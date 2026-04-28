"""Compute call-transition matrices per experiment from calls.csv + sync.csv.

Pipeline:
  1. add_exp_times(exp): load calls.csv and sync.csv, add experiment-seconds and
     wall-clock columns (start_time_experiment_sec / stop_time_experiment_sec /
     start_time_real / stop_time_real); rename assigned_channel -> channel.
  2. Filter by configured day/night hours (LIGHT_START_HOUR, LIGHT_END_HOUR).
  3. Write per-experiment, per-variant CSVs to
     <output_dir>/inputs/<variant>/<exp>/calls.csv. Each CSV's parent folder is
     the experiment id, so calc_transitions's `_source_exp` resolves correctly
     and transitions don't bridge across experiments. These derived inputs
     persist alongside the matrix outputs so you can inspect what fed each
     matrix; delete the entire <output_dir> to reset.
  4. For each variant in {all, day, night}: compute three interval-band
     transition matrices via calc_transitions.compute_and_save_arena_transitions
     (passing time_window='all' since filtering already happened upstream).
     Cross-file transitions are counted naturally via start_time_experiment_sec.
  5. Plot three overview figures: all / day / night.

Edit the values below and run:  python scripts/run_transitions.py
"""
import ast
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocalization_analysis.audio_processing_config import get_experiment_month
from vocalization_analysis.calc_transitions import (
    collect_inter_call_gaps,
    collect_self_inter_call_gaps,
    compute_and_save_arena_transitions,
    compute_shared_log_count_max,
    plot_transition_matrices,
)


# === Edit these before running ============================================
# One or more date folders to process. Experiments are auto-discovered from
# every integer-named sub-folder of each <BASE_PROCESSED>/<date_folder>/.
# A single run pools experiments across all listed date folders.
DATE_FOLDERS = ["2025_10"]                     # e.g. ["2025_07"] or ["2025_07", "2025_10", "2026_02"]

# Daytime window. Half-open: [LIGHT_START_HOUR, LIGHT_END_HOUR).
# Wraps midnight if LIGHT_START_HOUR > LIGHT_END_HOUR (e.g. 20 -> 8).
LIGHT_START_HOUR = 8
LIGHT_END_HOUR   = 20

# Inter-call interval bands (seconds).
VERY_SHORT_GAP_SEC = 0.05
SHORT_GAP_SEC      = 2
LONG_GAP_SEC       = 300

# Call-type order along matrix rows/cols. Matches what current DAS models emit.
CALL_TYPE_ORDER = ['newborn', 'high-freq', 'warble', 'alarm', 'stacks']

# Optional ad-hoc grouping, e.g. {'dense-stack': 'stacks'}. Empty = no grouping.
CALL_GROUP_MAP: dict[str, str] = {}

# Self-ICI histograms in the plots (one mini histogram per call type).
SELF_ICI_CALL_TYPES = ['high-freq', 'warble', 'alarm', 'stacks']

# Output goes to <BASE_PROCESSED>/<OUTPUT_FOLDER_NAME>/<dates_tag>/, where
# <dates_tag> = "_".join(DATE_FOLDERS). Each (date-folder combination) gets
# its own sub-folder, so different runs don't collide.
OUTPUT_FOLDER_NAME = "combined_transitions_outputs"
# ==========================================================================


if platform.system() == "Windows":
    BASE_RAW = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data")
    BASE_PROCESSED = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    BASE_RAW = Path("/mnt/home/neurostatslab/ceph/saneslab_data/big_setup/")
    BASE_PROCESSED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio")


def get_experiment_audio_dir(exp: int) -> Path:
    return BASE_PROCESSED / get_experiment_month(exp) / str(exp)


def list_experiment_ids_for_date(date_folder: str) -> list[int]:
    """Return all integer-named subfolders under BASE_PROCESSED/<date_folder>/, sorted."""
    folder = BASE_PROCESSED / date_folder
    if not folder.exists():
        raise SystemExit(f"Date folder not found: {folder}")
    return sorted(
        int(p.name) for p in folder.iterdir()
        if p.is_dir() and p.name.isdigit()
    )


def get_experiment_sync_path(exp: int) -> Path:
    return BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync" / "sync.csv"


def _parse_sync_field(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def _file_index_from_video_list(video_list) -> int | None:
    """Pull the trailing _NNN file index from a sync.csv `video` row.

    All entries within a single row share the same _NNN suffix regardless of
    camera angle (e.g. video_center_001, video_gily_center_001, video_nest_top_001
    all came from the same chunk), so we just try each entry until one parses.
    """
    if video_list is None or (isinstance(video_list, float) and np.isnan(video_list)):
        return None
    for entry in video_list:
        s = str(entry)
        try:
            return int(s.rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            continue
    return None


def add_exp_times(exp: int) -> pd.DataFrame:
    """Return calls for `exp` with experiment-seconds + wall-clock columns added.

    Reads calls.csv (output of combine_exp_calls.py) and the experiment's
    sync.csv. Each call's per-file time is shifted by its chunk's offset
    relative to experiment start.

    Adds columns:
      - channel (renamed from assigned_channel)
      - start_time_experiment_sec, stop_time_experiment_sec
      - start_time_real, stop_time_real
    """
    exp_dir = get_experiment_audio_dir(exp)
    calls_path = exp_dir / "calls.csv"
    sync_path = get_experiment_sync_path(exp)

    if not calls_path.exists():
        raise FileNotFoundError(f"calls.csv not found: {calls_path}")
    if not sync_path.exists():
        raise FileNotFoundError(f"sync.csv not found: {sync_path}")

    sync_df = pd.read_csv(sync_path)
    sync_df["timestamp"] = sync_df["timestamp"].apply(_parse_sync_field)
    if "video" not in sync_df.columns:
        raise ValueError(f"sync.csv missing 'video' column: {sync_path}")
    sync_df["video"] = sync_df["video"].apply(_parse_sync_field)
    sync_df["chunk_start_real"] = pd.to_datetime(sync_df["timestamp"].apply(lambda t: t[0]))
    sync_df["file_num"] = sync_df["video"].apply(_file_index_from_video_list)
    sync_df = sync_df.dropna(subset=["file_num", "chunk_start_real"]).copy()
    sync_df["file_num"] = sync_df["file_num"].astype(int)

    experiment_start_real = sync_df["chunk_start_real"].min()
    sync_df["chunk_offset_sec"] = (sync_df["chunk_start_real"] - experiment_start_real).dt.total_seconds()

    file_to_offset = dict(zip(sync_df["file_num"], sync_df["chunk_offset_sec"]))
    file_to_real = dict(zip(sync_df["file_num"], sync_df["chunk_start_real"]))

    calls = pd.read_csv(calls_path)
    if "assigned_channel" in calls.columns and "channel" not in calls.columns:
        calls = calls.rename(columns={"assigned_channel": "channel"})

    calls = calls.dropna(subset=["file_num"]).copy()
    calls["file_num"] = calls["file_num"].astype(int)

    missing = sorted({int(fn) for fn in calls["file_num"].unique() if fn not in file_to_offset})
    if missing:
        sample = ", ".join(str(m) for m in missing[:5]) + ("..." if len(missing) > 5 else "")
        print(f"  exp {exp}: dropping calls for {len(missing)} file_num(s) not in sync.csv: {sample}")
        calls = calls[calls["file_num"].isin(file_to_offset)].copy()

    if calls.empty:
        raise ValueError(
            f"No calls in exp {exp} have a file_num present in sync.csv "
            f"(sync had {len(file_to_offset)} mappable chunks)."
        )

    chunk_offset = calls["file_num"].map(file_to_offset)
    chunk_real = calls["file_num"].map(file_to_real)
    calls["start_time_experiment_sec"] = chunk_offset + calls["start_time_file_sec"]
    calls["stop_time_experiment_sec"]  = chunk_offset + calls["stop_time_file_sec"]
    calls["start_time_real"] = chunk_real + pd.to_timedelta(calls["start_time_file_sec"], unit="s")
    calls["stop_time_real"]  = chunk_real + pd.to_timedelta(calls["stop_time_file_sec"], unit="s")

    return calls


def _is_in_daytime(hours: pd.Series, light_start: int, light_end: int) -> pd.Series:
    light_start = light_start % 24
    light_end = light_end % 24
    if light_start == light_end:
        return pd.Series(False, index=hours.index)
    if light_start < light_end:
        return (hours >= light_start) & (hours < light_end)
    return (hours >= light_start) | (hours < light_end)


def filter_by_daynight(df: pd.DataFrame, time_window: str,
                       light_start: int, light_end: int) -> pd.DataFrame:
    if time_window == "all":
        return df.copy()
    if time_window not in {"day", "night"}:
        raise ValueError(f"Unsupported time_window: {time_window}")
    real_dt = pd.to_datetime(df["start_time_real"], errors="coerce")
    valid = real_dt.notna()
    is_day = _is_in_daytime(real_dt.dt.hour, light_start, light_end)
    keep = is_day if time_window == "day" else ~is_day
    return df.loc[valid & keep].copy()


def hours_in_window(span_start: pd.Timestamp, span_end: pd.Timestamp,
                    time_window: str, light_start: int, light_end: int) -> float:
    if pd.isna(span_start) or pd.isna(span_end) or span_end <= span_start:
        return 0.0
    span_sec = (span_end - span_start).total_seconds()
    if time_window == "all":
        return span_sec / 3600

    light_start = light_start % 24
    light_end = light_end % 24
    if light_start == light_end:
        return 0.0 if time_window == "day" else span_sec / 3600

    day_sec = 0.0
    cur = pd.Timestamp(span_start.date())
    end_day = pd.Timestamp(span_end.date()) + pd.Timedelta(days=1)
    while cur < end_day:
        if light_start < light_end:
            intervals = [(cur + pd.Timedelta(hours=light_start),
                          cur + pd.Timedelta(hours=light_end))]
        else:
            intervals = [
                (cur, cur + pd.Timedelta(hours=light_end)),
                (cur + pd.Timedelta(hours=light_start), cur + pd.Timedelta(days=1)),
            ]
        for w0, w1 in intervals:
            day_sec += max(0.0, (min(w1, span_end) - max(w0, span_start)).total_seconds())
        cur += pd.Timedelta(days=1)

    if time_window == "day":
        return day_sec / 3600
    return (span_sec - day_sec) / 3600


def write_for_transitions(df: pd.DataFrame, inputs_root: Path, variant: str, exp: int) -> Path:
    """Write filtered calls to <inputs_root>/<variant>/<exp>/calls.csv so that
    calc_transitions's `_source_exp` (parent dir name) becomes the experiment id."""
    target_dir = inputs_root / variant / str(exp)
    target_dir.mkdir(parents=True, exist_ok=True)
    out = target_dir / "calls.csv"
    df.to_csv(out, index=False)
    return out


def _tag(variant: str, band: str,
         long_s: float, short_s: float, very_short_s: float) -> str:
    if band == "left":
        return f"{variant}_gt{short_s}_le{long_s}".replace(".", "p")
    if band == "mid":
        return f"{variant}_gt{very_short_s}_le{short_s}".replace(".", "p")
    return f"{variant}_le{very_short_s}".replace(".", "p")


def main() -> int:
    if not DATE_FOLDERS:
        raise SystemExit("Set DATE_FOLDERS to at least one date folder.")

    experiment_ids: list[int] = []
    for date_folder in DATE_FOLDERS:
        experiment_ids.extend(list_experiment_ids_for_date(date_folder))
    experiment_ids = sorted(set(experiment_ids))
    if not experiment_ids:
        raise SystemExit(f"No experiments found in any of {DATE_FOLDERS}")

    dates_tag = "_".join(DATE_FOLDERS)
    output_dir = BASE_PROCESSED / OUTPUT_FOLDER_NAME / dates_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    call_group_map = dict(CALL_GROUP_MAP)
    call_type_order = list(CALL_TYPE_ORDER)

    print(f"Date folders     : {DATE_FOLDERS}")
    print(f"Output dir       : {output_dir}")
    print(f"Daytime window   : [{LIGHT_START_HOUR:02d}:00, {LIGHT_END_HOUR:02d}:00)")
    print(f"Call-type order  : {call_type_order}")
    sample = experiment_ids[:8]
    suffix = "..." if len(experiment_ids) > 8 else ""
    print(f"Experiments      : {len(experiment_ids)}  ({sample}{suffix})")

    # 1. Enrich each experiment's calls.csv with sync-derived times.
    enriched_by_exp: dict[int, pd.DataFrame] = {}
    failed: list[tuple[int, str]] = []
    for exp in experiment_ids:
        try:
            enriched_by_exp[exp] = add_exp_times(exp)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Skipping exp {exp}: {exc}")
            failed.append((exp, str(exc)))

    if not enriched_by_exp:
        raise SystemExit("No experiments could be enriched; nothing to compute.")

    band_specs = [
        ("left",  LONG_GAP_SEC,        SHORT_GAP_SEC),       # short_gap < gap <= long_gap
        ("mid",   SHORT_GAP_SEC,       VERY_SHORT_GAP_SEC),  # very_short_gap < gap <= short_gap
        ("right", VERY_SHORT_GAP_SEC,  None),                # gap <= very_short_gap
    ]
    variants = ("all", "day", "night")

    # Wipe stale per-variant inputs from any prior run so the on-disk state
    # always matches the current LIGHT_START_HOUR / LIGHT_END_HOUR settings.
    inputs_root = output_dir / "inputs"
    if inputs_root.exists():
        shutil.rmtree(inputs_root)
    inputs_root.mkdir(parents=True, exist_ok=True)
    print(f"Inputs dir       : {inputs_root}  (wiped fresh)")

    # 2. Per-variant pre-filtered CSVs + per-experiment recording spans.
    csv_paths_by_variant: dict[str, list[Path]] = {v: [] for v in variants}
    span_by_exp: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    for exp, enriched in enriched_by_exp.items():
        starts = pd.to_datetime(enriched["start_time_real"], errors="coerce")
        stops  = pd.to_datetime(enriched["stop_time_real"], errors="coerce")
        span_by_exp[exp] = (starts.min(), stops.max())
        for variant in variants:
            filtered = filter_by_daynight(enriched, variant, LIGHT_START_HOUR, LIGHT_END_HOUR)
            if filtered.empty:
                continue
            csv_paths_by_variant[variant].append(
                write_for_transitions(filtered, inputs_root, variant, exp)
            )

    # 3. For each variant: 3 interval-band transition matrices + ICI gaps + hours.
    gaps_by_variant: dict[str, np.ndarray] = {}
    self_ici_by_variant: dict[str, dict[str, np.ndarray]] = {}
    hours_by_variant: dict[str, float] = {}

    for variant in variants:
        paths = csv_paths_by_variant[variant]
        hours_by_variant[variant] = sum(
            hours_in_window(s, e, variant, LIGHT_START_HOUR, LIGHT_END_HOUR)
            for s, e in span_by_exp.values()
        )
        if not paths:
            print(f"Variant '{variant}' has no input CSVs; skipping.")
            gaps_by_variant[variant] = np.array([])
            self_ici_by_variant[variant] = {ct: np.array([]) for ct in SELF_ICI_CALL_TYPES}
            continue

        for band, upper, lower in band_specs:
            tag = _tag(variant, band, LONG_GAP_SEC, SHORT_GAP_SEC, VERY_SHORT_GAP_SEC)
            compute_and_save_arena_transitions(
                paths,
                inter_call_interval_sec=upper,
                output_dir=str(output_dir),
                min_inter_call_interval_sec=lower,
                call_group_map=call_group_map,
                call_type_order=call_type_order,
                file_tag=tag,
            )

        gaps_by_variant[variant] = collect_inter_call_gaps(paths)
        self_ici_by_variant[variant] = {
            ct: collect_self_inter_call_gaps(paths, ct, call_group_map=call_group_map)
            for ct in SELF_ICI_CALL_TYPES
        }

    # 4. Shared color scale across all 3 figures.
    shared_log_count_max = compute_shared_log_count_max(
        [(str(output_dir), _tag(v, b, LONG_GAP_SEC, SHORT_GAP_SEC, VERY_SHORT_GAP_SEC))
         for v in variants for b, _, _ in band_specs],
        arena_names=("arena", "underground"),
    )

    # 4b. Shared bins + y-max for the histograms across all 3 figures, so
    # day/night/all are visually comparable.
    zoom_min_sec = 0.05
    zoom_max_sec = 3.0
    zoom_bins = np.geomspace(zoom_min_sec, zoom_max_sec, 30)

    pooled = np.concatenate(
        [g for g in gaps_by_variant.values() if len(g) > 0]
    ) if any(len(g) > 0 for g in gaps_by_variant.values()) else np.array([])
    pooled_pos = pooled[pooled > 0] if len(pooled) > 0 else pooled
    hist_full_xmax_sec = 1000.0
    if len(pooled_pos) > 0:
        full_bins = np.geomspace(
            max(1e-3, float(pooled_pos.min())),
            min(float(pooled_pos.max()), hist_full_xmax_sec),
            100,
        )
    else:
        full_bins = None

    def _hist_max(g, bins):
        if g is None or len(g) == 0 or bins is None:
            return 0
        h, _ = np.histogram(g, bins=bins)
        return int(h.max()) if h.size else 0

    full_ymax = max(
        _hist_max(g[g > 0] if len(g) else g, full_bins)
        for g in gaps_by_variant.values()
    )
    zoom_ymax = max(
        _hist_max(g[(g >= zoom_min_sec) & (g <= zoom_max_sec)] if len(g) else g, zoom_bins)
        for g in gaps_by_variant.values()
    )
    self_ici_ymax_by_type = {
        ct: max(
            _hist_max(
                self_ici_by_variant[v][ct][
                    (self_ici_by_variant[v][ct] >= zoom_min_sec)
                    & (self_ici_by_variant[v][ct] <= zoom_max_sec)
                ] if len(self_ici_by_variant[v][ct]) else self_ici_by_variant[v][ct],
                zoom_bins,
            )
            for v in variants
        )
        for ct in SELF_ICI_CALL_TYPES
    }

    # 5. Plot per variant.
    exp_names = ", ".join(str(e) for e in sorted(enriched_by_exp.keys()))
    daytime_label = f"{LIGHT_START_HOUR:02d}:00-{LIGHT_END_HOUR:02d}:00"
    interval_label_left  = f"{SHORT_GAP_SEC}s < inter-call-interval <= {LONG_GAP_SEC}s"
    interval_label_mid   = f"{VERY_SHORT_GAP_SEC}s < inter-call-interval <= {SHORT_GAP_SEC}s"
    interval_label_right = f"inter-call-interval <= {VERY_SHORT_GAP_SEC}s"

    plot_specs = [
        ("all",   "transition_matrices_overview.png",  "all data", ""),
        ("day",   "transition_matrices_daytime.png",   "light",    f" | Daytime ({daytime_label})"),
        ("night", "transition_matrices_nighttime.png", "dark",     f" | Nighttime (outside {daytime_label})"),
    ]
    for variant, save_name, figure_title, label_extra in plot_specs:
        plot_note = (
            f"Exps: {exp_names}{label_extra} | Total analyzed duration: "
            f"{hours_by_variant[variant]:.2f} h"
        )
        plot_transition_matrices(
            str(output_dir), str(output_dir), str(output_dir),
            save_name=save_name,
            plot_note=plot_note,
            figure_title=figure_title,
            interval_left=LONG_GAP_SEC,
            interval_mid=SHORT_GAP_SEC,
            interval_right=VERY_SHORT_GAP_SEC,
            interval_left_label=interval_label_left,
            interval_mid_label=interval_label_mid,
            interval_right_label=interval_label_right,
            inter_call_gaps=gaps_by_variant[variant],
            self_ici_gaps_by_type=self_ici_by_variant[variant],
            self_ici_call_types=SELF_ICI_CALL_TYPES,
            shared_log_count_max=shared_log_count_max,
            hist_full_bins=full_bins,
            hist_full_ymax=full_ymax,
            hist_zoom_ymax=zoom_ymax,
            self_ici_ymax_by_type=self_ici_ymax_by_type,
            thresholds=[VERY_SHORT_GAP_SEC, SHORT_GAP_SEC, LONG_GAP_SEC],
            hist_full_xmax_sec=hist_full_xmax_sec,
            call_type_order=call_type_order,
            file_tag_left=_tag(variant, "left",  LONG_GAP_SEC, SHORT_GAP_SEC, VERY_SHORT_GAP_SEC),
            file_tag_mid =_tag(variant, "mid",   LONG_GAP_SEC, SHORT_GAP_SEC, VERY_SHORT_GAP_SEC),
            file_tag_right=_tag(variant, "right", LONG_GAP_SEC, SHORT_GAP_SEC, VERY_SHORT_GAP_SEC),
        )

    if failed:
        print("\nSkipped experiments:")
        for exp, reason in failed:
            print(f"  {exp}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
