"""Local-Mac variant of run_transitions.py.

Differs from run_transitions.py only in the data-loading layer:
  - Reads consolidated CSVs (e.g. all_calls_2025_10.csv) from a local
    Dropbox folder instead of per-experiment calls.csv + sync.csv from
    network storage.
  - These consolidated CSVs already contain `exp`,
    `start_time_experiment_sec`, `stop_time_experiment_sec`,
    `start_time_real`, `stop_time_real`, and `channel`, so add_exp_times()
    and sync.csv are not needed.

Everything downstream (day/night filtering, per-experiment split,
calc_transitions, plotting) is the same as run_transitions.py.

Edit the values below and run:  python scripts/run_transitions_mac.py
"""
import glob
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocalization_analysis.calc_transitions import (
    collect_inter_call_gaps,
    collect_self_inter_call_gaps,
    compute_arena_transitions,
    plot_transition_matrices,
    save_arena_transitions,
)


# === Edit these before running ============================================
# One or more date tags to process. Each tag XXXX_YY maps to the file
# <INPUT_DIR>/all_calls_XXXX_YY.csv. A single run pools experiments across
# all listed tags.
DATE_FOLDERS = ["2025_03"]                     # e.g. ["2025_07", "2025_10", "2026_02"]

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

# If False, all CSVs (per-experiment inputs/ and per-band counts_/call_counts_/
# probabilities_ matrices) are deleted after the plots are written. The plots
# themselves are always kept. Useful for fast dev iteration when you only care
# about the figures.
SAVE_CSVS = False

# Local Dropbox folder containing the consolidated all_calls_<date>.csv files.
INPUT_DIR = Path("/Users/gilyginosar/Dropbox (Personal)/Vocalizations_project/Data")

# Where to write outputs (per-variant inputs/, transition CSVs, plots).
# Each (date-folder combination) gets its own sub-folder.
OUTPUT_BASE = INPUT_DIR / "combined_transitions_outputs"
# ==========================================================================


def consolidated_csv_path(date_folder: str) -> Path:
    return INPUT_DIR / f"all_calls_{date_folder}.csv"


def load_consolidated(date_folder: str) -> pd.DataFrame:
    """Load one all_calls_<date>.csv. Assumes it already has exp,
    start/stop_time_experiment_sec, start/stop_time_real, channel,
    event_type, file_num — i.e. the same columns add_exp_times() produces."""
    path = consolidated_csv_path(date_folder)
    if not path.exists():
        raise SystemExit(f"Consolidated CSV not found: {path}")
    df = pd.read_csv(path)
    required = {
        "exp", "channel", "event_type",
        "start_time_experiment_sec", "stop_time_experiment_sec",
        "start_time_real", "stop_time_real",
    }
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"{path} missing columns: {sorted(missing)}")
    return df


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

    # Load all requested CSVs and split into per-experiment frames keyed by `exp`.
    enriched_by_exp: dict[int, pd.DataFrame] = {}
    for date_folder in DATE_FOLDERS:
        df = load_consolidated(date_folder)
        for exp_id, exp_df in df.groupby("exp"):
            exp_id_int = int(exp_id)
            if exp_id_int in enriched_by_exp:
                # If the same exp appears in multiple date CSVs, concatenate.
                enriched_by_exp[exp_id_int] = pd.concat(
                    [enriched_by_exp[exp_id_int], exp_df.copy()], ignore_index=True
                )
            else:
                enriched_by_exp[exp_id_int] = exp_df.copy()

    if not enriched_by_exp:
        raise SystemExit(f"No rows loaded from {DATE_FOLDERS}.")

    experiment_ids = sorted(enriched_by_exp.keys())
    dates_tag = "_".join(DATE_FOLDERS)
    output_dir = OUTPUT_BASE / dates_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    call_group_map = dict(CALL_GROUP_MAP)
    call_type_order = list(CALL_TYPE_ORDER)

    print(f"Date folders     : {DATE_FOLDERS}")
    print(f"Input dir        : {INPUT_DIR}")
    print(f"Output dir       : {output_dir}")
    print(f"Daytime window   : [{LIGHT_START_HOUR:02d}:00, {LIGHT_END_HOUR:02d}:00)")
    print(f"Call-type order  : {call_type_order}")
    sample = experiment_ids[:8]
    suffix = "..." if len(experiment_ids) > 8 else ""
    print(f"Experiments      : {len(experiment_ids)}  ({sample}{suffix})")

    band_specs = [
        ("left",  LONG_GAP_SEC,        SHORT_GAP_SEC),       # short_gap < gap <= long_gap
        ("mid",   SHORT_GAP_SEC,       VERY_SHORT_GAP_SEC),  # very_short_gap < gap <= short_gap
        ("right", VERY_SHORT_GAP_SEC,  None),                # gap <= very_short_gap
    ]
    variants = ("all", "day", "night")

    # Wipe any stale on-disk state from a prior SAVE_CSVS=True run so we don't
    # leave half-stale matrices around if someone toggles SAVE_CSVS off.
    inputs_root = output_dir / "inputs"
    if inputs_root.exists():
        shutil.rmtree(inputs_root)
    for prefix in ("counts_", "call_counts_", "probabilities_"):
        for path in glob.glob(str(output_dir / f"{prefix}*.csv")):
            os.remove(path)

    # Filter each experiment per variant — kept entirely in memory.
    arena_names = ("arena", "underground")
    filtered_by_variant: dict[str, dict[int, pd.DataFrame]] = {v: {} for v in variants}
    span_by_exp: dict[int, tuple[pd.Timestamp, pd.Timestamp]] = {}
    calls_by_variant: dict[str, int] = {v: 0 for v in variants}
    for exp, enriched in enriched_by_exp.items():
        starts = pd.to_datetime(enriched["start_time_real"], errors="coerce")
        stops  = pd.to_datetime(enriched["stop_time_real"], errors="coerce")
        span_by_exp[exp] = (starts.min(), stops.max())
        for variant in variants:
            filtered = filter_by_daynight(enriched, variant, LIGHT_START_HOUR, LIGHT_END_HOUR)
            if filtered.empty:
                continue
            calls_by_variant[variant] += len(filtered)
            filtered_by_variant[variant][exp] = filtered
    # Free the unfiltered DataFrames — we no longer need them.
    del enriched_by_exp

    # If saving CSVs, persist the per-experiment inputs so they're inspectable.
    if SAVE_CSVS:
        inputs_root.mkdir(parents=True, exist_ok=True)
        for variant, by_exp in filtered_by_variant.items():
            for exp, df in by_exp.items():
                write_for_transitions(df, inputs_root, variant, exp)
        print(f"Inputs dir       : {inputs_root}")

    # Compute transition matrices, gaps, and self-ICI gaps in memory.
    matrices_by_variant: dict[str, dict[str, dict]] = {v: {} for v in variants}
    gaps_by_variant: dict[str, np.ndarray] = {}
    self_ici_by_variant: dict[str, dict[str, np.ndarray]] = {}
    hours_by_variant: dict[str, float] = {}

    for variant in variants:
        hours_by_variant[variant] = sum(
            hours_in_window(s, e, variant, LIGHT_START_HOUR, LIGHT_END_HOUR)
            for s, e in span_by_exp.values()
        )
        by_exp = filtered_by_variant[variant]
        if not by_exp:
            print(f"Variant '{variant}' has no rows; skipping.")
            gaps_by_variant[variant] = np.array([])
            self_ici_by_variant[variant] = {ct: np.array([]) for ct in SELF_ICI_CALL_TYPES}
            continue

        for band, upper, lower in band_specs:
            arena_matrices = compute_arena_transitions(
                by_exp,
                inter_call_interval_sec=upper,
                min_inter_call_interval_sec=lower,
                call_group_map=call_group_map,
                call_type_order=call_type_order,
            )
            matrices_by_variant[variant][band] = arena_matrices
            if SAVE_CSVS:
                tag = _tag(variant, band, LONG_GAP_SEC, SHORT_GAP_SEC, VERY_SHORT_GAP_SEC)
                save_arena_transitions(arena_matrices, str(output_dir), file_tag=tag)

        gaps_by_variant[variant] = collect_inter_call_gaps(by_exp)
        self_ici_by_variant[variant] = {
            ct: collect_self_inter_call_gaps(by_exp, ct, call_group_map=call_group_map)
            for ct in SELF_ICI_CALL_TYPES
        }

    # Shared color scale across all 3 figures, computed from in-memory matrices.
    shared_log_count_max = 0.0
    for v in variants:
        for band in matrices_by_variant[v].values():
            for arena in arena_names:
                vals = band[arena]['counts'].values
                if vals.size:
                    shared_log_count_max = max(shared_log_count_max, float(np.log1p(vals).max()))

    # Shared bins + y-max for the histograms across all 3 figures, so
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

    # Plot per variant.
    dates_label = ", ".join(DATE_FOLDERS)
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
            f"Dates: {dates_label}{label_extra} | "
            f"Hours spanned (recording window): {hours_by_variant[variant]:.2f} h | "
            f"Total calls: {calls_by_variant[variant]:,}"
        )
        band_matrices = matrices_by_variant[variant]
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
            matrices_left=band_matrices.get("left"),
            matrices_mid=band_matrices.get("mid"),
            matrices_right=band_matrices.get("right"),
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())