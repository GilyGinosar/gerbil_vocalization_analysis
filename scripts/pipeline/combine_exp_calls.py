"""Combine DAS accepted calls per experiment and emit calls.csv.

Under the no-leakage assumption (true for datasets 2025_07, 2025_10, 2026_02):
the DAS detection channel IS the source arena. No cross-talk dedupe, no RMS
computation, no WAV reads — just map channel → arena directly.

For each selected experiment:
  1. Load all *_accepted_calls.csv from calls_confident/<subdir>/.
  2. Set assigned_location from the DAS channel (10→arena_1, 20→arena_2, 30→underground).
  3. Write calls.csv at the experiment root.

Run it on a whole date folder:

    python scripts/pipeline/combine_exp_calls.py --date-folder 2026_08

or on specific experiments:

    python scripts/pipeline/combine_exp_calls.py --experiment-id 785 790

The other route to the same calls.csv is run_rms_assignment.py, which computes
RMS across the three channels and dedupes overlaps. Use that when leakage
between arenas is possible.
"""
import argparse
from pathlib import Path

from scripts.pipeline.audio_processing_config import should_skip_experiment
from scripts.pipeline.paths import experiment_audio_dir, list_experiment_ids_for_date
from scripts.pipeline.rms_assignment import (
    DEFAULT_SOURCE_CHANNELS,
    build_qmc_metadata_csv,
    load_per_file_calls,
    sort_calls_by_time,
)
from vocalization_analysis.sync_times import attach_experiment_time, sync_csv_path

# Default DAS result folder to read when calls_confident holds more than one.
DEFAULT_CALLS_CONFIDENT_SUBDIR = "entropy_thr_default_0.30_hf_warble_0.60"


def resolve_calls_confident_dir(exp_audio_dir: Path, subdir: str | None) -> Path:
    root = exp_audio_dir / "calls_confident"
    if not root.exists():
        raise FileNotFoundError(f"calls_confident folder not found: {root}")
    if subdir is not None:
        chosen = root / subdir
        if not chosen.exists():
            raise FileNotFoundError(f"Configured calls-confident subdir not found: {chosen}")
        return chosen
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(subdirs) == 1:
        return subdirs[0]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No result subdirs found under: {root}")
    raise RuntimeError(
        "Multiple calls_confident subdirs found. Pass --calls-confident-subdir:\n"
        + "\n".join(f"  - {p.name}" for p in subdirs)
    )


def combine_for_experiment(exp: int, subdir: str | None, source_channels_override=None) -> Path:
    exp_audio_dir = experiment_audio_dir(exp)
    accepted_calls_dir = resolve_calls_confident_dir(exp_audio_dir, subdir)

    source_channels = dict(DEFAULT_SOURCE_CHANNELS if source_channels_override is None else source_channels_override)
    location_by_channel = {channel: location for location, channel in source_channels.items()}

    print(f"Accepted calls dir: {accepted_calls_dir}")

    combined_df = load_per_file_calls(accepted_calls_dir)
    n_channels = combined_df["channel"].nunique() if not combined_df.empty else 0
    print(f"Loaded {len(combined_df)} detections from {n_channels} channels.")

    if combined_df.empty:
        raise FileNotFoundError(f"No *_accepted_calls.csv files found in {accepted_calls_dir}")

    combined_df["assigned_location"] = combined_df["channel"].astype(str).map(location_by_channel)
    unmapped = combined_df["assigned_location"].isna().sum()
    if unmapped:
        print(f"Warning: {unmapped} detections have an unrecognized channel value; assigned_location is NaN for those.")

    combined_df = sort_calls_by_time(combined_df)
    qmc_df = build_qmc_metadata_csv(combined_df, source_channels)

    # Carry through entropy columns from the DAS accepted_calls.csv files.
    # Both build_qmc_metadata_csv and sort_calls_by_time use the same sort keys
    # (file_num, onset_s), so combined_df and qmc_df rows are aligned by index.
    for col in ("mean_entropy", "mean_entropy_norm"):
        if col in combined_df.columns:
            qmc_df[col] = combined_df[col].values

    # Align calls onto a single experiment timeline using sync.csv.
    # Adds time_from_exp_start_sec (offset from first chunk's start) and
    # wall_clock_dt (absolute datetime). NaN/NaT for any file_num missing from
    # sync.csv. Skipped with a warning if sync.csv is absent.
    sync_path = sync_csv_path(exp)
    if sync_path.exists():
        qmc_df = attach_experiment_time(qmc_df, exp)
        # Place the new time columns right after stop_time_file_sec, so all
        # timing columns sit together.
        new_time_cols = [c for c in ("time_from_exp_start_sec", "wall_clock_dt") if c in qmc_df.columns]
        if new_time_cols and "stop_time_file_sec" in qmc_df.columns:
            rest = [c for c in qmc_df.columns if c not in new_time_cols]
            insert_at = rest.index("stop_time_file_sec") + 1
            qmc_df = qmc_df[rest[:insert_at] + new_time_cols + rest[insert_at:]]
    else:
        print(f"Warning: sync.csv not found at {sync_path}; skipping experiment-time columns.")

    out_path = exp_audio_dir / "calls.csv"
    qmc_df.to_csv(out_path, index=False)
    print(f"Wrote {len(qmc_df)} calls to {out_path}")
    return out_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Turn DAS accepted-call CSVs into one calls.csv per experiment.",
        epilog="Example: python scripts/pipeline/combine_exp_calls.py --date-folder 2026_08",
    )
    sel = p.add_mutually_exclusive_group(required=True)
    sel.add_argument("--date-folder", help="Process every experiment present on disk in this date folder.")
    sel.add_argument("--experiment-id", nargs="+", type=int, help="Process these experiment ids.")
    p.add_argument("--calls-confident-subdir", default=DEFAULT_CALLS_CONFIDENT_SUBDIR,
                   help="DAS result folder under calls_confident/. Pass 'auto' to use the only one present.")
    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.date_folder:
        experiment_ids = [e for e in list_experiment_ids_for_date(args.date_folder)
                          if not should_skip_experiment(e)]
        if not experiment_ids:
            raise SystemExit(f"No experiment folders found for {args.date_folder}.")
    else:
        experiment_ids = args.experiment_id

    subdir = None if args.calls_confident_subdir == "auto" else args.calls_confident_subdir

    print(f"Experiments to process: {experiment_ids}")
    written, failed = [], []
    for exp in experiment_ids:
        print(f"\nExperiment: {exp}")
        try:
            written.append(combine_for_experiment(exp, subdir))
        except FileNotFoundError as exc:
            print(f"Skipping experiment {exp}: {exc}")
            failed.append((exp, str(exc)))

    print(f"\nWrote {len(written)} calls.csv file(s).")
    if failed:
        print("Skipped experiments (DAS output missing?):")
        for exp, reason in failed:
            print(f"  {exp}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
