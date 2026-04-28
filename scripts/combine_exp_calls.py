"""Combine DAS accepted calls per experiment and emit calls.csv.

Under the no-leakage assumption (true for datasets 2025_07, 2025_10, 2026_02):
the DAS detection channel IS the source arena. No cross-talk dedupe, no RMS
computation, no WAV reads — just map channel → arena directly.

For each experiment in EXPERIMENT_IDS:
  1. Load all *_accepted_calls.csv from calls_confident/<subdir>/.
  2. Set assigned_location from the DAS channel (10→arena_1, 20→arena_2, 30→underground).
  3. Write calls.csv at the experiment root.

Edit the values below and run:  python scripts/combine_exp_calls.py
"""
import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocalization_analysis.audio_processing_config import get_experiment_month
from vocalization_analysis.pipelines.rms_assignment import (
    DEFAULT_SOURCE_CHANNELS,
    build_qmc_metadata_csv,
    load_per_file_calls,
    sort_calls_by_time,
)

# === Edit these before running ============================================
#EXPERIMENT_IDS = [97,98,99] # explicit list
EXPERIMENT_IDS = list(range(112,113))     # inclusive 530..540



# Optional override. Leave as None to use the default mapping.
SOURCE_CHANNELS = None

# Optional override when calls_confident contains more than one result folder.
# Set to None to auto-detect when there is exactly one subdir.
CALLS_CONFIDENT_SUBDIR = "entropy_thr_default_0.30_hf_warble_0.60"
# ==========================================================================


if platform.system() == "Windows":
    BASE_PROCESSED = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    BASE_PROCESSED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio")


def get_experiment_audio_dir(exp: int) -> Path:
    return BASE_PROCESSED / get_experiment_month(exp) / str(exp)


def resolve_calls_confident_dir(exp_audio_dir: Path) -> Path:
    root = exp_audio_dir / "calls_confident"
    if not root.exists():
        raise FileNotFoundError(f"calls_confident folder not found: {root}")
    if CALLS_CONFIDENT_SUBDIR is not None:
        chosen = root / CALLS_CONFIDENT_SUBDIR
        if not chosen.exists():
            raise FileNotFoundError(f"Configured calls-confident subdir not found: {chosen}")
        return chosen
    subdirs = sorted(p for p in root.iterdir() if p.is_dir())
    if len(subdirs) == 1:
        return subdirs[0]
    if len(subdirs) == 0:
        raise FileNotFoundError(f"No result subdirs found under: {root}")
    raise RuntimeError(
        "Multiple calls_confident subdirs found. Set CALLS_CONFIDENT_SUBDIR:\n"
        + "\n".join(f"  - {p.name}" for p in subdirs)
    )


def combine_for_experiment(exp: int) -> Path:
    exp_audio_dir = get_experiment_audio_dir(exp)
    accepted_calls_dir = resolve_calls_confident_dir(exp_audio_dir)

    source_channels = dict(DEFAULT_SOURCE_CHANNELS if SOURCE_CHANNELS is None else SOURCE_CHANNELS)
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

    out_path = exp_audio_dir / "calls.csv"
    qmc_df.to_csv(out_path, index=False)
    print(f"Wrote {len(qmc_df)} calls to {out_path}")
    return out_path


def main() -> int:
    experiment_ids = list(EXPERIMENT_IDS)
    if not experiment_ids:
        raise SystemExit("Set EXPERIMENT_IDS to at least one experiment number.")

    print(f"Experiments to process: {experiment_ids}")
    failed: list[tuple[int, str]] = []
    for exp in experiment_ids:
        print(f"\nExperiment: {exp}")
        try:
            combine_for_experiment(exp)
        except FileNotFoundError as exc:
            print(f"Skipping experiment {exp}: {exc}")
            failed.append((exp, str(exc)))

    if failed:
        print("\nSkipped experiments:")
        for exp, reason in failed:
            print(f"  {exp}: {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
