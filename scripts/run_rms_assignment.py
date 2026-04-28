import platform
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vocalization_analysis.audio_processing_config import get_experiment_month
from vocalization_analysis.pipelines.rms_assignment import (
    RMSAssignmentConfig,
    run_rms_assignment,
)


# Edit these values before running the script.
EXPERIMENT_IDS = list(range(535,536))

OVERLAP_WINDOW_S = 0.015

# Optional override. Leave as `None` to use the default mapping.
# SOURCE_CHANNELS = {"arena_1": "10", "arena_2": "20", "underground": "30"}
SOURCE_CHANNELS = None

# Optional override when `calls_confident` contains more than one result folder.
# Example: "entropy_thr_default_0.30_hf_warble_0.60"
CALLS_CONFIDENT_SUBDIR = "entropy_thr_default_0.30_hf_warble_0.60"


if platform.system() == "Windows":
    base_processed = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    base_processed = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio")


def get_experiment_audio_dir(exp: int) -> Path:
    month_folder = get_experiment_month(exp)
    return base_processed / month_folder / str(exp)


def resolve_calls_confident_dir(exp_audio_dir: Path) -> Path:
    calls_confident_root = exp_audio_dir / "calls_confident"
    if not calls_confident_root.exists():
        raise FileNotFoundError(f"Calls-confident folder not found: {calls_confident_root}")

    if CALLS_CONFIDENT_SUBDIR is not None:
        chosen = calls_confident_root / CALLS_CONFIDENT_SUBDIR
        if not chosen.exists():
            raise FileNotFoundError(f"Configured calls-confident subdir not found: {chosen}")
        return chosen

    subdirs = sorted(path for path in calls_confident_root.iterdir() if path.is_dir())
    if len(subdirs) == 1:
        return subdirs[0]

    if len(subdirs) == 0:
        raise FileNotFoundError(f"No result subdirectories found under: {calls_confident_root}")

    raise RuntimeError(
        "Multiple calls_confident result folders found. Set CALLS_CONFIDENT_SUBDIR in the script to choose one:\n"
        + "\n".join(f"  - {path.name}" for path in subdirs)
    )


def build_config(exp: int) -> RMSAssignmentConfig:
    exp_audio_dir = get_experiment_audio_dir(exp)
    accepted_calls_dir = resolve_calls_confident_dir(exp_audio_dir)
    averaged_wavs_dir = exp_audio_dir / "Averaged_wavs_w_annotations"
    output_dir = exp_audio_dir / "rms_assignment"

    return RMSAssignmentConfig(
        accepted_calls_dir=accepted_calls_dir,
        averaged_wavs_dir=averaged_wavs_dir,
        output_dir=output_dir,
        experiment_dir=exp_audio_dir,
        overlap_window_s=OVERLAP_WINDOW_S,
        source_channels=SOURCE_CHANNELS,
    )


def main() -> int:
    if not EXPERIMENT_IDS:
        raise ValueError("Set EXPERIMENT_IDS to at least one experiment number.")

    for exp in EXPERIMENT_IDS:
        print(f"\nExperiment: {exp}")
        try:
            config = build_config(exp)
            print(f"Accepted calls dir: {config.accepted_calls_dir}")
            print(f"Averaged wavs dir: {config.averaged_wavs_dir}")
            print(f"Output dir: {config.output_dir}")
            outputs = run_rms_assignment(config)
        except FileNotFoundError as exc:
            print(f"Skipping experiment {exp}: {exc}")
            continue

        for name, path in outputs.items():
            print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
