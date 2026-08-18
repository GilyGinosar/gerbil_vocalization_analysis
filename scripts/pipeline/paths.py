"""Where the data lives — the single place that knows the roots.

Every script used to carry its own ``platform.system()`` branch and its own
copy of ``BASE_PROCESSED``. New code imports from here instead::

    from scripts.pipeline.paths import AUDIO_ROOT, experiment_audio_dir
"""
from __future__ import annotations

import platform
from pathlib import Path

from scripts.pipeline.audio_processing_config import get_experiment_month

if platform.system() == "Windows":
    BASE_RAW = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data")
    PROCESSED_ROOT = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data")
else:
    BASE_RAW = Path("/mnt/home/neurostatslab/ceph/saneslab_data/big_setup")
    PROCESSED_ROOT = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data")

# What the analysis scripts call BASE_PROCESSED: the Audio/ level, holding one
# folder per date folder plus the pooled all_calls/ directory.
AUDIO_ROOT = PROCESSED_ROOT / "Audio"
ALL_CALLS_DIR = AUDIO_ROOT / "all_calls"
PARQUET_DIR = ALL_CALLS_DIR / "parquet_cache"


def experiment_audio_dir(exp: int) -> Path:
    """<AUDIO_ROOT>/<date folder>/<exp>/ — where this experiment's outputs live."""
    return AUDIO_ROOT / get_experiment_month(exp) / str(exp)


def experiment_sync_path(exp: int) -> Path:
    """The raw sync.csv aligning audio chunks to wall-clock time."""
    return BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync" / "sync.csv"


def date_folder_dir(date_folder: str) -> Path:
    return AUDIO_ROOT / date_folder


def list_experiment_ids_for_date(date_folder: str) -> list[int]:
    """Experiment ids that actually have a processed folder on disk, sorted.

    This is the on-disk truth, as opposed to
    ``audio_processing_config.get_experiments_for_date``, which is the
    configured intent. They differ while a cohort is still being processed.
    """
    folder = date_folder_dir(date_folder)
    if not folder.exists():
        raise FileNotFoundError(f"Date folder not found: {folder}")
    return sorted(int(p.name) for p in folder.iterdir() if p.is_dir() and p.name.isdigit())
