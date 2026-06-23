"""Read per-experiment sync.csv to align audio file_num with wall-clock time.

Each big_setup experiment writes a sync.csv with one row per audio chunk file.
Row index == file_num (0-based). The ``timestamp`` column is a stringified
``[start_iso, end_iso]`` pair giving the wall-clock window for that chunk.

Use ``load_sync_table(exp)`` to get a per-chunk index, or
``attach_experiment_time(calls_df, exp)`` to add experiment-relative time to a
calls DataFrame that has ``file_num`` and ``start_time_file_sec`` columns.
"""
from __future__ import annotations

import ast
import platform
from pathlib import Path

import pandas as pd


if platform.system() == "Windows":
    BASE_RAW = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data")
else:
    BASE_RAW = Path("/mnt/home/neurostatslab/ceph/saneslab_data/big_setup")


def sync_csv_path(exp: int, base_raw: Path | None = None) -> Path:
    root = BASE_RAW if base_raw is None else base_raw
    return root / f"experiment_{exp}" / "concatenated_data_cam_mic_sync" / "sync.csv"


def load_sync_table(exp: int, base_raw: Path | None = None) -> pd.DataFrame:
    """Return per-chunk timing for ``exp``.

    Index is ``file_num`` (0-based, taken from sync.csv row order). Columns:
      - ``chunk_start_dt`` / ``chunk_end_dt``: wall-clock datetimes.
      - ``chunk_start_sec``: seconds since the first chunk's start.
    """
    path = sync_csv_path(exp, base_raw)
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError(f"sync.csv at {path} has no 'timestamp' column.")

    ts = df["timestamp"].apply(ast.literal_eval)
    starts = pd.to_datetime(ts.str[0])
    ends = pd.to_datetime(ts.str[-1])
    chunk_start_sec = (starts - starts.iloc[0]).dt.total_seconds()

    out = pd.DataFrame(
        {
            "chunk_start_dt": starts,
            "chunk_end_dt": ends,
            "chunk_start_sec": chunk_start_sec,
        }
    )
    out.index.name = "file_num"
    return out


def build_file_times(exp: int, base_raw: Path | None = None) -> pd.DataFrame:
    """Return the tidy per-file timing table for ``exp``.

    Columns: ``exp_num, file_num, start_date, start_time, start_sec_from_exp,
    end_date, end_time``. This is the slim, redundancy-free copy of the raw
    sync.csv (no video/audio file-name lists), with ``start_sec_from_exp``
    giving seconds since the start of file 0.
    """
    sync = load_sync_table(exp, base_raw=base_raw)
    starts = sync["chunk_start_dt"]
    ends = sync["chunk_end_dt"]
    return pd.DataFrame(
        {
            "exp_num": exp,
            "file_num": sync.index.to_numpy(),
            "start_date": starts.dt.strftime("%Y-%m-%d").to_numpy(),
            "start_time": starts.dt.strftime("%H:%M:%S").to_numpy(),
            "start_sec_from_exp": sync["chunk_start_sec"].to_numpy(),
            "end_date": ends.dt.strftime("%Y-%m-%d").to_numpy(),
            "end_time": ends.dt.strftime("%H:%M:%S").to_numpy(),
        }
    )


def write_file_times(
    exp: int, dest_dir: Path, base_raw: Path | None = None
) -> Path:
    """Write ``file_times.csv`` for ``exp`` into ``dest_dir`` and return its path."""
    out = build_file_times(exp, base_raw=base_raw)
    dest = Path(dest_dir) / "file_times.csv"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    return dest


def attach_experiment_time(
    calls_df: pd.DataFrame,
    exp: int,
    *,
    base_raw: Path | None = None,
    file_num_col: str = "file_num",
    onset_col: str = "start_time_file_sec",
) -> pd.DataFrame:
    """Add ``time_from_exp_start_sec`` and ``wall_clock_dt`` to ``calls_df``.

    Calls whose ``file_num`` is missing from sync.csv get NaN / NaT instead of
    raising, so this is safe to call when sync.csv is shorter than the call set.
    """
    sync = load_sync_table(exp, base_raw=base_raw)
    chunk_start_sec_by_file = sync["chunk_start_sec"].to_dict()
    chunk_start_dt_by_file = sync["chunk_start_dt"].to_dict()

    out = calls_df.copy()
    file_num = pd.to_numeric(out[file_num_col], errors="coerce")
    onset = pd.to_numeric(out[onset_col], errors="coerce")

    chunk_start_sec = file_num.map(chunk_start_sec_by_file)
    chunk_start_dt = pd.to_datetime(file_num.map(chunk_start_dt_by_file))

    out["time_from_exp_start_sec"] = chunk_start_sec + onset
    out["wall_clock_dt"] = chunk_start_dt + pd.to_timedelta(onset, unit="s")
    return out
