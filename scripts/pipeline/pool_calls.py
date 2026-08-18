"""Pool a date folder's per-experiment calls.csv into one file, and cache it.

This is the last stage before analysis. It replaces two hand-run steps:
``extract_calls_offline.py`` (which wrote to ~/offline_data, from where the
files were copied to ceph by hand) and the parquet-building cell inside
``notebooks/explore_calls_xplatform.ipynb``. Both are now one command:

    python scripts/pipeline/pool_calls.py --date-folder 2026_08

For each experiment in the date folder it reads calls.csv plus the raw sync.csv,
shifts every call's per-file time onto one experiment-wide clock, and writes:

    <all_calls>/all_calls_<date>.csv               pooled, self-contained
    <all_calls>/parquet_cache/all_calls_<date>.parquet   + date_folder,
                                                   day_in_experiment; dtypes kept

The parquet is what every analysis script actually reads
(``scripts/utils/ethogram_io.load_all_calls``).
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline.audio_processing_config import list_date_folders
from scripts.pipeline.paths import (
    ALL_CALLS_DIR,
    PARQUET_DIR,
    experiment_audio_dir,
    experiment_sync_path,
    list_experiment_ids_for_date,
)

DATETIME_COLS = ("start_time_real", "stop_time_real")


def _parse_sync_field(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def _file_index_from_video_list(video_list) -> int | None:
    """Pull the trailing _NNN chunk index from a sync.csv `video` row.

    Every entry in a row shares the suffix regardless of camera angle
    (video_center_001, video_nest_top_001, ...), so try each until one parses.
    """
    if video_list is None or (isinstance(video_list, float) and np.isnan(video_list)):
        return None
    for entry in video_list:
        try:
            return int(str(entry).rsplit("_", 1)[-1])
        except (ValueError, IndexError):
            continue
    return None


def chunk_start_times(exp: int) -> tuple[dict[int, pd.Timestamp], dict[int, float]]:
    """Map file_num -> (wall-clock start of that chunk, seconds since experiment start).

    Shared by the call and detection pooling: both put a per-file time onto the
    same experiment-wide clock, using the same sync.csv.
    """
    sync_path = experiment_sync_path(exp)
    if not sync_path.exists():
        raise FileNotFoundError(f"sync.csv not found: {sync_path}")
    sync_df = pd.read_csv(sync_path)
    if "video" not in sync_df.columns:
        raise ValueError(f"sync.csv missing 'video' column: {sync_path}")
    sync_df["timestamp"] = sync_df["timestamp"].apply(_parse_sync_field)
    sync_df["video"] = sync_df["video"].apply(_parse_sync_field)
    sync_df["chunk_start_real"] = pd.to_datetime(sync_df["timestamp"].apply(lambda t: t[0]))
    sync_df["file_num"] = sync_df["video"].apply(_file_index_from_video_list)
    sync_df = sync_df.dropna(subset=["file_num", "chunk_start_real"]).copy()
    sync_df["file_num"] = sync_df["file_num"].astype(int)

    start = sync_df["chunk_start_real"].min()
    sync_df["chunk_offset_sec"] = (sync_df["chunk_start_real"] - start).dt.total_seconds()
    return (dict(zip(sync_df["file_num"], sync_df["chunk_start_real"])),
            dict(zip(sync_df["file_num"], sync_df["chunk_offset_sec"])))


def add_exp_times(exp: int) -> pd.DataFrame:
    """Return calls for `exp` with experiment-seconds + wall-clock columns added.

    Reads calls.csv (output of combine_exp_calls.py) and the experiment's raw
    sync.csv; each call's per-file time is shifted by its chunk's offset from
    experiment start.

    Adds: channel (renamed from assigned_channel), start/stop_time_experiment_sec,
    start/stop_time_real.
    """
    calls_path = experiment_audio_dir(exp) / "calls.csv"
    if not calls_path.exists():
        raise FileNotFoundError(f"calls.csv not found: {calls_path}")
    file_to_real, file_to_offset = chunk_start_times(exp)

    calls = pd.read_csv(calls_path)
    if "assigned_channel" in calls.columns and "channel" not in calls.columns:
        calls = calls.rename(columns={"assigned_channel": "channel"})
    calls = calls.dropna(subset=["file_num"]).copy()
    calls["file_num"] = calls["file_num"].astype(int)

    missing = sorted({fn for fn in calls["file_num"].unique() if fn not in file_to_offset})
    if missing:
        print(f"    exp {exp}: {len(missing)} file_num(s) absent from sync.csv, dropped: {missing[:5]}")
        calls = calls[calls["file_num"].isin(file_to_offset)].copy()

    offsets = calls["file_num"].map(file_to_offset)
    chunk_real = calls["file_num"].map(file_to_real)
    calls["start_time_experiment_sec"] = calls["start_time_file_sec"] + offsets
    calls["stop_time_experiment_sec"] = calls["stop_time_file_sec"] + offsets
    calls["start_time_real"] = chunk_real + pd.to_timedelta(calls["start_time_file_sec"], unit="s")
    calls["stop_time_real"] = chunk_real + pd.to_timedelta(calls["stop_time_file_sec"], unit="s")
    return calls


def pool_date_folder(date_folder: str) -> tuple[pd.DataFrame, list[tuple[int, str]]]:
    """Concatenate every experiment in one date folder. Returns (df, failures)."""
    frames, failed = [], []
    for exp in list_experiment_ids_for_date(date_folder):
        try:
            df = add_exp_times(exp)
        except (FileNotFoundError, ValueError) as exc:
            failed.append((exp, str(exc)))
            continue
        df.insert(0, "exp", exp)
        frames.append(df)
    if not frames:
        return pd.DataFrame(), failed
    return pd.concat(frames, ignore_index=True), failed


def to_parquet(df: pd.DataFrame, date_folder: str, parquet_path: Path) -> Path:
    """Write the parquet cache, stamping date_folder and day_in_experiment.

    Same schema the notebook cell produced, so existing analysis reads it
    unchanged.
    """
    out = df.copy()
    for col in DATETIME_COLS:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")
    out["date_folder"] = date_folder
    if "start_time_real" in out.columns:
        out["day_in_experiment"] = (
            out["start_time_real"].dt.normalize().rank(method="dense").astype(int)
        )
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    return parquet_path


def run(date_folders: list[str], out_dir: Path, parquet_dir: Path, write_parquet: bool,
        dry_run: bool) -> int:
    for date_folder in date_folders:
        print(f"\n=== {date_folder}")
        df, failed = pool_date_folder(date_folder)
        if df.empty:
            print("  no calls.csv found — has combine_exp_calls.py run yet?")
            for exp, reason in failed[:5]:
                print(f"    skip {exp}: {reason[:110]}")
            continue

        n_exps = df["exp"].nunique()
        span = f"{df['start_time_real'].min()} .. {df['start_time_real'].max()}"
        print(f"  {len(df):,} calls from {n_exps} experiments")
        print(f"  spans {span}")
        if failed:
            print(f"  {len(failed)} experiment(s) skipped:")
            for exp, reason in failed[:5]:
                print(f"    {exp}: {reason[:110]}")

        csv_path = out_dir / f"all_calls_{date_folder}.csv"
        parquet_path = parquet_dir / f"all_calls_{date_folder}.parquet"
        if dry_run:
            print(f"  DRY RUN — would write {csv_path}")
            if write_parquet:
                print(f"  DRY RUN — would write {parquet_path}")
            continue

        out_dir.mkdir(parents=True, exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path}")
        if write_parquet:
            to_parquet(df, date_folder, parquet_path)
            print(f"  wrote {parquet_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pool per-experiment calls.csv into all_calls_<date>.csv + parquet cache.",
        epilog="Example: python scripts/pipeline/pool_calls.py --date-folder 2026_08",
    )
    p.add_argument("--date-folder", nargs="+", dest="date_folders",
                   help="Date folder(s) to pool. Default: every folder in experiments.toml.")
    p.add_argument("--out-dir", type=Path, default=ALL_CALLS_DIR,
                   help=f"Where the pooled CSV goes. Default: {ALL_CALLS_DIR}")
    p.add_argument("--parquet-dir", type=Path, default=PARQUET_DIR,
                   help=f"Where the parquet cache goes. Default: {PARQUET_DIR}")
    p.add_argument("--no-parquet", action="store_true", help="Write only the CSV.")
    p.add_argument("--dry-run", action="store_true", help="Report what would be written.")
    return p


def main() -> int:
    args = build_parser().parse_args()
    date_folders = args.date_folders or list_date_folders()
    return run(date_folders, args.out_dir, args.parquet_dir, not args.no_parquet, args.dry_run)


if __name__ == "__main__":
    raise SystemExit(main())
