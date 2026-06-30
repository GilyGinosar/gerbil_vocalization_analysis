"""Shared I/O + binning helpers for the call-analysis scripts.

Extracted from ``run_ethogram.py`` so the committed analysis scripts
(``run_ethogram_categorical``, ``run_bout_raster``, ``run_call_correlogram``)
don't depend on the larger exploratory ``run_ethogram`` module. Single source
for the pooled-call loader, the experiment / recording-coverage helpers, and the
circadian-day grids.
"""
from __future__ import annotations

import platform
from pathlib import Path

import numpy as np
import pandas as pd

# --- Display / grouping constants (shared across the ethogram-family scripts) ---
CALL_TYPE_ORDER = ["alarm", "high-freq", "warble", "stacks", "newborn"]
LOCATION_GROUPS = {                    # display group -> assigned_location values pooled into it
    "arena": ["arena_1", "arena_2"],
    "underground": ["underground"],
}
ROW_START_HOUR = 4                     # clock hour each circadian-day row starts at (4 = lights-on)
# Manually-noted events to annotate (red vertical line + label), per date folder.
EVENTS_BY_DATE: dict[str, list[tuple[str, str]]] = {
    "2026_02": [("2026-03-01 23:00", "new litter born")],
}

if platform.system() == "Windows":
    BASE_PROCESSED = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    BASE_PROCESSED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio")


def all_calls_path(date_folder: str) -> tuple[Path, str]:
    """Prefer the parquet cache; fall back to the CSV."""
    parquet = BASE_PROCESSED / "all_calls" / "parquet_cache" / f"all_calls_{date_folder}.parquet"
    if parquet.exists():
        return parquet, "parquet"
    csv = BASE_PROCESSED / "all_calls" / f"all_calls_{date_folder}.csv"
    return csv, "csv"


def load_all_calls(date_folder: str) -> pd.DataFrame:
    path, kind = all_calls_path(date_folder)
    if not path.exists():
        raise FileNotFoundError(f"No pooled calls file for {date_folder}: {path}")
    df = pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)
    df["start_time_real"] = pd.to_datetime(df["start_time_real"])
    print(f"{date_folder}: loaded {len(df):,} calls from {kind} ({path.name})")
    return df


def list_experiment_dirs(date_folder: str) -> list[Path]:
    folder = BASE_PROCESSED / date_folder
    if not folder.exists():
        raise FileNotFoundError(f"Date folder not found: {folder}")
    return sorted(
        (p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def make_bin_edges(df: pd.DataFrame, bin_minutes: int) -> pd.DatetimeIndex:
    freq = f"{bin_minutes}min"
    start = df["start_time_real"].min().floor(freq)
    end = df["start_time_real"].max().ceil(freq)
    return pd.date_range(start, end, freq=freq)


def recording_coverage_minutes(date_folder: str, bin_edges: pd.DatetimeIndex) -> np.ndarray:
    """Recorded minutes per time-bin, summed over all experiments' file_times.csv.

    Each ``file_times.csv`` row is a recorded chunk [start, end]; chunks are
    contiguous within an experiment and gappy between experiment restarts. We
    sum each chunk's overlap with every bin it touches.
    """
    edges_ns = bin_edges.asi8                       # int64 nanoseconds
    n_bins = len(bin_edges) - 1
    covered_ns = np.zeros(n_bins, dtype=np.float64)

    for exp_dir in list_experiment_dirs(date_folder):
        ft_path = exp_dir / "file_times.csv"
        if not ft_path.exists():
            continue
        ft = pd.read_csv(ft_path)
        starts = pd.to_datetime(ft["start_date"] + " " + ft["start_time"]).values.astype("int64")
        ends = pd.to_datetime(ft["end_date"] + " " + ft["end_time"]).values.astype("int64")
        for s, e in zip(starts, ends):
            i0 = max(np.searchsorted(edges_ns, s, "right") - 1, 0)
            i1 = min(np.searchsorted(edges_ns, e, "right") - 1, n_bins - 1)
            for b in range(i0, i1 + 1):
                lo = max(s, edges_ns[b])
                hi = min(e, edges_ns[b + 1])
                if hi > lo:
                    covered_ns[b] += hi - lo
    return covered_ns / 1e9 / 60.0                  # ns -> seconds -> minutes


def day_axis(df: pd.DataFrame, x0: int) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """Row dates for an actogram whose day starts at hour ``x0``.

    A "circadian day" runs [x0:00 of date D, x0:00 of D+1); a timestamp belongs
    to the row of ``(t - x0 h).date``. ``day0`` is the first such row date.
    """
    shifted = df["start_time_real"] - pd.Timedelta(hours=x0)
    day0 = shifted.min().normalize()
    days = pd.date_range(day0, shifted.max().normalize(), freq="D")
    return day0, days


def counts_grid(
    df_group: pd.DataFrame, day0: pd.Timestamp, n_days: int,
    n_hourbins: int, bin_minutes: int, x0: int = 0,
) -> np.ndarray:
    """(n_days, n_hourbins) raw call counts for one call-type × location subset.

    Rows are circadian days starting at hour ``x0``; within a row, time runs
    monotonically from ``x0`` (left) to ``x0 + 24`` (right).
    """
    t = df_group["start_time_real"]
    shifted = t - pd.Timedelta(hours=x0)
    day_idx = (shifted.dt.normalize() - day0).dt.days.to_numpy()
    mins = ((t.dt.hour * 60 + t.dt.minute) - x0 * 60) % 1440
    hb = (mins // bin_minutes).to_numpy().astype(int)
    grid = np.zeros((n_days, n_hourbins), dtype=float)
    ok = (day_idx >= 0) & (day_idx < n_days)
    np.add.at(grid, (day_idx[ok], hb[ok]), 1.0)
    return grid


def coverage_grid(
    date_folder: str, day0: pd.Timestamp, n_days: int,
    n_hourbins: int, bin_minutes: int, x0: int = 0,
) -> np.ndarray:
    """(n_days, n_hourbins) recorded minutes per (circadian-day, hour) cell."""
    cov = np.zeros((n_days, n_hourbins), dtype=float)
    x0h = pd.Timedelta(hours=x0)
    for exp_dir in list_experiment_dirs(date_folder):
        ft_path = exp_dir / "file_times.csv"
        if not ft_path.exists():
            continue
        ft = pd.read_csv(ft_path)
        starts = pd.to_datetime(ft["start_date"] + " " + ft["start_time"])
        ends = pd.to_datetime(ft["end_date"] + " " + ft["end_time"])
        for s, e in zip(starts, ends):
            cur = s
            while cur < e:                       # walk the chunk cell-by-cell
                d_idx = ((cur - x0h).normalize() - day0).days
                mod = cur.hour * 60 + cur.minute
                cell_end = cur.normalize() + pd.Timedelta(minutes=(mod // bin_minutes + 1) * bin_minutes)
                seg_end = min(e, cell_end)
                if 0 <= d_idx < n_days:
                    hb = int(((mod - x0 * 60) % 1440) // bin_minutes)
                    cov[d_idx, hb] += (seg_end - cur).total_seconds()
                cur = seg_end
    return cov / 60.0
