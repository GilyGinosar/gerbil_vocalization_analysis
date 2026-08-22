"""Reduce a date folder's video + calls into small tables you can hold at once.

The raw detections are ~260 MB per experiment, so cohort-wide behavioural work
cannot load them all. But almost every behavioural question only needs *counts
per unit time* and *counts per place* — and those are tiny. This walks the
experiments once and writes the reduction:

    behaviour/<date>/seconds_<date>.parquet     one row per (exp, location, second)
    behaviour/<date>/cells_<date>.parquet       one row per (exp, location, x, y) cell

**Per second, not per minute**, so downstream can re-bin to whatever it wants
without rebuilding: 1.4M rows for a 10-day cohort, about 15 MB.

The division into experiments is a recording artefact — a date folder is one
continuous experiment — so these tables carry `exp` but you are meant to ignore
it for behaviour and group by time instead. Keep the raw per-experiment files
for the analyses that genuinely are per experiment, like neural recordings.

    python scripts/pipeline/build_behaviour.py --date-folder 2026_02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.pipeline.audio_processing_config import list_date_folders
from scripts.pipeline.paths import PROCESSED_ROOT
from scripts.pipeline.pool_calls import add_exp_times
from scripts.utils.tracking_io import (FPS, experiments_in, load_detections,
                                       load_files_vetted, video_durations)

BEHAVIOUR_ROOT = PROCESSED_ROOT / "behaviour"

ARENA_WIDTH_CM, ARENA_HEIGHT_CM, BIN_CM = 121, 91, 2
X_EDGES = np.arange(0, ARENA_WIDTH_CM + BIN_CM, BIN_CM)
Y_EDGES = np.arange(0, ARENA_HEIGHT_CM + BIN_CM, BIN_CM)




def seconds_path(date_folder: str) -> Path:
    return BEHAVIOUR_ROOT / date_folder / f"seconds_{date_folder}.parquet"


def cells_path(date_folder: str) -> Path:
    return BEHAVIOUR_ROOT / date_folder / f"cells_{date_folder}.parquet"


def filmed_seconds(date_folder: str) -> pd.DataFrame:
    """Every (exp, location, second) that a camera actually recorded.

    This is the backbone: a second present here with no detections is a real
    zero, and a second missing from here was never observed.

    Bounded by the AUDIO, not the video. Durations are measured per video (the
    last chunk of an experiment runs until recording stopped, sometimes for
    hours) but audio stops long before that -- 25.5 h of 2026_02 is video with no
    audio. Since this table carries call counts, including those seconds would
    invent silence.
    """
    rows = []
    for video in video_durations(date_folder).itertuples():
        start = video.chunk_start_real.floor("s")
        for offset in range(int(round(video.duration_s))):
            rows.append({"exp": video.exp, "location": video.location,
                         "second": start + pd.Timedelta(seconds=offset)})
    return pd.DataFrame(rows).drop_duplicates()


def detections_per_second(date_folder: str, exp: int) -> pd.DataFrame:
    detections = load_detections(date_folder, exp=exp, quiet=True,
                                 columns=["exp", "location", "start_time_real"])
    if detections.empty:
        return pd.DataFrame(columns=["exp", "location", "second", "n_detections"])
    detections = detections.copy()
    detections["second"] = detections["start_time_real"].dt.floor("s")
    return (detections.groupby(["exp", "location", "second"], observed=True)
                      .size().reset_index(name="n_detections"))


def calls_per_second(exp: int) -> pd.DataFrame:
    calls = add_exp_times(exp)
    calls["second"] = calls["start_time_real"].dt.floor("s")
    counted = (calls.groupby(["assigned_location", "second"])
                    .size().reset_index(name="n_calls"))
    counted = counted.rename(columns={"assigned_location": "location"})
    counted["exp"] = exp
    return counted


def occupancy_cells(date_folder: str, exp: int) -> pd.DataFrame:
    """Animal-seconds per 2 cm cell, for one experiment and each location."""
    detections = load_detections(date_folder, exp=exp, quiet=True,
                                 columns=["location", "center_x", "center_y"])
    rows = []
    for location, points in detections.groupby("location", observed=True):
        counts, _, _ = np.histogram2d(points.center_x, points.center_y,
                                      bins=[X_EDGES, Y_EDGES])
        x_index, y_index = np.nonzero(counts)
        rows.append(pd.DataFrame({
            "exp": exp,
            "location": location,
            "x_cm": X_EDGES[x_index],
            "y_cm": Y_EDGES[y_index],
            "animal_seconds": counts[x_index, y_index] / FPS,
        }))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build(date_folder: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    backbone = filmed_seconds(date_folder)

    per_second, per_cell = [], []
    for exp in experiments_in(date_folder):
        per_second.append(detections_per_second(date_folder, exp))
        per_second.append(calls_per_second(exp))
        per_cell.append(occupancy_cells(date_folder, exp))

    detections_table = pd.concat([t for t in per_second if "n_detections" in t], ignore_index=True)
    calls_table = pd.concat([t for t in per_second if "n_calls" in t], ignore_index=True)

    seconds = backbone.merge(detections_table, on=["exp", "location", "second"], how="left")
    seconds = seconds.merge(calls_table, on=["exp", "location", "second"], how="left")
    seconds["n_detections"] = seconds["n_detections"].fillna(0).astype(int)
    seconds["n_calls"] = seconds["n_calls"].fillna(0).astype(int)
    seconds["animals"] = seconds["n_detections"] / FPS

    cells = pd.concat([t for t in per_cell if len(t)], ignore_index=True)
    return seconds, cells


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--date-folder", nargs="+", dest="date_folders")
    args = p.parse_args()

    for date_folder in args.date_folders or list_date_folders():
        print(f"\n=== {date_folder}")
        try:
            seconds, cells = build(date_folder)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue

        out = BEHAVIOUR_ROOT / date_folder
        out.mkdir(parents=True, exist_ok=True)
        seconds.to_parquet(seconds_path(date_folder), index=False)
        cells.to_parquet(cells_path(date_folder), index=False)

        hours = seconds.groupby("location", observed=True)["animals"].sum() / 3600
        print(f"  {len(seconds):,} filmed seconds, {len(cells):,} occupied cells")
        print(f"  animal-hours: {hours.round(1).to_dict()}")
        print(f"  calls in filmed seconds: {int(seconds.n_calls.sum()):,}")
        print(f"  wrote {seconds_path(date_folder)}  "
              f"({seconds_path(date_folder).stat().st_size/1e6:.1f} MB)")
        print(f"  wrote {cells_path(date_folder)}  "
              f"({cells_path(date_folder).stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
