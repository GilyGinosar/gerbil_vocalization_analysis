"""Loading the pooled tracking data, with its traps already handled.

The video-side counterpart of ``ethogram_io`` for calls. Use this rather than
reading the parquet directly, because two mistakes are easy to make and silent:

1. **Stationary detections.** The detector locks onto fixed objects (a piece of
   plastic in arena_2). ``pool_detections`` already removes them from the *pooled*
   file, so it is clean whatever you use to read it. The *per-experiment* files
   deliberately keep them as an audit trail, so this module drops them there.
2. **Size.** The pooled file is tens of millions of rows and a JupyterHub session
   here gets 16 GB, so loading it whole leaves little room. **Work one experiment
   at a time** — ``load_detections(date, exp=...)`` reads that experiment's own
   small file — and aggregate the small results. That is also the right unit
   scientifically: occupancy drifts between experiments, so a rate map's
   denominator has to come from the same experiment as its events.

3. **Untracked videos.** "Nobody visible" and "we never looked" are both absent
   rows in the detections. ``filmed_minutes`` builds the list of minutes that
   were actually filmed, so a quiet minute counts as zero and an untracked one
   does not count at all.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.pipeline.paths import (pooled_detections_path, pooled_files_vetted_path,
                                    video_detections_dir)
from scripts.pipeline.pool_detections import read_files_vetted

FPS = 30
FRAMES_PER_MINUTE = FPS * 60
NOMINAL_VIDEO_SECONDS = 360            # a recording chunk is 6 minutes...
                                       # ...except the LAST of each experiment, which runs until
                                       # recording stopped and can be hours long (up to 4.7 h in
                                       # 2026_02). Assuming 360 s there undercounts observed time
                                       # and silently drops the detections in the tail.


def load_files_vetted(date_folder: str) -> pd.DataFrame:
    """One row per video that was tracked."""
    return read_files_vetted(pooled_files_vetted_path(date_folder))


def experiments_in(date_folder: str) -> list[int]:
    """Experiment ids with pooled detections, oldest first."""
    return sorted(load_files_vetted(date_folder).exp.unique().tolist())


def load_detections(date_folder: str, exp: int | None = None,
                    include_stationary: bool = False,
                    columns: list[str] | None = None, location: str | None = None,
                    quiet: bool = False) -> pd.DataFrame:
    """Pooled detections for a date folder, stationary ones removed by default.

    Pass ``location="arena_1"`` to read only that arena — the filter is pushed down
    to the file, so it halves the memory instead of loading everything first.
    These files are tens of millions of rows; ask for the columns you need.

    Pass ``exp=`` to read a single experiment's file instead of the pooled one —
    much lighter, and the right choice inside a loop.

    Pass ``include_stationary=True`` only to inspect the artifact itself.
    """
    if columns is not None and exp is not None and "stationary" not in columns:
        columns = list(columns) + ["stationary"]     # per-experiment files still carry it

    # One experiment at a time reads its own small file; the pooled one is tens of
    # millions of rows and will exhaust memory if you hold it and a copy.
    if exp is None:
        path = pooled_detections_path(date_folder)
    else:
        path = video_detections_dir(date_folder, exp) / "detections.parquet"

    filters = [("location", "==", location)] if location else None
    detections = pd.read_parquet(path, columns=columns, filters=filters)
    # a handful of repeated strings: category costs a fraction of the memory
    if "location" in detections.columns:
        detections["location"] = detections["location"].astype("category")

    if include_stationary or "stationary" not in detections.columns:
        # The pooled file has no `stationary` column: pooling already dropped them.
        return detections

    n_before = len(detections)
    detections = detections[~detections["stationary"]]

    if not quiet:
        dropped = n_before - len(detections)
        print(f"{date_folder}: {len(detections):,} detections "
              f"({dropped:,} stationary dropped, {100*dropped/max(n_before,1):.1f}%)")
        warn_about_fallback(date_folder)

    return detections


def warn_about_fallback(date_folder: str) -> None:
    """Say which experiments still use our coarser stationary rule.

    The tracking repo computes the flag properly; until it has reached every
    experiment, the rest fall back to a local rule that under-flags. Those
    experiments are usable but provisional.
    """
    vetted = load_files_vetted(date_folder)
    if "stationary_source" not in vetted.columns:
        return
    fallback = vetted[vetted.stationary_source == "fallback"]
    if len(fallback):
        exps = sorted(fallback.exp.unique())
        print(f"  note: {len(exps)} experiment(s) still use the fallback flag and are "
              f"under-flagged: {exps}")


def video_durations(date_folder: str, exp: int | None = None) -> pd.DataFrame:
    """How long each video actually ran, in seconds.

    Measured, not assumed: the gap to the next chunk of the same experiment and
    location. The final chunk has no next one, so it falls back to its last
    detected frame, which is a lower bound but far better than 360 s.
    """
    vetted = load_files_vetted(date_folder)
    if exp is not None:
        vetted = vetted[vetted.exp == exp]

    vetted = vetted.sort_values(["exp", "location", "file_num"]).copy()
    next_start = vetted.groupby(["exp", "location"], observed=True)["chunk_start_real"].shift(-1)
    vetted["duration_s"] = (next_start - vetted["chunk_start_real"]).dt.total_seconds()

    # final chunk of each experiment: use the last frame we saw an animal in
    from_frames = (vetted["max_frame_id"].astype("Float64") + 1) / FPS
    vetted["duration_s"] = vetted["duration_s"].fillna(from_frames.astype(float))
    vetted["duration_s"] = vetted["duration_s"].fillna(NOMINAL_VIDEO_SECONDS)
    vetted.loc[vetted.duration_s < NOMINAL_VIDEO_SECONDS, "duration_s"] = NOMINAL_VIDEO_SECONDS
    return vetted


def filmed_minutes(date_folder: str, exp: int | None = None) -> pd.DataFrame:
    """Every (exp, location, minute) that was actually filmed.

    Merge occupancy onto this so a filmed-but-quiet minute is a real zero, and an
    untracked minute is simply absent rather than a fake zero.
    """
    rows = []
    for video in video_durations(date_folder, exp).itertuples():
        minutes = int(np.ceil(video.duration_s / 60))
        start = video.chunk_start_real.floor("1min")
        for minute in range(minutes):
            rows.append({"exp": video.exp,
                         "location": video.location,
                         "minute": start + pd.Timedelta(minutes=minute)})
    return pd.DataFrame(rows).drop_duplicates()


def animals_per_minute(date_folder: str, exp: int | None = None) -> pd.DataFrame:
    """Mean animals visible per frame, for each location and minute.

    Zero where the video was tracked and nobody was seen; absent where untracked.
    """
    # Read only what we need: with exp= this is one experiment's small file, not
    # the pooled 26M-row one. Loading the pooled file per experiment in a loop is
    # what used to exhaust the 16 GB the JupyterHub job allows.
    detections = load_detections(date_folder, exp=exp, quiet=True,
                                 columns=["exp", "location", "start_time_real"])
    detections = detections.copy()
    detections["minute"] = detections["start_time_real"].dt.floor("1min")

    counted = (detections.groupby(["exp", "location", "minute"], observed=True)
                     .size().reset_index(name="n_detections"))
    counted["animals"] = counted["n_detections"] / FRAMES_PER_MINUTE

    table = filmed_minutes(date_folder, exp).merge(
        counted[["exp", "location", "minute", "animals"]],
        on=["exp", "location", "minute"], how="left")
    table["animals"] = table["animals"].fillna(0)
    return table


# --- the reduced behavioural tables (built by scripts/pipeline/build_behaviour.py) ---
#
# Cohort-wide behaviour cannot hold the raw detections (~260 MB per experiment),
# but it does not need them: counts per second and per place are ~10 MB for a
# whole date folder. Load these instead, and keep the raw files for the analyses
# that really are per experiment, such as neural recordings.

def load_behaviour_seconds(date_folder: str) -> pd.DataFrame:
    """One row per (exp, location, filmed second): n_detections, animals, n_calls.

    Re-bin to whatever resolution you want, e.g.
        seconds.groupby([seconds.second.dt.floor("1min"), "location"]).agg(...)
    """
    from scripts.pipeline.build_behaviour import seconds_path
    return pd.read_parquet(seconds_path(date_folder))


def load_occupancy_cells(date_folder: str) -> pd.DataFrame:
    """One row per (exp, location, 2 cm cell): animal_seconds spent there."""
    from scripts.pipeline.build_behaviour import cells_path
    return pd.read_parquet(cells_path(date_folder))
