"""Pool per-frame animal detections onto the same clock as the calls.

Input is the output of the Gerbil-Detection-and-Tracking repo (see its
OUTPUT_FORMAT.md): one CSV per video, under

    <Video root>/<date folder>/<exp>/video_<camera>_<file_num>.csv

**These are detections, not tracks.** One row = one animal seen in one frame.
There is no tracker, so `det_id` is an index within its frame and says nothing
about identity across frames — you can count animals and locate them, but you
cannot follow an individual.

Written per experiment, then pooled per date folder — mirroring how calls.csv
sits in each experiment folder and all_calls_<date> pools them:

    <date>/<exp>/detections.parquet      that experiment's detections, timestamped
    <date>/<exp>/files_vetted.csv        one row per video of that experiment
    <date>/detections_<date>.parquet     every experiment concatenated
    <date>/files_vetted_<date>.csv       ditto

Detections are parquet (millions of rows); files_vetted is CSV, since it is a
couple of thousand rows and being able to just open it is worth more than the
format. Read it back with read_files_vetted(), which restores the timestamp and
integer dtypes that a text round trip drops.

The per-experiment files are the unit of work: pooling is a cheap concat of
them, so re-running after a few more experiments finish tracking does not
re-read every CSV (see --skip-existing).

Detections carry a `stationary` flag: True where the exact coordinate recurs
1000+ times within an experiment+location, which means the detector locked onto a
fixed object rather than an animal. Filter it out for occupancy
(`det[~det.stationary]`); it is 19.9% of arena_2 in 2026_02.

`files_vetted` records which videos were actually looked at. A detections table
alone cannot tell "nobody was visible" from "that video was never tracked" —
both are simply absent rows. Roughly a third of the videos legitimately contain
zero detections, and some experiments have only one arena filmed, so binning
occupancy without consulting files_vetted would invent an empty arena.

    python scripts/pipeline/pool_detections.py --date-folder 2026_02
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.pipeline.audio_processing_config import list_date_folders
from scripts.pipeline.paths import (
    pooled_detections_path,
    pooled_files_vetted_path,
    video_date_dir,
    video_detections_dir,
)
from scripts.pipeline.pool_calls import chunk_start_times

FPS = 30.0  # OUTPUT_FORMAT.md: seconds = frame_id / FPS

# Which arena each camera films. Cameras appear only in the input filenames; in
# the data itself we use `location`, the same vocabulary calls.csv uses for
# assigned_location (arena_1 / arena_2 / underground).
#
# VERIFIED 2026-08-17 against the calls themselves, on the 332 chunks of 2026_02
# where both arenas were filmed (97,159 calls). Presence in the same second as a
# call, versus that arena's baseline occupancy:
#
#                        seen in arena_1   seen in arena_2
#   call -> arena_1           96.4%             56.3%
#   call -> arena_2           77.7%             93.4%
#   call -> underground       39.6%             30.5%
#   baseline (any second)     72.0%             67.7%
#
# Each arena's calls coincide with presence in THAT arena far above its baseline,
# and underground calls fall well below both — so this mapping is the right way
# round, and the audio-based arena assignment agrees with the video.
CAMERA_TO_LOCATION = {
    "center": "arena_1",
    "gily_center": "arena_2",
}
# The burrow and nest cameras are not tracked, so `underground` is never filmed —
# occupancy there can only ever be inferred by subtraction from colony size.

CSV_PATTERN = re.compile(r"^video_(?P<camera>.+)_(?P<file_num>\d+)\.csv$")

DETECTION_COLS = ["exp", "location", "file_num", "frame_id", "det_id", "conf",
                  "center_x", "center_y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
                  "start_time_real", "stationary"]

# A detection is marked `stationary` when its exact (center_x, center_y) recurs at
# least this often within one experiment+location. At 30 fps that is ~33 s at an
# identical sub-millimetre point, which an animal cannot do — it is the detector
# locking onto an object. In 2026_02 this is a piece of plastic a gerbil dragged
# into arena_2 on 2026-02-22, and it accounts for 19.9% of that arena's detections
# (68.5% in exp 508). Flagged, never dropped: the caller decides.
STATIONARY_REPEATS = 1000


def read_files_vetted(path: Path) -> pd.DataFrame:
    """Read a files_vetted CSV with its dtypes intact.

    CSV loses two things on the round trip: chunk_start_real comes back as text,
    and max_frame_id becomes float (so <NA> shows up as NaN and 10799 as
    10799.0). Restore both here so callers never have to think about it.
    """
    df = pd.read_csv(path, parse_dates=["chunk_start_real"])
    if "max_frame_id" in df.columns:
        df["max_frame_id"] = df["max_frame_id"].astype("Int64")
    return df


def _parse_csv_name(path: Path) -> tuple[str, int] | None:
    m = CSV_PATTERN.match(path.name)
    if not m:
        return None
    return m.group("camera"), int(m.group("file_num"))


def flag_stationary(dets: pd.DataFrame, min_repeats: int = STATIONARY_REPEATS) -> pd.Series:
    """True where a detection sits on an exact coordinate that recurs implausibly often."""
    counts = dets.groupby(["location", "center_x", "center_y"])["frame_id"].transform("size")
    return counts >= min_repeats


def load_experiment_detections(date_folder: str, exp: int,
                               stationary_repeats: int = STATIONARY_REPEATS
                               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detections, files_vetted) for one experiment."""
    folder = video_detections_dir(date_folder, exp)
    if not folder.exists():
        raise FileNotFoundError(f"No tracking output for experiment {exp}: {folder}")

    file_to_real, _ = chunk_start_times(exp)

    det_frames, vetted, unmapped = [], [], set()
    for csv_path in sorted(folder.glob("video_*.csv")):
        parsed = _parse_csv_name(csv_path)
        if parsed is None:
            continue
        camera, file_num = parsed
        location = CAMERA_TO_LOCATION.get(camera)
        if location is None:
            unmapped.add(camera)
            continue
        chunk_start = file_to_real.get(file_num)

        df = pd.read_csv(csv_path)
        vetted.append({
            "exp": exp,
            "location": location,
            "file_num": file_num,
            "chunk_start_real": chunk_start,
            "n_detections": len(df),
            "max_frame_id": int(df["frame_id"].max()) if len(df) else pd.NA,
            "has_video": csv_path.with_suffix(".mp4").exists(),
            "in_sync_csv": chunk_start is not None,
        })
        if df.empty:
            continue
        if chunk_start is None:
            print(f"    exp {exp} {csv_path.name}: file_num {file_num} absent from sync.csv, dropped")
            continue

        df["exp"] = exp
        df["location"] = location
        df["file_num"] = file_num
        df["start_time_real"] = chunk_start + pd.to_timedelta(df["frame_id"] / FPS, unit="s")
        det_frames.append(df)

    if unmapped:
        print(f"    exp {exp}: ignoring cameras with no location mapping: {sorted(unmapped)}")

    vetted_df = pd.DataFrame(vetted)
    if not vetted_df.empty:
        # nullable Int64, so "no detections" reads as <NA> instead of demoting the
        # whole column to float and showing frame numbers as 10799.0
        vetted_df["max_frame_id"] = vetted_df["max_frame_id"].astype("Int64")

    if det_frames:
        dets = pd.concat(det_frames, ignore_index=True)
        dets["stationary"] = flag_stationary(dets, stationary_repeats)
        dets = dets[DETECTION_COLS]
    else:
        dets = pd.DataFrame(columns=DETECTION_COLS)
    return dets, vetted_df


def write_experiment(date_folder: str, exp: int, skip_existing: bool = False,
                     stationary_repeats: int = STATIONARY_REPEATS) -> tuple[Path, Path]:
    """Build one experiment's detections.parquet + files_vetted.parquet."""
    folder = video_detections_dir(date_folder, exp)
    det_path, vetted_path = folder / "detections.parquet", folder / "files_vetted.csv"
    if skip_existing and det_path.exists() and vetted_path.exists():
        return det_path, vetted_path
    dets, vetted = load_experiment_detections(date_folder, exp, stationary_repeats)
    dets.to_parquet(det_path, index=False)
    vetted.to_csv(vetted_path, index=False)
    return det_path, vetted_path


def pool_date_folder(date_folder: str, skip_existing: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
    """Write every experiment's files, then concatenate them for the date folder."""
    root = video_date_dir(date_folder)
    if not root.exists():
        raise FileNotFoundError(f"No tracking output for {date_folder}: {root}")
    exps = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())

    det_frames, vetted_frames, failed = [], [], []
    for exp in exps:
        try:
            det_path, vetted_path = write_experiment(date_folder, exp, skip_existing)
        except (FileNotFoundError, ValueError) as exc:
            failed.append((exp, str(exc)))
            continue
        dets, vetted = pd.read_parquet(det_path), read_files_vetted(vetted_path)
        if not dets.empty:
            det_frames.append(dets)
        if not vetted.empty:
            vetted_frames.append(vetted)
    dets = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    vetted = pd.concat(vetted_frames, ignore_index=True) if vetted_frames else pd.DataFrame()
    return dets, vetted, failed


def run(date_folders: list[str], dry_run: bool, skip_existing: bool) -> int:
    for date_folder in date_folders:
        print(f"\n=== {date_folder}")
        try:
            if dry_run:
                root = video_date_dir(date_folder)
                exps = sorted(int(q.name) for q in root.iterdir() if q.is_dir() and q.name.isdigit())
                print(f"  DRY RUN — would write per-experiment files for {len(exps)} experiments,")
                print(f"  DRY RUN — then {pooled_detections_path(date_folder)}")
                continue
            dets, vetted, failed = pool_date_folder(date_folder, skip_existing)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue
        if vetted.empty:
            print("  no tracking CSVs found")
            continue

        empty = int((vetted["n_detections"] == 0).sum())
        print(f"  {len(dets):,} detections from {vetted['exp'].nunique()} experiments, {len(vetted)} videos")
        print(f"  videos with zero detections: {empty}/{len(vetted)} ({100*empty/len(vetted):.0f}%)")
        if "stationary" in dets.columns and len(dets):
            st = dets.groupby("location")["stationary"].mean()
            print("  flagged `stationary` (detector stuck on an object): "
                  + ", ".join(f"{k} {100*v:.1f}%" for k, v in st.items()))
        print("  per-location:")
        for loc, grp in vetted.groupby("location"):
            print(f"    {loc:<12} {len(grp):4d} videos, {grp['exp'].nunique():2d} experiments, "
                  f"{int(grp['n_detections'].sum()):,} detections")
        if not dets.empty:
            print(f"  spans {dets['start_time_real'].min()} .. {dets['start_time_real'].max()}")
        if failed:
            print(f"  {len(failed)} experiment(s) skipped:")
            for exp, reason in failed[:5]:
                print(f"    {exp}: {reason[:110]}")

        det_path, vetted_path = pooled_detections_path(date_folder), pooled_files_vetted_path(date_folder)
        dets.to_parquet(det_path, index=False)
        vetted.to_csv(vetted_path, index=False)
        print(f"  wrote {det_path}")
        print(f"  wrote {vetted_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pool per-frame detections onto the calls' clock, with a files_vetted table.",
        epilog="Example: python scripts/pipeline/pool_detections.py --date-folder 2026_02",
    )
    p.add_argument("--date-folder", nargs="+", dest="date_folders",
                   help="Date folder(s). Default: every folder in experiments.toml.")
    p.add_argument("--skip-existing", action="store_true",
                   help="Reuse an experiment's detections.parquet if it already exists "
                        "(incremental re-pool after more experiments finish tracking).")
    p.add_argument("--dry-run", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    return run(args.date_folders or list_date_folders(), args.dry_run, args.skip_existing)


if __name__ == "__main__":
    raise SystemExit(main())
