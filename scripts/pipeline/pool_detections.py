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

    <date>/<exp>/detections.parquet     that experiment's detections, timestamped
    <date>/<exp>/coverage.parquet       one row per video of that experiment
    <date>/detections_<date>.parquet    every experiment concatenated
    <date>/coverage_<date>.parquet      ditto

The per-experiment files are the unit of work: pooling is a cheap concat of
them, so re-running after a few more experiments finish tracking does not
re-read every CSV (see --skip-existing).

The coverage table exists because a detections table alone cannot distinguish
"nobody was visible" from "that video was never tracked" — both are simply
absent rows. Roughly a third of the CSVs are legitimately empty, and some
experiments have only one camera tracked, so filling missing frames with zeros
would invent an empty arena. Bin occupancy against coverage, not against the
detections alone.

    python scripts/pipeline/pool_detections.py --date-folder 2026_02
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from scripts.pipeline.audio_processing_config import list_date_folders
from scripts.pipeline.paths import (
    pooled_coverage_path,
    pooled_detections_path,
    video_date_dir,
    video_detections_dir,
)
from scripts.pipeline.pool_calls import chunk_start_times

FPS = 30.0  # OUTPUT_FORMAT.md: seconds = frame_id / FPS

# Which arena each camera films, in the vocabulary calls.csv already uses
# (assigned_location: arena_1 / arena_2 / underground).
#
# VERIFIED 2026-08-17 against the calls themselves, on the 332 chunks of 2026_02
# where both cameras were tracked (97,159 calls). Presence in the same second as
# a call, versus that arena's baseline occupancy:
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
# The burrow and nest cameras are not tracked, so `underground` has no video
# coverage at all — occupancy there can only ever be inferred by subtraction.

CSV_PATTERN = re.compile(r"^video_(?P<camera>.+)_(?P<file_num>\d+)\.csv$")


def _parse_csv_name(path: Path) -> tuple[str, int] | None:
    m = CSV_PATTERN.match(path.name)
    if not m:
        return None
    return m.group("camera"), int(m.group("file_num"))


def load_experiment_detections(date_folder: str, exp: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (detections, coverage) for one experiment."""
    folder = video_detections_dir(date_folder, exp)
    if not folder.exists():
        raise FileNotFoundError(f"No tracking output for experiment {exp}: {folder}")

    file_to_real, _ = chunk_start_times(exp)

    det_frames, coverage = [], []
    for csv_path in sorted(folder.glob("video_*.csv")):
        parsed = _parse_csv_name(csv_path)
        if parsed is None:
            continue
        camera, file_num = parsed
        location = CAMERA_TO_LOCATION.get(camera)
        chunk_start = file_to_real.get(file_num)

        df = pd.read_csv(csv_path)
        coverage.append({
            "exp": exp,
            "location": location,
            "camera": camera,
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
        df["camera"] = camera
        df["file_num"] = file_num
        df["start_time_real"] = chunk_start + pd.to_timedelta(df["frame_id"] / FPS, unit="s")
        det_frames.append(df)

    cols = ["exp", "location", "camera", "file_num", "frame_id", "det_id", "conf",
            "center_x", "center_y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "start_time_real"]
    dets = pd.concat(det_frames, ignore_index=True)[cols] if det_frames else pd.DataFrame(columns=cols)
    return dets, pd.DataFrame(coverage)


def write_experiment(date_folder: str, exp: int, skip_existing: bool = False) -> tuple[Path, Path] | None:
    """Build one experiment's detections.parquet + coverage.parquet."""
    folder = video_detections_dir(date_folder, exp)
    det_path, cov_path = folder / "detections.parquet", folder / "coverage.parquet"
    if skip_existing and det_path.exists() and cov_path.exists():
        return det_path, cov_path
    dets, cov = load_experiment_detections(date_folder, exp)
    dets.to_parquet(det_path, index=False)
    cov.to_parquet(cov_path, index=False)
    return det_path, cov_path


def pool_date_folder(date_folder: str, skip_existing: bool = False
                     ) -> tuple[pd.DataFrame, pd.DataFrame, list[tuple[int, str]]]:
    """Write every experiment's files, then concatenate them for the date folder."""
    root = video_date_dir(date_folder)
    if not root.exists():
        raise FileNotFoundError(f"No tracking output for {date_folder}: {root}")
    exps = sorted(int(p.name) for p in root.iterdir() if p.is_dir() and p.name.isdigit())

    det_frames, cov_frames, failed = [], [], []
    for exp in exps:
        try:
            det_path, cov_path = write_experiment(date_folder, exp, skip_existing)
        except (FileNotFoundError, ValueError) as exc:
            failed.append((exp, str(exc)))
            continue
        dets, cov = pd.read_parquet(det_path), pd.read_parquet(cov_path)
        if not dets.empty:
            det_frames.append(dets)
        if not cov.empty:
            cov_frames.append(cov)
    dets = pd.concat(det_frames, ignore_index=True) if det_frames else pd.DataFrame()
    cov = pd.concat(cov_frames, ignore_index=True) if cov_frames else pd.DataFrame()
    return dets, cov, failed


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
            dets, cov, failed = pool_date_folder(date_folder, skip_existing)
        except FileNotFoundError as exc:
            print(f"  {exc}")
            continue
        if cov.empty:
            print("  no tracking CSVs found")
            continue

        n_exps = cov["exp"].nunique()
        empty = int((cov["n_detections"] == 0).sum())
        unmapped = sorted(set(cov.loc[cov["location"].isna(), "camera"]))
        print(f"  {len(dets):,} detections from {n_exps} experiments, {len(cov)} videos")
        print(f"  videos with zero detections: {empty}/{len(cov)} ({100*empty/len(cov):.0f}%)")
        print("  per-location coverage:")
        for (loc, cam), grp in cov.groupby([cov["location"].fillna("(unmapped)"), "camera"]):
            print(f"    {loc:<12} {cam:<14} {len(grp):4d} videos, "
                  f"{grp['exp'].nunique():2d} experiments, {int(grp['n_detections'].sum()):,} detections")
        if unmapped:
            print(f"  WARNING: cameras with no location mapping: {unmapped}")
        if not dets.empty:
            print(f"  spans {dets['start_time_real'].min()} .. {dets['start_time_real'].max()}")
        if failed:
            print(f"  {len(failed)} experiment(s) skipped:")
            for exp, reason in failed[:5]:
                print(f"    {exp}: {reason[:110]}")

        det_path, cov_path = pooled_detections_path(date_folder), pooled_coverage_path(date_folder)
        if dry_run:
            print(f"  DRY RUN — would write {det_path} and {cov_path}")
            continue
        dets.to_parquet(det_path, index=False)
        cov.to_parquet(cov_path, index=False)
        print(f"  wrote {det_path}")
        print(f"  wrote {cov_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Pool per-frame detections onto the calls' clock, with a coverage table.",
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
