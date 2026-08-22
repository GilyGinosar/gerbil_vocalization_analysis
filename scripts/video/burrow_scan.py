#!/usr/bin/env python
"""Scan whole burrow_side videos for tunnel traverses -- no motion detector, no curation.

`burrow_landmarks.py` refines candidate events proposed by the tracking repo's
frame-differencing detector. At 3,775 videos per date folder that two-stage
arrangement stops making sense: the candidate step exists only to avoid decoding
whole videos, and decoding is 94% of the cost either way. So this scans the file
end to end and lets the landmarks *be* the detection.

Per frame it records four things, because once a frame is decoded everything you
might want from it is nearly free (background subtraction + blobs is 0.84 ms
against 14.1 ms to decode, the frame difference another 0.07 ms):

  n_animals  blobs of animal size in the tunnel -- 0, 1, or more
  x          centroid of the single animal, 0 = nest end, 1 = arena end
  area       its area in px, which flags two animals merged into one blob
  moved      changed pixels vs the previous frame

`x` gives the landmarks. `moved` is not used for detection -- it is the audit
column: an animal present but never moving for a long stretch means the
background model has absorbed something, which is the one failure mode
background subtraction has and frame differencing does not.

A traverse is then just: one animal, crossing L and then R (`to_arena`) or R and
then L (`to_nest`), with the crossing times interpolated between frames.

Outputs per video, under --out-dir:
    tracks/<video>.parquet     the per-frame track, so L/R can be re-swept and
                               new statistics computed without touching video again
    traverses.csv              one row per detected traverse
    tiles/<traverse>.jpg       cached frame strip, so curation sheets can be
                               rebuilt in seconds instead of re-decoding

    # one experiment
    python scripts/video/burrow_scan.py --exp 492 --out-dir exports/scan_2026_02

    # one video (what the Slurm array runs)
    python scripts/video/burrow_scan.py --video <path.mp4> --exp 492 --out-dir ...
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pipeline.paths import BASE_RAW  # noqa: E402
from scripts.video.burrow_landmarks import (  # noqa: E402
    DEF_ROI, DW, DH, FG_THRESH, MIN_BLOB_PX, blobs_in,
)
from scripts.video.burrow_transit_picker import FPS, file_index  # noqa: E402

LEFT, RIGHT = 0.15, 0.75
BG_SAMPLE = 10          # every Nth frame builds the background median
DIFF_THRESH = 25        # per-pixel change counted as motion (audit column only)
MAX_TRAVERSE_S = 60.0   # refuse to pair landmark crossings further apart than this
MULTI_FRAC = 0.25       # traverse dropped if this fraction of its frames has >1 animal
TILE_FPS = 2            # cached strip frame rate
TILE_W = 300
BEFORE_S, AFTER_S = 2.0, 1.0


def decode_all(video: Path, roi) -> list[np.ndarray]:
    """Every frame of the video as a small grey ROI crop, in one forward pass."""
    x1, y1, x2, y2 = roi
    cap = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(cv2.resize(frame[y1:y2, x1:x2], (DW, DH)),
                                   cv2.COLOR_BGR2GRAY))
    cap.release()
    return frames


def build_track(frames: list[np.ndarray], background: np.ndarray) -> pd.DataFrame:
    """Per-frame n_animals / x / area / moved."""
    n_animals, xs, areas, moved = [], [], [], []
    previous = None
    for frame in frames:
        found = blobs_in(frame, background)
        n_animals.append(len(found))
        xs.append(found[0][0] if found else np.nan)
        areas.append(found[0][1] if found else 0)
        moved.append(int((cv2.absdiff(frame, previous) > DIFF_THRESH).sum())
                     if previous is not None else 0)
        previous = frame
    return pd.DataFrame({
        "frame": np.arange(len(frames), dtype=np.int32),
        "n_animals": np.asarray(n_animals, np.int8),
        "x": np.asarray(xs, np.float32),
        "area": np.asarray(areas, np.int32),
        "moved": np.asarray(moved, np.int32),
    })


def landmark_crossings(track: pd.DataFrame, level: float) -> list[tuple[float, int]]:
    """(time, direction) for every single-animal crossing of `level`.

    Only consecutive frames are joined: a gap where the animal was absent or
    doubled is not a crossing, it is missing data.
    """
    single = track[(track.n_animals == 1) & track.x.notna()]
    frame = single.frame.to_numpy()
    x = single.x.to_numpy()
    adjacent = np.diff(frame) == 1
    x0, x1 = x[:-1], x[1:]
    hit = adjacent & (((x0 < level) & (x1 >= level)) | ((x0 > level) & (x1 <= level)))
    out = []
    for i in np.flatnonzero(hit):
        span = x1[i] - x0[i]
        frac = 0.0 if span == 0 else (level - x0[i]) / span
        out.append(((frame[i] + frac) / FPS, 1 if span > 0 else -1))
    return out


def find_traverses(track: pd.DataFrame, left: float, right: float) -> list[dict]:
    """Pair landmark crossings into traverses.

    A `to_arena` traverse is a rightward crossing of L followed by a rightward
    crossing of R with no leftward L crossing in between -- i.e. the animal did
    not turn back and start again. `to_nest` is the mirror.
    """
    marks = ([(t, "L", d) for t, d in landmark_crossings(track, left)]
             + [(t, "R", d) for t, d in landmark_crossings(track, right)])
    marks.sort()
    traverses = []
    for i, (t_entry, mark, direction) in enumerate(marks):
        if mark == "L" and direction == 1:
            want, label = "R", "to_arena"
        elif mark == "R" and direction == -1:
            want, label = "L", "to_nest"
        else:
            continue
        for t_exit, mark2, dir2 in marks[i + 1:]:
            if t_exit - t_entry > MAX_TRAVERSE_S:
                break
            if mark2 == mark:            # came back to the start line: not this one
                break
            if mark2 == want and dir2 == direction:
                traverses.append({"direction": label, "t_entry": round(t_entry, 4),
                                  "t_exit": round(t_exit, 4),
                                  "traverse_s": round(t_exit - t_entry, 4)})
                break
    return traverses


def strip_for(frames: list[np.ndarray], t_entry: float, t_exit: float,
              roi, full: Path) -> np.ndarray | None:
    """The cached frame strip for one traverse, at TILE_FPS."""
    x1, y1, x2, y2 = roi
    tile_h = int(round(TILE_W * (y2 - y1) / (x2 - x1)))
    times = np.arange(t_entry - BEFORE_S, t_exit + AFTER_S, 1.0 / TILE_FPS)
    cap = cv2.VideoCapture(str(full))
    tiles = []
    position = 0
    for t in times:
        target = int(round(t * FPS))
        if target < 0 or target >= len(frames):
            tiles.append(np.zeros((tile_h, TILE_W, 3), np.uint8))
            continue
        if target < position or target - position > 250:
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            position = target
        while position < target:
            if not cap.grab():
                break
            position += 1
        ok, frame = cap.read()
        position += 1
        if not ok:
            tiles.append(np.zeros((tile_h, TILE_W, 3), np.uint8))
            continue
        tile = cv2.resize(frame[y1:y2, x1:x2], (TILE_W, tile_h))
        tiles.append(cv2.cvtColor(cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR))
    cap.release()
    return cv2.hconcat(tiles) if tiles else None


def scan_video(video: Path, roi, left: float, right: float, out_dir: Path,
               want_tiles: bool) -> list[dict]:
    frames = decode_all(video, roi)
    if not frames:
        print(f"{video.name}: decoded nothing", flush=True)
        return []
    background = np.median(np.stack(frames[::BG_SAMPLE]), axis=0).astype(np.uint8)
    track = build_track(frames, background)

    (out_dir / "tracks").mkdir(parents=True, exist_ok=True)
    track.to_parquet(out_dir / "tracks" / f"{video.stem}.parquet", index=False)

    # background fingerprint: the empty tunnel's own column profile. The camera is
    # not supposed to move within a date folder; if one does, this shifts and the
    # video shows up as an outlier instead of quietly producing wrong landmarks.
    profile = background.mean(axis=0)
    fingerprint = float(np.average(np.arange(DW), weights=profile) / DW)
    occupied = float((track.n_animals > 0).mean())

    rows = []
    for traverse in find_traverses(track, left, right):
        window = track[(track.frame >= traverse["t_entry"] * FPS)
                       & (track.frame <= traverse["t_exit"] * FPS)]
        multi = float((window.n_animals > 1).mean()) if len(window) else 0.0
        still = float((window.moved < 250).mean()) if len(window) else 0.0
        row = {"video": video.name, **traverse,
               "multi_animal_frac": round(multi, 3),
               "still_frac": round(still, 3),
               "single_animal": multi < MULTI_FRAC,
               "occupied_frac": round(occupied, 3),
               "bg_fingerprint": round(fingerprint, 4)}
        if want_tiles and row["single_animal"]:
            strip = strip_for(frames, traverse["t_entry"], traverse["t_exit"], roi, video)
            if strip is not None:
                (out_dir / "tiles").mkdir(parents=True, exist_ok=True)
                name = f"{video.stem}_t{traverse['t_entry']:.2f}".replace(".", "_") + ".jpg"
                cv2.imwrite(str(out_dir / "tiles" / name), strip,
                            [cv2.IMWRITE_JPEG_QUALITY, 85])
                row["tile"] = f"tiles/{name}"
        rows.append(row)
    print(f"{video.name}: {len(rows)} traverses, {occupied:.0%} occupied", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--video", help="one video (the Slurm array unit); default: all in the experiment")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datadir")
    parser.add_argument("--roi", default=",".join(map(str, DEF_ROI)))
    parser.add_argument("--left", type=float, default=LEFT)
    parser.add_argument("--right", type=float, default=RIGHT)
    parser.add_argument("--no-tiles", action="store_true", help="skip the cached frame strips")
    parser.add_argument("--limit", type=int, help="only the first N videos (for a trial run)")
    args = parser.parse_args()

    roi = tuple(int(v) for v in args.roi.split(","))
    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")
    out_dir = Path(args.out_dir) / str(args.exp)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = ([Path(args.video)] if args.video
              else sorted(datadir.glob("video_burrow_side_*.mp4")))
    if args.limit:
        videos = videos[:args.limit]
    if not videos:
        raise SystemExit(f"no burrow_side videos in {datadir}")

    rows = []
    for video in videos:
        rows.extend(scan_video(video, roi, args.left, args.right, out_dir, not args.no_tiles))

    suffix = f"_{Path(args.video).stem}" if args.video else ""
    path = out_dir / f"traverses{suffix}.csv"
    fields = ["video", "direction", "t_entry", "t_exit", "traverse_s", "multi_animal_frac",
              "still_frac", "single_animal", "occupied_frac", "bg_fingerprint", "tile"]
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    clean = [r for r in rows if r["single_animal"]]
    print(f"\n{len(videos)} videos -> {len(rows)} traverses ({len(clean)} single-animal) -> {path}")


if __name__ == "__main__":
    main()
