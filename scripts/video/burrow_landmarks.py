#!/usr/bin/env python
"""Precise tunnel-crossing landmarks by tracking the animal's body, not its motion.

`burrow_transits.py` finds candidate crossings by frame differencing, and its
`start_s` is the frame where the changed-pixel count first passes a threshold.
That boundary moves with the animal's speed and contrast, and the detector's
gap-bridging absorbs up to 0.5 s inside an event while its merge step joins runs
up to 3 s apart -- so `start_s` is a good proxy for "something happened here" and
a poor one for "the animal was at this point in the tunnel at this instant".

This script replaces it with a geometric landmark. For each candidate event it:

1. Builds a background image of the empty tunnel (per-pixel median over frames
   sampled across that video's event windows -- animals move, so the median is
   the empty tube).
2. Subtracts it per frame and keeps blobs of animal size. Unlike frame
   differencing this sees a **stationary** animal, so a pause does not break the
   track and no gap-bridging is needed.
3. Counts blobs -> frames with two animals in the tunnel are flagged, and events
   that are mostly two-animal are dropped (they make a centroid meaningless).
4. Records the single animal's centroid x each frame as a fraction of the tunnel
   (0 = nest end, 1 = arena end) -- the track, written out per event.
5. Reports the times the track crosses the two landmarks L and R, linearly
   interpolated between the straddling frames, so timing is sub-frame rather
   than threshold-dependent.

Direction falls out of the order: L before R is `to_arena`, R before L is
`to_nest`. An event that never crosses both never traversed the tunnel.

**The tracks are written to disk** so L and R can be re-swept later without
re-decoding any video (decoding is the expensive part -- the mp4s are on ceph
with 8.3 s keyframes).

    python scripts/video/burrow_landmarks.py \
        --from-csv .../transits_492_curated.csv --exp 492 --out-dir exports/burrow_landmarks_492

    # inspect the segmentation before trusting any of it
    python scripts/video/burrow_landmarks.py ... --debug-frames 12
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.pipeline.paths import BASE_RAW  # noqa: E402
from scripts.video.burrow_transit_picker import (  # noqa: E402
    DEF_ROI, FPS, GOP_FRAMES, file_index,
)

DW, DH = 460, 145      # ROI working resolution (half of the 920x290 crop)
PAD_S = 3.0            # seconds decoded either side of each candidate event
FG_THRESH = 22         # grey levels above background to count as foreground
MIN_BLOB_PX = 900      # smallest blob treated as an animal, at DWxDH
BG_SAMPLE = 4          # use every Nth decoded frame to build the background
MULTI_FRAC = 0.25      # drop an event if this fraction of its frames has >1 animal

LEFT, RIGHT = 0.15, 0.75   # default landmarks; see --left / --right


def windows_for_video(rows: list[dict], pad_s: float) -> list[tuple[int, int]]:
    """Merged [first_frame, last_frame] spans to decode for one video."""
    spans = sorted((int((float(r["start_s"]) - pad_s) * FPS),
                    int((float(r["end_s"]) + pad_s) * FPS)) for r in rows)
    merged: list[list[int]] = []
    for lo, hi in spans:
        lo = max(0, lo)
        if merged and lo <= merged[-1][1] + GOP_FRAMES:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return [(lo, hi) for lo, hi in merged]


def decode_windows(video: Path, spans: list[tuple[int, int]], roi) -> dict[int, np.ndarray]:
    """Grey ROI crops for every frame in the spans, decoded in one forward pass."""
    x1, y1, x2, y2 = roi
    cap = cv2.VideoCapture(str(video))
    frames: dict[int, np.ndarray] = {}
    position = 0
    for lo, hi in spans:
        if lo < position or lo - position > GOP_FRAMES:
            cap.set(cv2.CAP_PROP_POS_FRAMES, lo)
            position = lo
        while position < lo:
            if not cap.grab():
                cap.release()
                return frames
            position += 1
        while position <= hi:
            ok, frame = cap.read()
            if not ok:
                cap.release()
                return frames
            frames[position] = cv2.cvtColor(
                cv2.resize(frame[y1:y2, x1:x2], (DW, DH)), cv2.COLOR_BGR2GRAY)
            position += 1
    cap.release()
    return frames


def build_background(frames: dict[int, np.ndarray]) -> np.ndarray:
    """The empty tunnel: per-pixel median over a subsample of the decoded frames."""
    keys = sorted(frames)[::BG_SAMPLE]
    stack = np.stack([frames[k] for k in keys])
    return np.median(stack, axis=0).astype(np.uint8)


def blobs_in(frame: np.ndarray, background: np.ndarray) -> list[tuple[float, int]]:
    """Animal-sized blobs as (centroid_x_fraction, area), largest first."""
    diff = cv2.absdiff(frame, background)
    mask = (diff > FG_THRESH).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    count, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)
    found = []
    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= MIN_BLOB_PX:
            found.append((float(centroids[i][0]) / DW, area))
    found.sort(key=lambda b: -b[1])
    return found


def track_event(frames: dict[int, np.ndarray], background: np.ndarray,
                first: int, last: int) -> list[dict]:
    """Per-frame track over one event's padded window."""
    track = []
    for frame_no in range(first, last + 1):
        image = frames.get(frame_no)
        if image is None:
            continue
        found = blobs_in(image, background)
        track.append({
            "frame": frame_no,
            "t_s": round(frame_no / FPS, 4),
            "n_animals": len(found),
            "x": round(found[0][0], 4) if found else "",
            "area": found[0][1] if found else 0,
        })
    return track


def all_crossings(track: list[dict], level: float) -> list[float]:
    """Every time the single-animal track crosses `level`, sub-frame interpolated.

    Consecutive track points are only joined when they are adjacent frames --
    otherwise a gap where the animal was absent or doubled would be read as a
    crossing that never happened.
    """
    points = [(row["frame"], row["t_s"], float(row["x"])) for row in track
              if row["x"] != "" and row["n_animals"] == 1]
    times = []
    for (f0, t0, x0), (f1, t1, x1) in zip(points, points[1:]):
        if f1 != f0 + 1:
            continue
        if (x0 < level <= x1) or (x0 > level >= x1):
            frac = 0.0 if x1 == x0 else (level - x0) / (x1 - x0)
            times.append(round(t0 + frac * (t1 - t0), 4))
    return times


def crossing_near(track: list[dict], level: float, anchor: float) -> tuple[float | None, int]:
    """The crossing of `level` closest to `anchor`, and how many there were.

    Nearest-to-the-event rather than first-in-the-window: an animal that pokes
    past a landmark, retreats, and then makes the real traverse would otherwise
    be timed on the poke.
    """
    times = all_crossings(track, level)
    if not times:
        return None, 0
    return min(times, key=lambda t: abs(t - anchor)), len(times)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from-csv", required=True)
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--datadir")
    parser.add_argument("--roi", default=",".join(map(str, DEF_ROI)))
    parser.add_argument("--left", type=float, default=LEFT)
    parser.add_argument("--right", type=float, default=RIGHT)
    parser.add_argument("--pad", type=float, default=PAD_S)
    parser.add_argument("--debug-frames", type=int, default=0,
                        help="write this many segmentation check images and exit early")
    args = parser.parse_args()

    roi = tuple(int(v) for v in args.roi.split(","))
    datadir = (Path(args.datadir) if args.datadir
               else BASE_RAW / f"experiment_{args.exp}" / "concatenated_data_cam_mic_sync")
    out_dir = Path(args.out_dir)
    (out_dir / "tracks").mkdir(parents=True, exist_ok=True)

    rows = list(csv.DictReader(open(args.from_csv)))
    by_video = defaultdict(list)
    for row in rows:
        by_video[row["video"]].append(row)

    summary = []
    debug_written = 0
    for video_name in sorted(by_video):
        video = datadir / video_name
        if not video.exists():
            print(f"missing {video}")
            continue
        events = by_video[video_name]
        frames = decode_windows(video, windows_for_video(events, args.pad), roi)
        if not frames:
            print(f"{video_name}: decoded nothing")
            continue
        background = build_background(frames)

        if args.debug_frames and debug_written < args.debug_frames:
            for frame_no in sorted(frames)[::max(1, len(frames) // 6)]:
                if debug_written >= args.debug_frames:
                    break
                image = frames[frame_no]
                found = blobs_in(image, background)
                panel = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
                for x_frac, area in found:
                    x = int(x_frac * DW)
                    cv2.line(panel, (x, 0), (x, DH), (0, 255, 255), 1)
                    cv2.putText(panel, f"{area}", (x + 3, 14),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)
                for level, colour in ((args.left, (80, 255, 80)), (args.right, (80, 255, 80))):
                    cv2.line(panel, (int(level * DW), 0), (int(level * DW), DH), colour, 1)
                cv2.putText(panel, f"{video_name[:-4]} f{frame_no} n={len(found)}",
                            (4, DH - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
                stacked = cv2.vconcat([cv2.cvtColor(background, cv2.COLOR_GRAY2BGR), panel])
                cv2.imwrite(str(out_dir / f"debug_{debug_written:02d}.png"),
                            cv2.resize(stacked, (DW * 2, DH * 4)))
                debug_written += 1

        for row in events:
            first = int((float(row["start_s"]) - args.pad) * FPS)
            last = int((float(row["end_s"]) + args.pad) * FPS)
            track = track_event(frames, background, max(0, first), last)
            if not track:
                continue
            stem = f"{video_name[:-4]}_t{row['start_s'].replace('.', '_')}"
            with open(out_dir / "tracks" / f"{stem}.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(track[0]))
                writer.writeheader()
                writer.writerows(track)

            # how often is more than one animal in the tunnel during the event itself
            inside = [r for r in track
                      if float(row["start_s"]) <= r["t_s"] <= float(row["end_s"])]
            multi = sum(1 for r in inside if r["n_animals"] > 1) / max(1, len(inside))
            empty = sum(1 for r in inside if r["n_animals"] == 0) / max(1, len(inside))

            anchor = (float(row["start_s"]) + float(row["end_s"])) / 2
            t_left, n_left = crossing_near(track, args.left, anchor)
            t_right, n_right = crossing_near(track, args.right, anchor)
            traversed = t_left is not None and t_right is not None
            if traversed:
                landmark_dir = "to_arena" if t_left < t_right else "to_nest"
                entry, exit_ = sorted((t_left, t_right))
            else:
                landmark_dir, entry, exit_ = "", None, None

            summary.append({
                "video": video_name, "start_s": row["start_s"], "end_s": row["end_s"],
                "detector_direction": row["direction"],
                "multi_animal_frac": round(multi, 3),
                "empty_frac": round(empty, 3),
                "single_animal": multi < MULTI_FRAC,
                "t_left": t_left if t_left is not None else "",
                "t_right": t_right if t_right is not None else "",
                "n_left_crossings": n_left, "n_right_crossings": n_right,
                "traversed": traversed,
                "landmark_direction": landmark_dir,
                "traverse_s": round(exit_ - entry, 4) if traversed else "",
                "track_file": f"tracks/{stem}.csv",
            })
        print(f"{video_name}: {len(events)} events, {len(frames)} frames decoded", flush=True)
        if args.debug_frames and debug_written >= args.debug_frames:
            print(f"wrote {debug_written} debug images -> {out_dir}; stopping early")
            break

    if not summary:
        raise SystemExit("no events processed")
    with open(out_dir / "landmarks.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    clean = [s for s in summary if s["single_animal"] and s["traversed"]]
    agree = sum(1 for s in clean if s["landmark_direction"] == s["detector_direction"])
    print(f"\n{len(summary)} events -> {out_dir}/landmarks.csv")
    print(f"  single animal in tunnel : {sum(1 for s in summary if s['single_animal'])}")
    print(f"  crossed both landmarks  : {sum(1 for s in summary if s['traversed'])}")
    print(f"  clean single traverses  : {len(clean)}")
    print(f"  landmark direction agrees with the detector: {agree}/{len(clean)}")


if __name__ == "__main__":
    main()
