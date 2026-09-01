#!/usr/bin/env python
"""Look at the nest during the window the motion score was computed over.

`nest_motion.py` reduces the 5 s before a `to_nest` arrival to one number, and a
number that small hides the thing that matters: `motion_pre = 0` can mean the
nest was EMPTY, or it can mean every resident was buried under the bedding and
invisible. Those support opposite conclusions about who is calling, and no
statistic separates them. Only looking does.

So this draws the window rather than the instant. One row per traverse, frames
spread across [t_entry - pre, t_entry], each with the motion value measured at
that moment, and -- the point of the layout -- a difference map under every
frame. A still nest and an empty nest look identical in the frames; in the diff
maps a buried animal still breathes and shifts, and an empty nest does not. The
diff is the same absdiff on the same downscaled grey frames that produced the
score, at the same CHANGED threshold, so what you see is what was counted.

    python scripts/video/nest_motion_cards.py \
        --set still=exports/burrow/nest_motion/select_still.csv \
        --set active=exports/burrow/nest_motion/select_active.csv \
        --out-dir exports/burrow/nest_motion_cards
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.nest_motion import CHANGED, FPS, SMALL  # noqa: E402
from scripts.pipeline.paths import BASE_RAW  # noqa: E402

PANEL_H = 200           # px per frame panel
HEADER_W = 300          # px of caption column at the left of each row
GAP = 4
FONT = cv2.FONT_HERSHEY_SIMPLEX


def read_pair(cap, t: float):
    """The frame at time t and the one before it, read as a true consecutive pair.

    Seeking to each of two adjacent frames does NOT work: CAP_PROP_POS_FRAMES on
    H.264 snaps to the nearest keyframe, so both seeks land on the same frame and
    every difference comes out exactly zero -- which reads as a perfectly still
    nest and is entirely an artifact. So seek once and read forward, which is how
    nest_motion.py measured it in the first place.
    """
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t * FPS) - 1, 0))
    ok_a, first = cap.read()
    ok_b, second = cap.read()
    if not (ok_a and ok_b) or first is None or second is None:
        return None, None, None
    grey = lambda f: cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), SMALL)
    return second, grey(first), grey(second)


def panel(frame: np.ndarray, height: int) -> np.ndarray:
    """The nest view, rotated upright and scaled to a fixed height."""
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    scale = height / frame.shape[0]
    return cv2.resize(frame, (max(int(frame.shape[1] * scale), 1), height))


def diff_panel(prev_grey, grey, height: int, width: int) -> np.ndarray:
    """The changed-pixel mask that the motion score actually counts.

    Drawn at the same SMALL resolution and CHANGED threshold as the score, then
    blown up -- deliberately not smoothed, so a lone speckle stays visible as a
    speckle rather than becoming a smear that looks like an animal.
    """
    if prev_grey is None:
        mask = np.zeros(SMALL[::-1], np.uint8)
    else:
        mask = (cv2.absdiff(grey, prev_grey) > CHANGED).astype(np.uint8) * 255
    mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
    mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    out = np.zeros((height, width, 3), np.uint8)
    out[:, :, 2] = mask                                   # red = changed
    return out


def build_row(row, n_frames: int, pre: float) -> np.ndarray | None:
    datadir = BASE_RAW / f"experiment_{int(row.exp)}" / "concatenated_data_cam_mic_sync"
    path = datadir / f"video_nest_top_{int(row.file_num):03d}.mp4"
    if not path.exists():
        print(f"  exp {row.exp} file {row.file_num}: no nest_top video")
        return None
    cap = cv2.VideoCapture(str(path))
    times = np.linspace(row.t_entry - pre, row.t_entry, n_frames)
    cells = []
    for t in times:
        frame, before, grey = read_pair(cap, t)
        if frame is None:
            continue
        img = panel(frame, PANEL_H)
        d = diff_panel(before, grey, PANEL_H // 2, img.shape[1])
        changed = float((cv2.absdiff(grey, before) > CHANGED).mean())
        cv2.putText(img, f"t{t - row.t_entry:+.1f}s", (5, 16), FONT, 0.45,
                    (120, 235, 255), 1, cv2.LINE_AA)
        cv2.putText(d, f"{changed:.4f}", (5, 14), FONT, 0.4,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cells.append(cv2.vconcat([img, d]))
    cap.release()
    if not cells:
        return None
    height = cells[0].shape[0]
    spacer = np.zeros((height, GAP, 3), np.uint8)
    strip = cv2.hconcat([c for cell in cells for c in (cell, spacer)][:-1])

    head = np.zeros((height, HEADER_W, 3), np.uint8)
    calls = int(getattr(row, "arrival_calls", -1))
    lines = [f"exp {int(row.exp)}  file {int(row.file_num)}",
             f"t_entry {row.t_entry:.2f}s",
             f"motion_pre {row.motion_pre:.4f}",
             f"{calls} calls at arrival" if calls >= 0 else ""]
    for i, line in enumerate(lines):
        cv2.putText(head, line, (8, 26 + 24 * i), FONT, 0.5,
                    (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(head, f"top: nest_top   bottom: pixels changing by >{CHANGED} (red)",
                (8, height - 10), FONT, 0.38, (150, 150, 150), 1, cv2.LINE_AA)
    return cv2.hconcat([head, head[:, :GAP] * 0, strip])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--set", action="append", required=True, metavar="LABEL=CSV",
                        help="a named selection CSV (exp,file_num,t_entry,motion_pre); "
                             "repeat to put several blocks in one sheet")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--frames", type=int, default=6,
                        help="frames sampled across the pre-entry window")
    parser.add_argument("--pre", type=float, default=5.0,
                        help="seconds before entry; must match the nest_motion run")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in args.set:
        label, _, csv_path = spec.partition("=")
        table = pd.read_csv(csv_path)
        print(f"{label}: {len(table)} traverses")
        rows = []
        for r in table.itertuples():
            built = build_row(r, args.frames, args.pre)
            if built is not None:
                rows.append(built)
                print(f"  exp {int(r.exp)} file {int(r.file_num)} "
                      f"motion_pre={r.motion_pre:.4f}")
        if not rows:
            print(f"  nothing drawn for {label}")
            continue
        width = max(r.shape[1] for r in rows)
        padded = [cv2.copyMakeBorder(r, 0, GAP, 0, width - r.shape[1],
                                     cv2.BORDER_CONSTANT, value=(0, 0, 0)) for r in rows]
        banner = np.zeros((40, width, 3), np.uint8)
        cv2.putText(banner, f"{label.upper()}  --  {args.pre:g} s before arrival at the nest",
                    (10, 27), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        sheet = cv2.vconcat([banner] + padded)
        out = out_dir / f"nest_{label}.jpg"
        cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 88])
        print(f"wrote {out}  ({sheet.shape[1]}x{sheet.shape[0]})")


if __name__ == "__main__":
    main()
