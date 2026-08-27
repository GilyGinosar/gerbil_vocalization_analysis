#!/usr/bin/env python
"""Nest_top frames grouped by how many animals the census says are already home.

The census infers nest occupancy as `colony size - animals detected in the arenas
- the one in the tunnel`, because the nest itself cannot be counted from video:
the gerbils burrow under the bedding. This draws the frames behind that inference
so it can be checked by eye -- one row per inferred count, a few examples each,
taken at the moment the traverse starts (the traveller is at the far end of the
tunnel then, so the nest holds only the residents).

Rows for a NEGATIVE count are the diagnostic ones: they are impossible, and they
mean the arena detector over-counted, usually by splitting a huddle into several
boxes. If those rows show a busy nest, the census is under-counting the nest.

    python scripts/analysis/nest_occupancy_examples.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/burrow/nest_census
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

from scripts.pipeline.audio_processing_config import get_colony_size  # noqa: E402
from scripts.pipeline.paths import BASE_RAW, video_detections_dir  # noqa: E402

FPS = 30
TILE = (330, 440)          # each nest_top frame, after rotating to portrait


def nest_tile(exp: int, file_num: int, t: float, caption: str) -> np.ndarray | None:
    path = (BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync"
            / f"video_nest_top_{file_num:03d}.mp4")
    if not path.exists():
        return None
    cap = cv2.VideoCapture(str(path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(int(t * FPS), 0))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        return None
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = cv2.resize(frame, TILE)
    cv2.putText(frame, caption, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                (120, 235, 255), 1, cv2.LINE_AA)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--per-row", type=int, default=5)
    parser.add_argument("--experiments", type=int, default=14)
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    colony = get_colony_size(args.date)
    if colony is None:
        raise SystemExit(f"no n_animals recorded for {args.date} in experiments.toml")

    tv = pd.read_parquet(scan / f"traverses_{args.date}.parquet")
    tv = tv[tv.single_animal & (tv.direction == "to_nest")]
    rows = []
    for exp in sorted(tv.exp.unique())[:args.experiments]:
        try:
            det = pd.read_parquet(video_detections_dir(args.date, exp) / "detections.parquet")
        except FileNotFoundError:
            continue
        per = det.groupby(["file_num", "frame_id"]).size()
        for r in tv[tv.exp == exp].itertuples():
            lo, hi = int((r.t_entry - 2) * FPS), int(r.t_entry * FPS)
            vals = per.reindex([(r.file_num, f) for f in range(max(lo, 0), hi)])
            vals = vals.fillna(0).to_numpy()
            if not len(vals):
                continue
            arena = int(np.median(vals))
            rows.append({"exp": exp, "file_num": int(r.file_num), "t": float(r.t_entry),
                         "arena": arena, "nest": colony - arena - 1})
    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "census.csv", index=False)
    print(f"{len(table):,} to_nest traverses, colony N={colony}")
    print(table.nest.value_counts().sort_index().to_string())

    strips, labels = [], []
    for nest in sorted(table.nest.unique()):
        group = table[table.nest == nest].sample(frac=1.0, random_state=0)
        tiles = []
        for r in group.itertuples():
            tile = nest_tile(r.exp, r.file_num, r.t,
                             f"exp{r.exp} f{r.file_num} arena={r.arena}")
            if tile is not None:
                tiles.append(tile)
            if len(tiles) >= args.per_row:
                break
        if not tiles:
            continue
        while len(tiles) < args.per_row:
            tiles.append(np.zeros((TILE[1], TILE[0], 3), np.uint8))
        head = np.zeros((TILE[1], 190, 3), np.uint8)
        flag = "  IMPOSSIBLE" if nest < 0 else ""
        cv2.putText(head, f"nest = {nest}{flag}", (8, TILE[1] // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (80, 80, 240) if nest < 0 else (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(head, f"n={len(table[table.nest == nest]):,}", (8, TILE[1] // 2 + 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1, cv2.LINE_AA)
        strips.append(cv2.hconcat([head] + tiles))
        labels.append(nest)
    sheet = cv2.vconcat(strips)
    cv2.imwrite(str(out_dir / "nest_by_census.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"\nrows drawn for nest counts: {labels}")
    print(f"wrote {out_dir}/nest_by_census.jpg")


if __name__ == "__main__":
    main()
