#!/usr/bin/env python
"""Write a disBatch task file: one burrow_scan command per burrow_side video.

Per-video rather than per-experiment because experiments range from 2 to 229
videos -- an array indexed by experiment would leave most workers idle while one
runs for eight hours. Every task is ~2 minutes, so disBatch keeps the pool busy
and a failure costs one video rather than an experiment.

    python scripts/video/make_scan_tasks.py --date 2026_02 \
        --out-dir /mnt/ceph/users/gginosar/burrow_scan_2026_02 --tasks slurm/burrow_scan_2026_02.tasks
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.paths import AUDIO_ROOT, BASE_RAW  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--date", required=True, help="date folder, e.g. 2026_02")
    parser.add_argument("--out-dir", required=True, help="where the scan writes (use ceph: it is bulk data)")
    parser.add_argument("--tasks", required=True, help="task file to write")
    parser.add_argument("--python", default=str(REPO_ROOT / ".venv/bin/python"))
    parser.add_argument("--sample", type=int,
                        help="take only this many videos, spread evenly across experiments "
                             "and across time within each -- a representative subset rather "
                             "than the first N, which would all be one experiment's first hour")
    args = parser.parse_args()

    date_dir = AUDIO_ROOT / args.date
    lines, videos, experiments = [], 0, 0
    for exp_dir in sorted(p for p in date_dir.iterdir() if p.is_dir() and p.name.isdigit()):
        exp = int(exp_dir.name)
        raw = BASE_RAW / f"experiment_{exp}" / "concatenated_data_cam_mic_sync"
        if not raw.is_dir() or not (exp_dir / "calls.csv").exists():
            continue
        found = sorted(raw.glob("video_burrow_side_*.mp4"))
        if not found:
            continue
        experiments += 1
        for video in found:
            videos += 1
            lines.append(
                f"cd {REPO_ROOT} && {args.python} scripts/video/burrow_scan.py "
                f"--exp {exp} --video {video} --out-dir {args.out_dir}")

    if args.sample and args.sample < len(lines):
        step = len(lines) / args.sample
        lines = [lines[int(i * step)] for i in range(args.sample)]
        videos = len(lines)

    tasks = Path(args.tasks)
    tasks.parent.mkdir(parents=True, exist_ok=True)
    tasks.write_text("\n".join(lines) + "\n")
    print(f"{experiments} experiments, {videos} videos -> {tasks}")
    print(f"~{videos * 2 / 60:.0f} core-hours; on 64 workers about {videos * 2 / 60 / 64:.1f} h wall clock")


if __name__ == "__main__":
    main()
