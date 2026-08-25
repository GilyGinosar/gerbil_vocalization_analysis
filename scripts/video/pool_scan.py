#!/usr/bin/env python
"""Combine a date folder's per-video traverse CSVs into one table.

The disBatch run writes `traverses_video_burrow_side_NNN.csv` per video, per
experiment, because a task that owns one video can only safely write its own
file. This pools them and adds the experiment id and wall-clock time, so the
result joins straight onto `calls.csv` and the rest of the analysis.

    python scripts/video/pool_scan.py --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --date 2026_02
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.paths import experiment_sync_path  # noqa: E402


def file_start_times(exp: int) -> dict[int, pd.Timestamp]:
    """Wall-clock start of each file index, from the experiment's sync.csv."""
    path = experiment_sync_path(exp)
    if not path.exists():
        return {}
    sync = pd.read_csv(path)
    out = {}
    for i, stamp in enumerate(sync["timestamp"].apply(ast.literal_eval)):
        out[i] = pd.to_datetime(stamp[0])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--date", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    scan = Path(args.scan)
    frames = []
    for exp_dir in sorted(p for p in scan.iterdir() if p.is_dir() and p.name.isdigit()):
        exp = int(exp_dir.name)
        parts = [pd.read_csv(p) for p in sorted(exp_dir.glob("traverses*.csv"))]
        parts = [p for p in parts if len(p)]
        if not parts:
            continue
        table = pd.concat(parts, ignore_index=True)
        table["exp"] = exp
        starts = file_start_times(exp)
        table["file_num"] = table.video.str.extract(r"_(\d+)\.mp4$").astype(int)
        table["start_time_real"] = [
            starts.get(n, pd.NaT) + pd.to_timedelta(t, unit="s") if n in starts else pd.NaT
            for n, t in zip(table.file_num, table.t_entry)]
        frames.append(table)
        print(f"  exp {exp}: {len(table)} traverses from {len(parts)} videos")

    if not frames:
        raise SystemExit(f"no traverse CSVs under {scan}")
    pooled = pd.concat(frames, ignore_index=True).sort_values(["exp", "file_num", "t_entry"])
    out = Path(args.out) if args.out else scan / f"traverses_{args.date}.parquet"
    pooled.to_parquet(out, index=False)
    clean = pooled[pooled.single_animal]
    print(f"\n{len(pooled)} traverses from {pooled.exp.nunique()} experiments -> {out}")
    print(f"  single-animal: {len(clean)}")
    print(f"  by direction : {dict(clean.direction.value_counts())}")
    print(f"  median traverse {clean.traverse_s.median():.2f}s, "
          f"median time in tunnel {(clean.t_out - clean.t_entry).median():.2f}s")


if __name__ == "__main__":
    main()
