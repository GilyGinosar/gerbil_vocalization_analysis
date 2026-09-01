#!/usr/bin/env python
"""Concatenate the per-shard nest_motion CSVs into one table.

    python scripts/analysis/pool_nest_motion.py \
        --in-dir /mnt/ceph/users/gginosar/nest_motion_2026_02 \
        --out exports/burrow/nest_motion/nest_motion_full.csv
"""
import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    shards = sorted(Path(args.in_dir).glob("nest_motion_*.csv"))
    if not shards:
        raise SystemExit(f"no nest_motion_*.csv under {args.in_dir}")
    tables = [pd.read_csv(p) for p in shards]
    tables = [t for t in tables if len(t)]
    d = pd.concat(tables, ignore_index=True).sort_values(["exp", "file_num", "t_entry"])
    before = len(d)
    d = d.drop_duplicates(subset=["exp", "file_num", "t_entry"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"{len(shards)} shards -> {len(d)} traverses "
          f"({before - len(d)} duplicates dropped) -> {out}")
    print(f"  motion_pre: median {d.motion_pre.median():.4f}  "
          f"zero on {int((d.motion_pre == 0).sum())} traverses")


if __name__ == "__main__":
    main()
