#!/usr/bin/env python
"""One nest_motion task per nest_top video, for disBatch.

Sharding on (exp, file) rather than on experiment: experiments are very uneven
(one has 17 traverses in a single file, others a handful across many), and a
per-experiment split would leave one worker running long after the rest finished.
Per-file shards are 1,641 tasks of a median 2 traverses, none longer than ~2 min,
which disBatch packs tightly.

Only traverses that survive every filter get a task -- single_animal, to_nest, and
an OBSERVED t_out. A capped t_out is t_exit + MAX_LINGER_S, so its arrival window
points at the wrong time and the row would have to be thrown away afterwards.

    python slurm/make_nest_motion_tasks.py --date 2026_02 \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --out-dir /mnt/ceph/users/gginosar/nest_motion_2026_02 \
        > slurm/nest_motion_2026_02.tasks
"""
import argparse
from pathlib import Path

import pandas as pd

ROOT = "/mnt/home/gginosar/repos/gerbil_vocalization_analysis"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--date", default="2026_02")
    ap.add_argument("--scan", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    tv = pd.read_parquet(Path(args.scan) / f"traverses_{args.date}.parquet")
    tv = tv[tv.single_animal & (tv.direction == "to_nest")
            & (~tv.still_in_tunnel_at_cap)]
    for (exp, file_num), group in tv.groupby(["exp", "file_num"]):
        print(f"cd {ROOT} && {ROOT}/.venv/bin/python scripts/analysis/nest_motion.py "
              f"--scan {args.scan} --out-dir {args.out_dir} --date {args.date} "
              f"--limit 0 --exp {int(exp)} --file-num {int(file_num)}")


if __name__ == "__main__":
    main()
