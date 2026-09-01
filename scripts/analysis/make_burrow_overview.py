#!/usr/bin/env python
"""The current burrow overview, in one command.

`burrow_overview.py` grew a dozen flags and the version worth showing is one
combination of seven of them. Each is there because the default was wrong in a
way that changed what the figure said, so a half-remembered invocation is not a
slightly worse figure -- it is a misleading one:

  --category-csv   split EMPTY / SLEEPING / ACTIVE, scored by eye. Splitting on
                   nest MOTION cannot see the distinction that matters: empty
                   and sleeping nests have the same motion and a fourfold
                   difference in arrival burst.
  --no-localiser   drop the tunnel-origin/nest-origin series. That label is a
                   POSITION gradient -- an animal in the nest-end half of the
                   tunnel scores like the nest -- so drawing it invites a source
                   claim the data cannot support.
  --clear 0        keep back-to-back transits. `single_animal` already excludes
                   co-occupancy DURING a traverse; --clear additionally required
                   3 s of empty tunnel either side, which threw away a third of
                   the data and preferentially kept quiet periods -- the exact
                   base-rate axis that has misled this analysis twice.
  --pad 10/10      the burst is not a transient. At the old 2 s cut-off the
                   figure stopped at the peak; it still holds a third of its
                   excess 10 s after arrival.
  --column         the full 16-panel grid is 2400x5300, so on a screen it is
                   downscaled ~3x and the raster ticks vanish.
  --free-y         each row's rate panel on its own scale. Rows differ ~5x, and
                   a shared limit flattens the small ones into the axis. The
                   cost is that bar heights are no longer comparable BETWEEN
                   rows -- say so when showing it.

Never-usable rows (truncated last chunk, capped t_out) are dropped by the
loaders; see scripts/utils/data_rules.py.

    python scripts/analysis/make_burrow_overview.py --out-dir exports/overview

    # the light-cycle version, when someone asks about it
    python scripts/analysis/make_burrow_overview.py --out-dir exports/overview_ld \
        --split-light
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.publish import publish, today  # noqa: E402

CATEGORY_TABLE = REPO_ROOT / "data" / "nest_scoring" / "nest_category_full.csv"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scan", default="/mnt/ceph/users/gginosar/burrow_scan_2026_02")
    ap.add_argument("--date", default="2026_02")
    ap.add_argument("--category-csv", default=str(CATEGORY_TABLE))
    ap.add_argument("--column", default="usv:entry,usv:exit",
                    help="which panels to draw; 'all:exit' adds the non-USV classes")
    ap.add_argument("--pad", type=float, default=10.0,
                    help="seconds of epoch either side of the landmarks")
    ap.add_argument("--split-light", action="store_true",
                    help="split every row LIGHT/DARK. The phase effect is ~4x "
                         "smaller than the category effect and halves each n, so "
                         "this is for answering a question about the light cycle, "
                         "not for the headline figure.")
    ap.add_argument("--reuse-cache-from",
                    help="an out-dir whose collected.npz matches --clear/--calls; "
                         "copying it skips re-walking every traverse (~4 min)")
    ap.add_argument("--shared-y", action="store_true",
                    help="one y-limit across all rows, so heights compare by eye")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.reuse_cache_from:
        src = Path(args.reuse_cache_from)
        for slug in ("all", "usv"):
            cache = src / slug / "collected.npz"
            if cache.exists():
                (out_dir / slug).mkdir(parents=True, exist_ok=True)
                shutil.copy(cache, out_dir / slug / "collected.npz")
                print(f"reused {cache}")

    cmd = [sys.executable, "-u", str(REPO_ROOT / "scripts/analysis/burrow_overview.py"),
           "--scan", args.scan, "--date", args.date,
           "--out-dir", str(out_dir),
           "--category-csv", args.category_csv,
           "--no-localiser",
           "--clear", "0",
           "--column", args.column,
           "--pad-before", str(args.pad), "--pad-after", str(args.pad),
           "--max-lag", str(args.pad + 4)]
    if not args.shared_y:
        cmd.append("--free-y")
    if args.split_light:
        cmd.append("--split-light")
    print(" ".join(cmd), flush=True)
    rc = subprocess.call(cmd)
    if rc == 0:
        # name it for what it IS, not for the directory it happened to land in:
        # exports/ held a dozen */burrow_overview.png that only timestamps told apart
        sets = {c.split(":")[0] for c in args.column.split(",")}
        which = sets.pop() if len(sets) == 1 else "both"
        tag = f"burrow_overview_{which}"
        if args.split_light:
            tag += "_lightdark"
        publish(out_dir / "burrow_overview.png",
                name=f"{tag}_{args.date}_{today()}.png", date=args.date)
    raise SystemExit(rc)


if __name__ == "__main__":
    main()
