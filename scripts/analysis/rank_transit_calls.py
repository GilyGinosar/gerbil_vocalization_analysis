#!/usr/bin/env python
"""Rank curated crossings by how concentrated the calling is -- a reading order for the picker.

Not statistics: plain counts, so you know which of the 52 cards to open first instead of
scrolling blind. For each crossing it counts DAS underground calls that fall INSIDE the
crossing versus in the surrounding context window, and turns both into calls/s so a 30 s
crossing is not flattered over a 2 s one.

    python scripts/analysis/rank_transit_calls.py \
        --transits .../transits_492_curated.csv --exp 492 --out exports/burrow_look_492/reading_order.csv
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video import burrow_transit_picker as picker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--transits", required=True)
    parser.add_argument("--exp", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--context", type=float, default=picker.CONTEXT_S,
                        help="seconds each side of the crossing, matching the picker's --context")
    args = parser.parse_args()

    calls = picker.load_calls(args.exp)
    rows = list(csv.DictReader(open(args.transits)))

    ranked = []
    for row in rows:
        index = picker.file_index(row["video"])
        start_s, end_s = float(row["start_s"]), float(row["end_s"])
        middle = (start_s + end_s) / 2
        t0, t1 = middle - args.context, middle + args.context

        inside = around = 0
        for call_start, _stop, _event_type in calls.get(index, []):
            if start_s <= call_start <= end_s:
                inside += 1
            elif t0 <= call_start <= t1:
                around += 1

        duration = end_s - start_s
        around_duration = (t1 - t0) - duration
        rate_in = inside / duration if duration > 0 else 0.0
        rate_around = around / around_duration if around_duration > 0 else 0.0
        ranked.append({
            "video": row["video"], "start_s": row["start_s"], "direction": row["direction"],
            "dur_s": round(duration, 2),
            "calls_during": inside, "calls_around": around,
            "rate_during": round(rate_in, 3), "rate_around": round(rate_around, 3),
            # how much denser the calling is inside the crossing than beside it
            "concentration": round(rate_in / rate_around, 2) if rate_around > 0
                             else ("inf" if inside else 0),
        })

    # strongest first: the crossings most worth eyeballing
    ranked.sort(key=lambda r: -r["rate_during"])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ranked[0]))
        writer.writeheader()
        writer.writerows(ranked)

    silent = sum(1 for r in ranked if r["calls_during"] == 0)
    print(f"{len(ranked)} crossings -> {out}   ({silent} with no calls during the crossing)")
    for direction in ("to_arena", "to_nest", "reversal"):
        group = [r for r in ranked if r["direction"] == direction]
        if not group:
            continue
        during = sum(r["calls_during"] for r in group)
        during_s = sum(r["dur_s"] for r in group)
        around = sum(r["calls_around"] for r in group)
        around_s = sum(2 * args.context - r["dur_s"] for r in group)
        print(f"  {direction:<9} n={len(group):<3} during {during/during_s:.2f}/s   "
              f"around {around/around_s:.2f}/s")


if __name__ == "__main__":
    main()
