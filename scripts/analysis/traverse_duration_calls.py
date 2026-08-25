#!/usr/bin/env python
"""How long are traverses, and which ones have calls? Split by direction.

Durations are heavy-tailed (0.7 s to tens of seconds), so the histogram is over
log10(duration) with linear bins rather than log-spaced bins on a log axis --
the tail stays visible instead of being compressed into the last bar.

The bottom row is the part that answers "should we look only at short
traverses": the fraction with a tunnel-localised call, and the call RATE, as a
function of duration. Fraction alone is misleading, because a longer traverse
has a longer card window and therefore more opportunity for any call to land in
it -- so a rising fraction can be pure exposure. The rate divides that out.

    python scripts/analysis/traverse_duration_calls.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/duration_2026_02
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DIRECTION_COLOURS = {"to_arena": "#2f6fd0", "to_nest": "#d1642a"}
QUIET = "#c9c9c4"
BEFORE_S, AFTER_S = 3.0, 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    scan = Path(args.scan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tv = pd.read_parquet(scan / f"traverses_{args.date}.parquet")
    tv = tv[tv.single_animal]

    localised: dict[tuple[int, int], np.ndarray] = {}
    for path in sorted((scan / "origin").glob("*/call_origin.csv")):
        exp = int(path.parent.name)
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        reference = table.loc[table.state == "tunnel empty", "tunnel_db_over_nest"]
        if len(reference) < 50:
            continue
        hit = table[table.tunnel_db_over_nest > reference.quantile(0.95)]
        for file_num, group in hit.groupby("file"):
            localised[(exp, int(file_num))] = np.sort(group.start_s.to_numpy())

    tv = tv[[(e, f) in localised for e, f in zip(tv.exp, tv.file_num)]].copy()
    counts = []
    for row in tv.itertuples():
        times = localised[(row.exp, row.file_num)]
        t0, t1 = row.t_entry - BEFORE_S, row.t_out + AFTER_S
        counts.append(int(((times >= t0) & (times <= t1)).sum()))
    tv["n_calls"] = counts
    tv["window_s"] = (tv.t_out - tv.t_entry) + BEFORE_S + AFTER_S
    tv["has_calls"] = tv.n_calls > 0
    tv["log_dur"] = np.log10(tv.traverse_s.clip(lower=0.1))
    print(f"{len(tv)} traverses, {int(tv.has_calls.sum())} with a tunnel-localised call")

    edges = np.linspace(np.log10(0.5), np.log10(60), 31)
    centres = (edges[:-1] + edges[1:]) / 2
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)

    for col, direction in enumerate(("to_arena", "to_nest")):
        sub = tv[tv.direction == direction]
        colour = DIRECTION_COLOURS[direction]

        ax = axes[0][col]
        ax.hist([sub.loc[~sub.has_calls, "log_dur"], sub.loc[sub.has_calls, "log_dur"]],
                bins=edges, stacked=True, color=[QUIET, colour],
                label=[f"no tunnel call ({int((~sub.has_calls).sum()):,})",
                       f"has tunnel call ({int(sub.has_calls.sum()):,})"])
        ax.axvline(np.log10(sub.traverse_s.median()), color="0.25", ls="--", lw=1)
        ax.set_title(f"{direction}  (n={len(sub):,}, median {sub.traverse_s.median():.2f}s)",
                     loc="left", fontsize=11)
        ax.legend(frameon=False, fontsize=8, loc="upper right")
        if col == 0:
            ax.set_ylabel("traverses")

        ax = axes[1][col]
        idx = np.digitize(sub.log_dur, edges) - 1
        frac, rate, keep = [], [], []
        for b in range(len(centres)):
            m = idx == b
            if m.sum() >= 20:
                keep.append(centres[b])
                frac.append(sub.loc[m, "has_calls"].mean())
                rate.append((sub.loc[m, "n_calls"] / sub.loc[m, "window_s"]).mean())
        ax.plot(keep, frac, color=colour, lw=2, marker="o", ms=3, label="fraction with a call")
        ax2 = ax.twinx() if False else None      # never two y-scales on one axis
        ax.plot(keep, rate, color="0.35", lw=2, ls="--", marker="s", ms=3,
                label="tunnel calls / s of window")
        ax.set_ylim(0, None)
        ax.set_xlabel("traverse duration (s)")
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        if col == 0:
            ax.set_ylabel("fraction  /  calls per second")
        for a in (axes[0][col], axes[1][col]):
            a.grid(axis="y", color="0.93", lw=0.8)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
        ticks = [0.5, 1, 2, 5, 10, 20, 60]
        axes[1][col].set_xticks(np.log10(ticks))
        axes[1][col].set_xticklabels([str(t) for t in ticks])

    fig.suptitle("2026_02: traverse duration and tunnel-localised calling",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "duration_vs_calls.png", dpi=150)

    print("\n  duration band     to_arena: n / %with call / calls per s      to_nest: same")
    bands = [(0, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 1e9)]
    for lo, hi in bands:
        parts = []
        for direction in ("to_arena", "to_nest"):
            s = tv[(tv.direction == direction) & (tv.traverse_s >= lo) & (tv.traverse_s < hi)]
            if len(s):
                parts.append(f"{len(s):5d} / {100*s.has_calls.mean():4.0f}% / "
                             f"{(s.n_calls/s.window_s).mean():.3f}")
            else:
                parts.append("    - /    - /     -")
        label = f"{lo:>4.0f}-{hi:<4.0f}s" if hi < 1e9 else "  >16 s   "
        print(f"  {label}   {parts[0]}      {parts[1]}")
    tv.to_csv(out_dir / "traverse_calls.csv", index=False)
    print(f"\nwrote {out_dir}/duration_vs_calls.png")


if __name__ == "__main__":
    main()
