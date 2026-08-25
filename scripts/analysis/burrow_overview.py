#!/usr/bin/env python
"""Everything on one page: call set x alignment x condition, for deciding what to chase.

Four columns -- every DAS class vs USV only, each anchored on entering and on
leaving the tunnel. Four rows -- to_nest as a whole, to_nest split by whether the
nest had already been calling in the seconds before entry, and to_arena as the
opposite-direction control. Every panel carries the same three series, in the
animal's own spatial order and shaded light to dark:

    arena_1 (the compartment it came from / went to) · tunnel-origin · nest-origin

Every series gets its raster and the three of them share the rate panel below,
so a column is one condition read from ticks to average. Row order is shared down
a column -- the same traverse is the same line in all three rasters -- which is
what lets you follow a call handing over from the tunnel to the nest.

Two things this is built to show at a glance. Whether the non-USV classes
(stacks is 29% of underground calls, and tracks occupancy) are carrying any
result -- compare a row across the two call sets. And whether the prior-nest
association is specific -- compare the two middle rows.

    python scripts/analysis/burrow_overview.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/overview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.raster_and_rate_tunnel import (  # noqa: E402
    ALL_TYPES, COLOURS, PAD_AFTER, PAD_BEFORE, USV_ONLY,
    epoch_bounds, load_or_collect, localised_sides, observed_and_null, score, shade, tint,
)

TYPESETS = (("every DAS class", ALL_TYPES, "all"), ("USV only", USV_ONLY, "usv"))
ALIGNS = ("entry", "exit")


def prior_nest(entry: dict, nest: dict, window: float) -> bool:
    times = nest.get(entry["key"])
    return bool(times is not None and times.size
                and ((times >= entry["t_entry"] - window)
                     & (times < entry["t_entry"])).any())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clear", type=float, default=3.0)
    parser.add_argument("--calls", default="underground,arena_1")
    parser.add_argument("--localiser-quantile", type=float, default=0.99)
    parser.add_argument("--prior-window", type=float, default=5.0)
    parser.add_argument("--max-lag", type=float, default=8.0)
    parser.add_argument("--bin", type=float, default=0.25, dest="bin_s",
                        help="0.25 s by default: this figure is for comparing shapes across "
                             "16 panels, and the clock drift makes anything finer cosmetic")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="shuffles behind the chance column of summary.csv; 0 (the "
                             "default here) skips them -- this figure is for looking")
    parser.add_argument("--near", default="-0.5,2.0")
    parser.add_argument("--recollect", action="store_true")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(-args.max_lag, args.max_lag + args.bin_s, args.bin_s)
    centres = edges[:-1] + args.bin_s / 2
    near = tuple(float(v) for v in args.near.split(","))
    rng = np.random.default_rng(0)

    # one collection and one localiser pass per call set
    world = {}
    for name, types, slug in TYPESETS:
        print(f"\n=== {name}")
        data, calls_by_loc = load_or_collect(
            scan, out_dir / slug, args.calls, args.clear, None, args.recollect, types)
        tunnel, nest, _ = localised_sides(scan, quantile=args.localiser_quantile,
                                          types=types)
        world[name] = dict(data=data, arena=calls_by_loc["arena_1"],
                           tunnel=tunnel, nest=nest)
        print(f"  tunnel-origin {sum(len(v) for v in tunnel.values()):,}   "
              f"nest-origin {sum(len(v) for v in nest.values()):,}")

    # rows: the conditions. `prior` is None for "every traverse".
    rows = (("to_nest — every traverse", "to_nest", None),
            (f"to_nest — nest called in the {args.prior_window:g} s before entry",
             "to_nest", True),
            (f"to_nest — nest SILENT in the {args.prior_window:g} s before entry",
             "to_nest", False),
            ("to_arena — every traverse  (control direction)", "to_arena", None))

    columns = [(ts, al) for ts in TYPESETS for al in ALIGNS]
    fig = plt.figure(figsize=(22, 12.5 * len(rows)))
    subfigs = np.atleast_1d(fig.subfigures(len(rows), 1, hspace=0.015))
    summary, ceiling, panels = [], [], []

    for r, (row_label, direction, want_prior) in enumerate(rows):
        grid = subfigs[r].subplots(4, 4, height_ratios=[2.2, 2.2, 2.2, 1.35],
                                   gridspec_kw={"hspace": 0.10, "wspace": 0.16})
        base = COLOURS[direction]
        for c, ((set_name, types, _), align) in enumerate(columns):
            w = world[set_name]
            entries = w["data"][direction]
            if want_prior is not None:
                entries = [e for e in entries
                           if prior_nest(e, w["nest"], args.prior_window) == want_prior]
            series = (("arena_1", w["arena"], tint(base, 0.32)),
                      ("tunnel-origin", w["tunnel"], base),
                      ("nest-origin", w["nest"], shade(base, 0.52)))
            # one order for the whole column, so a row is one traverse in all three
            order = sorted(range(len(entries)), key=lambda i: -entries[i]["in_tunnel"])
            lo, hi = epoch_bounds(entries, align)
            field = "t_entry" if align == "entry" else "t_out"

            for s, (label, table, colour) in enumerate(series):
                ax = grid[s][c]
                xs_acc, ys_acc = [], []
                for row_i, i in enumerate(order):
                    e = entries[i]
                    times = table.get(e["key"])
                    if times is None or not times.size:
                        continue
                    lag = times - e[field]
                    lag = lag[(lag >= lo[i]) & (lag <= hi[i])]
                    if lag.size:
                        xs_acc.append(lag)
                        ys_acc.append(np.full(lag.size, row_i))
                if xs_acc:
                    ax.plot(np.concatenate(xs_acc), np.concatenate(ys_acc), "|",
                            color=colour, ms=1.3, mew=0.5, alpha=0.85,
                            rasterized=True, zorder=2)
                # the far end of the epoch: the other landmark, per traverse
                edge = [hi[i] - PAD_AFTER if align == "entry" else lo[i] + PAD_BEFORE
                        for i in order]
                ax.plot(edge, range(len(order)), color="0.25", lw=0.9, zorder=3)
                ax.set_ylim(-20, len(order) + 20)
                ax.set_yticks([])
                ax.set_xticklabels([])
                if c == 0:
                    ax.set_ylabel(label, fontsize=9, color=colour)
                panels.append((ax, align))

                observed, draws, _ = observed_and_null(
                    entries, table, align, edges, centres, max(args.shuffle, 1), rng,
                    require_call=False, before=PAD_BEFORE, after=PAD_AFTER)
                grid[3][c].plot(centres, observed, color=colour, lw=1.9,
                                label=label if (r == 0 and c == 0) else None)
                ceiling.append(float(np.nanmax(observed)))
                if args.shuffle:
                    summary.append({"call_set": set_name, "align": align,
                                    "condition": row_label, "series": label,
                                    "n": len(entries),
                                    **score(observed, draws, centres, near)})

            rate_ax = grid[3][c]
            rate_ax.set_xlabel(f"seconds from {align} of the tunnel", fontsize=9)
            rate_ax.grid(axis="y", color="0.93", lw=0.8)
            rate_ax.set_axisbelow(True)
            panels.append((rate_ax, align))
            if c == 0:
                rate_ax.set_ylabel("calls / s / traverse", fontsize=9)
            grid[0][c].set_title(f"{set_name} · aligned to {align.upper()}    "
                                 f"(n={len(entries):,})", loc="left", fontsize=9.5)
        subfigs[r].suptitle(row_label, x=0.005, ha="left", fontsize=13,
                            color=COLOURS[direction])
        subfigs[r].subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.05)
        if r == 0:
            grid[3][0].legend(frameon=False, fontsize=8.5, loc="upper left")
        rate_axes = [grid[3][c] for c in range(4)]
        for ax in rate_axes:
            ax.set_ylim(0, np.ceil(max(ceiling) * 10) / 10)

    for ax, align in panels:
        ax.set_xlim(max(-args.max_lag, -PAD_BEFORE - 0.4) if align == "entry"
                    else -args.max_lag,
                    args.max_lag if align == "entry"
                    else min(args.max_lag, PAD_AFTER + 0.4))
        ax.axvline(0, color="0.35", lw=1.0, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(f"2026_02 burrow calling — every condition, rasters and rates\n"
                 f"epoch {int(PAD_BEFORE)} s before entry to {int(PAD_AFTER)} s after "
                 f"leaving · localiser cut at q={args.localiser_quantile} · "
                 f"{args.bin_s:g} s bins · rows sorted by time in tunnel, shared down "
                 f"each column",
                 x=0.004, y=0.999, ha="left", va="top", fontsize=13)
    out = out_dir / "burrow_overview.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
