#!/usr/bin/env python
"""Arrival close-up: the same traverses on a +/-3 s axis, as a raster and as a heatmap.

`raster_and_rate.py` draws +/-10 s, which is the right window for showing that
the burst decays over ~10 s but the wrong one for asking what happens AT the
moment of arrival -- 0.25 s of behaviour occupies 1% of that axis. This is the
close-up, and the start of the figure aimed squarely at "do they call on
entering a new space".

Two views of one thing, stacked so they line up:

  raster   every traverse, one tick per call. Honest about sparseness, but
           2,700 rows in 700 px of panel is texture, not data.
  heatmap  the same rows averaged in blocks, so density is visible. It cannot
           show you a single traverse; that is what the raster above is for.

Row order is shared between them, and matters more than it looks. The default
sorts by time in the tunnel, which is independent of when the animal called.
--sort-rows latency sorts by the latency to the first call after the anchor, and
that WILL draw a clean diagonal edge even in Poisson noise -- it is sorting on
the very quantity you are then looking at. Use it to see the spread of latencies,
never as evidence that a response is time-locked.

    python scripts/analysis/arrival_response.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --collected exports/raster_clean/collected.npz --out-dir exports/arrival_3s
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.raster_and_rate import (  # noqa: E402
    CACHE_VERSION, COLOURS, COMPARTMENT, collect,
)

WINDOW = 3.0     # the close-up; the collection holds +/-10 s, this is a slice of it
BIN = 0.1
BLOCK = 25       # traverses averaged into one heatmap row


def load_or_collect(scan: Path, out_dir: Path, calls: str, clear: float,
                    collected: Path | None, recollect: bool):
    """The cached collection, from a sibling script's cache if one is offered.

    The npz format and its settings string are shared with `raster_and_rate.py`,
    so pointing --collected at that script's cache reuses a walk over 3,775
    videos rather than repeating it.
    """
    settings = f"v{CACHE_VERSION}|{calls}|{clear}"
    cache = collected or (out_dir / "collected.npz")
    if cache.exists() and not recollect:
        blob = np.load(cache, allow_pickle=True)
        if blob.get("settings", np.array("")).item() == settings:
            print(f"loaded {cache}")
            return blob["data"].item(), blob["calls_by_loc"].item()
        print(f"{cache} was built with other settings; rebuilding")
    data, occupancy, lag_sum, pos_edges, calls_by_loc = collect(
        scan, tuple(calls.split(",")), clear)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / "collected.npz"
    np.savez(target, data=np.array(data, dtype=object),
             occupancy=np.array(occupancy, dtype=object),
             lag_sum=np.array(lag_sum, dtype=object), pos_edges=pos_edges,
             calls_by_loc=np.array(calls_by_loc, dtype=object),
             settings=np.array(settings))
    print(f"collected and cached -> {target}")
    return data, calls_by_loc


def row_order(entries: list[dict], field: str, how: str) -> list[int]:
    """Which traverse goes on which row, shared by the raster and the heatmap."""
    if how == "latency":
        # traverses that never call have no latency; they go last rather than
        # being dropped, so the row count still matches the population
        def key(i):
            v = entries[i][field]
            after = v[v > 0]
            return float(after.min()) if after.size else np.inf
        return sorted(range(len(entries)), key=key)
    return sorted(range(len(entries)), key=lambda i: -entries[i]["in_tunnel"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--collected", help="reuse another run's collected.npz "
                                            "(e.g. exports/raster_clean/collected.npz)")
    parser.add_argument("--calls", default="underground,arena_1",
                        help="which assigned locations to pool (default both ends)")
    parser.add_argument("--clear", type=float, default=3.0,
                        help="clean-transit filter, as in raster_and_rate.py")
    parser.add_argument("--align", choices=("exit", "entry"), default="exit",
                        help="exit = arrival, where the effect is (default)")
    parser.add_argument("--sort-rows", choices=("in_tunnel", "latency"), default="in_tunnel",
                        help="row order. 'latency' draws a diagonal even in noise -- read "
                             "the docstring before using it as evidence.")
    parser.add_argument("--recollect", action="store_true")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, calls_by_loc = load_or_collect(
        scan, out_dir, args.calls, args.clear,
        Path(args.collected) if args.collected else None, args.recollect)

    field = "lags" if args.align == "exit" else "lags_entry"
    by_loc_field = "lags_by_loc" if args.align == "exit" else "entry_by_loc"
    moment = "leaving the tunnel (arrival)" if args.align == "exit" \
        else "entering the tunnel"
    edges = np.arange(-WINDOW, WINDOW + BIN, BIN)
    centres = edges[:-1] + BIN / 2

    # heatmaps first, so both directions can share one colour scale -- two panels
    # with independent scales invite reading a colour as a rate
    order, images = {}, {}
    for direction in COLOURS:
        entries = data[direction]
        order[direction] = row_order(entries, field, args.sort_rows)
        blocks = [order[direction][i:i + BLOCK]
                  for i in range(0, len(order[direction]), BLOCK)]
        image = np.zeros((len(blocks), len(centres)))
        for r, block in enumerate(blocks):
            lags = [entries[i][field] for i in block]
            lags = np.concatenate(lags) if any(v.size for v in lags) else np.empty(0)
            image[r] = np.histogram(lags, bins=edges)[0] / len(block) / BIN
        images[direction] = image
    top = max(np.percentile(im, 99) for im in images.values())

    fig, axes = plt.subplots(3, 2, figsize=(13, 13), height_ratios=[2.6, 1.7, 1.0],
                             gridspec_kw={"hspace": 0.12, "wspace": 0.16})
    rates = []
    for col, direction in enumerate(("to_nest", "to_arena")):
        entries = data[direction]
        colour = COLOURS[direction]
        rows = order[direction]

        # --- raster: one tick per call, coloured by the compartment that heard it
        acc = {loc: ([], []) for loc in COMPARTMENT}
        edge_x, ys = [], []
        for row, i in enumerate(rows):
            e = entries[i]
            # where the animal was on the other side of the anchor: for arrival that
            # is when it entered, so the line is the in-tunnel period running left
            edge_x.append(-e["in_tunnel"] if args.align == "exit" else e["in_tunnel"])
            ys.append(row)
            for loc, v in e.get(by_loc_field, {}).items():
                if loc in COMPARTMENT and v.size:
                    acc[loc][0].append(v)
                    acc[loc][1].append(np.full(v.size, row))
        for loc, (xs_acc, ys_acc) in acc.items():
            if xs_acc:
                axes[0][col].plot(np.concatenate(xs_acc), np.concatenate(ys_acc), "|",
                                  color=COMPARTMENT[loc], ms=1.4, mew=0.5, alpha=0.85,
                                  rasterized=True, zorder=2)
        axes[0][col].plot(edge_x, ys, color="0.25", lw=1.0, zorder=3)
        axes[0][col].set_ylim(-20, len(rows) + 20)
        axes[0][col].set_yticks([])
        axes[0][col].set_title(f"{direction}   (all {len(rows):,} clean transits)",
                               loc="left", fontsize=11, color=colour)

        # --- heatmap: the same rows, in blocks of BLOCK, so density is legible
        image = images[direction]
        im = axes[1][col].imshow(image, aspect="auto", origin="lower", cmap="magma",
                                 vmin=0, vmax=top, interpolation="nearest",
                                 extent=(-WINDOW, WINDOW, 0, len(rows)))
        axes[1][col].set_yticks([])
        bar = fig.colorbar(im, ax=axes[1][col], pad=0.01, fraction=0.045)
        bar.set_label("USV calls / s / traverse", fontsize=8)
        bar.ax.tick_params(labelsize=8)

        # --- the average underneath, so the heatmap has a scale to be read against
        lags = np.concatenate([e[field] for e in entries if e[field].size])
        rate = np.histogram(lags, bins=edges)[0] / len(entries) / BIN
        rates.append(rate)
        axes[2][col].plot(centres, rate, color=colour, lw=2)
        axes[2][col].set_xlabel(f"seconds from {moment}")

        for r in range(3):
            axes[r][col].set_xlim(-WINDOW, WINDOW)
            axes[r][col].axvline(0, color="0.85" if r == 1 else "0.35", lw=1.1, zorder=4)
            for side in ("top", "right"):
                axes[r][col].spines[side].set_visible(False)
        for r in (0, 1):
            axes[r][col].set_xticklabels([])
        axes[2][col].grid(axis="y", color="0.93", lw=0.8)
        axes[2][col].set_axisbelow(True)

    sorted_by = "time in tunnel" if args.sort_rows == "in_tunnel" \
        else "latency to first call  (SORTED ON THE MEASURE — see --help)"
    axes[0][0].set_ylabel(f"traverse\n(sorted by {sorted_by})", fontsize=9)
    axes[1][0].set_ylabel(f"the same rows,\n{BLOCK} traverses per line", fontsize=9)
    axes[2][0].set_ylabel("USV calls / s / traverse", fontsize=9)
    axes[0][0].legend(handles=[Line2D([0], [0], color=c, lw=5, label=loc)
                               for loc, c in COMPARTMENT.items()],
                      frameon=False, fontsize=8, loc="lower left",
                      title="DAS compartment", title_fontsize=8)
    top_rate = max(r.max() for r in rates)
    for ax in axes[2]:
        ax.set_ylim(0, np.ceil(top_rate * 10) / 10)
    fig.suptitle(f"2026_02: the {int(2 * WINDOW)} s around {moment} — every traverse, "
                 f"and the same traverses as density (high-freq + warble)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = out_dir / f"arrival_{int(WINDOW)}s_{args.align}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    for direction in ("to_nest", "to_arena"):
        entries = data[direction]
        lags = np.concatenate([e[field] for e in entries if e[field].size])
        rate = np.histogram(lags, bins=edges)[0] / len(entries) / BIN
        peak = centres[int(np.argmax(rate))]
        print(f"{direction}: n={len(entries):,}  peak {rate.max():.2f} calls/s at "
              f"{peak:+.2f} s  (edge bins {rate[0]:.2f} / {rate[-1]:.2f})")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
