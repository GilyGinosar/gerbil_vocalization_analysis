#!/usr/bin/env python
"""Do gerbils call when they enter the tunnel? The whole date folder, per origin.

Pools every experiment's localiser output and asks the question the single-
experiment version could only hint at: are calls that the microphones place AT
THE TUNNEL clustered around the moment an animal enters it?

Three things make this different from the earlier attempts, each of which was
wrong in a way that mattered:

* Calls are split by ORIGIN, not by whether they fall inside the tunnel period.
  Calling starts before the animal enters and continues after it leaves, so an
  "inside versus outside" test returns ~1.0 whether or not the effect is real.
* The tunnel/nest threshold is each experiment's own, because mic gain and tube
  acoustics differ between recordings.
* The null is a within-file shuffle of the anchor times, because calling comes
  in colony-wide bouts and any test against a flat rate will find "effects"
  wherever a bout happened to coincide with activity.

    python scripts/analysis/transit_call_result.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/transit_result_2026_02
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
N_SHUFFLES = 1000
FILE_S = 360.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=float, default=8.0)
    parser.add_argument("--bin", type=float, default=0.5)
    args = parser.parse_args()

    scan = Path(args.scan)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traverses = pd.read_parquet(scan / "traverses_2026_02.parquet")
    traverses = traverses[traverses.single_animal]

    # each experiment's own threshold, from its own tunnel-empty calls
    parts = []
    for path in sorted((scan / "origin").glob("*/call_origin.csv")):
        exp = int(path.parent.name)
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            # an experiment with no scored calls writes a headerless empty file;
            # a couple of the 2026_02 experiments are only 2-6 videos long
            print(f"  exp {exp}: no calls scored, skipped")
            continue
        if table.empty:
            continue
        reference = table.loc[table.state == "tunnel empty", "tunnel_db_over_nest"]
        if len(reference) < 50:
            print(f"  exp {exp}: only {len(reference)} calibration calls, skipped")
            continue
        table["exp"] = exp
        table["tunnel_localised"] = table.tunnel_db_over_nest > reference.quantile(0.95)
        parts.append(table)
    if not parts:
        raise SystemExit("no localiser output found -- has the localise task file run?")
    calls = pd.concat(parts, ignore_index=True)
    print(f"{len(calls):,} calls from {calls.exp.nunique()} experiments; "
          f"{int(calls.tunnel_localised.sum()):,} tunnel-localised")

    edges = np.arange(-args.window, args.window + args.bin, args.bin)
    centres = edges[:-1] + args.bin / 2
    rng = np.random.default_rng(0)
    near = (centres >= -2.5) & (centres <= 1.0)

    def rate(anchors, times_by_key):
        counts = np.zeros(len(centres))
        for key, anchor in anchors:
            times = times_by_key.get(key)
            if times is None:
                continue
            lags = times - anchor
            lags = lags[(lags >= edges[0]) & (lags < edges[-1])]
            if lags.size:
                counts += np.histogram(lags, bins=edges)[0]
        return counts / max(len(anchors), 1) / args.bin

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    summary = []
    for ax, localised in zip(axes, (True, False)):
        subset = calls[calls.tunnel_localised == localised]
        times_by_key = {k: g.start_s.to_numpy() for k, g in subset.groupby(["exp", "file"])}
        for direction in ("to_arena", "to_nest"):
            sub = traverses[traverses.direction == direction]
            anchors = list(zip(zip(sub.exp, sub.file_num), sub.t_entry))
            observed = rate(anchors, times_by_key)
            draws = np.empty(N_SHUFFLES)
            for i in range(N_SHUFFLES):
                fake = [(k, rng.uniform(0, FILE_S)) for k, _ in anchors]
                draws[i] = rate(fake, times_by_key)[near].mean()
            value = observed[near].mean()
            p = float((draws >= value).mean())
            summary.append({"origin": "tunnel" if localised else "nest",
                            "direction": direction, "n_traverses": len(sub),
                            "rate_near_entry": round(value, 4),
                            "chance": round(float(np.median(draws)), 4),
                            "ratio": round(value / max(np.median(draws), 1e-9), 2),
                            "p": p})
            ax.plot(centres, observed, color=DIRECTION_COLOURS[direction], lw=2,
                    label=f"{direction} (n={len(sub):,})")
        ax.axvline(0, color="0.3", lw=1)
        ax.set_title("calls placed AT THE TUNNEL" if localised else "calls placed in the nest",
                     loc="left", fontsize=11)
        ax.set_xlabel("seconds from entering the tunnel")
        ax.grid(axis="y", color="0.92", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("calls / s / traverse")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("2026_02: calling around tunnel entry, split by where the call came from",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "entry_by_origin.png", dpi=150)

    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {out_dir}/entry_by_origin.png and summary.csv")


if __name__ == "__main__":
    main()
