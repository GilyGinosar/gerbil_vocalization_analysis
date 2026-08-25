#!/usr/bin/env python
"""Is arrival special to the nest, or does arriving anywhere trigger calling?

Every earlier figure used underground calls only -- the two nest-end mics. So
"to_arena is flat at its own arrival" was never a real control: that animal
arrives in the arena, away from those microphones, and we simply were not
listening where it went.

This listens at both ends. The tunnel connects the nest to arena_1, so the
symmetric test is:

  to_nest  arriving  -> underground mics should peak   (already known)
                     -> arena_1 mics should not
  to_arena arriving  -> arena_1 mics SHOULD peak, if arrival per se drives calling
                     -> underground mics should not    (already known)

If arena_1 peaks for to_arena, the finding is "arrival triggers calling" and the
nest is not special. If it stays flat, the nest-specific claim finally rests on a
measured negative rather than on an absent microphone.

    python scripts/analysis/arrival_by_mic.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/arrival_by_mic
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.paths import experiment_audio_dir  # noqa: E402

COLOURS = {"to_arena": "#2f6fd0", "to_nest": "#d1642a"}
USV = {"high-freq", "warble"}
LOCATIONS = ("underground", "arena_1")
N_SHUFFLES = 500
FILE_S = 360.0


def load_by_location(exp: int) -> dict[str, dict[int, np.ndarray]]:
    """USV onsets per file index, for each assigned location."""
    path = experiment_audio_dir(exp) / "calls.csv"
    if not path.exists():
        return {}
    out: dict[str, dict[int, list]] = {loc: defaultdict(list) for loc in LOCATIONS}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] not in USV:
                continue
            location = row["assigned_location"]
            if location in out:
                out[location][int(row["file_num"])].append(float(row["start_time_file_sec"]))
    return {loc: {k: np.sort(np.array(v)) for k, v in files.items()}
            for loc, files in out.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--window", type=float, default=10.0)
    parser.add_argument("--bin", type=float, default=0.25)
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tv = pd.read_parquet(scan / f"traverses_{args.date}.parquet")
    tv = tv[tv.single_animal]

    tables: dict[str, dict[tuple[int, int], np.ndarray]] = {loc: {} for loc in LOCATIONS}
    for exp in sorted(tv.exp.unique()):
        per_location = load_by_location(int(exp))
        for location in LOCATIONS:
            for file_num, times in per_location.get(location, {}).items():
                tables[location][(int(exp), file_num)] = times
    for location in LOCATIONS:
        print(f"  {location:<12} {sum(len(v) for v in tables[location].values()):,} USV calls")

    edges = np.arange(-args.window, args.window + args.bin, args.bin)
    centres = edges[:-1] + args.bin / 2
    rng = np.random.default_rng(0)

    def rate(anchors, table):
        counts = np.zeros(len(centres))
        for key, anchor in anchors:
            times = table.get(key)
            if times is None:
                continue
            lags = times - anchor
            lags = lags[(lags >= edges[0]) & (lags < edges[-1])]
            if lags.size:
                counts += np.histogram(lags, bins=edges)[0]
        return counts / max(len(anchors), 1) / args.bin

    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, location in zip(axes, LOCATIONS):
        table = tables[location]
        for direction in ("to_arena", "to_nest"):
            sub = tv[tv.direction == direction]
            anchors = list(zip(zip(sub.exp, sub.file_num), sub.t_out))
            observed = rate(anchors, table)
            draws = np.empty((N_SHUFFLES, len(centres)))
            for i in range(N_SHUFFLES):
                draws[i] = rate([(k, rng.uniform(0, FILE_S)) for k, _ in anchors], table)
            lo, hi = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
            ax.fill_between(centres, lo, hi, color=COLOURS[direction], alpha=0.16, lw=0)
            ax.plot(centres, observed, color=COLOURS[direction], lw=2,
                    label=f"{direction} (n={len(sub):,})")
            near = (centres >= -0.5) & (centres <= 2.0)
            value, chance = observed[near].mean(), np.median(draws[:, near].mean(axis=1))
            summary.append({"mics": location, "arriving": direction,
                            "rate_at_arrival": round(float(value), 3),
                            "chance": round(float(chance), 3),
                            "ratio": round(float(value / max(chance, 1e-9)), 2),
                            "p": float((draws[:, near].mean(axis=1) >= value).mean())})
        ax.axvline(0, color="0.3", lw=1)
        where = "nest-end mics" if location == "underground" else "arena_1 mics"
        ax.set_title(f"heard on the {where}", loc="left", fontsize=11)
        ax.set_xlabel("seconds from leaving the tunnel (arrival)")
        ax.grid(axis="y", color="0.93", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("USV calls / s / traverse")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("2026_02: arrival, heard at both ends "
                 "(shaded = 95% of within-file shuffles)", x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "arrival_by_mic.png", dpi=150)

    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {out_dir}/arrival_by_mic.png")


if __name__ == "__main__":
    main()
