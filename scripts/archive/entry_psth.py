#!/usr/bin/env python
"""Call-rate PSTHs around tunnel entry and exit -- total rate, immune to relabelling.

The origin-split PSTH has a confound: while an animal is in the tunnel its calls
are classified tunnel-origin regardless of how much it is calling, so a peak in
that bin at t=0 can be pure bookkeeping. The dwell control suggests it largely
is.

TOTAL call rate has no such problem. Relabelling moves calls between the origin
bins; it cannot change how many calls there are. So if crossing makes an animal
call more, the total rate must rise at entry -- and if it does not, the "calls
at transit" story is about which mic heard them, not about behaviour.

Aligned to entering the tunnel and to leaving it, per direction, per call type,
against a within-file shuffle of the anchor times (calling comes in colony-wide
bouts, so a flat-rate null finds spurious structure wherever a bout coincided
with activity).

    python scripts/analysis/entry_psth.py --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --out-dir exports/entry_psth_2026_02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video.burrow_transit_picker import load_calls  # noqa: E402

DIRECTION_COLOURS = {"to_arena": "#2f6fd0", "to_nest": "#d1642a"}

# DAS separates high-freq from warble unreliably, and their PSTHs correlate at
# r=0.945 -- same baseline, rise and decay -- so the split carries no information
# and they are analysed together. Stacks is NOT folded in: it anti-correlates
# (r=-0.745), tracking occupancy rather than the event, and at a third of all
# calls it would cancel the signal. Newborn is pups in the nest, a confound for
# an arrival-at-the-nest question rather than a contribution to it.
GROUPS = {"USV (high-freq + warble)": ("high-freq", "warble"),
          "stacks": ("stacks",),
          "newborn": ("newborn",)}
N_SHUFFLES = 500
FILE_S = 360.0


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

    # every underground call, by (exp, file); no audio, no localiser
    times: dict[tuple[int, int], np.ndarray] = {}
    typed: dict[str, dict[tuple[int, int], np.ndarray]] = {}
    for exp in sorted(tv.exp.unique()):
        for file_num, rows in load_calls(int(exp)).items():
            key = (int(exp), int(file_num))
            starts = np.array([c for c, _, _ in rows])
            times[key] = np.sort(starts)
            for event_type in {t for _, _, t in rows}:
                sel = np.sort(np.array([c for c, _, t in rows if t == event_type]))
                typed.setdefault(event_type, {})[key] = sel
    print(f"{sum(len(v) for v in times.values()):,} underground calls, "
          f"{len(tv):,} traverses, {tv.exp.nunique()} experiments")

    usv: dict[tuple[int, int], np.ndarray] = {}
    for member in ("high-freq", "warble"):
        for key, arr in typed.get(member, {}).items():
            usv[key] = np.sort(np.concatenate([usv.get(key, np.empty(0)), arr]))
    print(f"  USV (high-freq + warble): {sum(len(v) for v in usv.values()):,} calls")

    edges = np.arange(-args.window, args.window + args.bin, args.bin)
    centres = edges[:-1] + args.bin / 2
    rng = np.random.default_rng(0)

    def rate(anchors, table):
        counts = np.zeros(len(centres))
        for key, anchor in anchors:
            t = table.get(key)
            if t is None:
                continue
            lags = t - anchor
            lags = lags[(lags >= edges[0]) & (lags < edges[-1])]
            if lags.size:
                counts += np.histogram(lags, bins=edges)[0]
        return counts / max(len(anchors), 1) / args.bin

    def band(anchors, table):
        draws = np.empty((N_SHUFFLES, len(centres)))
        for i in range(N_SHUFFLES):
            draws[i] = rate([(k, rng.uniform(0, FILE_S)) for k, _ in anchors], table)
        return np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)

    summary = []
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (align, column, label) in zip(axes, (("entry", "t_entry", "entering the tunnel"),
                                                 ("exit", "t_out", "leaving the tunnel"))):
        for direction in ("to_arena", "to_nest"):
            sub = tv[tv.direction == direction]
            anchors = list(zip(zip(sub.exp, sub.file_num), sub[column]))
            observed = rate(anchors, usv)
            lo, hi = band(anchors, usv)
            ax.fill_between(centres, lo, hi, color=DIRECTION_COLOURS[direction], alpha=0.16, lw=0)
            ax.plot(centres, observed, color=DIRECTION_COLOURS[direction], lw=2,
                    label=f"{direction} (n={len(sub):,})")
            near = (centres >= -1) & (centres <= 1)
            summary.append({"align": align, "direction": direction, "n": len(sub),
                            "rate_pm1s": round(float(observed[near].mean()), 4),
                            "chance": round(float((lo[near].mean() + hi[near].mean()) / 2), 4),
                            "above_band": bool(observed[near].mean() > hi[near].mean())})
        ax.axvline(0, color="0.3", lw=1)
        ax.set_title(f"aligned to {label}", loc="left", fontsize=11)
        ax.set_xlabel(f"seconds from {label}")
        ax.grid(axis="y", color="0.93", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("USV calls / s / traverse")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("2026_02: USV rate (high-freq + warble) around the tunnel "
                 "(shaded = 95% of within-file shuffles)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "total_rate.png", dpi=150)
    plt.close(fig)

    # grouped rather than per-type; see GROUPS for why
    grouped: dict[str, dict] = {}
    for name, members in GROUPS.items():
        merged: dict[tuple[int, int], np.ndarray] = {}
        for member in members:
            for key, arr in typed.get(member, {}).items():
                merged[key] = np.sort(np.concatenate([merged.get(key, np.empty(0)), arr]))
        if merged:
            grouped[name] = merged
    typed = grouped
    present = list(grouped)
    fig, axes = plt.subplots(1, len(present), figsize=(4.2 * len(present), 3.8), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, event_type in zip(axes, present):
        for direction in ("to_arena", "to_nest"):
            sub = tv[tv.direction == direction]
            anchors = list(zip(zip(sub.exp, sub.file_num), sub.t_entry))
            ax.plot(centres, rate(anchors, typed[event_type]),
                    color=DIRECTION_COLOURS[direction], lw=1.8, label=direction)
        ax.axvline(0, color="0.3", lw=1)
        ax.set_title(event_type, loc="left", fontsize=10)
        ax.set_xlabel("s from entry")
        ax.grid(axis="y", color="0.93", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("calls / s / traverse")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("2026_02: total call rate around tunnel entry, by call type",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_dir / "total_rate_by_type.png", dpi=150)

    table = pd.DataFrame(summary)
    table.to_csv(out_dir / "summary.csv", index=False)
    print()
    print(table.to_string(index=False))
    print(f"\nwrote {out_dir}/total_rate.png and total_rate_by_type.png")


if __name__ == "__main__":
    main()
