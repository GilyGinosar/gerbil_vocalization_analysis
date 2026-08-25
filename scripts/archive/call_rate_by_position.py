#!/usr/bin/env python
"""Where in the tunnel do animals call? Call rate against position, both directions.

The time-aligned PSTHs put t=0 at a different physical place for each direction
-- to_arena enters at the nest end, to_nest at the arena end -- so the two curves
are not directly comparable. On a position axis they are: x=0 is the nest end of
the tunnel and x=1 the arena end for both.

Rate is calls per second OF TIME SPENT at that position, not calls per bin. An
animal crosses the middle of the tube quickly and lingers at the ends, so raw
counts would show peaks at the ends that are pure dwell time. Dividing by
occupancy is the same correction the colony-wide rate maps need.

Only the in-tunnel portion of each traverse is on this axis, because that is
where the animal has a position. The arrival burst that dominates the
time-aligned figure happens just AFTER the animal leaves at x=0, so it sits off
the left edge here by construction.

    python scripts/analysis/call_rate_by_position.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/position_2026_02
"""
from __future__ import annotations

import argparse
import csv
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

from scripts.video.burrow_transit_picker import file_index, load_calls  # noqa: E402

DIRECTION_COLOURS = {"to_arena": "#2f6fd0", "to_nest": "#d1642a"}
USV = ("high-freq", "warble")
FPS = 30
LEFT, RIGHT = 0.15, 0.75


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--bins", type=int, default=25)
    parser.add_argument("--from-csv", help="replot from a previous run's CSV, no recompute")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(0, 1, args.bins + 1)
    centres = (edges[:-1] + edges[1:]) / 2

    occupancy = {d: np.zeros(args.bins) for d in DIRECTION_COLOURS}   # seconds
    call_counts = {d: np.zeros(args.bins) for d in DIRECTION_COLOURS}
    n_traverses = {d: 0 for d in DIRECTION_COLOURS}

    exp_dirs = sorted(p for p in scan.iterdir() if p.is_dir() and p.name.isdigit())
    for exp_dir in exp_dirs:
        exp = int(exp_dir.name)
        calls = load_calls(exp)
        if not calls:
            continue
        usv = {k: np.sort(np.array([c for c, _, t in v if t in USV]))
               for k, v in calls.items()}
        rows = [r for path in sorted(exp_dir.glob("traverses*.csv"))
                for r in csv.DictReader(open(path))
                if str(r["single_animal"]).lower() == "true"]
        by_video: dict[str, list[dict]] = {}
        for row in rows:
            by_video.setdefault(row["video"], []).append(row)

        for video, group in by_video.items():
            track_path = exp_dir / "tracks" / f"{Path(video).stem}.parquet"
            if not track_path.exists():
                continue
            track = pd.read_parquet(track_path)
            frames = track.frame.to_numpy()
            xs = track.x.to_numpy()
            n_animals = track.n_animals.to_numpy()
            times = usv.get(file_index(video))
            if times is None or not len(times):
                continue

            for row in group:
                direction = row["direction"]
                entry, out = float(row["t_entry"]), float(row["t_out"])
                lo, hi = int(entry * FPS), int(out * FPS)
                if hi <= lo or hi >= len(frames):
                    continue
                window_x = xs[lo:hi]
                window_n = n_animals[lo:hi]
                valid = (window_n == 1) & np.isfinite(window_x)
                if valid.sum() < 3:
                    continue
                n_traverses[direction] += 1
                # occupancy: each valid frame is 1/FPS of a second at its position
                occupancy[direction] += np.histogram(window_x[valid], bins=edges)[0] / FPS
                # calls: the animal's position at the moment of each call
                in_window = times[(times >= entry) & (times <= out)]
                if in_window.size:
                    idx = np.clip((in_window * FPS).astype(int) - lo, 0, len(window_x) - 1)
                    at = window_x[idx]
                    ok = np.isfinite(at) & (window_n[idx] == 1)
                    if ok.any():
                        call_counts[direction] += np.histogram(at[ok], bins=edges)[0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for direction, colour in DIRECTION_COLOURS.items():
        seconds = occupancy[direction]
        rate = np.divide(call_counts[direction], seconds,
                         out=np.full(args.bins, np.nan), where=seconds > 5)
        axes[0].plot(centres, rate, color=colour, lw=2,
                     label=f"{direction} (n={n_traverses[direction]:,}, "
                           f"{seconds.sum():.0f} s in tunnel)")
        axes[1].plot(centres, seconds, color=colour, lw=2, label=direction)
        print(f"{direction}: {n_traverses[direction]:,} traverses, "
              f"{int(call_counts[direction].sum()):,} USV calls in tunnel, "
              f"{seconds.sum():.0f} s occupancy, overall "
              f"{call_counts[direction].sum()/max(seconds.sum(),1e-9):.3f} calls/s")

    for ax, title, ylabel in ((axes[0], "USV rate, corrected for time spent",
                               "USV calls / s in the tunnel"),
                              (axes[1], "time spent at each position (the denominator)",
                               "seconds, all traverses")):
        for level, label in ((LEFT, "0.15"), (RIGHT, "0.75")):
            ax.axvline(level, color="0.7", ls="--", lw=1)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("position along the tunnel   (0 = nest end,  1 = arena end)")
        # a little air past both ends so the tunnel mouths are visible as landmarks
        # rather than being clipped against the frame
        ax.set_xlim(-0.1, 1.1)
        for end in (0.0, 1.0):
            ax.axvline(end, color="0.45", lw=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", color="0.93", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("2026_02: where in the tunnel the calling happens (high-freq + warble)",
                 x=0.01, ha="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_dir / "rate_by_position.png", dpi=150)

    pd.DataFrame({"x": centres,
                  **{f"{d}_calls": call_counts[d] for d in DIRECTION_COLOURS},
                  **{f"{d}_seconds": occupancy[d] for d in DIRECTION_COLOURS}}
                 ).to_csv(out_dir / "rate_by_position.csv", index=False)
    print(f"\nwrote {out_dir}/rate_by_position.png")


if __name__ == "__main__":
    main()
