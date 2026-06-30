"""Bout raster of dense periods — eyeball within-area "conversations".

A conversation is treated as turn-taking *within one area* (arena animals among
themselves, burrow among themselves; they don't hear across areas), so arena and
underground are rendered as SEPARATE figures and never mixed.

We detect bouts with vocalization_analysis.bouts (the single source of truth for
thresholds), drop singletons, and reduce each bout to one segment (start -> stop)
colored by call type. Then we pick the densest time windows and draw, per window,
a "piano-roll": call-type lanes on y, time on x, each bout a colored bar. A
conversation reads as bars alternating across lanes / colors within a few minutes.

Only DENSE windows are shown (quiet stretches are skipped): windows are ranked by
bout count and the top --top-n are drawn, tallest-plot style.

Usage:
    python scripts/analysis/run_bout_raster.py --dates 2026_02
    python scripts/analysis/run_bout_raster.py --dates 2026_02 --window-min 5 --top-n 16
    python scripts/analysis/run_bout_raster.py --dates 2026_02 --split-arenas   # arena_1, arena_2 separate
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import BASE_PROCESSED, load_all_calls  # noqa: E402
from vocalization_analysis.bouts import BOUT_THRESHOLDS, detect_bouts_for_types  # noqa: E402

# Date folder(s) to run on — EDIT HERE to switch experiment (e.g. ["2025_10"]).
# Available: 2024_12, 2025_03, 2025_07, 2025_10, 2026_02. Overridable with --dates.
DEFAULT_DATES = ["2026_02"]
WINDOW_MIN = 5                  # width of each dense window shown (minutes)
TOP_N = 12                      # number of windows to draw per area
MIN_BOUT_CALLS = 2             # drop singletons; keep bouts with >= this many calls

# Lane order top->bottom, matching the ethograms (newborn at the bottom).
# warble/high-freq/stacks/alarm are drawn as bouts (bars); newborn has no bout
# thresholds so each call is drawn as a tick.
CALL_TYPE_ORDER = ["alarm", "high-freq", "warble", "stacks", "newborn"]
TICK_TYPES = {"newborn"}        # plotted as one tick per call, not bouted
CALL_COLORS = {
    "warble": "#4daf4a",
    "high-freq": "#377eb8",
    "stacks": "#ff7f00",
    "alarm": "#e41a1c",
    "newborn": "#984ea3",
}
# Which assigned_location values make up each "area" (animals hear within an area).
AREAS_POOLED = {"arena": ["arena_1", "arena_2"], "underground": ["underground"]}
AREAS_SPLIT = {"arena_1": ["arena_1"], "arena_2": ["arena_2"], "underground": ["underground"]}


def bout_segments(df: pd.DataFrame) -> pd.DataFrame:
    """One row per bout (bars) plus one row per tick-type call (e.g. newborn).

    Columns: event_type, assigned_location, exp, bout_id, start, stop, size.
    """
    bout_types = [ct for ct in CALL_TYPE_ORDER if ct in BOUT_THRESHOLDS]
    bb = detect_bouts_for_types(df, bout_types)                  # adds bout_id/size/kind
    bb = bb[bb["bout_size"] >= MIN_BOUT_CALLS]
    seg = (
        bb.groupby(["event_type", "assigned_location", "exp", "bout_id"], observed=True)
        .agg(start=("start_time_real", "min"),
             stop=("stop_time_real", "max"),
             size=("bout_size", "first"))
        .reset_index()
    )
    # tick-types: every call is its own mark (no bout thresholds)
    tk = df[df["event_type"].isin(TICK_TYPES)]
    if not tk.empty:
        tk_seg = pd.DataFrame({
            "event_type": tk["event_type"].values,
            "assigned_location": tk["assigned_location"].values,
            "exp": tk["exp"].values,
            "bout_id": np.arange(len(tk)),
            "start": tk["start_time_real"].values,
            "stop": tk["start_time_real"].values,
            "size": 1,
        })
        seg = pd.concat([seg, tk_seg], ignore_index=True)
    return seg


def top_windows(seg_area: pd.DataFrame, window_min: int, top_n: int):
    """Windows ranked by call-type ALTERNATION (turn-taking), not raw count.

    Score = number of consecutive *bouts* whose call type differs from the
    previous one within the window. Tick-types (newborn) are still plotted but
    EXCLUDED from the score, so the ranking finds adult turn-taking rather than
    pup-call density. Returns up to top_n (w0, n_events, n_switches) chronological.
    """
    if seg_area.empty:
        return []
    s = seg_area.copy()
    s["win"] = s["start"].dt.floor(f"{window_min}min")
    rows = []
    for w0, g in s.groupby("win"):
        g = g.sort_values("start")
        adult = g[~g["event_type"].isin(TICK_TYPES)]["event_type"].to_numpy()
        switches = int((adult[1:] != adult[:-1]).sum()) if len(adult) > 1 else 0
        rows.append((w0, len(g), switches))
    rows.sort(key=lambda r: r[2], reverse=True)                  # most adult alternation first
    return sorted(rows[:top_n])                                  # chronological


def plot_area_raster(area, seg_area, windows, window_min, types, date_folder, out_path):
    n = len(windows)
    if n == 0:
        print(f"  {area}: no bouts to plot")
        return
    # y goes up, so reverse the top->bottom list to place the first type on top
    # and newborn (last) on the bottom, matching the ethograms.
    lane = {ct: len(types) - 1 - i for i, ct in enumerate(types)}
    fig, axes = plt.subplots(n, 1, figsize=(13, max(3.0, n * 0.85 + 1.2)),
                             squeeze=False, sharex=True)
    axes = axes[:, 0]
    for ax, (w0, nev, sw) in zip(axes, windows):
        w0 = pd.Timestamp(w0)
        w1 = w0 + pd.Timedelta(minutes=window_min)
        win = seg_area[(seg_area["start"] >= w0) & (seg_area["start"] < w1)]
        for _, b in win.iterrows():
            ct = b["event_type"]
            y = lane[ct]
            x0 = (b["start"] - w0).total_seconds() / 60.0
            if ct in TICK_TYPES:
                ax.vlines(x0, y - 0.35, y + 0.35, color=CALL_COLORS.get(ct, "k"), lw=0.8)
            else:
                x1 = max((b["stop"] - w0).total_seconds() / 60.0, x0 + window_min * 0.003)
                ax.hlines(y, x0, x1, color=CALL_COLORS.get(ct, "k"), lw=5)
        ax.set_ylim(-0.6, len(types) - 0.4)
        ax.set_yticks([])
        ax.set_xlim(0, window_min)
        ax.set_ylabel(f"{w0:%m-%d %H:%M}\n({nev} ev, {sw} sw)", rotation=0, ha="right",
                      va="center", fontsize=7)
        ax.grid(axis="x", color="0.9", lw=0.5)
    axes[-1].set_xlabel(f"minutes within window (window = {window_min} min)")
    handles = [mpatches.Patch(color=CALL_COLORS[ct], label=ct) for ct in types]
    fig.legend(handles=handles, loc="lower center", ncol=len(types), fontsize=9,
               framealpha=0.95, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(
        f"Bout raster — {date_folder}  [{area}] — {n} windows by adult alternation "
        f"({window_min}-min); bar = bout, tick = newborn call",
        fontsize=12, y=1.04,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  {area}: wrote {out_path}")


def run_for_date(date_folder, window_min, top_n, split_arenas, out_dir):
    df = load_all_calls(date_folder)
    seg = bout_segments(df)
    types = [ct for ct in CALL_TYPE_ORDER if ct in set(seg["event_type"])]
    areas = AREAS_SPLIT if split_arenas else AREAS_POOLED
    print(f"{date_folder}: {len(seg):,} bouts/ticks across types {types}")
    for area, locs in areas.items():
        seg_area = seg[seg["assigned_location"].isin(locs)]
        windows = top_windows(seg_area, window_min, top_n)
        out_path = out_dir / date_folder / f"bout_raster_{date_folder}_{area}_{window_min}min.png"
        plot_area_raster(area, seg_area, windows, window_min, types, date_folder, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--window-min", type=int, default=WINDOW_MIN)
    ap.add_argument("--top-n", type=int, default=TOP_N)
    ap.add_argument("--pool-arenas", action="store_true",
                    help="pool arena_1 + arena_2 into one 'arena' (default: keep them separate)")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.window_min, args.top_n, not args.pool_arenas, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
