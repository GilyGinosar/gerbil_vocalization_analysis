"""Daily call-rhythm curves: mean hour-of-day rate per call type, arena vs underground.

Collapses a date folder's whole experiment into one average day. For each
(location, call type) the per-day hour-of-day rate (calls / recorded minute) is
averaged across days; the line is the mean and the band is ±SEM across days.
This is the rigorous, day-averaged version of the ethogram's circadian story
(arena calls diurnal, burrow calls ~flat).

Reuses the circadian-day binning + coverage helpers from run_ethogram so the
day boundary (ROW_START_HOUR), location grouping and call-type order match.

Usage:
    python scripts/analysis/run_daily_rhythm.py --dates 2026_02
    python scripts/analysis/run_daily_rhythm.py --dates 2026_02 --bin-min 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from light_cycle import get_light_cycle_for_month  # noqa: E402
from run_ethogram import (  # noqa: E402  (shared helpers, single source of truth)
    BASE_PROCESSED,
    CALL_TYPE_ORDER,
    LOCATION_GROUPS,
    ROW_START_HOUR,
    counts_grid,
    coverage_grid,
    day_axis,
    load_all_calls,
)

BIN_MINUTES = 30                       # wider than the ethogram default: smoother curves
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}


def _fmt_hour(t: float) -> str:
    return f"{int(round(t)) % 24:02d}:00"


def plot_daily_rhythm(
    df, date_folder, day0, days, covered_min, call_types, bin_minutes,
    light_start, light_end, x0, out_path,
):
    groups = list(LOCATION_GROUPS.items())
    n_days, n_hourbins = len(days), covered_min.shape[1]
    centers = x0 + (np.arange(n_hourbins) + 0.5) * (bin_minutes / 60.0)
    enough = covered_min >= 0.5 * bin_minutes          # trust cells with >= half the bin recorded

    fig, axes = plt.subplots(len(call_types), 1, figsize=(9, 1.9 * len(call_types) + 1.2),
                             sharex=True, squeeze=False)
    axes = axes[:, 0]
    for i, ct in enumerate(call_types):
        ax = axes[i]
        ax.axvspan(x0, light_start, color="0.88", zorder=0)        # night before lights-on
        ax.axvspan(light_end, x0 + 24, color="0.88", zorder=0)     # night after lights-off
        ax.axvspan(light_start, light_end, color="#fff6c8", zorder=0)
        for gname, locs in groups:
            sub = df[df["assigned_location"].isin(locs)]
            counts = counts_grid(sub[sub["event_type"] == ct], day0, n_days,
                                 n_hourbins, bin_minutes, x0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = np.where(enough, counts / covered_min, np.nan)
            mean = np.nanmean(rate, axis=0)
            nval = np.sum(~np.isnan(rate), axis=0)
            sem = np.nanstd(rate, axis=0) / np.sqrt(np.maximum(nval, 1))
            color = LOCATION_COLORS.get(gname)
            ax.plot(centers, mean, color=color, lw=1.6, label=gname, zorder=3)
            ax.fill_between(centers, mean - sem, mean + sem, color=color, alpha=0.25, lw=0, zorder=2)
        ax.set_ylabel(f"{ct}\ncalls/min", fontsize=8)
        ax.set_xlim(x0, x0 + 24)
        ax.set_ylim(bottom=0)
        if i == 0:
            ax.legend(fontsize=8, loc="upper right", framealpha=0.9)

    ticks = np.arange(x0, x0 + 25, 2)
    axes[-1].set_xticks(ticks)
    axes[-1].set_xticklabels([_fmt_hour(t) for t in ticks])
    axes[-1].set_xlabel("hour of day")
    fig.suptitle(
        f"Daily call rhythm — {date_folder}   "
        f"(mean ±SEM over {n_days} days, {bin_minutes}-min bins; "
        f"lights {light_start:02d}:00–{light_end:02d}:00 = yellow)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, bin_minutes, out_dir, light_cycle=None):
    df = load_all_calls(date_folder)
    present = [ct for ct in CALL_TYPE_ORDER if ct in set(df["event_type"])]
    extra = sorted(set(df["event_type"]) - set(CALL_TYPE_ORDER))
    call_types = present + extra

    light_start, light_end = light_cycle or get_light_cycle_for_month(date_folder)
    if (24 * 60) % bin_minutes != 0 or (ROW_START_HOUR * 60) % bin_minutes != 0:
        raise SystemExit("--bin-min must divide 1440 and ROW_START_HOUR*60 evenly.")
    x0 = ROW_START_HOUR
    day0, days = day_axis(df, x0)
    n_hourbins = (24 * 60) // bin_minutes
    covered_min = coverage_grid(date_folder, day0, len(days), n_hourbins, bin_minutes, x0)
    out_path = out_dir / date_folder / f"daily_rhythm_{date_folder}_{bin_minutes}min.png"
    plot_daily_rhythm(df, date_folder, day0, days, covered_min, call_types,
                      bin_minutes, light_start, light_end, x0, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"], help="date folder(s)")
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES, help="time-bin width in minutes")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms",
                    help="output root (one subfolder per date)")
    ap.add_argument("--light-cycle", type=int, nargs=2, metavar=("ON", "OFF"), default=None,
                    help="override lights-on/off hours (default: per-date from light_cycle.py)")
    args = ap.parse_args()
    light_cycle = tuple(args.light_cycle) if args.light_cycle else None
    for date_folder in args.dates:
        run_for_date(date_folder, args.bin_min, args.out_dir, light_cycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
