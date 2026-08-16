"""Show that circadian gating is on the surface (arena), not in the burrow.

Two panels, both designed to isolate rhythm *shape* from rate *magnitude*:

  (A) Mean hour-of-day rate, each location's curve normalised to its own 24-h
      mean ("relative rate"). A flat line at 1.0 = arrhythmic; swings = rhythmic.
      Pooled over all call types, arena vs underground, ±SEM across days.
  (B) Modulation index = log2(day-rate / night-rate) per call type, arena vs
      underground. Bars above 0 = diurnal; near 0 = no day/night gating.

Reuses the circadian-day binning + coverage helpers from run_ethogram.

Usage:
    python scripts/analysis/run_circadian_gating.py --dates 2026_02
    python scripts/analysis/run_circadian_gating.py --dates 2026_02 --bin-min 30
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
from run_ethogram import (  # noqa: E402
    BASE_PROCESSED, CALL_TYPE_ORDER, LOCATION_GROUPS, ROW_START_HOUR,
    counts_grid, coverage_grid, day_axis, load_all_calls,
)

BIN_MINUTES = 30
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}


def _fmt_hour(t: float) -> str:
    return f"{int(round(t)) % 24:02d}:00"


def _mean_hour_profile(rate_days: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean and SEM across days of a (n_days, n_hourbins) per-day rate grid."""
    mean = np.nanmean(rate_days, axis=0)
    nval = np.sum(~np.isnan(rate_days), axis=0)
    sem = np.nanstd(rate_days, axis=0) / np.sqrt(np.maximum(nval, 1))
    return mean, sem


def plot_gating(df, date_folder, day0, days, covered_min, call_types, bin_minutes,
                light_start, light_end, x0, out_path):
    groups = list(LOCATION_GROUPS.items())
    n_days, n_hourbins = len(days), covered_min.shape[1]
    centers = x0 + (np.arange(n_hourbins) + 0.5) * (bin_minutes / 60.0)
    clock = (centers) % 24
    is_day = (clock >= light_start) & (clock < light_end)
    enough = covered_min >= 0.5 * bin_minutes

    def loc_rate(sub):
        counts = counts_grid(sub, day0, n_days, n_hourbins, bin_minutes, x0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(enough, counts / covered_min, np.nan)

    fig, (axA, axB) = plt.subplots(2, 1, figsize=(9, 7),
                                   gridspec_kw={"height_ratios": [1.3, 1]})

    # --- Panel A: pooled, mean-normalised hour-of-day rhythm ---
    axA.axvspan(x0, light_start, color="0.88", zorder=0)
    axA.axvspan(light_end, x0 + 24, color="0.88", zorder=0)
    axA.axvspan(light_start, light_end, color="#fff6c8", zorder=0)
    axA.axhline(1.0, color="0.4", lw=0.8, ls="--", zorder=1)
    for gname, locs in groups:
        rate = loc_rate(df[df["assigned_location"].isin(locs)])
        mean, sem = _mean_hour_profile(rate)
        base = np.nanmean(mean)
        if not base:
            continue
        m, s = mean / base, sem / base
        cv = np.nanstd(mean) / base                      # rhythm amplitude (CV of the profile)
        color = LOCATION_COLORS.get(gname)
        axA.plot(centers, m, color=color, lw=2.0, zorder=3,
                 label=f"{gname}  (amplitude CV={cv:.2f})")
        axA.fill_between(centers, m - s, m + s, color=color, alpha=0.25, lw=0, zorder=2)
    axA.set_xlim(x0, x0 + 24)
    axA.set_ylim(bottom=0)
    axA.set_ylabel("call rate relative\nto daily mean")
    axA.set_title("A. Daily rhythm of total calling — flat = arrhythmic, swings = gated", fontsize=10, loc="left")
    axA.legend(fontsize=9, loc="upper right", framealpha=0.9)
    ticks = np.arange(x0, x0 + 25, 2)
    axA.set_xticks(ticks); axA.set_xticklabels([_fmt_hour(t) for t in ticks])
    axA.set_xlabel("hour of day")

    # --- Panel B: per-call-type day/night modulation index ---
    eps = 1e-3
    width = 0.38
    xs = np.arange(len(call_types))
    for k, (gname, locs) in enumerate(groups):
        ratios = []
        for ct in call_types:
            rate = loc_rate(df[df["assigned_location"].isin(locs) & (df["event_type"] == ct)])
            mean, _ = _mean_hour_profile(rate)
            day_r = np.nanmean(mean[is_day]); night_r = np.nanmean(mean[~is_day])
            ratios.append(np.log2((day_r + eps) / (night_r + eps)))
        axB.bar(xs + (k - 0.5) * width, ratios, width, color=LOCATION_COLORS.get(gname),
                label=gname, zorder=3)
    axB.axhline(0, color="0.3", lw=0.8, zorder=2)
    axB.set_xticks(xs); axB.set_xticklabels(call_types)
    axB.set_ylabel("log₂(day / night rate)")
    axB.set_title("B. Day/night modulation per call type — >0 diurnal, ~0 no gating", fontsize=10, loc="left")
    axB.legend(fontsize=9, framealpha=0.9)
    axB.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"Circadian gating: surface vs burrow — {date_folder}  "
        f"(over {n_days} days; lights {light_start:02d}:00–{light_end:02d}:00)",
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
    call_types = present + sorted(set(df["event_type"]) - set(CALL_TYPE_ORDER))
    light_start, light_end = light_cycle or get_light_cycle_for_month(date_folder)
    if (24 * 60) % bin_minutes or (ROW_START_HOUR * 60) % bin_minutes:
        raise SystemExit("--bin-min must divide 1440 and ROW_START_HOUR*60 evenly.")
    x0 = ROW_START_HOUR
    day0, days = day_axis(df, x0)
    n_hourbins = (24 * 60) // bin_minutes
    covered_min = coverage_grid(date_folder, day0, len(days), n_hourbins, bin_minutes, x0)
    out_path = out_dir / date_folder / f"circadian_gating_{date_folder}_{bin_minutes}min.png"
    plot_gating(df, date_folder, day0, days, covered_min, call_types,
                bin_minutes, light_start, light_end, x0, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    ap.add_argument("--light-cycle", type=int, nargs=2, metavar=("ON", "OFF"), default=None)
    args = ap.parse_args()
    light_cycle = tuple(args.light_cycle) if args.light_cycle else None
    for date_folder in args.dates:
        run_for_date(date_folder, args.bin_min, args.out_dir, light_cycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
