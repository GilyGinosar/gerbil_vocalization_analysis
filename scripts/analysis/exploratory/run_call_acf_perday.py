"""Per-day autocorrelation of call activity — does a ~2 h ultradian rhythm recur?

The pooled, weeks-long autocorrelation (run_call_autocorrelation.py) averages over
all days and so is blind to an ultradian rhythm that drifts in phase or is only
present on some days (the Bialek-Shaevitz pooling problem). This script instead
computes the autocorrelation *within each circadian day separately*, for ALL calls
pooled together (every call type, both locations, one tick stream), and stacks the
days so a recurring bump (e.g. ~2 h) shows up as a vertical band.

  - top:    day x lag heatmap of the autocorrelation (a vertical stripe at some lag
            that persists down the rows = a rhythm that recurs day to day)
  - bottom: mean +/- SEM autocorrelation across days

Recording is gappy; each day's rate series is gap-aware (bins with too little
recording are NaN) and the autocorrelation is pairwise-complete.

Usage:
    python scripts/analysis/run_call_acf_perday.py --dates 2026_02
    python scripts/analysis/run_call_acf_perday.py --dates 2026_02 --bin-min 5 --max-lag-h 12
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

from run_ethogram import (  # noqa: E402
    BASE_PROCESSED, LOCATION_GROUPS, ROW_START_HOUR,
    counts_grid, coverage_grid, day_axis, load_all_calls,
)
from run_call_autocorrelation import gappy_acf  # noqa: E402

BIN_MINUTES = 1
MAX_LAG_H = 12.0
MIN_COVERAGE_FRAC = 0.5         # require >= this fraction of a bin recorded to use it
MIN_VALID_BINS = 24            # skip a day with fewer usable bins than this


def plot_perday_acf(date_folder, days, acf_rows, lags_h, valid_day, sel, out_path):
    fig, (axH, axM) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={"height_ratios": [2.6, 1]})
    vmax = np.nanpercentile(np.abs(acf_rows[:, 1:]), 98) if np.isfinite(acf_rows[:, 1:]).any() else 1.0

    im = axH.imshow(acf_rows, aspect="auto", origin="upper", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    extent=[lags_h[0], lags_h[-1], len(days) - 0.5, -0.5])
    axH.set_yticks(np.arange(len(days)))
    axH.set_yticklabels([d.strftime("%m-%d") for d in days], fontsize=7)
    axH.set_xlabel("lag (hours)")
    axH.set_ylabel("circadian day")
    axH.set_title(f"A. Per-day autocorrelation ({sel}) — vertical stripe = recurring rhythm",
                  fontsize=10, loc="left")
    for h in range(2, int(lags_h[-1]) + 1, 2):
        axH.axvline(h, color="0.3", lw=0.6, ls=":", alpha=0.5)
    fig.colorbar(im, ax=axH, label="autocorrelation", fraction=0.025, pad=0.01)

    n = np.sum(~np.isnan(acf_rows), axis=0)
    mean = np.nanmean(acf_rows, axis=0)
    sem = np.nanstd(acf_rows, axis=0) / np.sqrt(np.maximum(n, 1))
    axM.axhline(0, color="0.5", lw=0.8)
    axM.plot(lags_h, mean, color="#333333", lw=1.8, label=f"mean across {int(valid_day.sum())} days")
    axM.fill_between(lags_h, mean - sem, mean + sem, color="0.6", alpha=0.4, lw=0)
    for h in range(2, int(lags_h[-1]) + 1, 2):
        axM.axvline(h, color="0.3", lw=0.6, ls=":", alpha=0.5)
    axM.set_xlim(lags_h[0], lags_h[-1])
    axM.set_xlabel("lag (hours)"); axM.set_ylabel("autocorrelation")
    axM.set_title("B. Mean +/- SEM across days — a bump = a consistent rhythm at that period",
                  fontsize=10, loc="left")
    axM.legend(fontsize=9)

    # report the strongest mean-ACF peak beyond 1 h
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(np.nan_to_num(mean, nan=-1.0))
    except Exception:
        idx = 1 + np.where((mean[1:-1] > mean[:-2]) & (mean[1:-1] > mean[2:]))[0]
    idx = [i for i in idx if lags_h[i] >= 1.0 and np.isfinite(mean[i])]
    if idx:
        top = sorted(idx, key=lambda i: mean[i], reverse=True)[:3]
        peaks = ", ".join(f"{lags_h[i]:.1f}h (r={mean[i]:.2f})" for i in sorted(top, key=lambda i: lags_h[i]))
        print(f"  mean-ACF peaks > 1 h: {peaks}")
    else:
        print("  mean-ACF: no peak > 1 h")

    fig.suptitle(
        f"Per-day call autocorrelation — {date_folder} ({sel})  "
        f"({int((lags_h[1]-lags_h[0])*60)}-min bins; day starts {ROW_START_HOUR:02d}:00)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, bin_minutes, max_lag_h, min_frac, out_dir,
                 location="all", call_type="all"):
    if (24 * 60) % bin_minutes or (ROW_START_HOUR * 60) % bin_minutes:
        raise SystemExit("--bin-min must divide 1440 and ROW_START_HOUR*60 evenly.")
    df = load_all_calls(date_folder)
    label_bits = []
    if location != "all":
        if location not in LOCATION_GROUPS:
            raise SystemExit(f"--location must be one of {list(LOCATION_GROUPS) + ['all']}")
        df = df[df["assigned_location"].isin(LOCATION_GROUPS[location])]
        label_bits.append(location)
    if call_type != "all":
        df = df[df["event_type"] == call_type]
        label_bits.append(call_type)
    sel = " + ".join(label_bits) if label_bits else "all calls"
    print(f"{date_folder}: {len(df):,} calls selected ({sel})")
    x0 = ROW_START_HOUR
    day0, days = day_axis(df, x0)
    n_days = len(days)
    n_hourbins = (24 * 60) // bin_minutes
    counts = counts_grid(df, day0, n_days, n_hourbins, bin_minutes, x0)
    covered = coverage_grid(date_folder, day0, n_days, n_hourbins, bin_minutes, x0)
    enough = covered >= min_frac * bin_minutes
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(enough, counts / covered, np.nan)

    max_lag = int(round(max_lag_h * 60 / bin_minutes))
    lags_h = np.arange(max_lag + 1) * bin_minutes / 60.0
    acf_rows = np.full((n_days, max_lag + 1), np.nan)
    valid_day = np.zeros(n_days, dtype=bool)
    for d in range(n_days):
        if np.sum(~np.isnan(rate[d])) >= MIN_VALID_BINS:
            acf_rows[d] = gappy_acf(rate[d], max_lag)
            valid_day[d] = True
    print(f"{date_folder}: {int(valid_day.sum())}/{n_days} days with enough coverage")

    slug = "_".join(label_bits) if label_bits else "allcalls"
    out_path = out_dir / date_folder / f"call_acf_perday_{date_folder}_{slug}_{bin_minutes}min.png"
    plot_perday_acf(date_folder, list(days), acf_rows, lags_h, valid_day, sel, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES)
    ap.add_argument("--max-lag-h", type=float, default=MAX_LAG_H)
    ap.add_argument("--min-coverage-frac", type=float, default=MIN_COVERAGE_FRAC)
    ap.add_argument("--location", default="all", help="all | arena | underground")
    ap.add_argument("--call-type", default="all", help="all | alarm | high-freq | warble | stacks | newborn")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.bin_min, args.max_lag_h, args.min_coverage_frac,
                     args.out_dir, args.location, args.call_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
