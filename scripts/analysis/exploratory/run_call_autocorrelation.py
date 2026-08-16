"""Autocorrelation of call activity to probe rhythmicity (circadian / ultradian).

All call types are pooled into a single "any call" tick rate, computed separately
for arena and underground. The rate is binned on a regular grid; because
recording is gappy (experiment restarts), the autocorrelation at each lag is a
*pairwise-complete* Pearson correlation between the series and its lagged copy
(bins where either side has no recording are skipped) — so gaps neither fabricate
nor destroy structure.

Two panels:
  - full lag range (default 0-48 h) — look for a ~24 h circadian peak
  - zoom 0-8 h — look for ultradian peaks (e.g. the ~2 h arena oscillation)

Reuses the regular-binning + coverage helpers from run_ethogram.

Usage:
    python scripts/analysis/run_call_autocorrelation.py --dates 2026_02
    python scripts/analysis/run_call_autocorrelation.py --dates 2026_02 --bin-min 5 --max-lag-h 72
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
    BASE_PROCESSED, LOCATION_GROUPS, load_all_calls, make_bin_edges, recording_coverage_minutes,
)

BIN_MINUTES = 10
MAX_LAG_H = 48
MIN_COVERAGE_FRAC = 0.5          # require >= this fraction of a bin recorded to use it
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}


def location_rate_series(df, bin_edges, covered_min, locs, bin_minutes, min_frac):
    """Pooled calls/min per bin for one location; NaN where under-recorded."""
    sub = df[df["assigned_location"].isin(locs)]
    counts, _ = np.histogram(sub["start_time_real"].values.astype("int64"),
                             bins=bin_edges.values.astype("int64"))
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = counts / covered_min
    rate[covered_min < min_frac * bin_minutes] = np.nan
    return rate


def gappy_acf(x, max_lag):
    """Pairwise-complete autocorrelation (Pearson at each lag); acf[0] = 1."""
    n = len(x)
    acf = np.full(max_lag + 1, np.nan)
    for lag in range(max_lag + 1):
        a = x[: n - lag] if lag else x
        b = x[lag:]
        m = ~np.isnan(a) & ~np.isnan(b)
        if m.sum() >= 10:
            aa = a[m] - a[m].mean()
            bb = b[m] - b[m].mean()
            denom = np.sqrt((aa**2).sum() * (bb**2).sum())
            if denom > 0:
                acf[lag] = float((aa * bb).sum() / denom)
    return acf


def _report_peaks(name, lags_h, acf):
    """Print the strongest ACF peak beyond a short refractory lag."""
    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(np.nan_to_num(acf, nan=-1.0))
    except Exception:
        idx = 1 + np.where((acf[1:-1] > acf[:-2]) & (acf[1:-1] > acf[2:]))[0]
    idx = [i for i in idx if lags_h[i] >= 1.0 and np.isfinite(acf[i])]
    if not idx:
        print(f"  {name}: no clear peak > 1 h lag")
        return
    top = sorted(idx, key=lambda i: acf[i], reverse=True)[:4]
    peaks = ", ".join(f"{lags_h[i]:.1f}h (r={acf[i]:.2f})" for i in sorted(top, key=lambda i: lags_h[i]))
    print(f"  {name}: peaks -> {peaks}")


def plot_autocorrelation(df, date_folder, bin_edges, covered_min, bin_minutes,
                         max_lag_h, min_frac, out_path):
    groups = list(LOCATION_GROUPS.items())
    max_lag = int(round(max_lag_h * 60 / bin_minutes))
    lags_h = np.arange(max_lag + 1) * bin_minutes / 60.0

    fig, (ax_full, ax_zoom) = plt.subplots(2, 1, figsize=(10, 7))
    print(f"{date_folder}: autocorrelation ({bin_minutes}-min bins)")
    for gname, locs in groups:
        rate = location_rate_series(df, bin_edges, covered_min, locs, bin_minutes, min_frac)
        acf = gappy_acf(rate, max_lag)
        color = LOCATION_COLORS.get(gname)
        for ax in (ax_full, ax_zoom):
            ax.plot(lags_h, acf, color=color, lw=1.4, label=gname)
        _report_peaks(gname, lags_h, acf)

    for ax, span, title in (
        (ax_full, max_lag_h, f"A. Full range — circadian band"),
        (ax_zoom, min(8, max_lag_h), "B. Zoom 0-8 h — ultradian band"),
    ):
        ax.axhline(0, color="0.5", lw=0.8)
        for h in range(12, int(span) + 1, 12):
            ax.axvline(h, color="0.8", lw=0.8, ls="--", zorder=0)   # 12/24/36/48 h guides
        ax.set_xlim(0, span)
        ax.set_xlabel("lag (hours)")
        ax.set_ylabel("autocorrelation")
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(fontsize=9)
    ax_zoom.set_xticks(np.arange(0, min(8, max_lag_h) + 0.1, 1))

    fig.suptitle(
        f"Call-activity autocorrelation — {date_folder}  "
        f"(all call types pooled per location; {bin_minutes}-min bins, gap-aware)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, bin_minutes, max_lag_h, min_frac, out_dir):
    df = load_all_calls(date_folder)
    bin_edges = make_bin_edges(df, bin_minutes)
    covered_min = recording_coverage_minutes(date_folder, bin_edges)
    out_path = out_dir / date_folder / f"call_autocorrelation_{date_folder}_{bin_minutes}min.png"
    plot_autocorrelation(df, date_folder, bin_edges, covered_min, bin_minutes,
                         max_lag_h, min_frac, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES)
    ap.add_argument("--max-lag-h", type=float, default=MAX_LAG_H)
    ap.add_argument("--min-coverage-frac", type=float, default=MIN_COVERAGE_FRAC)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.bin_min, args.max_lag_h, args.min_coverage_frac, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
