"""Grid of call-rate autocorrelations — one column per date, one row per call type.

Each cell is the same autocorrelation shown in panel C of run_call_correlogram:
the binned per-location call rate (arena vs underground) over the whole recording,
autocorrelated with a pairwise-complete (gap-aware) Pearson correlation at each lag.

Layout:
  columns = date folders (passed via --dates)
  row 1   = all call types pooled ("any call", as panel C does now)
  rows 2+ = one call type on its own (alarm, high-freq, warble, stacks, newborn)

Within every cell, arena (blue) and underground (red) are plotted as two lines,
so you can read circadian / ultradian rhythmicity per call type and compare it
across dates and against the pooled row. Bin edges and recording coverage are
computed once from the full recording per date, then the rate series is rebuilt
per call type on that same grid, so the lag axis is identical across all rows.

Usage:
    python scripts/analysis/run_call_acf_grid.py --dates 2026_02
    python scripts/analysis/run_call_acf_grid.py --dates 2025_10 2026_02 --bin-min 5 --max-lag-h 48
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis",
          REPO_ROOT / "scripts" / "analysis" / "exploratory"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import (  # noqa: E402
    BASE_PROCESSED, CALL_TYPE_ORDER, LOCATION_GROUPS, load_all_calls,
    make_bin_edges, recording_coverage_minutes,
)
from run_call_autocorrelation import gappy_acf, location_rate_series  # noqa: E402

EXPORTS_DIR = REPO_ROOT / "exports"     # also drop the saved figure here for easy download
DATE_FOLDERS = ["2025_07", "2025_10", "2026_02"]   # one column per date
BIN_MINUTES = 5                  # rate-bin width (matches panel C default here)
MAX_LAG_H = 48.0                 # full autocorrelation lag range
MIN_FRAC = 0.5                   # min recorded fraction of a bin to use it
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}
# Per-call-type row-label colors (from run_ethogram_categorical.py).
CALL_TYPE_COLORS = {
    "alarm": "#e41a1c", "high-freq": "#377eb8", "warble": "#4daf4a",
    "stacks": "#ff7f00", "newborn": "#984ea3", "all": "0.15",
}


def acf_by_location(df_sub, bin_edges, covered_min, max_lag, bin_minutes, min_frac):
    """{location group -> gap-aware ACF array of length max_lag+1} for one subset.

    Recordings shorter than max_lag (e.g. a truncated date folder) are computed
    only up to their available length and NaN-padded, so every date shares the
    same lag axis.
    """
    out = {}
    for gname, locs in LOCATION_GROUPS.items():
        rate = location_rate_series(df_sub, bin_edges, covered_min, locs, bin_minutes, min_frac)
        eff_lag = min(max_lag, len(rate) - 1)
        acf = np.full(max_lag + 1, np.nan)
        if eff_lag >= 1:
            acf[: eff_lag + 1] = gappy_acf(rate, eff_lag)
        out[gname] = acf
    return out


def compute_for_date(date_folder, bin_minutes, max_lag_h, min_frac, row_types):
    """Return (lags_h, {row_name -> {loc -> acf}}, {row_name -> n_calls})."""
    df = load_all_calls(date_folder)
    bin_edges = make_bin_edges(df, bin_minutes)
    covered = recording_coverage_minutes(date_folder, bin_edges)
    max_lag = int(round(max_lag_h * 60 / bin_minutes))
    lags_h = np.arange(max_lag + 1) * bin_minutes / 60.0

    acf_by_row, n_by_row = {}, {}
    for row in row_types:
        sub = df if row == "all" else df[df["event_type"] == row]
        acf_by_row[row] = acf_by_location(sub, bin_edges, covered, max_lag, bin_minutes, min_frac)
        n_by_row[row] = len(sub)
    print(f"{date_folder}: {len(df):,} calls, "
          + ", ".join(f"{r}={n_by_row[r]:,}" for r in row_types))
    return lags_h, acf_by_row, n_by_row


def plot_grid(dates, per_date, row_types, bin_minutes, max_lag_h, out_path):
    nrows, ncols = len(row_types), len(dates)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    for j, date in enumerate(dates):
        lags_h, acf_by_row, n_by_row = per_date[date]
        for i, row in enumerate(row_types):
            ax = axes[i][j]
            ax.axhline(0, color="0.5", lw=0.7)
            for h in range(12, int(max_lag_h) + 1, 12):
                ax.axvline(h, color="0.88", lw=0.7, ls="--", zorder=0)
            for gname, acf in acf_by_row[row].items():
                ax.plot(lags_h, acf, lw=1.3, color=LOCATION_COLORS[gname], label=gname)
            ax.set_xlim(0, max_lag_h)
            ax.text(0.03, 0.93, f"n={n_by_row[row]:,}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color="0.5")
            if i == 0:
                ax.set_title(date, fontsize=11, fontweight="bold")
            if j == 0:
                label = "all calls" if row == "all" else row
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              color=CALL_TYPE_COLORS.get(row, "0.15"))
            if i == nrows - 1:
                ax.set_xlabel("lag (hours)")

    axes[0][0].set_ylim(-0.4, 1.05)
    axes[0][-1].legend(fontsize=8, framealpha=0.9, loc="upper right")
    fig.suptitle(
        f"Call-rate autocorrelation by type ({bin_minutes}-min bins, 0-{max_lag_h:g} h) "
        f"— arena vs underground, whole recording",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, EXPORTS_DIR / out_path.name)
    print(f"   + exports/{out_path.name}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DATE_FOLDERS, help="one column per date")
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES)
    ap.add_argument("--max-lag-h", type=float, default=MAX_LAG_H)
    ap.add_argument("--min-coverage-frac", type=float, default=MIN_FRAC)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()

    row_types = ["all"] + CALL_TYPE_ORDER
    per_date = {
        date: compute_for_date(date, args.bin_min, args.max_lag_h,
                               args.min_coverage_frac, row_types)
        for date in args.dates
    }
    tag = "_".join(args.dates)
    out_path = args.out_dir / f"call_acf_grid_{tag}_{args.bin_min}min.png"
    plot_grid(args.dates, per_date, row_types, args.bin_min, args.max_lag_h, out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
