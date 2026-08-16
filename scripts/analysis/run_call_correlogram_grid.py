"""Grid of call-pair correlograms — one column per date, one row per call type.

The sibling script run_call_acf_grid.py autocorrelates the *binned rate*, which
answers "is the calling rate at time t correlated with the rate at t+tau" over
long (circadian) timescales. This script instead asks a point-process question:

    given a call at time t, what is the relative likelihood of *another* call at
    t + tau, for ALL pairs of calls separated by tau (any number of other calls
    may occur in between) — not just consecutive calls.

That is the correlogram from run_call_correlogram.py: within each recorded
segment we histogram every pair lag and divide by the count expected under a
Poisson process of the same rate (triangular edge correction). ratio = 1 means
no correlation (chance); ratio > 1 means calls cluster at that delay.

Because it counts every pair, the correlogram is a short-timescale tool
(seconds-to-minutes bursting), shown log-log; the long-timescale circadian
structure is what run_call_acf_grid.py is for.

Layout:
  columns = date folders
  row 1   = all call types pooled
  rows 2+ = one call type on its own (alarm, high-freq, warble, stacks, newborn)
  each cell: arena (blue) vs underground (red), fold-over-chance vs tau (log-log)

Usage:
    python scripts/analysis/run_call_correlogram_grid.py
    python scripts/analysis/run_call_correlogram_grid.py --dates 2026_02 --max-lag-s 3600
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
)
from run_call_correlogram import (  # noqa: E402
    LOG_LAG_MIN_S, MAX_LAG_S, N_LOG_BINS, NS_PER_S, correlogram, recorded_segments,
)

EXPORTS_DIR = REPO_ROOT / "exports"     # also drop the saved figure here for easy download
DATE_FOLDERS = ["2025_07", "2025_10", "2026_02"]   # one column per date
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}
# Per-call-type row-label colors (from run_ethogram_categorical.py).
CALL_TYPE_COLORS = {
    "alarm": "#e41a1c", "high-freq": "#377eb8", "warble": "#4daf4a",
    "stacks": "#ff7f00", "newborn": "#984ea3", "all": "0.15",
}


# Minimum expected consecutive-interval count for a bin's fold-over-chance to be
# trustworthy. The exponential ICI null decays as exp(-rate*tau), so past a few
# mean intervals it underflows toward zero and obs/exp explodes without bound
# (a Poisson process essentially never waits that long between calls, so the
# ratio is "infinity x chance" — true but vacuous). Bins below this floor are
# masked to NaN so each curve self-truncates where its own null runs out of
# mass, rather than producing spurious rising limbs. Low-rate call types (alarm)
# keep valid support out to longer lags than high-rate ones (stacks).
MIN_EXPECTED_ICI = 10.0


def correlogram_consecutive(call_ns, segments, edges_s):
    """Fold-over-chance for *consecutive* calls only (the inter-call interval).

    Same shape as correlogram() but the observed histogram counts only each
    call's immediate successor (the k=1 pair), and the null is the exponential
    ICI a Poisson process of the same per-segment rate would produce, so the
    curve stays comparable to the all-pairs correlogram (both centred at 1).

    Bins whose summed expected count falls below MIN_EXPECTED_ICI are masked to
    NaN: there the null has essentially no mass and obs/exp is numerically
    meaningless (see the constant's note).
    """
    lo, hi = edges_s[:-1], edges_s[1:]
    obs = np.zeros(len(lo), dtype=np.float64)
    exp = np.zeros(len(lo), dtype=np.float64)
    for s, e in segments:
        i0 = np.searchsorted(call_ns, s, "left")
        i1 = np.searchsorted(call_ns, e, "right")
        t = (call_ns[i0:i1] - s) / NS_PER_S
        n, L = t.size, (e - s) / NS_PER_S
        if n < 2 or L <= 0:
            continue
        ici = np.diff(t)                                 # n-1 consecutive intervals
        sel = ici[ici < edges_s[-1]]
        if sel.size:
            obs += np.histogram(sel, bins=edges_s)[0]
        rate = n / L                                     # Poisson ICI ~ Exp(rate)
        exp += (n - 1) * (np.exp(-rate * lo) - np.exp(-rate * hi))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(exp >= MIN_EXPECTED_ICI, obs / exp, np.nan)
    return obs, exp, ratio


def ratio_by_location(df_sub, segments, log_edges, corr_fn):
    """{location group -> fold-over-chance correlogram} for one call-type subset."""
    out = {}
    for gname, locs in LOCATION_GROUPS.items():
        sub = df_sub[df_sub["assigned_location"].isin(locs)]
        call_ns = np.sort(sub["start_time_real"].values.astype("int64"))
        if call_ns.size < 2:
            out[gname] = np.full(len(log_edges) - 1, np.nan)
            continue
        _, _, ratio = corr_fn(call_ns, segments, log_edges)
        out[gname] = ratio
    return out


def compute_for_date(date_folder, log_edges, row_types, corr_fn):
    """Return ({row -> {loc -> ratio}}, {row -> n_calls})."""
    df = load_all_calls(date_folder)
    segments = recorded_segments(date_folder)
    ratio_by_row, n_by_row = {}, {}
    for row in row_types:
        sub = df if row == "all" else df[df["event_type"] == row]
        ratio_by_row[row] = ratio_by_location(sub, segments, log_edges, corr_fn)
        n_by_row[row] = len(sub)
    print(f"{date_folder}: {len(df):,} calls, "
          + ", ".join(f"{r}={n_by_row[r]:,}" for r in row_types))
    return ratio_by_row, n_by_row


def _time_markers(ax, lo, hi):
    for t, lab in ((1, "1 s"), (60, "1 min"), (3600, "1 h")):
        if lo <= t <= hi:
            ax.axvline(t, color="0.85", lw=0.7, zorder=0)
            ax.text(t, ax.get_ylim()[0], f" {lab}", fontsize=6, color="0.5",
                    va="bottom", ha="left", rotation=90)


def plot_grid(dates, per_date, row_types, log_centers, out_path, title):
    nrows, ncols = len(row_types), len(dates)
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.3 * nrows),
                             sharex=True, sharey=True, squeeze=False)

    gmax, gmin = 1.0, 1.0
    for date in dates:
        for row in row_types:
            for ratio in per_date[date][0][row].values():
                pos = ratio[np.isfinite(ratio) & (ratio > 0)]
                if pos.size:
                    gmax = max(gmax, pos.max())
                    gmin = min(gmin, pos.min())
    ytop = min(gmax * 1.3, 100.0)
    ybot = max(gmin * 0.8, 0.01)

    for j, date in enumerate(dates):
        ratio_by_row, n_by_row = per_date[date]
        for i, row in enumerate(row_types):
            ax = axes[i][j]
            ax.axhline(1.0, color="0.4", lw=0.9, ls="--", zorder=1)
            for gname, ratio in ratio_by_row[row].items():
                ax.plot(log_centers, ratio, "o-", ms=2.5, lw=1.1,
                        color=LOCATION_COLORS[gname], label=gname)
            ax.set_xscale("log"); ax.set_yscale("log")
            ax.set_xlim(log_centers.min(), log_centers.max())
            ax.set_ylim(ybot, ytop)
            _time_markers(ax, log_centers.min(), log_centers.max())
            ax.text(0.03, 0.94, f"n={n_by_row[row]:,}", transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, color="0.5")
            if i == 0:
                ax.set_title(date, fontsize=11, fontweight="bold")
            if j == 0:
                label = "all calls" if row == "all" else row
                ax.set_ylabel(label, fontsize=10, fontweight="bold",
                              color=CALL_TYPE_COLORS.get(row, "0.15"))
            if i == nrows - 1:
                ax.set_xlabel("lag τ (s)")

    axes[0][-1].legend(fontsize=8, framealpha=0.9, loc="upper right")
    fig.suptitle(title, fontsize=13)
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
    ap.add_argument("--max-lag-s", type=float, default=MAX_LAG_S, help="longest pair delay tau")
    ap.add_argument("--n-log-bins", type=int, default=N_LOG_BINS)
    ap.add_argument("--consecutive", action="store_true",
                    help="only each call's immediate successor (inter-call interval) "
                         "vs an exponential null, instead of all pairs")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()

    log_edges = np.logspace(np.log10(LOG_LAG_MIN_S), np.log10(args.max_lag_s), args.n_log_bins + 1)
    log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])

    corr_fn = correlogram_consecutive if args.consecutive else correlogram
    if args.consecutive:
        kind, title = "consecutive", (
            "Consecutive-call correlogram by type (fold over chance, log-log) "
            "— immediate next call at delay τ, arena vs underground")
    else:
        kind, title = "allpairs", (
            "Call-pair correlogram by type (fold over chance, log-log) "
            "— all pairs at delay τ, arena vs underground")

    row_types = ["all"] + CALL_TYPE_ORDER
    per_date = {date: compute_for_date(date, log_edges, row_types, corr_fn) for date in args.dates}

    tag = "_".join(args.dates)
    out_path = args.out_dir / f"call_correlogram_grid_{kind}_{tag}_{args.max_lag_s:.0f}s.png"
    plot_grid(args.dates, per_date, row_types, log_centers, out_path, title)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
