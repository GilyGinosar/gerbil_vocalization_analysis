"""Variance-scaling of call timing — are call correlations scale-free?

Motivated by Bialek & Shaevitz, PRL 132, 048401 (2024): instead of an
autocorrelation (which forces a per-lag mean subtraction that distorts long
timescales), characterise the process by how the variance of windowed counts
scales with window size T. We treat all calls in a date folder as ONE continuous
point process in absolute time (NOT folded into days, NOT pre-binned into an
ethogram grid), so structure is captured from the inter-call-interval scale up to
the circadian scale in a single curve.

For a counting window of width T we compute, per location:

  Fano factor   F(T) = Var[N(T)] / Mean[N(T)]
  Allan factor  A(T) = < (N_{i+1} - N_i)^2 > / (2 Mean[N(T)])     (adjacent windows)

Reading the curves (log-log):
  * Poisson (no correlation)        -> F = A = 1, flat at all T.
  * Clustered / long-range / fractal-> power-law rise  ~ T^alpha. A straight line
    over decades = scale-free (no characteristic timescale, cf. the paper).
  * The Allan factor DIFFERENCES adjacent windows, so a smooth slow trend (the
    circadian swing) is largely cancelled. So:
        Fano rises but Allan flattens at large T  -> the long-T rise is the
            smooth circadian trend, not genuine clustering.
        BOTH rise together                        -> genuine multi-scale clustering.

Recording is gappy (experiment restarts). We tile only *within* each recorded
segment (from file_times.csv): windows never straddle a gap, and adjacent-window
differences for the Allan factor are taken only within a segment.

Usage:
    python scripts/analysis/run_call_variance_scaling.py --dates 2026_02
    python scripts/analysis/run_call_variance_scaling.py --dates 2026_02 --tmin-s 1 --tmax-h 24
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

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_ethogram import (  # noqa: E402
    BASE_PROCESSED, LOCATION_GROUPS, list_experiment_dirs, load_all_calls,
)

TMIN_S = 1.0                     # smallest window (seconds)
TMAX_H = 24.0                    # largest window (hours)
N_SCALES = 40                    # log-spaced window sizes
MIN_WINDOWS = 30                 # need at least this many windows for a stable variance
GAP_TOL_S = 1.0                  # merge recorded chunks closer than this into one segment
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}
NS_PER_S = 1_000_000_000


def recorded_segments(date_folder: str) -> np.ndarray:
    """Merged [start_ns, end_ns] recorded intervals across all experiments.

    Each file_times.csv row is one contiguous recorded chunk; chunks abut within
    an experiment and gap between restarts. We merge chunks separated by < GAP_TOL
    so each returned segment is a single uninterrupted stretch of recording.
    """
    rows = []
    for exp_dir in list_experiment_dirs(date_folder):
        ft_path = exp_dir / "file_times.csv"
        if not ft_path.exists():
            continue
        ft = pd.read_csv(ft_path)
        s = pd.to_datetime(ft["start_date"] + " " + ft["start_time"]).values.astype("int64")
        e = pd.to_datetime(ft["end_date"] + " " + ft["end_time"]).values.astype("int64")
        rows.append(np.column_stack([s, e]))
    if not rows:
        raise FileNotFoundError(f"No file_times.csv found under {date_folder}")
    iv = np.concatenate(rows)
    iv = iv[np.argsort(iv[:, 0])]
    tol = int(GAP_TOL_S * NS_PER_S)
    merged = [iv[0].copy()]
    for s, e in iv[1:]:
        if s <= merged[-1][1] + tol:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append(np.array([s, e]))
    seg = np.array(merged)
    total_h = (seg[:, 1] - seg[:, 0]).sum() / NS_PER_S / 3600.0
    print(f"{date_folder}: {len(seg)} recorded segments, {total_h:.1f} h total")
    return seg


def window_counts(call_ns: np.ndarray, segments: np.ndarray, T_ns: int) -> np.ndarray:
    """Counts of calls in consecutive width-T windows tiled within each segment.

    Returns a list-of-arrays-per-segment flattened, plus a segment boundary mask is
    not needed because the Allan factor is computed per segment separately.
    """
    per_seg = []
    for s, e in segments:
        k = int((e - s) // T_ns)                 # number of whole windows in this segment
        if k < 1:
            continue
        edges = s + np.arange(k + 1, dtype=np.int64) * T_ns
        idx = np.searchsorted(call_ns, edges)
        per_seg.append(np.diff(idx))
    return per_seg


def fano_allan(per_seg: list[np.ndarray]) -> tuple[float, float, float, int]:
    """Fano factor, Allan factor, mean count, total #windows from per-segment counts."""
    allc = np.concatenate(per_seg) if per_seg else np.array([])
    n = allc.size
    mean = allc.mean() if n else np.nan
    fano = allc.var() / mean if (n and mean > 0) else np.nan
    # Allan factor: squared successive differences, within-segment only.
    diffs = [np.diff(c) for c in per_seg if c.size >= 2]
    allan = np.nan
    if diffs and mean > 0:
        d = np.concatenate(diffs)
        allan = (d**2).mean() / (2.0 * mean)
    return fano, allan, mean, n


def compute_curves(call_ns, segments, scales_ns):
    F, A, M, N = [], [], [], []
    for T in scales_ns:
        f, a, m, n = fano_allan(window_counts(call_ns, segments, int(T)))
        F.append(f); A.append(a); M.append(m); N.append(n)
    return map(np.array, (F, A, M, N))


def _fit_slope(T_s, y, lo_s, hi_s):
    """Power-law slope of y vs T over [lo_s, hi_s], using points clearly > 1."""
    m = (T_s >= lo_s) & (T_s <= hi_s) & np.isfinite(y) & (y > 1.2)
    if m.sum() < 3:
        return None
    coef = np.polyfit(np.log10(T_s[m]), np.log10(y[m]), 1)
    return float(coef[0])


def plot_variance_scaling(date_folder, scales_s, curves_by_loc, out_path):
    fig, (axF, axA) = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, title, key in (
        (axF, "Fano factor  Var[N]/Mean[N]  (sees circadian trend)", "F"),
        (axA, "Allan factor  (differences adjacent windows — trend-robust)", "A"),
    ):
        ax.axhline(1.0, color="0.4", lw=1.0, ls="--", zorder=1, label="Poisson (no corr.)")
        for gname, c in curves_by_loc.items():
            y = c[key]
            valid = c["N"] >= MIN_WINDOWS
            color = LOCATION_COLORS.get(gname)
            ax.plot(scales_s[valid], y[valid], "o-", ms=3.5, lw=1.4, color=color, label=gname)
            slope = _fit_slope(scales_s[valid], y[valid], 60.0, 3600.0)
            if slope is not None:
                print(f"  {gname} {key}: slope (1min-1h) = {slope:.2f}")
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("counting window T (s)")
        ax.set_title(title, fontsize=10, loc="left")
        for t, lab in ((60, "1 min"), (3600, "1 h"), (86400, "24 h")):
            if scales_s.min() <= t <= scales_s.max():
                ax.axvline(t, color="0.85", lw=0.8, zorder=0)
                ax.text(t, ax.get_ylim()[0], f" {lab}", fontsize=7, color="0.5",
                        va="bottom", ha="left", rotation=90)
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)
    axF.set_ylabel("dispersion (1 = Poisson)")
    fig.suptitle(
        f"Call-timing variance scaling — {date_folder}  "
        f"(one continuous point process; window {scales_s.min():.0f} s – {scales_s.max()/3600:.0f} h)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, tmin_s, tmax_h, n_scales, out_dir):
    df = load_all_calls(date_folder)
    segments = recorded_segments(date_folder)
    scales_s = np.unique(np.round(
        np.logspace(np.log10(tmin_s), np.log10(tmax_h * 3600.0), n_scales)))
    scales_ns = scales_s * NS_PER_S

    print(f"{date_folder}: variance scaling over {len(scales_s)} window sizes")
    curves_by_loc = {}
    for gname, locs in LOCATION_GROUPS.items():
        sub = df[df["assigned_location"].isin(locs)]
        call_ns = np.sort(sub["start_time_real"].values.astype("int64"))
        F, A, M, N = compute_curves(call_ns, segments, scales_ns)
        curves_by_loc[gname] = {"F": F, "A": A, "M": M, "N": N}

    out_path = out_dir / date_folder / f"call_variance_scaling_{date_folder}.png"
    plot_variance_scaling(date_folder, scales_s, curves_by_loc, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--tmin-s", type=float, default=TMIN_S)
    ap.add_argument("--tmax-h", type=float, default=TMAX_H)
    ap.add_argument("--n-scales", type=int, default=N_SCALES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.tmin_s, args.tmax_h, args.n_scales, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
