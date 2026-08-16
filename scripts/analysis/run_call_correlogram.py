"""Autocorrelogram of call timing — bursting structure from the timestamps.

Complements run_call_variance_scaling.py: instead of binning, this histograms the
*actual* time lags between pairs of calls, so it resolves correlations from the
inter-call-interval scale upward. All calls in a date folder are treated as one
continuous point process; pairs are only counted *within* a recorded segment
(from file_times.csv), so gaps between experiment restarts never create spurious
lags.

The raw correlogram is rate-dependent, so we normalise to "fold over chance":

    ratio(tau) = observed pairs at lag tau / expected pairs under a Poisson
                 process of the same per-segment rate

with a triangular edge correction (1 - tau/L) for the finite segment length L.
So ratio = 1 means no correlation (Poisson), > 1 means calls cluster at that lag.

Two panels:
  A. linear, short lags (default 0-60 s) — the near-zero bursting peak and how
     fast it falls toward chance.
  B. log-log, full lag range — a straight line = scale-free decay (the same
     fractal clustering the variance-scaling curve shows, seen as a correlogram).

Usage:
    python scripts/analysis/run_call_correlogram.py --dates 2026_02
    python scripts/analysis/run_call_correlogram.py --dates 2026_02 --max-lag-s 1800 --fine-lag-s 120
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
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis",
          REPO_ROOT / "scripts" / "analysis" / "exploratory"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import (  # noqa: E402
    BASE_PROCESSED, LOCATION_GROUPS, list_experiment_dirs, load_all_calls,
    make_bin_edges, recording_coverage_minutes,
)

# --- point-process helpers (vendored from the exploratory run_call_variance_scaling) ---
MIN_WINDOWS = 30                 # need >= this many windows for a stable variance
GAP_TOL_S = 1.0                  # merge recorded chunks closer than this into one segment
NS_PER_S = 1_000_000_000


def recorded_segments(date_folder: str) -> np.ndarray:
    """Merged [start_ns, end_ns] recorded intervals across all experiments."""
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


def window_counts(call_ns: np.ndarray, segments: np.ndarray, T_ns: int) -> list[np.ndarray]:
    """Counts of calls in consecutive width-T windows tiled within each segment."""
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
from run_call_autocorrelation import gappy_acf, location_rate_series  # noqa: E402

MAX_LAG_S = 600.0                # longest lag in the correlogram
FINE_LAG_S = 60.0               # span of the linear short-lag panel
FINE_BIN_S = 0.5                # bin width of the linear panel
N_LOG_BINS = 50                 # log-spaced bins for the log-log panel
LOG_LAG_MIN_S = 0.5             # smallest lag shown on the log panel
AC_BIN_MIN = 5                  # rate-bin width for the autocorrelation panels
AC_MAX_LAG_H = 48.0            # full autocorrelation lag range
AC_ZOOM_H = 2.0               # zoom autocorrelation lag range
AC_MIN_FRAC = 0.5             # min recorded fraction of a bin to use it
LOCATION_COLORS = {"arena": "#1f77b4", "underground": "#d62728"}
NS_PER_S = 1_000_000_000


def correlogram(call_ns: np.ndarray, segments: np.ndarray, edges_s: np.ndarray):
    """Observed / expected pair counts per lag bin, pooled over recorded segments.

    Within each segment, for sorted call times we add successive shifts
    t[k:]-t[:-k] until the smallest such lag exceeds the max lag (the min lag at
    shift k is non-decreasing in k, so this terminates correctly).
    """
    nb = len(edges_s) - 1
    obs = np.zeros(nb, dtype=np.float64)
    exp = np.zeros(nb, dtype=np.float64)
    max_lag = edges_s[-1]
    centers = 0.5 * (edges_s[:-1] + edges_s[1:])
    widths = np.diff(edges_s)

    for s, e in segments:
        i0 = np.searchsorted(call_ns, s, "left")
        i1 = np.searchsorted(call_ns, e, "right")
        t = (call_ns[i0:i1] - s) / NS_PER_S          # seconds from segment start
        n = t.size
        L = (e - s) / NS_PER_S
        if n < 2 or L <= 0:
            continue
        k = 1
        while k < n:
            d = t[k:] - t[:-k]
            if d.min() >= max_lag:
                break
            sel = d[d < max_lag]
            if sel.size:
                obs += np.histogram(sel, bins=edges_s)[0]
            k += 1
        # expected ordered-pair count per bin under uniform-rate Poisson on [0, L]
        exp += (n * (n - 1) / L) * widths * np.clip(1.0 - centers / L, 0.0, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(exp > 0, obs / exp, np.nan)
    return obs, exp, ratio


def _time_markers(ax, lo, hi):
    for t, lab in ((60, "1 min"), (3600, "1 h"), (86400, "24 h")):
        if lo <= t <= hi:
            ax.axvline(t, color="0.85", lw=0.8, zorder=0)
            ax.text(t, ax.get_ylim()[0], f" {lab}", fontsize=7, color="0.5",
                    va="bottom", ha="left", rotation=90)


def _draw_ac(ax, ac_lags_h, ac_by_loc, span, title, zoom):
    ax.axhline(0, color="0.5", lw=0.8)
    for gname, acf in ac_by_loc.items():
        ax.plot(ac_lags_h, acf, lw=1.5, color=LOCATION_COLORS.get(gname), label=gname)
    for h in range(12, int(span) + 1, 12):
        ax.axvline(h, color="0.85", lw=0.8, ls="--", zorder=0)
    if zoom:
        step = 0.5 if span <= 3 else 1
        ax.set_xticks(np.arange(0, span + 1e-9, step))
    ax.set_xlim(0, span)
    ax.set_xlabel("lag (hours)"); ax.set_ylabel("autocorrelation")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(fontsize=9, framealpha=0.9)


def plot_correlogram(date_folder, fine_edges, log_centers, fine_by_loc, log_by_loc,
                     baseline_by_loc, scales_s, var_by_loc, ac_lags_h, ac_by_loc,
                     ac_zoom_h, ac_bin_min, fine_lag_s, out_path):
    """One consolidated multiscale figure, rows = method, cols = two views.

    Row 1 Correlogram   A. raw conditional rate, short lags (bursting, absolute)
                        B. fold over chance, log-log (scale-free decay)
    Row 2 Autocorrelation (whole recording, binned rate)
                        C. full range 0-48 h (24 h circadian peak)
                        D. zoom 0-8 h (ultradian band)
    Row 3 Variance scaling
                        E. Fano factor to 24 h (sees circadian trend)
                        F. Allan factor to 24 h (trend-robust)

    fine_by_loc / log_by_loc: loc -> (ratio, raw_intensity_per_min).
    var_by_loc:               loc -> {"F","A","N"} (Fano, Allan, #windows).
    ac_by_loc:                loc -> autocorrelation array aligned to ac_lags_h.
    """
    fine_centers = 0.5 * (fine_edges[:-1] + fine_edges[1:])
    fig, ((axA, axB), (axC, axD), (axE, axF)) = plt.subplots(3, 2, figsize=(13, 14))

    # A. raw conditional intensity, short lags (absolute calls/min)
    for gname, (_, raw) in fine_by_loc.items():
        color = LOCATION_COLORS.get(gname)
        base = baseline_by_loc[gname]
        axA.plot(fine_centers, raw, lw=1.6, color=color, label=f"{gname} (baseline {base:.1f}/min)")
        axA.axhline(base, color=color, lw=0.9, ls=":", zorder=1)
    axA.set_xlim(0, fine_lag_s); axA.set_ylim(bottom=0)
    axA.set_xlabel("lag (s)"); axA.set_ylabel("conditional rate (calls/min)")
    axA.set_title(f"A. Correlogram, raw (0-{fine_lag_s:.0f} s) — bursting; dotted = baseline",
                  fontsize=10, loc="left")
    axA.legend(fontsize=9, framealpha=0.9)

    # B. fold over chance, log-log (scale-free decay)
    axB.axhline(1.0, color="0.4", lw=1.0, ls="--", label="Poisson (chance)")
    for gname, (ratio, _) in log_by_loc.items():
        axB.plot(log_centers, ratio, "o-", ms=3, lw=1.3, color=LOCATION_COLORS.get(gname), label=gname)
    axB.set_xscale("log"); axB.set_yscale("log")
    axB.set_xlabel("lag (s)"); axB.set_ylabel("fold over chance")
    axB.set_title("B. Correlogram, fold over chance (log-log) — straight = scale-free",
                  fontsize=10, loc="left")
    _time_markers(axB, log_centers.min(), log_centers.max())
    axB.legend(fontsize=9, framealpha=0.9)

    # C/D. autocorrelation of the binned rate (whole recording)
    _draw_ac(axC, ac_lags_h, ac_by_loc, ac_lags_h[-1],
             f"C. Autocorrelation of rate ({ac_bin_min}-min bins, whole recording) — 24 h circadian peak",
             zoom=False)
    _draw_ac(axD, ac_lags_h, ac_by_loc, ac_zoom_h,
             f"D. Autocorrelation — zoom 0-{ac_zoom_h:g} h ({ac_bin_min}-min bins)", zoom=True)

    # E. Fano factor & F. Allan factor, log-log to 24 h
    for ax, key, title in (
        (axE, "F", "E. Fano factor (var/mean window counts) — sees circadian trend"),
        (axF, "A", "F. Allan factor (trend-robust) — arena rise past 1 h = circadian"),
    ):
        ax.axhline(1.0, color="0.4", lw=1.0, ls="--", label="Poisson (no corr.)")
        for gname, v in var_by_loc.items():
            valid = v["N"] >= MIN_WINDOWS
            ax.plot(scales_s[valid], v[key][valid], "o-", ms=3.5, lw=1.4,
                    color=LOCATION_COLORS.get(gname), label=gname)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("counting window T (s)"); ax.set_ylabel("dispersion (1 = Poisson)")
        ax.set_title(title, fontsize=10, loc="left")
        _time_markers(ax, scales_s.min(), scales_s.max())
        ax.legend(fontsize=9, loc="upper left", framealpha=0.9)

    fig.suptitle(
        f"Call-timing structure — {date_folder}  "
        f"(one continuous point process; correlogram + autocorrelation + variance scaling)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, max_lag_s, fine_lag_s, fine_bin_s, n_log_bins,
                 var_tmax_h, var_n_scales, out_dir):
    df = load_all_calls(date_folder)
    segments = recorded_segments(date_folder)
    total_min = (segments[:, 1] - segments[:, 0]).sum() / NS_PER_S / 60.0
    fine_edges = np.arange(0.0, fine_lag_s + fine_bin_s, fine_bin_s)
    log_edges = np.logspace(np.log10(LOG_LAG_MIN_S), np.log10(max_lag_s), n_log_bins + 1)
    log_centers = np.sqrt(log_edges[:-1] * log_edges[1:])
    # window sizes for Fano/Allan: 1 s up to var_tmax_h (default 24 h)
    scales_s = np.unique(np.round(np.logspace(0.0, np.log10(var_tmax_h * 3600.0), var_n_scales)))
    scales_ns = scales_s * NS_PER_S

    print(f"{date_folder}: correlogram (max lag {max_lag_s:.0f} s) + variance scaling (to {var_tmax_h:.0f} h)")
    fine_by_loc, log_by_loc, baseline_by_loc, var_by_loc = {}, {}, {}, {}
    for gname, locs in LOCATION_GROUPS.items():
        sub = df[df["assigned_location"].isin(locs)]
        call_ns = np.sort(sub["start_time_real"].values.astype("int64"))
        baseline = len(call_ns) / total_min                  # calls/min, pooled mean rate
        baseline_by_loc[gname] = baseline
        _, _, fine_ratio = correlogram(call_ns, segments, fine_edges)
        _, _, log_ratio = correlogram(call_ns, segments, log_edges)
        # raw conditional intensity = fold-over-chance x baseline (same edge correction)
        fine_by_loc[gname] = (fine_ratio, fine_ratio * baseline)
        log_by_loc[gname] = (log_ratio, log_ratio * baseline)
        F, A, _, N = compute_curves(call_ns, segments, scales_ns)
        var_by_loc[gname] = {"F": F, "A": A, "N": N}
        print(f"  {gname}: {len(call_ns):,} calls, baseline {baseline:.2f}/min, "
              f"near-zero peak = {np.nanmax(fine_ratio):.1f}x chance "
              f"({np.nanmax(fine_ratio) * baseline:.1f}/min)")

    # autocorrelation of the binned rate over the whole recording
    ac_bin_edges = make_bin_edges(df, AC_BIN_MIN)
    ac_covered = recording_coverage_minutes(date_folder, ac_bin_edges)
    ac_max_lag = int(round(AC_MAX_LAG_H * 60 / AC_BIN_MIN))
    ac_lags_h = np.arange(ac_max_lag + 1) * AC_BIN_MIN / 60.0
    ac_by_loc = {}
    for gname, locs in LOCATION_GROUPS.items():
        rate = location_rate_series(df, ac_bin_edges, ac_covered, locs, AC_BIN_MIN, AC_MIN_FRAC)
        ac_by_loc[gname] = gappy_acf(rate, ac_max_lag)

    out_path = out_dir / date_folder / f"call_timing_structure_{date_folder}.png"
    plot_correlogram(date_folder, fine_edges, log_centers, fine_by_loc, log_by_loc,
                     baseline_by_loc, scales_s, var_by_loc, ac_lags_h, ac_by_loc,
                     AC_ZOOM_H, AC_BIN_MIN, fine_lag_s, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--max-lag-s", type=float, default=MAX_LAG_S)
    ap.add_argument("--fine-lag-s", type=float, default=FINE_LAG_S)
    ap.add_argument("--fine-bin-s", type=float, default=FINE_BIN_S)
    ap.add_argument("--n-log-bins", type=int, default=N_LOG_BINS)
    ap.add_argument("--var-tmax-h", type=float, default=24.0, help="max window for Fano/Allan")
    ap.add_argument("--var-n-scales", type=int, default=40)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.max_lag_s, args.fine_lag_s, args.fine_bin_s,
                     args.n_log_bins, args.var_tmax_h, args.var_n_scales, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
