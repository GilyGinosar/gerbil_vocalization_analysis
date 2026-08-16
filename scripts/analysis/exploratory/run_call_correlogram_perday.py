"""Per-day autocorrelogram (binless) — the per-day ACF without rate binning.

Same question and layout as run_call_acf_perday.py, but instead of binning calls
into a 10-min rate series and correlating, this works from the *actual call
timestamps*: within each circadian day it histograms exact pairwise lags at fine
(1-min) resolution and smooths with a Gaussian kernel of a chosen bandwidth, i.e.
a kernel-density estimate of the autocorrelogram (no rate bins). It is normalised
to fold-over-chance (Poisson = 1) with the same per-segment edge correction as the
pooled correlogram, and pairs are only formed within a recorded segment so gaps
are never crossed.

  - top:    day x lag heatmap of (fold over chance - 1); a vertical stripe at some
            lag, recurring down the rows = a rhythm at that period.
  - bottom: mean fold-over-chance across days; peaks at 2/4/6 h = a 2 h rhythm.

Usage:
    python scripts/analysis/run_call_correlogram_perday.py --dates 2026_02 --location arena
    python scripts/analysis/run_call_correlogram_perday.py --dates 2026_02 --location arena --bandwidth-min 6
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
from scipy.ndimage import gaussian_filter1d

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_ethogram import (  # noqa: E402
    BASE_PROCESSED, LOCATION_GROUPS, ROW_START_HOUR, day_axis, load_all_calls,
)
from run_call_variance_scaling import recorded_segments  # noqa: E402
from run_call_correlogram import correlogram  # noqa: E402

MAX_LAG_H = 8.0
GRID_S = 60.0                    # evaluation grid (<< bandwidth, so effectively binless)
BANDWIDTH_MIN = 12.0            # Gaussian kernel bandwidth
MIN_CALLS_DAY = 200            # skip a day/segment-set with fewer calls than this
MIN_EXP_PAIRS = 5.0            # mask lags with fewer expected pairs (kills noisy tail)
NS_PER_S = 1_000_000_000


def clip_segments(segments: np.ndarray, w0: int, w1: int) -> np.ndarray:
    s = np.maximum(segments[:, 0], w0)
    e = np.minimum(segments[:, 1], w1)
    keep = e > s
    return np.column_stack([s[keep], e[keep]])


def plot_perday_correlogram(date_folder, days, ratio_rows, lags_h, sel, bandwidth_min, n_days_used, out_path):
    fig, (axH, axM) = plt.subplots(2, 1, figsize=(10, 9),
                                   gridspec_kw={"height_ratios": [2.6, 1]})
    excess = ratio_rows - 1.0
    far = lags_h >= 1.0                          # ignore the central bursting peak for the scale
    vmax = np.nanpercentile(np.abs(excess[:, far]), 98) if np.isfinite(excess[:, far]).any() else 1.0
    vmax = max(vmax, 0.05)

    im = axH.imshow(excess, aspect="auto", origin="upper", cmap="RdBu_r",
                    vmin=-vmax, vmax=vmax,
                    extent=[lags_h[0], lags_h[-1], len(days) - 0.5, -0.5])
    axH.set_yticks(np.arange(len(days)))
    axH.set_yticklabels([d.strftime("%m-%d") for d in days], fontsize=7)
    axH.set_xlabel("lag (hours)"); axH.set_ylabel("circadian day")
    axH.set_title(f"A. Per-day autocorrelogram ({sel}, binless) — vertical stripe = recurring rhythm",
                  fontsize=10, loc="left")
    for h in range(2, int(lags_h[-1]) + 1, 2):
        axH.axvline(h, color="0.3", lw=0.6, ls=":", alpha=0.5)
    fig.colorbar(im, ax=axH, label="fold over chance − 1", fraction=0.025, pad=0.01)

    n = np.sum(~np.isnan(ratio_rows), axis=0)
    mean = np.nanmean(ratio_rows, axis=0)
    sem = np.nanstd(ratio_rows, axis=0) / np.sqrt(np.maximum(n, 1))
    axM.axhline(1.0, color="0.5", lw=0.8, ls="--", label="Poisson (chance)")
    axM.plot(lags_h, mean, color="#333333", lw=1.8, label=f"mean across {n_days_used} days")
    axM.fill_between(lags_h, mean - sem, mean + sem, color="0.6", alpha=0.4, lw=0)
    for h in range(2, int(lags_h[-1]) + 1, 2):
        axM.axvline(h, color="0.3", lw=0.6, ls=":", alpha=0.5)
    axM.set_xlim(lags_h[0], lags_h[-1])
    axM.set_xlabel("lag (hours)"); axM.set_ylabel("fold over chance")
    axM.set_title("B. Mean across days — peaks at 2/4/6 h = a consistent ~2 h rhythm",
                  fontsize=10, loc="left")
    axM.legend(fontsize=9)

    try:
        from scipy.signal import find_peaks
        idx, _ = find_peaks(np.nan_to_num(mean, nan=-1.0))
    except Exception:
        idx = 1 + np.where((mean[1:-1] > mean[:-2]) & (mean[1:-1] > mean[2:]))[0]
    idx = [i for i in idx if lags_h[i] >= 1.0 and np.isfinite(mean[i])]
    if idx:
        top = sorted(idx, key=lambda i: mean[i], reverse=True)[:3]
        peaks = ", ".join(f"{lags_h[i]:.1f}h (x{mean[i]:.2f})" for i in sorted(top, key=lambda i: lags_h[i]))
        print(f"  mean-correlogram peaks > 1 h: {peaks}")
    else:
        print("  mean-correlogram: no peak > 1 h")

    fig.suptitle(
        f"Per-day call autocorrelogram (binless) — {date_folder} ({sel})  "
        f"(kernel bandwidth {bandwidth_min:.0f} min; day starts {ROW_START_HOUR:02d}:00)",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, location, call_type, max_lag_h, grid_s, bandwidth_min, out_dir):
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

    segments = recorded_segments(date_folder)
    call_ns = np.sort(df["start_time_real"].values.astype("int64"))
    x0 = ROW_START_HOUR
    day0, days = day_axis(df, x0)
    n_days = len(days)

    fine_edges = np.arange(0.0, max_lag_h * 3600.0 + grid_s, grid_s)
    lags_h = (0.5 * (fine_edges[:-1] + fine_edges[1:])) / 3600.0
    sigma_bins = (bandwidth_min * 60.0) / grid_s

    ratio_rows = np.full((n_days, len(lags_h)), np.nan)
    used = 0
    for d in range(n_days):
        w0 = (day0 + pd.Timedelta(days=d, hours=x0)).value
        w1 = w0 + 24 * 3600 * NS_PER_S
        day_segs = clip_segments(segments, w0, w1)
        if len(day_segs) == 0:
            continue
        i0 = np.searchsorted(call_ns, w0, "left")
        i1 = np.searchsorted(call_ns, w1, "right")
        if i1 - i0 < MIN_CALLS_DAY:
            continue
        obs, exp, _ = correlogram(call_ns, day_segs, fine_edges)
        obs_s = gaussian_filter1d(obs, sigma_bins, mode="nearest")
        exp_s = gaussian_filter1d(exp, sigma_bins, mode="nearest")
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(exp_s > MIN_EXP_PAIRS, obs_s / exp_s, np.nan)
        ratio_rows[d] = r
        used += 1
    print(f"{date_folder}: {used}/{n_days} days with >= {MIN_CALLS_DAY} calls")

    slug = "_".join(label_bits) if label_bits else "allcalls"
    out_path = out_dir / date_folder / f"call_correlogram_perday_{date_folder}_{slug}.png"
    plot_perday_correlogram(date_folder, list(days), ratio_rows, lags_h, sel, bandwidth_min, used, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--location", default="arena", help="all | arena | underground")
    ap.add_argument("--call-type", default="all", help="all | alarm | high-freq | warble | stacks | newborn")
    ap.add_argument("--max-lag-h", type=float, default=MAX_LAG_H)
    ap.add_argument("--grid-s", type=float, default=GRID_S)
    ap.add_argument("--bandwidth-min", type=float, default=BANDWIDTH_MIN)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.location, args.call_type, args.max_lag_h,
                     args.grid_s, args.bandwidth_min, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
