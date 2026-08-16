"""Wavelet scalogram of call rate — find *intermittent* rhythms (e.g. a ~2 h bout).

Autocorrelation assumes stationarity, so a rhythm that drifts in phase or comes and
goes averages away (see run_call_acf_perday / run_call_correlogram_perday: no fixed
2 h period survives). A continuous wavelet transform does not assume stationarity:
it shows power vs period *and* time, so an intermittent band lights up only on the
days/hours it actually occurs.

Method: Morlet CWT (Torrence & Compo 1998), computed by FFT. The call-rate series
is coverage-normalised on a regular grid; short recording gaps are linearly
interpolated for the transform but the gap columns are masked (grey) in the display,
and the cone-of-influence (edge-affected region) is hatched.

Panels:
  - top:   the arena call-rate trace (night shaded)
  - main:  scalogram, time x period, colour = log10 wavelet power; 2/12/24 h marked
  - right: global wavelet spectrum (time-averaged power vs period) = the Fourier-like
           summary; a peak at 2 h would mean an *on-average* ultradian rhythm.

Usage:
    python scripts/analysis/run_call_wavelet.py --dates 2026_02 --location arena
    python scripts/analysis/run_call_wavelet.py --dates 2026_02 --location arena --bin-min 5 --pmax-h 48
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from light_cycle import get_light_cycle_for_month  # noqa: E402
from run_ethogram import (  # noqa: E402
    BASE_PROCESSED, LOCATION_GROUPS, load_all_calls, make_bin_edges, recording_coverage_minutes,
)

BIN_MINUTES = 10
PMIN_H = 1 / 3.0                 # shortest period shown (20 min)
PMAX_H = 48.0                    # longest period shown
N_PERIODS = 96
W0 = 6.0                        # Morlet central frequency
MIN_COVERAGE_FRAC = 0.5


def morlet_cwt(x, dt, periods, w0=W0):
    """FFT Morlet CWT (Torrence & Compo). x: real signal; dt, periods in same units.

    Returns power (n_periods, n) and the cone-of-influence period at each sample.
    """
    n = len(x)
    fourier_factor = (4 * np.pi) / (w0 + np.sqrt(2 + w0**2))   # period = fourier_factor * scale
    scales = periods / fourier_factor
    X = np.fft.fft(x)
    omega = 2 * np.pi * np.fft.fftfreq(n, d=dt)               # angular frequency
    W = np.empty((len(scales), n), dtype=complex)
    for i, s in enumerate(scales):
        norm = np.sqrt(2 * np.pi * s / dt) * np.pi ** (-0.25)
        psi = norm * np.exp(-0.5 * (s * omega - w0) ** 2) * (omega > 0)
        W[i] = np.fft.ifft(X * psi)
    power = np.abs(W) ** 2
    # cone of influence: e-folding for Morlet ~ sqrt(2)*scale -> period sqrt(2)*period
    edge = np.minimum(np.arange(n), np.arange(n)[::-1]) * dt
    coi = fourier_factor * np.sqrt(2) * edge                  # reliable up to this period
    return power, coi


def _night_spans(ax, t0, t1, light_start, light_end):
    """Shade dark hours across [t0, t1] (datetimes)."""
    day = pd.Timestamp(t0).normalize()
    while day <= pd.Timestamp(t1):
        # dark = [light_end, next light_start]
        a = day + pd.Timedelta(hours=light_end)
        b = day + pd.Timedelta(days=1, hours=light_start)
        ax.axvspan(max(a, pd.Timestamp(t0)), min(b, pd.Timestamp(t1)),
                   color="0.85", alpha=0.5, lw=0, zorder=0)
        day += pd.Timedelta(days=1)


def plot_wavelet(date_folder, centers, rate, power, coi, periods, valid,
                 light_start, light_end, sel, out_path):
    tnum = mdates.date2num(centers)
    Z = np.log10(power)
    gap = ~valid
    Zm = np.ma.array(Z, mask=np.broadcast_to(gap, Z.shape))
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("0.8")

    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[5, 1], height_ratios=[1, 4],
                          hspace=0.06, wspace=0.04)
    axR = fig.add_subplot(gs[0, 0])                  # rate strip
    axS = fig.add_subplot(gs[1, 0], sharex=axR)      # scalogram
    axG = fig.add_subplot(gs[1, 1], sharey=axS)      # global spectrum

    # --- rate strip ---
    _night_spans(axR, centers[0], centers[-1], light_start, light_end)
    axR.plot(centers, np.where(valid, rate, np.nan), color="#1f77b4", lw=0.6)
    axR.set_ylabel("calls/min", fontsize=9)
    axR.set_title(f"Wavelet scalogram — {date_folder} ({sel}); night shaded",
                  fontsize=12, loc="left")
    axR.tick_params(labelbottom=False)
    axR.margins(x=0)

    # --- scalogram ---
    pcm = axS.pcolormesh(tnum, periods, Zm, cmap=cmap, shading="nearest")
    axS.set_yscale("log")
    axS.set_ylim(periods[0], periods[-1])
    axS.set_ylabel("period (hours)")
    for p, lab in ((2, "2 h"), (12, "12 h"), (24, "24 h")):
        if periods[0] <= p <= periods[-1]:
            axS.axhline(p, color="white", lw=0.8, ls="--", alpha=0.7)
            axS.text(tnum[1], p, f" {lab}", color="white", fontsize=8, va="bottom")
    # cone of influence (hatch the unreliable region)
    coi_p = np.clip(coi, periods[0], periods[-1])
    axS.fill_between(tnum, coi_p, periods[-1], color="none", hatch="xx",
                     edgecolor="white", lw=0, alpha=0.25)
    axS.plot(tnum, coi_p, color="white", lw=0.8, alpha=0.6)
    axS.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    axS.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axS.set_xlabel("date")
    fig.colorbar(pcm, ax=axG, label="log₁₀ power", fraction=0.5, pad=0.05)

    # --- global wavelet spectrum (time-averaged over valid columns) ---
    gws = np.nanmean(np.where(valid[None, :], power, np.nan), axis=1)
    axG.plot(gws, periods, color="#333333", lw=1.6)
    axG.set_yscale("log")
    for p in (2, 12, 24):
        if periods[0] <= p <= periods[-1]:
            axG.axhline(p, color="0.6", lw=0.8, ls="--")
    axG.set_xlabel("mean power")
    axG.tick_params(labelleft=False)
    axG.set_title("global spectrum", fontsize=9, loc="left")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # report the dominant global-spectrum period and whether anything peaks near 2 h
    order = np.argsort(gws)[::-1]
    top = periods[order[0]]
    print(f"  global spectrum: dominant period = {top:.1f} h")
    band = (periods >= 1.3) & (periods <= 3.5)
    if band.any():
        i = np.argmax(gws * band)
        print(f"  ultradian band 1.3-3.5 h: max power at {periods[i]:.1f} h "
              f"(= {gws[i] / np.nanmean(gws):.2f}x mean power)")
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, location, call_type, bin_minutes, pmin_h, pmax_h,
                 n_periods, min_frac, out_dir):
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

    bin_edges = make_bin_edges(df, bin_minutes)
    centers = bin_edges[:-1] + pd.Timedelta(minutes=bin_minutes / 2)
    counts, _ = np.histogram(df["start_time_real"].values.astype("int64"),
                             bins=bin_edges.values.astype("int64"))
    covered = recording_coverage_minutes(date_folder, bin_edges)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = counts / covered
    valid = covered >= min_frac * bin_minutes
    rate[~valid] = np.nan

    # interpolate across gaps for the transform; mask them in the display
    x = rate - np.nanmean(rate)
    idx = np.arange(len(x))
    x = np.interp(idx, idx[valid], x[valid])

    dt = bin_minutes / 60.0                                   # hours
    periods = np.logspace(np.log10(pmin_h), np.log10(pmax_h), n_periods)
    power, coi = morlet_cwt(x, dt, periods)
    print(f"{date_folder}: CWT {power.shape[0]} periods x {power.shape[1]} samples "
          f"({100*valid.mean():.0f}% bins recorded)")

    light_start, light_end = get_light_cycle_for_month(date_folder)
    slug = "_".join(label_bits) if label_bits else "allcalls"
    out_path = out_dir / date_folder / f"call_wavelet_{date_folder}_{slug}_{bin_minutes}min.png"
    plot_wavelet(date_folder, centers, rate, power, coi, periods, valid,
                 light_start, light_end, sel, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=["2026_02"])
    ap.add_argument("--location", default="arena", help="all | arena | underground")
    ap.add_argument("--call-type", default="all", help="all | alarm | high-freq | warble | stacks | newborn")
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES)
    ap.add_argument("--pmin-h", type=float, default=PMIN_H)
    ap.add_argument("--pmax-h", type=float, default=PMAX_H)
    ap.add_argument("--n-periods", type=int, default=N_PERIODS)
    ap.add_argument("--min-coverage-frac", type=float, default=MIN_COVERAGE_FRAC)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    args = ap.parse_args()
    for date_folder in args.dates:
        run_for_date(date_folder, args.location, args.call_type, args.bin_min,
                     args.pmin_h, args.pmax_h, args.n_periods, args.min_coverage_frac, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
