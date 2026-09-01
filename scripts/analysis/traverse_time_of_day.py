#!/usr/bin/env python
"""When do traverses and calls happen? For scheduling a recording session.

Four things a logger schedule depends on:

1. Time of day. Traverses peak sharply ~3 h after lights on. Peak-to-trough is
   only ~2.6x, but the peak hour is the best hour at every window length up to
   7 h, and the 8 h window containing it ties the best evening window to within
   0.4% -- so there is no duration at which the morning is the wrong answer.
2. Coverage. Recording is NOT uniform across the clock, so raw counts per hour
   are a recording schedule as much as a behaviour. Every rate here is divided by
   the hours that actually exist in that hour bin, video and audio separately,
   and the coverage panel shows where each denominator is thin.
3. Error. The bands are a bootstrap over DAYS, not Poisson. Counting error on a
   pooled hour is ~5%; the day-to-day spread is 30-60%, so a Poisson bar would
   understate the uncertainty that actually matters by an order of magnitude and
   make every hour look reliably different from its neighbour.
4. Day of experiment. The rate halves on 2026-03-02, the day the new litter
   appears in the logs -- a step, not a slope. Bigger than any choice of hour.

Calls are DAS output and inherit its known problems -- playbacks counted as
calls, no cross-channel arbitration, per-experiment threshold drift. They are
here as an activity proxy, not as a measurement; read the shape, not the height.

    python scripts/analysis/traverse_time_of_day.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --out-dir exports/burrow/time_of_day
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.arena_occupancy_by_hour import (  # noqa: E402
    ARENAS, occupancy_grids,
)
from scripts.utils.data_rules import load_traverses  # noqa: E402
from scripts.utils.ethogram_io import load_all_calls  # noqa: E402

AUDIO_BASE = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/"
                  "Processed_data/Audio")
CHUNK_S = 360.0
# validated pair (dataviz validator, light surface): CVD dE 24.3, normal 27.3
ACCENT = UNDER_C = "#8c3a1e"
ABOVE_C = "#1a6fc4"
INK, MUTED, GRID, DARK_BAND = "#1c1a19", "#6b6560", "#e3dedb", "#eceaf2"
N_BOOT = 2000


def spread(t0: pd.Timestamp, dur_s: float, total: float, acc: dict, day) -> None:
    """Add `total` into (day, hour) bins, split by how the window straddles them."""
    rem, cur = dur_s, t0
    while rem > 0:
        nxt = (cur + pd.Timedelta(hours=1)).floor("h")
        take = min(rem, (nxt - cur).total_seconds())
        # key on the day this SLICE falls in, not the day the window started: a
        # file beginning at 23:57 contributes real 00:00 coverage to the next day,
        # and its events are keyed by their own timestamp
        acc[(cur.date(), cur.hour)] = (acc.get((cur.date(), cur.hour), 0.0)
                                       + total * take / dur_s)
        cur, rem = cur + pd.Timedelta(seconds=take), rem - take


def to_grid(per_day_hour: dict, days: list) -> np.ndarray:
    """dict[(day, hour)] -> (n_days, 24) array, in `days` order."""
    idx = {d: i for i, d in enumerate(days)}
    out = np.zeros((len(days), 24))
    for (d, h), v in per_day_hour.items():
        if d in idx:
            out[idx[d], h] = v
    return out


def bootstrap_band(counts: np.ndarray, cov: np.ndarray, n_boot: int = N_BOOT,
                   lo: float = 2.5, hi: float = 97.5, seed: int = 0):
    """Pooled rate per hour, with a CI from resampling DAYS with replacement.

    Resampling days rather than events is the whole point: the uncertainty that
    matters for "what will one recording session get me" is which day you land
    on, and that is an order of magnitude larger than the counting error.
    """
    rng = np.random.default_rng(seed)
    n = counts.shape[0]
    pooled = np.divide(counts.sum(0), cov.sum(0), out=np.zeros(24),
                       where=cov.sum(0) > 0)
    draws = np.empty((n_boot, 24))
    for b in range(n_boot):
        take = rng.integers(0, n, n)
        c, v = counts[take].sum(0), cov[take].sum(0)
        draws[b] = np.divide(c, v, out=np.full(24, np.nan), where=v > 0)
    return (pooled, np.nanpercentile(draws, lo, axis=0),
            np.nanpercentile(draws, hi, axis=0))


def occupancy_band(date: str, n_boot: int = N_BOOT, seed: int = 0):
    """Animals visible in the arenas per hour, with a day-bootstrap CI.

    Each arena is its own ratio (detections / observed frames) before the two are
    summed, so an hour where one camera recorded more than the other is not
    weighted toward that camera.
    """
    dets, frames, days = occupancy_grids(date)
    dg = {a: to_grid(dets[a], days) for a in ARENAS}
    fg = {a: to_grid(frames[a], days) for a in ARENAS}

    def pooled(rows):
        out = np.zeros(24)
        for a in ARENAS:
            d, f = dg[a][rows].sum(0), fg[a][rows].sum(0)
            out += np.divide(d, f, out=np.zeros(24), where=f > 0)
        return out

    rng = np.random.default_rng(seed)
    n = len(days)
    draws = np.array([pooled(rng.integers(0, n, n)) for _ in range(n_boot)])
    return (pooled(np.arange(n)), np.nanpercentile(draws, 2.5, axis=0),
            np.nanpercentile(draws, 97.5, axis=0), min(days), max(days))


def video_coverage(scan: Path, traverses: pd.DataFrame) -> dict:
    """(day, hour) -> hours of burrow_side video, from every SCANNED file.

    Files with no traverse at all are exactly the quiet hours, so their times are
    extrapolated from each experiment's linear file_num -> clock map rather than
    left out, which would flatten the pattern being measured.
    """
    known = traverses.groupby(["exp", "file_num"]).file_start.first().reset_index()
    scanned = []
    for d in sorted(scan.glob("*/")):
        if d.name.isdigit():
            scanned += [(int(d.name), int(p.stem.split("_")[-1]))
                        for p in d.glob("traverses_video_burrow_side_*.csv")]
    scanned = pd.DataFrame(scanned, columns=["exp", "file_num"])

    acc: dict = {}
    for exp, g in known.groupby("exp"):
        s = scanned[scanned.exp == exp]
        if s.empty:
            continue
        origin = g.file_start.min()
        if len(g) == 1:
            pred = origin + pd.to_timedelta(
                (s.file_num - g.file_num.iloc[0]) * CHUNK_S, unit="s")
        else:
            slope, icept = np.polyfit(
                g.file_num.to_numpy(float),
                (g.file_start - origin).dt.total_seconds().to_numpy(), 1)
            pred = origin + pd.to_timedelta(
                icept + slope * s.file_num.to_numpy(float), unit="s")
        for t0 in pred:
            spread(t0, CHUNK_S, CHUNK_S / 3600.0, acc, t0.date())
    return acc


def audio_coverage(date: str, keep_exps: set) -> dict:
    """(day, hour) -> hours of audio, from each experiment's file_times.csv.

    Restricted to experiments that actually reached the calls table: one with
    recorded audio but no DAS output would add denominator with no possible
    numerator and drag its hours artificially quiet.
    """
    acc: dict = {}
    for f in sorted(glob.glob(str(AUDIO_BASE / date / "*" / "file_times.csv"))):
        t = pd.read_csv(f)
        if t.empty or int(t.exp_num.iloc[0]) not in keep_exps:
            continue
        start = pd.to_datetime(t.start_date.astype(str) + " " + t.start_time.astype(str),
                               errors="coerce")
        end = pd.to_datetime(t.end_date.astype(str) + " " + t.end_time.astype(str),
                             errors="coerce")
        for t0, t1 in zip(start, end):
            if pd.isna(t0):
                continue
            dur = (t1 - t0).total_seconds() if pd.notna(t1) else CHUNK_S
            if not np.isfinite(dur) or dur <= 0 or dur > 4 * CHUNK_S:
                dur = CHUNK_S
            spread(t0, dur, dur / 3600.0, acc, t0.date())
    return acc


def best_window(counts: np.ndarray, cov: np.ndarray, width: int,
                prefer_hour: int | None = None, tol: float = 0.05):
    """Best contiguous window, breaking near-ties toward the peak hour.

    At 8 h the top window by raw rate misses the morning peak and wins by 0.4% --
    inside the noise, and it would send you to the wrong half of the day on a coin
    flip. Conditioned on days that actually have coverage, the peak-containing
    window is also the steadier one (CV 0.28 vs 0.63).
    """
    scored = []
    for s in range(24):
        idx = [(s + i) % 24 for i in range(width)]
        if cov[idx].sum() <= 0:
            continue
        scored.append((counts[idx].sum() / cov[idx].sum(), s,
                       counts[idx].sum() / counts.sum()))
    if not scored:
        return None
    scored.sort(reverse=True)
    best = scored[0]
    if prefer_hour is not None:
        for cand in scored:
            if prefer_hour in [(cand[1] + i) % 24 for i in range(width)]:
                return cand if cand[0] >= best[0] * (1 - tol) else best
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--date", default="2026_02")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--event", default="2026-03-02",
                    help="dated colony event to split the decline panel on; for 2026_02 "
                         "the first log mention of 'babies'. Pass '' for a median split.")
    ap.add_argument("--lights", default="4,16")
    args = ap.parse_args()

    scan = Path(args.scan)
    on, off = (int(v) for v in args.lights.split(","))

    tv = load_traverses(scan, args.date, keep_capped=True)
    tv["file_start"] = tv.start_time_real - pd.to_timedelta(tv.t_entry, unit="s")
    tv["hour"] = tv.start_time_real.dt.hour
    tv["day"] = tv.start_time_real.dt.date

    vcov = video_coverage(scan, tv)
    vdays = sorted({d for d, _ in vcov})
    tcnt = to_grid(dict(tv.groupby(["day", "hour"]).size()), vdays)
    vgrid = to_grid(vcov, vdays)
    trate, tlo, thi = bootstrap_band(tcnt, vgrid)

    calls = load_all_calls(args.date)          # drops the truncated last chunk
    calls["hour"] = calls.start_time_real.dt.hour
    calls["day"] = calls.start_time_real.dt.date
    acov = audio_coverage(args.date, set(calls.exp.unique()))
    adays = sorted({d for d, _ in acov})
    agrid = to_grid(acov, adays)
    bands = {}
    for lab, sub in (("underground", calls[calls.assigned_location == "underground"]),
                     ("arenas", calls[calls.assigned_location != "underground"])):
        bands[lab] = bootstrap_band(
            to_grid(dict(sub.groupby(["day", "hour"]).size()), adays), agrid)

    hours = np.arange(24)
    occ, olo, ohi, oday0, oday1 = occupancy_band(args.date)

    fig = plt.figure(figsize=(16.5, 13.0))
    gs = fig.add_gridspec(4, 2, width_ratios=[1.75, 0.85],
                          height_ratios=[2.2, 1.9, 1.7, 0.8], hspace=0.30, wspace=0.17)
    ax = fig.add_subplot(gs[0, 0])
    axk = fig.add_subplot(gs[1, 0], sharex=ax)
    axo = fig.add_subplot(gs[2, 0], sharex=ax)
    axc = fig.add_subplot(gs[3, 0], sharex=ax)
    axd = fig.add_subplot(gs[0, 1])

    for a in (ax, axk, axo, axc):
        if off > on:
            a.axvspan(-0.5, on - 0.5, color=DARK_BAND, zorder=0)
            a.axvspan(off - 0.5, 23.5, color=DARK_BAND, zorder=0)
        a.set_xlim(-0.5, 23.5)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    ax.fill_between(hours, tlo, thi, color=ACCENT, alpha=0.20, lw=0, zorder=2)
    ax.plot(hours, trate, color=ACCENT, lw=2.4, marker="o", ms=5, zorder=3)
    peak = int(np.argmax(trate))
    ax.annotate(f"peak {trate[peak]:.0f}/h at {peak:02d}:00\n"
                f"(+{(peak - on) % 24} h after lights on)",
                xy=(peak, thi[peak]), xytext=(peak + 1.6, thi[peak] * 1.00),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    r8, s8, share8 = best_window(tcnt.sum(0), vgrid.sum(0), 8, prefer_hour=peak)
    top = ax.get_ylim()[1]
    ax.plot([s8 - 0.5, s8 + 7.5], [top * 0.06] * 2, color=ACCENT, lw=3.0,
            solid_capstyle="butt", zorder=5)
    ax.text(s8 + 3.5, top * 0.09,
            f"best 8 h: {s8:02d}:00–{(s8 + 8) % 24:02d}:00 · {r8:.0f}/h · "
            f"{100 * share8:.0f}% of traverses", ha="center", fontsize=9, color=ACCENT)
    ax.set_ylabel("traverses per hour")
    ax.set_title(f"{args.date} — traverses and calls by hour", loc="left", fontsize=12)
    ax.text(0.012, 0.955, f"lights on {on:02d}:00, off {off:02d}:00 (shaded = dark)  ·  "
                          f"bands = 95% CI, bootstrap over days",
            transform=ax.transAxes, fontsize=8.5, color=MUTED, va="top")
    ax.tick_params(labelbottom=False)

    for lab, colour, mk in (("underground", UNDER_C, "o"), ("arenas", ABOVE_C, "s")):
        r, blo, bhi = bands[lab]
        axk.fill_between(hours, blo, bhi, color=colour, alpha=0.18, lw=0, zorder=2)
        axk.plot(hours, r, color=colour, lw=2.4, marker=mk, ms=5, zorder=3,
                 label=f"DAS calls — {lab}")
    axk.set_ylabel("calls per hour")
    axk.legend(frameon=False, fontsize=9.5, loc="upper right",
               bbox_to_anchor=(1.0, 0.88))
    axk.text(0.012, 0.97, "DAS output — an activity proxy, not a measurement",
             transform=axk.transAxes, fontsize=8.5, color=MUTED, va="top")
    axk.tick_params(labelbottom=False)

    axo.fill_between(hours, olo, ohi, color=ABOVE_C, alpha=0.18, lw=0, zorder=2)
    axo.plot(hours, occ, color=ABOVE_C, lw=2.4, marker="o", ms=5, zorder=3)
    opk = int(np.argmax(occ))
    axo.annotate(f"{occ[opk]:.1f} of 6 outside at {opk:02d}:00 — "
                 f"{occ.max() / max(occ.min(), 1e-9):.0f}× the {int(np.argmin(occ))}:00 low",
                 xy=(opk, occ[opk]), xytext=(opk + 1.4, occ[opk] * 0.90),
                 fontsize=9, color=INK,
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    axo.set_ylabel("animals visible\nin the arenas")
    axo.text(0.012, 0.97, f"video tracking only reaches {oday0:%b %d}–{oday1:%b %d}, "
                          f"i.e. before the litter — occupancy swings harder (9×) than "
                          f"traverses do (~3×)",
             transform=axo.transAxes, fontsize=8.5, color=MUTED, va="top")
    axo.tick_params(labelbottom=False)

    axc.bar(hours - 0.19, vgrid.sum(0), width=0.36, color=MUTED, label="video")
    axc.bar(hours + 0.19, agrid.sum(0), width=0.36, color="#b9b2ac", label="audio")
    axc.set_ylabel("hours\nrecorded")
    axc.set_xlabel("hour of day")
    axc.xaxis.set_major_locator(MultipleLocator(2))
    axc.legend(frameon=False, fontsize=8.5, ncol=2, loc="upper right")

    keep = vgrid.sum(1) >= 6.0
    dd = pd.DataFrame({"day": np.array(vdays)[keep], "n": tcnt.sum(1)[keep],
                       "cov_h": vgrid.sum(1)[keep]})
    dd["rate"] = dd.n / dd.cov_h
    x = np.arange(len(dd))
    axd.plot(x, dd.rate, color=ACCENT, lw=2.0, marker="o", ms=5)
    hi_i = int(np.argmax(dd.rate.to_numpy()))
    if dd.cov_h.iloc[hi_i] < 12:
        axd.annotate(f"{dd.cov_h.iloc[hi_i]:.0f} h only",
                     xy=(hi_i, dd.rate.iloc[hi_i]),
                     xytext=(hi_i + 1.5, dd.rate.iloc[hi_i] * 0.94),
                     fontsize=7.5, color=MUTED,
                     arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    labels = ("before litter", "after litter")
    if args.event:
        ev = pd.Timestamp(args.event).date()
        half = int(np.searchsorted(dd.day.to_numpy(), ev))
        axd.axvline(half - 0.5, color=INK, lw=1.2, ls=":", zorder=4)
        axd.text(half - 0.35, axd.get_ylim()[1] * 0.97, f" litter\n {ev:%b %d}",
                 fontsize=8, color=INK, va="top")
    else:
        half, labels = len(dd) // 2, ("first half", "second half")
    for lo_i, hi_j, lab in ((0, half, labels[0]), (half, len(dd), labels[1])):
        if hi_j <= lo_i:
            continue
        m = dd.n.iloc[lo_i:hi_j].sum() / dd.cov_h.iloc[lo_i:hi_j].sum()
        axd.hlines(m, lo_i - 0.4, hi_j - 0.6, color=INK, lw=1.4, ls="--", zorder=5)
        axd.text((lo_i + hi_j) / 2 - 0.5, m, f"{lab} {m:.0f}/h", va="bottom",
                 ha="center", fontsize=8, color=INK)
    axd.set_xticks(x[::4])
    axd.set_xticklabels([d.strftime("%b %d") for d in dd.day[::4]], rotation=45,
                        ha="right", fontsize=8)
    axd.set_ylabel("traverses / h", fontsize=9)
    axd.tick_params(labelsize=8)
    axd.set_title("…and it halves the day\nthe new litter appears", loc="left",
                  fontsize=10.5)
    axd.grid(axis="y", color=GRID, lw=0.8)
    axd.set_axisbelow(True)
    for side in ("top", "right"):
        axd.spines[side].set_visible(False)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "traverse_time_of_day.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    pd.DataFrame({
        "hour": hours, "video_h": vgrid.sum(0), "audio_h": agrid.sum(0),
        "traverses": tcnt.sum(0), "traverses_per_h": trate,
        "traverses_lo95": tlo, "traverses_hi95": thi,
        "calls_underground_per_h": bands["underground"][0],
        "calls_underground_lo95": bands["underground"][1],
        "calls_underground_hi95": bands["underground"][2],
        "calls_arenas_per_h": bands["arenas"][0],
        "calls_arenas_lo95": bands["arenas"][1],
        "calls_arenas_hi95": bands["arenas"][2],
    }).to_csv(out_dir / "traverses_by_hour.csv", index=False)
    print(f"wrote {out}\nwrote {out_dir}/traverses_by_hour.csv")
    print(f"peak traverses {trate[peak]:.1f}/h at {peak:02d}:00 "
          f"[95% CI {tlo[peak]:.1f}-{thi[peak]:.1f}]")
    for lab in ("underground", "arenas"):
        r = bands[lab][0]
        h = int(np.argmax(r))
        print(f"  DAS {lab:<12} peak {r[h]:8.0f}/h at {h:02d}:00 "
              f"[{bands[lab][1][h]:.0f}-{bands[lab][2][h]:.0f}]")
    for w in (2, 4, 8, 12):
        r, s, share = best_window(tcnt.sum(0), vgrid.sum(0), w, prefer_hour=peak)
        print(f"  best {w:2d} h: {s:02d}:00-{(s + w) % 24:02d}:00  {r:5.1f}/h  "
              f"{100 * share:4.1f}% of traverses")


if __name__ == "__main__":
    main()
