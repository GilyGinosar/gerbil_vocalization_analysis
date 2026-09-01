#!/usr/bin/env python
"""Traverses, and how many animals were actually outside to make them.

The traverse rate peaks sharply in the morning, and the obvious reading -- "that
is when they move between compartments" -- turns out to be mostly wrong. Arena
occupancy swings 9x across the day while the traverse rate swings only ~4x, so
most of the morning peak is simply that more animals are out. Divide by the pool
that could actually make each crossing and the picture inverts:

    to_arena per animal INSIDE   peaks in the morning  -> emergence is the real
                                                          circadian signal
    to_nest  per animal OUTSIDE  peaks at NIGHT        -> the few animals out
                                                          after dark go straight
                                                          back in; morning animals
                                                          stay out

This is the occupancy normalisation that per-location call rates have been
waiting for: a raw per-hour rate confounds "how often does an animal do this"
with "how many animals were there to do it".

Denominators, both of which matter:
  * observation time comes from `files_vetted.video_s`, NOT `max_frame_id` --
    38% of rows have no max_frame_id precisely because nothing was detected, and
    those are the empty hours the denominator most needs.
  * `stationary` detections are dropped (the plastic object in arena_2 that the
    tracking repo now flags), otherwise arena_2 occupancy is inflated by a fifth.

Everything is restricted to the tracking window, which is shorter than the
burrow scan's and ends before the 2026-03-02 litter, so this says nothing about
the post-litter regime.

    python scripts/analysis/arena_occupancy_by_hour.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 \
        --out-dir exports/burrow/occupancy
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

from scripts.utils.data_rules import load_traverses  # noqa: E402
from scripts.utils.publish import publish  # noqa: E402

VIDEO_BASE = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/"
                  "Processed_data/Video")
FPS, CHUNK_S = 30.0, 360.0
ARENAS = ("arena_1", "arena_2")
# validated pair (dataviz validator, light surface): CVD dE 24.3, normal 27.3
IN_C, OUT_C = "#8c3a1e", "#1a6fc4"
INK, MUTED, GRID, DARK_BAND = "#1c1a19", "#6b6560", "#e3dedb", "#eceaf2"


def spread(t0: pd.Timestamp, dur_s: float, total: float, acc, day=None) -> None:
    """Add `total` into hour bins, split by how the window straddles them.

    Writes into a (day, hour) dict when `day` is given, else a plain 24-vector.
    """
    rem, cur = dur_s, t0
    while rem > 0:
        nxt = (cur + pd.Timedelta(hours=1)).floor("h")
        take = min(rem, (nxt - cur).total_seconds())
        part = total * take / dur_s
        if day is None:
            acc[cur.hour] += part
        else:
            # key on the day this SLICE falls in, not the day the file started:
            # a file beginning at 23:57 puts real 00:00 frames in the next day, and
            # detections are keyed by their own timestamp, so booking them to the
            # start date orphans every post-midnight detection
            acc[(cur.date(), cur.hour)] = acc.get((cur.date(), cur.hour), 0.0) + part
        cur, rem = cur + pd.Timedelta(seconds=take), rem - take


def occupancy_grids(date: str):
    """Per-(day, hour) detections and observed frames, per arena.

    Returned as grids rather than a pooled curve so a caller can bootstrap over
    days. Frames come from `video_s`, NOT `max_frame_id`: 38% of files_vetted rows
    have no max_frame_id precisely because nothing was detected in them, and those
    are the empty hours the denominator most needs.
    """
    frames = {a: {} for a in ARENAS}
    dets = {a: {} for a in ARENAS}
    for f in sorted(glob.glob(str(VIDEO_BASE / date / "*" / "files_vetted.csv"))):
        fv = pd.read_csv(f)
        fv["chunk_start_real"] = pd.to_datetime(fv.chunk_start_real)
        for r in fv.itertuples():
            if r.location not in frames or not r.has_video:
                continue
            dur = r.video_s if pd.notna(r.video_s) and r.video_s > 0 else CHUNK_S
            spread(r.chunk_start_real, float(dur), float(dur) * FPS,
                   frames[r.location], r.chunk_start_real.date())
        d = pd.read_parquet(f.replace("files_vetted.csv", "detections.parquet"),
                            columns=["location", "start_time_real", "stationary"])
        d = d[~d.stationary]
        t = pd.to_datetime(d.start_time_real)
        d = d.assign(day=t.dt.date, h=t.dt.hour)
        for (loc, day, h), c in d.groupby(["location", "day", "h"]).size().items():
            if loc in dets:
                dets[loc][(day, h)] = dets[loc].get((day, h), 0) + c
    days = sorted({d for a in ARENAS for d, _ in frames[a]})
    return dets, frames, days


def arena_occupancy(date: str):
    """Mean animals visible per arena per hour-of-day, and the tracking window."""
    dets, frames, days = occupancy_grids(date)
    pooled_d = {a: np.zeros(24) for a in ARENAS}
    pooled_f = {a: np.zeros(24) for a in ARENAS}
    for a in ARENAS:
        for (_, h), v in dets[a].items():
            pooled_d[a][h] += v
        for (_, h), v in frames[a].items():
            pooled_f[a][h] += v
    occ = {a: np.divide(pooled_d[a], pooled_f[a], out=np.zeros(24),
                        where=pooled_f[a] > 0) for a in ARENAS}
    return occ, min(days), max(days)


def traverse_rate(scan: Path, date: str, lo, hi):
    """Traverses per hour of recording, by direction, over [lo, hi]."""
    tv = load_traverses(scan, date, keep_capped=True, quiet=True)
    tv["file_start"] = tv.start_time_real - pd.to_timedelta(tv.t_entry, unit="s")
    known = tv.groupby(["exp", "file_num"]).file_start.first().reset_index()

    scanned = []
    for d in sorted(scan.glob("*/")):
        if d.name.isdigit():
            scanned += [(int(d.name), int(p.stem.split("_")[-1]))
                        for p in d.glob("traverses_video_burrow_side_*.csv")]
    scanned = pd.DataFrame(scanned, columns=["exp", "file_num"])

    parts = []
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
        parts.append(pd.DataFrame({"t0": pred}))
    files = pd.concat(parts, ignore_index=True)
    files = files[(files.t0.dt.date >= lo) & (files.t0.dt.date <= hi)]

    cov = np.zeros(24)
    for t0 in files.t0:
        spread(t0, CHUNK_S, CHUNK_S, cov)
    cov /= 3600.0

    w = tv[(tv.start_time_real.dt.date >= lo) & (tv.start_time_real.dt.date <= hi)].copy()
    w["hour"] = w.start_time_real.dt.hour

    def rate(sub):
        c = sub.groupby("hour").size().reindex(range(24), fill_value=0).to_numpy()
        return np.divide(c, cov, out=np.zeros(24), where=cov > 0)

    return rate(w[w.direction == "to_nest"]), rate(w[w.direction == "to_arena"]), cov


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", required=True)
    ap.add_argument("--date", default="2026_02")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-animals", type=int, default=6,
                    help="colony size; experiments.toml has 6 for 2026_02")
    ap.add_argument("--lights", default="4,16")
    args = ap.parse_args()

    on, off = (int(v) for v in args.lights.split(","))
    occ, lo, hi = arena_occupancy(args.date)
    outside = occ["arena_1"] + occ["arena_2"]
    inside = np.clip(args.n_animals - outside, 0.01, None)
    r_nest, r_arena, cov = traverse_rate(Path(args.scan), args.date, lo, hi)
    per_out = np.divide(r_nest, outside, out=np.zeros(24), where=outside > 0.05)
    per_in = r_arena / inside

    hours = np.arange(24)
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 11.0), sharex=True,
                             gridspec_kw={"hspace": 0.16})
    for a in axes:
        if off > on:
            a.axvspan(-0.5, on - 0.5, color=DARK_BAND, zorder=0)
            a.axvspan(off - 0.5, 23.5, color=DARK_BAND, zorder=0)
        a.grid(axis="y", color=GRID, lw=0.8)
        a.set_axisbelow(True)
        a.set_xlim(-0.5, 23.5)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    ax = axes[0]
    ax.fill_between(hours, outside, color=OUT_C, alpha=0.16, zorder=2)
    ax.plot(hours, outside, color=OUT_C, lw=2.4, marker="o", ms=5, zorder=3)
    pk = int(np.argmax(outside))
    ax.annotate(f"{outside[pk]:.1f} of {args.n_animals} outside at {pk:02d}:00\n"
                f"{outside.max() / max(outside.min(), 1e-9):.0f}× the {int(np.argmin(outside))}:00 low",
                xy=(pk, outside[pk]), xytext=(pk + 1.2, outside[pk] * 0.86),
                fontsize=9, color=INK,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.9))
    ax.set_ylabel("animals visible in the arenas")
    ax.set_title(f"{args.date} · {lo:%b %d}–{hi:%b %d} (tracking window, before the litter)"
                 f"   ·   lights on {on:02d}:00, off {off:02d}:00 (shaded = dark)",
                 loc="left", fontsize=12)

    ax = axes[1]
    ax.plot(hours, r_nest + r_arena, color=MUTED, lw=2.4, marker="o", ms=5)
    ax.set_ylabel("traverses per hour\n(raw — what you'd schedule on)")
    ax.text(0.5, 0.92, "raw rate swings ~4×, occupancy swings 9× — most of the "
                       "morning peak is just more animals being out",
            transform=ax.transAxes, ha="center", va="top", fontsize=9, color=MUTED)

    ax = axes[2]
    ax.plot(hours, per_in, color=IN_C, lw=2.4, marker="o", ms=5,
            label="to_arena, per animal INSIDE  (going out)")
    ax.plot(hours, per_out, color=OUT_C, lw=2.4, marker="s", ms=5,
            label="to_nest, per animal OUTSIDE  (coming in)")
    ax.set_ylabel("traverses per hour\nper available animal")
    ax.set_xlabel("hour of day")
    ax.xaxis.set_major_locator(MultipleLocator(2))
    ax.legend(frameon=False, fontsize=9.5, loc="upper left",
              bbox_to_anchor=(0.015, 0.82))
    a_pk, o_pk = int(np.argmax(per_in)), int(np.argmax(per_out))
    ax.annotate("emergence peaks\nin the morning", xy=(a_pk, per_in[a_pk]),
                xytext=(a_pk - 5.0, per_in[a_pk] * 1.55), fontsize=9, color=IN_C,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=IN_C, lw=0.9))
    ax.annotate("but the few animals out at night\ngo back in fastest",
                xy=(o_pk, per_out[o_pk]),
                xytext=(o_pk - 5.5, per_out[o_pk] * 0.72), fontsize=9, color=OUT_C,
                ha="center",
                arrowprops=dict(arrowstyle="-", color=OUT_C, lw=0.9))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "arena_occupancy_by_hour.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    pd.DataFrame({"hour": hours, "coverage_h": cov,
                  "outside_arena_1": occ["arena_1"], "outside_arena_2": occ["arena_2"],
                  "outside_total": outside, "to_nest_per_h": r_nest,
                  "to_arena_per_h": r_arena, "to_nest_per_outside": per_out,
                  "to_arena_per_inside": per_in}).to_csv(
        out_dir / "occupancy_by_hour.csv", index=False)
    print(f"wrote {out}\nwrote {out_dir}/occupancy_by_hour.csv")
    publish(out, date=args.date)
    print(f"tracking window {lo} .. {hi}")
    for lab, arr in (("animals outside", outside), ("raw traverses/h", r_nest + r_arena),
                     ("to_arena per inside", per_in), ("to_nest per outside", per_out)):
        good = arr[np.isfinite(arr) & (arr > 0)]
        print(f"  {lab:<22} peak {good.max():6.2f} at {int(np.argmax(arr)):02d}:00   "
              f"peak/trough {good.max() / good.min():.1f}×")


if __name__ == "__main__":
    main()
