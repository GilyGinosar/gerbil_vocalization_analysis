#!/usr/bin/env python
"""What the tunnel-vs-nest localiser actually does, and what its threshold costs.

`localise_calls.py` gives every underground call one number -- how much louder it
was on ch01 (at the tunnel) than on ch00 (deeper in the nest), in dB -- and then
calls it tunnel-origin if that number clears a threshold. This draws the three
things you need in order to read any figure built on that label:

  left    the two populations the threshold has to separate, and how far they
          overlap. Calls made while the tracks say the tunnel was EMPTY cannot
          have come from the tunnel, so they are the nest-origin reference.
  middle  what the label is actually made of. The empty-tunnel population is
          ~11x larger, so its small false-alarm tail outnumbers the genuine hits
          -- at the default cut, most "tunnel-origin" calls had nobody in the
          tunnel. This is a base-rate effect, not a bug in the measurement.
  right   the trade. Raising the quantile buys precision and spends sensitivity.
  far right  what the threshold is REALLY selecting. For calls made while one
          animal was in the tunnel, the level difference tracks where that animal
          was standing: about -8 dB at the nest end, near 0 at the arena end. The
          nest-end half of the tunnel scores like the nest-origin reference, so a
          call from an animal genuinely inside the tunnel is labelled nest-origin
          if it was made in the half nearer the nest. "Tunnel-origin" is closer to
          "arena-half of the tunnel" than to "from the tunnel".

The cut is computed PER EXPERIMENT (mic gain and tube acoustics differ), so the
left panel marks the median cut across experiments while the middle and right
panels use each experiment's own.

    python scripts/analysis/localiser_threshold.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/localiser
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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.video.burrow_transit_picker import file_index  # noqa: E402

# the two states the threshold has to tell apart. Same hues the rest of the burrow
# figures use; validated ΔE 25.2 protan / 29.3 tritan / 31.3 normal vision.
STATE = {"tunnel empty": "#d1642a", "animal in tunnel": "#2f6fd0"}
FPS = 30
INK, MUTED = "#222222", "#888888"
QUANTILES = (0.95, 0.98, 0.99, 0.995)
MIN_REFERENCE = 50


def load(scan: Path) -> pd.DataFrame:
    frames = []
    for path in sorted((scan / "origin").glob("*/call_origin.csv")):
        try:
            frames.append(pd.read_csv(path).assign(exp=int(path.parent.name)))
        except pd.errors.EmptyDataError:
            continue
    if not frames:
        raise SystemExit(f"no call_origin.csv under {scan}/origin")
    return pd.concat(frames, ignore_index=True)


def counts_at(table: pd.DataFrame, q: float) -> tuple[int, int]:
    """(genuine hits, false alarms) among the calls labelled tunnel-origin."""
    hits = false = 0
    for _, group in table.groupby("exp"):
        reference = group.loc[group.state == "tunnel empty", "tunnel_db_over_nest"]
        if len(reference) < MIN_REFERENCE:
            continue
        above = group[group.tunnel_db_over_nest > reference.quantile(q)]
        hits += int((above.state == "animal in tunnel").sum())
        false += int((above.state == "tunnel empty").sum())
    return hits, false


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--position-experiments", type=int, default=20,
                        help="how many experiments to walk for the position panel; it "
                             "has to open every track parquet, so the default is a sample")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    table = load(scan)
    empty = table.loc[table.state == "tunnel empty", "tunnel_db_over_nest"]
    occupied = table.loc[table.state == "animal in tunnel", "tunnel_db_over_nest"]
    n_empty, n_occupied = len(empty), len(occupied)
    median_cut = {q: float(np.median([g.loc[g.state == "tunnel empty",
                                            "tunnel_db_over_nest"].quantile(q)
                                      for _, g in table.groupby("exp")
                                      if (g.state == "tunnel empty").sum() >= MIN_REFERENCE]))
                  for q in QUANTILES}

    # where was the animal when each in-tunnel call was made? the tracks know, and
    # this is the variable the dB threshold turns out to be selecting on
    pos_rows = []
    for exp_dir in sorted(p for p in scan.iterdir()
                          if p.is_dir() and p.name.isdigit())[:args.position_experiments]:
        origin = scan / "origin" / exp_dir.name / "call_origin.csv"
        if not origin.exists():
            continue
        try:
            occ = pd.read_csv(origin)
        except pd.errors.EmptyDataError:
            continue
        occ = occ[occ.state == "animal in tunnel"]
        if occ.empty:
            continue
        by_file = {f: g for f, g in occ.groupby("file")}
        for track_path in (exp_dir / "tracks").glob("*.parquet"):
            g = by_file.get(file_index(track_path.stem + ".mp4"))
            if g is None:
                continue
            tr = pd.read_parquet(track_path)
            xs, na = tr.x.to_numpy(), tr.n_animals.to_numpy()
            for start, db in zip(g.start_s, g.tunnel_db_over_nest):
                i = int(start * FPS)
                if 0 <= i < len(xs) and na[i] == 1 and np.isfinite(xs[i]):
                    pos_rows.append((xs[i], db))
    pos = pd.DataFrame(pos_rows, columns=["x", "db"])

    fig, axes = plt.subplots(1, 4, figsize=(22, 5.0),
                             gridspec_kw={"wspace": 0.30,
                                          "width_ratios": [1.25, 1, 1, 1.1]})

    # ---- left: the overlap the threshold has to live with --------------------
    ax = axes[0]
    edges = np.arange(-30, 30.5, 1.0)
    for label, series in (("tunnel empty", empty), ("animal in tunnel", occupied)):
        ax.hist(series, bins=edges, density=True, color=STATE[label], alpha=0.55,
                label=f"{label}  (n={len(series):,})")
    top = ax.get_ylim()[1]
    for q, style, height in ((0.95, "--", 0.60), (0.99, "-", 0.42)):
        ax.axvline(median_cut[q], color=INK, lw=1.6, ls=style, zorder=5)
        ax.annotate(f"q={q}  ({median_cut[q]:+.1f} dB)", (median_cut[q], top * height),
                    textcoords="offset points", xytext=(7, 0), color=INK,
                    fontsize=9, va="center", ha="left",
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.85))
    ax.set_xlabel("call louder at the tunnel mic  →\n"
                  "20·log10(rms ch01 / rms ch00), dB")
    ax.set_ylabel("share of calls (density)")
    ax.set_title("the two populations overlap heavily", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper left")

    # ---- middle: what the label is made of -----------------------------------
    ax = axes[1]
    hits = [counts_at(table, q) for q in QUANTILES]
    x = np.arange(len(QUANTILES))
    true_n = np.array([h for h, _ in hits], float)
    false_n = np.array([f for _, f in hits], float)
    ax.bar(x, false_n, 0.62, color=STATE["tunnel empty"],
           label="tunnel was EMPTY (false alarm)")
    # a 2 px surface gap between the stacked segments, so the split is legible
    ax.bar(x, true_n, 0.62, bottom=false_n + (false_n + true_n).max() * 0.004,
           color=STATE["animal in tunnel"], label="animal WAS in the tunnel")
    for i, (h, f) in enumerate(hits):
        ax.text(i, h + f + (true_n + false_n).max() * 0.03,
                f"{100 * h / (h + f):.0f}%", ha="center", fontsize=9.5, color=INK)
    ax.set_xticks(x, [f"q={q}" for q in QUANTILES])
    ax.set_ylabel("calls labelled tunnel-origin")
    ax.set_xlabel("percent above each bar = share that are genuine")
    ax.set_title("most of the label is the false-alarm tail", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=9, loc="upper right")

    # ---- right: the trade ----------------------------------------------------
    # one axis: both series are percentages, so they belong on the same scale
    ax = axes[2]
    sweep = np.round(np.arange(0.90, 0.9985, 0.005), 4)
    precision, sensitivity = [], []
    for q in sweep:
        h, f = counts_at(table, q)
        precision.append(100 * h / max(h + f, 1))
        sensitivity.append(100 * h / n_occupied)
    ax.plot(sweep, precision, color=INK, lw=2, label="precision — of the calls "
                                                     "labelled tunnel, share that are genuine")
    ax.plot(sweep, sensitivity, color=MUTED, lw=2, ls="--",
            label="sensitivity — of the real in-tunnel calls, share kept")
    for q in (0.95, 0.99):
        h, f = counts_at(table, q)
        for value, colour, dy in ((100 * h / (h + f), INK, 9),
                                  (100 * h / n_occupied, MUTED, -14)):
            ax.plot([q], [value], "o", color=colour, ms=7, zorder=5)
            ax.annotate(f"{value:.0f}%", (q, value), textcoords="offset points",
                        xytext=(7, dy), fontsize=9, color=colour)
        ax.axvline(q, color="0.85", lw=1, zorder=0)
    ax.set_xlabel("quantile of the tunnel-empty reference used as the cut")
    ax.set_ylabel("percent")
    ax.set_ylim(0, 100)
    ax.set_title("raising the cut buys precision, spends sensitivity",
                 loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower left")

    # ---- far right: the threshold is really a position cut ------------------
    ax = axes[3]
    edges_x = np.linspace(0, 1, 11)
    centres_x = (edges_x[:-1] + edges_x[1:]) / 2
    pos["bin"] = pd.cut(pos.x, edges_x)
    grouped = pos.groupby("bin", observed=True).db
    med = grouped.median().to_numpy()
    q25, q75 = grouped.quantile(0.25).to_numpy(), grouped.quantile(0.75).to_numpy()
    ax.fill_between(centres_x[:len(med)], q25, q75, color=STATE["animal in tunnel"],
                    alpha=0.20, lw=0)
    ax.plot(centres_x[:len(med)], med, color=STATE["animal in tunnel"], lw=2.2,
            label="median call, one animal in the tunnel")
    for q, style in ((0.95, "--"), (0.99, "-")):
        ax.axhline(median_cut[q], color=INK, lw=1.5, ls=style)
        ax.annotate(f"q={q} cut", (0.02, median_cut[q]), textcoords="offset points",
                    xytext=(0, 5), fontsize=8.5, color=INK)
    ax.axhline(float(empty.median()), color=STATE["tunnel empty"], lw=1.6, ls=":")
    ax.annotate("median of the nest-origin reference", (0.35, float(empty.median())),
                textcoords="offset points", xytext=(0, -14), fontsize=8.5,
                color=STATE["tunnel empty"])
    ax.set_xlabel("where the animal actually was\n(0 = nest end of the tunnel, "
                  "1 = arena end)")
    ax.set_ylabel("tunnel_db_over_nest, dB")
    ax.set_title(f"the cut is a POSITION cut  (n={len(pos):,})", loc="left", fontsize=11)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")

    for ax in axes:
        ax.grid(axis="y", color="0.93", lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    fig.suptitle(f"How a call becomes 'tunnel-origin' — {len(table):,} scored calls, "
                 f"{table.exp.nunique()} experiments of 2026_02\n"
                 f"the cut is recomputed per experiment; the left panel marks the "
                 f"median across them",
                 x=0.005, y=0.995, ha="left", va="top", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    out = out_dir / "localiser_threshold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")

    print(f"{len(table):,} scored calls  ({n_empty:,} tunnel empty, "
          f"{n_occupied:,} animal in tunnel — {n_empty / n_occupied:.0f}x more)")
    for q in QUANTILES:
        h, f = counts_at(table, q)
        print(f"  q={q:<6} cut {median_cut[q]:+6.1f} dB   labelled tunnel {h + f:6,}   "
              f"precision {100 * h / (h + f):3.0f}%   sensitivity {100 * h / n_occupied:3.0f}%")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
