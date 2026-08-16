"""Inter-call-interval (ICI) histograms: all calls on top, one per type below.

A single stacked figure with a shared x-axis:
  - top panel   : gap between *every* pair of consecutive calls (any type)
  - one panel   : per call type, the gap between consecutive calls of *that same
    per type    type only (the self-ICI distribution)

All panels share the same log-spaced x bins, so the per-type distributions line
up under the pooled one and are directly comparable. Gaps are log10-transformed
and binned with linspace bins (density=True) rather than log-x + geomspace, so
the long tail stays visible; the x-axis is relabelled back to real seconds.

A gap is next.start - curr.stop within a (date_folder, exp, assigned_location)
group -- ordering by start time, never bridging experiments or locations. Only
positive gaps are kept (overlapping/simultaneous calls on different channels are
dropped). Bouts are disregarded; this is the raw call stream.

Usage:
    python scripts/analysis/run_ici_histograms.py                    # all dates pooled
    python scripts/analysis/run_ici_histograms.py --per-date         # one figure per date
    python scripts/analysis/run_ici_histograms.py --dates 2025_10 2026_02 --format png
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
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import BASE_PROCESSED, load_all_calls  # noqa: E402
from vocalization_analysis.bouts import BOUT_THRESHOLDS, DEFAULT_GROUP_COLS  # noqa: E402

# === config ================================================================
DEFAULT_DATES = ["2025_03", "2025_07", "2025_10", "2026_02"]
CALL_TYPES = list(BOUT_THRESHOLDS)             # ["warble", "high-freq", "alarm", "stacks"]
GROUP_COLS = list(DEFAULT_GROUP_COLS)          # gaps never cross these
TYPE_COLORS = {"warble": "#2A9D8F", "high-freq": "#457B9D",
               "alarm": "#E63946", "stacks": "#E9C46A", "newborn": "#984EA3"}
ALL_COLOR = "#6C757D"                           # pooled top panel
CROSS_COLOR = "#495057"                          # cross-type (off-diagonal) transitions

GAP_MIN_S, GAP_MAX_S = 0.003, 3600.0            # x range (3 ms .. 1 h); gaps outside are clipped out
N_BINS = 60                                     # linspace bins over log10(gap)
REGIME_MARKERS = [(0.035, "35 ms"), (2, "2 s"), (300, "300 s")]  # dashed reference lines
EXPORTS_DIR = REPO_ROOT / "exports"             # also drop every figure here for easy download
# ===========================================================================


def _save_and_export(fig, out_path):
    """Save the figure to out_path, then drop a copy under exports/."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, EXPORTS_DIR / out_path.name)
    print(f"wrote {out_path}\n   + exports/{out_path.name}")


def load_calls(dates: list[str]) -> pd.DataFrame:
    """Pool per-date calls; keep only rows with a usable type and time span."""
    calls = pd.concat([load_all_calls(d) for d in dates], ignore_index=True)
    for col in ("start_time_real", "stop_time_real"):
        calls[col] = pd.to_datetime(calls[col], errors="coerce")
    return calls.dropna(subset=["event_type", "start_time_real", "stop_time_real"])


def consecutive_gaps(calls: pd.DataFrame, group_cols: list[str]) -> np.ndarray:
    """Positive gaps (s) between consecutive calls, within each group.

    Within each group the calls are ordered by start time and every adjacent
    pair contributes tau = next.start - curr.stop. Negative/zero gaps
    (overlapping or simultaneous calls) are dropped.
    """
    parts = []
    for _, g in calls.groupby(group_cols):
        g = g.sort_values("start_time_real")
        st = g["start_time_real"].to_numpy()
        sp = g["stop_time_real"].to_numpy()
        parts.append((st[1:] - sp[:-1]) / np.timedelta64(1, "s"))
    if not parts:
        return np.array([])
    tau = np.concatenate(parts)
    return tau[tau > 0]


def consecutive_gap_pairs(calls: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Full-stream adjacent gaps with the prev/next call type on each.

    Unlike consecutive_gaps applied to a type subset (which measures the interval to
    the next *same-type* call, skipping others), this keeps every adjacent pair in
    the full call stream and tags it with (prev, next) type. Classifying these pairs
    -- diagonal per type vs off-diagonal 'cross' -- partitions the pooled gaps
    exactly, so the per-row contributions sum to the pooled panel.
    """
    parts = []
    for _, g in calls.groupby(group_cols):
        g = g.sort_values("start_time_real")
        st = g["start_time_real"].to_numpy()
        sp = g["stop_time_real"].to_numpy()
        ty = g["event_type"].to_numpy()
        tau = (st[1:] - sp[:-1]) / np.timedelta64(1, "s")
        parts.append(pd.DataFrame({"gap": tau, "prev": ty[:-1], "next": ty[1:]}))
    if not parts:
        return pd.DataFrame(columns=["gap", "prev", "next"])
    pairs = pd.concat(parts, ignore_index=True)
    return pairs[pairs["gap"] > 0]


PAIR_CROSS_COLORS = ["#E76F51", "#6A4C93"]       # the two directions of a cross pair


def build_panels(calls: pd.DataFrame, rows_mode: str, pair=None, split_cross=None):
    """List of (name, color, gaps) panels for the chosen decomposition.

    'adjacent' : pooled + immediate same-type rows (X->X) + a cross-type row; the
                 diagonals plus cross partition the pooled gaps exactly. If
                 split_cross=(A, B) is given, A->B and B->A are pulled out of the
                 cross row into their own rows (the rest becomes 'other cross-type'),
                 so the partition still closes but you see that pair's contribution.
    'self'     : pooled + per-type inter-same-type intervals (skipping other calls);
                 rows are on-scale but do NOT sum to the pooled panel.
    'pair'     : pooled + T1->T1, T2->T2, T1->T2, T2->T1 for a suspected-confusable
                 type pair. If the detector confuses T1/T2, the two cross rows should
                 spike at the same very-short (within-bout) gaps as the diagonals,
                 rather than at the longer gaps a real behavioural switch would give.
    """
    if rows_mode == "pair":
        t1, t2 = pair
        pairs = consecutive_gap_pairs(calls, GROUP_COLS)
        sel = lambda a, b: pairs.loc[(pairs["prev"] == a) & (pairs["next"] == b), "gap"].to_numpy()
        return [
            ("all consecutive calls", ALL_COLOR, pairs["gap"].to_numpy()),
            (f"{t1}→{t1}", TYPE_COLORS.get(t1, ALL_COLOR), sel(t1, t1)),
            (f"{t2}→{t2}", TYPE_COLORS.get(t2, ALL_COLOR), sel(t2, t2)),
            (f"{t1}→{t2}", PAIR_CROSS_COLORS[0], sel(t1, t2)),
            (f"{t2}→{t1}", PAIR_CROSS_COLORS[1], sel(t2, t1)),
        ]
    if rows_mode == "self":
        panels = [("all consecutive calls", ALL_COLOR, consecutive_gaps(calls, GROUP_COLS))]
        for ct in CALL_TYPES:
            panels.append((ct, TYPE_COLORS.get(ct, ALL_COLOR),
                           consecutive_gaps(calls[calls["event_type"] == ct], GROUP_COLS)))
        return panels
    pairs = consecutive_gap_pairs(calls, GROUP_COLS)
    # Every type present needs its own diagonal row, else its self-pairs (e.g.
    # newborn->newborn) fall into neither a diagonal nor 'cross' and the partition
    # leaks. Bout types first (in their canonical order), then any extras (newborn).
    row_types = list(CALL_TYPES) + [t for t in pd.unique(pairs[["prev", "next"]].values.ravel())
                                    if t not in CALL_TYPES]
    panels = [("all consecutive calls", ALL_COLOR, pairs["gap"].to_numpy())]
    for ct in row_types:
        m = (pairs["prev"] == ct) & (pairs["next"] == ct)
        panels.append((f"{ct}→{ct}", TYPE_COLORS.get(ct, ALL_COLOR),
                       pairs.loc[m, "gap"].to_numpy()))
    cross_mask = pairs["prev"] != pairs["next"]
    if split_cross:                                  # break out A->B and B->A into own rows
        a, b = split_cross
        for i, (u, v) in enumerate(((a, b), (b, a))):
            sel = (pairs["prev"] == u) & (pairs["next"] == v)
            panels.append((f"{u}→{v}", PAIR_CROSS_COLORS[i], pairs.loc[sel, "gap"].to_numpy()))
            cross_mask &= ~sel
    cross_name = "other cross-type" if split_cross else "cross-type"
    panels.append((cross_name, CROSS_COLOR, pairs.loc[cross_mask, "gap"].to_numpy()))
    return panels


def _decade_ticks(lo: float, hi: float):
    """log10 tick positions + second labels at every decade within [lo, hi]."""
    d0, d1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    ticks, labels = [], []
    for d in range(d0, d1 + 1):
        v = 10.0 ** d
        if v < lo or v > hi:
            continue
        ticks.append(d)
        labels.append(f"{v * 1000:g} ms" if v < 1 else f"{v:g} s")
    return ticks, labels


def plot_ici_histograms(panels, title: str) -> plt.Figure:
    """Two-column shared-x histograms from precomputed (name, color, gaps) panels.

    Left column  -- 'shape': each panel is its own density (integral = 1), so
      distributions are comparable regardless of how many calls each type has.
      This is deliberately blind to prevalence: alarm looks as tall as warble.
    Right column -- 'contribution': each panel is weighted so its integral equals
      that panel's share of ALL consecutive gaps, and every panel shares the pooled
      panel's y-range. A rare type (few alarms) therefore shows as a small sliver,
      making it obvious it can't move the full histogram.

    With rows_mode='adjacent' the diagonal (X->X) rows plus the cross-type row
    partition the pooled gaps, so their contribution areas sum to 1 (the pooled).
    """
    log_lo, log_hi = np.log10(GAP_MIN_S), np.log10(GAP_MAX_S)
    bins = np.linspace(log_lo, log_hi, N_BINS + 1)
    dbin = bins[1] - bins[0]
    centers = 0.5 * (bins[:-1] + bins[1:])

    # Per-panel gap counts, clipped to the plotted range. The pooled panel's N is
    # the denominator that puts every row on the same "share of all gaps" scale.
    counts_by_panel, n_by_panel = [], []
    for _, _, gaps in panels:
        gaps = np.asarray(gaps, dtype=float)
        gaps = gaps[(gaps >= GAP_MIN_S) & (gaps <= GAP_MAX_S)]
        counts_by_panel.append(np.histogram(np.log10(gaps), bins=bins)[0] if gaps.size
                               else np.zeros(N_BINS))
        n_by_panel.append(int(gaps.size))
    n_pooled = max(n_by_panel[0], 1)

    shape = [c / (max(n, 1) * dbin) for c, n in zip(counts_by_panel, n_by_panel)]   # each area = 1
    contrib = [c / (n_pooled * dbin) for c in counts_by_panel]                      # area = n_type / n_pooled
    shape_ymax = max((s.max() for s in shape if s.size), default=1) * 1.05
    contrib_ymax = (contrib[0].max() if contrib[0].size else 1) * 1.05             # pooled sets the scale

    fig, axes = plt.subplots(len(panels), 2, figsize=(14, 2.0 * len(panels)),
                             sharex=True, squeeze=False)
    ticks, labels = _decade_ticks(GAP_MIN_S, GAP_MAX_S)

    for r, (name, color, _gaps) in enumerate(panels):
        for col, (y, ymax) in enumerate(((shape[r], shape_ymax), (contrib[r], contrib_ymax))):
            ax = axes[r, col]
            ax.bar(centers, y, width=dbin, color=color, edgecolor="white",
                   linewidth=0.3, align="center")
            for mv, _ in REGIME_MARKERS:
                ax.axvline(np.log10(mv), color="gray", ls=":", lw=0.8)
            ax.set_ylim(0, ymax)
            ax.set_xlim(log_lo, log_hi)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.text(0.995, 0.92, f"{name}\nn={n_by_panel[r]:,}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8, color=color, fontweight="bold")
            if col == 0:
                ax.set_ylabel("density", fontsize=8)

    axes[0, 0].set_title("shape  (each ∫ = 1)", fontsize=10, pad=22)
    axes[0, 1].set_title("contribution  (∫ = share of all gaps)", fontsize=10, pad=22)
    for mv, mlab in REGIME_MARKERS:
        for col in (0, 1):
            axes[0, col].text(np.log10(mv), 1.04, mlab, transform=axes[0, col].get_xaxis_transform(),
                              ha="center", va="bottom", fontsize=7, color="gray")
    for col in (0, 1):
        axes[-1, col].set_xticks(ticks)
        axes[-1, col].set_xticklabels(labels)
        axes[-1, col].set_xlabel("inter-call gap  (log scale)")
    fig.suptitle(title, y=0.997, fontsize=12)
    fig.tight_layout()
    return fig


def run(dates: list[str], out_dir: Path, fmt: str, rows_mode: str, pair=None,
        split_cross=None) -> None:
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    panels = build_panels(calls, rows_mode, pair=pair, split_cross=split_cross)
    if rows_mode == "adjacent":                     # sanity: rows below pooled == pooled
        below = sum(len(g) for _, _, g in panels[1:])
        print(f"  partition check: rows sum {below:,} vs pooled {len(panels[0][2]):,}")
    tag = "+".join(dates)
    if rows_mode == "pair":
        suffix = f"_pair_{pair[0]}_{pair[1]}"
        title = (f"Inter-call-gap histograms  (dates: {', '.join(dates)})  "
                 f"[{pair[0]} vs {pair[1]} — detection-confusion check]")
    else:
        suffix = "" if rows_mode == "adjacent" else f"_{rows_mode}"
        if split_cross:
            suffix += f"_split_{split_cross[0]}_{split_cross[1]}"
        note = "[X→X rows + cross partition the pooled]" if rows_mode == "adjacent" \
            else "[per-type self-interval rows]"
        if split_cross:
            note = f"[X→X + {split_cross[0]}↔{split_cross[1]} + other cross = pooled]"
        title = f"Inter-call-gap histograms  (dates: {', '.join(dates)})  {note}"
    fig = plot_ici_histograms(panels, title)
    _save_and_export(fig, out_dir / f"ici_histograms_{tag}{suffix}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--rows", choices=["adjacent", "self", "pair"], default="adjacent",
                    help="'adjacent' (default): X->X immediate rows + cross-type row that "
                         "partition the pooled; 'self': per-type inter-same-type intervals; "
                         "'pair': the two directions of a suspected-confusable type pair")
    ap.add_argument("--pair", nargs=2, metavar=("TYPE1", "TYPE2"),
                    default=["high-freq", "warble"],
                    help="for --rows pair: the two call types to cross-tabulate")
    ap.add_argument("--split-cross", nargs=2, metavar=("TYPE1", "TYPE2"), default=None,
                    help="for --rows adjacent: pull TYPE1->TYPE2 and TYPE2->TYPE1 out of "
                         "the cross-type row into their own rows (rest = 'other cross-type')")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates together")
    args = ap.parse_args()
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.rows, pair=args.pair,
            split_cross=args.split_cross)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
