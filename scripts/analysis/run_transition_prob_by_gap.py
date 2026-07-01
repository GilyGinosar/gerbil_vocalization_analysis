"""Transition probability vs inter-event gap (tau), split by location.

For each *current* type (one panel), plot P(next type | current type, gap ~ tau)
against the silent gap tau (log x-axis), one line per *next* type, in arena vs
underground rows. This is the conditional transition probability resolved by the
gap -- it is NOT base-rate corrected (a common type dominates every row); see the
log2(observed/expected) enrichment view for that. Lifted from the "tau-resolved"
figures in notebooks/bout_transitions.ipynb.

Unit: consecutive individual calls (bouts disregarded). Within each
(date_folder, exp, assigned_location) group, calls are ordered by start time and
each adjacent pair contributes tau = next.start - curr.stop. Only the four main
call types are used (warble, high-freq, alarm, stacks).

Usage:
    python scripts/analysis/run_transition_prob_by_gap.py                 # all dates pooled
    python scripts/analysis/run_transition_prob_by_gap.py --per-date      # one figure per date
    python scripts/analysis/run_transition_prob_by_gap.py --dates 2025_10 2026_02 --format png
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
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import BASE_PROCESSED, load_all_calls  # noqa: E402
from vocalization_analysis.bouts import BOUT_THRESHOLDS, DEFAULT_GROUP_COLS  # noqa: E402

# === config ================================================================
DEFAULT_DATES = ["2025_03", "2025_07", "2025_10", "2026_02"]
GAP_LO_S, GAP_HI_S = 2, 300            # gap band shaded for reference (seconds)
BOUT_CALL_TYPES = list(BOUT_THRESHOLDS)        # ["warble", "high-freq", "alarm", "stacks"]
GROUP_COLS = list(DEFAULT_GROUP_COLS)          # transitions never cross these
LOC_GROUPS = {"arena_1": "arena", "arena_2": "arena", "underground": "underground"}
TYPE_COLORS = {"warble": "#2A9D8F", "high-freq": "#457B9D",
               "alarm": "#E63946", "stacks": "#E9C46A"}
TAU_BINS = np.logspace(np.log10(0.005), np.log10(10800), 41)  # log-spaced gap bins (s), 5 ms to 3 h
TAU_MARKERS = [0.035, GAP_LO_S, GAP_HI_S]                   # regime boundaries: 35 ms, 2 s, 300 s
TAU_MARKER_LABELS = ["35 ms", "2 s", "300 s"]
MIN_COUNT = 30                         # blank a bin with fewer transitions than this
WILSON_Z = 1.0                         # error-bar half-width in sigmas (1.0 ~ 68%; 1.96 = 95% CI)
EXPORTS_DIR = REPO_ROOT / "exports"    # also drop every saved figure here for easy download
# ===========================================================================


def _save_and_export(fig, out_path):
    """Save the figure to its out_path, then drop a copy under exports/."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(out_path, EXPORTS_DIR / out_path.name)
    print(f"wrote {out_path}\n   + exports/{out_path.name}")


def load_calls(dates: list[str]) -> pd.DataFrame:
    """Pool the per-date calls; keep only the columns the analysis needs."""
    calls = pd.concat([load_all_calls(d) for d in dates], ignore_index=True)
    for col in ("start_time_real", "stop_time_real"):
        calls[col] = pd.to_datetime(calls[col], errors="coerce")
    return calls.dropna(subset=["event_type", "start_time_real", "stop_time_real"])


def transition_pairs(events, type_order, group_cols, start_col, stop_col, type_col="event_type"):
    """All consecutive transitions among `type_order` events, with their gap.

    Events are filtered to `type_order`, ordered by start within each group, and each
    adjacent pair yields (curr_type, next_type, tau_s = next.start - curr.stop).
    """
    ev = events[events[type_col].isin(type_order)]
    parts = []
    for _, g in ev.groupby(list(group_cols)):
        g = g.sort_values(start_col)
        t = g[type_col].to_numpy()
        st = g[start_col].to_numpy()
        sp = g[stop_col].to_numpy()
        tau = (st[1:] - sp[:-1]) / np.timedelta64(1, "s")
        parts.append(pd.DataFrame({"curr_type": t[:-1], "next_type": t[1:], "tau_s": tau}))
    if not parts:
        return pd.DataFrame(columns=["curr_type", "next_type", "tau_s"])
    return pd.concat(parts, ignore_index=True)


def loc_split_pairs(events, type_order, group_cols, start_col, stop_col):
    """Transition pairs computed separately for arena and underground."""
    ev = events.assign(_loc2=events["assigned_location"].map(LOC_GROUPS))
    return {loc: transition_pairs(ev[ev["_loc2"] == loc], type_order, group_cols,
                                  start_col, stop_col)
            for loc in ["arena", "underground"]}


def plot_tau_curves_by_loc(pairs_by_loc, type_order, title, tau_bins=TAU_BINS, min_count=MIN_COUNT):
    """Grid of tau curves: rows = location, cols = current type, lines = next type."""
    rows = list(pairs_by_loc)
    centers = np.sqrt(tau_bins[:-1] * tau_bins[1:])
    fig, axes = plt.subplots(len(rows), len(type_order),
                             figsize=(3.3 * len(type_order), 3.0 * len(rows)),
                             sharex=True, sharey=True, squeeze=False)
    for r, loc in enumerate(rows):
        P = pairs_by_loc[loc]
        P = P[P["tau_s"] > 0].copy()
        # chance level = marginal P(next type) in this location (independent of current/tau);
        # a curve only departs from its own dashed line where there is real dependence.
        base = P["next_type"].value_counts(normalize=True).reindex(type_order, fill_value=0)
        print(f"  chance ({loc}): " + ", ".join(f"{Y} {base[Y]:.3f}" for Y in type_order))
        P["bin"] = np.digitize(P["tau_s"].to_numpy(), tau_bins) - 1
        P = P[(P["bin"] >= 0) & (P["bin"] < len(tau_bins) - 1)]
        for c, X in enumerate(type_order):
            ax = axes[r, c]
            for Y in type_order:                         # dashed per-type chance-level reference
                ax.axhline(base[Y], color=TYPE_COLORS.get(Y), ls="--", lw=0.8, alpha=0.45, zorder=0)
            sub = P[P["curr_type"] == X]
            counts = np.zeros((len(centers), len(type_order)))
            for j, Y in enumerate(type_order):
                s = sub[sub["next_type"] == Y].groupby("bin").size()
                counts[s.index.to_numpy(), j] = s.to_numpy()
            total = counts.sum(axis=1)
            n = total[:, None]
            with np.errstate(invalid="ignore", divide="ignore"):
                prob = counts / n
                # Wilson score interval (+/- WILSON_Z sigma) for each proportion --
                # stays within [0, 1] and behaves at small n / extreme p, unlike Wald.
                z = WILSON_Z
                denom = 1 + z**2 / n
                center = (prob + z**2 / (2 * n)) / denom
                half = (z / denom) * np.sqrt(prob * (1 - prob) / n + z**2 / (4 * n**2))
                lo, hi = center - half, center + half
            mask = total < min_count
            prob[mask] = np.nan
            lo[mask] = np.nan
            hi[mask] = np.nan
            for j, Y in enumerate(type_order):
                yerr = np.clip(np.vstack([prob[:, j] - lo[:, j], hi[:, j] - prob[:, j]]), 0, None)
                ax.errorbar(centers, prob[:, j], yerr=yerr, marker="o", ms=2.5, lw=1.3,
                            color=TYPE_COLORS.get(Y), label=Y, elinewidth=0.6, capsize=0)
            for mx, mlab in zip(TAU_MARKERS, TAU_MARKER_LABELS):
                ax.axvline(mx, color="gray", ls=":", lw=0.7)
                ax.text(mx, 0.985, mlab, transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=6, color="gray",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.6))
            ax.set_xscale("log")
            ax.set_ylim(0, 1)
            ax.text(0.96, 0.94, f"n={int(total.sum()):,}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=7, color="gray")
            if r == 0:
                ax.set_title(f"current = {X}", fontsize=10,
                             color=TYPE_COLORS.get(X), fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"{loc}\nP(next | current, tau)", fontsize=9)
            if r == len(rows) - 1:
                ax.set_xlabel("gap tau (s)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
    axes[0, -1].legend(title="next type", fontsize=7, loc="upper right")
    fig.suptitle(title, y=1.01, fontsize=13)
    fig.tight_layout()
    return fig


def run(dates, out_dir, fmt, min_count):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "+".join(dates)
    pairs = loc_split_pairs(calls, BOUT_CALL_TYPES, GROUP_COLS,
                            "start_time_real", "stop_time_real")
    title = f"Call-call transition probability  (dates: {', '.join(dates)})"
    fig = plot_tau_curves_by_loc(pairs, BOUT_CALL_TYPES, title, min_count=min_count)
    _save_and_export(fig, out_dir / f"transition_prob_by_gap_call_{tag}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--min-count", type=int, default=MIN_COUNT,
                    help="blank a tau bin with fewer transitions than this")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates together")
    args = ap.parse_args()
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.min_count)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
