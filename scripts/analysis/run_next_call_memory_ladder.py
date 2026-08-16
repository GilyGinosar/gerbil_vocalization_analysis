"""How much does history add beyond the last call?  (nested-model memory ladder)

The next-call softmax model (run_next_call_logit.py) is dominated by self-repetition:
recent X -> next X. That is exactly what a first-order transition matrix already
tells you. So the fair question is: does looking at ALL recent calls jointly buy
anything OVER the single-last-call model you already have -- i.e. does gerbil
calling have memory deeper than first-order Markov? And if so, on what timescale?

We answer it with a ladder of NESTED models, each strictly adding features, scored
by leakage-free GroupKFold cross-validation (grouped by date/exp, like the sibling
script). Lower log-loss = more information about the next call.

    M0  base rate only .............. chance floor (predict the marginal)
    M1  + last call's type (1-hot) .. FIRST-ORDER MARKOV == the transition matrix
                                      you already have. No window, no timescale.
    M2  + recent per-type counts .... does ACCUMULATION over the last S s add
        (window S)                    anything beyond just the last call?
    M3  + time-of-day + location .... do the covariates add on top of that?

Everything is data-driven: no bout definition, no threshold. The only timescale
knob is the window S, swept on a log axis. M0 and M1 do not depend on S (flat
lines); M2 and M3 are curves. The gap M1 -> M2 is the headline -- the predictive
value of "all recent calls" beyond "the last call". Where that gap is largest is
the memory timescale, read straight off the data.

All models are scored on the SAME rows (every call that has a preceding call in
its (date, exp, location) block), so the log-losses are directly comparable.

Usage:
    python scripts/analysis/run_next_call_memory_ladder.py --per-date --format png
    python scripts/analysis/run_next_call_memory_ladder.py --dates 2026_02
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the sibling script's data plumbing so the two analyses stay identical.
from run_next_call_logit import (  # noqa: E402
    BOUT_CALL_TYPES, DEFAULT_C, DEFAULT_DATES, DEFAULT_S_SWEEP, GROUP_COLS, LOC_GROUPS,
    NS_PER_DAY, NS_PER_S, _save_and_export, load_calls, window_counts,
)
from ethogram_io import BASE_PROCESSED  # noqa: E402

N_SPLITS = 5
MODEL_COLORS = {"M0 base rate": "#9AA0A6", "M1 last call": "#000000",
                "M2 +counts(S)": "#457B9D", "M3 +time+loc": "#2A9D8F"}


# -----------------------------------------------------------------------------
# Build the invariant design once (everything except the S-dependent counts)
# -----------------------------------------------------------------------------
def build_ladder(calls, type_order):
    """Assemble the rows shared by all models, plus per-group arrays for counts.

    One row per call that HAS a preceding call in its (date, exp, location) block
    (so M1's "last call" is defined). Returns:
        y      int[N]        next-call type index (the label)
        groups str[N]        date/exp tag for GroupKFold
        prev   float[N, T]   one-hot of the immediately preceding call's type (M1)
        tod    float[N, 2]   sin/cos of time of day (part of M3)
        loc    float[N, 1]   underground indicator (part of M3)
        segs   list[(start_ns, code)]  per-group sorted arrays, to compute counts
                             at any window S in the SAME row order as y.
    """
    code_of = {t: i for i, t in enumerate(type_order)}
    nt = len(type_order)
    ev = calls[calls["event_type"].isin(type_order)].copy()
    ev["_loc2"] = ev["assigned_location"].map(LOC_GROUPS)
    ev = ev.dropna(subset=["_loc2"])

    Y, PREV, TOD, LOC, GRP, segs = [], [], [], [], [], []
    for (date, exp, loc), g in ev.groupby(GROUP_COLS + ["_loc2"]):
        g = g.sort_values("start_time_real")
        start_ns = g["start_time_real"].to_numpy().astype("int64")
        code = g["event_type"].map(code_of).to_numpy()
        n = len(code)
        if n < 2:
            continue                                    # no target has a predecessor
        # targets are calls 1..n-1; their predecessor is call i-1
        Y.append(code[1:])
        oh = np.zeros((n - 1, nt))
        oh[np.arange(n - 1), code[:-1]] = 1.0           # one-hot of the previous type
        PREV.append(oh)
        tod = (start_ns[1:] % NS_PER_DAY) / NS_PER_DAY
        ang = 2 * np.pi * tod
        TOD.append(np.column_stack([np.sin(ang), np.cos(ang)]))
        LOC.append(np.full(n - 1, 1.0 if loc == "underground" else 0.0))
        GRP.append(np.full(n - 1, f"{date}/{exp}"))
        segs.append((start_ns, code))                   # full history for count windows

    y = np.concatenate(Y)
    groups = np.concatenate(GRP)
    prev = np.vstack(PREV)
    tod = np.vstack(TOD)
    loc = np.concatenate(LOC)[:, None]
    return y, groups, prev, tod, loc, segs


def counts_at(segs, nt, window_s):
    """Per-type counts in [t - S, t) for every target row, in build_ladder's order.

    For each group the targets are calls 1..n-1 (start_ns[1:]); window_counts sees
    the full group history so a target counts all same-location prior calls within S.
    """
    window_ns = int(window_s * NS_PER_S)
    return np.vstack([window_counts(s, c, s[1:], nt, window_ns) for s, c in segs])


# -----------------------------------------------------------------------------
# Cross-validation (leakage-free, grouped by date/exp)
# -----------------------------------------------------------------------------
def _folds(groups, n_splits=N_SPLITS):
    return GroupKFold(n_splits=min(n_splits, np.unique(groups).size))


def cv_model(X, y, groups, classes, C):
    """GroupKFold mean (log_loss, accuracy) for a fitted softmax model on X."""
    ll, acc = [], []
    for tr, te in _folds(groups).split(X, y, groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(X[tr], y[tr])
        raw = pipe.predict_proba(X[te])
        proba = np.zeros((te.size, classes.size))       # align to global class set
        col = {c: j for j, c in enumerate(classes)}
        for j, c in enumerate(pipe.classes_):
            proba[:, col[c]] = raw[:, j]
        acc.append(accuracy_score(y[te], classes[proba.argmax(1)]))
        ll.append(log_loss(y[te], proba, labels=classes))
    return float(np.mean(ll)), float(np.mean(acc))


def cv_marginal(y, groups, classes):
    """M0: predict the TRAIN marginal for every test row (no features)."""
    ll, acc = [], []
    dummy = np.zeros((y.size, 1))
    for tr, te in _folds(groups).split(dummy, y, groups):
        marg = np.array([(y[tr] == c).mean() for c in classes])
        proba = np.repeat(marg[None], te.size, axis=0)
        acc.append(accuracy_score(y[te], np.full(te.size, classes[marg.argmax()])))
        ll.append(log_loss(y[te], proba, labels=classes))
    return float(np.mean(ll)), float(np.mean(acc))


# -----------------------------------------------------------------------------
# Figure: log-loss and accuracy vs S, one line per model
# -----------------------------------------------------------------------------
def _style_S_axis(ax, Ss):
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ax.set_xticks(Ss)
    ax.set_xticklabels([f"{s:g}" for s in Ss], rotation=45, fontsize=7)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("window S (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def render(Ss, m0, m1, m2_curve, m3_curve, dates):
    """m0, m1: (ll, acc) scalars (S-independent). m2_curve, m3_curve: lists of (ll, acc)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), constrained_layout=True)

    for ax, idx, ylab, title, lower_better in (
        (axes[0], 0, "CV log-loss", "log-loss vs window  (lower = more info)", True),
        (axes[1], 1, "CV accuracy", "accuracy vs window", False),
    ):
        ax.axhline(m0[idx], color=MODEL_COLORS["M0 base rate"], ls="--", lw=1.4, label="M0 base rate")
        ax.axhline(m1[idx], color=MODEL_COLORS["M1 last call"], ls="--", lw=1.4, label="M1 last call")
        ax.plot(Ss, [c[idx] for c in m2_curve], "o-", color=MODEL_COLORS["M2 +counts(S)"],
                lw=1.6, label="M2 +counts(S)")
        ax.plot(Ss, [c[idx] for c in m3_curve], "s-", color=MODEL_COLORS["M3 +time+loc"],
                lw=1.6, label="M3 +time+loc")
        _style_S_axis(ax, Ss)
        ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8, frameon=False)

    # annotate the headline: best (largest) M1 -> M2 log-loss gain over the sweep
    m2_ll = np.array([c[0] for c in m2_curve])
    gain = m1[0] - m2_ll
    j = int(np.argmax(gain))
    axes[0].annotate(f"max gain over M1:\n{gain[j]:+.3f} nats @ S={Ss[j]:g}s",
                     xy=(Ss[j], m2_ll[j]), xytext=(0.5, 0.15), textcoords="axes fraction",
                     fontsize=8, ha="left",
                     arrowprops=dict(arrowstyle="->", color="0.4", lw=0.8))

    fig.suptitle(f"Memory ladder: does history add beyond the last call?  "
                 f"(dates: {', '.join(dates)})", fontsize=13)
    return fig


# -----------------------------------------------------------------------------
# Orchestrate one run
# -----------------------------------------------------------------------------
def run(dates, out_dir, fmt, s_sweep, C):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    nt = len(BOUT_CALL_TYPES)
    y, groups, prev, tod, loc, segs = build_ladder(calls, BOUT_CALL_TYPES)
    classes = np.unique(y)
    print(f"ladder rows: {y.size:,} (calls with a predecessor); "
          f"{np.unique(groups).size} date/exp groups")

    # S-independent rungs -- compute once
    m0 = cv_marginal(y, groups, classes)
    m1 = cv_model(prev, y, groups, classes, C)
    print(f"  M0 base rate : logloss {m0[0]:.4f}  acc {m0[1]:.3f}")
    print(f"  M1 last call : logloss {m1[0]:.4f}  acc {m1[1]:.3f}   "
          f"(= first-order Markov / transition matrix)")

    # S-dependent rungs -- sweep the window
    Ss = sorted(set(s_sweep))
    m2_curve, m3_curve = [], []
    for S in Ss:
        cnt = counts_at(segs, nt, S)
        m2 = cv_model(np.hstack([prev, cnt]), y, groups, classes, C)
        m3 = cv_model(np.hstack([prev, cnt, tod, loc]), y, groups, classes, C)
        m2_curve.append(m2)
        m3_curve.append(m3)
        print(f"  S={S:6.1f}s  M2 ll {m2[0]:.4f} acc {m2[1]:.3f} "
              f"(gain over M1: {m1[0]-m2[0]:+.4f})   M3 ll {m3[0]:.4f} acc {m3[1]:.3f}")

    fig = render(Ss, m0, m1, m2_curve, m3_curve, dates)
    tag = "+".join(dates)
    _save_and_export(fig, out_dir / f"next_call_memory_ladder_{tag}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--s-sweep", type=float, nargs="+", default=DEFAULT_S_SWEEP,
                    help="window lengths (s) to evaluate for M2/M3")
    ap.add_argument("--C", type=float, default=DEFAULT_C,
                    help="inverse L2 strength for LogisticRegression (larger = weaker)")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.s_sweep, args.C)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
