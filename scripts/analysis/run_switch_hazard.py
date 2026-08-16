"""PILOT: can we predict WHEN a call-type switch happens (bout termination)?

The next-call model reduces to "predict the current dominant type" -- it is blind
to switches because its features are dominated by the ongoing bout. This pilot
targets the switch directly and gives it features that describe the DECAY of the
current bout, not its identity:

    target  : will the NEXT call be a different type?  (a switch; base rate ~31%)
    features: current type (one-hot)               -- hazard differs by type
              log run-length so far                -- bout age in # calls
              log time-in-run (s)                  -- bout age in seconds
              log recent inter-call gap (s)        -- is calling slowing down?
              ici trend = log(gap) - log(prev gap) -- is it DECELERATING? (+ = slower)
              log recent call count (context)      -- overall busyness
              sin/cos time-of-day, underground

Hypothesis: bouts end when they slow down / get old, so time-in-run and the ICI /
ICI-trend features should carry positive weight and lift AUC above 0.5. If AUC
stays near chance, switching is NOT set by internal call dynamics -- evidence it is
externally triggered (needs behaviour/video, not audio).

Model: binary LogisticRegression, GroupKFold by date/exp. We report ROC-AUC (the
right metric for an imbalanced yes/no), accuracy, and log-loss vs the base rate,
plus MODEL-FREE hazard curves: P(switch) vs bout age and vs recent gap.

Usage:
    python scripts/analysis/run_switch_hazard.py --dates 2026_02 --format png
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
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_next_call_logit import (  # noqa: E402
    BOUT_CALL_TYPES, DEFAULT_C, DEFAULT_DATES, GROUP_COLS, LOC_GROUPS, NS_PER_DAY,
    NS_PER_S, TYPE_COLORS, _save_and_export, load_calls,
)
from ethogram_io import BASE_PROCESSED  # noqa: E402

CTX_WINDOW_S = 3.0             # window for the "recent call count" context feature
ICI_EPS_S = 1e-3              # floor for log of inter-call gaps
N_SPLITS = 5


def build_switch_design(calls, type_order, ctx_window_s=CTX_WINDOW_S):
    """One row per call that has both a predecessor-of-predecessor and a successor.

    Returns X, feature names, y (1 = next call is a switch), groups, and a dict of
    raw (unlogged) bout-age / recent-gap arrays for the model-free hazard curves.
    """
    code_of = {t: i for i, t in enumerate(type_order)}
    nt = len(type_order)
    ctx_ns = int(ctx_window_s * NS_PER_S)
    ev = calls[calls["event_type"].isin(type_order)].copy()
    ev["_loc2"] = ev["assigned_location"].map(LOC_GROUPS)
    ev = ev.dropna(subset=["_loc2"])

    rows_X, rows_y, rows_g, age_s_all, ici_s_all = [], [], [], [], []
    for (date, exp, loc), g in ev.groupby(GROUP_COLS + ["_loc2"]):
        g = g.sort_values("start_time_real")
        start = g["start_time_real"].to_numpy().astype("int64")
        code = g["event_type"].map(code_of).to_numpy()
        n = code.size
        if n < 4:
            continue
        idx = np.arange(n)
        change = np.empty(n, bool)                      # True where a new run starts
        change[0] = True
        change[1:] = code[1:] != code[:-1]
        run_start = np.maximum.accumulate(np.where(change, idx, -1))     # last run-start index
        run_len = idx - run_start + 1                   # calls in the current run so far
        time_in_run_s = (start - start[run_start]) / NS_PER_S           # bout age (s)

        ici_s = np.full(n, np.nan)                      # gap into call i
        ici_s[1:] = (start[1:] - start[:-1]) / NS_PER_S
        ici_prev_s = np.full(n, np.nan)                 # the gap before that
        ici_prev_s[2:] = ici_s[1:-1]

        # recent call count in [t - ctx, t) (all types) -- context busyness
        hi = np.searchsorted(start, start, side="left")
        lo = np.searchsorted(start, start - ctx_ns, side="left")
        n_recent = hi - lo

        y_next_switch = np.empty(n, bool)               # will the NEXT call be a switch?
        y_next_switch[:-1] = change[1:]
        v = np.arange(2, n - 1)                         # valid targets (need ici_prev2 and a successor)

        onehot = np.zeros((v.size, nt))
        onehot[np.arange(v.size), code[v]] = 1.0
        ici_trend = np.log(ici_s[v] + ICI_EPS_S) - np.log(ici_prev_s[v] + ICI_EPS_S)
        ang = 2 * np.pi * (start[v] % NS_PER_DAY) / NS_PER_DAY
        X = np.column_stack([
            onehot,
            np.log1p(run_len[v]),
            np.log1p(time_in_run_s[v]),
            np.log(ici_s[v] + ICI_EPS_S),
            ici_trend,
            np.log1p(n_recent[v]),
            np.sin(ang), np.cos(ang),
            np.full(v.size, 1.0 if loc == "underground" else 0.0),
        ])
        rows_X.append(X)
        rows_y.append(y_next_switch[v])
        rows_g.append(np.full(v.size, f"{date}/{exp}"))
        age_s_all.append(time_in_run_s[v])
        ici_s_all.append(ici_s[v])

    names = ([f"cur_{t}" for t in type_order]
             + ["log_run_len", "log_time_in_run", "log_ici_prev", "ici_trend",
                "log_n_recent", "sin_tod", "cos_tod", "underground"])
    return (np.vstack(rows_X), names, np.concatenate(rows_y).astype(int),
            np.concatenate(rows_g),
            {"age_s": np.concatenate(age_s_all), "ici_s": np.concatenate(ici_s_all)})


def cv_metrics(X, y, groups, C):
    """GroupKFold mean ROC-AUC, accuracy and log-loss for the switch classifier."""
    gkf = GroupKFold(n_splits=min(N_SPLITS, np.unique(groups).size))
    auc, acc, ll = [], [], []
    for tr, te in gkf.split(X, y, groups):
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(X[tr], y[tr])
        proba = pipe.predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y[te], proba))
        acc.append(accuracy_score(y[te], proba >= 0.5))
        ll.append(log_loss(y[te], proba, labels=[0, 1]))
    return np.mean(auc), np.mean(acc), np.mean(ll)


def fit_coef(X, y, C):
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
    pipe.fit(X, y)
    return pipe.named_steps["logisticregression"].coef_[0]


def _hazard_curve(value, y, lo, hi, nbins=12):
    """Model-free P(switch) in log-spaced bins of `value`; returns centers, rate, n."""
    edges = np.logspace(np.log10(lo), np.log10(hi), nbins + 1)
    which = np.clip(np.digitize(value, edges) - 1, 0, nbins - 1)
    centers = np.sqrt(edges[:-1] * edges[1:])
    rate = np.array([y[which == b].mean() if (which == b).any() else np.nan
                     for b in range(nbins)])
    cnt = np.array([(which == b).sum() for b in range(nbins)])
    return centers, rate, cnt


def render(names, coef, extras, y, base, auc, acc, dates):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.25, 1, 1]})

    # --- coefficient bar ------------------------------------------------------
    ax = axes[0]
    order = np.arange(len(names))
    colors = ["#E63946" if c < 0 else "#2A9D8F" for c in coef]
    ax.barh(order, coef, color=colors)
    ax.set_yticks(order); ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_xlabel("logistic weight (per 1 SD)  -> P(switch)")
    ax.set_title("what predicts a switch", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # --- hazard vs bout age ---------------------------------------------------
    ax = axes[1]
    c, r, n = _hazard_curve(extras["age_s"], y, 0.05, 300)
    ax.plot(c, r, "o-", color="#264653")
    ax.axhline(base, color="gray", ls="--", lw=1, label=f"base {base:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("time in current bout (s)"); ax.set_ylabel("P(next call is a switch)")
    ax.set_title("hazard vs bout age", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # --- hazard vs recent gap -------------------------------------------------
    ax = axes[2]
    c, r, n = _hazard_curve(extras["ici_s"], y, 0.01, 300)
    ax.plot(c, r, "o-", color="#264653")
    ax.axhline(base, color="gray", ls="--", lw=1, label=f"base {base:.2f}")
    ax.set_xscale("log")
    ax.set_xlabel("gap before current call (s)"); ax.set_ylabel("P(next call is a switch)")
    ax.set_title("hazard vs recent inter-call gap", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(f"Switch-hazard pilot  (dates: {', '.join(dates)})   "
                 f"AUC={auc:.3f}  acc={acc:.3f}  (base switch rate {base:.3f})", fontsize=13)
    return fig


def run(dates, out_dir, fmt, C):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    X, names, y, groups, extras = build_switch_design(calls, BOUT_CALL_TYPES)
    base = y.mean()
    print(f"{y.size:,} rows; base switch rate {base:.3f}; {np.unique(groups).size} groups")

    auc, acc, ll = cv_metrics(X, y, groups, C)
    print(f"  CV  AUC {auc:.3f}   acc {acc:.3f} (base {max(base, 1-base):.3f})   logloss {ll:.3f}")
    coef = fit_coef(X, y, C)
    print("  coef (per SD): " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(names, coef)))

    fig = render(names, coef, extras, y, base, auc, acc, dates)
    tag = "+".join(dates)
    _save_and_export(fig, out_dir / f"switch_hazard_{tag}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--C", type=float, default=DEFAULT_C)
    ap.add_argument("--per-date", action="store_true")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.C)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
