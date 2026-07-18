"""Next-window call counts: given the per-type counts in a window, predict the next.

Self-excitation (run_call_rate_excitation.py) asks a scalar question: does total
activity beget total activity? This asks the multivariate one: given HOW MANY of
each call type occurred in a window, how many of each type occur in the NEXT
window? The payoff is the cross-type transfer matrix -- e.g. does a window heavy
in alarms predict more STACKS next, not just more alarms.

METHOD (a vector-autoregressive Poisson model)
  * Bin each (date, exp, location) block into fixed W-second windows; count each
    call type per window.
  * For every target type j, fit a POISSON regression predicting its NEXT-window
    count from the current window's per-type counts (log1p, all four types) plus
    time-of-day (sin/cos) and location:
        E[count_j(next)] = exp( b0 + sum_i w_ij * log1p(count_i(now)) + time + loc )
  * The weight matrix w_ij is the transfer matrix: row i = predictor type (this
    window), column j = target type (next window). Diagonal = self-persistence,
    off-diagonal = cross-type excitation (+) or suppression (-).

For each target type we also report held-out Poisson D^2 (GroupKFold by date/exp),
and the gain of the full model over a time+location-only baseline -- i.e. how much
knowing the current counts helps predict next-window counts beyond circadian.

log1p on the count features keeps the exp-link cross-validation numerically stable
(raw counts let a held-out burst blow the predicted rate up to nonsense).

Usage:
    python scripts/analysis/run_next_window_counts.py --per-date --format png
    python scripts/analysis/run_next_window_counts.py --dates 2026_02 --window 30
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_poisson_deviance
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from run_next_call_logit import (  # noqa: E402
    BOUT_CALL_TYPES, DEFAULT_DATES, GROUP_COLS, LOC_GROUPS, NS_PER_DAY, NS_PER_S,
    TYPE_COLORS, _save_and_export, load_calls,
)
from ethogram_io import BASE_PROCESSED  # noqa: E402

DEFAULT_WINDOW_S = 30.0        # window width whose per-type counts we predict
DEFAULT_ALPHA = 1e-2           # L2 strength for PoissonRegressor
N_SPLITS = 5


# -----------------------------------------------------------------------------
# Build the binned per-type count design
# -----------------------------------------------------------------------------
def build_window_design(calls, type_order, window_s):
    """Per-window per-type counts, paired current-window -> next-window.

    Returns:
        prev   float[N, T]   log1p per-type counts in the current window (predictors)
        tod    float[N, 2]   sin/cos of time of day at the NEXT window's start
        loc    float[N, 1]   underground indicator
        Y      int[N, T]     per-type counts in the NEXT window (targets)
        groups str[N]        date/exp tag for GroupKFold
    """
    code_of = {t: i for i, t in enumerate(type_order)}
    nt = len(type_order)
    win_ns = int(window_s * NS_PER_S)
    ev = calls[calls["event_type"].isin(type_order)].copy()
    ev["_loc2"] = ev["assigned_location"].map(LOC_GROUPS)
    ev = ev.dropna(subset=["_loc2"])

    PREV, TOD, LOC, Y, GRP = [], [], [], [], []
    for (date, exp, loc), g in ev.groupby(GROUP_COLS + ["_loc2"]):
        start_ns = g["start_time_real"].to_numpy().astype("int64")
        code = g["event_type"].map(code_of).to_numpy()
        order = np.argsort(start_ns)
        start_ns, code = start_ns[order], code[order]
        span = start_ns[-1] - start_ns[0]
        nbins = int(span // win_ns) + 1
        if nbins < 2:
            continue
        edges = start_ns[0] + np.arange(nbins + 1) * win_ns
        # per-type counts per bin -> C[type, bin]
        C = np.vstack([np.diff(np.searchsorted(start_ns[code == k], edges))
                       for k in range(nt)]).astype(np.float64)

        PREV.append(np.log1p(C[:, :-1].T))              # current window (bins 0..n-2)
        Y.append(C[:, 1:].T.astype(int))                # next window    (bins 1..n-1)
        ang = 2 * np.pi * (edges[1:nbins] % NS_PER_DAY) / NS_PER_DAY   # next-bin start time
        TOD.append(np.column_stack([np.sin(ang), np.cos(ang)]))
        LOC.append(np.full(nbins - 1, 1.0 if loc == "underground" else 0.0))
        GRP.append(np.full(nbins - 1, f"{date}/{exp}"))

    return (np.vstack(PREV), np.vstack(TOD), np.concatenate(LOC)[:, None],
            np.vstack(Y), np.concatenate(GRP))


# -----------------------------------------------------------------------------
# Poisson cross-validation and coefficient fit
# -----------------------------------------------------------------------------
def _folds(groups):
    return GroupKFold(n_splits=min(N_SPLITS, np.unique(groups).size))


def cv_d2(X, y, groups, alpha):
    """GroupKFold Poisson D^2 = 1 - deviance(model)/deviance(train-mean null)."""
    dev_m, dev_0 = 0.0, 0.0
    for tr, te in _folds(groups).split(X, y, groups):
        pipe = make_pipeline(StandardScaler(),
                             PoissonRegressor(alpha=alpha, max_iter=1000))
        pipe.fit(X[tr], y[tr])
        pred = np.clip(pipe.predict(X[te]), 1e-9, None)
        null = np.full(te.size, max(y[tr].mean(), 1e-9))
        dev_m += mean_poisson_deviance(y[te], pred) * te.size
        dev_0 += mean_poisson_deviance(y[te], null) * te.size
    return 1.0 - dev_m / dev_0


def fit_coef(X, y, alpha):
    """Standardized Poisson weights (per feature) for one target type on all data."""
    pipe = make_pipeline(StandardScaler(),
                         PoissonRegressor(alpha=alpha, max_iter=1000))
    pipe.fit(X, y)
    return pipe.named_steps["poissonregressor"].coef_


# -----------------------------------------------------------------------------
# Figure: transfer-matrix heatmap + per-type predictability
# -----------------------------------------------------------------------------
def render(coef, names, type_order, d2_full, d2_base, window_s, dates):
    fig = plt.figure(figsize=(12.5, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.5, 1])

    # --- coefficient heatmap (features x next-window type) --------------------
    ax = fig.add_subplot(gs[0, 0])
    vmax = np.abs(coef).max() or 1.0
    im = ax.imshow(coef.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels([f"next {t}" for t in type_order], rotation=30, ha="right", fontsize=8)
    for x, t in enumerate(type_order):
        ax.get_xticklabels()[x].set_color(TYPE_COLORS.get(t, "black"))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):
        for j in range(len(type_order)):
            ax.text(j, i, f"{coef[j, i]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(f"count transfer weights (per 1 SD)  W={window_s:g}s\n"
                 "row = this window, col = next window", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Poisson coef")

    # --- per-type predictability ----------------------------------------------
    ax = fig.add_subplot(gs[0, 1])
    xpos = np.arange(len(type_order))
    ax.bar(xpos - 0.2, d2_base, width=0.4, color="#9AA0A6", label="time+loc")
    ax.bar(xpos + 0.2, d2_full, width=0.4,
           color=[TYPE_COLORS.get(t, "#457B9D") for t in type_order], label="+counts")
    ax.set_xticks(xpos)
    ax.set_xticklabels(type_order, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("held-out Poisson D$^2$")
    ax.set_title("next-window count predictability", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(f"Next-window per-type call counts  (dates: {', '.join(dates)})  "
                 f"W={window_s:g}s", fontsize=13)
    return fig


# -----------------------------------------------------------------------------
# Orchestrate
# -----------------------------------------------------------------------------
def run(dates, out_dir, fmt, window_s, alpha):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    prev, tod, loc, Y, groups = build_window_design(calls, BOUT_CALL_TYPES, window_s)
    nt = len(BOUT_CALL_TYPES)
    print(f"binned design: {Y.shape[0]:,} windows x {nt} types; "
          f"mean counts/window " + ", ".join(
              f"{t} {Y[:, i].mean():.2f}" for i, t in enumerate(BOUT_CALL_TYPES))
          + f"; {np.unique(groups).size} date/exp groups")

    X_full = np.hstack([prev, tod, loc])                # counts + time + loc
    X_base = np.hstack([tod, loc])                      # time + loc only
    coef = np.zeros((nt, X_full.shape[1]))              # [target_type, feature]
    d2_full, d2_base = [], []
    for j, t in enumerate(BOUT_CALL_TYPES):
        d2_full.append(cv_d2(X_full, Y[:, j], groups, alpha))
        d2_base.append(cv_d2(X_base, Y[:, j], groups, alpha))
        coef[j] = fit_coef(X_full, Y[:, j], alpha)
        print(f"  next {t:10s}: D2 {d2_full[-1]:.4f}  (time+loc only {d2_base[-1]:.4f}, "
              f"gain {d2_full[-1]-d2_base[-1]:+.4f})")

    names = [f"n_{t}" for t in BOUT_CALL_TYPES] + ["sin_tod", "cos_tod", "underground"]
    fig = render(coef, names, BOUT_CALL_TYPES, np.array(d2_full), np.array(d2_base),
                 window_s, dates)
    tag = "+".join(dates)
    _save_and_export(fig, out_dir / f"next_window_counts_{tag}_W{window_s:g}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--window", type=float, default=DEFAULT_WINDOW_S,
                    help="window width (s) whose per-type counts are predicted")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="L2 strength for PoissonRegressor")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.window, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
