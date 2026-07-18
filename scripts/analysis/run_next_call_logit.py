"""Softmax (multinomial logistic) model of the next call type.

==============================================================================
WHAT THIS SCRIPT ANSWERS
==============================================================================
The transition figures (run_transition_prob_by_gap*.py) are *descriptive*: they
show P(next type | current type, tau) as curves. This script asks the same idea
as a *prediction*:

    Given the recent acoustic context, the time of day, and the location,
    what type is the next call?

We treat EVERY call as one prediction target. For a target call that starts at
time t, we look back over the window [t - S, t) and count how many calls of each
type happened -- but only calls at the *same location* and inside the *same
recording block*, so a window never spans a recording gap or the arena/burrow
boundary. Those counts, plus a time-of-day angle and a location flag, are the
inputs. The label is the target call's own type.

==============================================================================
THE MODEL (this is exactly what your spec described)
==============================================================================
Multinomial logistic regression = softmax regression. For feature vector x and
call type k:

        exp(w_k . x + b_k)
    P(next = k | x) = ------------------------          (softmax over the 4 types)
                      sum_j exp(w_j . x + b_j)

  * w_k  = the weight vector for type k (one number per feature)
  * b_k  = the per-type intercept == the "bias / baseline rate" for type k
  * There is no separate multi_class argument to set: scikit-learn's
    LogisticRegression is multinomial by default whenever y has >2 classes.
    (The old multi_class kwarg is deprecated in 1.8/1.9.)

FEATURE VECTOR x (7 numbers per call):
    n_warble, n_high-freq, n_alarm, n_stacks   counts in the last S s (same loc)
    sin_tod, cos_tod                           time of day, as sin/cos of an angle
    underground                                1 if underground else 0 (arena=ref)

==============================================================================
WHAT IT PRODUCES  (one figure, four panels)
==============================================================================
  1-2. TWO coefficient heatmaps side by side, on a shared colour scale: one fit
       at the best-accuracy window S, one at the largest S in the sweep. Cells are
       the standardized weights w_k (effect on each next-type's log-odds per 1 SD
       of each feature), so short-vs-long window structure is directly comparable.
  3. Accuracy vs window S   -- cross-validated, against a base-rate baseline.
  4. Log-loss vs window S   -- cross-validated, against the same baseline.
  The two S-swept panels tick at the exact windows evaluated, and mark the two
  heatmap windows (green dotted = best accuracy, grey dotted = largest).

The cross-validation folds are grouped by (date_folder, exp): a whole recording
block is either all-train or all-test, never split. That matters because calls
close in time share overlapping windows and are heavily autocorrelated -- if a
block straddled the split, the test score would be inflated by leakage.

==============================================================================
SCIKIT-LEARN CONCEPTS  (skip if you know sklearn)
==============================================================================
* THE DATA SHAPE. Every sklearn model takes X and y:
      X : 2-D array [n_samples, n_features].  Here one ROW = one call (every call
          is a prediction target), and the COLUMNS are its 7 features. The calls
          that fell inside a target's look-back window do NOT get their own rows --
          they are collapsed into that row's 4 count features. So 842k calls ->
          842k rows x 7 columns.
      y : 1-D array [n_samples].  The label for each row = that call's own type.

* THE MODEL. LogisticRegression learns a linear score  w_k . x + b_k  per class k,
  then softmax turns the 4 scores into 4 probabilities that sum to 1. Fitting =
  choosing the weights w_k (one per feature per class) and biases b_k that make the
  observed labels most likely. A positive w means "more of this feature -> higher
  odds of that next type". With >2 classes it is multinomial automatically -- the
  old multi_class argument is deprecated and we do not set it.

* THE API. Every estimator is  fit(X, y)  then query:
      predict(X)        -> hard class labels
      predict_proba(X)  -> the softmax probabilities   (we use this)
      .coef_            -> weights, shape [n_classes, n_features]
      .intercept_       -> the biases b_k
      .classes_         -> which label each coef_ row / proba column means.
  GOTCHA: .coef_ rows follow .classes_ (sorted labels), not our preferred order --
  fit_full() re-aligns them so the heatmap rows never silently shuffle.

* REGULARIZATION (C). LogisticRegression penalizes large weights (L2) by default.
  C is the INVERSE strength: large C = weak penalty, small C = heavy shrink to 0.

* SCALING + PIPELINE. StandardScaler z-scores each feature (mean 0, sd 1) so the
  penalty is fair across features and weights read as "effect per 1 SD".
  make_pipeline(scaler, model) chains them so the scaler is fit on the TRAIN fold
  only -- fitting it on all data would leak test statistics into training.

* CROSS-VALIDATION + LEAKAGE. To measure generalization we train on some folds and
  test on a held-out one. We use GroupKFold with group = (date, exp) so a whole
  recording block is all-train or all-test. Two calls seconds apart have nearly
  identical windows; letting them straddle the split would leak the answer and
  inflate the score. This is the single most important choice in the script.

* METRICS. accuracy = fraction with the right top class (blunt: always-guess-warble
  already scores ~0.49). log_loss = rewards calibrated probabilities, punishes
  confident wrong answers, lower is better. Both are shown vs a base-rate baseline.

==============================================================================
USAGE
==============================================================================
    python scripts/analysis/run_next_call_logit.py                       # all dates pooled
    python scripts/analysis/run_next_call_logit.py --per-date --format png
    python scripts/analysis/run_next_call_logit.py --dates 2026_02 --s-sweep 1 5 30 300
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")                        # headless backend: render to file, never a window

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- make the repo's shared modules importable regardless of CWD -------------
REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Reuse the transition script's building blocks so both analyses stay consistent:
#   BOUT_CALL_TYPES  the 4 modelled types, in a fixed order  -> index 0..3
#   LOC_GROUPS       maps assigned_location -> "arena" / "underground"
#   TYPE_COLORS      the shared per-type colour scheme (for tick labels)
#   load_calls       loads + cleans the pooled per-date call tables
#   _save_and_export saves the figure and drops a copy under exports/
from run_transition_prob_by_gap import (  # noqa: E402
    BOUT_CALL_TYPES, DEFAULT_DATES as _ALL_DATES, LOC_GROUPS, TYPE_COLORS,
    _save_and_export, load_calls,
)
from ethogram_io import BASE_PROCESSED  # noqa: E402

# 2025_03 is a ~2-day experiment (vs weeks for the others) with only 2 exp
# segments -- too short to be comparable or to cross-validate cleanly, so it is
# excluded here. Pass it explicitly via --dates 2025_03 if you ever want it.
DEFAULT_DATES = [d for d in _ALL_DATES if d != "2025_03"]

# === constants ===============================================================
NS_PER_S = 1_000_000_000                      # pandas datetimes are int64 nanoseconds
NS_PER_DAY = 86_400 * NS_PER_S
GROUP_COLS = ["date_folder", "exp"]           # a recording block; windows/CV folds never cross it
# Windows scored in the accuracy/log-loss panels. Dense at the short end because the
# predictive signal lives there (within-bout repetition); below ~0.5s the window is
# mostly empty and only measures the time+location floor. The two coefficient
# heatmaps are fit at the best-accuracy S and the largest S, both taken from this list.
DEFAULT_S_SWEEP = [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 300.0]
DEFAULT_C = 1.0                               # inverse L2 strength (larger C = weaker regularization)
N_SPLITS = 5                                  # cross-validation folds
# =============================================================================


# -----------------------------------------------------------------------------
# STEP 1 - count calls in each target's look-back window
# -----------------------------------------------------------------------------
def window_counts(start_ns, code, target_ns, n_types, window_ns):
    """For each target call, count how many prior calls of each type fall in [t - S, t).

    Inputs (all for a SINGLE (date, exp, location) block, sorted by start time):
        start_ns   int64[n]  start time of every call, ascending
        code       int[n]    type index (0..n_types-1) of every call
        target_ns  int64[n]  the target start times (same array as start_ns here)
        window_ns  int       window length S, in nanoseconds

    Returns float[n, n_types]: row i = the per-type counts seen just before call i.

    How the counting works, per type k, with no Python loop over calls:
        ts = the sorted start times of just the type-k calls.
        hi = searchsorted(ts, t,        "left")  -> # of type-k calls with start <  t
        lo = searchsorted(ts, t - S,    "left")  -> # of type-k calls with start <  t - S
        hi - lo                                  -> # with start in [t - S, t)
    "left" means an entry exactly equal to t is placed to its LEFT, i.e. NOT
    counted -- so a call never counts itself or another call at the identical
    timestamp. searchsorted is vectorised over all targets at once (O(n log n)).
    """
    out = np.zeros((target_ns.size, n_types), dtype=np.float64)
    for k in range(n_types):
        ts = start_ns[code == k]                       # start times of type-k calls only
        if ts.size == 0:
            continue                                   # this type never occurs in this block
        hi = np.searchsorted(ts, target_ns, side="left")               # start <  t
        lo = np.searchsorted(ts, target_ns - window_ns, side="left")   # start <  t - S
        out[:, k] = hi - lo                            # count in the half-open window
    return out


# -----------------------------------------------------------------------------
# STEP 2 - assemble the full design matrix X, labels y, and CV groups
# -----------------------------------------------------------------------------
def build_design(calls, type_order, window_s, switches_only=False):
    """Turn the call table into (X, feature_names, y, groups) for scikit-learn.

    One row per call. We process each (date, exp, location) block independently so
    that windows respect recording gaps and the arena/burrow split, then stack all
    the blocks together at the end.

    switches_only : if True, keep only rows where the call's type DIFFERS from the
        immediately preceding call in its block -- i.e. across-type transitions,
        dropping self-continuations. The label is then the switch DESTINATION, and
        the chance level becomes the marginal over destinations. This is a purely
        data-defined "switch" (no bout threshold); it isolates the off-diagonal
        transition syntax that self-repetition otherwise dominates.

    Returns:
        X       float[N, 7]   feature matrix (counts + sin/cos tod + underground)
        names   list[str]     the 7 feature names, in column order
        y       int[N]        label = the call's own type index
        groups  str[N]        "date/exp" tag per row, used to keep CV folds clean
    """
    code_of = {t: i for i, t in enumerate(type_order)}   # "warble" -> 0, ... fixed order
    nt = len(type_order)

    # keep only the modelled types, and attach the coarse arena/underground label
    ev = calls[calls["event_type"].isin(type_order)].copy()
    ev["_loc2"] = ev["assigned_location"].map(LOC_GROUPS)
    ev = ev.dropna(subset=["_loc2"])                     # drop calls at unmapped locations

    # per-block feature pieces, collected then stacked (cheaper than growing arrays)
    count_blocks, tod_blocks, loc_blocks, y_blocks, grp_blocks = [], [], [], [], []
    window_ns = int(window_s * NS_PER_S)

    for (date, exp, loc), g in ev.groupby(GROUP_COLS + ["_loc2"]):
        g = g.sort_values("start_time_real")             # windows need ascending time
        start_ns = g["start_time_real"].to_numpy().astype("int64")   # ns since epoch
        code = g["event_type"].map(code_of).to_numpy()               # type index per call

        # (a) the look-back per-type counts -- the acoustic-context features
        counts = window_counts(start_ns, code, start_ns, nt, window_ns)

        # (b) time of day as a smooth angle. tod is the fraction of the day [0,1);
        #     2*pi*tod is the clock angle; sin+cos encode it WITHOUT the midnight
        #     discontinuity you'd get from using the raw hour as a number.
        tod = (start_ns % NS_PER_DAY) / NS_PER_DAY
        ang = 2 * np.pi * tod
        tod2 = np.column_stack([np.sin(ang), np.cos(ang)])

        # (c) location flag: 1 for underground, 0 for arena (arena is the reference level)
        loc_col = np.full(g.shape[0], 1.0 if loc == "underground" else 0.0)
        grp = np.full(g.shape[0], f"{date}/{exp}")

        # (d) optionally keep only across-type transitions (type != previous type).
        #     row 0 has no predecessor, so it is never a switch.
        if switches_only:
            keep = np.zeros(code.size, dtype=bool)
            keep[1:] = code[1:] != code[:-1]
            counts, tod2, loc_col, code, grp = (
                counts[keep], tod2[keep], loc_col[keep], code[keep], grp[keep])

        count_blocks.append(counts)
        tod_blocks.append(tod2)
        loc_blocks.append(loc_col)
        y_blocks.append(code)                            # label = this call's type
        grp_blocks.append(grp)

    # glue the blocks into one matrix. column order here defines `names` below.
    X = np.column_stack([
        np.vstack(count_blocks),                 # 4 count columns
        np.vstack(tod_blocks),                   # sin_tod, cos_tod
        np.concatenate(loc_blocks)[:, None],     # underground
    ])
    y = np.concatenate(y_blocks)
    groups = np.concatenate(grp_blocks)
    names = [f"n_{t}" for t in type_order] + ["sin_tod", "cos_tod", "underground"]
    return X, names, y, groups


# -----------------------------------------------------------------------------
# STEP 3 - fit the softmax model on ALL rows, for the coefficient picture
# -----------------------------------------------------------------------------
def fit_full(X, y, type_order, C):
    """Fit softmax regression on every row and return (pipeline, coef[type, feature]).

    Pipeline = StandardScaler -> LogisticRegression:
      * StandardScaler z-scores each feature (mean 0, sd 1). Two reasons: L2
        regularization is fairer when features share a scale, and the resulting
        weights read as "effect per 1 SD", comparable across features.
      * LogisticRegression with 4 classes IS the softmax model. fit_intercept is
        True by default, so it learns the per-type bias b_k for us.

    scikit-learn stores weights in clf.coef_ ordered by clf.classes_ (the sorted
    labels actually seen). We copy each row into a fixed [type_order] layout so the
    heatmap rows always mean the same type even if some class were ever missing.
    """
    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(C=C, max_iter=2000))
    pipe.fit(X, y)
    clf = pipe.named_steps["logisticregression"]

    coef = np.zeros((len(type_order), X.shape[1]))       # [n_types, n_features]
    for row, cls in enumerate(clf.classes_):             # cls is a type index (0..3)
        coef[cls] = clf.coef_[row]                       # align to our fixed type order
    return pipe, coef


# -----------------------------------------------------------------------------
# STEP 4 - honest predictive score via grouped cross-validation
# -----------------------------------------------------------------------------
def cv_scores(X, y, groups, C, n_splits=N_SPLITS):
    """Grouped-CV accuracy & log-loss for the model AND a base-rate baseline.

    GroupKFold puts every row of a given (date, exp) block entirely in train OR
    test -- never both. Without this, two calls a few seconds apart (with almost
    identical windows) could land on opposite sides of the split and leak the
    answer, making the model look better than it is.

    Baseline = "know nothing but the overall frequencies": predict the TRAIN-set
    class marginal for every test row.
      * accuracy  -> always guess the single most common train class
      * log-loss  -> score the full marginal distribution (a fair probabilistic ref)

    Returns the mean-over-folds of the four numbers.
    """
    classes = np.unique(y)
    # GroupKFold needs n_splits <= number of groups; per-family some dates have only
    # a couple of exp segments (e.g. 2025_03 has 2), so cap the fold count to fit.
    n_groups = np.unique(groups).size
    n_splits = min(n_splits, n_groups)
    gkf = GroupKFold(n_splits=n_splits)
    acc_m, acc_b, ll_m, ll_b = [], [], [], []            # model/baseline accuracy & log-loss

    for tr, te in gkf.split(X, y, groups):
        # --- the model ---
        pipe = make_pipeline(StandardScaler(),
                             LogisticRegression(C=C, max_iter=2000))
        pipe.fit(X[tr], y[tr])
        # align proba columns to the GLOBAL class set: if a rare class (alarm) is
        # absent from this train fold, predict_proba omits it -- fill that column
        # with 0 so log_loss's label axis still lines up (it clips 0 internally).
        raw = pipe.predict_proba(X[te])                  # [n_test, n_train_classes]
        proba = np.zeros((te.size, classes.size))
        col = {c: j for j, c in enumerate(classes)}
        for j, c in enumerate(pipe.classes_):
            proba[:, col[c]] = raw[:, j]
        acc_m.append(accuracy_score(y[te], classes[proba.argmax(1)]))
        ll_m.append(log_loss(y[te], proba, labels=classes))

        # --- the base-rate baseline (fit only the train marginal) ---
        marg = np.array([(y[tr] == c).mean() for c in classes])   # train class frequencies
        base_proba = np.repeat(marg[None], te.size, axis=0)       # same row for every test call
        acc_b.append(accuracy_score(y[te], np.full(te.size, classes[marg.argmax()])))
        ll_b.append(log_loss(y[te], base_proba, labels=classes))

    return {"acc_model": np.mean(acc_m), "acc_base": np.mean(acc_b),
            "ll_model": np.mean(ll_m), "ll_base": np.mean(ll_b)}


# -----------------------------------------------------------------------------
# STEP 5 - draw the figure: two coefficient heatmaps + accuracy + log-loss
# -----------------------------------------------------------------------------
def _style_S_axis(ax, Ss):
    """Log x-axis whose ticks sit exactly on the S values we actually evaluated.

    matplotlib's default log ticks are decade marks (10, 100, ...); here we replace
    them with one tick per swept window (0.5, 1, 2, ..., 300) so the reader sees the
    real sampling, and we drop the minor decade ticks that would otherwise clutter.
    """
    from matplotlib.ticker import NullLocator
    ax.set_xscale("log")
    ax.set_xticks(Ss)
    ax.set_xticklabels([f"{s:g}" for s in Ss], rotation=45, fontsize=7)
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xlabel("window S (s)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _heatmap(ax, coef, names, type_order, vmax, title):
    """One coefficient heatmap (features x next-type) on a shared colour scale."""
    im = ax.imshow(coef.T, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(type_order)))
    ax.set_xticklabels([f"next = {t}" for t in type_order], rotation=30, ha="right", fontsize=8)
    for x, t in enumerate(type_order):                   # colour each column label by its type
        ax.get_xticklabels()[x].set_color(TYPE_COLORS.get(t, "black"))
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    for i in range(len(names)):                          # print the numeric weight in each cell
        for j in range(len(type_order)):
            ax.text(j, i, f"{coef[j, i]:+.2f}", ha="center", va="center", fontsize=6.5)
    ax.set_title(title, fontsize=10)
    return im


def render(coef_best, S_best, coef_large, S_large, names, type_order, sweep, dates,
           switches_only=False):
    """Two heatmaps (best-accuracy S vs largest S) + accuracy-vs-S + log-loss-vs-S.

    The two heatmaps share one symmetric colour scale (vmax over both) so a cell's
    colour means the same weight in each -- you can compare short vs long window
    directly. The right two panels mark both chosen windows with dotted lines.
    """
    fig = plt.figure(figsize=(18, 4.9), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 1.35, 1.05, 1.05])
    acc_of = {s["window_s"]: s["acc_model"] for s in sweep}
    vmax = max(np.abs(coef_best).max(), np.abs(coef_large).max()) or 1.0

    # --- panels 1 & 2: the two coefficient heatmaps ---------------------------
    ax0 = fig.add_subplot(gs[0, 0])
    _heatmap(ax0, coef_best, names, type_order, vmax,
             f"best accuracy:  S={S_best:g}s   (acc {acc_of[S_best]:.3f})")
    ax1 = fig.add_subplot(gs[0, 1])
    im = _heatmap(ax1, coef_large, names, type_order, vmax,
                  f"largest window:  S={S_large:g}s   (acc {acc_of[S_large]:.3f})")
    fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04, label="coef (per 1 SD)")

    Ss = [s["window_s"] for s in sweep]

    def _mark(ax):                                        # dotted lines at the two heatmap windows
        ax.axvline(S_best, color="#2A9D8F", ls=":", lw=1.1)
        ax.axvline(S_large, color="0.4", ls=":", lw=1.1)

    # --- panel 3: accuracy vs window S ----------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    ax.plot(Ss, [s["acc_model"] for s in sweep], "o-", color="#264653", label="model")
    ax.plot(Ss, [s["acc_base"] for s in sweep], "s--", color="gray", label="base rate")
    _mark(ax)
    _style_S_axis(ax, Ss)
    ax.set_ylabel("CV accuracy"); ax.set_title("accuracy vs window", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    # --- panel 4: log-loss vs window S (lower is better) ----------------------
    ax = fig.add_subplot(gs[0, 3])
    ax.plot(Ss, [s["ll_model"] for s in sweep], "o-", color="#264653", label="model")
    ax.plot(Ss, [s["ll_base"] for s in sweep], "s--", color="gray", label="base rate")
    _mark(ax)
    _style_S_axis(ax, Ss)
    ax.set_ylabel("CV log-loss (lower better)"); ax.set_title("log-loss vs window", fontsize=10)
    ax.legend(fontsize=8, frameon=False)

    mode = "  [SWITCHES ONLY: across-type transitions]" if switches_only else ""
    fig.suptitle(f"Next-call softmax model  (dates: {', '.join(dates)})   "
                 f"[green dotted = best-acc S, grey dotted = largest S]{mode}", fontsize=13)
    return fig


# -----------------------------------------------------------------------------
# STEP 6 - orchestrate one run (load -> sweep CV -> fit 2 heatmaps -> draw -> save)
# -----------------------------------------------------------------------------
def run(dates, out_dir, fmt, s_sweep, C, switches_only=False):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}"
          + ("  [switches only]" if switches_only else ""))

    # (a) CV score at every window in the sweep -> the accuracy/log-loss curves
    sweep = []
    for S in sorted(set(s_sweep)):
        Xs, _, ys, gs = build_design(calls, BOUT_CALL_TYPES, S, switches_only)
        sc = cv_scores(Xs, ys, gs, C)
        sc["window_s"] = S
        sweep.append(sc)
        print(f"  S={S:6.1f}s  acc {sc['acc_model']:.3f} (base {sc['acc_base']:.3f})  "
              f"logloss {sc['ll_model']:.3f} (base {sc['ll_base']:.3f})")

    # (b) pick the two windows to show as coefficient heatmaps
    S_best = max(sweep, key=lambda s: s["acc_model"])["window_s"]   # best CV accuracy
    S_large = max(s["window_s"] for s in sweep)                     # longest window
    print(f"  -> heatmaps at S_best={S_best:g}s (best acc) and S_large={S_large:g}s (largest)")

    # (c) fit coefficients on ALL data at each of those two windows
    Xb, names, yb, _ = build_design(calls, BOUT_CALL_TYPES, S_best, switches_only)
    print(f"  {yb.size:,} target rows; class balance @S_best: "
          + ", ".join(f"{t} {(yb == i).mean():.3f}" for i, t in enumerate(BOUT_CALL_TYPES)))
    _, coef_best = fit_full(Xb, yb, BOUT_CALL_TYPES, C)
    Xl, _, yl, _ = build_design(calls, BOUT_CALL_TYPES, S_large, switches_only)
    _, coef_large = fit_full(Xl, yl, BOUT_CALL_TYPES, C)

    # (d) draw and save (also copied under exports/ by _save_and_export)
    fig = render(coef_best, S_best, coef_large, S_large, names, BOUT_CALL_TYPES, sweep, dates,
                 switches_only)
    tag = "+".join(dates)
    suffix = "_switches" if switches_only else ""
    _save_and_export(fig, out_dir / f"next_call_logit_{tag}{suffix}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--s-sweep", type=float, nargs="+", default=DEFAULT_S_SWEEP,
                    help="window lengths (s) to evaluate; heatmaps use best-acc & largest")
    ap.add_argument("--C", type=float, default=DEFAULT_C,
                    help="inverse L2 strength for LogisticRegression (larger = weaker)")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates")
    ap.add_argument("--switches-only", action="store_true",
                    help="model only across-type transitions (drop self-continuations); "
                         "predict the switch destination")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.s_sweep, args.C, args.switches_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
