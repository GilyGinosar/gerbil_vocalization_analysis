"""Self-excitation of calling: predict the number of calls in the next time-bin.

The next-call models ask "given a call happens, which type?". This asks the more
basic temporal question: do calls beget calls, and over what timescale? It is a
point-process / Hawkes-style view, and it is immune to call-type confusion because
the target is TOTAL call activity, not a type label.

METHOD
  * Bin each (date, exp, location) recording block into fixed Delta-second bins and
    count calls per bin (all types pooled -> total activity).
  * For each target bin, the features are the counts in the preceding L bins
    (lags 1..L, i.e. an autoregressive history), plus time-of-day (sin/cos) and a
    location flag.
  * Model the count with POISSON regression (log link):
        E[count_now] = exp( b0 + sum_k w_k * count_{k bins ago} + time + location )
    A positive lag weight w_k = "more calls k bins ago -> higher rate now" = self-
    excitation. The weights vs lag time ARE the excitation kernel; its decay is the
    memory timescale of calling.

NESTED MODELS (does excitation add beyond the circadian rhythm?)
    H1  time-of-day + location .......... the known circadian/spatial rate structure
    H2  H1 + excitation kernel (lags) ... does recent calling predict more calling
                                          ON TOP OF time of day?
  Scored by leakage-free GroupKFold (grouped by date/exp) using the Poisson
  deviance D^2 (fraction of deviance explained vs a constant-rate null). The
  H1 -> H2 gain is the self-excitation signal beyond circadian.

CAVEAT: bins tile each block from its first to its last call, assuming the block is
continuously recorded (exp segments are contiguous runs). A long unrecorded gap
inside a block would inject false zero-count bins; if that becomes a concern, mask
bins by recording coverage (file_times.csv) -- not done here.

Usage:
    python scripts/analysis/run_call_rate_excitation.py --per-date --format png
    python scripts/analysis/run_call_rate_excitation.py --dates 2026_02 --delta 5 --n-lags 24
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
    _save_and_export, load_calls,
)
from ethogram_io import BASE_PROCESSED  # noqa: E402

DEFAULT_DELTA_S = 5.0          # bin width (s) = the "next window" whose count we predict
DEFAULT_N_LAGS = 24            # history depth in bins (24 * 5s = 120s of look-back)
DEFAULT_ALPHA = 1e-2           # L2 strength for PoissonRegressor (smooths collinear lags)
N_SPLITS = 5


# -----------------------------------------------------------------------------
# Build the binned design: lag-count features + time + location, target = count
# -----------------------------------------------------------------------------
def build_rate_design(calls, type_order, delta_s, n_lags):
    """Bin every (date, exp, location) block and assemble the AR-Poisson design.

    Returns:
        y      int[N]          calls in the target bin
        lags   float[N, L]     counts in the previous 1..L bins (column k-1 = lag k)
        tod    float[N, 2]     sin/cos of time of day at the bin start
        loc    float[N, 1]     underground indicator
        groups str[N]          date/exp tag for GroupKFold
        lag_t  float[L]        lag times in seconds (k * delta), for the kernel x-axis
    """
    delta_ns = int(delta_s * NS_PER_S)
    ev = calls[calls["event_type"].isin(type_order)].copy()
    ev["_loc2"] = ev["assigned_location"].map(LOC_GROUPS)
    ev = ev.dropna(subset=["_loc2"])

    Y, LAG, TOD, LOC, GRP = [], [], [], [], []
    for (date, exp, loc), g in ev.groupby(GROUP_COLS + ["_loc2"]):
        starts = np.sort(g["start_time_real"].to_numpy().astype("int64"))
        span = starts[-1] - starts[0]
        nbins = int(span // delta_ns) + 1
        if nbins <= n_lags + 1:
            continue                                    # block too short to have history
        edges = starts[0] + np.arange(nbins + 1) * delta_ns
        counts = np.diff(np.searchsorted(starts, edges)).astype(np.float64)  # calls per bin

        tgt = np.arange(n_lags, nbins)                  # bins that have L bins of history
        Y.append(counts[tgt])
        # lag column k (1..L): the count k bins before each target. log1p-compress the
        # counts: with the Poisson log-link, raw counts give rate = exp(sum w*count),
        # which explodes when a held-out burst exceeds anything seen in training
        # (deviance -> huge, D^2 -> nonsense). log1p makes the rate a tame power law
        # of (1 + count) and keeps cross-validation numerically stable.
        LAG.append(np.log1p(np.column_stack([counts[tgt - k] for k in range(1, n_lags + 1)])))
        bin_start_ns = edges[tgt]                       # time at the start of each target bin
        ang = 2 * np.pi * (bin_start_ns % NS_PER_DAY) / NS_PER_DAY
        TOD.append(np.column_stack([np.sin(ang), np.cos(ang)]))
        LOC.append(np.full(tgt.size, 1.0 if loc == "underground" else 0.0))
        GRP.append(np.full(tgt.size, f"{date}/{exp}"))

    y = np.concatenate(Y)
    lags = np.vstack(LAG)
    tod = np.vstack(TOD)
    loc = np.concatenate(LOC)[:, None]
    groups = np.concatenate(GRP)
    lag_t = np.arange(1, n_lags + 1) * delta_s
    return y, lags, tod, loc, groups, lag_t


# -----------------------------------------------------------------------------
# Fit / score
# -----------------------------------------------------------------------------
def _folds(groups):
    return GroupKFold(n_splits=min(N_SPLITS, np.unique(groups).size))


def cv_d2(X, y, groups, alpha):
    """GroupKFold Poisson D^2 = 1 - deviance(model) / deviance(constant-rate null).

    Deviances are pooled over folds (weighted by test size) so the ratio is a proper
    held-out fraction-of-deviance-explained. Null predicts the TRAIN mean rate.
    """
    dev_m, dev_0, n = 0.0, 0.0, 0
    for tr, te in _folds(groups).split(X, y, groups):
        pipe = make_pipeline(StandardScaler(),
                             PoissonRegressor(alpha=alpha, max_iter=1000))
        pipe.fit(X[tr], y[tr])
        pred = np.clip(pipe.predict(X[te]), 1e-9, None)
        null = np.full(te.size, max(y[tr].mean(), 1e-9))
        dev_m += mean_poisson_deviance(y[te], pred) * te.size
        dev_0 += mean_poisson_deviance(y[te], null) * te.size
        n += te.size
    return 1.0 - dev_m / dev_0


def fit_kernel(lags, tod, loc, y, alpha):
    """Fit H2 on all data; return the standardized lag weights (the excitation kernel)."""
    X = np.hstack([lags, tod, loc])
    pipe = make_pipeline(StandardScaler(),
                         PoissonRegressor(alpha=alpha, max_iter=1000))
    pipe.fit(X, y)
    return pipe.named_steps["poissonregressor"].coef_[:lags.shape[1]]   # lag columns only


# -----------------------------------------------------------------------------
# Figure
# -----------------------------------------------------------------------------
def render(lag_t, kernel, d2_h1, d2_h2, delta_s, mean_rate, dates):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True,
                             gridspec_kw={"width_ratios": [1.6, 1]})

    # --- excitation kernel ----------------------------------------------------
    ax = axes[0]
    ax.axhline(0, color="0.6", lw=0.8)
    ax.plot(lag_t, kernel, "o-", color="#264653", lw=1.6)
    ax.set_xscale("log")
    ax.set_xlabel("lag before the predicted bin (s)")
    ax.set_ylabel("Poisson weight (per 1 SD)")
    ax.set_title(f"excitation kernel  (Delta={delta_s:g}s bins)\n"
                 "positive = recent calls raise the rate now", fontsize=10)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # --- variance explained: circadian vs +excitation -------------------------
    ax = axes[1]
    bars = ax.bar(["H1\ntime+loc", "H2\n+excitation"], [d2_h1, d2_h2],
                  color=["#9AA0A6", "#2A9D8F"])
    ax.bar_label(bars, fmt="%.3f", fontsize=9)
    ax.set_ylabel("held-out Poisson D$^2$")
    ax.set_title(f"deviance explained\n(+excitation gain: {d2_h2 - d2_h1:+.3f})", fontsize=10)
    ax.set_ylim(0, max(d2_h2, 0.01) * 1.25)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle(f"Call-rate self-excitation  (dates: {', '.join(dates)})   "
                 f"mean {mean_rate:.2f} calls/{delta_s:g}s bin", fontsize=13)
    return fig


# -----------------------------------------------------------------------------
# Orchestrate
# -----------------------------------------------------------------------------
def run(dates, out_dir, fmt, delta_s, n_lags, alpha):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    y, lags, tod, loc, groups, lag_t = build_rate_design(calls, BOUT_CALL_TYPES, delta_s, n_lags)
    print(f"binned design: {y.size:,} bins x {n_lags} lags; "
          f"mean {y.mean():.3f} calls/bin ({(y > 0).mean():.1%} of bins have >=1 call); "
          f"{np.unique(groups).size} date/exp groups")

    tl = np.hstack([tod, loc])                          # H1 features
    d2_h1 = cv_d2(tl, y, groups, alpha)
    d2_h2 = cv_d2(np.hstack([lags, tod, loc]), y, groups, alpha)
    print(f"  H1 time+loc      : D2 {d2_h1:.4f}")
    print(f"  H2 +excitation   : D2 {d2_h2:.4f}   (gain {d2_h2 - d2_h1:+.4f})")

    kernel = fit_kernel(lags, tod, loc, y, alpha)
    print("  kernel (per-SD weight by lag s): "
          + ", ".join(f"{t:g}s={w:+.2f}" for t, w in zip(lag_t, kernel)))

    fig = render(lag_t, kernel, d2_h1, d2_h2, delta_s, y.mean(), dates)
    tag = "+".join(dates)
    _save_and_export(fig, out_dir / f"call_rate_excitation_{tag}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--delta", type=float, default=DEFAULT_DELTA_S,
                    help="bin width (s) = the window whose call count is predicted")
    ap.add_argument("--n-lags", type=int, default=DEFAULT_N_LAGS,
                    help="history depth in bins (kernel spans n_lags * delta seconds)")
    ap.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                    help="L2 strength for PoissonRegressor (larger = smoother kernel)")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.delta, args.n_lags, args.alpha)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
