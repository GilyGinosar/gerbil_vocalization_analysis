"""All-pairs transition composition vs separation (tau), split by location.

Sibling of run_transition_prob_by_gap.py. That script conditions on the
*immediate next* call: given a call of type X whose very next call arrives after
silent gap tau, what type is it? This script drops the "immediate next"
condition and instead asks, for EVERY later call at separation ~ tau (any number
of calls may occur in between):

    of all calls occurring ~tau after a call of type X, what fraction are type Y?

i.e. P(some call at separation tau is Y | a call now is X), resolved by tau.
Same grid layout as the consecutive figure (rows = arena/underground, cols =
current type X, lines = next type Y) so the two are directly comparable.

Why compare the two:
  * This all-pairs curve CONVERGES to the base rate at large tau -- far-apart
    calls are independent, so the composition at separation tau relaxes to the
    overall marginal P(Y). It therefore shows the *timescale* over which "who
    just called" stops predicting "who calls next".
  * The consecutive curve does NOT converge to base rate at large tau, because a
    large *consecutive* gap means no call happened for tau seconds -- that
    conditions on lulls. The gap between the two figures at a given tau measures
    how much that "nothing in between" condition distorts the picture.

Note on the x-axis: here tau is a *separation* (calls occur in between), NOT a
silent gap as in the consecutive figure. The axis is relabelled accordingly.

Pairing respects the same boundaries as the consecutive figure
(date_folder, exp, assigned_location): pairs are never formed across them. Within
a group every call is paired with all later calls whose separation
tau = later.start - earlier.stop is in (0, --max-sep-s]. Caveat: a group may
span a recording gap, so a handful of pairs at large tau can straddle
non-recorded time; at large tau their composition is ~base rate anyway, so this
diffuses into the asymptote rather than creating false structure.

Usage:
    python scripts/analysis/run_transition_prob_by_gap_allpairs.py --dates 2026_02 --format png
    python scripts/analysis/run_transition_prob_by_gap_allpairs.py            # all dates pooled
    python scripts/analysis/run_transition_prob_by_gap_allpairs.py --per-date --max-sep-s 3600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import BASE_PROCESSED  # noqa: E402
from run_transition_prob_by_gap import (  # noqa: E402
    BOUT_CALL_TYPES, DEFAULT_DATES, LOC_GROUPS, MIN_COUNT, TAU_BINS,
    _save_and_export, load_calls, render_grid_from_counts,
)

NS_PER_S = 1_000_000_000
GROUP_COLS = ["date_folder", "exp"]        # location handled by the outer split
DEFAULT_MAX_SEP_S = 3600.0                 # longest separation to count pairs over (s)
SRC_BLOCK = 4096                           # sources per vectorised chunk (bounds peak memory)


def _accumulate_group(start_ns, stop_ns, code, tau_bins, max_sep_ns, counts):
    """Add every forward pair within max_sep to the (curr, bin, next) count cube.

    start_ns must be sorted ascending. For each source i, the eligible targets j
    are those whose start lands in [stop_i, stop_i + max_sep]; each contributes a
    pair with tau = start_j - stop_i. Done in source-blocks so the materialised
    per-block pair arrays stay bounded regardless of group size.
    """
    n = start_ns.size
    if n < 2:
        return
    nb, nt = counts.shape[1], counts.shape[2]
    lo_all = np.searchsorted(start_ns, stop_ns, side="left")          # first j with start_j >= stop_i
    hi_all = np.searchsorted(start_ns, stop_ns + max_sep_ns, side="right")
    for b0 in range(0, n, SRC_BLOCK):
        b1 = min(b0 + SRC_BLOCK, n)
        lo, hi = lo_all[b0:b1], hi_all[b0:b1]
        cnt = hi - lo
        total = int(cnt.sum())
        if total == 0:
            continue
        src = np.arange(b0, b1)
        i_idx = np.repeat(src, cnt)
        # j index for each pair: lo_i + (running offset within that source's window)
        offsets = np.arange(total) - np.repeat(np.cumsum(cnt) - cnt, cnt)
        j_idx = np.repeat(lo, cnt) + offsets
        tau_s = (start_ns[j_idx] - stop_ns[i_idx]) / NS_PER_S
        bins = np.digitize(tau_s, tau_bins) - 1
        keep = (tau_s > 0) & (bins >= 0) & (bins < nb)
        if not keep.any():
            continue
        flat = (code[i_idx[keep]] * nb + bins[keep]) * nt + code[j_idx[keep]]
        counts.reshape(-1)[:] += np.bincount(flat, minlength=counts.size)


def all_pairs_counts_by_loc(calls, type_order, tau_bins, max_sep_s, group_cols=GROUP_COLS):
    """{loc -> count cube [n_curr, n_bins, n_next]} and {loc -> base rate [n_next]}.

    base[y] = fraction of all calls in the location that are type_order[y] -- the
    asymptote the all-pairs composition relaxes to as tau grows.
    """
    code_of = {t: i for i, t in enumerate(type_order)}
    nt, nb = len(type_order), len(tau_bins) - 1
    max_sep_ns = int(max_sep_s * NS_PER_S)
    ev = calls[calls["event_type"].isin(type_order)].copy()
    ev["_loc2"] = ev["assigned_location"].map(LOC_GROUPS)

    counts_by_loc, base_by_loc = {}, {}
    for loc in ("arena", "underground"):
        sub = ev[ev["_loc2"] == loc]
        counts = np.zeros((nt, nb, nt), dtype=np.float64)
        base_by_loc[loc] = (sub["event_type"].value_counts(normalize=True)
                            .reindex(type_order, fill_value=0).to_numpy())
        n_pairs_before = counts.sum()
        for _, g in sub.groupby(group_cols):
            g = g.sort_values("start_time_real")
            start_ns = g["start_time_real"].to_numpy().astype("int64")
            stop_ns = g["stop_time_real"].to_numpy().astype("int64")
            code = g["event_type"].map(code_of).to_numpy()
            _accumulate_group(start_ns, stop_ns, code, tau_bins, max_sep_ns, counts)
        counts_by_loc[loc] = counts
        print(f"  {loc}: {len(sub):,} calls -> {int(counts.sum() - n_pairs_before):,} forward pairs "
              f"(<= {max_sep_s:.0f}s)")
    return counts_by_loc, base_by_loc


def run(dates, out_dir, fmt, min_count, max_sep_s, baseline="calls"):
    calls = load_calls(dates)
    print(f"{len(calls):,} calls pooled across {dates}")
    out_dir.mkdir(parents=True, exist_ok=True)
    counts_by_loc, base_by_loc = all_pairs_counts_by_loc(
        calls, BOUT_CALL_TYPES, TAU_BINS, max_sep_s)
    if baseline == "transitions":
        # pair-weighted marginal: fraction of all forward pairs ending in each type
        # (weights each target call by how many sources precede it within max_sep).
        base_by_loc = {loc: (c.sum(axis=(0, 1)) / c.sum() if c.sum() else c.sum(axis=(0, 1)))
                       for loc, c in counts_by_loc.items()}
    suffix = "_transbase" if baseline == "transitions" else ""
    tag = "+".join(dates)
    title = ("All-pairs call composition at separation tau  "
             f"(dates: {', '.join(dates)})"
             + ("  [chance = pair marginal]" if baseline == "transitions"
                else "  [chance = call abundance]"))
    fig = render_grid_from_counts(
        counts_by_loc, base_by_loc, BOUT_CALL_TYPES, title,
        tau_bins=TAU_BINS, min_count=min_count,
        ylabel_stat="P(type at sep tau | current)", xlabel="separation tau (s)")
    _save_and_export(fig, out_dir / f"transition_prob_by_gap_allpairs_call_{tag}{suffix}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="pdf")
    ap.add_argument("--min-count", type=int, default=MIN_COUNT,
                    help="blank a tau bin with fewer pairs than this")
    ap.add_argument("--max-sep-s", type=float, default=DEFAULT_MAX_SEP_S,
                    help="longest separation tau to count pairs over (s); larger = slower")
    ap.add_argument("--baseline", choices=["calls", "transitions"], default="calls",
                    help="chance line: 'calls' = raw call abundance n_Y/N (default, the "
                         "true asymptote); 'transitions' = pair-weighted marginal")
    ap.add_argument("--per-date", action="store_true",
                    help="one figure per date instead of pooling all dates together")
    args = ap.parse_args()
    date_groups = [[d] for d in args.dates] if args.per_date else [args.dates]
    for dates in date_groups:
        run(dates, args.out_dir, args.format, args.min_count, args.max_sep_s, args.baseline)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
