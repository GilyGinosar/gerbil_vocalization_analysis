"""Before- vs after-litter transition figures, overlaid, per family.

Each date folder is a distinct gerbil family, and each of these three families had a
litter born mid-experiment (LITTER_BOUNDARY in run_transition_prob_by_gap.py). This
script re-makes the two transition views we built -- the consecutive one
(run_transition_prob_by_gap.py) and the all-pairs one
(run_transition_prob_by_gap_allpairs.py) -- but splits each family's calls at its
litter boundary and OVERLAYS before (dashed, open markers) and after (solid, filled)
on the same axes, so the change in vocal-sequence structure around the birth is read
off directly. Colour = next-type; line style = period. Chance line = call abundance
*within that period* (so before/after each get their own faint reference).

One figure per (date, mode). Rows = arena/underground, cols = current type.

Usage:
    python scripts/analysis/run_transition_litter_split.py                       # both modes, 3 families
    python scripts/analysis/run_transition_litter_split.py --dates 2026_02 --mode consecutive --format png
    python scripts/analysis/run_transition_litter_split.py --mode allpairs --max-sep-s 3600
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from ethogram_io import BASE_PROCESSED  # noqa: E402
from run_transition_prob_by_gap import (  # noqa: E402
    BOUT_CALL_TYPES, GROUP_COLS, LITTER_BOUNDARY, MIN_COUNT, TAU_BINS,
    _counts_from_pairs, _save_and_export, call_composition_by_loc, load_calls,
    loc_split_pairs, render_grid_overlay, split_by_litter,
)
from run_transition_prob_by_gap_allpairs import (  # noqa: E402
    DEFAULT_MAX_SEP_S, all_pairs_counts_by_loc,
)

# before = dashed + open markers; after = solid + filled. Same colour = same next-type.
PERIOD_STYLE = {
    "before litter": dict(ls="--", mfc="none", alpha=0.75, lw=1.1),
    "after litter":  dict(ls="-",  mfc=None,   alpha=1.0,  lw=1.6),
}


def _consecutive_counts(period_df):
    """(counts_by_loc, base_by_loc) for the consecutive-transition view of one period."""
    pairs = loc_split_pairs(period_df, BOUT_CALL_TYPES, GROUP_COLS,
                            "start_time_real", "stop_time_real")
    counts_by_loc = {loc: _counts_from_pairs(pairs[loc], BOUT_CALL_TYPES, TAU_BINS)[0]
                     for loc in pairs}
    return counts_by_loc, call_composition_by_loc(period_df, BOUT_CALL_TYPES)


def _allpairs_counts(period_df, max_sep_s):
    """(counts_by_loc, base_by_loc) for the all-pairs view of one period."""
    return all_pairs_counts_by_loc(period_df, BOUT_CALL_TYPES, TAU_BINS, max_sep_s)


def build_periods(before_df, after_df, mode, max_sep_s):
    """Assemble the overlay period list (before, after) for one family and mode."""
    out = []
    for label, df in (("before litter", before_df), ("after litter", after_df)):
        if mode == "consecutive":
            counts, base = _consecutive_counts(df)
        else:
            counts, base = _allpairs_counts(df, max_sep_s)
        out.append(dict(label=label, counts_by_loc=counts, base_by_loc=base,
                        **PERIOD_STYLE[label]))
    return out


def run_date(date, mode, out_dir, fmt, min_count, max_sep_s):
    calls = load_calls([date])
    before_df, after_df = split_by_litter(calls, date)
    e, f = LITTER_BOUNDARY[date]
    print(f"{date}: {len(before_df):,} before / {len(after_df):,} after "
          f"(litter at exp {e}, file {f})  [{mode}]")
    periods = build_periods(before_df, after_df, mode, max_sep_s)
    if mode == "consecutive":
        ylabel, xlabel, tau = "P(next | current, tau)", "gap tau (s)", ""
        title = f"Consecutive transitions, before vs after litter  ({date})"
    else:
        ylabel, xlabel = "P(type at sep tau | current)", "separation tau (s)"
        tau = f"_{max_sep_s:.0f}s"
        title = f"All-pairs composition at separation tau, before vs after litter  ({date})"
    fig = render_grid_overlay(periods, BOUT_CALL_TYPES, title, tau_bins=TAU_BINS,
                              min_count=min_count, ylabel_stat=ylabel, xlabel=xlabel)
    out_dir.mkdir(parents=True, exist_ok=True)
    _save_and_export(fig, out_dir / f"transition_litter_{mode}_{date}{tau}.{fmt}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=list(LITTER_BOUNDARY),
                    help="families with a defined litter boundary")
    ap.add_argument("--mode", choices=["consecutive", "allpairs", "both"], default="both")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "transition_analysis")
    ap.add_argument("--format", choices=["pdf", "png"], default="png")
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--max-sep-s", type=float, default=DEFAULT_MAX_SEP_S,
                    help="all-pairs only: longest separation tau to count over (s)")
    args = ap.parse_args()
    modes = ["consecutive", "allpairs"] if args.mode == "both" else [args.mode]
    for date in args.dates:
        for mode in modes:
            run_date(date, mode, args.out_dir, args.format, args.min_count, args.max_sep_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
