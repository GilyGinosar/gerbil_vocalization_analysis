#!/usr/bin/env python
"""Everything on one page: call set x alignment x condition, for deciding what to chase.

Four columns -- every DAS class vs USV only, each anchored on entering and on
leaving the tunnel. Four rows -- to_nest as a whole, to_nest split by whether the
nest had already been calling in the seconds before entry, and to_arena as the
opposite-direction control. Every panel carries the same three series, in the
animal's own spatial order and shaded light to dark:

    arena_1 (the compartment it came from / went to) · tunnel-origin · nest-origin

Every series gets its raster and the three of them share the rate panel below,
so a column is one condition read from ticks to average. Row order is shared down
a column -- the same traverse is the same line in all three rasters -- which is
what lets you follow a call handing over from the tunnel to the nest.

Two things this is built to show at a glance. Whether the non-USV classes
(stacks is 29% of underground calls, and tracks occupancy) are carrying any
result -- compare a row across the two call sets. And whether the prior-nest
association is specific -- compare the two middle rows.

    python scripts/analysis/burrow_overview.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/overview
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.utils.data_rules import load_traverses  # noqa: E402
from scripts.analysis.raster_and_rate_tunnel import (  # noqa: E402
    ALL_TYPES, COLOURS, PAD_AFTER, PAD_BEFORE, USV_ONLY,
    epoch_bounds, light_dark, load_or_collect, localised_sides, observed_and_null,
    score, shade, tint,
)

TYPESETS = (("every DAS class", ALL_TYPES, "all"), ("USV only", USV_ONLY, "usv"))
ALIGNS = ("entry", "exit")


def prior_nest(entry: dict, nest: dict, window: float) -> bool:
    times = nest.get(entry["key"])
    return bool(times is not None and times.size
                and ((times >= entry["t_entry"] - window)
                     & (times < entry["t_entry"])).any())


def drop_capped(data: dict, scan: Path, date: str) -> dict:
    """Remove traverses the never-usable rules exclude, from a prebuilt cache.

    `burrow_scan` reports t_out as the first sustained empty-tunnel moment, but
    when no such moment turns up within MAX_LINGER_S it returns t_exit + 5 s and
    sets `still_in_tunnel_at_cap`. Every exit-aligned panel anchors on t_out, so
    those traverses smear the arrival response by up to 5 s. The collection cache
    predates the flag, so match back to the traverse table by (exp, file, t_entry).
    """
    # the raw table on purpose: this function needs the rows the rules REMOVE, so
    # it can find them in a cache that was built before the rules existed
    raw = load_traverses(scan, date, keep_last_file=True, keep_capped=True, quiet=True)
    last = raw.groupby("exp").file_num.transform("max")
    tv = raw[raw.still_in_tunnel_at_cap | (raw.file_num == last)]
    bad: dict[tuple[int, int], list] = {}
    for r in tv.itertuples():
        bad.setdefault((int(r.exp), int(r.file_num)), []).append(float(r.t_entry))
    out, removed = {}, 0
    for direction, entries in data.items():
        keep = []
        for e in entries:
            hits = bad.get((int(e["key"][0]), int(e["key"][1])), ())
            if any(abs(t - e["t_entry"]) < 0.01 for t in hits):
                removed += 1
            else:
                keep.append(e)
        out[direction] = keep
    print(f"  dropped {removed} unusable traverses "
          f"(invented t_out, or the truncated last chunk)")
    return out


def category_lookup(path: Path):
    """(exp, file_num, t_entry) -> nest category, from the picker's verdicts.

    `empty` and `sleeping` are indistinguishable by motion -- 0.0002 vs 0.0007
    mean pre-entry motion, both dead still -- and separable only by eye. Keeping
    them apart matters: their arrival bursts differ ~4x, while `sleeping` and
    `active`, which differ 20x in motion, do not differ at all.
    """
    table = pd.read_csv(path)
    by_file: dict[tuple[int, int], list] = {}
    for r in table.itertuples():
        by_file.setdefault((int(r.exp), int(r.file_num)), []).append(
            (float(r.t_entry), str(r.cat)))

    def get(entry: dict, tol: float = 0.01):
        best, gap = None, tol
        for t, value in by_file.get((int(entry["key"][0]), int(entry["key"][1])), ()):
            d = abs(t - entry["t_entry"])
            if d <= gap:
                best, gap = value, d
        return best

    return get


def motion_lookup(path: Path):
    """(exp, file_num, t_entry) -> pre-entry nest motion, from a nest_motion.py run.

    Nearest-t_entry within a hundredth of a second rather than equality: the score
    table stores t_entry as text, and the last digit does not always survive the
    round trip. Returns None for a traverse the motion pass never measured, which
    is most of them -- the pilot sampled 400 of 4,168 to_nest traverses, so a
    motion-split row is a subset row and its n says so.
    """
    table = pd.read_csv(path)
    by_file: dict[tuple[int, int], list] = {}
    for r in table.itertuples():
        by_file.setdefault((int(r.exp), int(r.file_num)), []).append(
            (float(r.t_entry), float(r.motion_pre)))

    def get(entry: dict, tol: float = 0.01):
        best, gap = None, tol
        for t, value in by_file.get((int(entry["key"][0]), int(entry["key"][1])), ()):
            d = abs(t - entry["t_entry"])
            if d <= gap:
                best, gap = value, d
        return best

    return get, float(table.motion_pre.median())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--clear", type=float, default=3.0)
    parser.add_argument("--date", default="2026_02")
    parser.add_argument("--calls", default="underground,arena_1")
    parser.add_argument("--localiser-quantile", type=float, default=0.99)
    parser.add_argument("--column", metavar="SET:ALIGN[,...]",
                        help="draw only these columns, in this order, e.g. "
                             "'usv:entry,usv:exit'. The full grid is 40 panels on a "
                             "2400x5300 page, so on screen it is downscaled ~3x and the "
                             "raster ticks vanish; naming the columns you want gets the "
                             "same content at a size you can actually read. SET is 'all' "
                             "or 'usv', ALIGN is 'entry' (start of the traverse) or "
                             "'exit' (arrival).")
    parser.add_argument("--no-localiser", action="store_true",
                        help="drop the tunnel-origin/nest-origin series and draw one "
                             "underground series instead. The dB label is a POSITION "
                             "gradient, not a compartment split -- an animal in the "
                             "nest-end half of the tunnel scores nest-origin -- so the "
                             "split invites a source reading the data cannot support. "
                             "Nothing legitimate is lost: what remains is every "
                             "underground call against the arena_1 control.")
    parser.add_argument("--prior-window", type=float, default=5.0)
    parser.add_argument("--split-light", action="store_true",
                        help="split every row into LIGHT and DARK. The phase effect is "
                             "~4x smaller than the nest-category effect, so this halves "
                             "each n to show a weaker result -- worth it only when the "
                             "question is specifically about the light cycle.")
    parser.add_argument("--free-y", action="store_true",
                        help="let each ROW's rate panels scale to their own data. The "
                             "default shares one y-limit across the whole figure so bar "
                             "heights are comparable between rows; with rows differing "
                             "5x that flattens the small ones into the axis.")
    parser.add_argument("--keep-capped", action="store_true",
                        help="keep traverses whose t_out is t_exit + MAX_LINGER_S rather "
                             "than an observed empty tunnel; dropped by default because "
                             "every exit-aligned panel anchors on t_out")
    parser.add_argument("--category-csv",
                        help="nest_category.csv (exp,file_num,t_entry,cat) from the "
                             "nest_empty_picker verdicts. Splits to_nest into EMPTY / "
                             "SLEEPING / ACTIVE instead of the motion median, which "
                             "cannot see the empty-vs-sleeping distinction at all.")
    parser.add_argument("--motion-csv",
                        help="nest_motion.py output. Replaces the two prior-nest-CALL "
                             "rows with a split on nest MOTION before entry, which is "
                             "independent of the audio instead of nearly circular.")
    parser.add_argument("--motion-cut", type=float,
                        help="still/active boundary; defaults to the median of the "
                             "motion table, which is the split the pilot tested")
    parser.add_argument("--max-lag", type=float, default=8.0)
    parser.add_argument("--pad-before", type=float, default=PAD_BEFORE,
                        help=f"seconds of run-up before entering (default {PAD_BEFORE:g})")
    parser.add_argument("--pad-after", type=float, default=PAD_AFTER,
                        help=f"seconds of follow-through after leaving (default "
                             f"{PAD_AFTER:g}). Widening this past --clear is not free: the "
                             f"tunnel is only guaranteed empty for --clear seconds either "
                             f"side, so a longer window can run into the next transit.")
    parser.add_argument("--bin", type=float, default=0.25, dest="bin_s",
                        help="0.25 s by default: this figure is for comparing shapes across "
                             "16 panels, and the clock drift makes anything finer cosmetic")
    parser.add_argument("--shuffle", type=int, default=0,
                        help="shuffles behind the chance column of summary.csv; 0 (the "
                             "default here) skips them -- this figure is for looking")
    parser.add_argument("--near", default="-0.5,2.0")
    parser.add_argument("--recollect", action="store_true")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(-args.max_lag, args.max_lag + args.bin_s, args.bin_s)
    centres = edges[:-1] + args.bin_s / 2
    near = tuple(float(v) for v in args.near.split(","))
    pad_before, pad_after = args.pad_before, args.pad_after
    if max(pad_before, pad_after) > args.clear:
        print(f"note: epoch runs {pad_before:g}/{pad_after:g} s beyond the landmarks but "
              f"--clear is {args.clear:g} s, so the far end of the window is not "
              f"guaranteed to be a clear tunnel")
    rng = np.random.default_rng(0)

    # one collection and one localiser pass per call set
    world = {}
    for name, types, slug in TYPESETS:
        print(f"\n=== {name}")
        data, calls_by_loc = load_or_collect(
            scan, out_dir / slug, args.calls, args.clear, None, args.recollect, types)
        if not args.keep_capped:
            data = drop_capped(data, scan, args.date)
        world[name] = dict(data=data, arena=calls_by_loc["arena_1"],
                           underground=calls_by_loc["underground"])
        if args.no_localiser:
            print(f"  underground {sum(len(v) for v in calls_by_loc['underground'].values()):,}"
                  f"   (localiser not used)")
        else:
            tunnel, nest, _ = localised_sides(scan, quantile=args.localiser_quantile,
                                              types=types)
            world[name].update(tunnel=tunnel, nest=nest)
            print(f"  tunnel-origin {sum(len(v) for v in tunnel.values()):,}   "
                  f"nest-origin {sum(len(v) for v in nest.values()):,}")

    # rows: (label, direction, keep). `keep` is None for "every traverse", else a
    # predicate on one entry given that call set's world.
    phase_of = None
    if args.split_light:
        phase_of, (on_h, off_h) = light_dark(scan, args.date)
        print(f"  light {on_h:02d}:00-{off_h:02d}:00")

    def phased(base_keep, want):
        """base_keep AND this light phase."""
        def f(e, w):
            if base_keep is not None and not base_keep(e, w):
                return False
            key = (int(e["key"][0]), int(e["key"][1]), round(e["t_entry"], 3))
            return phase_of.get(key) == want
        return f

    if args.category_csv:
        cat_of = category_lookup(Path(args.category_csv))
        def want(name):
            return lambda e, w: cat_of(e) == name
        names = (("EMPTY  (nobody home)", "empty"),
                 ("OCCUPIED but STILL  (sleeping residents)", "sleeping"),
                 ("ACTIVE  (residents moving)", "active"))
        if args.split_light:
            rows = tuple(
                (f"to_nest — nest {label} · {ph.upper()}", "to_nest",
                 phased(want(key), ph))
                for label, key in names for ph in ("light", "dark"))
        else:
            rows = (("to_nest — every traverse", "to_nest", None),
                    *((f"to_nest — nest {label}", "to_nest", want(key))
                      for label, key in names),
                    ("to_arena — every traverse  (control direction)", "to_arena", None))
    elif args.motion_csv:
        motion_of, median_cut = motion_lookup(Path(args.motion_csv))
        cut = args.motion_cut if args.motion_cut is not None else median_cut
        measured = lambda e, w: motion_of(e) is not None
        still = lambda e, w: (lambda v: v is not None and v <= cut)(motion_of(e))
        active = lambda e, w: (lambda v: v is not None and v > cut)(motion_of(e))
        rows = (("to_nest — every traverse", "to_nest", None),
                ("to_nest — the traverses nest motion was measured on  (subset)",
                 "to_nest", measured),
                (f"to_nest — nest STILL before entry  (motion <= {cut:.4f})",
                 "to_nest", still),
                (f"to_nest — nest ACTIVE before entry  (motion > {cut:.4f})",
                 "to_nest", active),
                ("to_arena — every traverse  (control direction)", "to_arena", None))
    elif args.split_light:
        rows = (("to_nest — LIGHT", "to_nest", phased(None, "light")),
                ("to_nest — DARK", "to_nest", phased(None, "dark")),
                ("to_arena — LIGHT  (control)", "to_arena", phased(None, "light")),
                ("to_arena — DARK  (control)", "to_arena", phased(None, "dark")))
    else:
        rows = (("to_nest — every traverse", "to_nest", None),
                (f"to_nest — nest called in the {args.prior_window:g} s before entry",
                 "to_nest", lambda e, w: prior_nest(e, w["nest"], args.prior_window)),
                (f"to_nest — nest SILENT in the {args.prior_window:g} s before entry",
                 "to_nest", lambda e, w: not prior_nest(e, w["nest"], args.prior_window)),
                ("to_arena — every traverse  (control direction)", "to_arena", None))

    columns = [(ts, al) for ts in TYPESETS for al in ALIGNS]
    if args.column:
        by_slug = {ts[2]: ts for ts in TYPESETS}
        columns = []
        for spec in args.column.split(","):
            want_set, _, want_align = spec.strip().partition(":")
            if want_set not in by_slug or want_align not in ALIGNS:
                raise SystemExit(
                    f"--column {spec!r} is not valid; SET is one of "
                    f"{sorted(by_slug)}, ALIGN is one of {list(ALIGNS)}")
            columns.append((by_slug[want_set], want_align))
    n_col = len(columns)
    n_series = 2 if args.no_localiser else 3
    row_h = 4.0 * n_series + 1.4
    fig = plt.figure(figsize=(5.5 * n_col + 1.2, row_h * len(rows) + 1.6))
    bands = np.atleast_1d(fig.subfigures(len(rows) + 1, 1, hspace=0.015,
                                         height_ratios=[1.6 / row_h] + [1.0] * len(rows)))
    heading, subfigs = bands[0], bands[1:]
    summary, ceiling, panels = [], [], []

    for r, (row_label, direction, keep) in enumerate(rows):
        grid = subfigs[r].subplots(n_series + 1, n_col, squeeze=False,
                                   height_ratios=[2.2] * n_series + [1.35],
                                   gridspec_kw={"hspace": 0.10, "wspace": 0.16})
        base = COLOURS[direction]
        row_ceiling = []
        for c, ((set_name, types, _), align) in enumerate(columns):
            w = world[set_name]
            entries = w["data"][direction]
            if keep is not None:
                entries = [e for e in entries if keep(e, w)]
            if args.no_localiser:
                series = (("arena_1  (control)", w["arena"], tint(base, 0.32)),
                          ("underground — every call", w["underground"],
                           shade(base, 0.52)))
            else:
                series = (("arena_1", w["arena"], tint(base, 0.32)),
                          ("tunnel-origin", w["tunnel"], base),
                          ("nest-origin", w["nest"], shade(base, 0.52)))
            # one order for the whole column, so a row is one traverse in all three
            order = sorted(range(len(entries)), key=lambda i: -entries[i]["in_tunnel"])
            lo, hi = epoch_bounds(entries, align, pad_before, pad_after)
            field = "t_entry" if align == "entry" else "t_out"
            # how many of these traverses had ANY underground call in the epoch --
            # the rate curve is a mean and hides whether it is a few loud traverses
            # or most of them
            with_calls = 0
            for i, e in enumerate(entries):
                t = w["underground"].get(e["key"])
                if t is None or not t.size:
                    continue
                lag = t - e[field]
                if ((lag >= lo[i]) & (lag <= hi[i])).any():
                    with_calls += 1

            for s, (label, table, colour) in enumerate(series):
                ax = grid[s][c]
                xs_acc, ys_acc = [], []
                for row_i, i in enumerate(order):
                    e = entries[i]
                    times = table.get(e["key"])
                    if times is None or not times.size:
                        continue
                    lag = times - e[field]
                    lag = lag[(lag >= lo[i]) & (lag <= hi[i])]
                    if lag.size:
                        xs_acc.append(lag)
                        ys_acc.append(np.full(lag.size, row_i))
                if xs_acc:
                    ms = float(np.clip(3.0 * np.sqrt(500.0 / max(len(order), 1)),
                                       1.3, 5.0))
                    ax.plot(np.concatenate(xs_acc), np.concatenate(ys_acc), "|",
                            color=colour, ms=ms, mew=min(0.5 * ms / 1.3, 1.1),
                            alpha=0.85, rasterized=True, zorder=2)
                # the far end of the epoch: the other landmark, per traverse
                edge = [hi[i] - pad_after if align == "entry" else lo[i] + pad_before
                        for i in order]
                ax.plot(edge, range(len(order)), color="0.25", lw=0.9, zorder=3)
                # proportional margin, not a fixed +/-20 rows: on a 51-traverse row
                # that fixed pad was 44% of the panel height and read as data with no
                # calls, while on a 3,786-row panel it is 1% and invisible
                pad_rows = max(1.0, 0.02 * len(order))
                ax.set_ylim(-pad_rows, len(order) + pad_rows)
                ax.set_yticks([])
                ax.set_xticklabels([])
                if c == 0:
                    ax.set_ylabel(label, fontsize=9, color=colour)
                panels.append((ax, align))

                observed, draws, _ = observed_and_null(
                    entries, table, align, edges, centres, max(args.shuffle, 1), rng,
                    require_call=False, before=pad_before, after=pad_after)
                grid[n_series][c].plot(centres, observed, color=colour, lw=1.9,
                                label=label if (r == 0 and c == 0) else None)
                ceiling.append(float(np.nanmax(observed)))
                row_ceiling.append(float(np.nanmax(observed)))
                if args.shuffle:
                    summary.append({"call_set": set_name, "align": align,
                                    "condition": row_label, "series": label,
                                    "n": len(entries),
                                    **score(observed, draws, centres, near)})

            rate_ax = grid[n_series][c]
            rate_ax.set_xlabel(f"seconds from {align} of the tunnel", fontsize=9)
            rate_ax.grid(axis="y", color="0.93", lw=0.8)
            rate_ax.set_axisbelow(True)
            panels.append((rate_ax, align))
            if c == 0:
                rate_ax.set_ylabel("calls / s / traverse", fontsize=9)
            pct = 100 * with_calls / len(entries) if entries else 0
            grid[0][c].set_title(f"{set_name} · aligned to {align.upper()}    "
                                 f"(n={len(entries):,};  {with_calls:,} with calls "
                                 f"= {pct:.0f}%)", loc="left", fontsize=9.5)
        subfigs[r].suptitle(row_label, x=0.005, ha="left", fontsize=13,
                            color=COLOURS[direction])
        subfigs[r].subplots_adjust(left=0.055, right=0.99, top=0.93, bottom=0.05)
        if r == 0:
            grid[n_series][0].legend(frameon=False, fontsize=8.5, loc="upper left")
        rate_axes = [grid[n_series][c] for c in range(n_col)]
        top = (max(row_ceiling) if args.free_y and row_ceiling else max(ceiling))
        for ax in rate_axes:
            ax.set_ylim(0, np.ceil(top * 10) / 10)

    for ax, align in panels:
        ax.set_xlim(max(-args.max_lag, -pad_before - 0.4) if align == "entry"
                    else -args.max_lag,
                    args.max_lag if align == "entry"
                    else min(args.max_lag, pad_after + 0.4))
        ax.axvline(0, color="0.35", lw=1.0, zorder=0)
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    series_note = ("series: every underground call vs arena_1 — the dB tunnel/nest "
                   "label is a POSITION gradient and is not drawn"
                   if args.no_localiser else
                   f"series: localiser cut at q={args.localiser_quantile}")
    if args.category_csv:
        split_note = ("nest split EMPTY / SLEEPING / ACTIVE — scored by eye from the "
                      "nest_top frames; empty and sleeping are indistinguishable by motion")
        if args.split_light:
            split_note += "  ·  each category further split LIGHT / DARK"
    elif args.split_light:
        split_note = "to_nest split LIGHT / DARK by the wall clock"
    elif args.motion_csv:
        split_note = "nest split on MOTION before entry (video, audio-independent)"
    else:
        split_note = (f"nest split on prior CALLS in {args.prior_window:g} s "
                      f"before entry")
    heading.text(0.004, 0.92,
                 "2026_02 burrow calling — every condition, rasters and rates",
                 ha="left", va="top", fontsize=15)
    heading.text(0.004, 0.52, f"{split_note}\n{series_note}",
                 ha="left", va="top", fontsize=11.5)
    heading.text(0.004, 0.10,
                 f"epoch {pad_before:g} s before entry to {pad_after:g} s after "
                 f"leaving · {args.bin_s:g} s bins · rows sorted by time in tunnel, "
                 f"shared down each column",
                 ha="left", va="top", fontsize=10, color="0.35")
    out = out_dir / "burrow_overview.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    if summary:
        pd.DataFrame(summary).to_csv(out_dir / "summary.csv", index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
