#!/usr/bin/env python
"""Raster and rate over the traverse EPOCH: 5 s before entry to 2 s after leaving.

A second version of `raster_and_rate.py`, which stays as it is. Three differences:

1. The window is the traverse plus asymmetric margins, not a fixed span around
   one anchor: 5 s of run-up before entering, 2 s of follow-through after
   leaving. Every call from `t_entry - 5` to `t_out + 2` is in,
   everything else is out. Shown twice: anchored on entry and anchored on
   leaving, the same epoch read from either end.

2. Only the `underground` compartment -- the nest-end pair, which carries the
   tunnel mic (raw channel 01). The wide version pools underground with arena_1,
   which means each direction is heard by whichever mics it is walking towards
   and the two columns stop being comparable. Here BOTH directions are on the
   same microphones whether or not the animal is near them, so to_arena is a real
   negative control rather than a silence caused by walking away from the mic.
   It costs 26 of 8,319 traverses (0.3%) that sit in a file where the nest mics
   heard no USV at all.

3. Because traverses differ in length, the number contributing to a lag bin FALLS
   as you move away from the anchor -- at -8 s from leaving, only traverses that
   spent over 5 s in the tunnel exist. So the rate is divided by the traverses
   actually covering each bin, not by the total, and the grey trace on the rate
   row is that coverage. Read the far tail knowing it is a subset, and a biased
   one: only slow traverses reach it.

    python scripts/analysis/raster_and_rate_tunnel.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/tunnel_epoch2
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.analysis.raster_and_rate import (  # noqa: E402
    CACHE_VERSION, COLOURS, collect, union,
)
from scripts.analysis.raster_and_rate import WINDOW as COLLECT_WINDOW  # noqa: E402

PAD_BEFORE = 5.0  # seconds of margin before entering
PAD_AFTER = 2.0   # seconds of margin after leaving
BIN = 0.1       # default rate bin; --bin overrides. Note the audio/video
                # clocks drift by up to ~0.25 s across a file and nothing
                # corrects it, so bins finer than that buy smoothness, not
                # resolution -- do not read structure narrower than ~0.25 s
FILE_S = 360.0
# every DAS class. The compartments differ a lot in what they contain -- stacks are
# 29% of underground calls but 4% of arena_1 -- so comparing compartments means
# counting the same set on both sides.
ALL_TYPES = ("high-freq", "warble", "stacks", "newborn", "alarm")
USV_ONLY = ("high-freq", "warble")


def load_or_collect(scan: Path, out_dir: Path, calls: str, clear: float,
                    collected: Path | None, recollect: bool,
                    types: tuple[str, ...] = ALL_TYPES):
    """The cached collection, reusing a sibling run's cache when the settings match.

    The npz layout and settings string are shared with `raster_and_rate.py`, so a
    cache built there for the same --calls/--clear is reused rather than
    re-walking 3,775 videos. The epoch lags are rebuilt here from the stored call
    trains, so the cache does not need to know about the epoch at all.
    """
    # the type set is part of the identity: a cache of USV-only calls and one of
    # every call are not interchangeable
    settings = f"v{CACHE_VERSION}|{calls}|{clear}|{'+'.join(sorted(types))}"
    cache = collected or (out_dir / "collected.npz")
    if cache.exists() and not recollect:
        blob = np.load(cache, allow_pickle=True)
        if blob.get("settings", np.array("")).item() == settings:
            print(f"loaded {cache}")
            return blob["data"].item(), blob["calls_by_loc"].item()
        print(f"{cache} was built with other settings; collecting")
    data, occupancy, lag_sum, pos_edges, calls_by_loc = collect(
        scan, tuple(calls.split(",")), clear, types)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "collected.npz", data=np.array(data, dtype=object),
             occupancy=np.array(occupancy, dtype=object),
             lag_sum=np.array(lag_sum, dtype=object), pos_edges=pos_edges,
             calls_by_loc=np.array(calls_by_loc, dtype=object),
             settings=np.array(settings))
    print(f"collected and cached -> {out_dir}/collected.npz")
    return data, calls_by_loc



def light_dark(scan: Path, date: str) -> dict:
    """(exp, file_num, t_entry) -> "light" or "dark", from the wall clock.

    The collection stores only file-relative times, so the phase has to come back
    from the pooled parquet's `start_time_real`. 2026_02 ran lights 04:00-16:00;
    the cycle per cohort lives in scripts/utils/light_cycle.py.
    """
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
    from light_cycle import get_light_cycle_for_month
    on, off = get_light_cycle_for_month(date)
    from scripts.utils.data_rules import load_traverses
    tv = load_traverses(scan, date, keep_capped=True, quiet=True)
    hour = tv.start_time_real.dt.hour
    phase = np.where((hour >= on) & (hour < off), "light", "dark")
    print(f"  light cycle for {date}: {on:02d}:00-{off:02d}:00")
    return {(int(e), int(f), round(float(x), 3)): ph
            for e, f, x, ph in zip(tv.exp, tv.file_num, tv.t_entry, phase)}, (on, off)


def shade(colour: str, factor: float) -> tuple:
    """A darker version of a direction's colour, for a later series in a block."""
    r, g, b = mcolors.to_rgb(colour)
    return (r * factor, g * factor, b * factor)


def tint(colour: str, factor: float) -> tuple:
    """A lighter version, mixed towards white."""
    r, g, b = mcolors.to_rgb(colour)
    return tuple(c + (1 - c) * factor for c in (r, g, b))


def localised_sides(scan: Path, quantile: float = 0.95, min_reference: int = 50,
                    types: tuple[str, ...] | None = None):
    """Both sides of the localiser threshold: tunnel-origin and nest-origin calls.

    ch01 sits at the tunnel mouth and ch00 deeper in the nest, so the level
    difference between them puts a call on that axis without seeing the animal.
    The threshold is calibrated per experiment from moments the TRACKS say the
    tunnel was empty: those calls cannot have come from the tunnel, so they are
    the nest-origin reference, and anything above its `quantile` is called
    tunnel-origin.

    Returns the per-(exp, file) call trains plus a breakdown by tunnel state --
    read that breakdown before trusting anything downstream. Selecting calls that
    are loud at the tunnel selects moments an animal was IN the tunnel, which is
    also when traverses happen. See the note in main().
    """
    tunnel, nest = {}, {}
    empty_pass = empty_all = occupied_pass = occupied_all = used = 0
    for path in sorted((scan / "origin").glob("*/call_origin.csv")):
        exp = int(path.parent.name)
        try:
            table = pd.read_csv(path)
        except pd.errors.EmptyDataError:
            continue
        if types is not None:
            table = table[table.event_type.isin(types)]
        reference = table.loc[table.state == "tunnel empty", "tunnel_db_over_nest"]
        if len(reference) < min_reference:
            continue
        used += 1
        cut = reference.quantile(quantile)
        hit = table[table.tunnel_db_over_nest > cut]
        miss = table[table.tunnel_db_over_nest <= cut]
        is_empty = table.state == "tunnel empty"
        empty_all += int(is_empty.sum())
        occupied_all += int((~is_empty).sum())
        empty_pass += int((hit.state == "tunnel empty").sum())
        occupied_pass += int((hit.state != "tunnel empty").sum())
        for file_num, group in hit.groupby("file"):
            tunnel[(exp, int(file_num))] = np.sort(group.start_s.to_numpy())
        for file_num, group in miss.groupby("file"):
            nest[(exp, int(file_num))] = np.sort(group.start_s.to_numpy())
    return tunnel, nest, {"experiments": used,
                          "empty": (empty_pass, empty_all),
                          "occupied": (occupied_pass, occupied_all)}


def epoch_bounds(entries: list[dict], align: str,
                 before: float = PAD_BEFORE, after: float = PAD_AFTER):
    """Each traverse's epoch in lag space, for the chosen anchor.

    The epoch in absolute time is always [t_entry - before, t_out + after]; the
    two alignments are the same seconds of audio read from opposite ends. The
    margins are deliberately asymmetric -- 5 s of run-up to the tunnel, 2 s of
    follow-through after leaving it.
    """
    inside = np.array([e["in_tunnel"] for e in entries])
    if align == "entry":
        return np.full(len(entries), -before), inside + after
    return -(inside + before), np.full(len(entries), after)


def flatten_epoch(entries: list[dict], table: dict, align: str,
                  before: float = PAD_BEFORE, after: float = PAD_AFTER):
    """Call times of the anchored files, with the anchor and epoch of each traverse."""
    field = "t_entry" if align == "entry" else "t_out"
    lo, hi = epoch_bounds(entries, align, before, after)
    times, owner, anchors = [], [], []
    for i, e in enumerate(entries):
        anchors.append(e[field])
        t = table.get(e["key"])
        if t is None or not t.size:
            continue
        times.append(t)
        owner.append(np.full(t.size, i))
    all_times = np.concatenate(times) if times else np.empty(0)
    owner = np.concatenate(owner) if owner else np.empty(0, int)
    return all_times, owner, np.array(anchors), lo, hi


def coverage(lo: np.ndarray, hi: np.ndarray, centres: np.ndarray) -> np.ndarray:
    """How many traverses have each lag bin inside their epoch."""
    slo, shi = np.sort(lo), np.sort(hi)
    return (np.searchsorted(slo, centres, "right")
            - np.searchsorted(shi, centres, "left")).astype(float)


def epoch_psth(all_times, owner, anchor_t, lo, hi, edges, centres, require_call):
    """Calls per second per CONTRIBUTING traverse, for one set of anchor times.

    require_call keeps only traverses that have at least one call inside their
    epoch. It must be applied to the SHUFFLES as well as to the data: selecting
    traverses because they called and then comparing them against a null that
    included silent windows would credit the selection to the behaviour.
    """
    if not all_times.size:
        return np.zeros(len(edges) - 1), np.zeros(len(centres))
    lags = all_times - anchor_t[owner]
    keep = (lags >= lo[owner]) & (lags <= hi[owner])
    lags, owned = lags[keep], owner[keep]
    if require_call:
        has = np.zeros(len(lo), bool)
        has[owned] = True
        denom = coverage(lo[has], hi[has], centres)
    else:
        denom = coverage(lo, hi, centres)
    counts = np.histogram(lags, bins=edges)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rate = np.where(denom > 0, counts / np.maximum(denom, 1e-9) / (edges[1] - edges[0]),
                        np.nan)
    return rate, denom


def observed_and_null(entries, table, align, edges, centres, n_shuffles, rng,
                      require_call=False, before=PAD_BEFORE, after=PAD_AFTER):
    """The real epoch PSTH and its within-file shuffles.

    The shuffle redraws each anchor uniformly inside its own 6-minute audio file
    but keeps that traverse's duration, so the fake epoch has the same shape and
    length. The null therefore answers "how much calling would these same seconds
    of these same files contain at a random moment", holding the colony's overall
    noisiness fixed and destroying only the timing relative to the traverse.

    With require_call the denominator is recomputed per draw, because which
    traverses pass the "has a call" filter changes with the anchor. Without it,
    every draw shares the same coverage.
    """
    all_times, owner, anchor_t, lo, hi = flatten_epoch(entries, table, align,
                                                       before, after)
    observed, denom = epoch_psth(all_times, owner, anchor_t, lo, hi, edges, centres,
                                 require_call)
    draws = np.empty((n_shuffles, len(centres)))
    for i in range(n_shuffles):
        draws[i] = epoch_psth(all_times, owner, rng.uniform(0, FILE_S, len(entries)),
                              lo, hi, edges, centres, require_call)[0]
    return observed, draws, denom


def score(observed, draws, centres, near) -> dict:
    sel = (centres >= near[0]) & (centres <= near[1]) & np.isfinite(observed)
    value = float(observed[sel].mean())
    spread = np.nanmean(draws[:, sel], axis=1)
    chance = float(np.median(spread))
    return {"rate": round(value, 3), "chance": round(chance, 3),
            "ratio": round(value / max(chance, 1e-9), 2),
            "p": float((spread >= value).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--collected", help="reuse another run's collected.npz if its "
                                            "--calls/--clear match")
    parser.add_argument("--calls", default="underground",
                        help="assigned locations to listen to (default: the nest-end pair, "
                             "which carries the tunnel mic)")
    parser.add_argument("--clear", type=float, default=3.0,
                        help="seconds the tunnel must be empty before and after for a "
                             "traverse to count as a clean transit; 0 disables")
    parser.add_argument("--traverses", choices=("vocal", "all"), default="vocal",
                        help="'vocal' (default) draws only traverses with at least one call "
                             "inside their epoch, and conditions the shuffle null the same "
                             "way; 'all' keeps the silent ones in the denominator")
    parser.add_argument("--max-lag", type=float, default=10.0,
                        help="how far from the anchor to draw. The epoch of a long traverse "
                             "runs further; beyond this it is off the axis, not excluded.")
    parser.add_argument("--bin", type=float, default=BIN, dest="bin_s",
                        help=f"rate bin in seconds (default {BIN}). The effective "
                             "resolution is ~0.25 s whatever this is set to; see the note "
                             "by BIN in the source.")
    parser.add_argument("--tick", type=float, default=2.0,
                        help="spacing of the LABELLED x ticks, in seconds; unlabelled minor "
                             "ticks go at half that, so the default is a mark every second")
    parser.add_argument("--shuffle", type=int, default=500,
                        help="within-file shuffles behind the null band; 0 draws no null")
    parser.add_argument("--date", default="2026_02",
                        help="date folder, used for the pooled parquet and the light cycle")
    parser.add_argument("--direction", choices=("to_nest", "to_arena"),
                        help="draw only this direction")
    parser.add_argument("--split-prior-nest", action="store_true",
                        help="split by whether the NEST was already calling in the seconds "
                             "before the animal entered the tunnel (needs --localiser). "
                             "Confounded by colony-wide bouts -- see the control below.")
    parser.add_argument("--prior-window", type=float, default=5.0,
                        help="how far before entry to look for a prior nest call")
    parser.add_argument("--split-light", action="store_true",
                        help="split every block by the light cycle, so each direction "
                             "appears twice: daytime traverses and night-time ones")
    parser.add_argument("--types", default="all",
                        help="'all' (default) counts every DAS class in every series; "
                             "'usv' restricts to high-freq + warble. Whichever is chosen "
                             "applies to arena_1, tunnel-origin and nest-origin alike.")
    parser.add_argument("--localiser-quantile", type=float, default=0.95,
                        help="how strict the tunnel-origin cut is, as a quantile of the "
                             "tunnel-empty reference. 0.95 keeps more real tunnel calls but "
                             "its false-alarm tail outnumbers them; 0.99 trades sensitivity "
                             "for precision.")
    parser.add_argument("--localiser", action="store_true",
                        help="keep only calls the tunnel-vs-nest localiser puts in the "
                             "TUNNEL (ch01 louder than ch00, against a per-experiment "
                             "tunnel-empty reference). Read the warning it prints.")
    parser.add_argument("--null-band", action="store_true",
                        help="shade the shuffle null on the rate panel. Off by default; the "
                             "numbers still go to summary.csv either way")
    parser.add_argument("--near", default="-0.5,2.0",
                        help="lo,hi seconds around the anchor that summary.csv scores")
    parser.add_argument("--recollect", action="store_true")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    locations = tuple(args.calls.split(","))
    types = ALL_TYPES if args.types == "all" else USV_ONLY
    data, calls_by_loc = load_or_collect(
        scan, out_dir, args.calls, args.clear,
        Path(args.collected) if args.collected else None, args.recollect, types)

    tunnel_calls = nest_calls = None
    if args.localiser:
        tunnel_calls, nest_calls, stats = localised_sides(
            scan, quantile=args.localiser_quantile, types=types)

    subsets, subset_direction = {}, {}
    if args.direction:
        data = {d: v for d, v in data.items() if d == args.direction}
    if args.split_light:
        phase_of, _ = light_dark(scan, args.date)
        missing = 0
        for direction, entries in data.items():
            for phase in ("light", "dark"):
                subsets[f"{direction} · {phase}"] = []
                subset_direction[f"{direction} · {phase}"] = direction
            for e in entries:
                key = (e["key"][0], e["key"][1], round(e["t_entry"], 3))
                phase = phase_of.get(key)
                if phase is None:
                    missing += 1
                    continue
                subsets[f"{direction} · {phase}"].append(e)
        if missing:
            print(f"  {missing} traverses had no wall-clock match and were dropped")
    elif args.split_prior_nest:
        if nest_calls is None:
            raise SystemExit("--split-prior-nest needs --localiser")
        for direction, entries in data.items():
            for label in ("prior nest call", "no prior nest call"):
                subsets[f"{direction} · {label}"] = []
                subset_direction[f"{direction} · {label}"] = direction
            for e in entries:
                times = nest_calls.get(e["key"])
                prior = False
                if times is not None and times.size:
                    prior = bool(((times >= e["t_entry"] - args.prior_window)
                                  & (times < e["t_entry"])).any())
                label = "prior nest call" if prior else "no prior nest call"
                subsets[f"{direction} · {label}"].append(e)
    else:
        subsets = {d: list(v) for d, v in data.items()}
        subset_direction = {d: d for d in data}

    kept = {}
    if args.traverses == "vocal":
        pooled_pre = union(calls_by_loc)
        for name, entries in subsets.items():
            vocal = []
            for e in entries:
                times = pooled_pre.get(e["key"])
                if times is None or not times.size:
                    continue
                lag = times - e["t_out"]
                if ((lag >= -(e["in_tunnel"] + PAD_BEFORE))
                        & (lag <= PAD_AFTER)).any():
                    vocal.append(e)
            kept[name] = (len(vocal), len(entries))
            subsets[name] = vocal
    else:
        kept = {n: (len(v), len(v)) for n, v in subsets.items()}

    edges = np.arange(-args.max_lag, args.max_lag + args.bin_s, args.bin_s)
    centres = edges[:-1] + args.bin_s / 2
    near = tuple(float(v) for v in args.near.split(","))
    rng = np.random.default_rng(0)
    pooled = union(calls_by_loc)
    for loc in locations:
        print(f"  {loc:<12} {sum(len(v) for v in calls_by_loc[loc].values()):,} calls")
    # A block is two SERIES drawn as two rasters plus the rate they average to.
    # Without the localiser the two series are the two directions. With it, the
    # block is one direction and the series are the two sides of the threshold --
    # so a call handing over from tunnel to nest is read down a single column.
    blocks = [("all underground USV",
               [(n, n, pooled, COLOURS[subset_direction[n]]) for n in subsets])]
    if args.localiser:
        # shade runs light -> dark along the animal's own axis: arena, then the
        # tunnel, then the nest. arena_1 is a compartment from calls.csv, not a
        # side of the localiser -- the localiser only splits the underground pair.
        def series_for(name):
            colour = COLOURS[subset_direction[name]]
            out = []
            if "arena_1" in calls_by_loc:
                # only lightly tinted: a sparse raster of 1.4 px ticks disappears
                # against white long before the colour stops looking distinct
                out.append(("arena_1", name, calls_by_loc["arena_1"], tint(colour, 0.32)))
            out += [("tunnel-origin", name, tunnel_calls, colour),
                    ("nest-origin", name, nest_calls, shade(colour, 0.52))]
            return out
        blocks = [(n, series_for(n)) for n in subsets]
        if "arena_1" not in calls_by_loc:
            print("  (no arena_1 in this collection -- pass --calls underground,arena_1)")
        pooled = tunnel_calls
        ep, ea = stats["empty"]
        op, oa = stats["occupied"]
        print(f"  localiser: {sum(len(v) for v in tunnel_calls.values()):,} tunnel-origin "
              f"and {sum(len(v) for v in nest_calls.values()):,} nest-origin calls "
              f"from {stats['experiments']} experiments")
        print(f"    of calls scored while the tunnel was EMPTY   : {ep:,}/{ea:,} "
              f"= {100*ep/max(ea,1):.1f}% pass "
              f"({100*(1-args.localiser_quantile):.0f}% by construction)")
        print(f"    of calls scored while an ANIMAL was in tunnel: {op:,}/{oa:,} "
              f"= {100*op/max(oa,1):.1f}% pass")
        print(f"    -> the filter is {100*op/max(oa,1) / max(100*ep/max(ea,1), 1e-9):.1f}x "
              f"more likely to keep a call when the tunnel is occupied.")
        print("    The within-file shuffle null does NOT know this: a random anchor "
              "usually\n    has an empty tunnel, so its tunnel-origin rate is low and the "
              "ratio below\n    is inflated by the selection. Treat it as an upper bound.")

    # One block of three rows per call set: two rasters and the rate they average
    # to. With --localiser that is the two sides of the threshold, drawn on the
    # SAME traverses in the SAME row order, so a row can be read down the figure.
    n_blocks = len(blocks)
    n_series = len(blocks[0][1])
    rate_row = n_series
    per_block = (4.5 if n_blocks <= 2 else 3.1) * n_series + 3.2
    fig = plt.figure(figsize=(13, per_block * n_blocks))
    subfigs = np.atleast_1d(fig.subfigures(n_blocks, 1, hspace=0.02))
    grids = [sf.subplots(n_series + 1, 2, height_ratios=[2.6] * n_series + [1.2],
                         gridspec_kw={"hspace": 0.09, "wspace": 0.20})
             for sf in subfigs]
    RASTER_ROW = {"to_arena": 0, "to_nest": 1}
    span = {0: (max(-args.max_lag, -PAD_BEFORE - 0.4), args.max_lag),
            1: (-args.max_lag, min(args.max_lag, PAD_AFTER + 0.4))}
    summary, curves, ceiling = [], {}, []

    for b, (tag, series) in enumerate(blocks):
        axes = grids[b]
        for row, (label, subset, table, colour) in enumerate(series):
            entries = subsets[subset]
            for align in ("entry", "exit"):
                observed, draws, denom = observed_and_null(
                    entries, table, align, edges, centres, args.shuffle or 1, rng,
                    require_call=args.traverses == "vocal")
                band = (np.percentile(draws, 2.5, axis=0),
                        np.percentile(draws, 97.5, axis=0)) \
                    if args.shuffle and args.null_band else None
                curves[(tag, label, align)] = (observed, band, denom, colour, subset)
                ceiling.append(float(np.nanmax(observed)))
                if band is not None:
                    ceiling.append(float(np.nanmax(band[1])))
                if args.shuffle:
                    n_vocal, n_total = kept[subset]
                    summary.append({"block": tag, "calls": label, "align": align,
                                    "mics": args.calls, "subset": subset,
                                    "direction": subset_direction[subset],
                                    "traverses": args.traverses, "n": len(entries),
                                    "n_with_calls": n_vocal, "n_all": n_total,
                                    "pct_with_calls": round(100 * n_vocal / n_total, 1),
                                    **score(observed, draws, centres, near)})

            # longest traverses at the bottom, so the epoch edge is a smooth curve
            # and the region beyond it is visibly empty rather than looking silent.
            # Both series in a block use the SAME order, so a row is one traverse
            # in both rasters and the handover can be read straight down.
            order = sorted(range(len(entries)), key=lambda i: -entries[i]["in_tunnel"])
            for col, align in ((0, "entry"), (1, "exit")):
                lo, hi = epoch_bounds(entries, align)
                field = "t_entry" if align == "entry" else "t_out"
                xs_acc, ys_acc, edge, ys = [], [], [], []
                for r_i, i in enumerate(order):
                    e = entries[i]
                    times = table.get(e["key"])
                    if times is not None and times.size:
                        lag = times - e[field]
                        lag = lag[(lag >= lo[i]) & (lag <= hi[i])]
                        if lag.size:
                            xs_acc.append(lag)
                            ys_acc.append(np.full(lag.size, r_i))
                    edge.append(hi[i] - PAD_AFTER if align == "entry"
                                else lo[i] + PAD_BEFORE)
                    ys.append(r_i)
                if xs_acc:
                    axes[row][col].plot(np.concatenate(xs_acc), np.concatenate(ys_acc), "|",
                                        color=colour, ms=1.4, mew=0.5, alpha=0.85,
                                        rasterized=True, zorder=2)
                axes[row][col].plot(edge, ys, color="0.25", lw=1.0, zorder=3)
                axes[row][col].set_ylim(-20, len(order) + 20)
            n_vocal, n_total = kept[subset]
            drawn = f"{n_vocal:,} / {n_total:,} = {100 * n_vocal / n_total:.0f}% had a call " \
                    f"in the epoch" if args.traverses == "vocal" else f"all {n_total:,}"
            axes[row][0].text(0.012, 0.985, f"{label}   ({subset},  {drawn})",
                              transform=axes[row][0].transAxes,
                              color=colour, fontsize=10, va="top", ha="left")
            axes[row][0].set_ylabel("traverse\n(sorted by time in tunnel)", fontsize=9)

        for col, align in ((0, "entry"), (1, "exit")):
            cover_ax = axes[rate_row][col].twinx()
            for label, _, _, _ in series:
                observed, band, denom, colour, subset = curves[(tag, label, align)]
                axes[rate_row][col].plot(centres, observed, color=colour, lw=2,
                                  label=label if col == 0 else None)
                if band is not None:
                    axes[rate_row][col].fill_between(centres, band[0], band[1], color=colour,
                                              alpha=0.16, lw=0)
                cover_ax.plot(centres, 100 * denom / max(len(subsets[subset]), 1),
                              color=colour, lw=0.9, ls=":", alpha=0.7)
            cover_ax.set_ylim(0, 105)
            if col == 0:
                cover_ax.set_ylabel("% of traverses covering this lag (dotted)",
                                    fontsize=8, color="0.35")
            cover_ax.tick_params(labelsize=8, colors="0.35")
            for side in ("top", "left"):
                cover_ax.spines[side].set_visible(False)
        axes[rate_row][0].set_ylabel("calls / s / traverse", fontsize=9)
        axes[rate_row][0].legend(frameon=False, fontsize=9, loc="upper left")
        axes[rate_row][0].set_xlabel("seconds from ENTERING the tunnel\n"
                              f"(epoch: -{int(PAD_BEFORE)} s to "
                              f"{int(PAD_AFTER)} s past leaving)")
        axes[rate_row][1].set_xlabel("seconds from LEAVING the tunnel\n"
                              f"(epoch: {int(PAD_BEFORE)} s before entering "
                              f"to +{int(PAD_AFTER)} s)")
        axes[0][0].set_title(f"{tag.upper()} — aligned to entering the tunnel",
                             loc="left", fontsize=11, pad=14)
        axes[0][1].set_title(f"{tag.upper()} — aligned to leaving the tunnel",
                             loc="left", fontsize=11, pad=14)

        for r in range(n_series + 1):
            for c in (0, 1):
                axes[r][c].set_xlim(*span[c])
                axes[r][c].xaxis.set_major_locator(MultipleLocator(args.tick))
                axes[r][c].xaxis.set_minor_locator(MultipleLocator(args.tick / 2))
                axes[r][c].axvline(0, color="0.35", lw=1.1, zorder=0)
                for side in ("top", "right"):
                    axes[r][c].spines[side].set_visible(False)
            if r < rate_row:
                for c in (0, 1):
                    axes[r][c].set_xticklabels([])
                    axes[r][c].set_yticks([])
            else:
                for c in (0, 1):
                    axes[r][c].grid(axis="y", color="0.93", lw=0.8)
                    axes[r][c].set_axisbelow(True)

    # one rate scale across every block, so the two sides of the threshold are
    # directly comparable rather than each filling its own panel
    top = np.ceil(max(ceiling) * 10) / 10
    for grid in grids:
        for c in (0, 1):
            grid[rate_row][c].set_ylim(0, top)

    heard = "nest-end mics only" if args.calls == "underground" else args.calls
    shaded = f", shaded = 95% of {args.shuffle} within-file shuffles" \
        if args.shuffle and args.null_band else ""
    drawn_note = "only traverses with a call in the epoch, null conditioned the same way" \
        if args.traverses == "vocal" else "every traverse, silent ones included"
    sides = " — the two sides of the tunnel-vs-nest localiser, same traverses" \
        if args.localiser else ""
    # wrapped: a single long line makes bbox_inches="tight" widen the whole canvas
    # to fit the text, leaving the panels stranded in the corner
    fig.suptitle(f"2026_02: the traverse epoch — {int(PAD_BEFORE)} s before entering "
                 f"to {int(PAD_AFTER)} s after leaving\n"
                 f"heard on the {heard}{sides}\n"
                 f"{drawn_note}\n"
                 f"{'every DAS class' if args.types == 'all' else 'high-freq + warble'}, "
                 f"{args.bin_s:g} s bins{shaded}",
                 x=0.01, y=0.995, ha="left", va="top", fontsize=11)
    for sf in subfigs:
        sf.subplots_adjust(left=0.09, right=0.93, top=0.94, bottom=0.07)
    subfigs[0].subplots_adjust(top=0.88 if n_blocks > 1 else 0.94)
    fig.savefig(out_dir / "raster_and_rate_tunnel.png", dpi=150, bbox_inches="tight")

    for name in subsets:
        n_vocal, n_total = kept[name]
        print(f"{name}: {n_vocal:,} / {n_total:,} = {100 * n_vocal / max(n_total, 1):.0f}% "
              f"had a call in the epoch" + ("  (drawn)" if args.traverses == "vocal"
                                            else "  (all drawn)"))
    for tag, series in blocks:
        for label, subset, _, _ in series:
            for align in ("entry", "exit"):
                observed, _, denom, _, _ = curves[(tag, label, align)]
                # relative to this series' OWN peak coverage: a series whose calls
                # are rare (tunnel-origin on to_arena) never reaches half of all
                # traverses, and thresholding on that gives an empty mask
                solid = denom > 0.5 * denom.max() if denom.max() > 0 else denom > 0
                if not solid.any() or np.isnan(observed[solid]).all():
                    print(f"{tag:<9} {label:<14} {align:<5} no covered bins")
                    continue
                at = np.nanargmax(np.where(solid, observed, np.nan))
                print(f"{tag:<9} {label:<14} {align:<5} peak {observed[at]:.2f} "
                      f"calls/s at {centres[at]:+.2f} s  "
                      f"(max coverage {100 * denom.max() / max(len(subsets[subset]), 1):.0f}% "
                      f"of drawn traverses)")
    if summary:
        table = pd.DataFrame(summary)
        table.to_csv(out_dir / "summary.csv", index=False)
        print(f"\nrate in {near[0]} to {near[1]} s around the anchor, vs the shuffle null:")
        print(table.to_string(index=False))
    print(f"\nwrote {out_dir}/raster_and_rate_tunnel.png")


if __name__ == "__main__":
    main()
