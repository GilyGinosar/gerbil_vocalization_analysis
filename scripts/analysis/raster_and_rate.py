#!/usr/bin/env python
"""The burrow-calls figure set: rasters, rate curves, the shuffle null, and the mic control.

One collection pass over the scan feeds every panel, because they are all the
same calls counted differently and letting each figure walk the tracks its own
way is how two versions of "the rate at arrival" start disagreeing.

A rate curve is an average, and averages hide whether an effect is a few loud
traverses or most of them. The raster underneath each curve shows the calls the
curve is made of: one row per traverse, one tick per call.

Left column is time anchored on ENTERING the tunnel, middle column on LEAVING
it -- that second one is where the effect lives, and anchoring on exit removes
the jitter that variable traverse duration adds. Right column is position along
the tunnel, which puts both directions on one physical axis (their entry points
are at opposite ends, so a time axis alone is not comparable between them).

Rows within each direction are sorted by how long the animal was in the tunnel,
so the grey band showing the in-tunnel period forms a smooth edge and departures
from it are visible.

The null is a within-file shuffle of the anchor times: calling comes in
colony-wide bouts, so a flat-rate null finds spurious structure wherever a bout
happened to coincide with tunnel activity. It runs on exactly the traverses the
rasters draw, so the band and the curve are the same population.

    python scripts/analysis/raster_and_rate.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/raster_clean
    python scripts/analysis/raster_and_rate.py \
        --scan /mnt/ceph/users/gginosar/burrow_scan_2026_02 --out-dir exports/raster_clean --by-mic

Writes raster_and_rate.png, position_vs_time.png, rate_by_position.csv, summary.csv,
and arrival_by_mic.png with --by-mic.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.pipeline.paths import experiment_audio_dir  # noqa: E402
from scripts.video.burrow_transit_picker import file_index  # noqa: E402


def load_usv(exp: int, locations: tuple[str, ...],
             types: tuple[str, ...] = None) -> dict[int, np.ndarray]:
    """Call onsets per file, from the given assigned locations and event types.

    `types` defaults to the USV pair, which is what this script has always used.
    Pass a wider set to count every call -- the compartments have very different
    inventories (stacks are 29% of underground but 4% of arena_1), so a comparison
    ACROSS compartments has to use the same set on both sides.

    calls.csv puts each call in ONE compartment, by which mic pair was loudest. An
    animal calling in the arena before it enters the tunnel is therefore filed
    under arena_1 and vanishes from an underground-only analysis -- even though
    the tunnel mic heard it and it is part of the same behaviour. Taking the union
    of underground and arena_1 puts those calls back; the tunnel joins exactly
    those two compartments.
    """
    types = USV if types is None else types
    path = experiment_audio_dir(exp) / "calls.csv"
    if not path.exists():
        return {}
    out: dict[int, list] = {}
    with open(path) as handle:
        for row in csv.DictReader(handle):
            if row["event_type"] in types \
                    and row["assigned_location"] in locations:
                out.setdefault(int(row["file_num"]), []).append(
                    float(row["start_time_file_sec"]))
    return {k: np.sort(np.array(v)) for k, v in out.items()}

COLOURS = {"to_arena": "#2f6fd0", "to_nest": "#d1642a"}
# Raster ticks are coloured by the COMPARTMENT das assigned the call to, not by
# direction -- the direction is already the row. A clean transit should hand over
# from one colour to the other at the tunnel; where it does not, that is the
# location-assignment error visible rather than hidden. Deliberately a different
# pair from the direction colours so one hue never means two things in one figure.
# Validated: adjacent dE 11.1 deutan / 16.3 tritan, both above the CVD floor.
COMPARTMENT = {"underground": "#8b5fbf", "arena_1": "#00a0a0"}
# DAS separates high-freq from warble unreliably, and their PSTHs correlate at
# r=0.945 -- same baseline, rise and decay -- so the split carries no information
# and they are analysed together. Stacks is NOT folded in: it anti-correlates
# (r=-0.745), tracking occupancy rather than the event, and at a third of all
# calls it would cancel the signal.
USV = ("high-freq", "warble")
FPS = 30
WINDOW = 10.0
BIN = 0.25
POS_BINS = 25
FILE_S = 360.0     # one audio file; the shuffle redraws anchors uniformly inside it
SATURATE_S = 0.3   # final seconds before the animal leaves the crop: the centroid
                   # stops moving there, so those frames all report the same position
                   # and pile into a single bin that looks like a real concentration
CACHE_VERSION = 3  # bump when a field is added to the collection, so old caches rebuild


def collect(scan: Path, locations: tuple[str, ...], clear_s: float,
            types: tuple[str, ...] = None):
    """Per traverse: call lags from exit, call positions, the in-tunnel span, and
    the (exp, file) key plus absolute anchor times the shuffle null needs.

    Also returns every USV in those files keyed by (exp, file) and by compartment
    -- the shuffle draws its fake anchors against the same call trains, and the
    mic control needs the two compartments kept apart.
    """
    out = {d: [] for d in COLOURS}
    occupancy = {d: np.zeros(POS_BINS) for d in COLOURS}
    # seconds-weighted sum of (frame time - exit) per position bin, so a position
    # can be placed on a time axis and the two views compared directly
    lag_sum = {d: np.zeros(POS_BINS) for d in COLOURS}
    calls_by_loc: dict[str, dict[tuple[int, int], np.ndarray]] = {loc: {} for loc in locations}
    pos_edges = np.linspace(0, 1, POS_BINS + 1)
    for exp_dir in sorted(p for p in scan.iterdir() if p.is_dir() and p.name.isdigit()):
        exp = int(exp_dir.name)
        per_loc = {loc: load_usv(exp, (loc,), types) for loc in locations}
        usv = {}
        for loc, files in per_loc.items():
            for k, v in files.items():
                usv.setdefault(k, []).append((loc, v))
                calls_by_loc[loc][(exp, k)] = v
        if not usv:
            continue
        rows = [r for path in sorted(exp_dir.glob("traverses*.csv"))
                for r in csv.DictReader(open(path))
                if str(r["single_animal"]).lower() == "true"]
        by_video: dict[str, list[dict]] = {}
        for row in rows:
            by_video.setdefault(row["video"], []).append(row)
        for video, group in by_video.items():
            track_path = exp_dir / "tracks" / f"{Path(video).stem}.parquet"
            if not track_path.exists():
                continue
            track = pd.read_parquet(track_path)
            xs, n_animals = track.x.to_numpy(), track.n_animals.to_numpy()
            file_num = file_index(video)
            by_loc = dict(usv.get(file_num, []))
            times = np.sort(np.concatenate([v for v in by_loc.values()])) \
                if by_loc else np.empty(0)
            if not times.size:
                continue
            occupied = n_animals > 0
            for row in group:
                direction = row["direction"]
                entry, out_s = float(row["t_entry"]), float(row["t_out"])
                lo, hi = int(entry * FPS), int(out_s * FPS)
                if hi <= lo or hi >= len(xs):
                    continue
                if clear_s:
                    # A clean transit: the animal enters the ROI from outside, crosses,
                    # and leaves without coming back. The test must bracket the
                    # OCCUPANCY RUN, not the landmark crossings -- the animal is
                    # already inside the crop before it reaches x=0.75, so requiring
                    # an empty tunnel just before t_entry is impossible by
                    # construction and rejected every traverse.
                    run_start = lo
                    while run_start > 0 and occupied[run_start - 1]:
                        run_start -= 1
                    run_end = hi
                    while run_end < len(occupied) - 1 and occupied[run_end]:
                        run_end += 1
                    pad = int(clear_s * FPS)
                    before_ok = run_start - pad >= 0 and not occupied[run_start - pad:run_start].any()
                    after_ok = run_end + pad < len(occupied) and not occupied[run_end:run_end + pad].any()
                    if not (before_ok and after_ok):
                        continue
                # Position uses the whole ROI VISIT, not the landmark window. to_nest
                # enters at x=0.75 and to_arena at x=0.15, so a landmark-bounded
                # window truncates each direction at its own entry (0.09-0.76 and
                # 0.14-0.93) even though the tracking reaches 0.08-0.94 for both.
                vis_start, vis_end = lo, hi
                while vis_start > 0 and n_animals[vis_start - 1] > 0:
                    vis_start -= 1
                while vis_end < len(n_animals) - 1 and n_animals[vis_end] > 0:
                    vis_end += 1
                # Drop the last SATURATE_S before it vanishes: as the body leaves the
                # crop the centroid stops moving and parks near the edge, so those
                # frames all report ~0.12 (to_nest) or ~0.77 (to_arena) and pile into
                # a single bin that looks like a real concentration.
                vis_end = max(vis_start + 1, vis_end - int(SATURATE_S * FPS))
                window_x, window_n = xs[vis_start:vis_end], n_animals[vis_start:vis_end]
                valid = (window_n == 1) & np.isfinite(window_x)
                if valid.sum() >= 3:
                    px = window_x[valid]
                    occupancy[direction] += np.histogram(px, bins=pos_edges)[0] / FPS
                    lag = (np.arange(vis_start, vis_end)[valid] / FPS) - out_s
                    lag_sum[direction] += np.histogram(px, bins=pos_edges, weights=lag)[0] / FPS
                # calls outside the tunnel have no tracked position, but we know
                # which SIDE the animal is on from its direction: before entry it is
                # on the side it came from, after exit on the side it went to. That
                # puts every call in the window onto the position axis.
                before = int(((times >= entry - WINDOW) & (times < entry)).sum())
                after = int(((times > out_s) & (times <= out_s + WINDOW)).sum())
                near = times[(times >= out_s - WINDOW) & (times <= out_s + WINDOW)]
                near_entry = times[(times >= entry - WINDOW) & (times <= entry + WINDOW)]
                inside = times[(times >= vis_start / FPS) & (times <= vis_end / FPS)]
                positions = np.empty(0)
                if inside.size:
                    idx = np.clip((inside * FPS).astype(int), vis_start, vis_end - 1)
                    p = xs[idx]
                    positions = p[np.isfinite(p) & (n_animals[idx] == 1)]
                lags_by_loc = {loc: v[(v >= out_s - WINDOW) & (v <= out_s + WINDOW)] - out_s
                               for loc, v in by_loc.items()}
                entry_by_loc = {loc: v[(v >= entry - WINDOW) & (v <= entry + WINDOW)] - entry
                                for loc, v in by_loc.items()}
                out[direction].append({"lags_by_loc": lags_by_loc,
                                       "entry_by_loc": entry_by_loc,
                                       "before": before, "after": after,
                                       "lags": near - out_s,
                                       "lags_entry": near_entry - entry,
                                       "positions": positions,
                                       "key": (exp, file_num),
                                       "t_entry": entry, "t_out": out_s,
                                       "in_tunnel": out_s - entry})
    return out, occupancy, lag_sum, pos_edges, calls_by_loc


# ---- the shuffle null -----------------------------------------------------
# Flattening the anchors once and histogramming whole arrays keeps 500 shuffles
# to a few seconds; the obvious loop over anchors inside a loop over shuffles is
# a few million Python iterations and takes minutes.

def flatten(anchors: list[tuple], table: dict) -> tuple[np.ndarray, np.ndarray, int]:
    """All call times of the anchored files, plus which anchor each belongs to."""
    times, owner = [], []
    for i, (key, _) in enumerate(anchors):
        t = table.get(key)
        if t is None or not t.size:
            continue
        times.append(t)
        owner.append(np.full(t.size, i))
    if not times:
        return np.empty(0), np.empty(0, int), len(anchors)
    return np.concatenate(times), np.concatenate(owner), len(anchors)


def psth(times: np.ndarray, owner: np.ndarray, anchor_t: np.ndarray,
         edges: np.ndarray, n_anchors: int) -> np.ndarray:
    """Calls per second per traverse, for one set of anchor times."""
    if not times.size:
        return np.zeros(len(edges) - 1)
    lags = times - anchor_t[owner]
    lags = lags[(lags >= edges[0]) & (lags < edges[-1])]
    return np.histogram(lags, bins=edges)[0] / max(n_anchors, 1) / (edges[1] - edges[0])


def observed_and_null(anchors: list[tuple], table: dict, edges: np.ndarray,
                      n_shuffles: int, rng) -> tuple[np.ndarray, np.ndarray]:
    """The real PSTH, and n_shuffles PSTHs with the anchors redrawn inside the file."""
    times, owner, n_anchors = flatten(anchors, table)
    observed = psth(times, owner, np.array([t for _, t in anchors]), edges, n_anchors)
    draws = np.empty((n_shuffles, len(edges) - 1))
    for i in range(n_shuffles):
        draws[i] = psth(times, owner, rng.uniform(0, FILE_S, n_anchors), edges, n_anchors)
    return observed, draws


def union(calls_by_loc: dict) -> dict[tuple[int, int], np.ndarray]:
    """One call train per file, pooling the compartments that were loaded."""
    out: dict[tuple[int, int], np.ndarray] = {}
    for files in calls_by_loc.values():
        for key, arr in files.items():
            out[key] = np.sort(np.concatenate([out.get(key, np.empty(0)), arr]))
    return out


def score(observed: np.ndarray, draws: np.ndarray, centres: np.ndarray,
          near: tuple[float, float]) -> dict:
    """Rate in the window of interest against the shuffle distribution."""
    sel = (centres >= near[0]) & (centres <= near[1])
    value = float(observed[sel].mean())
    spread = draws[:, sel].mean(axis=1)
    chance = float(np.median(spread))
    return {"rate": round(value, 3), "chance": round(chance, 3),
            "ratio": round(value / max(chance, 1e-9), 2),
            "p": float((spread >= value).mean())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--rows", type=int, default=0,
                        help="unused; every traverse is drawn (kept so old commands still run)")
    parser.add_argument("--calls", default="underground,arena_1",
                        help="which assigned locations to pool (default both ends of the tunnel)")
    parser.add_argument("--clear", type=float, default=3.0,
                        help="seconds the tunnel must be empty before entry and after exit for a "
                             "traverse to count as a clean transit; 0 disables the filter")
    parser.add_argument("--shuffle", type=int, default=500,
                        help="within-file shuffles behind the null band and the p-values; "
                             "0 draws the curves with no null")
    parser.add_argument("--near", default="-0.5,2.0",
                        help="lo,hi seconds around the anchor that summary.csv scores")
    parser.add_argument("--by-mic", action="store_true",
                        help="extra figure: the same arrival heard on each compartment's mics "
                             "separately, which is what makes the nest-specific claim a "
                             "measured negative rather than an absent microphone")
    parser.add_argument("--recollect", action="store_true",
                        help="re-walk the tracks instead of using the cached collection")
    args = parser.parse_args()

    scan, out_dir = Path(args.scan), Path(args.out_dir)
    locations = tuple(args.calls.split(","))
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "collected.npz"
    settings = f"v{CACHE_VERSION}|{args.calls}|{args.clear}"
    stale = True
    if cache.exists() and not args.recollect:
        blob = np.load(cache, allow_pickle=True)
        stale = blob.get("settings", np.array("")).item() != settings
        if stale:
            print("cached collection was built with other settings; rebuilding")
        else:
            data = blob["data"].item()
            occupancy = blob["occupancy"].item()
            pos_edges = blob["pos_edges"]
            lag_sum = blob["lag_sum"].item()
            calls_by_loc = blob["calls_by_loc"].item()
            print(f"loaded {cache} (pass --recollect to rebuild from the tracks)")
    if stale:
        data, occupancy, lag_sum, pos_edges, calls_by_loc = collect(
            scan, locations, args.clear)
        np.savez(cache, data=np.array(data, dtype=object),
                 occupancy=np.array(occupancy, dtype=object),
                 lag_sum=np.array(lag_sum, dtype=object), pos_edges=pos_edges,
                 calls_by_loc=np.array(calls_by_loc, dtype=object),
                 settings=np.array(settings))
        print(f"collected and cached -> {cache}")
    pos_centres = (pos_edges[:-1] + pos_edges[1:]) / 2
    edges = np.arange(-WINDOW, WINDOW + BIN, BIN)
    centres = edges[:-1] + BIN / 2
    near = tuple(float(v) for v in args.near.split(","))
    rng = np.random.default_rng(0)
    pooled = union(calls_by_loc)
    for loc in locations:
        print(f"  {loc:<12} {sum(len(v) for v in calls_by_loc[loc].values()):,} USV calls")

    # the null band, per alignment and direction, on exactly the drawn traverses
    summary = []
    bands: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    if args.shuffle:
        for align, field in (("entry", "t_entry"), ("exit", "t_out")):
            for direction in COLOURS:
                anchors = [(e["key"], e[field]) for e in data[direction]]
                observed, draws = observed_and_null(anchors, pooled, edges, args.shuffle, rng)
                bands[(align, direction)] = (np.percentile(draws, 2.5, axis=0),
                                             np.percentile(draws, 97.5, axis=0))
                summary.append({"figure": "rate", "align": align, "mics": args.calls,
                                "direction": direction, "n": len(anchors),
                                **score(observed, draws, centres, near)})

    # one raster row per direction, then the rates -- stacking both directions
    # inside a single axes made the block boundary read as a feature
    fig, axes = plt.subplots(3, 3, figsize=(18, 15), height_ratios=[2.6, 2.6, 1.2],
                             gridspec_kw={"hspace": 0.09, "wspace": 0.22})
    RASTER_ROW = {"to_arena": 0, "to_nest": 1}
    for direction in ("to_nest", "to_arena"):
        r = RASTER_ROW[direction]
        entries = data[direction]
        colour = COLOURS[direction]
        # every traverse, sorted so the shortest are at the top. Drawing 4,000 rows
        # one plot() call at a time is far too slow, so each panel gets a single
        # call with all its points -- and rasterized, or the vector output is
        # hundreds of thousands of separate marks.
        order = sorted(range(len(entries)), key=lambda i: -entries[i]["in_tunnel"])
        acc = {(col, loc): ([], []) for col in (0, 1) for loc in COMPARTMENT}
        pos_acc = ([], [])
        entry_x, exit_x, ys = [], [], []
        for row, i in enumerate(order):
            e = entries[i]
            entry_x.append(-e["in_tunnel"])
            exit_x.append(e["in_tunnel"])
            ys.append(row)
            for col, key in ((0, "entry_by_loc"), (1, "lags_by_loc")):
                for loc, v in e.get(key, {}).items():
                    if loc in COMPARTMENT and v.size:
                        acc[(col, loc)][0].append(v)
                        acc[(col, loc)][1].append(np.full(v.size, row))
            v = e["positions"]
            if v.size:
                pos_acc[0].append(v)
                pos_acc[1].append(np.full(v.size, row))
        for (col, loc), (xs_acc, ys_acc) in acc.items():
            if xs_acc:
                axes[r][col].plot(np.concatenate(xs_acc), np.concatenate(ys_acc), "|",
                                  color=COMPARTMENT[loc], ms=0.9, mew=0.4, alpha=0.8,
                                  rasterized=True, zorder=2,
                                  label=loc if (r == 0 and col == 0) else None)
        if pos_acc[0]:
            axes[r][2].plot(np.concatenate(pos_acc[0]), np.concatenate(pos_acc[1]), "|",
                            color=colour, ms=0.9, mew=0.4, alpha=0.75,
                            rasterized=True, zorder=2)
        axes[r][0].plot(exit_x, ys, color="0.25", lw=1.0, zorder=3)
        axes[r][1].plot(entry_x, ys, color="0.25", lw=1.0, zorder=3)
        for c in range(3):
            axes[r][c].set_ylim(-20, len(order) + 20)
        axes[r][0].text(-WINDOW + 0.3, len(order), f"{direction}   "
                        f"(all {len(order):,})", color=colour, fontsize=10, va="top")

    ceiling = []
    for direction in ("to_nest", "to_arena"):
        entries = data[direction]
        colour = COLOURS[direction]
        # column 0 is entry-aligned, column 1 exit-aligned -- each must use its own
        # lag array, or the curve sits under the wrong axis label
        for col, align, field in ((0, "entry", "lags_entry"), (1, "exit", "lags")):
            lags = np.concatenate([e[field] for e in entries if e[field].size])
            axes[2][col].plot(centres, np.histogram(lags, bins=edges)[0] / len(entries) / BIN,
                              color=colour, lw=2,
                              label=f"{direction} (n={len(entries):,})" if col == 0 else None)
            if (align, direction) in bands:
                lo, hi = bands[(align, direction)]
                axes[2][col].fill_between(centres, lo, hi, color=colour, alpha=0.16, lw=0)
                ceiling.append(float(hi.max()))
        pos = np.concatenate([e["positions"] for e in entries if e["positions"].size])
        counts = np.histogram(pos, bins=pos_edges)[0]
        seconds = occupancy[direction]
        with np.errstate(invalid="ignore", divide="ignore"):
            prate = np.where(seconds > 5, counts / seconds, np.nan)
        axes[2][2].plot(pos_centres, prate, color=colour, lw=2)

        # the two flanks: WINDOW seconds either side of the traverse, placed on the
        # side of the tunnel the animal was actually on
        nest_side = sum(e["after"] for e in entries) if direction == "to_nest" \
            else sum(e["before"] for e in entries)
        arena_side = sum(e["before"] for e in entries) if direction == "to_nest" \
            else sum(e["after"] for e in entries)
        flank_seconds = len(entries) * WINDOW
        for x, n in ((-0.13, nest_side), (1.13, arena_side)):
            axes[2][2].plot([x - 0.05, x + 0.05], [n / flank_seconds] * 2,
                            color=colour, lw=3, solid_capstyle="butt")
        for ax in (axes[0][2], axes[1][2], axes[2][2]):
            ax.set_xlim(-0.22, 1.22)

    # rows 0 and 1 are rasters (one direction each); row 2 is the rate curves
    # proxy handles: the raster's own markers are 0.9 px ticks, and a legend
    # built from them draws a key with nothing visible in it
    axes[0][0].legend(handles=[Line2D([0], [0], color=colour, lw=5, label=loc)
                               for loc, colour in COMPARTMENT.items()],
                      frameon=False, fontsize=8, loc="lower left",
                      title="DAS compartment", title_fontsize=8)
    for r in (0, 1):
        axes[r][0].set_ylabel("traverse\n(sorted by time in tunnel)", fontsize=9)
    for r in range(3):
        axes[r][0].set_xlim(-WINDOW, WINDOW)
        axes[r][1].set_xlim(-WINDOW, WINDOW)
        axes[r][2].set_xlim(-0.22, 1.22)
    axes[2][0].set_xlabel("seconds from ENTERING the tunnel")
    axes[2][1].set_xlabel("seconds from LEAVING the tunnel")
    axes[2][2].set_xlabel("position along the tunnel\n"
                          "bars outside 0-1 = animal in the nest / in the arena")
    axes[2][0].set_ylabel("USV calls / s / traverse", fontsize=9)
    # both time panels share one scale so the two alignments are comparable, but
    # the limit must come from BOTH of them -- matching the left panel alone
    # clipped the exit-aligned peak at 1.9 -- and from the null band, which can
    # sit above a flat curve
    top = max(ceiling + [line.get_ydata().max()
                         for ax in (axes[2][0], axes[2][1]) for line in ax.get_lines()
                         if len(line.get_ydata())])
    for ax in (axes[2][0], axes[2][1]):
        ax.set_ylim(0, np.ceil(top * 10) / 10)
    axes[2][2].set_ylabel("USV calls / s in tunnel", fontsize=9)
    # small direction-of-travel arrows on the position rate panel, in a blended
    # transform so x is position and y is a fraction of the axes height
    blend = matplotlib.transforms.blended_transform_factory(
        axes[2][2].transData, axes[2][2].transAxes)
    for direction, y in (("to_arena", 0.90), ("to_nest", 0.78)):
        tail, head = (0.30, 0.62) if direction == "to_arena" else (0.62, 0.30)
        axes[2][2].annotate("", xy=(head, y), xytext=(tail, y), xycoords=blend,
                            textcoords=blend,
                            arrowprops=dict(arrowstyle="-|>", color=COLOURS[direction],
                                            lw=1.4, mutation_scale=11))
        axes[2][2].text(0.66, y, f" {direction}",
                        transform=blend, color=COLOURS[direction], fontsize=8,
                        va="center", ha="left")
    axes[2][0].legend(frameon=False, fontsize=9)
    axes[0][0].set_title("aligned to entering the tunnel", loc="left", fontsize=11)
    axes[0][1].set_title("aligned to leaving the tunnel", loc="left", fontsize=11)
    axes[0][2].set_title("by position in the tunnel", loc="left", fontsize=11)

    for col, marks in ((0, [0.0]), (1, [0.0]), (2, [0.0, 0.15, 0.75, 1.0])):
        for r in (0, 1):
            ax = axes[r][col]
            for m in marks:
                ax.axvline(m, color="0.4" if m in (0.0, 1.0) else "0.75",
                           lw=1.1 if m in (0.0, 1.0) else 0.9,
                           ls="-" if m in (0.0, 1.0) else "--", zorder=0)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        for r in (0, 1):
            axes[r][col].set_xticklabels([])
            axes[r][col].set_yticks([])
        axes[2][col].grid(axis="y", color="0.93", lw=0.8)
        axes[2][col].set_axisbelow(True)

    shaded = f" (shaded = 95% of {args.shuffle} within-file shuffles)" if args.shuffle else ""
    fig.suptitle("2026_02: USV calls around the tunnel — rasters and the rates they average to "
                 f"(high-freq + warble){shaded}", x=0.01, ha="left", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_dir / "raster_and_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- companion figure: position, its denominator, and time ----------------
    # Two different x units, deliberately. The point is to see the views one above
    # the other, not to force position onto a time axis -- doing that would bake in
    # an average trajectory and hide how much traverse speed varies. The occupancy
    # panel sits directly under the rate it divides, because peaks at the tunnel
    # mouths are exactly what a missing denominator invents.
    fig2, ax2 = plt.subplots(3, 1, figsize=(9, 10.5), gridspec_kw={"hspace": 0.38})
    position_table = {"x": pos_centres}
    for direction, colour in COLOURS.items():
        entries = data[direction]
        seconds = occupancy[direction]
        counts = np.histogram(
            np.concatenate([e["positions"] for e in entries if e["positions"].size]),
            bins=pos_edges)[0]
        with np.errstate(invalid="ignore", divide="ignore"):
            prate = np.where(seconds > 5, counts / seconds, np.nan)
        ax2[0].plot(pos_centres, prate, color=colour, lw=2,
                    label=f"{direction} (n={len(entries):,}, {seconds.sum():.0f} s in tunnel)")
        ax2[1].plot(pos_centres, seconds, color=colour, lw=2, label=direction)
        position_table[f"{direction}_calls"] = counts
        position_table[f"{direction}_seconds"] = seconds
        nest_side = sum(e["after"] for e in entries) if direction == "to_nest" \
            else sum(e["before"] for e in entries)
        arena_side = sum(e["before"] for e in entries) if direction == "to_nest" \
            else sum(e["after"] for e in entries)
        flank = len(entries) * WINDOW
        for x, n in ((-0.13, nest_side), (1.13, arena_side)):
            ax2[0].plot([x - 0.05, x + 0.05], [n / flank] * 2, color=colour, lw=3,
                        solid_capstyle="butt")
        lags = np.concatenate([e["lags"] for e in entries if e["lags"].size])
        ax2[2].plot(centres, np.histogram(lags, bins=edges)[0] / len(entries) / BIN,
                    color=colour, lw=2)
        if ("exit", direction) in bands:
            lo, hi = bands[("exit", direction)]
            ax2[2].fill_between(centres, lo, hi, color=colour, alpha=0.16, lw=0)

    for ax in ax2[:2]:
        ax.set_xlim(-0.22, 1.22)
        for level in (0.15, 0.75):
            ax.axvline(level, color="0.75", ls="--", lw=0.9)
        for end in (0.0, 1.0):
            ax.axvline(end, color="0.45", lw=1.1)
    ax2[0].set_xlabel("position along the tunnel   (0 = nest end,  1 = arena end)\n"
                      "bars outside 0-1 = animal in the nest / in the arena")
    ax2[0].set_ylabel("USV calls / s in tunnel")
    ax2[1].set_xlabel("position along the tunnel")
    ax2[1].set_ylabel("seconds spent there\n(the denominator above)")
    ax2[2].set_xlim(-WINDOW, WINDOW)
    ax2[2].axvline(0, color="0.3", lw=1)
    ax2[2].set_xlabel("seconds from leaving the tunnel")
    ax2[2].set_ylabel("USV calls / s / traverse")
    ax2[0].legend(frameon=False, fontsize=9)
    # the flank bars are drawn from plain lists, so get_ydata() is not always an
    # ndarray; and NaN-padded position bins must not swallow the max. Only the two
    # RATE panels share a scale -- the occupancy panel is in seconds.
    top = max(np.nanmax(np.asarray(l.get_ydata(), dtype=float))
              for a in (ax2[0], ax2[2]) for l in a.get_lines() if len(l.get_ydata()))
    for a in (ax2[0], ax2[2]):
        a.set_ylim(0, np.ceil(top * 10) / 10)
    for a in ax2:
        a.grid(axis="y", color="0.93", lw=0.8)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
    fig2.suptitle("2026_02: the same calls by position, the time spent there, and by time",
                  x=0.01, ha="left", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.97))
    fig2.savefig(out_dir / "position_vs_time.png", dpi=150)
    plt.close(fig2)
    pd.DataFrame(position_table).to_csv(out_dir / "rate_by_position.csv", index=False)

    # ---- the mic control ------------------------------------------------------
    # Every other panel pools the compartments, so "to_arena is flat at its own
    # arrival" would be an absent microphone rather than a measured negative: that
    # animal arrives in the arena. Here each compartment is asked separately.
    if args.by_mic:
        if len(locations) < 2:
            raise SystemExit("--by-mic needs at least two locations in --calls")
        if not args.shuffle:
            raise SystemExit("--by-mic needs --shuffle > 0 for its null")
        fig3, axes3 = plt.subplots(1, len(locations), figsize=(6 * len(locations), 4.8),
                                   sharex=True, sharey=True)
        axes3 = np.atleast_1d(axes3)
        for ax, location in zip(axes3, locations):
            table = calls_by_loc[location]
            for direction in COLOURS:
                anchors = [(e["key"], e["t_out"]) for e in data[direction]]
                observed, draws = observed_and_null(anchors, table, edges, args.shuffle, rng)
                lo, hi = np.percentile(draws, 2.5, axis=0), np.percentile(draws, 97.5, axis=0)
                ax.fill_between(centres, lo, hi, color=COLOURS[direction], alpha=0.16, lw=0)
                ax.plot(centres, observed, color=COLOURS[direction], lw=2,
                        label=f"{direction} (n={len(anchors):,})")
                summary.append({"figure": "by_mic", "align": "exit", "mics": location,
                                "direction": direction, "n": len(anchors),
                                **score(observed, draws, centres, near)})
            ax.axvline(0, color="0.3", lw=1)
            where = "nest-end mics" if location == "underground" else f"{location} mics"
            ax.set_title(f"heard on the {where}", loc="left", fontsize=11)
            ax.set_xlabel("seconds from leaving the tunnel (arrival)")
            ax.grid(axis="y", color="0.93", lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
        axes3[0].set_ylabel("USV calls / s / traverse")
        axes3[0].legend(frameon=False, fontsize=9)
        fig3.suptitle(f"2026_02: arrival, heard at each end separately "
                      f"(shaded = 95% of {args.shuffle} within-file shuffles)",
                      x=0.01, ha="left", fontsize=12)
        fig3.tight_layout()
        fig3.savefig(out_dir / "arrival_by_mic.png", dpi=150)
        plt.close(fig3)

    for direction in ("to_nest", "to_arena"):
        e = data[direction]
        vocal = sum(1 for x in e if x["lags"].size)
        print(f"{direction}: {len(e):,} clean transits, {vocal:,} with a USV in the +/-10 s window "
              f"({100*vocal/len(e):.0f}%)")
    if summary:
        table = pd.DataFrame(summary)
        table.to_csv(out_dir / "summary.csv", index=False)
        print(f"\nrate in {near[0]} to {near[1]} s around the anchor, vs the shuffle null:")
        print(table.to_string(index=False))
    print(f"\nwrote {out_dir}/raster_and_rate.png, position_vs_time.png, rate_by_position.csv"
          + (", arrival_by_mic.png" if args.by_mic else ""))


if __name__ == "__main__":
    main()
