"""Categorical ("dominant call type") ethogram — each call type a color.

A different view from the per-type heatmap: instead of one row per call type,
each location gets ONE strip per day, and every time-bin is painted the color of
the call type that is most active there — but only if it clears a threshold.

  - cell color   = dominant call type (argmax of per-type-normalised rate),
                   shown only when that normalised rate >= --threshold
  - quiet cell   = day/night background (light yellow = light hours, grey = dark)
  - no recording = neutral grey

"Per-type-normalised" means each call type is scaled to its own 99th-percentile
rate (shared across locations), so a bin is colored by which call type is
unusually active *for that type* — otherwise abundant warble would win every bin.

Reuses the circadian-day binning + coverage helpers from run_ethogram.

Usage:
    python scripts/analysis/run_ethogram_categorical.py --dates 2026_02 --bin-min 10
    python scripts/analysis/run_ethogram_categorical.py --dates 2026_02 --threshold 0.4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
for p in (REPO_ROOT, REPO_ROOT / "scripts" / "utils", REPO_ROOT / "scripts" / "analysis"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from light_cycle import get_light_cycle_for_month  # noqa: E402
from ethogram_io import (  # noqa: E402
    BASE_PROCESSED, CALL_TYPE_ORDER, EVENTS_BY_DATE, LOCATION_GROUPS, ROW_START_HOUR,
    counts_grid, coverage_grid, day_axis, load_all_calls,
)

# Date folder(s) to run on — EDIT HERE to switch experiment (e.g. ["2025_10"]).
# Available: 2024_12, 2025_03, 2025_07, 2025_10, 2026_02. Overridable with --dates.
DEFAULT_DATES = ["2025_03"]
BIN_MINUTES = 5
THRESHOLD = 0.2                 # min per-type-normalised rate to paint a cell
PCT = 99.0                      # percentile used as each call type's scale
CALL_COLORS = {                 # one distinct color per call type
    "alarm": "#e41a1c",
    "high-freq": "#377eb8",
    "warble": "#4daf4a",
    "stacks": "#ff7f00",
    "newborn": "#984ea3",
}
QUIET_DAY = (1.0, 0.988, 0.816)     # #fff6d0-ish (light hours, quiet)
QUIET_NIGHT = (0.90, 0.90, 0.90)    # dark hours, quiet
NO_REC = (0.74, 0.74, 0.74)         # no recording


def _fmt_hour(t: float) -> str:
    return f"{int(round(t)) % 24:02d}:00"


def plot_categorical(df, date_folder, day0, days, covered_min, call_types, bin_minutes,
                     light_start, light_end, x0, threshold, dominance, out_path):
    groups = list(LOCATION_GROUPS.items())
    n_days, n_ct = len(days), len(call_types)
    n_hourbins = covered_min.shape[1]
    centers = x0 + (np.arange(n_hourbins) + 0.5) * (bin_minutes / 60.0)
    is_day = ((centers % 24) >= light_start) & ((centers % 24) < light_end)
    no_rec = covered_min <= 0
    color_rgb = np.array([mcolors.to_rgb(CALL_COLORS.get(ct, "#000000")) for ct in call_types])

    # Metric driving the argmax per location: (n_ct, n_days, n_hourbins).
    #   normalized -> rate / each type's 99th pct (who is most active for itself)
    #   raw        -> rate in calls/min            (who is literally most frequent)
    metric_by_group = {}
    for gname, locs in groups:
        sub = df[df["assigned_location"].isin(locs)]
        r = np.full((n_ct, n_days, n_hourbins), np.nan)
        for i, ct in enumerate(call_types):
            counts = counts_grid(sub[sub["event_type"] == ct], day0, n_days, n_hourbins, bin_minutes, x0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = counts / covered_min
            rr[no_rec] = np.nan
            r[i] = rr
        if dominance == "normalized":
            scale = np.nanpercentile(r, PCT, axis=(1, 2))
            scale[~np.isfinite(scale) | (scale == 0)] = np.nan
            metric_by_group[gname] = r / scale[:, None, None]
        else:
            metric_by_group[gname] = r

    def strip_rgb(gname, d):
        """(n_hourbins, 3) RGB for one location-day strip."""
        v = metric_by_group[gname][:, d, :]               # (n_ct, n_hourbins)
        amax = np.nanargmax(np.where(np.isnan(v), -np.inf, v), axis=0)
        mval = v[amax, np.arange(n_hourbins)]
        out = np.empty((n_hourbins, 3))
        for hb in range(n_hourbins):
            if no_rec[d, hb]:
                out[hb] = NO_REC
            elif np.isfinite(mval[hb]) and mval[hb] >= threshold:
                out[hb] = color_rgb[amax[hb]]
            else:
                out[hb] = QUIET_DAY if is_day[hb] else QUIET_NIGHT
        return out

    fig = plt.figure(figsize=(13, max(5.0, n_days * 0.55 + 1.6)))
    subfigs_all = fig.subfigures(n_days + 2, 1, height_ratios=[0.8] + [1] * n_days + [0.6], hspace=0.5)
    subfigs = subfigs_all[1:-1]
    day_labels = [d.strftime("%a %m-%d") for d in days]
    events = [(pd.Timestamp(t), lbl) for t, lbl in EVENTS_BY_DATE.get(date_folder, [])]

    for d in range(n_days):
        axs = subfigs[d].subplots(len(groups), 1, sharex=True, squeeze=False,
                                  gridspec_kw={"hspace": 0.0})[:, 0]
        axs[0].set_title(day_labels[d], loc="left", fontsize=9, pad=3)
        for gi, (gname, _locs) in enumerate(groups):
            ax = axs[gi]
            ax.imshow(strip_rgb(gname, d)[None, :, :], aspect="auto",
                      extent=(x0, x0 + 24, 0, 1), origin="lower", interpolation="nearest", zorder=1)
            ax.set_yticks([])
            ax.set_ylabel(gname, rotation=0, ha="right", va="center", fontsize=7, labelpad=20)
            if gi < len(groups) - 1:
                ax.spines["bottom"].set(visible=True, color="white", linewidth=2.5, zorder=5)
            if gi > 0:
                ax.spines["top"].set_visible(False)
            ax.set_xlim(x0, x0 + 24)
            is_bottom = (gi == len(groups) - 1)
            ticks = np.arange(x0, x0 + 25, 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([_fmt_hour(t) for t in ticks] if is_bottom else [], fontsize=4)
            if d == n_days - 1 and is_bottom:
                ax.set_xlabel("hour of day")

        for ev_dt, lbl in events:
            if ((ev_dt - pd.Timedelta(hours=x0)).normalize() - day0).days != d:
                continue
            ev_x = x0 + (((ev_dt.hour * 60 + ev_dt.minute) - x0 * 60) % 1440) / 60.0
            for ax in axs:
                ax.axvline(ev_x, color="black", lw=1.3, zorder=6)
            axs[0].text(ev_x, 0.5, " " + lbl, rotation=90, color="black", fontsize=6,
                        fontweight="bold", va="center", ha="left",
                        transform=axs[0].get_xaxis_transform(), zorder=7)

    handles = [mpatches.Patch(color=CALL_COLORS.get(ct, "#000"), label=ct) for ct in call_types]
    handles += [mpatches.Patch(color=QUIET_DAY, label="quiet (light)"),
                mpatches.Patch(color=QUIET_NIGHT, label="quiet (dark)"),
                mpatches.Patch(color=NO_REC, label="no recording")]
    fig.legend(handles=handles, loc="upper right", fontsize=8, ncol=1, framealpha=0.95)
    thr_unit = "" if dominance == "normalized" else " calls/min"
    fig.suptitle(
        f"Categorical ethogram ({dominance} dominance, thr={threshold}{thr_unit}) — {date_folder}   "
        f"({len(df):,} calls, {n_days} days, {bin_minutes}-min bins; lights "
        f"{light_start:02d}:00–{light_end:02d}:00)",
        fontsize=12, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def plot_per_type_threshold(df, date_folder, day0, days, covered_min, call_types, bin_minutes,
                            light_start, light_end, x0, threshold, out_path):
    """Per-call-type rows (like the stacked heatmap) but binary-colored by threshold.

    Each call type keeps its own row; a cell is painted that type's color where its
    per-type-normalised rate >= threshold, else day/night background. Unlike the
    winner-take-all view, every type is shown independently, so a rhythmic
    modulation of one type appears as on/off stripes in that type's row.
    """
    groups = list(LOCATION_GROUPS.items())
    n_days, n_ct = len(days), len(call_types)
    n_hourbins = covered_min.shape[1]
    centers = x0 + (np.arange(n_hourbins) + 0.5) * (bin_minutes / 60.0)
    is_day = ((centers % 24) >= light_start) & ((centers % 24) < light_end)
    no_rec = covered_min <= 0
    color_rgb = np.array([mcolors.to_rgb(CALL_COLORS.get(ct, "#000000")) for ct in call_types])

    metric_by_group = {}
    for gname, locs in groups:
        sub = df[df["assigned_location"].isin(locs)]
        r = np.full((n_ct, n_days, n_hourbins), np.nan)
        for i, ct in enumerate(call_types):
            counts = counts_grid(sub[sub["event_type"] == ct], day0, n_days, n_hourbins, bin_minutes, x0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = counts / covered_min
            rr[no_rec] = np.nan
            r[i] = rr
        scale = np.nanpercentile(r, PCT, axis=(1, 2))
        scale[~np.isfinite(scale) | (scale == 0)] = np.nan
        metric_by_group[gname] = r / scale[:, None, None]

    bg_row = np.where(is_day[:, None], np.array(QUIET_DAY), np.array(QUIET_NIGHT))  # (n_hourbins, 3)

    def strip_img(gname, d):
        """(n_ct, n_hourbins, 3) RGB; row i = call_types[i]."""
        v = metric_by_group[gname][:, d, :]
        out = np.empty((n_ct, n_hourbins, 3))
        for i in range(n_ct):
            on = np.isfinite(v[i]) & (v[i] >= threshold)
            out[i] = np.where(on[:, None], color_rgb[i], bg_row)
            out[i][no_rec[d]] = NO_REC
        return out

    fig = plt.figure(figsize=(13, max(6.0, n_days * 0.9 + 1.8)))
    subfigs_all = fig.subfigures(n_days + 2, 1, height_ratios=[0.7] + [1] * n_days + [0.6], hspace=0.5)
    subfigs = subfigs_all[1:-1]
    day_labels = [d.strftime("%a %m-%d") for d in days]
    events = [(pd.Timestamp(t), lbl) for t, lbl in EVENTS_BY_DATE.get(date_folder, [])]

    for d in range(n_days):
        axs = subfigs[d].subplots(len(groups), 1, sharex=True, squeeze=False,
                                  gridspec_kw={"hspace": 0.18})[:, 0]
        axs[0].set_title(day_labels[d], loc="left", fontsize=9, pad=3)
        for gi, (gname, _locs) in enumerate(groups):
            ax = axs[gi]
            ax.imshow(strip_img(gname, d), aspect="auto", extent=(x0, x0 + 24, n_ct, 0),
                      origin="upper", interpolation="nearest", zorder=1)
            ax.set_yticks(np.arange(n_ct) + 0.5)
            ax.set_yticklabels(call_types, fontsize=4)
            ax.set_ylim(n_ct, 0)
            ax.set_ylabel(gname, rotation=0, ha="right", va="center", fontsize=7, labelpad=20)
            ax.set_xlim(x0, x0 + 24)
            is_bottom = (gi == len(groups) - 1)
            ticks = np.arange(x0, x0 + 25, 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels([_fmt_hour(t) for t in ticks] if is_bottom else [], fontsize=4)
            if d == n_days - 1 and is_bottom:
                ax.set_xlabel("hour of day")

        for ev_dt, lbl in events:
            if ((ev_dt - pd.Timedelta(hours=x0)).normalize() - day0).days != d:
                continue
            ev_x = x0 + (((ev_dt.hour * 60 + ev_dt.minute) - x0 * 60) % 1440) / 60.0
            for ax in axs:
                ax.axvline(ev_x, color="black", lw=1.2, zorder=6)
            axs[0].text(ev_x, 0.5, " " + lbl, rotation=90, color="black", fontsize=6,
                        fontweight="bold", va="center", ha="left",
                        transform=axs[0].get_xaxis_transform(), zorder=7)

    handles = [mpatches.Patch(color=CALL_COLORS.get(ct, "#000"), label=ct) for ct in call_types]
    handles += [mpatches.Patch(color=QUIET_DAY, label="quiet (light)"),
                mpatches.Patch(color=QUIET_NIGHT, label="quiet (dark)"),
                mpatches.Patch(color=NO_REC, label="no recording")]
    fig.legend(handles=handles, loc="upper right", fontsize=8, framealpha=0.95)
    fig.suptitle(
        f"Per-type threshold ethogram (thr={threshold} of each type's {PCT:g}th pct) — {date_folder}   "
        f"({len(df):,} calls, {n_days} days, {bin_minutes}-min bins; lights "
        f"{light_start:02d}:00–{light_end:02d}:00)",
        fontsize=12, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(date_folder, bin_minutes, out_dir, threshold, dominance, style, light_cycle=None):
    df = load_all_calls(date_folder)
    present = [ct for ct in CALL_TYPE_ORDER if ct in set(df["event_type"])]
    call_types = present + sorted(set(df["event_type"]) - set(CALL_TYPE_ORDER))
    light_start, light_end = light_cycle or get_light_cycle_for_month(date_folder)
    if (24 * 60) % bin_minutes or (ROW_START_HOUR * 60) % bin_minutes:
        raise SystemExit("--bin-min must divide 1440 and ROW_START_HOUR*60 evenly.")
    x0 = ROW_START_HOUR
    day0, days = day_axis(df, x0)
    n_hourbins = (24 * 60) // bin_minutes
    covered_min = coverage_grid(date_folder, day0, len(days), n_hourbins, bin_minutes, x0)
    if style == "per-type":
        out_path = out_dir / date_folder / f"ethogram_pertype_{date_folder}_{bin_minutes}min.png"
        plot_per_type_threshold(df, date_folder, day0, days, covered_min, call_types,
                                bin_minutes, light_start, light_end, x0, threshold, out_path)
    else:
        out_path = out_dir / date_folder / f"ethogram_categorical_{dominance}_{date_folder}_{bin_minutes}min.png"
        plot_categorical(df, date_folder, day0, days, covered_min, call_types,
                         bin_minutes, light_start, light_end, x0, threshold, dominance, out_path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES)
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES)
    ap.add_argument("--style", choices=["winner", "per-type"], default="per-type",
                    help="'winner' = one strip/location colored by dominant type; "
                         "'per-type' = one row per call type, binary-colored by threshold")
    ap.add_argument("--dominance", choices=["normalized", "raw"], default="raw",
                    help="(winner style) 'normalized' = most active vs each type's own 99th pct; "
                         "'raw' = literally most frequent call type")
    ap.add_argument("--threshold", type=float, default=THRESHOLD,
                    help="min metric to paint a cell (winner-raw: calls/min; else 0-1 fraction)")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms")
    ap.add_argument("--light-cycle", type=int, nargs=2, metavar=("ON", "OFF"), default=None)
    args = ap.parse_args()
    light_cycle = tuple(args.light_cycle) if args.light_cycle else None
    for date_folder in args.dates:
        run_for_date(date_folder, args.bin_min, args.out_dir, args.threshold,
                     args.dominance, args.style, light_cycle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
