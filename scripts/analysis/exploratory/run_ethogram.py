"""Ethogram of call types across a date-folder experiment timeline.

A date folder (e.g. 2026_02) is one continuous weeks-long experiment; the
per-experiment `exp` numbers are just restart segments. This script renders one
ethogram figure per date folder from the pooled ``all_calls_<date>.parquet``
(which already carries each call's absolute wall-clock time in
``start_time_real``).

For each location group it bins time and shows a heatmap of call rate
(calls / minute) per call type:

    rows  = call types
    x     = wall-clock time bins
    color = rate (per-row normalised by default, so rare types like alarm stay
            visible alongside warble)

Recording is NOT continuous: there are gaps between exp restarts. Coverage per
time-bin is computed from each experiment's ``file_times.csv`` (the slim sync
table); bins with no recording are greyed out, and partially-covered bins are
divided by their actual recorded minutes so the rate stays honest.

Locations are grouped: ``arena`` = arena_1 + arena_2, ``underground`` = underground.

Usage:
    python scripts/analysis/run_ethogram.py --dates 2026_02
    python scripts/analysis/run_ethogram.py --dates 2026_02 2025_10 --bin-min 30
"""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts" / "utils") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "utils"))
from light_cycle import get_light_cycle_for_month  # noqa: E402

# === Defaults (override on the command line) ==============================
# Date folder(s) to run on — EDIT HERE to switch experiment (e.g. ["2025_10"]).
# Available: 2024_12, 2025_03, 2025_07, 2025_10, 2026_02. Overridable with --dates.
DEFAULT_DATES = ["2026_02"]
BIN_MINUTES = 10                       # time-bin width
# Day/night comes from scripts/utils/light_cycle.py (per date folder); CLI can override.
CALL_TYPE_ORDER = ["alarm", "high-freq", "warble", "stacks", "newborn"]
LOCATION_GROUPS = {                    # display group -> assigned_location values pooled into it
    "arena": ["arena_1", "arena_2"],
    "underground": ["underground"],
}
NORMALIZE_PER_ROW = True               # scale each call-type row to its own max
# Folded (hour-of-day) layouts: clock hour each day-row starts at (row spans 24 h,
# time monotonic left->right). Edit to shift the day boundary; 4 = lights-on.
ROW_START_HOUR = 4
# Manually-noted events to annotate (red vertical line + label), per date folder.
# Each entry: (approx wall-clock datetime, label).
EVENTS_BY_DATE: dict[str, list[tuple[str, str]]] = {
    "2026_02": [("2026-03-01 23:00", "new litter born")],
}
# ==========================================================================

if platform.system() == "Windows":
    BASE_PROCESSED = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data\Audio")
else:
    BASE_PROCESSED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio")


def all_calls_path(date_folder: str) -> tuple[Path, str]:
    """Prefer the parquet cache; fall back to the CSV."""
    parquet = BASE_PROCESSED / "all_calls" / "parquet_cache" / f"all_calls_{date_folder}.parquet"
    if parquet.exists():
        return parquet, "parquet"
    csv = BASE_PROCESSED / "all_calls" / f"all_calls_{date_folder}.csv"
    return csv, "csv"


def load_all_calls(date_folder: str) -> pd.DataFrame:
    path, kind = all_calls_path(date_folder)
    if not path.exists():
        raise FileNotFoundError(f"No pooled calls file for {date_folder}: {path}")
    df = pd.read_parquet(path) if kind == "parquet" else pd.read_csv(path)
    df["start_time_real"] = pd.to_datetime(df["start_time_real"])
    print(f"{date_folder}: loaded {len(df):,} calls from {kind} ({path.name})")
    return df


def list_experiment_dirs(date_folder: str) -> list[Path]:
    folder = BASE_PROCESSED / date_folder
    if not folder.exists():
        raise FileNotFoundError(f"Date folder not found: {folder}")
    return sorted(
        (p for p in folder.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name),
    )


def make_bin_edges(df: pd.DataFrame, bin_minutes: int) -> pd.DatetimeIndex:
    freq = f"{bin_minutes}min"
    start = df["start_time_real"].min().floor(freq)
    end = df["start_time_real"].max().ceil(freq)
    return pd.date_range(start, end, freq=freq)


def recording_coverage_minutes(date_folder: str, bin_edges: pd.DatetimeIndex) -> np.ndarray:
    """Recorded minutes per time-bin, summed over all experiments' file_times.csv.

    Each ``file_times.csv`` row is a recorded chunk [start, end]; chunks are
    contiguous within an experiment and gappy between experiment restarts. We
    sum each chunk's overlap with every bin it touches.
    """
    edges_ns = bin_edges.asi8                       # int64 nanoseconds
    n_bins = len(bin_edges) - 1
    covered_ns = np.zeros(n_bins, dtype=np.float64)

    for exp_dir in list_experiment_dirs(date_folder):
        ft_path = exp_dir / "file_times.csv"
        if not ft_path.exists():
            continue
        ft = pd.read_csv(ft_path)
        starts = pd.to_datetime(ft["start_date"] + " " + ft["start_time"]).values.astype("int64")
        ends = pd.to_datetime(ft["end_date"] + " " + ft["end_time"]).values.astype("int64")
        for s, e in zip(starts, ends):
            i0 = max(np.searchsorted(edges_ns, s, "right") - 1, 0)
            i1 = min(np.searchsorted(edges_ns, e, "right") - 1, n_bins - 1)
            for b in range(i0, i1 + 1):
                lo = max(s, edges_ns[b])
                hi = min(e, edges_ns[b + 1])
                if hi > lo:
                    covered_ns[b] += hi - lo
    return covered_ns / 1e9 / 60.0                  # ns -> seconds -> minutes


def rate_matrix(
    df_group: pd.DataFrame,
    bin_edges: pd.DatetimeIndex,
    call_types: list[str],
    covered_min: np.ndarray,
) -> np.ndarray:
    """(n_call_types, n_bins) array of calls/min; NaN where no recording."""
    cats = pd.cut(df_group["start_time_real"], bins=bin_edges, right=False)
    counts = (
        df_group.groupby([df_group["event_type"], cats], observed=False)
        .size()
        .unstack(fill_value=0)
        .reindex(index=call_types, fill_value=0)
        .reindex(columns=cats.cat.categories, fill_value=0)
    )
    mat = counts.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = mat / covered_min[np.newaxis, :]
    rate[:, covered_min <= 0] = np.nan
    return rate


def _shade_nights(ax, bin_edges: pd.DatetimeIndex, light_start: int, light_end: int) -> None:
    """Translucent shading over night spans (outside [light_start, light_end))."""
    day0 = bin_edges[0].normalize()
    day1 = bin_edges[-1].normalize() + pd.Timedelta(days=1)
    for day in pd.date_range(day0, day1, freq="D"):
        dusk = day + pd.Timedelta(hours=light_end)
        dawn_next = day + pd.Timedelta(days=1, hours=light_start)
        ax.axvspan(mdates.date2num(dusk), mdates.date2num(dawn_next),
                   color="black", alpha=0.07, lw=0, zorder=3)


def plot_ethogram(
    df: pd.DataFrame,
    date_folder: str,
    bin_edges: pd.DatetimeIndex,
    covered_min: np.ndarray,
    call_types: list[str],
    light_start: int,
    light_end: int,
    out_path: Path,
) -> None:
    groups = list(LOCATION_GROUPS.items())
    x = mdates.date2num(bin_edges.to_pydatetime())
    y = np.arange(len(call_types) + 1)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgrey")

    fig, axes = plt.subplots(
        len(groups), 1, figsize=(16, 2.6 * len(groups) + 1.5),
        sharex=True, squeeze=False,
    )
    axes = axes[:, 0]

    for ax, (group_name, locations) in zip(axes, groups):
        sub = df[df["assigned_location"].isin(locations)]
        rate = rate_matrix(sub, bin_edges, call_types, covered_min)

        if NORMALIZE_PER_ROW:
            row_max = np.nanmax(rate, axis=1, keepdims=True)
            row_max[row_max == 0] = np.nan
            disp = rate / row_max
            vmin, vmax = 0.0, 1.0
        else:
            disp = rate
            vmin, vmax = 0.0, np.nanpercentile(rate, 99)

        mesh = ax.pcolormesh(
            x, y, np.ma.masked_invalid(disp),
            cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
        )
        _shade_nights(ax, bin_edges, light_start, light_end)
        ax.set_yticks(np.arange(len(call_types)) + 0.5)
        ax.set_yticklabels(call_types)
        ax.set_ylim(0, len(call_types))
        ax.invert_yaxis()
        n_calls = len(sub)
        ax.set_title(f"{group_name}  (n = {n_calls:,} calls)", fontsize=11, loc="left")

        cbar = fig.colorbar(mesh, ax=ax, pad=0.01, fraction=0.025)
        cbar.set_label("rate / row max" if NORMALIZE_PER_ROW else "calls / min")

    axes[-1].set_xlim(x[0], x[-1])
    axes[-1].xaxis.set_major_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[-1].set_xlabel("date")
    for label in axes[-1].get_xticklabels():
        label.set_rotation(0)

    rec_h = covered_min.sum() / 60.0
    span_h = (bin_edges[-1] - bin_edges[0]).total_seconds() / 3600.0
    fig.suptitle(
        f"Call ethogram — {date_folder}   "
        f"({len(df):,} calls, {rec_h:.0f} h recorded over {span_h / 24:.1f} days, "
        f"{BIN_MINUTES}-min bins; night shaded)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


# --- daily ("actogram") layout: rows = calendar days, x = hour-of-day ----------

def day_axis(df: pd.DataFrame, x0: int) -> tuple[pd.Timestamp, pd.DatetimeIndex]:
    """Row dates for an actogram whose day starts at hour ``x0``.

    A "circadian day" runs [x0:00 of date D, x0:00 of D+1); a timestamp belongs
    to the row of ``(t - x0 h).date``. ``day0`` is the first such row date.
    """
    shifted = df["start_time_real"] - pd.Timedelta(hours=x0)
    day0 = shifted.min().normalize()
    days = pd.date_range(day0, shifted.max().normalize(), freq="D")
    return day0, days


def counts_grid(
    df_group: pd.DataFrame, day0: pd.Timestamp, n_days: int,
    n_hourbins: int, bin_minutes: int, x0: int = 0,
) -> np.ndarray:
    """(n_days, n_hourbins) raw call counts for one call-type × location subset.

    Rows are circadian days starting at hour ``x0``; within a row, time runs
    monotonically from ``x0`` (left) to ``x0 + 24`` (right).
    """
    t = df_group["start_time_real"]
    shifted = t - pd.Timedelta(hours=x0)
    day_idx = (shifted.dt.normalize() - day0).dt.days.to_numpy()
    mins = ((t.dt.hour * 60 + t.dt.minute) - x0 * 60) % 1440
    hb = (mins // bin_minutes).to_numpy().astype(int)
    grid = np.zeros((n_days, n_hourbins), dtype=float)
    ok = (day_idx >= 0) & (day_idx < n_days)
    np.add.at(grid, (day_idx[ok], hb[ok]), 1.0)
    return grid


def coverage_grid(
    date_folder: str, day0: pd.Timestamp, n_days: int,
    n_hourbins: int, bin_minutes: int, x0: int = 0,
) -> np.ndarray:
    """(n_days, n_hourbins) recorded minutes per (circadian-day, hour) cell."""
    cov = np.zeros((n_days, n_hourbins), dtype=float)
    x0h = pd.Timedelta(hours=x0)
    for exp_dir in list_experiment_dirs(date_folder):
        ft_path = exp_dir / "file_times.csv"
        if not ft_path.exists():
            continue
        ft = pd.read_csv(ft_path)
        starts = pd.to_datetime(ft["start_date"] + " " + ft["start_time"])
        ends = pd.to_datetime(ft["end_date"] + " " + ft["end_time"])
        for s, e in zip(starts, ends):
            cur = s
            while cur < e:                       # walk the chunk cell-by-cell
                d_idx = ((cur - x0h).normalize() - day0).days
                mod = cur.hour * 60 + cur.minute
                cell_end = cur.normalize() + pd.Timedelta(minutes=(mod // bin_minutes + 1) * bin_minutes)
                seg_end = min(e, cell_end)
                if 0 <= d_idx < n_days:
                    hb = int(((mod - x0 * 60) % 1440) // bin_minutes)
                    cov[d_idx, hb] += (seg_end - cur).total_seconds()
                cur = seg_end
    return cov / 60.0


def _draw_light_ribbon(ax, light_start: int, light_end: int, x0: float) -> None:
    """Night ribbon over [x0, x0+24]: grey = night, light-yellow carve = light hours."""
    ax.set_xlim(x0, x0 + 24)
    ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((x0, 0), 24, 1, facecolor="0.55", edgecolor="black", lw=0.6))
    ax.axvspan(light_start, light_end, color="#fffdd0", lw=0)   # carve the light period
    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _fmt_hour(t: float) -> str:
    return f"{int(round(t)) % 24:02d}:00"


def _hour_ticks(ax, x0: float, *, labels: bool, fontsize: int = 6) -> None:
    """Tick every hour across [x0, x0+24]; labels are clock hours (mod 24)."""
    ticks = np.arange(x0, x0 + 24 + 1e-6, 1.0)
    ax.set_xticks(ticks)
    if labels:
        ax.set_xticklabels([_fmt_hour(t) for t in ticks], fontsize=fontsize)
    else:
        ax.set_xticklabels([])


def plot_ethogram_daily(
    df: pd.DataFrame, date_folder: str, day0: pd.Timestamp, days: pd.DatetimeIndex,
    covered_min: np.ndarray, call_types: list[str], bin_minutes: int,
    light_start: int, light_end: int, x0: float, out_path: Path,
) -> None:
    groups = list(LOCATION_GROUPS.items())
    n_days = len(days)
    n_hourbins = covered_min.shape[1]
    x_edges = x0 + np.arange(0, 24 + 1e-9, bin_minutes / 60.0)
    y_edges = np.arange(n_days + 1)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgrey")
    no_rec = covered_min <= 0

    n_rows = len(call_types) + 1                  # +1 ribbon row on top
    fig, axes = plt.subplots(
        n_rows, len(groups),
        figsize=(5.0 * len(groups) + 1.5, 0.16 * n_days * len(call_types) + 2.0),
        sharex=True, squeeze=False,
        gridspec_kw={"height_ratios": [0.4] + [1] * len(call_types)},
    )

    day_labels = [d.strftime("%m-%d") for d in days]
    mesh = None
    for j, (group_name, locations) in enumerate(groups):
        _draw_light_ribbon(axes[0, j], light_start, light_end, x0)
        axes[0, j].set_title(f"{group_name}", fontsize=12)
        sub_loc = df[df["assigned_location"].isin(locations)]
        for i, ct in enumerate(call_types):
            ax = axes[i + 1, j]
            counts = counts_grid(sub_loc[sub_loc["event_type"] == ct],
                                 day0, n_days, n_hourbins, bin_minutes, x0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rate = counts / covered_min
            rate[no_rec] = np.nan
            row_max = np.nanmax(rate) if np.isfinite(np.nanmax(rate)) else 1.0
            disp = rate / row_max if (NORMALIZE_PER_ROW and row_max > 0) else rate
            mesh = ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(disp),
                                 cmap=cmap, vmin=0.0, vmax=1.0 if NORMALIZE_PER_ROW else None,
                                 shading="flat")
            ax.invert_yaxis()
            if j == 0:
                ax.set_ylabel(ct, fontsize=11)
                ax.set_yticks(np.arange(n_days) + 0.5)
                ax.set_yticklabels(day_labels, fontsize=6)
            else:
                ax.set_yticks([])
            ax.set_xlim(x0, x0 + 24)
            _hour_ticks(ax, x0, labels=(i == len(call_types) - 1), fontsize=6)

    for j in range(len(groups)):
        axes[-1, j].set_xlabel("hour of day")

    if mesh is not None:
        cbar = fig.colorbar(mesh, ax=axes[1:, :], fraction=0.02, pad=0.02)
        cbar.set_label("rate / panel max" if NORMALIZE_PER_ROW else "calls / min")

    rec_h = covered_min.sum() / 60.0
    fig.suptitle(
        f"Call ethogram (daily) — {date_folder}   "
        f"({len(df):,} calls, {rec_h:.0f} h recorded over {n_days} days, "
        f"{bin_minutes}-min bins; night = dark ribbon)",
        fontsize=13,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def _norm_scales(
    rate_by_group: dict[str, np.ndarray], groups: list, n_ct: int, norm: str, pct: float,
) -> tuple[dict[str, np.ndarray], str]:
    """Per-(group, call-type) divisor for display, by normalization mode.

    Uses the ``pct`` percentile (not the raw max) so a single bout doesn't flatten
    the rest. Modes:
      - 'per-type-loc' : each call type scaled within its own location
      - 'per-type'     : each call type scaled by a max shared across locations
      - 'absolute'     : one shared scale across all call types and locations
    """
    gnames = [g for g, _ in groups]
    if norm == "absolute":
        allvals = np.concatenate([rate_by_group[g].ravel() for g in gnames])
        glob = np.nanpercentile(allvals, pct) if np.isfinite(np.nanmax(allvals)) else np.nan
        scales = {g: np.full(n_ct, glob) for g in gnames}
        label = f"rate / global {pct:g}th pct"
    elif norm == "per-type":
        shared = np.array([
            np.nanpercentile(np.concatenate([rate_by_group[g][i].ravel() for g in gnames]), pct)
            if np.isfinite(np.nanmax([np.nanmax(rate_by_group[g][i]) for g in gnames])) else np.nan
            for i in range(n_ct)
        ])
        scales = {g: shared.copy() for g in gnames}
        label = f"rate / call-type {pct:g}th pct"
    else:  # per-type-loc
        scales = {g: np.nanpercentile(rate_by_group[g], pct, axis=(1, 2)) for g in gnames}
        label = f"rate / call-type×loc {pct:g}th pct"
    for g in gnames:
        s = scales[g]
        s[~np.isfinite(s) | (s == 0)] = np.nan
    return scales, label


def plot_ethogram_stacked(
    df: pd.DataFrame, date_folder: str, day0: pd.Timestamp, days: pd.DatetimeIndex,
    covered_min: np.ndarray, call_types: list[str], bin_minutes: int,
    light_start: int, light_end: int, x0: float,
    norm: str, pct: float, out_path: Path,
) -> None:
    """One mini two-panel strip per day, stacked vertically.

    Each day: a thin night-ribbon rectangle, then arena (5 call-type rows) above
    underground (5 rows); x = hour of day starting ``x0`` (a few hours before
    lights-on). Normalisation is controlled by ``norm`` (see :func:`_norm_scales`)
    using the ``pct`` percentile, so days are comparable.
    """
    groups = list(LOCATION_GROUPS.items())
    n_days, n_ct = len(days), len(call_types)
    n_hourbins = covered_min.shape[1]
    x_edges = x0 + np.arange(0, 24 + 1e-9, bin_minutes / 60.0)
    y_edges = np.arange(n_ct + 1)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("lightgrey")
    no_rec = covered_min <= 0

    rate_by_group = {}
    for gname, locations in groups:
        sub = df[df["assigned_location"].isin(locations)]
        r = np.full((n_ct, n_days, n_hourbins), np.nan)
        for i, ct in enumerate(call_types):
            counts = counts_grid(sub[sub["event_type"] == ct],
                                 day0, n_days, n_hourbins, bin_minutes, x0)
            with np.errstate(divide="ignore", invalid="ignore"):
                rr = counts / covered_min
            rr[no_rec] = np.nan
            r[i] = rr
        rate_by_group[gname] = r
    scale_by_group, cbar_label = _norm_scales(rate_by_group, groups, n_ct, norm, pct)

    fig = plt.figure(figsize=(13, max(6.0, n_days * 0.9 + 1.8)))
    # Empty first/last sub-figure rows reserve headroom for the title and the
    # bottom x-axis labels (which otherwise clip off the figure edge).
    subfigs_all = fig.subfigures(
        n_days + 2, 1, height_ratios=[0.6] + [1] * n_days + [0.5], hspace=0.5)
    for sf in subfigs_all:
        sf.patch.set_alpha(0.0)               # transparent, so the day/night bands show through
    subfigs = subfigs_all[1:-1]
    day_labels = [d.strftime("%a %m-%d") for d in days]

    events = [(pd.Timestamp(t), lbl) for t, lbl in EVENTS_BY_DATE.get(date_folder, [])]
    top_ax = bot_ax = None

    for d in range(n_days):
        axs = subfigs[d].subplots(
            len(groups), 1, sharex=True, squeeze=False,
            gridspec_kw={"hspace": 0.0},
        )[:, 0]
        if d == 0:
            top_ax = axs[0]
        if d == n_days - 1:
            bot_ax = axs[-1]
        axs[0].set_title(day_labels[d], loc="left", fontsize=9, pad=3)
        for gi, (gname, _locations) in enumerate(groups):
            ax = axs[gi]
            disp = rate_by_group[gname][:, d, :] / scale_by_group[gname][:, None]
            ax.pcolormesh(x_edges, y_edges, np.ma.masked_invalid(disp),
                          cmap=cmap, vmin=0.0, vmax=1.0, shading="flat", zorder=1)
            ax.set_yticks(np.arange(n_ct) + 0.5)
            ax.set_yticklabels(call_types, fontsize=4)
            ax.set_ylim(0, n_ct)
            ax.invert_yaxis()
            ax.set_ylabel(gname, rotation=0, ha="right", va="center", fontsize=7, labelpad=20)
            # White separator line between the stacked location panels.
            if gi < len(groups) - 1:
                ax.spines["bottom"].set(visible=True, color="white", linewidth=2.5, zorder=5)
            if gi > 0:
                ax.spines["top"].set_visible(False)
            ax.set_xlim(x0, x0 + 24)
            is_bottom = (gi == len(groups) - 1)          # underground row of each day
            _hour_ticks(ax, x0, labels=is_bottom, fontsize=4)
            if d == n_days - 1 and is_bottom:
                ax.set_xlabel("hour of day")

        # Event markers (e.g. new litter) on the matching circadian-day row.
        for ev_dt, lbl in events:
            if ((ev_dt - pd.Timedelta(hours=x0)).normalize() - day0).days != d:
                continue
            ev_x = x0 + (((ev_dt.hour * 60 + ev_dt.minute) - x0 * 60) % 1440) / 60.0
            for ax in axs:
                ax.axvline(ev_x, color="red", lw=1.3, zorder=6)
            axs[0].text(ev_x, 0.5, " " + lbl, rotation=90, color="red",
                        fontsize=6, fontweight="bold", va="center", ha="left",
                        transform=axs[0].get_xaxis_transform(), zorder=7)

    # Day/night background bands, bounded vertically to the data rows (so they
    # don't extend above the first row or below the last), aligned to the data
    # columns' x-extent so they line up down the figure.
    fig.canvas.draw()
    inv = fig.transFigure.inverted()
    pad = 0.008                                          # let bands run a little past the rows
    y_top = inv.transform((0, top_ax.bbox.ymax))[1] + pad
    y_bot = inv.transform((0, bot_ax.bbox.ymin))[1] - pad
    L = plt.rcParams["figure.subplot.left"]
    R = plt.rcParams["figure.subplot.right"]
    fx = lambda h: L + (h - x0) / 24.0 * (R - L)
    bg = fig.add_axes([0, y_bot, 1, y_top - y_bot], zorder=-10)
    bg.set_xlim(0, 1); bg.set_ylim(0, 1); bg.axis("off")
    bg.axvspan(fx(x0), fx(light_start), color="0.85")              # night (grey)
    bg.axvspan(fx(light_end), fx(x0 + 24), color="0.85")
    bg.axvspan(fx(light_start), fx(light_end), color="#fff6c8")    # day (light yellow)

    sm = plt.cm.ScalarMappable(norm=plt.Normalize(0, 1), cmap=cmap)
    cax = fig.add_axes([0.93, 0.35, 0.012, 0.3])
    fig.colorbar(sm, cax=cax).set_label(cbar_label, fontsize=8)

    rec_h = covered_min.sum() / 60.0
    fig.suptitle(
        f"Call ethogram (per-day strips) — {date_folder}   "
        f"({len(df):,} calls, {rec_h:.0f} h over {n_days} days, {bin_minutes}-min bins; "
        f"lights {light_start:02d}:00–{light_end:02d}:00 = yellow / night = grey; norm={norm})",
        fontsize=13, y=0.995,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)        # explicit margins; no bbox='tight' (it eats the top gap)
    plt.close(fig)
    print(f"{date_folder}: wrote {out_path}")


def run_for_date(
    date_folder: str, bin_minutes: int, out_dir: Path, layout: str,
    light_cycle: tuple[int, int] | None = None,
    norm: str = "per-type", pct: float = 99.0,
) -> None:
    df = load_all_calls(date_folder)
    present = [ct for ct in CALL_TYPE_ORDER if ct in set(df["event_type"])]
    extra = sorted(set(df["event_type"]) - set(CALL_TYPE_ORDER))
    if extra:
        print(f"{date_folder}: note — unlisted event types present, appending: {extra}")
    call_types = present + extra

    light_start, light_end = light_cycle or get_light_cycle_for_month(date_folder)
    print(f"{date_folder}: light cycle {light_start:02d}:00–{light_end:02d}:00")

    if layout == "timeline":
        bin_edges = make_bin_edges(df, bin_minutes)
        covered_min = recording_coverage_minutes(date_folder, bin_edges)
        out_path = out_dir / date_folder / f"ethogram_{date_folder}_{bin_minutes}min.png"
        plot_ethogram(df, date_folder, bin_edges, covered_min, call_types,
                      light_start, light_end, out_path)
    elif layout in ("actogram", "stacked"):
        if (24 * 60) % bin_minutes != 0:
            raise SystemExit("--bin-min must divide 1440 (24 h) evenly for this layout.")
        x0 = ROW_START_HOUR                       # clock hour each row starts at
        if (x0 * 60) % bin_minutes != 0:
            raise SystemExit("ROW_START_HOUR * 60 must be divisible by --bin-min.")
        day0, days = day_axis(df, x0)
        n_hourbins = (24 * 60) // bin_minutes
        covered_min = coverage_grid(date_folder, day0, len(days), n_hourbins, bin_minutes, x0)
        if layout == "actogram":
            out_path = out_dir / date_folder / f"ethogram_actogram_{date_folder}_{bin_minutes}min.png"
            plot_ethogram_daily(df, date_folder, day0, days, covered_min, call_types,
                                bin_minutes, light_start, light_end, x0, out_path)
        else:
            out_path = out_dir / date_folder / f"ethogram_stacked_{date_folder}_{bin_minutes}min.png"
            plot_ethogram_stacked(df, date_folder, day0, days, covered_min, call_types,
                                  bin_minutes, light_start, light_end, x0,
                                  norm, pct, out_path)
    else:
        raise SystemExit(f"unknown layout: {layout}")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dates", nargs="+", default=DEFAULT_DATES,
                    help="date folder(s), e.g. 2026_02 2025_10")
    ap.add_argument("--bin-min", type=int, default=BIN_MINUTES, help="time-bin width in minutes")
    ap.add_argument("--layout", choices=["timeline", "actogram", "stacked"], default="stacked",
                    help="'stacked' = one mini two-panel strip per day (arena/underground), "
                         "x = hour-of-day, night ribbon above each day; "
                         "'actogram' = days as rows in a grid of call-type × location heatmaps; "
                         "'timeline' = one continuous wall-clock x-axis")
    ap.add_argument("--out-dir", type=Path, default=BASE_PROCESSED / "ethograms",
                    help="output root (one subfolder per date)")
    ap.add_argument("--light-cycle", type=int, nargs=2, metavar=("ON", "OFF"), default=None,
                    help="override lights-on/off hours (default: per-date from light_cycle.py)")
    ap.add_argument("--norm", choices=["per-type", "per-type-loc", "absolute"], default="per-type",
                    help="color normalization (stacked layout): 'per-type' shares a scale across "
                         "locations per call type; 'per-type-loc' scales each location separately; "
                         "'absolute' uses one scale for all call types")
    ap.add_argument("--norm-pct", type=float, default=99.0,
                    help="percentile used as the normalization ceiling (default 99)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    light_cycle = tuple(args.light_cycle) if args.light_cycle else None
    for date_folder in args.dates:
        run_for_date(date_folder, args.bin_min, args.out_dir, args.layout,
                     light_cycle, args.norm, args.norm_pct)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
