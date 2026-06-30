"""Bout detection for consecutive vocalizations of the same call type.

A "bout" is a run of consecutive calls of one event_type within a single
(date_folder, exp, assigned_location) group, where each silent gap (ICG)
between adjacent calls sits inside [min_icg_s, max_icg_s]. Gaps outside
that window start a new bout.

This module is the single source of truth for thresholds. If you tune a
threshold here, every notebook that imports `detect_bouts` will see the
change next reload - no chance of stale parquets with different settings.

Output column naming: every call to `detect_bouts` writes a set of columns
prefixed by `prefix` (default "bout"):
    {prefix}_id, {prefix}_size, {prefix}_position, {prefix}_kind
plus the shared `ici_s` and `icg_s` (same regardless of scale - they're
call-level quantities).

For multi-scale call types (currently only alarm: bouts at 2 s,
events at 30 s, possibly more later), call `detect_bouts` repeatedly
with different prefixes - or use the `detect_alarm_scales` helper which
does this for you.

Terminology:
    ICI (inter-call interval) - this.START - previous.START between consecutive
                                calls. Period. 1/median(ICI) = calling rate in Hz.
    ICG (inter-call gap)      - this.START - previous.STOP between consecutive
                                calls. Silent gap. Used to detect bouts.

At the group scale (one row per bout / event in summarize_scale's output),
the analogous quantities are just called `interval_s` and `gap_s` - no
prefix needed because the scale is implicit in which DataFrame (bouts_meta
vs events_meta) you're looking at.

Public entry points:
    BOUT_THRESHOLDS        : per-call-type thresholds (single-scale defaults)
    ALARM_SCALES           : multi-scale config for alarm calls
    detect_bouts           : add bout columns to one call type's DataFrame
    detect_bouts_for_types : convenience loop over multiple call types
    detect_alarm_scales    : apply ALARM_SCALES to a DataFrame of alarm calls
    summarize_scale        : roll up calls -> one row per bout/event (bouts_meta / events_meta)
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# Single source of truth for thresholds.
# --------------------------------------------------------------------------
# Tune these here, nowhere else.
#
# Bout detection compares the silent GAP (icg_s) to these thresholds:
# - min_icg_s    : sub-threshold gaps are treated as artifacts and start a new
#                  bout. Set to None to skip the lower bound (alarm calls
#                  don't suffer from sub-50 ms segmentation overlap).
# - max_icg_s    : gaps above this are between-bout silences.
# - min_bout_size: bouts with >= this many calls are labelled "in_bout".
#                  Bouts with 2..(min_bout_size - 1) calls get "small_bout"
#                  and are typically dropped from comparisons; single calls
#                  get "singleton".
BOUT_THRESHOLDS: dict[str, dict[str, float | int | None]] = {
    "warble":    dict(min_icg_s=0.05, max_icg_s=0.20, min_bout_size=5),
    "high-freq": dict(min_icg_s=0.05, max_icg_s=0.20, min_bout_size=5),
    "alarm":     dict(min_icg_s=None, max_icg_s=2.0,  min_bout_size=5),
    "stacks":    dict(min_icg_s=None, max_icg_s=2.0,  min_bout_size=5),
}

# Multi-scale config for alarm calls. The "bout" scale uses the same
# thresholds as BOUT_THRESHOLDS["alarm"] (kept in sync by reference below).
# Add a third scale by appending another dict here, e.g.
#     dict(name="episode", min_icg_s=None, max_icg_s=300.0, min_bout_size=2)
ALARM_SCALES: list[dict] = [
    dict(name="bout",  **BOUT_THRESHOLDS["alarm"]),
    dict(name="event", min_icg_s=None, max_icg_s=30.0, min_bout_size=5),
]

# What scopes a bout: bouts cannot span across different values of these columns.
DEFAULT_GROUP_COLS: tuple[str, ...] = ("date_folder", "exp", "assigned_location")


def detect_bouts(
    df: pd.DataFrame,
    call_type: str,
    *,
    prefix: str = "bout",
    with_kind: bool = True,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_icg_s: float | None = None,
    max_icg_s: float | None = None,
    min_bout_size: int | None = None,
) -> pd.DataFrame:
    """Add ici_s, icg_s, and {prefix}_(id|size|position|kind) columns to df.

    The DataFrame is assumed to already be filtered to a single call type
    (no check). `call_type` is used only to look up defaults in
    BOUT_THRESHOLDS; pass keyword args to override them for sweeps or for
    multi-scale work.

    Parameters
    ----------
    df : DataFrame
        One row per call. Must have `start_time_real`, `stop_time_real`, and
        the columns in `group_cols` (start/stop are datetime-like).
    call_type : str
        Key into BOUT_THRESHOLDS. Provides defaults for the threshold kwargs.
    prefix : str, default "bout"
        Prefix for the output columns.
    with_kind : bool, default True
        Whether to add a `{prefix}_kind` (singleton/in_bout/small_bout) column.
        Set False for coarser scales (e.g. event, episode) where a kind label
        based on call count duplicates information already on the finer scale.
    group_cols : tuple of str, optional
        Columns that scope "consecutive". Default: (date_folder, exp, assigned_location).
    min_icg_s, max_icg_s, min_bout_size : optional overrides
        If None, pulled from BOUT_THRESHOLDS[call_type]. Bout detection
        compares the silent gap (icg_s) to [min_icg_s, max_icg_s].

    Returns
    -------
    DataFrame
        Sorted by (*group_cols, start_time_real), reset_index. New columns:
            ici_s             - inter-call interval (this.start - prev.start), seconds
            icg_s             - inter-call gap      (this.start - prev.stop ), seconds
            {prefix}_id       - integer, unique per consecutive run
            {prefix}_size     - number of calls in this call's bout
            {prefix}_position - 1-based index of this call within its bout
            {prefix}_kind     - "singleton" | "in_bout" | "small_bout"
                                (only if with_kind=True)
    """
    cfg = BOUT_THRESHOLDS.get(call_type, {})
    if min_icg_s     is None: min_icg_s     = cfg.get("min_icg_s")
    if max_icg_s     is None: max_icg_s     = cfg.get("max_icg_s")
    if min_bout_size is None: min_bout_size = cfg.get("min_bout_size")

    if max_icg_s is None or min_bout_size is None:
        raise ValueError(
            f"detect_bouts(call_type={call_type!r}, prefix={prefix!r}): "
            f"max_icg_s and min_bout_size must be set, either via "
            f"BOUT_THRESHOLDS[{call_type!r}] or as kwargs."
        )

    group_cols = list(group_cols)
    df = df.sort_values([*group_cols, "start_time_real"]).reset_index(drop=True)

    # Two flavors of inter-call timing, computed within each group:
    #   icg_s : silent gap     (this.start - previous.stop)  - used for bout detection
    #   ici_s : start-to-start (this.start - previous.start) - period; 1/ici = Hz
    grp = df.groupby(group_cols)
    prev_stop  = grp["stop_time_real"].shift(1)
    prev_start = grp["start_time_real"].shift(1)
    df["icg_s"] = (df["start_time_real"] - prev_stop ).dt.total_seconds()
    df["ici_s"] = (df["start_time_real"] - prev_start).dt.total_seconds()

    # A new bout starts when no predecessor, or the GAP is outside the window.
    too_long  = df["icg_s"] > max_icg_s
    too_short = (
        df["icg_s"] < min_icg_s
        if min_icg_s is not None
        else pd.Series(False, index=df.index)
    )
    is_new = df["icg_s"].isna() | too_long | too_short

    id_col, size_col, pos_col, kind_col = (
        f"{prefix}_id", f"{prefix}_size", f"{prefix}_position", f"{prefix}_kind",
    )
    df[id_col]   = is_new.cumsum()
    df[size_col] = df.groupby(id_col)[id_col].transform("size")
    df[pos_col]  = df.groupby(id_col).cumcount() + 1
    if with_kind:
        df[kind_col] = np.select(
            [df[size_col] == 1, df[size_col] >= min_bout_size],
            ["singleton", "in_bout"],
            default="small_bout",
        )
    return df


def detect_bouts_for_types(
    df: pd.DataFrame,
    call_types: list[str],
    **detect_kwargs,
) -> pd.DataFrame:
    """Run `detect_bouts` independently per call type, then concatenate.

    Bout detection is per-type because a warble followed by a high-freq call
    shouldn't share a bout - they're different vocalizations.

    Parameters
    ----------
    df : DataFrame
        Calls of one or more types, with an `event_type` column.
    call_types : list of str
        Which event_types to keep + detect bouts for. Any type not in this
        list is dropped.
    **detect_kwargs
        Forwarded to `detect_bouts` (e.g. prefix=, group_cols=).

    Returns
    -------
    DataFrame
        Concatenation of `detect_bouts(df_one_type, call_type, **kwargs)`
        for each non-empty type. Index is reset.
    """
    parts = []
    for ct in call_types:
        sub = df[df["event_type"] == ct]
        if not sub.empty:
            parts.append(detect_bouts(sub, ct, **detect_kwargs))
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True)


def detect_alarm_scales(
    df: pd.DataFrame,
    scales: list[dict] | None = None,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
) -> pd.DataFrame:
    """Apply detect_bouts to alarm calls once per scale in `scales`.

    After the loop the returned DataFrame carries one (id, size, position[, kind])
    column set per scale, all on the same rows. E.g. with the default
    ALARM_SCALES list you get the call-level timing pair (ici_s, icg_s) plus:
        bout_id, bout_size, bout_position, bout_kind,
        event_id, event_size, event_position

    Parameters
    ----------
    df : DataFrame
        Calls already filtered to event_type == "alarm".
    scales : list of dict, optional
        Each dict needs the keys: name, min_icg_s, max_icg_s, min_bout_size.
        Defaults to ALARM_SCALES.
    group_cols : tuple of str, optional
        Forwarded to detect_bouts. Default (date_folder, exp, assigned_location).
    """
    scales = scales if scales is not None else ALARM_SCALES
    # Only the finest (first) scale gets a {prefix}_kind column; coarser scales
    # repackage the same calls so a kind label there would be redundant.
    for i, s in enumerate(scales):
        df = detect_bouts(
            df,
            call_type="alarm",
            prefix=s["name"],
            with_kind=(i == 0),
            group_cols=group_cols,
            min_icg_s=s["min_icg_s"],
            max_icg_s=s["max_icg_s"],
            min_bout_size=s["min_bout_size"],
        )
    return df


def summarize_scale(
    df: pd.DataFrame,
    prefix: str = "bout",
    *,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    extra_first_cols: tuple[str, ...] = (),
    finer_scale: str | None = None,
) -> pd.DataFrame:
    """Roll up a calls DataFrame into one row per {prefix} (bouts_meta / events_meta).

    Each output row summarizes one bout/event:
        * n_calls (alias for `{prefix}_size`) and, if `{prefix}_kind` exists, kind.
        * span: start_time, stop_time, duration_s.
        * within-{prefix} timing stats (computed over position > 1, i.e.
          inside the bout/event only):
              mean_icg_s, std_icg_s, cv_icg  - based on silent gap
              mean_ici_s, std_ici_s, cv_ici  - based on start-to-start period
              (1 / mean_ici_s = within-bout calling rate in Hz)
        * inter-{prefix} timing (between consecutive bouts/events of this scale):
              gap_s      - this.start - prev.stop
              interval_s - this.start - prev.start
          Both scoped by `group_cols`.
        * group_cols + extra_first_cols copied from each {prefix}'s first call.
        * `n_{finer_scale}s` if finer_scale is given (e.g. n_bouts for events).

    Parameters
    ----------
    df : DataFrame
        A calls-level DataFrame that already has `{prefix}_id`, `{prefix}_size`,
        `{prefix}_position`, plus `ici_s`, `icg_s`, `start_time_real`,
        `stop_time_real`, and the columns in `group_cols`. (i.e. run
        `detect_bouts` or `detect_alarm_scales` first.) `{prefix}_kind` is
        included in the output if present on `df`, otherwise omitted.
    prefix : str, default "bout"
        Which scale to summarize.
    group_cols : tuple of str, optional
        IGG / IGI don't cross these. Default: (date_folder, exp, assigned_location).
    extra_first_cols : tuple of str, optional
        Columns to copy from each {prefix}'s first call. Useful for things
        like classifier confidence (e.g. ("meanprob_alarm",)).
    finer_scale : str, optional
        If given (e.g. "bout" when summarizing the event scale), the output
        gains an `n_{finer_scale}s` column counting how many unique
        `{finer_scale}_id`s fall inside each row.

    Returns
    -------
    DataFrame indexed by `{prefix}_id`.
    """
    id_col   = f"{prefix}_id"
    size_col = f"{prefix}_size"
    kind_col = f"{prefix}_kind"
    pos_col  = f"{prefix}_position"

    # First-call rows give us identifier columns + per-{prefix} metadata.
    # kind is optional - coarser scales (event, episode, ...) don't get a kind
    # column from detect_bouts, so we only include it if present.
    first_cols = list(group_cols) + [size_col]
    if kind_col in df.columns:
        first_cols.append(kind_col)
    first_cols.extend(extra_first_cols)
    first_call = df[df[pos_col] == 1].set_index(id_col)[first_cols]

    # Span: start of first call, stop of last call, duration in seconds.
    span = df.groupby(id_col).agg(
        start_time=("start_time_real", "min"),
        stop_time =("stop_time_real",  "max"),
    )
    span["duration_s"] = (span["stop_time"] - span["start_time"]).dt.total_seconds()

    # Within-{prefix} ICG / ICI stats.
    # For each bout, position-1 has icg_s/ici_s but they span the BETWEEN-bout gap
    # (or are NaN if it's also the first call in its group_cols group). Excluding
    # position-1 keeps only within-{prefix} timing.
    within = df[df[pos_col] > 1]
    within_stats = within.groupby(id_col).agg(
        mean_icg_s=("icg_s", "mean"),
        std_icg_s =("icg_s", "std"),
        mean_ici_s=("ici_s", "mean"),
        std_ici_s =("ici_s", "std"),
    )

    # Combine.
    meta = first_call.join(span).join(within_stats)
    meta["cv_icg"] = meta["std_icg_s"] / meta["mean_icg_s"]
    meta["cv_ici"] = meta["std_ici_s"] / meta["mean_ici_s"]

    # Readable aliases / extras.
    meta["n_calls"] = meta[size_col]
    if finer_scale is not None:
        finer_id_col = f"{finer_scale}_id"
        if finer_id_col not in df.columns:
            raise KeyError(
                f"summarize_scale: finer_scale={finer_scale!r} requested but "
                f"column {finer_id_col!r} is not on df. Did you run "
                f"detect_bouts(prefix={finer_scale!r}, ...) first?"
            )
        n_finer = df.groupby(id_col)[finer_id_col].nunique()
        n_finer.name = f"n_{finer_scale}s"
        meta = meta.join(n_finer)

    # Between-{prefix} timing, scoped to group_cols.
    #   gap_s      = this.start - prev.stop   (silent time between two bouts/events)
    #   interval_s = this.start - prev.start  (period; 1/median = rate of bouts/events)
    gc = list(group_cols)
    meta = meta.sort_values(gc + ["start_time"])
    prev_stop  = meta.groupby(gc)["stop_time"].shift(1)
    prev_start = meta.groupby(gc)["start_time"].shift(1)
    meta["gap_s"]      = (meta["start_time"] - prev_stop ).dt.total_seconds()
    meta["interval_s"] = (meta["start_time"] - prev_start).dt.total_seconds()

    return meta
