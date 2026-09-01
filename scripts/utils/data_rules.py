#!/usr/bin/env python
"""Data that is never usable, enforced in one place instead of documented in many.

These rules were written down in README.md and in the loaders' docstrings, and
were still missed -- the nest-motion analysis ran for two days on traverses from
the truncated last chunk of every experiment before anyone noticed. A rule that
lives in prose does not propagate; a default argument in a shared function does.

So: every loader here applies the rules by DEFAULT and takes an explicit
``keep_*`` flag to opt out. Anything that reads the raw files itself bypasses
them, which is why `check_direct_reads.py` exists.

The rules
---------
**The last file of every experiment is cut short.** Recording stops part-way
through it. In 2026_02 that is 53 chunks, 24 of them under a full 6 minutes,
median 159 s. It costs ~1.2% of calls and ~2.3% of traverses. The video side
already dropped it (`tracking_io.load_files_vetted`); the call and traverse sides
did not.

**A capped `t_out` is not a measurement.** `burrow_scan` reports t_out as the
first sustained empty-tunnel moment, but when none turns up within MAX_LINGER_S
it returns `t_exit + 5 s` and sets `still_in_tunnel_at_cap`. Anything anchored on
arrival must drop those or it measures a window up to 5 s off the event.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd


def drop_last_file(df: pd.DataFrame, exp: str = "exp",
                   file_num: str = "file_num", quiet: bool = False) -> pd.DataFrame:
    """Remove rows from each experiment's final, truncated chunk.

    Works on anything carrying an experiment and a chunk number -- calls,
    traverses, detections -- because the audio and video are cut into the same
    numbered chunks, so the same file_num is bad on both sides.
    """
    if df.empty or exp not in df.columns or file_num not in df.columns:
        return df
    last = df.groupby(exp)[file_num].transform("max")
    keep = df[file_num] < last
    if not quiet and (~keep).any():
        print(f"  dropped {int((~keep).sum()):,} rows from the truncated last chunk "
              f"of {df.loc[~keep, exp].nunique()} experiments")
    return df[keep]


def load_traverses(scan: Path, date: str = "2026_02", *,
                   keep_last_file: bool = False,
                   keep_capped: bool = False,
                   single_animal: bool | None = None,
                   direction: str | None = None,
                   quiet: bool = False) -> pd.DataFrame:
    """The burrow-scan traverse table, with the never-usable rows already gone.

    Read traverses through this rather than `pd.read_parquet` so the rules travel
    with the data. `keep_last_file` and `keep_capped` exist for the rare analysis
    that genuinely wants them -- auditing the truncated chunks, say -- and both
    default to dropping.
    """
    tv = pd.read_parquet(Path(scan) / f"traverses_{date}.parquet")
    n0 = len(tv)
    if not keep_last_file:
        tv = drop_last_file(tv, quiet=quiet)
    if not keep_capped and "still_in_tunnel_at_cap" in tv.columns:
        capped = int(tv.still_in_tunnel_at_cap.sum())
        tv = tv[~tv.still_in_tunnel_at_cap]
        if not quiet and capped:
            print(f"  dropped {capped:,} traverses whose t_out was capped, not observed")
    if single_animal is not None and "single_animal" in tv.columns:
        tv = tv[tv.single_animal == single_animal]
    if direction is not None:
        tv = tv[tv.direction == direction]
    if not quiet:
        print(f"  traverses: {len(tv):,} of {n0:,} kept")
    return tv
