#!/usr/bin/env python
"""Where a finished figure goes, so it is in both places and easy to find.

Every figure script writes into a working directory -- caches, selection tables,
sheet folders -- and that is the right place for intermediates. It is the wrong
place for the figure itself: `exports/` accumulated a dozen near-identical
`*/burrow_overview.png`, all superseded, and telling them apart meant reading
timestamps.

So the finished image is copied to two destinations, which answer different
questions:

  1. exports/<today>/<name>      what did this session produce. Grouped by the
                                 day it was RUN, so clearing out old work is
                                 deleting a folder rather than picking through
                                 files and reading timestamps to tell a dozen
                                 near-identical figures apart.
  2. Combined/<date>/<name>      the record, on ceph beside the Audio and Video
                                 trees where a figure drawn from both belongs.
                                 Nothing is overwritten: a rerun adds a version
                                 rather than replacing one, because the figure
                                 someone put in a talk should still exist after
                                 the analysis behind it changes.

Every filename carries BOTH dates -- ``burrow_overview_2026_02_2026-09-01.png``
is the 2026_02 date folder, drawn on 1 Sept. In exports the run date repeats
the folder it sits in, deliberately: the failure this replaces was a dozen files
called `burrow_overview.png` whose identity lived entirely in their directory, so
one dragged anywhere became unidentifiable. A name that survives being moved is
worth one redundant token.

Sorting a directory groups every version of a figure together, latest last.
"""
from __future__ import annotations

import shutil
from datetime import date as _date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPORTS = REPO_ROOT / "exports"
COMBINED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/"
                "Processed_data/Combined")


def today() -> str:
    return _date.today().isoformat()


def destinations(date: str = "2026_02") -> list[Path]:
    """Where a figure goes: this session's folder, and the current-figure tree."""
    return [EXPORTS / today(), COMBINED / date]


def publish(src: Path | str, name: str | None = None, date: str = "2026_02",
            quiet: bool = False) -> list[Path]:
    """Copy one finished figure to exports/ and to Combined/<date>/.

    Lands in ``exports/<today>/`` and in ``Combined/<date>/``, under the same
    name in both. `name` defaults to the source filename with the experiment
    date and the run date appended -- `burrow_overview.png` becomes
    `burrow_overview_2026_02_2026-09-01.png` -- so nothing is overwritten and
    the file identifies itself wherever it ends up.
    """
    src = Path(src)
    if not src.exists():
        if not quiet:
            print(f"  publish: {src} does not exist, skipped")
        return []
    if name is None:
        name = f"{src.stem}_{date}_{today()}{src.suffix}"
    out = []
    for d in destinations(date):
        try:
            d.mkdir(parents=True, exist_ok=True)
            dst = d / name
            shutil.copy2(src, dst)
            out.append(dst)
            if not quiet:
                print(f"  published {dst}")
        except OSError as e:
            # ceph being unavailable must not lose the figure that is already
            # written locally, so this reports and carries on
            print(f"  publish FAILED to {d}: {e}")
    return out


def publish_many(paths, prefix: str, date: str = "2026_02",
                 quiet: bool = False) -> list[Path]:
    """Publish a numbered set (card sheets) under one flat prefix."""
    paths = sorted(Path(p) for p in paths)
    out = []
    for i, p in enumerate(paths, 1):
        suffix = f"_{i:02d}" if len(paths) > 1 else ""
        out += publish(p, name=f"{prefix}_{date}_{today()}{suffix}{p.suffix}",
                       date=date, quiet=quiet)
    return out
