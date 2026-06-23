"""Export a tidy per-file sync table into each experiment's processed folder.

The raw big_setup ``sync.csv`` has one row per audio chunk file but stores the
full video/audio file-name lists (redundant) and packs both timestamps into a
single stringified ``[start, end]`` cell. This script writes a slim copy with
just:

    exp_num, file_num, start_date, start_time, end_date, end_time

Output goes to the experiment's processed-data folder as ``file_times.csv``.

The tidy table is built by ``vocalization_analysis.sync_times.build_file_times``
— the same helper the ``average_audio`` pipeline calls, so new experiments get
their ``file_times.csv`` automatically and this script only needs to be run to
(re)generate older ones.

Usage:
    python scripts/utils/export_sync_tidy.py 492            # one experiment
    python scripts/utils/export_sync_tidy.py 492 493 494    # several
    python scripts/utils/export_sync_tidy.py --all          # every exp under Processed_data
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from vocalization_analysis.audio_processing_config import get_experiment_month
from vocalization_analysis.sync_times import build_file_times

BASE_PROCESSED = Path(
    "/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data/Audio"
)


def processed_dir(exp: int) -> Path:
    return BASE_PROCESSED / get_experiment_month(exp) / str(exp)


def export(exp: int, dry_run: bool = False) -> Path | None:
    dest = processed_dir(exp) / "file_times.csv"
    try:
        out = build_file_times(exp)
    except FileNotFoundError:
        print(f"exp {exp}: SKIP (no raw sync.csv)")
        return None
    except Exception as e:  # noqa: BLE001
        print(f"exp {exp}: SKIP ({type(e).__name__}: {e})")
        return None
    if dry_run:
        print(f"[dry-run] exp {exp}: {len(out)} rows -> {dest}")
    else:
        dest.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(dest, index=False)
        print(f"exp {exp}: wrote {len(out)} rows -> {dest}")
    return dest


def discover_experiments() -> list[int]:
    """All numeric experiment folders under every month dir in Processed_data."""
    exps: list[int] = []
    for month_dir in sorted(BASE_PROCESSED.iterdir()):
        if not month_dir.is_dir():
            continue
        for sub in month_dir.iterdir():
            if sub.is_dir() and sub.name.isdigit():
                exps.append(int(sub.name))
    return sorted(exps)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("experiments", type=int, nargs="*", help="experiment number(s)")
    ap.add_argument("--all", action="store_true",
                    help="process every experiment folder under Processed_data")
    ap.add_argument("--dry-run", action="store_true", help="print, do not write")
    args = ap.parse_args()

    exps = discover_experiments() if args.all else args.experiments
    if not exps:
        ap.error("give experiment number(s) or --all")

    written = 0
    for exp in exps:
        if export(exp, dry_run=args.dry_run) is not None:
            written += 1
    print(f"\n{written}/{len(exps)} experiments processed.")


if __name__ == "__main__":
    main()
