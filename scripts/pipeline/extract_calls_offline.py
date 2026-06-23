"""Extract per-date-folder enriched calls.csv files for offline analysis.

For each date folder under <BASE_PROCESSED>/, loads every experiment's
calls.csv + sync.csv via add_exp_times (so experiment-seconds and wall-clock
columns are baked in), concatenates, and writes one CSV per date folder to
OUTPUT_DIR. The result is self-contained (no separate sync.csv needed).

Output filenames: all_calls_<date_folder>.csv

Edit the values below and run:  python scripts/extract_calls_offline.py
"""
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "scripts" / "analysis") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts" / "analysis"))

from run_transitions import (  # type: ignore
    BASE_PROCESSED,
    add_exp_times,
    list_experiment_ids_for_date,
)

# === Edit these before running ============================================
# Where the per-date-folder CSVs land. Created if missing.
OUTPUT_DIR = Path.home() / "offline_data"

# Which date folders to extract. Empty list = auto-discover every YYYY_MM
# folder under BASE_PROCESSED.
DATE_FOLDERS: list[str] = []
# ==========================================================================


def _is_date_folder(name: str) -> bool:
    return len(name) == 7 and name[4] == "_" and name[:4].isdigit() and name[5:7].isdigit()


def discover_date_folders() -> list[str]:
    return sorted(p.name for p in BASE_PROCESSED.iterdir() if p.is_dir() and _is_date_folder(p.name))


def extract_date_folder(date: str, out_dir: Path) -> tuple[Path | None, int, list[tuple[int, str]]]:
    """Return (output_path or None, n_rows, [(exp, error_msg), ...])."""
    rows = []
    failed: list[tuple[int, str]] = []
    for exp in list_experiment_ids_for_date(date):
        try:
            df = add_exp_times(exp)
            df.insert(0, "exp", exp)
            rows.append(df)
        except (FileNotFoundError, ValueError) as exc:
            failed.append((exp, str(exc)))

    if not rows:
        return None, 0, failed

    out_path = out_dir / f"all_calls_{date}.csv"
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(out_path, index=False)
    return out_path, len(combined), failed


def main() -> int:
    out_dir = OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    date_folders = list(DATE_FOLDERS) if DATE_FOLDERS else discover_date_folders()
    if not date_folders:
        raise SystemExit(f"No date folders found under {BASE_PROCESSED}")

    print(f"Output dir   : {out_dir}")
    print(f"Date folders : {date_folders}\n")

    summary = []
    for date in date_folders:
        out_path, n_rows, failed = extract_date_folder(date, out_dir)
        if out_path is None:
            print(f"{date}: no data")
            summary.append((date, 0, 0, len(failed)))
            continue
        n_exps = len(pd.read_csv(out_path, usecols=["exp"])["exp"].unique())
        print(f"{date}: wrote {n_rows:>8,} rows from {n_exps:>3} exps -> {out_path.name}"
              + (f"  (skipped {len(failed)} exps)" if failed else ""))
        for exp, reason in failed[:5]:
            print(f"   skip {exp}: {reason[:100]}")
        summary.append((date, n_rows, n_exps, len(failed)))

    print("\n=== summary ===")
    for date, n_rows, n_exps, n_failed in summary:
        print(f"  {date}: {n_rows:>8,} rows  {n_exps:>3} exps  ({n_failed} skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
