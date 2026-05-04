"""Cache the consolidated all_calls_<date>.csv files to parquet for fast reload.

Run once now, and once more whenever a new all_calls_<date>.csv lands in
Dropbox. Each input CSV becomes one .parquet alongside it under parquet_cache/.

Why parquet:
  - ~10x faster to load than CSV (binary, columnar, no parsing).
  - Preserves dtypes — datetime stays datetime, ints stay ints. CSV reload
    would parse `start_time_real` back as a string every time.
  - About 5x smaller on disk.

Usage:
    uv run python scripts/cache_calls_to_parquet.py
"""
from pathlib import Path

import pandas as pd

INPUT_DIR = Path("/Users/gilyginosar/Dropbox (Personal)/Vocalizations_project/Data")
PARQUET_DIR = INPUT_DIR / "parquet_cache"

DATETIME_COLS = ("start_time_real", "stop_time_real")


def cache_one(csv_path: Path) -> Path:
    date_tag = csv_path.stem.replace("all_calls_", "")  # e.g. "2025_10"
    parquet_path = PARQUET_DIR / f"all_calls_{date_tag}.parquet"

    print(f"  reading {csv_path.name} ...")
    df = pd.read_csv(csv_path)

    for col in DATETIME_COLS:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Stamp every row with its source so you can recover provenance after
    # concatenating multiple months together.
    df["date_folder"] = date_tag

    # Sequential day-of-recording within this date_folder: 1 for the earliest
    # calendar day, 2 for the next, ... Independent of the upstream `exp` column.
    df["day_in_experiment"] = (
        df["start_time_real"].dt.normalize().rank(method="dense").astype(int)
    )

    df.to_parquet(parquet_path, index=False)
    print(f"    -> {parquet_path.name}  ({len(df):,} rows)")
    return parquet_path


def main() -> int:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    csvs = sorted(INPUT_DIR.glob("all_calls_*.csv"))
    if not csvs:
        raise SystemExit(f"No all_calls_*.csv files found in {INPUT_DIR}")
    print(f"Caching {len(csvs)} CSV(s) -> {PARQUET_DIR}")
    for csv_path in csvs:
        cache_one(csv_path)
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())