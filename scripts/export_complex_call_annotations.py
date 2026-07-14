"""Export hand-labeled complex-call annotations to a single CSV.

Walks a TRAINING_data_complex/<subset>/ folder of <stem>.wav +
<stem>_annotations.csv pairs and emits one tidy CSV that downstream analysis
(e.g. gerbil_vocalization_analysis) can load directly.

Output schema:
  file_stem, wav_path, label, onset_s, offset_s, duration_s, channel

Edit the constants below and run:  python scripts/export_complex_call_annotations.py
"""
from pathlib import Path

import pandas as pd

# === Edit these before running ============================================
DATA_ROOT = Path(
    "/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/"
    "Pre_processing/das/models/TRAINING_data_complex"
)
SUBSET = "latest"          # e.g. "latest", "with_dwn_stack", "235"

OUTPUT_DIR = Path(
    "/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/"
    "Processed_data/Audio/combined/complex_calls"
)
# ==========================================================================


def load_annotations(csv_path: Path) -> pd.DataFrame:
    # DAS prepends a class-registration row per label with NaN start_seconds,
    # stop_seconds=0, channel=-1. Drop those — they are not real annotations.
    df = pd.read_csv(csv_path).dropna(subset=["start_seconds"]).reset_index(drop=True)
    return df.rename(
        columns={"name": "label", "start_seconds": "onset_s", "stop_seconds": "offset_s"}
    )


def main() -> None:
    data_dir = DATA_ROOT / SUBSET
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Subset dir not found: {data_dir}")

    rows: list[pd.DataFrame] = []
    missing: list[str] = []
    for wav in sorted(data_dir.glob("*.wav")):
        csv = wav.with_name(wav.stem + "_annotations.csv")
        if not csv.exists():
            missing.append(wav.name)
            continue
        df = load_annotations(csv)
        df["file_stem"] = wav.stem
        df["wav_path"] = str(wav)
        rows.append(df)

    if not rows:
        raise FileNotFoundError(
            f"No <name>.wav + <name>_annotations.csv pairs found in {data_dir}"
        )

    out = pd.concat(rows, ignore_index=True)
    out["duration_s"] = out["offset_s"] - out["onset_s"]
    out = (
        out[["file_stem", "wav_path", "label", "onset_s", "offset_s", "duration_s", "channel"]]
        .sort_values(["file_stem", "onset_s"])
        .reset_index(drop=True)
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"complex_calls_{SUBSET}.csv"
    out.to_csv(output_path, index=False)

    print(f"Subset:      {data_dir}")
    print(f"Wrote:       {output_path}")
    print(f"Annotations: {len(out):,} rows / {out['file_stem'].nunique()} files / {out['label'].nunique()} labels")
    print("Per-label counts:")
    print(out["label"].value_counts().to_string())
    if missing:
        print(f"\n[warn] {len(missing)} wav files had no annotations CSV (skipped):")
        for name in missing:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
