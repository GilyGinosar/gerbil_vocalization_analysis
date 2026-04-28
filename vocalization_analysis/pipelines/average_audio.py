from __future__ import annotations

import argparse
import ast
import json
import platform
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile

from vocalization_analysis.audio_processing_config import (
    detect_raw_naming_scheme,
    get_channel_mapping,
    get_experiment_month,
    should_skip_experiment,
)

if platform.system() == "Windows":
    BASE_RAW = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Raw_data")
    BASE_PROCESSED = Path(r"\\sanesstorage.cns.nyu.edu\archive\ginosar\Processed_data")
else:
    BASE_RAW = Path("/mnt/home/neurostatslab/ceph/saneslab_data/big_setup/")
    BASE_PROCESSED = Path("/mnt/home/neurostatslab/ceph/saneslab_data/gily_data/Processed_data")

MAX_STANDARD_WAV_SIZE_BYTES = (4 * 1024**3) - 1


def build_experiment_paths(exp: int) -> dict[str, Path | str]:
    month_folder = get_experiment_month(exp)
    experiment_root = BASE_RAW / f"experiment_{exp}"
    folder_path_raw_wavs = experiment_root / "concatenated_data_cam_mic_sync"
    folder_path_sync = folder_path_raw_wavs
    folder_path_averaged_wavs = BASE_PROCESSED / "Audio" / month_folder / str(exp) / "Averaged_wavs_w_annotations"
    folder_path_audio = BASE_PROCESSED / "Audio" / month_folder / str(exp)
    return {
        "month_folder": month_folder,
        "experiment_root": experiment_root,
        "raw_wavs": folder_path_raw_wavs,
        "sync": folder_path_sync,
        "averaged_wavs": folder_path_averaged_wavs,
        "audio": folder_path_audio,
    }


def load_sync_file(exp: int, folder_path_sync: Path) -> tuple[pd.Timestamp, pd.DataFrame]:
    sync_path = Path(folder_path_sync) / "sync.csv"
    if not sync_path.exists():
        raise FileNotFoundError(f"Sync file not found: {sync_path}")

    sync_df = pd.read_csv(sync_path)
    print(list(sync_df.columns))

    sync_df["timestamp"] = sync_df["timestamp"].apply(ast.literal_eval)
    sync_df["start_time"] = pd.to_datetime(sync_df["timestamp"].str[0])
    start_time_experiment = sync_df.iloc[0]["start_time"]
    print(f"Experiment {exp} started at: {start_time_experiment}")

    return start_time_experiment, sync_df


def collect_file_nums(input_folder: Path, exp: int) -> list[int]:
    raw_naming_scheme = detect_raw_naming_scheme(exp, input_folder)
    source_channels = sorted({channel for pair in get_channel_mapping(exp).values() for channel in pair})
    file_nums: set[int] = set()

    for source_channel in source_channels:
        if raw_naming_scheme == "legacy":
            pattern = re.compile(rf"channel_{source_channel}_(\d+)\.wav$")
        else:
            pattern = re.compile(rf"channel_{source_channel:02d}_file_(\d+)\.wav$")

        for file_path in input_folder.iterdir():
            match = pattern.match(file_path.name)
            if match:
                file_nums.add(int(match.group(1)))

    return sorted(file_nums)


def collect_chunk_paths(input_folder: Path, file_num: int, raw_naming_scheme: str) -> list[Path]:
    if raw_naming_scheme == "legacy":
        suffix = f"_{file_num}"
    else:
        suffix = f"_{file_num:03d}"

    chunk_paths = [
        path for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() in {".wav", ".mp4"} and path.stem.endswith(suffix)
    ]
    return sorted(chunk_paths)


def find_oversized_chunk_files(chunk_paths: list[Path]) -> list[Path]:
    return [path for path in chunk_paths if path.suffix.lower() == ".wav" and path.stat().st_size > MAX_STANDARD_WAV_SIZE_BYTES]


def remove_terminal_problem_chunk(
    input_folder: Path, file_num: int, raw_naming_scheme: str, problem_paths: list[Path]
) -> None:
    chunk_paths = collect_chunk_paths(input_folder, file_num, raw_naming_scheme)
    print(
        f"Removing terminal problematic chunk {file_num:03d}; oversized WAV files: "
        f"{', '.join(path.name for path in problem_paths)}"
    )
    for path in chunk_paths:
        path.unlink()
        print(f"  Deleted {path.name}")


def average_microphone_pairs(exp: int, input_folder: Path, output_folder: Path) -> dict[str, int]:
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    channel_mapping = get_channel_mapping(exp)
    raw_naming_scheme = detect_raw_naming_scheme(exp, input_folder)

    print(f"\n--- Processing experiment {exp} ---")
    print(f"Reading raw WAV files from: {input_folder}")
    print(f"Saving averaged WAV files to: {output_folder}")
    print(f"Raw naming scheme: {raw_naming_scheme}")

    for virtual_ch, real_pair in channel_mapping.items():
        print(f"  virtual channel {virtual_ch} <- real channels {real_pair}")

    file_nums = collect_file_nums(input_folder, exp)
    print(f"Found {len(file_nums)} file chunks")

    n_saved = 0
    n_missing = 0
    n_shape_mismatch = 0
    n_rate_mismatch = 0
    dropped_terminal_chunks: list[int] = []

    for index, file_num in enumerate(file_nums):
        file_num_str = f"{file_num:03d}"
        chunk_paths = collect_chunk_paths(input_folder, file_num, raw_naming_scheme)
        oversized_chunk_files = find_oversized_chunk_files(chunk_paths)
        if oversized_chunk_files:
            is_last_chunk = index == len(file_nums) - 1
            if not is_last_chunk:
                oversized_names = ", ".join(path.name for path in oversized_chunk_files)
                raise RuntimeError(
                    f"Problematic non-terminal chunk {file_num_str} in experiment {exp}; "
                    f"oversized WAV files: {oversized_names}"
                )

            remove_terminal_problem_chunk(input_folder, file_num, raw_naming_scheme, oversized_chunk_files)
            dropped_terminal_chunks.append(file_num)
            continue

        for virtual_ch, real_pair in channel_mapping.items():
            ch1, ch2 = real_pair
            if raw_naming_scheme == "legacy":
                path1 = input_folder / f"channel_{ch1}_{file_num}.wav"
                path2 = input_folder / f"channel_{ch2}_{file_num}.wav"
            else:
                path1 = input_folder / f"channel_{ch1:02d}_file_{file_num_str}.wav"
                path2 = input_folder / f"channel_{ch2:02d}_file_{file_num_str}.wav"

            if not path1.exists() or not path2.exists():
                print("  Skipping because one or both files are missing")
                print(f"  Expected file: {path1.name}")
                print(f"  Expected file: {path2.name}")
                n_missing += 1
                continue

            rate1, data1 = wavfile.read(path1)
            rate2, data2 = wavfile.read(path2)

            if rate1 != rate2:
                print(f"  Skipping because sample rates differ: {rate1} vs {rate2}")
                n_rate_mismatch += 1
                continue

            if data1.shape != data2.shape:
                print(f"  Skipping because shapes differ: {data1.shape} vs {data2.shape}")
                n_shape_mismatch += 1
                continue

            avg_data = (data1.astype(np.float32, copy=False) + data2.astype(np.float32, copy=False)) / 2.0
            out_path = output_folder / f"channel_{virtual_ch}_file_{file_num_str}.wav"
            wavfile.write(out_path, rate1, avg_data)
            n_saved += 1

    print("\n--- Done ---")
    print(f"Saved files: {n_saved}")
    print(f"Skipped because of missing files: {n_missing}")
    print(f"Skipped because of shape mismatch: {n_shape_mismatch}")
    print(f"Skipped because of sample rate mismatch: {n_rate_mismatch}")
    print(f"Dropped terminal problematic chunks: {dropped_terminal_chunks}")

    return {
        "saved_files": n_saved,
        "missing_pairs": n_missing,
        "shape_mismatch": n_shape_mismatch,
        "rate_mismatch": n_rate_mismatch,
        "dropped_terminal_chunks": len(dropped_terminal_chunks),
    }


def copy_experiment_log_file(exp: int, experiment_root: Path, destination_folder: Path) -> Path | None:
    experiment_root = Path(experiment_root)
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    log_files = sorted(experiment_root.glob(f"experiment_{exp}_log*.txt"))
    if not log_files:
        print(f"Warning: no log txt file found in {experiment_root}")
        return None

    log_path = log_files[0]
    destination_path = destination_folder / log_path.name
    shutil.copy2(log_path, destination_path)
    print(f"Copied log file to: {destination_path}")
    return destination_path


def process_experiment(exp: int) -> dict[str, object]:
    print(f"\n{'=' * 80}")
    print(f"Starting experiment {exp}")

    if should_skip_experiment(exp):
        print(f"Skipping experiment {exp} by design")
        return {"exp": exp, "skipped": True, "reason": "configured skip"}

    paths = build_experiment_paths(exp)
    print("Raw WAV folder:", paths["raw_wavs"])
    print("Sync folder:", paths["sync"])
    print("Output folder:", paths["averaged_wavs"])
    print("Processed experiment folder:", paths["audio"])

    start_time_experiment, sync_df = load_sync_file(exp, paths["sync"])
    log_copy_path = copy_experiment_log_file(exp, paths["experiment_root"], paths["audio"])
    summary = average_microphone_pairs(exp, paths["raw_wavs"], paths["averaged_wavs"])

    return {
        "exp": exp,
        "month_folder": paths["month_folder"],
        "start_time_experiment": str(start_time_experiment),
        "sync_rows": len(sync_df),
        "copied_log_file": None if log_copy_path is None else str(log_copy_path),
        **summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Average raw microphone pairs into Audio outputs.")
    parser.add_argument("--experiment-id", type=int, help="Single experiment id to process.")
    parser.add_argument("--start-exp", type=int, help="Start of inclusive experiment range.")
    parser.add_argument("--end-exp", type=int, help="End of inclusive experiment range.")
    parser.add_argument("--stop-on-error", action="store_true", help="Raise immediately on first failure.")
    parser.add_argument("--json", action="store_true", help="Print results as JSON as well as the tabular summary.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.experiment_id is not None:
        experiment_ids = [args.experiment_id]
    elif args.start_exp is not None and args.end_exp is not None:
        experiment_ids = list(range(args.start_exp, args.end_exp + 1))
    else:
        raise SystemExit("Provide either --experiment-id or both --start-exp and --end-exp.")

    results = []
    for exp in experiment_ids:
        try:
            results.append(process_experiment(exp))
        except Exception as exc:
            print(f"Failed for experiment {exp}: {exc}")
            results.append({"exp": exp, "error": str(exc)})
            if args.stop_on_error:
                raise

    results_df = pd.DataFrame(results)
    print("\nSummary:")
    print(results_df.to_string(index=False))

    if args.json:
        print("\nJSON:")
        print(json.dumps(results, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
