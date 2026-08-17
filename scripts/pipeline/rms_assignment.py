from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.io import wavfile


DEFAULT_SOURCE_CHANNELS = {
    "arena_1": "10",
    "arena_2": "20",
    "underground": "30",
}


@dataclass(slots=True)
class RMSAssignmentConfig:
    accepted_calls_dir: Path
    averaged_wavs_dir: Path
    output_dir: Path
    experiment_dir: Path | None = None
    overlap_window_s: float = 0.015
    source_channels: dict[str, str] | None = None

    def resolved_source_channels(self) -> dict[str, str]:
        return dict(DEFAULT_SOURCE_CHANNELS if self.source_channels is None else self.source_channels)


def compute_rms(signal: np.ndarray, eps: float = 1e-12) -> float:
    if signal.size == 0:
        return np.nan
    signal = signal.astype(np.float32, copy=False)
    return float(20 * np.log10(np.sqrt(np.mean(signal ** 2)) + eps))


def load_per_file_calls(accepted_calls_dir: Path) -> pd.DataFrame:
    csv_paths = sorted(path for path in accepted_calls_dir.glob("*_accepted_calls.csv") if path.is_file())
    rows: list[pd.DataFrame] = []

    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        stem = csv_path.stem
        parts = stem.split("_")
        channel = parts[1] if len(parts) > 1 else ""
        try:
            file_number = parts[3]
        except IndexError:
            file_number = ""

        df.insert(0, "file_number", file_number)
        df.insert(0, "channel", channel)
        rows.append(df)

    if not rows:
        return pd.DataFrame(columns=["channel", "file_number"])

    return pd.concat(rows, ignore_index=True)


def sort_calls_by_time(calls_df: pd.DataFrame) -> pd.DataFrame:
    if calls_df.empty:
        return calls_df

    sort_df = calls_df.copy()
    file_col = "file_number" if "file_number" in sort_df.columns else "file_num"
    sort_df["_file_number_sort"] = pd.to_numeric(sort_df[file_col], errors="coerce")
    sort_cols = ["_file_number_sort"]
    if "onset_s" in sort_df.columns:
        sort_cols.append("onset_s")
    elif "start_time_file_sec" in sort_df.columns:
        sort_cols.append("start_time_file_sec")

    sort_df = sort_df.sort_values(sort_cols, kind="stable", na_position="last").reset_index(drop=True)
    return sort_df.drop(columns=["_file_number_sort"], errors="ignore")


def _load_wav_cache_for_file(file_number: str, averaged_wavs_dir: Path, source_channels: dict[str, str]) -> dict[str, tuple[int, np.ndarray]]:
    cache: dict[str, tuple[int, np.ndarray]] = {}
    for source_name, channel in source_channels.items():
        wav_path = averaged_wavs_dir / f"channel_{channel}_file_{int(file_number):03d}.wav"
        rate, data = wavfile.read(wav_path)
        if getattr(data, "ndim", 1) > 1:
            data = data[:, 0]
        cache[source_name] = (rate, data)
    return cache


def attach_rms_columns(calls_df: pd.DataFrame, averaged_wavs_dir: Path, source_channels: dict[str, str]) -> pd.DataFrame:
    calls_df = calls_df.copy()
    for source_name in source_channels:
        calls_df[f"RMS_{source_name}"] = np.nan

    if calls_df.empty:
        return calls_df

    required_cols = {"file_number", "onset_s", "offset_s"}
    missing = required_cols.difference(calls_df.columns)
    if missing:
        raise ValueError(f"Calls dataframe is missing required columns for RMS assignment: {sorted(missing)}")

    for file_number, group_idx in calls_df.groupby("file_number").groups.items():
        wav_cache = _load_wav_cache_for_file(str(file_number), averaged_wavs_dir, source_channels)
        for row_i in group_idx:
            onset_s = float(calls_df.at[row_i, "onset_s"])
            offset_s = float(calls_df.at[row_i, "offset_s"])
            for source_name, (rate, data) in wav_cache.items():
                start = max(0, int(onset_s * rate))
                stop = max(start, int(offset_s * rate))
                calls_df.at[row_i, f"RMS_{source_name}"] = compute_rms(data[start:stop])

    return calls_df


def select_highest_rms_calls(calls_df: pd.DataFrame, overlap_window_s: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if calls_df.empty:
        return calls_df.copy(), pd.DataFrame()

    calls_df = calls_df.sort_values(["file_number", "onset_s"]).reset_index(drop=True)
    rms_cols = [col for col in calls_df.columns if col.startswith("RMS_")]

    merged: list[pd.Series] = []
    merged_log_rows: list[pd.DataFrame] = []
    used_indices: set[int] = set()

    for i, row in calls_df.iterrows():
        if i in used_indices:
            continue

        group = [row]
        used_indices.add(i)

        same_file = calls_df[calls_df["file_number"] == row["file_number"]]
        same_file = same_file[~same_file.index.isin(used_indices)]

        for j, other_row in same_file.iterrows():
            if abs(float(other_row["onset_s"]) - float(row["onset_s"])) <= overlap_window_s:
                group.append(other_row)
                used_indices.add(j)

        group_df = pd.DataFrame(group)
        if group_df.empty or group_df[rms_cols].isna().all(axis=1).all():
            continue

        rms_sum = group_df[rms_cols].sum(axis=1)
        best_idx = rms_sum.idxmax()
        best_row = group_df.loc[best_idx].copy()
        merged.append(best_row)

        if len(group_df) > 1:
            log_df = group_df.copy()
            log_df.insert(0, "is_chosen", False)
            log_df.loc[best_idx, "is_chosen"] = True
            merged_log_rows.append(log_df)
            merged_log_rows.append(pd.DataFrame([{}]))

    if merged:
        merged_df = pd.DataFrame(merged).reset_index(drop=True)
    else:
        merged_df = calls_df.head(0).copy()
    merged_log_df = pd.concat(merged_log_rows, ignore_index=True) if merged_log_rows else pd.DataFrame()
    return merged_df, merged_log_df


def assign_source_locations(calls_df: pd.DataFrame, source_channels: dict[str, str]) -> pd.DataFrame:
    calls_df = calls_df.copy()
    source_names = list(source_channels.keys())
    rms_cols = [f"RMS_{source_name}" for source_name in source_names]

    if calls_df.empty:
        calls_df["assigned_location"] = pd.Series(index=calls_df.index, dtype="object")
        return calls_df

    def get_best_rms_location(row: pd.Series) -> str | float:
        if row[rms_cols].isna().all():
            return np.nan
        rms_vals = {source_name: row[f"RMS_{source_name}"] for source_name in source_names}
        return max(rms_vals, key=rms_vals.get)

    calls_df["assigned_location"] = calls_df.apply(get_best_rms_location, axis=1)
    return calls_df


def build_qmc_metadata_csv(calls_df: pd.DataFrame, source_channels: dict[str, str]) -> pd.DataFrame:
    metadata_df = calls_df.copy()
    channel_map = {location: int(channel) for location, channel in source_channels.items()}
    meanprob_cols = [col for col in metadata_df.columns if col.startswith("meanprob_")]

    metadata_df["file_num"] = pd.to_numeric(metadata_df["file_number"], errors="coerce").astype("Int64")
    metadata_df["event_type"] = metadata_df["label"]
    metadata_df["start_time_file_sec"] = metadata_df["onset_s"]
    metadata_df["stop_time_file_sec"] = metadata_df["offset_s"]
    if "duration_s" in metadata_df.columns:
        metadata_df["duration_sec"] = metadata_df["duration_s"]
    else:
        metadata_df["duration_sec"] = metadata_df["stop_time_file_sec"] - metadata_df["start_time_file_sec"]
    metadata_df["assigned_channel"] = metadata_df["assigned_location"].map(channel_map).astype("Int64")

    output_columns = [
        "file_num",
        "event_type",
        "start_time_file_sec",
        "stop_time_file_sec",
        "duration_sec",
        "assigned_location",
        "assigned_channel",
        *meanprob_cols,
    ]
    metadata_df = metadata_df[output_columns].copy()
    return sort_calls_by_time(metadata_df)


def run_rms_assignment(config: RMSAssignmentConfig) -> dict[str, Path]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    source_channels = config.resolved_source_channels()
    experiment_dir = config.output_dir.parent if config.experiment_dir is None else config.experiment_dir
    experiment_dir.mkdir(parents=True, exist_ok=True)

    combined_df = load_per_file_calls(config.accepted_calls_dir)
    combined_with_rms_df = attach_rms_columns(combined_df, config.averaged_wavs_dir, source_channels)
    selected_df, merge_log_df = select_highest_rms_calls(combined_with_rms_df, config.overlap_window_s)
    assigned_df = assign_source_locations(selected_df, source_channels)
    combined_with_rms_df = sort_calls_by_time(combined_with_rms_df)
    assigned_df = sort_calls_by_time(assigned_df)
    qmc_metadata_df = build_qmc_metadata_csv(assigned_df, source_channels)

    combined_out = config.output_dir / "accepted_calls_combined.csv"
    with_rms_out = config.output_dir / "accepted_calls_with_rms.csv"
    selected_out = config.output_dir / "accepted_calls_rms_selected.csv"
    qmc_metadata_out = experiment_dir / "calls.csv"
    merge_log_out = config.output_dir / "accepted_calls_rms_merge_log.csv"

    combined_df.to_csv(combined_out, index=False)
    combined_with_rms_df.to_csv(with_rms_out, index=False)
    assigned_df.to_csv(selected_out, index=False)
    qmc_metadata_df.to_csv(qmc_metadata_out, index=False)
    if not merge_log_df.empty:
        merge_log_df.to_csv(merge_log_out, index=False)

    outputs = {
        "combined_csv": combined_out,
        "with_rms_csv": with_rms_out,
        "selected_csv": selected_out,
        "qmc_metadata_csv": qmc_metadata_out,
    }
    if not merge_log_df.empty:
        outputs["merge_log_csv"] = merge_log_out
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine per-file accepted calls, compute RMS across arenas, and keep the strongest overlapping call."
    )
    parser.add_argument("--accepted-calls-dir", type=Path, required=True)
    parser.add_argument("--averaged-wavs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overlap-window-s", type=float, default=0.015)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    outputs = run_rms_assignment(
        RMSAssignmentConfig(
            accepted_calls_dir=args.accepted_calls_dir,
            averaged_wavs_dir=args.averaged_wavs_dir,
            output_dir=args.output_dir,
            overlap_window_s=args.overlap_window_s,
        )
    )
    for name, path in outputs.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
