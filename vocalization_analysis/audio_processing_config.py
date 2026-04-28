from pathlib import Path


SKIPPED_EXPERIMENTS: set[int] = {236, 502, 507, 524, 528, 536, 537, 538, 539, 551}


def get_experiment_month(exp: int) -> str:
    if 97 <= exp <= 116:
        return "2024_12"
    if 235 <= exp <= 237:
        return "2025_03"
    if 272 <= exp <= 278:
        return "2025_07"
    if 332 <= exp <= 344:
        return "2025_10"
    if 492 <= exp <= 567 :
        return "2026_02"
    raise ValueError(f"Unknown experiment range for {exp}")


def get_channel_mapping(exp: int) -> dict[str, list[int]]:
    if exp < 272:
        return {
            "10": [0, 1],
            "20": [2, 3],
            "30": [4, 5],
        }
    return {
        "10": [2, 3],
        "20": [4, 5],
        "30": [0, 1],
    }


def get_experiments_to_skip() -> set[int]:
    return set(SKIPPED_EXPERIMENTS)


def should_skip_experiment(exp: int) -> bool:
    return exp in SKIPPED_EXPERIMENTS


def _chunk_ids_for_scheme(raw_audio_folder: Path, channels: list[int], scheme: str) -> set[int]:
    chunk_ids: set[int] = set()

    for channel in channels:
        pattern = f"channel_{channel:02d}_file_*.wav" if scheme == "modern" else f"channel_{channel}_*.wav"
        for path in raw_audio_folder.glob(pattern):
            try:
                if scheme == "modern":
                    chunk_ids.add(int(path.stem.split("_file_")[1]))
                else:
                    chunk_ids.add(int(path.stem.split(f"channel_{channel}_")[1]))
            except (IndexError, ValueError):
                continue

    return chunk_ids


def detect_raw_naming_scheme(exp: int, raw_audio_folder: Path) -> str:
    channel_mapping = get_channel_mapping(exp)
    source_channels = sorted({channel for pair in channel_mapping.values() for channel in pair})

    has_modern = any(
        any(raw_audio_folder.glob(f"channel_{channel:02d}_file_*.wav"))
        for channel in source_channels
    )
    has_legacy = any(
        any(raw_audio_folder.glob(f"channel_{channel}_*.wav"))
        for channel in source_channels
    )

    if has_modern and not has_legacy:
        return "modern"
    if has_legacy and not has_modern:
        return "legacy"
    if has_modern and has_legacy:
        modern_chunk_ids = _chunk_ids_for_scheme(raw_audio_folder, source_channels, "modern")
        legacy_chunk_ids = _chunk_ids_for_scheme(raw_audio_folder, source_channels, "legacy")
        if modern_chunk_ids == legacy_chunk_ids:
            return "modern"
        raise ValueError(
            f"Ambiguous raw naming scheme for experiment {exp}: found both legacy and modern source-channel files in {raw_audio_folder}"
        )

    raise FileNotFoundError(
        f"Could not detect raw naming scheme for experiment {exp}: no legacy or modern source-channel files found in {raw_audio_folder}"
    )
