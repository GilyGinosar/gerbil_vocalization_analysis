"""Single source of truth for which experiment belongs to which date folder.

The mapping itself lives in ``experiments.toml`` beside this file, so starting a
new date folder is a data edit rather than a code change. Everything here just
reads that file and answers questions about it.

Still code, because they are rules rather than lists: the mic-pair wiring
(:func:`get_channel_mapping`) and the raw-filename scheme detection
(:func:`detect_raw_naming_scheme`).
"""
from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

CONFIG_PATH = Path(__file__).with_name("experiments.toml")


def _parse_id_spec(spec: str, date_folder: str) -> list[int]:
    """Expand "97-116, 785" into [97, ..., 116, 785]."""
    ids: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if "-" in part:
                start_text, _, end_text = part.partition("-")
                start, end = int(start_text), int(end_text)
                if end < start:
                    raise ValueError(f"range runs backwards: {part!r}")
                ids.extend(range(start, end + 1))
            else:
                ids.append(int(part))
        except ValueError as exc:
            raise ValueError(
                f"Bad `experiments` entry {part!r} for [{date_folder}] in {CONFIG_PATH}: {exc}"
            ) from None
    if not ids:
        raise ValueError(f"[{date_folder}] in {CONFIG_PATH} lists no experiments.")
    return ids


@lru_cache(maxsize=1)
def _load() -> dict[str, dict]:
    """Parse experiments.toml into {date_folder: {"ids": [...], "skip": frozenset}}."""
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Experiment config not found: {CONFIG_PATH}")

    with CONFIG_PATH.open("rb") as handle:
        raw = tomllib.load(handle)

    folders: dict[str, dict] = {}
    owner: dict[int, str] = {}  # experiment id -> date folder, to catch overlaps

    for date_folder, entry in raw.get("date_folders", {}).items():
        ids = _parse_id_spec(entry["experiments"], date_folder)

        for exp in ids:
            if exp in owner:
                raise ValueError(
                    f"Experiment {exp} is claimed by both [{owner[exp]}] and "
                    f"[{date_folder}] in {CONFIG_PATH}."
                )
            owner[exp] = date_folder

        skip = frozenset(entry.get("skip", []))
        stray = sorted(skip - set(ids))
        if stray:
            raise ValueError(
                f"[{date_folder}] in {CONFIG_PATH} skips {stray}, which are not in "
                f"its own `experiments` range."
            )

        folders[date_folder] = {"ids": ids, "skip": skip, "note": entry.get("note", ""),
                                "n_animals": entry.get("n_animals")}

    if not folders:
        raise ValueError(f"No [date_folders.*] entries found in {CONFIG_PATH}.")
    return folders


def get_colony_size(date_folder: str) -> int | None:
    """Total gerbils in the setup for a date folder, or None if not recorded.

    None means "not known", never zero. It is the denominator of a census: the
    nest cannot be counted from video because the animals burrow under bedding,
    so occupancy there is inferred as this minus the animals detected in the
    arenas and the tunnel. Without it that inference cannot be made at all.
    """
    folders = _load()
    if date_folder not in folders:
        raise ValueError(
            f"Unknown date folder {date_folder!r}. Configured: {', '.join(sorted(folders))}."
        )
    return folders[date_folder]["n_animals"]


def list_date_folders() -> list[str]:
    """Every configured date folder, oldest first."""
    return sorted(_load())


def get_experiments_for_date(date_folder: str, include_skipped: bool = False) -> list[int]:
    """Experiment ids belonging to one date folder, skips removed by default."""
    folders = _load()
    if date_folder not in folders:
        raise ValueError(
            f"Unknown date folder {date_folder!r}. Configured: {', '.join(sorted(folders))}. "
            f"Add it to {CONFIG_PATH}."
        )
    entry = folders[date_folder]
    if include_skipped:
        return list(entry["ids"])
    return [exp for exp in entry["ids"] if exp not in entry["skip"]]


def get_experiment_month(exp: int) -> str:
    for date_folder, entry in _load().items():
        if exp in entry["ids"]:
            return date_folder
    raise ValueError(
        f"Unknown experiment range for {exp}. Add it to a [date_folders.*] block in {CONFIG_PATH}."
    )


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
    return {exp for entry in _load().values() for exp in entry["skip"]}


def should_skip_experiment(exp: int) -> bool:
    return exp in get_experiments_to_skip()


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


# Back-compat: the old module exposed this as a module-level constant.
SKIPPED_EXPERIMENTS: set[int] = get_experiments_to_skip()
