"""Light cycle (day/night) per experiment / date folder — single source of truth.

A light cycle is ``(lights_on_hour, lights_off_hour)`` on a 24-h clock, half-open
``[on, off)``: daytime is ``on <= hour < off``. Most cohorts ran 08:00-20:00; the
2026_02 experiment logs note "dark" / "lights off" around 16:00, i.e. a 12-h
photoperiod 04:00-16:00. Add new cohorts to ``LIGHT_CYCLE_BY_MONTH`` as recorded.

Use from other scripts after putting ``scripts/utils`` on ``sys.path``::

    from light_cycle import get_light_cycle_for_month, get_light_cycle
    on, off = get_light_cycle_for_month("2026_02")   # -> (4, 16)
"""
from __future__ import annotations

DEFAULT_LIGHT_CYCLE: tuple[int, int] = (8, 20)

LIGHT_CYCLE_BY_MONTH: dict[str, tuple[int, int]] = {
    "2024_12": (8, 20),
    "2025_03": (8, 20),
    "2025_07": (8, 20),
    "2025_10": (8, 20),
    "2026_02": (4, 16),
}


def get_light_cycle_for_month(month: str) -> tuple[int, int]:
    """Return (lights_on_hour, lights_off_hour) for a month / date folder."""
    return LIGHT_CYCLE_BY_MONTH.get(month, DEFAULT_LIGHT_CYCLE)


def get_light_cycle(exp: int) -> tuple[int, int]:
    """Return (lights_on_hour, lights_off_hour) for an experiment id.

    Maps exp -> month via the experiment-month table (still in the library for
    now; this stays correct as that logic migrates into scripts/).
    """
    from scripts.pipeline.audio_processing_config import get_experiment_month

    return get_light_cycle_for_month(get_experiment_month(exp))
