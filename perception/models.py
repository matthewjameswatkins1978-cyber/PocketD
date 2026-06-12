"""Data types for the perception engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

FrequencyRegion = Literal["sub", "low", "low_mid", "high_mid", "high", "unknown"]


@dataclass(frozen=True)
class MusicalEvent:
    """A single detected musical event extracted from audio.

    This is the fundamental output unit of the perception engine.
    It describes *what happened* musically, not which note was played.

    Parameters
    ----------
    time_seconds : float
        The time at which the event was detected.
    strength : float
        Normalised attack strength in [0, 1]. How hard the attack hit.
    frequency_region : FrequencyRegion
        The rough frequency band where the energy concentrates.
        Helps distinguish kick (sub/low) from snare (low_mid) from hats (high).
    energy : float
        Local RMS energy in the vicinity of the attack, normalised [0, 1].
    density : float
        Local attack density — how many nearby attacks occurred in a short window.
    """
    time_seconds: float
    strength: float = 0.0
    frequency_region: FrequencyRegion = "unknown"
    energy: float = 0.0
    density: float = 0.0


# Frequency band definitions (Hz)
FREQUENCY_REGIONS: list[tuple[str, float, float]] = [
    # name,            low_hz,  high_hz
    ("sub",             20.0,    80.0),
    ("low",             80.0,   250.0),
    ("low_mid",        250.0,   800.0),
    ("high_mid",       800.0,  2500.0),
    ("high",          2500.0, 20000.0),
]

# Default FFT size for frequency analysis
FFT_SIZE: int = 2048
# Hop size for sliding window analysis
HOP_SIZE: int = 512