"""Core data types for Bunny Deluxe."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

DrummerMode = Literal["listening", "locked", "recovery"]

# General MIDI drum notes
KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
RIDE = 51

DRUM_NOTES = {
    "kick": KICK,
    "snare": SNARE,
    "hat": CLOSED_HAT,
    "open_hat": OPEN_HAT,
    "crash": CRASH,
    "ride": RIDE,
}


@dataclass(frozen=True)
class AccentEvent:
    time_seconds: float
    strength: float


@dataclass
class PulseState:
    bpm: float = 120.0
    confidence: float = 0.0
    beat_phase: float = 0.0
    downbeat_confidence: float = 0.0


@dataclass
class Groove:
    id: str
    name: str
    bars: int
    steps: int
    kick_steps: list[int]
    snare_steps: list[int]
    hat_steps: list[int]
    energy: int
    density: str
    risk: str
    # Simple Brain metadata (optional — safe defaults)
    simple_brain_enabled: bool = False
    ideal_density: str = ""
    min_stability: float = 0.0
    feel_tags: list[str] = field(default_factory=list)
    description: str = ""


@dataclass
class DrummerState:
    mode: DrummerMode = "listening"
    timing_confidence: float = 0.0
    groove_confidence: float = 0.0
    current_groove_id: str | None = None
    complexity_level: int = 5


@dataclass
class DrummerSnapshot:
    """Thread-safe snapshot for GUI / logging."""

    pulse: PulseState = field(default_factory=PulseState)
    drummer: DrummerState = field(default_factory=DrummerState)
    current_step: int = 0
    recent_accents: list[AccentEvent] = field(default_factory=list)
