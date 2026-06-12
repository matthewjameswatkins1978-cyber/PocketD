"""Drummer package — event generation, scheduling, humanization, and feel."""

from __future__ import annotations

from drummer.feel import (
    DrummerFeelEngine,
    DrummerProfile,
    GrooveEvent,
    TimingStrategy,
)
from drummer.humanize import humanize_events

__all__ = [
    "DrummerFeelEngine",
    "DrummerProfile",
    "GrooveEvent",
    "TimingStrategy",
    "humanize_events",
]
