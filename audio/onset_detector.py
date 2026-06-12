"""Onset detection helpers for the fake-pulse diagnostic path."""

from __future__ import annotations

from typing import Protocol

from models import AccentEvent


class SupportsOnset(Protocol):
    time_seconds: float
    strength: float


def detect_onsets_from_events(
    events: list[SupportsOnset],
    sample_rate: int = 16000,
    min_interval: float = 0.05,
) -> list[AccentEvent]:
    """Convert synthetic pulse events into onset detections.

    This is intentionally small and deterministic so the fake-pulse stage can
    be verified before real microphone/audio capture is added.
    """
    detected: list[AccentEvent] = []
    previous_time: float | None = None

    for event in events:
        timestamp = float(getattr(event, "time_seconds", 0.0))
        strength = float(getattr(event, "strength", 1.0))

        if previous_time is None or (timestamp - previous_time) >= min_interval:
            detected.append(AccentEvent(time_seconds=timestamp, strength=strength))
            previous_time = timestamp

    return detected
