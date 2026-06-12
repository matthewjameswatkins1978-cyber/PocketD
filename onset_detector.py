"""Lightweight onset detection for the Pocket Drummer prototype."""

from __future__ import annotations

import numpy as np

from models import AccentEvent


def detect_onsets(
    signal: list[float] | np.ndarray,
    sample_rate: int = 16000,
    min_interval: float = 0.05,
) -> list[AccentEvent]:
    """Detect significant onset events using simple energy/flux analysis."""
    data = np.asarray(signal, dtype=float)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.size < 4:
        return []

    diff = np.abs(np.diff(data))
    window = max(3, min(25, int(sample_rate * 0.005)))
    smoothed = np.convolve(diff, np.ones(window) / window, mode="same")

    threshold = max(float(np.mean(smoothed) * 0.75 + np.std(smoothed) * 0.35), 1e-4)

    events: list[AccentEvent] = []
    for index in range(1, len(smoothed) - 1):
        is_peak = (
            smoothed[index] > smoothed[index - 1]
            and smoothed[index] >= smoothed[index + 1]
            and smoothed[index] > threshold
        )
        if is_peak:
            timestamp = index / sample_rate
            if not events or timestamp - events[-1].time_seconds >= min_interval:
                strength = float(min(1.0, smoothed[index] / max(threshold, 1e-3)))
                events.append(AccentEvent(time_seconds=timestamp, strength=strength))

    return events
