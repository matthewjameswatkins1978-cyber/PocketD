"""Confidence estimation for the Bunny Deluxe analysis pipeline."""

from __future__ import annotations

import statistics


def estimate_timing_confidence(
    onset_times: list[float], expected_bpm: float = 120.0
) -> float:
    """Estimate how stable a pulse sequence is around an expected tempo.

    This keeps the confidence stage intentionally small: compare the observed
    inter-onset spacing to the expected quarter-note interval and clamp the
    result to a 0.0-1.0 range.
    """
    if len(onset_times) < 2:
        return 0.5

    intervals = [b - a for a, b in zip(onset_times, onset_times[1:]) if b > a]
    if not intervals:
        return 0.5

    interval = statistics.median(intervals)
    if interval <= 0:
        return 0.5

    expected_interval = 60.0 / expected_bpm
    variance = abs(interval - expected_interval) / expected_interval
    confidence = max(0.0, 1.0 - min(variance * 2.0, 1.0))
    return max(0.0, min(1.0, confidence))


def calculate_confidence(
    tempo_stability: float,
    accent_consistency: float,
    fingerprint_repeat: float,
    downbeat_confidence: float,
) -> float:
    """Return a 0.0-1.0 confidence score for the current groove interpretation."""

    raw = (
        tempo_stability * 0.35
        + accent_consistency * 0.25
        + fingerprint_repeat * 0.25
        + downbeat_confidence * 0.15
    )
    return max(0.0, min(1.0, raw))
