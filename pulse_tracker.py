"""Simple tempo estimation from onset timings."""

from __future__ import annotations

import statistics


def estimate_tempo(onset_times: list[float]) -> float:
    """Estimate BPM from a list of onset timestamps.

    The detector uses the median inter-onset interval and treats short intervals
    as eighth-note spacing, which keeps the prototype stable for live playing.
    """
    if len(onset_times) < 2:
        return 120.0

    intervals = [b - a for a, b in zip(onset_times, onset_times[1:]) if b > a]
    if not intervals:
        return 120.0

    interval = statistics.median(intervals)
    if interval <= 0:
        return 120.0

    # Short intervals usually represent eighth-note pulses in the live input.
    bpm = 30.0 / interval if interval < 0.35 else 60.0 / interval
    return max(40.0, min(240.0, bpm))
