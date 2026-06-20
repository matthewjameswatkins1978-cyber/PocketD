"""Simple tempo estimation from onset timings."""

from __future__ import annotations

import statistics


def estimate_tempo(onset_times: list[float]) -> float:
    """Estimate BPM from a list of onset timestamps.

    The detector uses median timing across one-, two-, and four-beat spans where
    possible. Longer spans make the estimate more tolerant of human push/pull
    and alternating long-short timing while staying deterministic.

    Very short per-beat intervals are treated as eighth-note spacing, which
    keeps the prototype stable for dense live input.
    """
    if len(onset_times) < 2:
        return 120.0

    span_estimates: list[float] = []
    for span in (1, 2, 4):
        if len(onset_times) <= span:
            continue

        candidates: list[float] = []
        intervals: list[float] = []
        for start_index in range(len(onset_times) - span):
            elapsed = onset_times[start_index + span] - onset_times[start_index]
            if elapsed <= 0:
                continue

            interval_per_beat = elapsed / span
            intervals.append(interval_per_beat)
            candidates.append(60.0 / interval_per_beat)

        if candidates:
            span_bpm = statistics.median(candidates)
            if intervals and statistics.median(intervals) < 0.35:
                span_bpm *= 0.5
            span_estimates.append(span_bpm)

    if not span_estimates:
        return 120.0

    bpm = statistics.median(span_estimates)
    return max(40.0, min(240.0, bpm))
