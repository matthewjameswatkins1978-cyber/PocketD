"""Synthetic pulse helpers that simulate realistic playing or detection problems.

Every function returns a list of ``PulseEvent`` objects compatible with the
existing ``detect_onsets_from_events`` pipeline.

Design principle:
    Detection and decision stay separate. These helpers only create imperfect
    synthetic input. The tempo/confidence analysis and groove decision layers
    consume the output untouched.
"""

from __future__ import annotations

import copy
import random

from audio.pulse_generator import PulseEvent, generate_pulse_events


def make_steady_pulse(bpm: float = 120.0, bars: int = 2) -> list[PulseEvent]:
    """Return a clean steady pulse for *bars* quarter-note bars.

    Equivalent to ``generate_pulse_events`` with a bar-based duration.
    """
    seconds_per_bar = 4 * (60.0 / bpm)
    duration = bars * seconds_per_bar
    return generate_pulse_events(bpm=bpm, duration_seconds=duration)


def add_timing_jitter(
    pulses: list[PulseEvent],
    amount_ms: float = 10.0,
    seed: int | None = None,
) -> list[PulseEvent]:
    """Return a new pulse list with random timing offsets applied.

    Each pulse time is shifted by a random value in
    ``[-amount_ms, +amount_ms]`` milliseconds.  Timing is clamped to stay
    non-negative.
    """
    rng = random.Random(seed)
    amount_seconds = amount_ms / 1000.0
    jittered: list[PulseEvent] = []
    for p in pulses:
        offset = rng.uniform(-amount_seconds, amount_seconds)
        new_time = max(0.0, p.time_seconds + offset)
        jittered.append(
            PulseEvent(time_seconds=new_time, strength=p.strength, label=p.label)
        )
    # Keep events in chronological order after jitter
    jittered.sort(key=lambda e: e.time_seconds)
    return jittered


def drop_pulse(pulses: list[PulseEvent], index: int) -> list[PulseEvent]:
    """Return a new pulse list with the pulse at *index* removed.

    If *index* is out of range the original list is returned unchanged.
    """
    if index < 0 or index >= len(pulses):
        return list(pulses)
    return [p for i, p in enumerate(pulses) if i != index]


def add_extra_pulse(
    pulses: list[PulseEvent],
    timestamp: float,
    strength: float = 0.5,
) -> list[PulseEvent]:
    """Return a new pulse list with an extra pulse inserted at *timestamp*."""
    extra = PulseEvent(time_seconds=timestamp, strength=strength, label="extra")
    extended = list(pulses) + [extra]
    extended.sort(key=lambda e: e.time_seconds)
    return extended


def make_tempo_change_pulse(
    start_bpm: float = 120.0,
    end_bpm: float = 90.0,
    bars_each: int = 1,
) -> list[PulseEvent]:
    """Return pulses for two concatenated steady sections at different tempos.

    The result is a single pulse list: *bars_each* bars at *start_bpm*
    immediately followed by *bars_each* bars at *end_bpm*.
    """
    first = make_steady_pulse(bpm=start_bpm, bars=bars_each)
    last_pulse_time = first[-1].time_seconds if first else 0.0

    # Add a small silence gap (1 quarter note) to represent a clear section break
    gap = 60.0 / start_bpm

    second = make_steady_pulse(bpm=end_bpm, bars=bars_each)
    shifted = [
        PulseEvent(
            time_seconds=p.time_seconds + last_pulse_time + gap,
            strength=p.strength,
            label=p.label,
        )
        for p in second
    ]
    return first + shifted


def make_sparse_or_garbage_pulse(seed: int | None = None) -> list[PulseEvent]:
    """Return a noisy, low-confidence pulse sequence (sparse random timestamps).

    This simulates garbage input: a few random hits with no stable tempo.
    The output should always result in very low confidence.
    """
    rng = random.Random(seed)
    num_pulses = rng.randint(2, 5)
    times: set[float] = set()
    for _ in range(num_pulses):
        times.add(round(rng.uniform(0.0, 2.0), 3))
    return [
        PulseEvent(time_seconds=t, strength=0.3, label="noise") for t in sorted(times)
    ]