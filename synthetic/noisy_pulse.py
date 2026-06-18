"""Synthetic pulse helpers that simulate realistic playing or detection problems.

Every function returns a list of ``PulseEvent`` objects compatible with the
existing ``detect_onsets_from_events`` pipeline.

Design principle:
    Detection and decision stay separate. These helpers only create imperfect
    synthetic input. The tempo/confidence analysis and groove decision layers
    consume the output untouched.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from audio.pulse_generator import PulseEvent, generate_pulse_events


@dataclass(frozen=True)
class HumanTimingProfile:
    """A deterministic profile for human-like pulse timing.

    Parameters
    ----------
    name:
        Short profile label used in batch reports.
    jitter_ms:
        Random timing variation applied to every beat.
    drift_ms_per_beat:
        Linear push or drag over time. Positive values get later each beat,
        negative values get earlier.
    swing_ms:
        Alternating long/short feel. Odd-numbered beats are pushed later and
        even-numbered beats after the first are pulled earlier.
    drop_every:
        Optional regular missing-beat interval. For example, 7 drops every
        seventh generated pulse.
    extra_offbeat_every:
        Optional regular extra hit interval. For example, 4 adds a quieter
        offbeat after every fourth generated pulse.
    """

    name: str
    jitter_ms: float = 0.0
    drift_ms_per_beat: float = 0.0
    swing_ms: float = 0.0
    drop_every: int | None = None
    extra_offbeat_every: int | None = None


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


def make_human_pulse(
    bpm: float = 120.0,
    bars: int = 4,
    profile: HumanTimingProfile | None = None,
    seed: int | None = None,
) -> list[PulseEvent]:
    """Return pulse events with repeatable human-like timing imperfections.

    This keeps generation separate from analysis: callers still feed the output
    through the same onset and tempo pipeline as real detections.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if bars <= 0:
        raise ValueError("bars must be positive")

    profile = profile or HumanTimingProfile(name="steady")
    rng = random.Random(seed)
    interval = 60.0 / bpm
    beat_count = bars * 4
    pulses: list[PulseEvent] = []

    for beat_index in range(beat_count):
        if (
            profile.drop_every is not None
            and profile.drop_every > 0
            and beat_index > 0
            and beat_index % profile.drop_every == 0
        ):
            continue

        base_time = beat_index * interval
        jitter = rng.uniform(
            -profile.jitter_ms / 1000.0,
            profile.jitter_ms / 1000.0,
        )
        drift = beat_index * profile.drift_ms_per_beat / 1000.0
        swing = 0.0
        if beat_index > 0 and profile.swing_ms:
            swing_direction = 1.0 if beat_index % 2 else -1.0
            swing = swing_direction * profile.swing_ms / 1000.0

        time_seconds = max(0.0, base_time + jitter + drift + swing)
        strength = 1.0 if beat_index % 4 == 0 else 0.72
        pulses.append(
            PulseEvent(
                time_seconds=time_seconds,
                strength=strength,
                label="downbeat" if beat_index % 4 == 0 else "human_pulse",
            )
        )

        if (
            profile.extra_offbeat_every is not None
            and profile.extra_offbeat_every > 0
            and beat_index > 0
            and beat_index % profile.extra_offbeat_every == 0
        ):
            extra_jitter = rng.uniform(-0.015, 0.015)
            pulses.append(
                PulseEvent(
                    time_seconds=max(0.0, base_time + interval * 0.5 + extra_jitter),
                    strength=0.38,
                    label="extra",
                )
            )

    pulses.sort(key=lambda e: e.time_seconds)
    return pulses


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
