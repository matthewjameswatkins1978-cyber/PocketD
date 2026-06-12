"""Test that the synthetic pipeline extends through the scheduler stage."""

from __future__ import annotations

import sys
from pathlib import Path

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from confidence_engine import estimate_timing_confidence
from groove_matcher import select_groove
from models import KICK, SNARE, CLOSED_HAT
from pulse_tracker import estimate_tempo
from scheduler import hits_at_step, step_duration_seconds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_scheduler_produces_hits_across_two_bars() -> None:
    pulse_events = generate_pulse_events(bpm=120.0, duration_seconds=2.0)
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)
    bpm = estimate_tempo([event.time_seconds for event in onsets])
    confidence = estimate_timing_confidence(
        [event.time_seconds for event in onsets], expected_bpm=120.0
    )

    groove = select_groove(
        {"density": 0.6, "syncopation": 0.2, "strong_beats": [1, 3]},
        confidence=confidence,
        personality="Anchor",
    )

    assert groove is not None

    step_dur = step_duration_seconds(bpm)
    total_steps = groove.steps * 2  # 2 bars
    events: list[dict] = []

    for step_idx in range(total_steps):
        step = step_idx % groove.steps
        timestamp = step_idx * step_dur
        hits = hits_at_step(groove, step, complexity_level=3)
        for instrument, note in hits:
            velocity = 80 if instrument == "hat" else 100
            events.append(
                {
                    "timestamp": timestamp,
                    "step": step,
                    "instrument": instrument,
                    "note": note,
                    "velocity": velocity,
                }
            )

    # At least one scheduled event across 2 bars
    assert len(events) > 0, "Scheduler should produce at least one hit in 2 bars"

    # All instruments are valid
    valid_notes = {KICK, SNARE, CLOSED_HAT}
    valid_instruments = {"kick", "snare", "hat"}
    for ev in events:
        assert ev["instrument"] in valid_instruments
        assert ev["note"] in valid_notes
        assert 0 <= ev["velocity"] <= 127

    # Timestamps are monotonically increasing
    timestamps = [ev["timestamp"] for ev in events]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], "Timestamps must be non-decreasing"

    # All timestamps are non-negative
    assert all(t >= 0.0 for t in timestamps)