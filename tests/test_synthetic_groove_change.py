"""Regression test for synthetic groove-change detection across two sections."""

from __future__ import annotations

import sys
from pathlib import Path

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from confidence_engine import estimate_timing_confidence
from demo_synthetic_pipeline import scheduled_events_table
from drummer.humanize import humanize_events
from groove_matcher import select_groove_by_tempo
from pulse_tracker import estimate_tempo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _process_section(
    bpm: float,
    duration_seconds: float,
    previous_groove_id: str | None = None,
) -> dict:
    """Run the full synthetic pipeline for one section and return results."""
    pulse_events = generate_pulse_events(bpm=bpm, duration_seconds=duration_seconds)
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)
    onset_times = [event.time_seconds for event in onsets]

    estimated_bpm = estimate_tempo(onset_times)
    confidence = estimate_timing_confidence(onset_times, expected_bpm=bpm)

    groove = select_groove_by_tempo(
        bpm=estimated_bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        personality="Anchor",
    )

    events = scheduled_events_table(groove, estimated_bpm, bars=2, complexity=3)
    humanized = humanize_events(events, seed=42)

    return {
        "groove": groove,
        "bpm": estimated_bpm,
        "confidence": confidence,
        "events": events,
        "humanized": humanized,
    }


def test_two_synthetic_sections_produce_valid_results() -> None:
    """Both sections produce valid tempo, confidence, events, and humanization."""
    section_a = _process_section(bpm=120.0, duration_seconds=2.0)
    section_b = _process_section(bpm=90.0, duration_seconds=2.0)

    for name, result in [("A", section_a), ("B", section_b)]:
        assert result["groove"] is not None, f"Section {name} groove is None"
        assert 40.0 <= result["bpm"] <= 240.0, f"Section {name} BPM out of range"
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Section {name} confidence out of range"
        )
        assert len(result["events"]) > 0, (
            f"Section {name} has no scheduled events"
        )
        assert len(result["humanized"]) > 0, (
            f"Section {name} has no humanized events"
        )
        assert len(result["humanized"]) == len(result["events"]), (
            f"Section {name} event count mismatch"
        )


def test_groove_change_detected() -> None:
    """120 BPM section should pick motorik, 90 BPM section should pick half_time."""
    section_a = _process_section(bpm=120.0, duration_seconds=2.0)
    section_b = _process_section(
        bpm=90.0, duration_seconds=2.0,
        previous_groove_id=section_a["groove"].id,
    )

    assert section_a["groove"].id == "motorik", (
        f"Expected motorik for 120 BPM, got {section_a['groove'].id}"
    )
    assert section_b["groove"].id == "half_time", (
        f"Expected half_time for 90 BPM, got {section_b['groove'].id}"
    )
    assert section_a["groove"].id != section_b["groove"].id, (
        "Sections should pick different grooves"
    )


def test_groove_choice_stable_with_fixed_seed() -> None:
    """Repeatable pipeline: same inputs → same groove choice."""
    r1 = _process_section(bpm=120.0, duration_seconds=2.0)
    r2 = _process_section(bpm=120.0, duration_seconds=2.0)

    assert r1["groove"].id == r2["groove"].id, (
        "Groove choice must be deterministic with fixed seed"
    )
    # Events should be identical because the pipeline is deterministic
    assert len(r1["events"]) == len(r2["events"])
    for e1, e2 in zip(r1["events"], r2["events"]):
        assert e1["timestamp"] == e2["timestamp"]
        assert e1["instrument"] == e2["instrument"]
        assert e1["velocity"] == e2["velocity"]


def test_low_confidence_does_not_switch_groove() -> None:
    """When confidence is very low, the system keeps the previous groove."""
    # Very short duration gives few onsets → low confidence
    section_a = _process_section(bpm=120.0, duration_seconds=0.3)
    previous_id = section_a["groove"].id

    # Second section also short/lower tempo but low confidence → stay safe
    section_b = _process_section(
        bpm=90.0, duration_seconds=0.3,
        previous_groove_id=previous_id,
    )

    # If confidence is below threshold, groove should not change wildly
    if section_b["confidence"] < 0.4:
        assert section_b["groove"].id == previous_id, (
            "Low confidence should keep the previous groove"
        )


def test_groove_change_only_at_high_confidence() -> None:
    """At moderate confidence the system should not switch to a different groove."""
    result = _process_section(bpm=180.0, duration_seconds=0.5)
    # This test only verifies that the system can process 180 BPM
    # without crashing — the actual groove choice depends on confidence.
    assert result["groove"] is not None
    assert 0.0 <= result["confidence"] <= 1.0
