"""Regression tests for synthetic humanization (drummer/humanize.py)."""

from __future__ import annotations

import sys
from pathlib import Path

from drummer.humanize import humanize_events
from models import KICK, SNARE, CLOSED_HAT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---- helpers ----------------------------------------------------------------


def _make_fake_events(count: int = 8) -> list[dict]:
    """Build a repeatable set of synthetic scheduler events."""
    return [
        {"timestamp": t, "step": i, "instrument": inst, "note": note, "velocity": vel}
        for i, (t, inst, note, vel) in enumerate(
            [
                (0.000, "kick", KICK, 100),
                (0.125, "hat", CLOSED_HAT, 80),
                (0.250, "snare", SNARE, 100),
                (0.375, "hat", CLOSED_HAT, 80),
                (0.500, "kick", KICK, 100),
                (0.625, "hat", CLOSED_HAT, 80),
                (0.750, "snare", SNARE, 100),
                (0.875, "hat", CLOSED_HAT, 80),
            ]
        )
    ]


def _all_fields_match(
    original: list[dict], humanized: list[dict], key: str
) -> bool:
    return all(
        orig[key] == hum[key]
        for orig, hum in zip(original, humanized)
    )


# ---- tests ------------------------------------------------------------------


def test_humanize_preserves_event_count() -> None:
    events = _make_fake_events(8)
    result = humanize_events(events, seed=0)
    assert len(result) == len(events)


def test_humanize_preserves_instruments() -> None:
    events = _make_fake_events(8)
    result = humanize_events(events, seed=0)
    assert _all_fields_match(events, result, "instrument")


def test_humanize_preserves_midi_notes() -> None:
    events = _make_fake_events(8)
    result = humanize_events(events, seed=0)
    assert _all_fields_match(events, result, "note")


def test_humanize_velocities_stay_in_range() -> None:
    events = _make_fake_events(8)
    result = humanize_events(events, seed=0)
    for ev in result:
        assert 0 <= ev["velocity"] <= 127, f"Velocity out of range: {ev['velocity']}"


def test_humanize_timestamps_stay_non_negative() -> None:
    events = _make_fake_events(8)
    result = humanize_events(events, seed=0)
    for ev in result:
        assert ev["timestamp"] >= 0.0, f"Negative timestamp: {ev['timestamp']}"


def test_humanize_event_order_is_sensible() -> None:
    events = _make_fake_events(8)
    result = humanize_events(events, seed=0)
    timestamps = [ev["timestamp"] for ev in result]
    for i in range(1, len(timestamps)):
        assert timestamps[i] >= timestamps[i - 1], (
            f"Timestamps not monotonic at index {i}: {timestamps[i - 1]} -> {timestamps[i]}"
        )


def test_humanize_deterministic_same_seed() -> None:
    events = _make_fake_events(8)
    r1 = humanize_events(events, seed=42)
    r2 = humanize_events(events, seed=42)
    for e1, e2 in zip(r1, r2):
        assert e1["timestamp"] == e2["timestamp"]
        assert e1["velocity"] == e2["velocity"]


def test_humanize_changes_something() -> None:
    """At least one timing or velocity value should differ from the original."""
    events = _make_fake_events(16)
    result = humanize_events(events, seed=7)
    # Check timing
    timing_changed = any(
        orig["timestamp"] != hum["timestamp"]
        for orig, hum in zip(events, result)
    )
    velocity_changed = any(
        orig["velocity"] != hum["velocity"]
        for orig, hum in zip(events, result)
    )
    assert timing_changed or velocity_changed, (
        "Humanization should alter at least one timing or velocity value"
    )


def test_humanize_different_seed_different_result() -> None:
    """Different seeds should produce different output."""  # noqa: D401
    events = _make_fake_events(16)
    r1 = humanize_events(events, seed=1)
    r2 = humanize_events(events, seed=999)
    # It's vanishingly unlikely that two different seeds produce identical
    # results for 16 events.
    identical = all(
        e1["timestamp"] == e2["timestamp"] and e1["velocity"] == e2["velocity"]
        for e1, e2 in zip(r1, r2)
    )
    assert not identical, "Different seeds should produce different humanization"