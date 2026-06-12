"""Tests for synthetic noisy pulse input.

Verifies that the decision layer stays calm under imperfect input — jittered
timing, missing hits, extra hits, tempo changes, and garbage input — without
making wild groove switches.

Each test runs the full pulse→onset→tempo→confidence→decision pipeline.
"""

from __future__ import annotations

import sys
from pathlib import Path

from audio.onset_detector import detect_onsets_from_events
from confidence_engine import estimate_timing_confidence
from groove_matcher import select_groove_decision
from pulse_tracker import estimate_tempo
from synthetic.noisy_pulse import (
    add_extra_pulse,
    add_timing_jitter,
    drop_pulse,
    make_sparse_or_garbage_pulse,
    make_steady_pulse,
    make_tempo_change_pulse,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _analyse(pulses: list, expected_bpm: float = 120.0) -> dict:
    """Run pulse→onset→tempo→confidence and return analysis results."""
    onsets = detect_onsets_from_events(pulses, min_interval=0.05)
    onset_times = [e.time_seconds for e in onsets]
    estimated_bpm = estimate_tempo(onset_times)
    confidence = estimate_timing_confidence(onset_times, expected_bpm=expected_bpm)
    return {
        "estimated_bpm": estimated_bpm,
        "confidence": confidence,
        "onset_count": len(onset_times),
    }


# ---------------------------------------------------------------------------
# 1. Steady pulse
# ---------------------------------------------------------------------------


def test_steady_120_still_selects_motorik() -> None:
    """Steady 120 BPM pulse still gives high confidence and selects motorik."""
    pulses = make_steady_pulse(bpm=120.0, bars=2)
    a = _analyse(pulses, expected_bpm=120.0)
    d = select_groove_decision(bpm=a["estimated_bpm"], confidence=a["confidence"])
    assert a["confidence"] > 0.7, f"Expected high confidence, got {a['confidence']:.2f}"
    assert d.selected_groove_id == "motorik", f"Expected motorik, got {d.selected_groove_id}"
    assert d.reason == "high_confidence_motorik"


# ---------------------------------------------------------------------------
# 2. Slight jitter
# ---------------------------------------------------------------------------


def test_jittered_120_no_wild_change() -> None:
    """Slight timing jitter around 120 BPM does not cause a wild groove change."""
    steady = make_steady_pulse(bpm=120.0, bars=2)
    jittered = add_timing_jitter(steady, amount_ms=15.0, seed=42)
    a = _analyse(jittered, expected_bpm=120.0)
    d = select_groove_decision(
        bpm=a["estimated_bpm"],
        confidence=a["confidence"],
        previous_groove_id="motorik",
    )
    # Even with jitter, confidence should be high enough to keep motorik
    # or at least not switch to something inappropriate
    assert d.selected_groove_id in ("motorik", "simple_rock"), (
        f"Jittered pulse caused unexpected groove: {d.selected_groove_id}"
    )


# ---------------------------------------------------------------------------
# 3. Missing hit
# ---------------------------------------------------------------------------


def test_missing_hit_120_no_wild_change() -> None:
    """One missing hit does not cause a wild groove change if previous exists."""
    steady = make_steady_pulse(bpm=120.0, bars=2)
    # Drop beat index 3 (the fourth pulse)
    missing = drop_pulse(steady, index=3)
    a = _analyse(missing, expected_bpm=120.0)
    d = select_groove_decision(
        bpm=a["estimated_bpm"],
        confidence=a["confidence"],
        previous_groove_id="motorik",
    )
    # Should keep previous (motorik) or at worst fall to simple_rock,
    # not switch to something like half_time
    assert d.selected_groove_id in ("motorik", "simple_rock"), (
        f"Missing hit caused unexpected groove: {d.selected_groove_id}"
    )
    # If confidence is low, reason should be keep_previous
    if a["confidence"] < 0.4:
        assert d.reason == "low_confidence_keep_previous", (
            f"Expected keep_previous reason, got {d.reason}"
        )


# ---------------------------------------------------------------------------
# 4. Extra offbeat hit
# ---------------------------------------------------------------------------


def test_extra_offbeat_120_no_wild_change() -> None:
    """One extra offbeat hit does not cause a wild groove change if previous exists."""
    steady = make_steady_pulse(bpm=120.0, bars=2)
    # Add an offbeat hit halfway between beats 1 and 2
    quarter = 60.0 / 120.0
    extra_time = steady[1].time_seconds + quarter / 2
    extra = add_extra_pulse(steady, timestamp=extra_time, strength=0.5)
    a = _analyse(extra, expected_bpm=120.0)
    d = select_groove_decision(
        bpm=a["estimated_bpm"],
        confidence=a["confidence"],
        previous_groove_id="motorik",
    )
    assert d.selected_groove_id in ("motorik", "simple_rock"), (
        f"Extra hit caused unexpected groove: {d.selected_groove_id}"
    )
    if a["confidence"] < 0.4:
        assert d.reason == "low_confidence_keep_previous"


# ---------------------------------------------------------------------------
# 5. Tempo change (clean high-confidence)
# ---------------------------------------------------------------------------


def test_tempo_change_120_to_90_switches_groove() -> None:
    """A clean high-confidence tempo change from 120 to 90 BPM can switch to half_time."""
    pulses = make_tempo_change_pulse(start_bpm=120.0, end_bpm=90.0, bars_each=2)
    # Analyse the second half (90 BPM portion)
    # The tempo change pulse adds both sections; take the second half
    quarter_90 = 60.0 / 90.0
    # Find the break point — pulses after the gap are the second section
    # The gap is 1 quarter of start_bpm = 0.5s
    gap = 60.0 / 120.0
    first_duration = 4 * (60.0 / 120.0) * 2  # 2 bars at 120 BPM
    cut_time = first_duration + gap
    second_pulses = [p for p in pulses if p.time_seconds >= cut_time]

    a = _analyse(second_pulses, expected_bpm=90.0)
    d = select_groove_decision(
        bpm=a["estimated_bpm"],
        confidence=a["confidence"],
        previous_groove_id="motorik",
    )

    # If confidence is high enough, should switch to half_time
    if a["confidence"] >= 0.7:
        assert d.selected_groove_id == "half_time", (
            f"Expected half_time for 90 BPM, got {d.selected_groove_id}"
        )
        assert d.reason == "high_confidence_half_time"
        assert d.changed is True
    # If confidence is moderate, it should hold previous
    else:
        assert d.selected_groove_id == "motorik", (
            f"Expected motorik (hold) at moderate conf, got {d.selected_groove_id}"
        )
        assert d.changed is False


# ---------------------------------------------------------------------------
# 6. Garbage input with previous groove
# ---------------------------------------------------------------------------


def test_garbage_with_previous_keeps_groove() -> None:
    """Low-confidence garbage input keeps the previous groove if provided."""
    pulses = make_sparse_or_garbage_pulse(seed=99)
    a = _analyse(pulses, expected_bpm=120.0)
    d = select_groove_decision(
        bpm=a["estimated_bpm"],
        confidence=a["confidence"],
        previous_groove_id="motorik",
    )
    assert a["confidence"] < 0.4, (
        f"Garbage pulse should give low confidence, got {a['confidence']:.2f}"
    )
    assert d.selected_groove_id == "motorik", (
        f"Expected to keep motorik, got {d.selected_groove_id}"
    )
    assert d.reason == "low_confidence_keep_previous"
    assert d.changed is False


# ---------------------------------------------------------------------------
# 7. Garbage input with no previous groove
# ---------------------------------------------------------------------------


def test_garbage_no_previous_falls_back_to_simple_rock() -> None:
    """Low-confidence garbage input falls back to simple_rock if no previous."""
    pulses = make_sparse_or_garbage_pulse(seed=99)
    a = _analyse(pulses, expected_bpm=120.0)
    d = select_groove_decision(
        bpm=a["estimated_bpm"],
        confidence=a["confidence"],
    )
    assert a["confidence"] < 0.4, (
        f"Garbage pulse should give low confidence, got {a['confidence']:.2f}"
    )
    assert d.selected_groove_id == "simple_rock", (
        f"Expected simple_rock fallback, got {d.selected_groove_id}"
    )
    assert d.reason == "low_confidence_default"
    assert d.changed is None


# ---------------------------------------------------------------------------
# 8. Reason codes present for each scenario
# ---------------------------------------------------------------------------


def test_reason_code_present_across_scenarios() -> None:
    """Each noisy scenario produces a valid reason code."""
    scenarios = [
        ("steady", make_steady_pulse(bpm=120.0, bars=2), 120.0),
        ("jittered", add_timing_jitter(make_steady_pulse(bpm=120.0, bars=2), seed=7), 120.0),
        ("missing", drop_pulse(make_steady_pulse(bpm=120.0, bars=2), index=2), 120.0),
        ("extra", add_extra_pulse(make_steady_pulse(bpm=120.0, bars=2), timestamp=0.6), 120.0),
        ("garbage", make_sparse_or_garbage_pulse(seed=99), 120.0),
    ]
    valid_reasons = {
        "high_confidence_motorik",
        "high_confidence_half_time",
        "high_confidence_simple_rock",
        "low_confidence_keep_previous",
        "low_confidence_default",
        "unknown_fallback",
    }
    for name, pulses, expected_bpm in scenarios:
        a = _analyse(pulses, expected_bpm=expected_bpm)
        d = select_groove_decision(
            bpm=a["estimated_bpm"],
            confidence=a["confidence"],
            previous_groove_id="motorik",
        )
        assert d.reason in valid_reasons, (
            f"Scenario '{name}' produced invalid reason '{d.reason}'"
        )
        assert d.selected_groove_id in ("motorik", "half_time", "simple_rock"), (
            f"Scenario '{name}' produced unexpected groove '{d.selected_groove_id}'"
        )


# ---------------------------------------------------------------------------
# 9. Seeded noisy input is deterministic
# ---------------------------------------------------------------------------


def test_seeded_noisy_pulse_deterministic() -> None:
    """Seeded noisy input produces identical results every time."""
    r1 = add_timing_jitter(make_steady_pulse(bpm=120.0, bars=1), amount_ms=10.0, seed=42)
    r2 = add_timing_jitter(make_steady_pulse(bpm=120.0, bars=1), amount_ms=10.0, seed=42)
    r3 = add_timing_jitter(make_steady_pulse(bpm=120.0, bars=1), amount_ms=10.0, seed=42)

    for i in range(len(r1)):
        assert r1[i].time_seconds == r2[i].time_seconds == r3[i].time_seconds, (
            f"Jittered pulse {i} times differ despite same seed"
        )

    g1 = make_sparse_or_garbage_pulse(seed=42)
    g2 = make_sparse_or_garbage_pulse(seed=42)
    g3 = make_sparse_or_garbage_pulse(seed=42)
    for i in range(len(g1)):
        assert g1[i].time_seconds == g2[i].time_seconds == g3[i].time_seconds, (
            f"Garbage pulse {i} times differ despite same seed"
        )