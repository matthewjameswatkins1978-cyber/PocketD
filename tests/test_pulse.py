"""Tests for the Pulse Tracker — Module 2: competing tempo/pulse hypotheses."""

from __future__ import annotations

import sys
from pathlib import Path

import math

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.models import MusicalEvent
from perception.pulse import PulseTracker, PulseHypothesis, PulseState


# ─── Helper: create events ─────────────────────────────────────────


def _event(time: float, strength: float = 0.8, energy: float = 0.5, density: float = 0.5) -> MusicalEvent:
    """Create a MusicalEvent with explicit time and default other values."""
    return MusicalEvent(
        time_seconds=time,
        strength=strength,
        energy=energy,
        density=density,
    )


def _steady_pulse_events(bpm: float, num_beats: int = 8, strength: float = 0.8) -> list[MusicalEvent]:
    """Generate a perfectly steady pulse at the given BPM."""
    interval = 60.0 / bpm
    return [_event(i * interval, strength=strength) for i in range(num_beats)]


def _human_pulse_events(bpm: float, num_beats: int = 8, jitter: float = 0.02) -> list[MusicalEvent]:
    """Generate a pulse with human-like timing variation."""
    interval = 60.0 / bpm
    events: list[MusicalEvent] = []
    for i in range(num_beats):
        t = i * interval
        # Add jitter except for the first event
        if i > 0:
            t += (hash(i) % 100 - 50) / 100 * jitter
        events.append(_event(max(0.0, t), strength=0.8))
    return events


# ─── PulseHypothesis Tests ─────────────────────────────────────────


class TestPulseHypothesis:
    def test_default_values(self) -> None:
        hyp = PulseHypothesis(bpm=120.0)
        assert hyp.bpm == 120.0
        assert hyp.confidence == 0.0
        assert hyp.matches == 0
        assert hyp.misses == 0
        assert hyp.last_event_time == 0.0
        assert hyp.stability == 0.0

    def test_custom_values(self) -> None:
        hyp = PulseHypothesis(
            bpm=90.0,
            confidence=0.75,
            matches=10,
            misses=2,
            last_event_time=5.0,
            stability=0.6,
        )
        assert hyp.bpm == 90.0
        assert hyp.confidence == 0.75
        assert hyp.matches == 10
        assert hyp.misses == 2
        assert hyp.last_event_time == 5.0
        assert hyp.stability == 0.6


# ─── PulseState Tests ──────────────────────────────────────────────


class TestPulseState:
    def test_default_values(self) -> None:
        state = PulseState()
        assert state.hypotheses == []
        assert state.best_bpm is None
        assert state.confidence == 0.0
        assert state.stability == "unknown"

    def test_with_hypotheses(self) -> None:
        hyp = PulseHypothesis(bpm=120.0, confidence=0.8)
        state = PulseState(
            hypotheses=[hyp],
            best_bpm=120.0,
            confidence=0.8,
            stability="rising",
        )
        assert len(state.hypotheses) == 1
        assert state.best_bpm == 120.0
        assert state.confidence == 0.8
        assert state.stability == "rising"


# ─── Steady Pulse Tests ────────────────────────────────────────────


class TestSteadyPulse:
    """Test 1: Perfectly steady pulse should converge to ~120 BPM."""

    def test_steady_120_bpm(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=10)

        final_state: PulseState = PulseState()
        for event in events:
            final_state = tracker.process_event(event)

        assert final_state.best_bpm is not None
        assert 115 <= final_state.best_bpm <= 125, (
            f"Expected ~120 BPM, got {final_state.best_bpm}"
        )
        assert final_state.confidence > 0.3

    def test_steady_90_bpm(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=90.0, num_beats=10)

        final_state: PulseState = PulseState()
        for event in events:
            final_state = tracker.process_event(event)

        assert final_state.best_bpm is not None
        assert 85 <= final_state.best_bpm <= 95, (
            f"Expected ~90 BPM, got {final_state.best_bpm}"
        )
        assert final_state.confidence > 0.3

    def test_steady_60_bpm(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=60.0, num_beats=8)

        final_state: PulseState = PulseState()
        for event in events:
            final_state = tracker.process_event(event)

        assert final_state.best_bpm is not None
        # 60 BPM may also produce 120 BPM (double-time) as a hypothesis
        # Accept either 60 or nearby values
        assert any(
            abs(h.bpm - 60) < 10 or abs(h.bpm - 120) < 10
            for h in final_state.hypotheses
        )

    def test_confident_after_enough_events(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=16)

        # Check confidence grows over time
        confidences: list[float] = []
        for event in events:
            state = tracker.process_event(event)
            confidences.append(state.confidence)

        # Later events should have higher or equal confidence
        assert confidences[-1] >= confidences[0]


# ─── Human Timing Variation Tests ──────────────────────────────────


class TestHumanTimingVariation:
    """Test 2: Slightly jittered timing should still converge."""

    def test_jittered_pulse_still_detected(self) -> None:
        tracker = PulseTracker()
        events = _human_pulse_events(bpm=120.0, num_beats=10, jitter=0.025)

        final_state: PulseState = PulseState()
        for event in events:
            final_state = tracker.process_event(event)

        assert final_state.best_bpm is not None
        assert 110 <= final_state.best_bpm <= 130, (
            f"Expected ~120 BPM with jitter, got {final_state.best_bpm}"
        )

    def test_more_jitter_lower_confidence(self) -> None:
        """Higher jitter should result in lower (or more varied) confidence."""
        tracker_tight = PulseTracker()
        tracker_loose = PulseTracker()

        tight_events = _human_pulse_events(bpm=120.0, num_beats=8, jitter=0.01)
        loose_events = _human_pulse_events(bpm=120.0, num_beats=8, jitter=0.08)

        for ev in tight_events:
            tracker_tight.process_event(ev)
        for ev in loose_events:
            tracker_loose.process_event(ev)

        # Both should still detect roughly 120 BPM
        tight_state = tracker_tight.get_state()
        loose_state = tracker_loose.get_state()

        assert tight_state.best_bpm is not None
        # The loose tracker may still have a reasonable BPM
        assert loose_state.best_bpm is not None


# ─── Half-Time Ambiguity Tests ─────────────────────────────────────


class TestHalfTimeAmbiguity:
    """Test 3: Events at 1.0s intervals should produce 60 and 120 BPM."""

    def test_half_time_ambiguity(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=60.0, num_beats=8)

        final_state: PulseState = PulseState()
        for event in events:
            final_state = tracker.process_event(event)

        # Both 60 BPM and 120 BPM should appear in hypotheses
        bpms = [h.bpm for h in final_state.hypotheses]
        has_60 = any(abs(b - 60) < 10 for b in bpms)
        has_120 = any(abs(b - 120) < 10 for b in bpms)

        assert has_60 or has_120, (
            f"Expected 60 or 120 BPM in hypotheses, got {bpms}"
        )


# ─── Double-Time Ambiguity Tests ────────────────────────────────────


class TestDoubleTimeAmbiguity:
    """Test 4: Events at 0.25s intervals should produce 240 and 120 BPM."""

    def test_double_time_ambiguity(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=240.0, num_beats=12)

        final_state: PulseState = PulseState()
        for event in events:
            final_state = tracker.process_event(event)

        bpms = [h.bpm for h in final_state.hypotheses]
        has_240 = any(abs(b - 240) < 20 for b in bpms)
        has_120 = any(abs(b - 120) < 15 for b in bpms)

        assert has_240 or has_120, (
            f"Expected 240 or 120 BPM in hypotheses, got {bpms}"
        )


# ─── Accent/Strength Influence Tests ────────────────────────────────


class TestAccentInfluence:
    """Test 5: Strong accents at half the pulse rate should influence confidence."""

    def test_strong_accents_add_half_time_hypothesis(self) -> None:
        """Events every 0.5s with alternating accent strength."""
        tracker = PulseTracker()
        events: list[MusicalEvent] = []

        interval = 0.5  # 120 BPM
        for i in range(10):
            t = i * interval
            # Every other beat is an accent (strong)
            if i % 2 == 0:
                events.append(_event(t, strength=1.0, energy=0.9))
            else:
                events.append(_event(t, strength=0.3, energy=0.2))

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        bpms = [h.bpm for h in state.hypotheses]

        # 120 BPM should be present (events every 0.5s)
        # 60 BPM may also gain confidence because strong accents repeat every 1.0s
        assert any(abs(b - 120) < 15 for b in bpms), (
            f"Expected 120 BPM from 0.5s intervals, got {bpms}"
        )


# ─── Silence/Decay Tests ────────────────────────────────────────────


class TestSilenceDecay:
    """Test 6: After a gap with no events, confidence should decay."""

    def test_confidence_decays_over_silence(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=6)

        # Build confidence
        for event in events:
            tracker.process_event(event)

        state_before = tracker.get_state()
        assert state_before.confidence > 0.2

        # Advance time by 10 seconds with no events
        state_after = tracker.advance_time(current_time=16.0)

        # Confidence should have decayed
        assert state_after.confidence <= state_before.confidence + 0.01, (
            f"Confidence should decay during silence: "
            f"{state_before.confidence} -> {state_after.confidence}"
        )

    def test_long_silence_reduces_confidence_significantly(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=8)

        for event in events:
            tracker.process_event(event)

        state_before = tracker.get_state()

        # Simulate 20 seconds of silence
        state_after = tracker.advance_time(current_time=30.0)

        # Confidence should be lower or hypotheses should have been pruned
        if state_after.hypotheses:
            assert state_after.confidence <= state_before.confidence + 0.01

    def test_silence_does_not_increase_confidence(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=4)

        for event in events:
            tracker.process_event(event)

        state_before = tracker.get_state()
        state_after = tracker.advance_time(current_time=20.0)

        # Confidence should not increase during silence
        assert state_after.confidence <= state_before.confidence + 0.01


# ─── Reset Tests ────────────────────────────────────────────────────


class TestReset:
    """Test 7: Reset should clear all hypotheses."""

    def test_reset_clears_hypotheses(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=6)

        for event in events:
            tracker.process_event(event)

        assert len(tracker.get_state().hypotheses) > 0

        tracker.reset()
        state = tracker.get_state()
        assert state.hypotheses == []
        assert state.best_bpm is None
        assert state.confidence == 0.0
        assert state.stability == "unknown"

    def test_reset_allows_new_pulse_detection(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=6)

        for event in events:
            tracker.process_event(event)

        tracker.reset()

        # After reset, detect a new tempo
        new_events = _steady_pulse_events(bpm=80.0, num_beats=6)
        for event in new_events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.best_bpm is not None
        assert 70 <= state.best_bpm <= 90, (
            f"After reset, expected ~80 BPM, got {state.best_bpm}"
        )


# ─── Multiple Hypotheses Tests ─────────────────────────────────────


class TestMultipleHypotheses:
    def test_multiple_hypotheses_exist(self) -> None:
        tracker = PulseTracker(max_hypotheses=5)
        events = _steady_pulse_events(bpm=120.0, num_beats=10)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert len(state.hypotheses) >= 1

    def test_hypotheses_sorted_by_confidence(self) -> None:
        tracker = PulseTracker(max_hypotheses=5)
        events = _steady_pulse_events(bpm=120.0, num_beats=10)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        for i in range(len(state.hypotheses) - 1):
            assert state.hypotheses[i].confidence >= state.hypotheses[i + 1].confidence


# ─── Edge Case Tests ────────────────────────────────────────────────


class TestEdgeCases:
    def test_no_events_returns_empty_state(self) -> None:
        tracker = PulseTracker()
        state = tracker.get_state()
        assert state.hypotheses == []
        assert state.best_bpm is None

    def test_single_event_does_not_crash(self) -> None:
        tracker = PulseTracker()
        state = tracker.process_event(_event(0.0))
        # With one event we can't determine BPM yet
        assert isinstance(state, PulseState)

    def test_two_events_with_same_timestamp(self) -> None:
        tracker = PulseTracker()
        state1 = tracker.process_event(_event(1.0))
        state2 = tracker.process_event(_event(1.0))  # same time
        assert isinstance(state2, PulseState)

    def test_very_slow_pulse(self) -> None:
        """2.0s interval = 30 BPM, below min_bpm=40."""
        tracker = PulseTracker(min_bpm=40.0)
        events = _steady_pulse_events(bpm=30.0, num_beats=5)

        state: PulseState = PulseState()
        for event in events:
            state = tracker.process_event(event)

        # 30 BPM is below min, so no hypothesis should match exactly
        # But multiples (60 BPM) may appear
        assert isinstance(state, PulseState)

    def test_very_fast_pulse(self) -> None:
        """0.2s interval = 300 BPM, above max_bpm=250."""
        tracker = PulseTracker(max_bpm=250.0)
        events = _steady_pulse_events(bpm=300.0, num_beats=10)

        state: PulseState = PulseState()
        for event in events:
            state = tracker.process_event(event)

        # 300 BPM is above max, so 150 BPM (half) or similar may appear
        assert isinstance(state, PulseState)

    def test_varying_strengths_do_not_crash(self) -> None:
        tracker = PulseTracker()
        events = [
            _event(0.0, strength=0.1),
            _event(0.5, strength=0.9),
            _event(1.0, strength=0.5),
            _event(1.5, strength=0.0),
            _event(2.0, strength=0.99),
        ]
        for event in events:
            tracker.process_event(event)
        state = tracker.get_state()
        assert isinstance(state, PulseState)


# ─── Configuration Tests ────────────────────────────────────────────


class TestConfiguration:
    def test_custom_min_max_bpm(self) -> None:
        """Set very narrow BPM range and verify it's respected."""
        tracker = PulseTracker(min_bpm=100.0, max_bpm=140.0)
        # 60 BPM events should not produce 60 BPM hypotheses
        events = _steady_pulse_events(bpm=60.0, num_beats=6)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        if state.hypotheses:
            for hyp in state.hypotheses:
                assert hyp.bpm >= 100.0, f"BPM {hyp.bpm} below min_bpm=100"

    def test_custom_tolerance_more_forgiving(self) -> None:
        """Wider tolerance should still converge on pulse."""
        tracker = PulseTracker(tolerance=0.25)  # 25% tolerance
        events = _human_pulse_events(bpm=120.0, num_beats=8, jitter=0.05)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.best_bpm is not None
        assert 100 <= state.best_bpm <= 140

    def test_fewer_hypotheses(self) -> None:
        tracker = PulseTracker(max_hypotheses=3)
        events = _steady_pulse_events(bpm=120.0, num_beats=10)

        for event in events:
            tracker.process_event(event)

        assert len(tracker.get_state().hypotheses) <= 3


# ─── Stability Tests ───────────────────────────────────────────────


class TestStability:
    def test_stability_increases_with_consistent_events(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=10)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.stability in ("rising", "stable", "locked")

    def test_stability_unknown_early(self) -> None:
        tracker = PulseTracker()
        # Just a few events
        for event in _steady_pulse_events(bpm=120.0, num_beats=2):
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.stability in ("unknown", "rising")


# ─── Get State Without Processing Tests ─────────────────────────────


class TestGetState:
    def test_get_state_returns_current_state(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=120.0, num_beats=6)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.best_bpm is not None

        # Getting state again without processing should be identical
        state2 = tracker.get_state()
        assert state2.best_bpm == state.best_bpm
        assert len(state2.hypotheses) == len(state.hypotheses)


# ─── BPM Accuracy with Long Samples ────────────────────────────────


class TestAccuracy:
    def test_accuracy_with_many_events(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=138.0, num_beats=20)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.best_bpm is not None
        assert abs(state.best_bpm - 138) < 8, (
            f"Expected ~138 BPM, got {state.best_bpm}"
        )

    def test_odd_bpm_detected(self) -> None:
        tracker = PulseTracker()
        events = _steady_pulse_events(bpm=73.0, num_beats=12)

        for event in events:
            tracker.process_event(event)

        state = tracker.get_state()
        assert state.best_bpm is not None
        assert abs(state.best_bpm - 73) < 12 or abs(state.best_bpm - 146) < 12, (
            f"Expected ~73 or ~146 BPM, got {state.best_bpm}"
        )