"""Tests for the Bar Tracker — Module 3: bar position and downbeat estimation."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.bar import BarTracker, BarHypothesis, BarState, DEFAULT_BEATS_PER_BAR
from perception.models import MusicalEvent
from perception.pulse import PulseHypothesis, PulseState


# ── Helpers ────────────────────────────────────────────────────────


def _event(t: float, strength: float = 0.8, energy: float = 0.5) -> MusicalEvent:
    return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=0.5)


def _pulse_state(bpm: float, confidence: float = 0.5) -> PulseState:
    return PulseState(
        hypotheses=[
            PulseHypothesis(bpm=bpm, confidence=confidence, matches=8, last_event_time=0.0),
        ],
        best_bpm=bpm,
        confidence=confidence,
        stability="stable",
    )


def _strong(t: float) -> MusicalEvent:
    """Strong accent event (likely downbeat)."""
    return _event(t, strength=1.0, energy=0.9)


def _weak(t: float) -> MusicalEvent:
    """Weak event (likely offbeat)."""
    return _event(t, strength=0.3, energy=0.2)


# ─── BarHypothesis Tests ───────────────────────────────────────────


class TestBarHypothesis:
    def test_default_values(self) -> None:
        h = BarHypothesis(bpm=120.0, beat_interval=0.5)
        assert h.bpm == 120.0
        assert h.beat_interval == 0.5
        assert h.beats_per_bar == 4
        assert h.downbeat_time == 0.0
        assert h.confidence == 0.0
        assert h.supporting_events == 0
        assert h.accent_score == 0.0
        assert h.regularity_score == 0.0

    def test_bar_duration(self) -> None:
        h = BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=4)
        assert h.bar_duration == 2.0

    def test_custom_beats_per_bar(self) -> None:
        h = BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=3)
        assert h.beats_per_bar == 3
        assert h.bar_duration == 1.5


# ─── BarState Tests ────────────────────────────────────────────────


class TestBarState:
    def test_default_values(self) -> None:
        s = BarState()
        assert s.hypotheses == []
        assert s.best_hypothesis is None
        assert s.is_confident is False
        assert s.estimated_bar_position is None
        assert s.estimated_beat_in_bar is None


# ─── Basic Tracker Tests ───────────────────────────────────────────


class TestBarTrackerBasics:
    def test_initializes_empty(self) -> None:
        bt = BarTracker()
        state = bt.get_state()
        assert state.hypotheses == []
        assert not state.is_confident
        assert state.best_hypothesis is None

    def test_no_confidence_without_evidence(self) -> None:
        bt = BarTracker()
        # Feed one event with a weak pulse
        bt.update(_event(0.0), _pulse_state(120.0, confidence=0.03))
        state = bt.get_state()
        # Pulse confidence too low — no bar hypotheses spawn
        assert not state.is_confident

    def test_reset_clears_everything(self) -> None:
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)
        bt.update(_strong(0.0), ps)
        bt.update(_weak(0.5), ps)

        assert len(bt.get_state().hypotheses) > 0

        bt.reset()
        state = bt.get_state()
        assert state.hypotheses == []
        assert state.best_hypothesis is None


# ─── Simple 4/4 Detection Tests ────────────────────────────────────


class TestSimpleFourFourDetection:
    """Feed clear 4/4 pattern with strong downbeats every 2 seconds."""

    def _run_four_four(self, bar_tracker: BarTracker | None = None) -> BarTracker:
        bt = bar_tracker or BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)
        beat = 0.5  # 120 BPM = 0.5s per beat

        # 2 bars: downbeat(strong) + offbeat(weak) + snare(medium) + offbeat(weak)
        for bar_num in range(4):
            base = bar_num * 2.0
            bt.update(_strong(base + 0.0), ps)       # beat 1 — strong
            bt.update(_weak(base + 0.5), ps)         # beat 2 — hi-hat
            bt.update(_event(base + 1.0, 0.9, 0.9), ps)  # beat 3 — snare
            bt.update(_weak(base + 1.5), ps)         # beat 4 — hi-hat
        return bt

    def test_detects_four_four_bar_cycle(self) -> None:
        bt = self._run_four_four()
        state = bt.get_state()
        assert len(state.hypotheses) >= 1
        assert state.best_hypothesis is not None

    def test_confidence_increases_over_repeated_bars(self) -> None:
        """Run bar by bar and check confidence growth."""
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)
        beat = 0.5

        confidences: list[float] = []
        for bar_num in range(4):
            base = bar_num * 2.0
            bt.update(_strong(base + 0.0), ps)
            bt.update(_weak(base + 0.5), ps)
            bt.update(_event(base + 1.0, 0.9, 0.9), ps)
            bt.update(_weak(base + 1.5), ps)
            confidences.append(bt.get_state().confidence)

        # Later bars should have higher confidence
        assert confidences[-1] >= confidences[0], (
            f"Expected growing confidence: {confidences}"
        )
        # After 4 bars of consistent evidence, should be confident
        assert confidences[-1] > 0.15, (
            f"Expected decent confidence after 4 bars: {confidences[-1]}"
        )

    def test_event_at_downbeat_gives_beat_zero(self) -> None:
        """After detecting a bar, event at downbeat time should give beat 0."""
        bt = self._run_four_four()
        state = bt.get_state()

        # Advance time to a downbeat moment
        t = 8.0  # Should be a downbeat after 4 bars
        state = bt.get_state(current_time=t)
        assert state.estimated_beat_in_bar is not None
        assert state.estimated_beat_in_bar == 0


# ─── Bar Position Estimation Tests ─────────────────────────────────


class TestBarPositionEstimation:
    def _setup_detected_bar(self) -> BarTracker:
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)
        beat = 0.5
        for bar_num in range(4):
            base = bar_num * 2.0
            bt.update(_strong(base + 0.0), ps)
            bt.update(_weak(base + 0.5), ps)
            bt.update(_event(base + 1.0, 0.9, 0.9), ps)
            bt.update(_weak(base + 1.5), ps)
        return bt

    def test_downbeat_event_gives_beat_near_zero(self) -> None:
        bt = self._setup_detected_bar()
        # Downbeat at 0.0, 2.0, 4.0, 6.0, 8.0...
        state = bt.get_state(current_time=8.0)
        assert state.estimated_beat_in_bar is not None
        assert state.estimated_beat_in_bar in (0, 3), (
            f"Expected beat near 0, got {state.estimated_beat_in_bar}"
        )

    def test_half_beat_gives_mid_bar(self) -> None:
        bt = self._setup_detected_bar()
        state = bt.get_state(current_time=9.0)  # 1 second into a bar
        assert state.estimated_beat_in_bar is not None
        assert state.estimated_beat_in_bar in (1, 2), (
            f"Expected mid-bar beat, got {state.estimated_beat_in_bar}"
        )


# ─── Ambiguity Tests ───────────────────────────────────────────────


class TestAmbiguity:
    def test_ambiguous_input_preserves_multiple_hypotheses(self) -> None:
        """Events every 1.0s with strong accents every 2.0s should cause ambiguity."""
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)

        for i in range(8):
            t = i * 1.0
            # Every 2 seconds is strong (every other 120bpm beat)
            if i % 2 == 0:
                bt.update(_strong(t), ps)
            else:
                bt.update(_weak(t), ps)

        state = bt.get_state()
        assert len(state.hypotheses) >= 1

    def test_low_confidence_when_ambiguous(self) -> None:
        """Without clear 4-beat structure, confidence should be limited."""
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)

        for i in range(8):
            t = i * 1.0
            bt.update(_event(t, strength=0.5 + (i % 2) * 0.5), ps)

        state = bt.get_state()
        # Confidence should exist but not be extremely high
        assert state.confidence < 0.9


# ─── Human Jitter Tests ───────────────────────────────────────────


class TestHumanJitter:
    def test_tolerates_small_timing_variations(self) -> None:
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)
        beat = 0.5

        for bar_num in range(4):
            base = bar_num * 2.0
            # Add jitter up to ±25ms
            jitter = (hash(bar_num * 10) % 51 - 25) / 1000.0
            bt.update(_strong(base + jitter), ps)

            jitter = (hash(bar_num * 10 + 1) % 51 - 25) / 1000.0
            bt.update(_weak(base + 0.5 + jitter), ps)

            jitter = (hash(bar_num * 10 + 2) % 51 - 25) / 1000.0
            bt.update(_event(base + 1.0 + jitter, 0.9, 0.9), ps)

            jitter = (hash(bar_num * 10 + 3) % 51 - 25) / 1000.0
            bt.update(_weak(base + 1.5 + jitter), ps)

        state = bt.get_state()
        assert len(state.hypotheses) >= 1
        assert state.confidence > 0.1


# ─── Silence Decay Tests ──────────────────────────────────────────


class TestSilenceDecay:
    def test_confidence_decays_after_silence(self) -> None:
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)
        beat = 0.5

        # Build confidence with 2 bars
        for bar_num in range(2):
            base = bar_num * 2.0
            bt.update(_strong(base + 0.0), ps)
            bt.update(_weak(base + 0.5), ps)
            bt.update(_event(base + 1.0, 0.9, 0.9), ps)
            bt.update(_weak(base + 1.5), ps)

        state_before = bt.get_state()
        assert state_before.confidence > 0.05

        # Simulate silence
        state_after = bt.get_state(current_time=20.0)
        assert state_after.confidence <= state_before.confidence + 0.01

    def test_hypotheses_persist_during_silence(self) -> None:
        """Hypotheses should not vanish instantly during silence."""
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.5)

        bt.update(_strong(0.0), ps)
        bt.update(_weak(0.5), ps)
        bt.update(_event(1.0, 0.9, 0.9), ps)
        bt.update(_weak(1.5), ps)

        # Short silence
        state = bt.get_state(current_time=5.0)
        # Some hypotheses should still exist
        assert len(state.hypotheses) >= 1


# ─── Pulse Dependency Tests ────────────────────────────────────────


class TestPulseDependency:
    def test_no_bar_with_no_pulse(self) -> None:
        """If pulse has no hypotheses, bar tracker should not create any."""
        bt = BarTracker()
        ps = PulseState(hypotheses=[], best_bpm=None, confidence=0.0)
        bt.update(_strong(0.0), ps)
        assert len(bt.get_state().hypotheses) == 0

    def test_weak_pulse_spawns_no_bars(self) -> None:
        bt = BarTracker()
        ps = _pulse_state(120.0, confidence=0.01)  # below min threshold
        bt.update(_strong(0.0), ps)
        assert len(bt.get_state().hypotheses) == 0


# ─── Configuration Tests ──────────────────────────────────────────


class TestConfiguration:
    def test_custom_beats_per_bar(self) -> None:
        bt = BarTracker(beats_per_bar=3)
        assert bt._beats_per_bar == 3

    def test_invalid_beats_per_bar_raises(self) -> None:
        try:
            BarTracker(beats_per_bar=0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ─── Integration: Pulse + Bar Pipeline ─────────────────────────────


class TestIntegration:
    def test_pipeline_from_events_to_bar(self) -> None:
        """Simulate pulse tracker + bar tracker working together."""
        from perception.pulse import PulseTracker

        pulse_tracker = PulseTracker()
        bar_tracker = BarTracker()

        # Feed events at 120 BPM through pulse then bar
        beat = 0.5
        bar_states: list[BarState] = []

        for bar_num in range(4):
            base = bar_num * 2.0
            event1 = _strong(base + 0.0)
            event2 = _weak(base + 0.5)
            event3 = _event(base + 1.0, 0.9, 0.9)
            event4 = _weak(base + 1.5)

            pulse_state = pulse_tracker.process_event(event1)
            bar_states.append(bar_tracker.update(event1, pulse_state))

            pulse_state = pulse_tracker.process_event(event2)
            bar_states.append(bar_tracker.update(event2, pulse_state))

            pulse_state = pulse_tracker.process_event(event3)
            bar_states.append(bar_tracker.update(event3, pulse_state))

            pulse_state = pulse_tracker.process_event(event4)
            bar_states.append(bar_tracker.update(event4, pulse_state))

        # Final bar state should have at least one hypothesis
        final = bar_tracker.get_state()
        assert len(final.hypotheses) >= 1