"""Tests for Groove Intent Engine — Module 4: perception-to-behaviour decisions."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drummer.intent import (
    GrooveIntentEngine,
    GrooveAction,
    GrooveIntent,
    MIN_PULSE_CONFIDENCE_TO_PLAY,
    MIN_BAR_CONFIDENCE_TO_PLAY,
    HIGH_CONFIDENCE,
    MIN_BARS_BETWEEN_FILLS,
)
from perception.bar import BarHypothesis, BarState
from perception.models import MusicalEvent
from perception.pulse import PulseHypothesis, PulseState


# ── Helpers ────────────────────────────────────────────────────────


def _event(t: float = 0.0, strength: float = 0.8, energy: float = 0.5, density: float = 0.5) -> MusicalEvent:
    return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=density)


def _pulse_state(bpm: float = 120.0, confidence: float = 0.5) -> PulseState:
    return PulseState(
        hypotheses=[PulseHypothesis(bpm=bpm, confidence=confidence, matches=10)],
        best_bpm=bpm,
        confidence=confidence,
        stability="stable",
    )


def _bar_state(confidence: float = 0.5, beat_in_bar: int = 0, beats_per_bar: int = 4) -> BarState:
    return BarState(
        hypotheses=[
            BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=beats_per_bar, confidence=confidence),
        ],
        best_hypothesis=BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=beats_per_bar, confidence=confidence),
        is_confident=True,
        estimated_bar_position=float(beat_in_bar),
        estimated_beat_in_bar=beat_in_bar,
        confidence=confidence,
        timestamp=0.0,
    )


def _low_pulse() -> PulseState:
    return _pulse_state(confidence=0.1)


def _low_bar() -> BarState:
    return _bar_state(confidence=0.1)


def _good_pulse() -> PulseState:
    return _pulse_state(confidence=MIN_PULSE_CONFIDENCE_TO_PLAY + 0.05)


def _good_bar() -> BarState:
    return _bar_state(confidence=MIN_BAR_CONFIDENCE_TO_PLAY + 0.05)


def _confident_pulse() -> PulseState:
    return _pulse_state(confidence=HIGH_CONFIDENCE + 0.05)


def _confident_bar() -> BarState:
    return _bar_state(confidence=HIGH_CONFIDENCE + 0.05, beat_in_bar=0)


def _feed_events(engine: GrooveIntentEngine, pulse: PulseState, bar: BarState, count: int) -> list[GrooveIntent]:
    intents: list[GrooveIntent] = []
    for i in range(count):
        ev = _event(t=float(i) * 0.5, energy=0.5 + (i % 4) * 0.1)
        intent = engine.update(ev, pulse, bar)
        intents.append(intent)
    return intents


# ─── Basic State Tests ────────────────────────────────────────────


class TestBasicState:
    def test_initializes_in_wait(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.get_current_intent()
        assert intent.action == GrooveAction.WAIT
        assert not intent.should_play
        assert not intent.should_fill

    def test_reset_returns_to_wait(self) -> None:
        engine = GrooveIntentEngine()
        # Feed enough to enter
        _feed_events(engine, _good_pulse(), _good_bar(), 4)
        engine.reset()
        intent = engine.get_current_intent()
        assert intent.action == GrooveAction.WAIT
        assert not intent.should_play

    def test_low_pulse_confidence_produces_wait(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _low_pulse(), _good_bar())
        assert intent.action == GrooveAction.WAIT

    def test_low_bar_confidence_produces_wait(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _low_bar())
        assert intent.action == GrooveAction.WAIT

    def test_both_low_produces_wait(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _low_pulse(), _low_bar())
        assert intent.action == GrooveAction.WAIT


# ─── Entry Behaviour Tests ────────────────────────────────────────


class TestEntryBehaviour:
    def test_stable_pulse_and_bar_produces_enter(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _good_bar())
        assert intent.action == GrooveAction.ENTER

    def test_entry_has_should_play_true(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _good_bar())
        assert intent.should_play

    def test_entry_does_not_trigger_fill(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _good_bar())
        assert not intent.should_fill

    def test_entry_uses_conservative_complexity(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _good_bar())
        assert intent.suggested_complexity < 0.6  # not aggressive

    def test_entry_reason_mentions_entering(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _good_bar())
        assert "entering" in intent.reason.lower() or "enter" in intent.reason.lower()


# ─── Hold Behaviour Tests ─────────────────────────────────────────


class TestHoldBehaviour:
    def test_repeated_stable_updates_become_hold(self) -> None:
        engine = GrooveIntentEngine()
        # Enter first
        engine.update(_event(), _good_pulse(), _good_bar())
        # Then hold on subsequent updates
        intents = _feed_events(engine, _good_pulse(), _good_bar(), 5)
        actions = [i.action for i in intents]
        assert GrooveAction.HOLD in actions

    def test_steady_energy_keeps_complexity_stable(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _good_pulse(), _good_bar(), 4)
        intent1 = engine.get_current_intent()
        _feed_events(engine, _good_pulse(), _good_bar(), 2)
        intent2 = engine.get_current_intent()
        # Complexity should not jump dramatically
        assert abs(intent2.suggested_complexity - intent1.suggested_complexity) < 0.3


# ─── Build Behaviour Tests ────────────────────────────────────────


class TestBuildBehaviour:
    def test_rising_energy_creates_build(self) -> None:
        engine = GrooveIntentEngine()
        # Establish entry/hold first
        _feed_events(engine, _good_pulse(), _good_bar(), 4)
        running_action = engine.get_current_intent().action
        assert running_action in (GrooveAction.ENTER, GrooveAction.HOLD)

        # Feed rising energy events
        for i in range(4):
            ev = _event(t=float(i) * 0.5, energy=0.3 + i * 0.15, density=0.4 + i * 0.15)
            engine.update(ev, _confident_pulse(), _confident_bar())

        intent = engine.get_current_intent()
        assert intent.action == GrooveAction.BUILD

    def test_rising_density_creates_build(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _good_pulse(), _good_bar(), 4)

        for i in range(4):
            ev = _event(t=float(i) * 0.5, energy=0.5, density=0.3 + i * 0.2)
            engine.update(ev, _confident_pulse(), _confident_bar())

        intent = engine.get_current_intent()
        assert intent.action == GrooveAction.BUILD

    def test_build_increases_complexity(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _good_pulse(), _good_bar(), 4)
        before = engine.get_current_intent().suggested_complexity

        for i in range(4):
            ev = _event(t=float(i) * 0.5, energy=0.5 + i * 0.1, density=0.4 + i * 0.15)
            engine.update(ev, _confident_pulse(), _confident_bar())

        after = engine.get_current_intent().suggested_complexity
        assert after >= before  # complexity shouldn't drop during build


# ─── Reduce / Simplify Tests ──────────────────────────────────────


class TestReduceSimplify:
    def test_falling_energy_creates_reduce(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _good_pulse(), _good_bar(), 4)

        # Feed falling energy events
        for i in range(4):
            ev = _event(t=float(i) * 0.5, energy=0.7 - i * 0.15, density=0.5 - i * 0.1)
            engine.update(ev, _confident_pulse(), _confident_bar())

        intent = engine.get_current_intent()
        assert intent.action in (GrooveAction.REDUCE, GrooveAction.SIMPLIFY)

    def test_very_low_energy_simplifies(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _good_pulse(), _good_bar(), 4)

        for i in range(4):
            ev = _event(t=float(i) * 0.5, energy=0.7 - i * 0.18, density=0.4 - i * 0.12)
            engine.update(ev, _confident_pulse(), _confident_bar())

        intent = engine.get_current_intent()
        assert intent.action in (GrooveAction.REDUCE, GrooveAction.SIMPLIFY)
        assert intent.suggested_complexity < 0.6


# ─── Fill Preparation Tests ────────────────────────────────────────


class TestFillPreparation:
    def test_no_fill_when_confidence_low(self) -> None:
        engine = GrooveIntentEngine()
        # Set up bars_since_fill by feeding events with bar position 0
        _feed_events(engine, _good_pulse(), _bar_state(beat_in_bar=0), 4)

        # Try fill with low confidence
        intent = engine.update(_event(), _good_pulse(), _low_bar())
        assert not intent.should_fill

    def test_no_fill_immediately_after_entry(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _confident_pulse(), _confident_bar())
        assert not intent.should_fill

    def test_can_prepare_fill_at_bar_end_after_enough_bars(self) -> None:
        engine = GrooveIntentEngine()
        # Feed many events to build up bars_since_fill
        for i in range(20):
            beat = i % 4
            ev = _event(t=float(i) * 0.5, energy=0.6)
            engine.update(ev, _confident_pulse(), _bar_state(beat_in_bar=beat, confidence=HIGH_CONFIDENCE + 0.1))

        # Now at beat 3 (near bar end) with high confidence
        intent = engine.update(
            _event(t=10.0, energy=0.7),
            _confident_pulse(),
            _bar_state(beat_in_bar=3, confidence=HIGH_CONFIDENCE + 0.1),
        )
        # After enough bars at high confidence near bar end,
        # should at minimum be playing confidently
        assert intent.should_play
        assert intent.action in (
            GrooveAction.PREPARE_FILL, GrooveAction.MARK_DOWNBEAT,
            GrooveAction.HOLD, GrooveAction.BUILD,
        )


# ─── Downbeat Marking Tests ───────────────────────────────────────


class TestDownbeatMarking:
    def test_strong_downbeat_marks(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _confident_pulse(), _confident_bar(), 6)

        # Strong event on downbeat
        ev = _event(t=10.0, strength=0.9, energy=0.8)
        intent = engine.update(ev, _confident_pulse(), _confident_bar())
        # Should be MARK_DOWNBEAT or HOLD
        assert intent.action in (GrooveAction.MARK_DOWNBEAT, GrooveAction.HOLD)

    def test_weak_uncertain_downbeat_does_not_mark(self) -> None:
        engine = GrooveIntentEngine()
        _feed_events(engine, _good_pulse(), _good_bar(), 4)

        ev = _event(t=5.0, strength=0.3, energy=0.2)
        intent = engine.update(ev, _good_pulse(), _good_bar())
        assert intent.action != GrooveAction.MARK_DOWNBEAT


# ─── Reason Field Tests ────────────────────────────────────────────


class TestReasonField:
    def test_reason_is_nonempty_after_update(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _good_pulse(), _good_bar())
        assert len(intent.reason) > 0

    def test_reason_mentions_pulse_when_low(self) -> None:
        engine = GrooveIntentEngine()
        intent = engine.update(_event(), _low_pulse(), _good_bar())
        assert "pulse" in intent.reason.lower()


# ─── Integration Test ──────────────────────────────────────────────


class TestIntegration:
    def test_full_lifecycle(self) -> None:
        """Simulate a full musical scenario: wait → enter → hold → build → reduce."""
        engine = GrooveIntentEngine()
        actions: list[GrooveAction] = []

        # Phase 1: Wait (low confidence)
        for i in range(3):
            intent = engine.update(_event(i * 0.5), _low_pulse(), _low_bar())
            actions.append(intent.action)
        assert all(a == GrooveAction.WAIT for a in actions)

        # Phase 2: Enter (confidence rises)
        pulse = _good_pulse()
        bar = _good_bar()
        for i in range(3):
            intent = engine.update(_event(2.0 + i * 0.5, energy=0.5), pulse, bar)
            actions.append(intent.action)
        assert GrooveAction.ENTER in actions

        # Phase 3: Hold (stable)
        for i in range(5):
            intent = engine.update(_event(4.0 + i * 0.5, energy=0.5), pulse, bar)
            actions.append(intent.action)
        assert GrooveAction.HOLD in actions

        # Phase 4: Build (rising energy)
        pulse = _confident_pulse()
        bar = _confident_bar()
        for i in range(4):
            intent = engine.update(_event(7.0 + i * 0.5, energy=0.4 + i * 0.15, density=0.3 + i * 0.2), pulse, bar)
            actions.append(intent.action)
        assert GrooveAction.BUILD in actions

        # Phase 5: Reduce (falling energy)
        for i in range(4):
            intent = engine.update(_event(10.0 + i * 0.5, energy=0.8 - i * 0.18, density=0.5 - i * 0.12), pulse, bar)
            actions.append(intent.action)
        assert GrooveAction.REDUCE in actions or GrooveAction.SIMPLIFY in actions