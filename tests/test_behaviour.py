"""Tests for Behaviour Engine — Stages 1, 2, and 3.

Stage 1: BAIL logic, EMA smoothing, intent tracking.
Stage 2: LISTEN / ENTER_SOFT / ENTER_FULL / MAINTAIN.
Stage 3: BUILD / REDUCE / DROP dynamic energy-response.
"""

from __future__ import annotations

import pytest

from drummer.behaviour import (
    BehaviourDecision,
    BehaviourEngine,
    BehaviourIntent,
    ConservativePocketDrummer,
    DrummerProfile,
)
from perception.models import MusicalEvent


# ============================================================================
# Mock state classes
# ============================================================================


class MockPulseState:
    """Minimal mock of PulseState for behaviour engine testing."""

    def __init__(
        self,
        confidence: float = 0.0,
        stability: str = "unknown",
    ) -> None:
        self.confidence = confidence
        self.stability = stability


class MockBarState:
    """Minimal mock of BarState for behaviour engine testing."""

    def __init__(
        self,
        confidence: float = 0.0,
        is_confident: bool = False,
    ) -> None:
        self.confidence = confidence
        self.is_confident = is_confident


# ============================================================================
# Helpers
# ============================================================================


def _prepare_engine(
    engine: BehaviourEngine,
    eval_time: float,
    *,
    first_event_time: float = 0.0,
) -> None:
    """Feed events from first_event_time to eval_time to satisfy observation window."""
    engine.evaluate(
        current_time=first_event_time,
        recent_events=[MusicalEvent(time_seconds=first_event_time, energy=0.5)],
    )
    step = 0.25
    t = first_event_time + step
    while t < eval_time:
        engine.evaluate(
            current_time=t,
            recent_events=[MusicalEvent(time_seconds=t, energy=0.5)],
        )
        t += step
    engine.evaluate(
        current_time=eval_time,
        recent_events=[MusicalEvent(time_seconds=eval_time, energy=0.5)],
    )


def _enter_and_seed_ema(
    engine: BehaviourEngine,
    entry_time: float = 2.0,
    *,
    pulse_conf: float = 0.88,
    bar_conf: float = 0.86,
) -> None:
    """Prepare engine: feed events bridging 0→entry_time, then enter."""
    _prepare_engine(engine, eval_time=entry_time)
    engine.evaluate(
        current_time=entry_time,
        recent_events=[],
        pulse_state=MockPulseState(confidence=pulse_conf, stability="stable"),
        bar_state=MockBarState(confidence=bar_conf),
    )


# ============================================================================
# Stage 1 Tests
# ============================================================================


class TestBehaviourIntent:
    """Verify the BehaviourIntent enum is complete."""

    def test_contains_all_expected_values(self) -> None:
        expected = {
            "LISTEN", "BAIL", "ENTER_SOFT", "ENTER_FULL", "MAINTAIN",
            "BUILD", "REDUCE", "FILL", "CRASH", "DROP",
        }
        actual = set(BehaviourIntent.__members__.keys())
        assert actual == expected

    def test_keeptime_not_present(self) -> None:
        assert "KEEP_TIME" not in BehaviourIntent.__members__


class TestConservativePocketDrummer:
    """Verify the conservative default profile."""

    def test_exists(self) -> None:
        assert ConservativePocketDrummer is not None

    def test_name(self) -> None:
        assert ConservativePocketDrummer.name == "Conservative Pocket Drummer"

    def test_hysteresis_margin(self) -> None:
        assert ConservativePocketDrummer.hysteresis_margin == 0.10

    def test_bail_silence_seconds(self) -> None:
        assert ConservativePocketDrummer.bail_silence_seconds == 0.50

    def test_density_inversion_threshold(self) -> None:
        assert ConservativePocketDrummer.density_inversion_threshold == 0.75

    def test_fill_probability_base(self) -> None:
        assert ConservativePocketDrummer.fill_probability_base == 0.05

    def test_energy_ema_alpha(self) -> None:
        assert ConservativePocketDrummer.energy_ema_alpha == 0.10

    def test_density_ema_alpha(self) -> None:
        assert ConservativePocketDrummer.density_ema_alpha == 0.10

    def test_bail_is_in_seconds_not_ms(self) -> None:
        assert ConservativePocketDrummer.bail_silence_seconds < 1.0


class TestBehaviourDecision:
    """Verify BehaviourDecision dataclass stores the required fields."""

    def test_stores_intent(self) -> None:
        d = BehaviourDecision(
            intent=BehaviourIntent.LISTEN,
            confidence=0.5,
            reason="test",
            scores={"a": 1.0},
            evaluated_at=1.5,
        )
        assert d.intent == BehaviourIntent.LISTEN

    def test_stores_confidence(self) -> None:
        d = BehaviourDecision(
            intent=BehaviourIntent.LISTEN,
            confidence=0.75,
            reason="test",
            scores={},
            evaluated_at=0.0,
        )
        assert d.confidence == 0.75

    def test_stores_reason(self) -> None:
        d = BehaviourDecision(
            intent=BehaviourIntent.BAIL,
            confidence=1.0,
            reason="Silence exceeded bail threshold",
            scores={},
            evaluated_at=3.0,
        )
        assert "Silence" in d.reason

    def test_stores_scores(self) -> None:
        d = BehaviourDecision(
            intent=BehaviourIntent.BAIL,
            confidence=1.0,
            reason="test",
            scores={"silence_duration": 2.0, "bail_silence_seconds": 0.5},
            evaluated_at=5.0,
        )
        assert d.scores["silence_duration"] == 2.0

    def test_stores_evaluated_at(self) -> None:
        d = BehaviourDecision(
            intent=BehaviourIntent.LISTEN,
            confidence=0.0,
            reason="test",
            scores={},
            evaluated_at=42.0,
        )
        assert d.evaluated_at == 42.0


class TestEngineInstantiation:
    """Verify engine construction."""

    def test_default_construction(self) -> None:
        engine = BehaviourEngine()
        assert engine.profile is ConservativePocketDrummer
        assert engine.previous_intent == BehaviourIntent.LISTEN
        assert engine.smoothed_energy is None
        assert engine.smoothed_density is None
        assert engine.last_event_time is None
        assert engine.has_seen_event is False

    def test_custom_profile_construction(self) -> None:
        custom = DrummerProfile(
            name="Test Drummer",
            hysteresis_margin=0.20,
            bail_silence_seconds=1.0,
            density_inversion_threshold=0.60,
            fill_probability_base=0.10,
            energy_ema_alpha=0.20,
            density_ema_alpha=0.20,
        )
        engine = BehaviourEngine(profile=custom)
        assert engine.profile is custom
        assert engine.profile.bail_silence_seconds == 1.0


class TestNoBailBeforeEvents:
    """BAIL is an emergency silence override — only possible after events."""

    def test_fresh_engine_with_no_events_does_not_bail(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(current_time=10.0, recent_events=[])
        assert decision.intent != BehaviourIntent.BAIL
        assert decision.confidence == 0.0
        assert decision.reason == "Stage 1 fallback state"

    def test_fresh_engine_returns_listen(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(current_time=100.0, recent_events=[])
        assert decision.intent == BehaviourIntent.LISTEN

    def test_no_bail_even_with_large_silence_when_no_events_seen(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(current_time=999.0, recent_events=[])
        assert decision.intent != BehaviourIntent.BAIL


class TestEventTimeTracking:
    """Verify engine tracks event timing."""

    def test_last_event_time_updated(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, strength=0.5, energy=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        assert engine.last_event_time == 1.0

    def test_last_event_time_tracks_latest(self) -> None:
        engine = BehaviourEngine()
        events = [
            MusicalEvent(time_seconds=0.5, strength=0.3, energy=0.3),
            MusicalEvent(time_seconds=2.0, strength=0.7, energy=0.7),
        ]
        engine.evaluate(current_time=2.0, recent_events=events)
        assert engine.last_event_time == 2.0

    def test_has_seen_event_set_to_true(self) -> None:
        engine = BehaviourEngine()
        assert engine.has_seen_event is False
        events = [MusicalEvent(time_seconds=0.5, strength=0.3, energy=0.3)]
        engine.evaluate(current_time=0.5, recent_events=events)
        assert engine.has_seen_event is True


class TestSmoothedEnergy:
    """Verify EMA energy smoothing works."""

    def test_smoothed_energy_initialised(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=0.5, energy=0.8)]
        engine.evaluate(current_time=0.5, recent_events=events)
        assert engine.smoothed_energy is not None
        assert engine.smoothed_energy > 0.0

    def test_smoothed_energy_ema(self) -> None:
        profile = DrummerProfile(
            name="test",
            hysteresis_margin=0.10,
            bail_silence_seconds=0.50,
            density_inversion_threshold=0.75,
            fill_probability_base=0.05,
            energy_ema_alpha=0.10,
            density_ema_alpha=0.10,
        )
        engine = BehaviourEngine(profile=profile)
        engine.evaluate(
            current_time=1.0,
            recent_events=[MusicalEvent(time_seconds=1.0, energy=1.0)],
        )
        assert engine.smoothed_energy == pytest.approx(1.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[MusicalEvent(time_seconds=2.0, energy=0.0)],
        )
        assert engine.smoothed_energy == pytest.approx(0.90)

    def test_smoothed_energy_unchanged_with_no_events(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, energy=0.6)]
        engine.evaluate(current_time=1.0, recent_events=events)
        before = engine.smoothed_energy
        engine.evaluate(current_time=2.0, recent_events=[])
        assert engine.smoothed_energy == before


class TestSmoothedDensity:
    """Verify EMA density smoothing works."""

    def test_smoothed_density_initialised(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=0.5, density=0.7)]
        engine.evaluate(current_time=0.5, recent_events=events)
        assert engine.smoothed_density is not None

    def test_smoothed_density_ema(self) -> None:
        profile = DrummerProfile(
            name="test",
            hysteresis_margin=0.10,
            bail_silence_seconds=0.50,
            density_inversion_threshold=0.75,
            fill_probability_base=0.05,
            energy_ema_alpha=0.10,
            density_ema_alpha=0.10,
        )
        engine = BehaviourEngine(profile=profile)
        engine.evaluate(
            current_time=1.0,
            recent_events=[MusicalEvent(time_seconds=1.0, density=0.6)],
        )
        assert engine.smoothed_density == pytest.approx(0.6)
        engine.evaluate(
            current_time=2.0,
            recent_events=[MusicalEvent(time_seconds=2.0, density=1.0)],
        )
        assert engine.smoothed_density == pytest.approx(0.64)

    def test_smoothed_density_unchanged_with_no_events(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, density=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        before = engine.smoothed_density
        engine.evaluate(current_time=2.0, recent_events=[])
        assert engine.smoothed_density == before

    def test_density_falls_back_to_zero_safely(self) -> None:
        engine = BehaviourEngine()
        evt = MusicalEvent(time_seconds=1.0, energy=0.5)
        engine.evaluate(current_time=1.0, recent_events=[evt])
        assert engine.smoothed_density is not None


class TestBailAfterEvents:
    """BAIL should trigger when silence exceeds bail_silence_seconds."""

    def test_bail_triggers_after_silence(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, energy=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        decision = engine.evaluate(current_time=1.6, recent_events=[])
        assert decision.intent == BehaviourIntent.BAIL
        assert decision.confidence == 1.0
        assert "Silence" in decision.reason
        assert "silence_duration" in decision.scores
        assert decision.scores["silence_duration"] == pytest.approx(0.60)

    def test_bail_does_not_trigger_before_threshold(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, energy=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        decision = engine.evaluate(current_time=1.4, recent_events=[])
        assert decision.intent != BehaviourIntent.BAIL

    def test_bail_updates_previous_intent(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, energy=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        decision = engine.evaluate(current_time=2.0, recent_events=[])
        assert decision.intent == BehaviourIntent.BAIL
        assert engine.previous_intent == BehaviourIntent.BAIL

    def test_bail_with_custom_profile(self) -> None:
        profile = DrummerProfile(
            name="test",
            hysteresis_margin=0.10,
            bail_silence_seconds=2.0,
            density_inversion_threshold=0.75,
            fill_probability_base=0.05,
            energy_ema_alpha=0.10,
            density_ema_alpha=0.10,
        )
        engine = BehaviourEngine(profile=profile)
        events = [MusicalEvent(time_seconds=1.0, energy=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        decision = engine.evaluate(current_time=2.5, recent_events=[])
        assert decision.intent != BehaviourIntent.BAIL
        decision = engine.evaluate(current_time=4.0, recent_events=[])
        assert decision.intent == BehaviourIntent.BAIL


class TestDefaultListen:
    """Default fallback intent is LISTEN."""

    def test_new_engine_returns_listen(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(current_time=0.0, recent_events=[])
        assert decision.intent == BehaviourIntent.LISTEN

    def test_returns_listen_after_events_without_bailing(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=1.0, energy=0.5)]
        engine.evaluate(current_time=1.0, recent_events=events)
        decision = engine.evaluate(current_time=1.2, recent_events=[])
        assert decision.intent == BehaviourIntent.LISTEN


class TestPreviousIntent:
    """Engine should track and remember previous_intent across calls."""

    def test_previous_intent_starts_as_listen(self) -> None:
        engine = BehaviourEngine()
        assert engine.previous_intent == BehaviourIntent.LISTEN

    def test_previous_intent_updated_after_decision(self) -> None:
        engine = BehaviourEngine()
        engine.evaluate(current_time=0.0, recent_events=[])
        assert engine.previous_intent == BehaviourIntent.LISTEN
        events = [MusicalEvent(time_seconds=0.5, energy=0.5)]
        engine.evaluate(current_time=0.5, recent_events=events)
        engine.evaluate(current_time=1.5, recent_events=[])
        assert engine.previous_intent == BehaviourIntent.BAIL


class TestFallbackConfidence:
    """Stage 1 fallback decisions should have zero confidence."""

    def test_listen_fallback_has_zero_confidence(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(current_time=0.0, recent_events=[])
        assert decision.intent == BehaviourIntent.LISTEN
        assert decision.confidence == 0.0
        assert decision.reason == "Stage 1 fallback state"

    def test_maintain_fallback_has_zero_confidence(self) -> None:
        engine = BehaviourEngine()
        engine.previous_intent = BehaviourIntent.MAINTAIN
        decision = engine.evaluate(current_time=0.0, recent_events=[])
        assert decision.intent == BehaviourIntent.MAINTAIN
        assert decision.confidence == 0.0
        assert decision.reason == "Stage 1 fallback state"

    def test_bail_has_full_confidence(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=0.5, energy=0.5)]
        engine.evaluate(current_time=0.5, recent_events=events)
        decision = engine.evaluate(current_time=1.5, recent_events=[])
        assert decision.intent == BehaviourIntent.BAIL
        assert decision.confidence == 1.0


# ============================================================================
# Stage 2 Tests
# ============================================================================


class TestStage2ListenNoPulse:
    """Stage 2: missing pulse state should cause LISTEN."""

    def test_no_pulse_state_causes_listen(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=None,
            bar_state=MockBarState(confidence=0.8),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "no pulse state" in decision.reason.lower()

    def test_listen_reason_includes_scores(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=None,
            bar_state=MockBarState(confidence=0.8),
        )
        assert "pulse_confidence" in decision.scores
        assert decision.scores["pulse_confidence"] == 0.0

    def test_no_pulse_no_bar_causes_fallback(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=None,
            bar_state=None,
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert decision.confidence == 0.0
        assert decision.reason == "Stage 1 fallback state"


class TestStage2ListenNoBar:
    """Stage 2: missing bar state should cause LISTEN."""

    def test_no_bar_state_causes_listen(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.8, stability="stable"),
            bar_state=None,
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "no bar state" in decision.reason.lower()

    def test_missing_bar_confidence_in_scores(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.8),
            bar_state=MockBarState(confidence=0.0),
        )
        assert decision.scores["bar_confidence"] == 0.0


class TestStage2LowPulseConfidence:
    """Stage 2: pulse confidence below threshold should cause LISTEN."""

    def test_low_pulse_confidence_causes_listen(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.50, stability="stable"),
            bar_state=MockBarState(confidence=0.85),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "pulse confidence below threshold" in decision.reason.lower()

    def test_pulse_confidence_just_below_threshold_listens(self) -> None:
        just_below = ConservativePocketDrummer.min_pulse_confidence - 0.01
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=just_below, stability="stable"),
            bar_state=MockBarState(confidence=0.85),
        )
        assert decision.intent == BehaviourIntent.LISTEN


class TestStage2LowBarConfidence:
    """Stage 2: bar confidence below threshold should cause LISTEN."""

    def test_low_bar_confidence_causes_listen(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.85, stability="stable"),
            bar_state=MockBarState(confidence=0.40),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "bar confidence below threshold" in decision.reason.lower()

    def test_bar_confidence_just_below_threshold_listens(self) -> None:
        just_below = ConservativePocketDrummer.min_bar_confidence - 0.01
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.85, stability="stable"),
            bar_state=MockBarState(confidence=just_below),
        )
        assert decision.intent == BehaviourIntent.LISTEN


class TestStage2ObservationWindow:
    """Stage 2: insufficient observation time should block entry."""

    def test_insufficient_observation_causes_listen(self) -> None:
        engine = BehaviourEngine()
        engine.evaluate(
            current_time=0.0,
            recent_events=[MusicalEvent(time_seconds=0.0, energy=0.5)],
        )
        engine.evaluate(
            current_time=0.5,
            recent_events=[MusicalEvent(time_seconds=0.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=0.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.85, stability="stable"),
            bar_state=MockBarState(confidence=0.80),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "observation window incomplete" in decision.reason.lower()

    def test_observation_readiness_partial_in_scores(self) -> None:
        engine = BehaviourEngine()
        engine.evaluate(
            current_time=0.0,
            recent_events=[MusicalEvent(time_seconds=0.0, energy=0.5)],
        )
        engine.evaluate(
            current_time=0.75,
            recent_events=[MusicalEvent(time_seconds=0.75, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=0.75,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.85, stability="stable"),
            bar_state=MockBarState(confidence=0.80),
        )
        assert "observation_readiness" in decision.scores
        assert decision.scores["observation_readiness"] == pytest.approx(0.5)

    def test_no_events_no_observation_readiness(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(
            current_time=100.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.85, stability="stable"),
            bar_state=MockBarState(confidence=0.80),
        )
        assert decision.intent == BehaviourIntent.LISTEN


class TestStage2EnterSoft:
    """Stage 2: meet entry thresholds → ENTER_SOFT."""

    def test_entry_score_below_soft_threshold_listens(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.80, stability="stable"),
            bar_state=MockBarState(confidence=0.78),
        )
        assert decision.intent == BehaviourIntent.LISTEN

    def test_entry_soft_with_qualifying_score(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert decision.intent == BehaviourIntent.ENTER_SOFT
        assert "Enter soft" in decision.reason
        assert "entry_score" in decision.scores
        assert decision.confidence == pytest.approx(0.7568, abs=0.001)

    def test_enter_soft_sets_has_entered(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert engine.has_entered is True
        assert engine.entered_at == 2.0

    def test_enter_soft_updates_previous_intent(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert engine.previous_intent == BehaviourIntent.ENTER_SOFT


class TestStage2EnterFull:
    """Stage 2: very high confidence should trigger ENTER_FULL."""

    def test_very_high_confidence_enters_full(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.95, stability="stable"),
            bar_state=MockBarState(confidence=0.92),
        )
        assert decision.intent == BehaviourIntent.ENTER_FULL
        assert "Enter full" in decision.reason
        assert decision.confidence == pytest.approx(0.874, abs=0.001)

    def test_enter_full_sets_has_entered(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.95, stability="stable"),
            bar_state=MockBarState(confidence=0.92),
        )
        assert engine.has_entered is True
        assert engine.entered_at == 2.0

    def test_enter_full_updates_previous_intent(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.95, stability="stable"),
            bar_state=MockBarState(confidence=0.92),
        )
        assert engine.previous_intent == BehaviourIntent.ENTER_FULL

    def test_rising_stability_lowers_entry_score(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.95, stability="rising"),
            bar_state=MockBarState(confidence=0.92),
        )
        assert decision.intent == BehaviourIntent.LISTEN

    def test_bar_is_confident_provides_stability(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.95, stability="rising"),
            bar_state=MockBarState(confidence=0.92, is_confident=True),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert decision.scores["entry_score"] == pytest.approx(0.7429, abs=0.001)


class TestStage2MaintainStable:
    """Stage 2: after entering, stable confidence should MAINTAIN."""

    def test_after_entry_stable_confidence_maintains(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert engine.has_entered is True
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.87, stability="stable"),
            bar_state=MockBarState(confidence=0.85),
        )
        assert decision.intent == BehaviourIntent.MAINTAIN
        assert "Maintain" in decision.reason

    def test_maintain_includes_thresholds_in_scores(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.87, stability="stable"),
            bar_state=MockBarState(confidence=0.85),
        )
        # Stage 3 MAINTAIN includes pulse/bar confidence and energy trend
        assert "pulse_confidence" in decision.scores
        assert "bar_confidence" in decision.scores
        assert "energy_trend" in decision.scores


class TestStage2MaintainHysteresis:
    """Stage 2: small confidence dips should still MAINTAIN."""

    def test_small_pulse_dip_maintains(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.70, stability="stable"),
            bar_state=MockBarState(confidence=0.84),
        )
        assert decision.intent == BehaviourIntent.MAINTAIN

    def test_small_bar_dip_maintains(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.85, stability="stable"),
            bar_state=MockBarState(confidence=0.65),
        )
        assert decision.intent == BehaviourIntent.MAINTAIN

    def test_both_marginally_below_entry_still_maintains(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.66, stability="falling"),
            bar_state=MockBarState(confidence=0.62),
        )
        assert decision.intent == BehaviourIntent.MAINTAIN


class TestStage2SevereCollapse:
    """Stage 2: severe confidence collapse causes LISTEN temporarily."""

    def test_severe_pulse_collapse_causes_listen(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.20, stability="falling"),
            bar_state=MockBarState(confidence=0.70),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "confidence collapsed" in decision.reason.lower()

    def test_severe_bar_collapse_causes_listen(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.70),
            bar_state=MockBarState(confidence=0.10),
        )
        assert decision.intent == BehaviourIntent.LISTEN
        assert "confidence collapsed" in decision.reason.lower()

    def test_severe_collapse_includes_severe_threshold_in_scores(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.20, stability="falling"),
            bar_state=MockBarState(confidence=0.70),
        )
        assert "severe_uncertainty_threshold" in decision.scores

    def test_confidence_just_above_severe_reduces(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        engine.evaluate(
            current_time=2.5,
            recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)],
        )
        decision = engine.evaluate(
            current_time=2.5,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.36, stability="falling"),
            bar_state=MockBarState(confidence=0.70),
        )
        assert decision.intent == BehaviourIntent.REDUCE


class TestStage2BailOverride:
    """Stage 2: BAIL must still take priority over musical decisions."""

    def test_bail_overrides_even_with_good_pulse_bar(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        decision = engine.evaluate(
            current_time=2.6,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.90, stability="stable"),
            bar_state=MockBarState(confidence=0.88),
        )
        assert decision.intent == BehaviourIntent.BAIL
        assert decision.confidence == 1.0

    def test_bail_overrides_even_when_entered(self) -> None:
        engine = BehaviourEngine()
        events = [MusicalEvent(time_seconds=5.0, energy=0.5)]
        engine.evaluate(current_time=5.0, recent_events=events)
        engine.has_entered = True
        engine.entered_at = 5.0
        decision = engine.evaluate(
            current_time=6.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert decision.intent == BehaviourIntent.BAIL


class TestStage2DecisionScores:
    """Stage 2: decisions must include appropriate scores."""

    def test_listen_scores_include_all_fields(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.50, stability="rising"),
            bar_state=MockBarState(confidence=0.80),
        )
        assert "pulse_confidence" in decision.scores
        assert "bar_confidence" in decision.scores
        assert "stability" in decision.scores
        assert "observation_readiness" in decision.scores
        assert "entry_score" in decision.scores

    def test_enter_soft_scores(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert "entry_score" in decision.scores
        assert decision.scores["entry_score"] == pytest.approx(0.7568, abs=0.001)

    def test_enter_full_scores(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.95, stability="stable"),
            bar_state=MockBarState(confidence=0.92),
        )
        assert decision.confidence == pytest.approx(0.874, abs=0.001)
        assert "entry_score" in decision.scores


class TestStage2FallbackConfidence:
    """Stage 2: fallback confidence 0.0 only when no real logic applies."""

    def test_stage1_fallback_has_zero_confidence(self) -> None:
        engine = BehaviourEngine()
        decision = engine.evaluate(current_time=0.0, recent_events=[])
        assert decision.confidence == 0.0
        assert decision.reason == "Stage 1 fallback state"

    def test_real_listen_has_nonzero_confidence(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.60, stability="stable"),
            bar_state=MockBarState(confidence=0.80),
        )
        assert decision.confidence > 0.0

    def test_enter_soft_has_nonzero_confidence(self) -> None:
        engine = BehaviourEngine()
        _prepare_engine(engine, eval_time=2.0)
        decision = engine.evaluate(
            current_time=2.0,
            recent_events=[],
            pulse_state=MockPulseState(confidence=0.88, stability="stable"),
            bar_state=MockBarState(confidence=0.86),
        )
        assert decision.intent == BehaviourIntent.ENTER_SOFT
        assert decision.confidence > 0.0


class TestStage2ProfileDefaults:
    """Stage 2: verify conservative default thresholds."""

    def test_min_pulse_confidence_default(self) -> None:
        assert ConservativePocketDrummer.min_pulse_confidence == 0.75

    def test_min_bar_confidence_default(self) -> None:
        assert ConservativePocketDrummer.min_bar_confidence == 0.70

    def test_full_entry_confidence_default(self) -> None:
        assert ConservativePocketDrummer.full_entry_confidence == 0.85

    def test_soft_entry_confidence_default(self) -> None:
        assert ConservativePocketDrummer.soft_entry_confidence == 0.75

    def test_min_observation_seconds_default(self) -> None:
        assert ConservativePocketDrummer.min_observation_seconds == 1.50

    def test_severe_uncertainty_threshold_default(self) -> None:
        assert ConservativePocketDrummer.severe_uncertainty_threshold == 0.35

    def test_maintain_hysteresis_margin_default(self) -> None:
        assert ConservativePocketDrummer.maintain_hysteresis_margin == 0.10

    def test_profile_is_frozen(self) -> None:
        with pytest.raises(Exception):
            ConservativePocketDrummer.min_pulse_confidence = 0.50  # type: ignore[misc]


# ============================================================================
# Stage 3 Tests — BUILD / REDUCE / DROP (appended cleanly)
# ============================================================================


class TestStage3DualEma:
    def test_dual_ema_initialised_on_first_energy(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.0, recent_events=[MusicalEvent(time_seconds=0.0, energy=0.60)])
        assert e.fast_energy_ema == pytest.approx(0.60)
        assert e.slow_energy_ema == pytest.approx(0.60)

    def test_fast_ema_responds_faster_than_slow(self) -> None:
        e = BehaviourEngine()
        for _ in range(3):
            e.evaluate(current_time=0.0, recent_events=[MusicalEvent(time_seconds=0.0, energy=0.50)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        t = 0.25
        for _ in range(5):
            e.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.90)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
            t += 0.25
        trend = e.fast_energy_ema - e.slow_energy_ema  # type: ignore[operator]
        assert trend > 0.05

    def test_energy_trend_positive_after_sustained_rise(self) -> None:
        e = BehaviourEngine()
        _enter_and_seed_ema(e, entry_time=2.0)
        t = 2.25
        for _ in range(8):
            e.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.85)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
            t += 0.25
        assert e._energy_trend() > 0.05

    def test_energy_trend_negative_after_sustained_fall(self) -> None:
        e = BehaviourEngine()
        _enter_and_seed_ema(e, entry_time=2.0)
        t = 2.25
        for _ in range(8):
            e.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.30)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
            t += 0.25
        assert e._energy_trend() < -0.05

    def test_energy_trend_zero_when_no_energy(self) -> None:
        assert BehaviourEngine()._energy_trend() == 0.0


class TestStage3Build:
    def test_build_triggers_on_sustained_rise(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.70)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.BUILD

    def test_build_blocked_before_entry(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.70)], pulse_state=MockPulseState(confidence=0.90, stability="stable"), bar_state=MockBarState(confidence=0.85))
        assert d.intent != BehaviourIntent.BUILD

    def test_build_blocked_when_density_too_high(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        e.smoothed_density = 0.85
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.70, density=0.85)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent != BehaviourIntent.BUILD

    def test_build_blocked_when_pulse_confidence_weak(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.70)], pulse_state=MockPulseState(confidence=0.60, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent != BehaviourIntent.BUILD

    def test_build_blocked_when_bar_confidence_weak(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.70)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.50))
        assert d.intent != BehaviourIntent.BUILD

    def test_one_loud_event_does_not_trigger_build(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.50; e.slow_energy_ema = 0.48
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.95)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent != BehaviourIntent.BUILD

    def test_build_scores_include_energy_fields(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.70)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.BUILD
        assert "fast_energy_ema" in d.scores
        assert "energy_trend" in d.scores


class TestStage3Reduce:
    def test_reduce_on_confidence_dip(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.50)], pulse_state=MockPulseState(confidence=0.64, stability="falling"), bar_state=MockBarState(confidence=0.72))
        assert d.intent == BehaviourIntent.REDUCE

    def test_reduce_scores_include_energy_fields(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.50)], pulse_state=MockPulseState(confidence=0.64, stability="falling"), bar_state=MockBarState(confidence=0.72))
        assert d.intent == BehaviourIntent.REDUCE
        assert "fast_energy_ema" in d.scores
        assert "energy_trend" in d.scores


class TestStage3Drop:
    def test_drop_triggers_on_severe_collapse(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.10; e.slow_energy_ema = 0.50; e.smoothed_energy = 0.10
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.DROP

    def test_drop_does_not_trigger_from_mild_negative_trend(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.50; e.slow_energy_ema = 0.70
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.50)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent != BehaviourIntent.DROP

    def test_drop_blocked_before_entry(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.fast_energy_ema = 0.10; e.slow_energy_ema = 0.50
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent != BehaviourIntent.DROP

    def test_drop_scores_include_energy_thresholds(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.10; e.slow_energy_ema = 0.50
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.DROP
        assert "drop_trend_threshold" in d.scores
        assert "low_energy_threshold_for_drop" in d.scores


class TestStage3BailOverride:
    def test_bail_overrides_build(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        t = 2.25
        for _ in range(10):
            e.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.85)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
            t += 0.25
        d = e.evaluate(current_time=2.25 + 10 * 0.25 + 0.6, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.BAIL

    def test_bail_overrides_reduce(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.evaluate(current_time=2.25, recent_events=[MusicalEvent(time_seconds=2.25, energy=0.40)], pulse_state=MockPulseState(confidence=0.64, stability="falling"), bar_state=MockBarState(confidence=0.72))
        d = e.evaluate(current_time=2.25 + 0.6, recent_events=[], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.BAIL

    def test_bail_overrides_drop(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.10; e.slow_energy_ema = 0.50
        d = e.evaluate(current_time=2.6, recent_events=[], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.BAIL


class TestStage3Cooldowns:
    def test_min_build_duration_prevents_immediate_switch(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.has_entered = True
        e.previous_intent = BehaviourIntent.BUILD
        e.last_intent_change_time = 2.5
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        d = e.evaluate(current_time=3.0, recent_events=[MusicalEvent(time_seconds=3.0, energy=0.70)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.BUILD
        assert "Cooldown" in d.reason

    def test_min_reduce_duration_prevents_immediate_switch(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.50)], pulse_state=MockPulseState(confidence=0.64, stability="falling"), bar_state=MockBarState(confidence=0.72))
        assert d.intent == BehaviourIntent.REDUCE
        e.evaluate(current_time=2.6, recent_events=[MusicalEvent(time_seconds=2.6, energy=0.50)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        d2 = e.evaluate(current_time=2.6, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d2.intent == BehaviourIntent.REDUCE
        assert "Cooldown" in d2.reason

    def test_min_drop_duration_prevents_immediate_switch(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.10; e.slow_energy_ema = 0.50
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.DROP
        e.evaluate(current_time=2.6, recent_events=[MusicalEvent(time_seconds=2.6, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        d2 = e.evaluate(current_time=2.6, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d2.intent == BehaviourIntent.DROP
        assert "Cooldown" in d2.reason

    def test_cooldown_does_not_block_bail(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        t = 2.25
        for _ in range(15):
            e.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.90)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
            t += 0.25
        t_build = 2.25 + 15 * 0.25
        e.evaluate(current_time=t_build, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        d = e.evaluate(current_time=t_build + 0.6, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.BAIL

    def test_maintain_has_no_cooldown(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.55)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        t = 2.75
        for _ in range(10):
            e.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.90)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
            t += 0.25
        d = e.evaluate(current_time=2.75 + 10 * 0.25, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent != BehaviourIntent.MAINTAIN


class TestStage3PostEntryMaintain:
    def test_after_enter_soft_default_to_maintain(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.55)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.MAINTAIN


class TestStage3Regression:
    def test_stage1_fallback_still_works(self) -> None:
        d = BehaviourEngine().evaluate(current_time=0.0, recent_events=[])
        assert d.intent == BehaviourIntent.LISTEN and d.confidence == 0.0

    def test_stage2_entry_still_works(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        assert d.intent == BehaviourIntent.ENTER_SOFT

    def test_stage2_listen_still_works(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.50, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.LISTEN

    def test_bail_still_works(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0, energy=0.5)])
        assert e.evaluate(current_time=1.6, recent_events=[]).intent == BehaviourIntent.BAIL


class TestStage3ProfileDefaults:
    def test_fast_energy_ema_alpha(self) -> None:
        assert ConservativePocketDrummer.fast_energy_ema_alpha == 0.15

    def test_slow_energy_ema_alpha(self) -> None:
        assert ConservativePocketDrummer.slow_energy_ema_alpha == 0.02

    def test_build_trend_threshold(self) -> None:
        assert ConservativePocketDrummer.build_trend_threshold == 0.15

    def test_reduce_trend_threshold(self) -> None:
        assert ConservativePocketDrummer.reduce_trend_threshold == -0.10

    def test_drop_trend_threshold(self) -> None:
        assert ConservativePocketDrummer.drop_trend_threshold == -0.30

    def test_max_density_for_build(self) -> None:
        assert ConservativePocketDrummer.max_density_for_build == 0.80

    def test_low_energy_threshold_for_drop(self) -> None:
        assert ConservativePocketDrummer.low_energy_threshold_for_drop == 0.25

    def test_density_collapse_ratio_for_drop(self) -> None:
        assert ConservativePocketDrummer.density_collapse_ratio_for_drop == 0.35

    def test_min_build_duration_seconds(self) -> None:
        assert ConservativePocketDrummer.min_build_duration_seconds == 2.0

    def test_min_reduce_duration_seconds(self) -> None:
        assert ConservativePocketDrummer.min_reduce_duration_seconds == 2.0

    def test_min_drop_duration_seconds(self) -> None:
        assert ConservativePocketDrummer.min_drop_duration_seconds == 1.0