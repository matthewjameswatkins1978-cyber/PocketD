"""Tests for Stage 1, Stage 2 & Stage 3 Behaviour Engine."""
from __future__ import annotations
import pytest
from drummer.behaviour import BehaviourDecision, BehaviourEngine, BehaviourIntent, ConservativePocketDrummer, DrummerProfile
from perception.models import MusicalEvent


class MockPulseState:
    def __init__(self, confidence: float = 0.0, stability: str = "unknown") -> None:
        self.confidence = confidence
        self.stability = stability


class MockBarState:
    def __init__(self, confidence: float = 0.0, is_confident: bool = False) -> None:
        self.confidence = confidence
        self.is_confident = is_confident


# ── Stage 1 ────────────────────────────────────────────────────────────────

class TestBehaviourIntent:
    def test_contains_all_expected_values(self) -> None:
        expected = {"LISTEN","BAIL","ENTER_SOFT","ENTER_FULL","MAINTAIN","BUILD","REDUCE","FILL","CRASH","DROP"}
        assert set(BehaviourIntent.__members__.keys()) == expected
    def test_keeptime_not_present(self) -> None:
        assert "KEEP_TIME" not in BehaviourIntent.__members__

class TestConservativePocketDrummer:
    def test_exists(self) -> None: assert ConservativePocketDrummer is not None
    def test_name(self) -> None: assert ConservativePocketDrummer.name == "Conservative Pocket Drummer"
    def test_hysteresis_margin(self) -> None: assert ConservativePocketDrummer.hysteresis_margin == 0.10
    def test_bail_silence_seconds(self) -> None: assert ConservativePocketDrummer.bail_silence_seconds == 0.50
    def test_bail_is_in_seconds_not_ms(self) -> None: assert ConservativePocketDrummer.bail_silence_seconds < 1.0

class TestBehaviourDecision:
    def test_stores_intent(self) -> None:
        d = BehaviourDecision(intent=BehaviourIntent.LISTEN, confidence=0.5, reason="test", scores={"a":1.0}, evaluated_at=1.5)
        assert d.intent == BehaviourIntent.LISTEN
    def test_stores_confidence(self) -> None:
        d = BehaviourDecision(intent=BehaviourIntent.LISTEN, confidence=0.75, reason="test", scores={}, evaluated_at=0.0)
        assert d.confidence == 0.75
    def test_stores_reason(self) -> None:
        d = BehaviourDecision(intent=BehaviourIntent.BAIL, confidence=1.0, reason="Silence exceeded bail threshold", scores={}, evaluated_at=3.0)
        assert "Silence" in d.reason
    def test_stores_scores(self) -> None:
        d = BehaviourDecision(intent=BehaviourIntent.BAIL, confidence=1.0, reason="test", scores={"silence_duration":2.0}, evaluated_at=5.0)
        assert d.scores["silence_duration"] == 2.0
    def test_stores_evaluated_at(self) -> None:
        d = BehaviourDecision(intent=BehaviourIntent.LISTEN, confidence=0.0, reason="test", scores={}, evaluated_at=42.0)
        assert d.evaluated_at == 42.0

class TestEngineInstantiation:
    def test_default_construction(self) -> None:
        e = BehaviourEngine()
        assert e.profile is ConservativePocketDrummer
        assert e.previous_intent == BehaviourIntent.LISTEN

class TestNoBailBeforeEvents:
    def test_fresh_engine_with_no_events_does_not_bail(self) -> None:
        d = BehaviourEngine().evaluate(current_time=10.0, recent_events=[])
        assert d.intent != BehaviourIntent.BAIL
    def test_fresh_engine_returns_listen(self) -> None:
        assert BehaviourEngine().evaluate(current_time=100.0, recent_events=[]).intent == BehaviourIntent.LISTEN

class TestEventTimeTracking:
    def test_last_event_time_updated(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0, strength=0.5, energy=0.5)])
        assert e.last_event_time == 1.0
    def test_has_seen_event_set_to_true(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.5, recent_events=[MusicalEvent(time_seconds=0.5, strength=0.3, energy=0.3)])
        assert e.has_seen_event is True

class TestSmoothedEnergy:
    def test_smoothed_energy_initialised(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.5, recent_events=[MusicalEvent(time_seconds=0.5, energy=0.8)])
        assert e.smoothed_energy is not None
    def test_smoothed_energy_ema(self) -> None:
        p = DrummerProfile(name="test",hysteresis_margin=0.10,bail_silence_seconds=0.50,density_inversion_threshold=0.75,fill_probability_base=0.05,energy_ema_alpha=0.10,density_ema_alpha=0.10)
        e = BehaviourEngine(profile=p)
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0,energy=1.0)])
        assert e.smoothed_energy == pytest.approx(1.0)
        e.evaluate(current_time=2.0, recent_events=[MusicalEvent(time_seconds=2.0,energy=0.0)])
        assert e.smoothed_energy == pytest.approx(0.90)

class TestSmoothedDensity:
    def test_smoothed_density_initialised(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.5, recent_events=[MusicalEvent(time_seconds=0.5,density=0.7)])
        assert e.smoothed_density is not None

class TestBailAfterEvents:
    def test_bail_triggers_after_silence(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0,energy=0.5)])
        d = e.evaluate(current_time=1.6, recent_events=[])
        assert d.intent == BehaviourIntent.BAIL
    def test_bail_does_not_trigger_before_threshold(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0,energy=0.5)])
        assert e.evaluate(current_time=1.4, recent_events=[]).intent != BehaviourIntent.BAIL
    def test_bail_updates_previous_intent(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0,energy=0.5)])
        e.evaluate(current_time=2.0, recent_events=[])
        assert e.previous_intent == BehaviourIntent.BAIL

class TestDefaultListen:
    def test_new_engine_returns_listen(self) -> None:
        assert BehaviourEngine().evaluate(current_time=0.0, recent_events=[]).intent == BehaviourIntent.LISTEN

class TestPreviousIntent:
    def test_previous_intent_starts_as_listen(self) -> None:
        assert BehaviourEngine().previous_intent == BehaviourIntent.LISTEN

class TestFallbackConfidence:
    def test_listen_fallback_has_zero_confidence(self) -> None:
        d = BehaviourEngine().evaluate(current_time=0.0, recent_events=[])
        assert d.confidence == 0.0
    def test_bail_has_full_confidence(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.5, recent_events=[MusicalEvent(time_seconds=0.5,energy=0.5)])
        assert e.evaluate(current_time=1.5, recent_events=[]).confidence == 1.0


# ── Helpers ─────────────────────────────────────────────────────────────────

def _prepare_engine(engine: BehaviourEngine, eval_time: float, *, first_event_time: float = 0.0) -> None:
    """Feed events from first_event_time to eval_time (step 0.25) to satisfy observation window."""
    engine.evaluate(current_time=first_event_time, recent_events=[MusicalEvent(time_seconds=first_event_time, energy=0.5)])
    step = 0.25; t = first_event_time + step
    while t < eval_time:
        engine.evaluate(current_time=t, recent_events=[MusicalEvent(time_seconds=t, energy=0.5)])
        t += step
    engine.evaluate(current_time=eval_time, recent_events=[MusicalEvent(time_seconds=eval_time, energy=0.5)])


def _enter_and_seed_ema(engine: BehaviourEngine, entry_time: float = 2.0, *, pulse_conf: float = 0.88, bar_conf: float = 0.86) -> None:
    _prepare_engine(engine, eval_time=entry_time)
    engine.evaluate(current_time=entry_time, recent_events=[], pulse_state=MockPulseState(confidence=pulse_conf, stability="stable"), bar_state=MockBarState(confidence=bar_conf))


# ═══════════════════════════════════════════════════════════════════════════
# Stage 2 Tests
# ═══════════════════════════════════════════════════════════════════════════

class TestStage2ListenNoPulse:
    def test_no_pulse_state_causes_listen(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=None, bar_state=MockBarState(confidence=0.8))
        assert d.intent == BehaviourIntent.LISTEN
        assert "no pulse state" in d.reason.lower()
    def test_no_pulse_no_bar_causes_fallback(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=None, bar_state=None)
        assert d.intent == BehaviourIntent.LISTEN and d.confidence == 0.0

class TestStage2ListenNoBar:
    def test_no_bar_state_causes_listen(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.8, stability="stable"), bar_state=None)
        assert d.intent == BehaviourIntent.LISTEN and "no bar state" in d.reason.lower()

class TestStage2LowPulseConfidence:
    def test_low_pulse_confidence_causes_listen(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.50, stability="stable"), bar_state=MockBarState(confidence=0.85))
        assert d.intent == BehaviourIntent.LISTEN

class TestStage2LowBarConfidence:
    def test_low_bar_confidence_causes_listen(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.40))
        assert d.intent == BehaviourIntent.LISTEN

class TestStage2ObservationWindow:
    def test_insufficient_observation_causes_listen(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.0, recent_events=[MusicalEvent(time_seconds=0.0, energy=0.5)])
        e.evaluate(current_time=0.5, recent_events=[MusicalEvent(time_seconds=0.5, energy=0.5)])
        d = e.evaluate(current_time=0.5, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.LISTEN

class TestStage2EnterSoft:
    def test_entry_soft_with_qualifying_score(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        assert d.intent == BehaviourIntent.ENTER_SOFT
        assert d.confidence == pytest.approx(0.7568, abs=0.001)
    def test_enter_soft_sets_has_entered(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        assert e.has_entered is True

class TestStage2EnterFull:
    def test_very_high_confidence_enters_full(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.95, stability="stable"), bar_state=MockBarState(confidence=0.92))
        assert d.intent == BehaviourIntent.ENTER_FULL

class TestStage2MaintainStable:
    def test_after_entry_stable_confidence_maintains(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.87, stability="stable"), bar_state=MockBarState(confidence=0.85))
        assert d.intent == BehaviourIntent.MAINTAIN

class TestStage2MaintainHysteresis:
    def test_small_pulse_dip_maintains(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.70, stability="stable"), bar_state=MockBarState(confidence=0.84))
        assert d.intent == BehaviourIntent.MAINTAIN

class TestStage2SevereCollapse:
    def test_severe_pulse_collapse_causes_listen(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.20, stability="falling"), bar_state=MockBarState(confidence=0.70))
        assert d.intent == BehaviourIntent.LISTEN and "confidence collapsed" in d.reason.lower()
    def test_severe_bar_collapse_causes_listen(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.70), bar_state=MockBarState(confidence=0.10))
        assert d.intent == BehaviourIntent.LISTEN and "confidence collapsed" in d.reason.lower()
    def test_confidence_just_above_severe_reduces(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.5)])
        d = e.evaluate(current_time=2.5, recent_events=[], pulse_state=MockPulseState(confidence=0.36, stability="falling"), bar_state=MockBarState(confidence=0.70))
        assert d.intent == BehaviourIntent.REDUCE

class TestStage2BailOverride:
    def test_bail_overrides_even_with_good_pulse_bar(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        d = e.evaluate(current_time=2.6, recent_events=[], pulse_state=MockPulseState(confidence=0.90, stability="stable"), bar_state=MockBarState(confidence=0.88))
        assert d.intent == BehaviourIntent.BAIL
    def test_bail_overrides_even_when_entered(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=5.0, recent_events=[MusicalEvent(time_seconds=5.0, energy=0.5)])
        e.has_entered = True
        d = e.evaluate(current_time=6.0, recent_events=[], pulse_state=MockPulseState(confidence=0.88, stability="stable"), bar_state=MockBarState(confidence=0.86))
        assert d.intent == BehaviourIntent.BAIL

class TestStage2FallbackConfidence:
    def test_stage1_fallback_has_zero_confidence(self) -> None:
        assert BehaviourEngine().evaluate(current_time=0.0, recent_events=[]).confidence == 0.0
    def test_real_listen_has_nonzero_confidence(self) -> None:
        e = BehaviourEngine(); _prepare_engine(e, eval_time=2.0)
        d = e.evaluate(current_time=2.0, recent_events=[], pulse_state=MockPulseState(confidence=0.60, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.confidence > 0.0

class TestStage2ProfileDefaults:
    def test_min_pulse_confidence_default(self) -> None: assert ConservativePocketDrummer.min_pulse_confidence == 0.75
    def test_min_bar_confidence_default(self) -> None: assert ConservativePocketDrummer.min_bar_confidence == 0.70
    def test_full_entry_confidence_default(self) -> None: assert ConservativePocketDrummer.full_entry_confidence == 0.85
    def test_soft_entry_confidence_default(self) -> None: assert ConservativePocketDrummer.soft_entry_confidence == 0.75
    def test_severe_uncertainty_threshold_default(self) -> None: assert ConservativePocketDrummer.severe_uncertainty_threshold == 0.35
    def test_profile_is_frozen(self) -> None:
        with pytest.raises(Exception):
            ConservativePocketDrummer.min_pulse_confidence = 0.50  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Stage 3 Tests
# ═══════════════════════════════════════════════════════════════════════════


class TestStage3DualEma:
    def test_dual_ema_initialised_on_first_energy(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=0.0, recent_events=[MusicalEvent(time_seconds=0.0, energy=0.60)])
        assert e.fast_energy_ema == pytest.approx(0.60)
        assert e.slow_energy_ema == pytest.approx(0.60)

    def test_fast_ema_responds_faster_than_slow(self) -> None:
        e = BehaviourEngine()
        # Feed with pulse/bar to avoid fallback
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
        # Feed rising energy
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
        e.fast_energy_ema = 0.80
        e.slow_energy_ema = 0.60
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
        # Use event with high density so smoothed_density stays > max_density_for_build (0.80)
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
        # Seed EMAs at low level, feed one loud event — trend won't reach 0.15
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
        e.fast_energy_ema = 0.10
        e.slow_energy_ema = 0.50
        e.smoothed_energy = 0.10
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.DROP

    def test_drop_does_not_trigger_from_mild_negative_trend(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.50
        e.slow_energy_ema = 0.70
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
        # Directly set state: entered, in BUILD, change_time just now
        e.has_entered = True
        e.previous_intent = BehaviourIntent.BUILD
        e.last_intent_change_time = 2.5
        e.fast_energy_ema = 0.80; e.slow_energy_ema = 0.60
        # Evaluate 0.5s later — within 2.0s min_build_duration — should cooldown
        d = e.evaluate(current_time=3.0, recent_events=[MusicalEvent(time_seconds=3.0, energy=0.70)], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d.intent == BehaviourIntent.BUILD
        assert "Cooldown" in d.reason

    def test_min_reduce_duration_prevents_immediate_switch(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.50)], pulse_state=MockPulseState(confidence=0.64, stability="falling"), bar_state=MockBarState(confidence=0.72))
        assert d.intent == BehaviourIntent.REDUCE
        # Feed bridge event WITH pulse/bar so previous_intent stays REDUCE (avoids fallback reset)
        e.evaluate(current_time=2.6, recent_events=[MusicalEvent(time_seconds=2.6, energy=0.50)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        d2 = e.evaluate(current_time=2.6, recent_events=[], pulse_state=MockPulseState(confidence=0.85, stability="stable"), bar_state=MockBarState(confidence=0.80))
        assert d2.intent == BehaviourIntent.REDUCE
        assert "Cooldown" in d2.reason

    def test_min_drop_duration_prevents_immediate_switch(self) -> None:
        e = BehaviourEngine(); _enter_and_seed_ema(e, entry_time=2.0)
        e.fast_energy_ema = 0.10; e.slow_energy_ema = 0.50
        d = e.evaluate(current_time=2.5, recent_events=[MusicalEvent(time_seconds=2.5, energy=0.10)], pulse_state=MockPulseState(confidence=0.80, stability="stable"), bar_state=MockBarState(confidence=0.75))
        assert d.intent == BehaviourIntent.DROP
        # Feed bridge event WITH pulse/bar so previous_intent stays DROP
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
    def test_bail_still_works(self) -> None:
        e = BehaviourEngine()
        e.evaluate(current_time=1.0, recent_events=[MusicalEvent(time_seconds=1.0, energy=0.5)])
        assert e.evaluate(current_time=1.6, recent_events=[]).intent == BehaviourIntent.BAIL


class TestStage3ProfileDefaults:
    def test_fast_energy_ema_alpha(self) -> None: assert ConservativePocketDrummer.fast_energy_ema_alpha == 0.15
    def test_slow_energy_ema_alpha(self) -> None: assert ConservativePocketDrummer.slow_energy_ema_alpha == 0.02
    def test_build_trend_threshold(self) -> None: assert ConservativePocketDrummer.build_trend_threshold == 0.15
    def test_reduce_trend_threshold(self) -> None: assert ConservativePocketDrummer.reduce_trend_threshold == -0.10
    def test_drop_trend_threshold(self) -> None: assert ConservativePocketDrummer.drop_trend_threshold == -0.30
    def test_max_density_for_build(self) -> None: assert ConservativePocketDrummer.max_density_for_build == 0.80
    def test_low_energy_threshold_for_drop(self) -> None: assert ConservativePocketDrummer.low_energy_threshold_for_drop == 0.25
    def test_min_build_duration_seconds(self) -> None: assert ConservativePocketDrummer.min_build_duration_seconds == 2.0
    def test_min_reduce_duration_seconds(self) -> None: assert ConservativePocketDrummer.min_reduce_duration_seconds == 2.0
    def test_min_drop_duration_seconds(self) -> None: assert ConservativePocketDrummer.min_drop_duration_seconds == 1.0