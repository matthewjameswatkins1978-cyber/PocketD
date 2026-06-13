"""Tests for Stage 4 — Feature-Driven Behaviour Engine.

Tests FeatureDrivenBehaviourEngine using FeatureSnapshot inputs,
covering ENTER, MAINTAIN, BUILD, REDUCE, ANCHOR, BAIL, hysteresis,
and backward compatibility.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from drummer.behaviour import (
    BehaviourDecision,
    BehaviourIntent,
    ConservativePocketDrummer,
    DrummerProfile,
    FeatureDrivenBehaviourEngine,
)
from perception.features import FeatureMonitorConfig, FeatureSnapshot


# ============================================================================
# Helpers
# ============================================================================


def _snap(
    timestamp: float = 0.0,
    *,
    input_density: float = 0.0,
    strength_ema: float = 0.0,
    fast_strength_ema: float = 0.0,
    slow_strength_ema: float = 0.0,
    change_score: float = 0.0,
    silence_duration: float = 0.0,
    repetition_stability: float = 0.0,
    phase_alignment: float | None = None,
    player_certainty: float = 0.0,
) -> FeatureSnapshot:
    """Shorthand for building a FeatureSnapshot with only the needed fields."""
    return FeatureSnapshot(
        timestamp=timestamp,
        input_density=input_density,
        strength_ema=strength_ema,
        fast_strength_ema=fast_strength_ema,
        slow_strength_ema=slow_strength_ema,
        change_score=change_score,
        silence_duration=silence_duration,
        repetition_stability=repetition_stability,
        phase_alignment=phase_alignment,
        player_certainty=player_certainty,
    )


# ============================================================================
# 1. Feature-driven ENTER
# ============================================================================


class TestFeatureEnter:
    """Stable repeated snapshots transition from LISTEN to ENTER."""

    def test_stable_repeated_snapshots_enter(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # Feed multiple confirming snapshots
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Now it should have entered
        assert eng.has_entered

    def test_one_good_snapshot_not_enough(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        s = _snap(
            timestamp=0.0,
            repetition_stability=0.85,
            player_certainty=0.80,
            phase_alignment=0.70,
        )
        d = eng.evaluate(s)
        assert d.intent == BehaviourIntent.LISTEN
        assert eng.has_entered is False

    def test_two_good_snapshots_not_enough_with_default_confirmation(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(2):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        assert eng.has_entered is False  # default needs 3

    def test_low_repetition_prevents_enter(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(5):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.55,  # below 0.70 threshold
                player_certainty=0.80,
                phase_alignment=0.70,
            )
            d = eng.evaluate(s)
        assert eng.has_entered is False
        assert d.intent == BehaviourIntent.LISTEN
        assert "repetition_stability" in d.reason.lower()

    def test_low_certainty_prevents_enter(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(5):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.80,
                player_certainty=0.50,  # below 0.65 threshold
                phase_alignment=0.70,
            )
            d = eng.evaluate(s)
        assert eng.has_entered is False
        assert d.intent == BehaviourIntent.LISTEN
        assert "player_certainty" in d.reason.lower()

    def test_confirmation_resets_on_bad_snapshot(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # Two good ones
        for i in range(2):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Bad snapshot resets counter
        bad = _snap(
            timestamp=1.5,
            repetition_stability=0.50,
            player_certainty=0.50,
        )
        eng.evaluate(bad)
        # Good ones again — need 3 fresh
        for i in range(3):
            s = _snap(
                timestamp=2.0 + float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        assert eng.has_entered

    def test_custom_confirmation_count(self) -> None:
        profile = DrummerProfile(
            name="test",
            hysteresis_margin=0.10,
            bail_silence_seconds=0.50,
            density_inversion_threshold=0.75,
            fill_probability_base=0.05,
            energy_ema_alpha=0.10,
            density_ema_alpha=0.10,
            enter_confirmation_snapshots=1,
        )
        eng = FeatureDrivenBehaviourEngine(profile=profile)
        s = _snap(
            timestamp=0.0,
            repetition_stability=0.75,
            player_certainty=0.70,
            phase_alignment=0.60,
        )
        d = eng.evaluate(s)
        assert eng.has_entered
        assert d.intent == BehaviourIntent.ENTER_SOFT


# ============================================================================
# 2. MAINTAIN bias
# ============================================================================


class TestFeatureMaintain:
    """Once active, ordinary stable snapshots remain MAINTAIN."""

    def test_default_to_maintain_after_entry(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # Enter
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Now a neutral snapshot
        neutral = _snap(
            timestamp=2.0,
            repetition_stability=0.80,
            player_certainty=0.75,
            input_density=0.3,
            change_score=0.05,
        )
        d = eng.evaluate(neutral)
        assert d.intent == BehaviourIntent.MAINTAIN

    def test_small_fluctuations_do_not_change_state(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Slightly lower certainty but still above anchor threshold
        mild = _snap(
            timestamp=2.0,
            repetition_stability=0.70,
            player_certainty=0.60,
            input_density=0.4,
        )
        d = eng.evaluate(mild)
        assert d.intent == BehaviourIntent.MAINTAIN


# ============================================================================
# 3. BUILD
# ============================================================================


class TestFeatureBuild:
    """Rising strength/change_score triggers BUILD."""

    def test_build_triggers_with_change_score(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # Enter
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Build-like snapshot
        build_snap = _snap(
            timestamp=2.0,
            change_score=0.30,  # >= 0.20 threshold
            player_certainty=0.65,  # >= 0.55 threshold
            input_density=0.3,  # below reduce threshold
            repetition_stability=0.75,
        )
        d = eng.evaluate(build_snap)
        assert d.intent == BehaviourIntent.BUILD
        assert "BUILD" in d.reason

    def test_build_blocked_by_low_certainty(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # High change_score but low certainty
        snap = _snap(
            timestamp=2.0,
            change_score=0.35,
            player_certainty=0.40,  # below 0.55
            input_density=0.3,
        )
        d = eng.evaluate(snap)
        assert d.intent != BehaviourIntent.BUILD

    def test_build_blocked_by_high_density(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # High change_score but density is too busy
        snap = _snap(
            timestamp=2.0,
            change_score=0.35,
            player_certainty=0.65,
            input_density=0.80,  # >= reduce_density_threshold (0.75)
        )
        d = eng.evaluate(snap)
        assert d.intent != BehaviourIntent.BUILD

    def test_build_hysteresis_keeps_build(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Trigger BUILD
        eng.evaluate(_snap(
            timestamp=2.0,
            change_score=0.30,
            player_certainty=0.65,
            input_density=0.3,
            repetition_stability=0.75,
            phase_alignment=0.60,
        ))
        # Slightly lower change_score — hysteresis keeps BUILD
        mild = _snap(
            timestamp=2.5,
            change_score=0.15,  # below entry threshold but above exit
            player_certainty=0.65,
            input_density=0.3,
            repetition_stability=0.75,
            phase_alignment=0.60,
        )
        d = eng.evaluate(mild)
        assert d.intent == BehaviourIntent.BUILD

    def test_build_exits_when_change_drops_below_hysteresis(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Trigger BUILD
        eng.evaluate(_snap(timestamp=2.0, change_score=0.30, player_certainty=0.65, input_density=0.3))
        # Drop far below hysteresis exit threshold
        exit_snap = _snap(
            timestamp=2.5,
            change_score=0.05,  # well below (0.20 - 0.10) = 0.10
            player_certainty=0.65,
            input_density=0.3,
        )
        d = eng.evaluate(exit_snap)
        assert d.intent != BehaviourIntent.BUILD


# ============================================================================
# 4. REDUCE / Density Inversion
# ============================================================================


class TestFeatureReduce:
    """High input density triggers REDUCE."""

    def test_high_density_triggers_reduce(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Dense snapshot — needs adequate stability so ANCHOR doesn't trigger first
        dense = _snap(
            timestamp=2.0,
            input_density=0.85,
            player_certainty=0.70,
            change_score=0.05,
            repetition_stability=0.75,
            phase_alignment=0.60,
        )
        d = eng.evaluate(dense)
        assert d.intent == BehaviourIntent.REDUCE
        assert "density inversion" in d.reason.lower()

    def test_density_reduce_has_hysteresis(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Trigger REDUCE
        eng.evaluate(_snap(
            timestamp=2.0, input_density=0.85, player_certainty=0.70,
            repetition_stability=0.75, phase_alignment=0.60,
        ))
        # Slightly lower density — hysteresis keeps REDUCE
        mild = _snap(
            timestamp=2.5, input_density=0.72, player_certainty=0.70,
            repetition_stability=0.75, phase_alignment=0.60,
        )
        d = eng.evaluate(mild)
        assert d.intent == BehaviourIntent.REDUCE

    def test_dense_but_stable_not_panic(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # High density triggers REDUCE (not BAIL)
        dense = _snap(
            timestamp=2.0,
            input_density=0.90,
            player_certainty=0.70,
            repetition_stability=0.75,
            phase_alignment=0.60,
        )
        d = eng.evaluate(dense)
        assert d.intent == BehaviourIntent.REDUCE
        # REDUCE is musical restraint, not panic

    def test_reduce_density_inversion_overrides_build(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # High change_score AND high density → ANCHOR or REDUCE wins, not BUILD
        # (REDUCE has lower priority than ANCHOR but both block BUILD)
        snap = _snap(
            timestamp=2.0,
            input_density=0.85,
            player_certainty=0.70,
            change_score=0.30,
            repetition_stability=0.75,
            phase_alignment=0.60,
        )
        d = eng.evaluate(snap)
        assert d.intent == BehaviourIntent.REDUCE  # not BUILD


# ============================================================================
# 5. ANCHOR / uncertainty
# ============================================================================


class TestFeatureAnchor:
    """Low certainty triggers ANCHOR."""

    def test_low_certainty_triggers_anchor(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Low certainty
        weak = _snap(
            timestamp=2.0,
            player_certainty=0.25,  # < 0.40
            repetition_stability=0.50,
            phase_alignment=0.60,
        )
        d = eng.evaluate(weak)
        assert d.intent == BehaviourIntent.ANCHOR

    def test_poor_phase_triggers_anchor(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Bad phase alignment
        weak = _snap(
            timestamp=2.0,
            player_certainty=0.55,
            repetition_stability=0.75,
            phase_alignment=0.30,  # < 0.45
        )
        d = eng.evaluate(weak)
        assert d.intent == BehaviourIntent.ANCHOR
        assert "phase_alignment" in d.reason

    def test_erratic_repetition_triggers_anchor(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Erratic repetition
        weak = _snap(
            timestamp=2.0,
            player_certainty=0.60,
            repetition_stability=0.20,  # < 0.35
        )
        d = eng.evaluate(weak)
        assert d.intent == BehaviourIntent.ANCHOR
        assert "repetition_stability" in d.reason

    def test_anchor_hysteresis_keeps_anchor(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Trigger ANCHOR
        eng.evaluate(_snap(timestamp=2.0, player_certainty=0.25, repetition_stability=0.50))
        # Marginally recovered — still in ANCHOR
        mid = _snap(
            timestamp=2.5,
            player_certainty=0.45,
            repetition_stability=0.40,
            phase_alignment=0.50,
        )
        d = eng.evaluate(mid)
        assert d.intent == BehaviourIntent.ANCHOR

    def test_anchor_recovery_exits(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Trigger ANCHOR
        eng.evaluate(_snap(timestamp=2.0, player_certainty=0.25, repetition_stability=0.50))
        # Strong recovery
        strong = _snap(
            timestamp=2.5,
            player_certainty=0.70,
            repetition_stability=0.80,
            phase_alignment=0.60,
        )
        d = eng.evaluate(strong)
        assert d.intent != BehaviourIntent.ANCHOR


# ============================================================================
# 6. BAIL / silence
# ============================================================================


class TestFeatureBail:
    """Long silence after active input triggers BAIL."""

    def test_long_silence_triggers_bail(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # Enter first
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Silence
        silent = _snap(
            timestamp=5.0,
            silence_duration=2.0,  # > 1.50 feature bail threshold
            player_certainty=0.0,
            repetition_stability=0.0,
        )
        d = eng.evaluate(silent)
        assert d.intent == BehaviourIntent.BAIL

    def test_brief_silence_does_not_trigger_bail(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Brief silence
        silent = _snap(
            timestamp=2.5,
            silence_duration=0.5,  # < 1.50
            player_certainty=0.60,
            repetition_stability=0.60,
        )
        d = eng.evaluate(silent)
        assert d.intent != BehaviourIntent.BAIL

    def test_empty_initial_state_does_not_bail(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # No prior entry, just a silent snapshot
        silent = _snap(
            timestamp=10.0,
            silence_duration=5.0,
        )
        d = eng.evaluate(silent)
        # Should not be BAIL — should be LISTEN (or at least not BAIL)
        assert d.intent == BehaviourIntent.LISTEN

    def test_bail_overrides_everything(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Start BUILD
        eng.evaluate(_snap(timestamp=2.0, change_score=0.30, player_certainty=0.65, input_density=0.3))
        # BAIL should override BUILD
        bail = _snap(
            timestamp=5.0,
            silence_duration=2.0,
        )
        d = eng.evaluate(bail)
        assert d.intent == BehaviourIntent.BAIL


# ============================================================================
# 7. Hysteresis
# ============================================================================


class TestFeatureHysteresis:
    """Values hovering around thresholds do not cause rapid toggling."""

    def test_build_hover_does_not_flicker(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Enter BUILD
        eng.evaluate(_snap(timestamp=2.0, change_score=0.30, player_certainty=0.65, input_density=0.3))
        # Hover around the threshold boundary
        intents: list[BehaviourIntent] = []
        for i in range(10):
            # Alternates between just above and just below entry threshold
            change = 0.19 if i % 2 == 0 else 0.21
            s = _snap(
                timestamp=2.5 + float(i) * 0.1,
                change_score=change,
                player_certainty=0.65,
                input_density=0.3,
            )
            d = eng.evaluate(s)
            intents.append(d.intent)
        # Should not flip rapidly — hysteresis should keep BUILD stable
        unique = set(intents)
        assert len(unique) == 1  # all the same (hysteresis prevents flickering)

    def test_anchor_hover_does_not_flicker(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Enter ANCHOR
        eng.evaluate(_snap(timestamp=2.0, player_certainty=0.25, repetition_stability=0.50))
        # Hover around recovery boundary
        for _ in range(5):
            s = _snap(
                timestamp=3.0,
                player_certainty=0.47,  # between 0.40 and 0.50
                repetition_stability=0.42,  # between 0.35 and 0.45
            )
            d = eng.evaluate(s)
            assert d.intent == BehaviourIntent.ANCHOR  # stays ANCHOR

    def test_reduce_hover_does_not_flicker(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        # Enter REDUCE
        eng.evaluate(_snap(
            timestamp=2.0, input_density=0.85, player_certainty=0.70,
            repetition_stability=0.75, phase_alignment=0.60,
        ))
        # Hover around exit boundary
        for _ in range(3):
            s = _snap(
                timestamp=3.0,
                input_density=0.68,  # between 0.65 and 0.75
                player_certainty=0.70,
                repetition_stability=0.75,
                phase_alignment=0.60,
            )
            d = eng.evaluate(s)
            assert d.intent == BehaviourIntent.REDUCE  # stays REDUCE


# ============================================================================
# 8. Backward compatibility
# ============================================================================


class TestFeatureBackwardCompat:
    """Existing BehaviourEngine and profile defaults are untouched."""

    def test_conservative_profile_stage4_defaults(self) -> None:
        p = ConservativePocketDrummer
        assert p.enter_certainty_threshold == 0.65
        assert p.enter_repetition_threshold == 0.70
        assert p.enter_confirmation_snapshots == 3
        assert p.build_change_threshold == 0.20
        assert p.build_certainty_threshold == 0.55
        assert p.reduce_density_threshold == 0.75
        assert p.anchor_certainty_threshold == 0.40
        assert p.anchor_repetition_threshold == 0.35
        assert p.anchor_phase_threshold == 0.45
        assert p.feature_bail_silence_seconds == 1.50
        assert p.feature_hysteresis_margin == 0.10

    def test_engine_default_profile_is_conservative(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        assert eng.profile is ConservativePocketDrummer

    def test_custom_profile_respected(self) -> None:
        profile = DrummerProfile(
            name="test",
            hysteresis_margin=0.10,
            bail_silence_seconds=0.50,
            density_inversion_threshold=0.75,
            fill_probability_base=0.05,
            energy_ema_alpha=0.10,
            density_ema_alpha=0.10,
            enter_certainty_threshold=0.80,
        )
        eng = FeatureDrivenBehaviourEngine(profile=profile)
        assert eng.profile.enter_certainty_threshold == 0.80

    def test_reset_clears_engine_state(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        # Enter
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        assert eng.has_entered is True
        eng.reset()
        assert eng.has_entered is False
        assert eng.previous_intent == BehaviourIntent.LISTEN
        assert eng.last_snapshot is None

    def test_feature_engine_does_not_affect_original_engine(self) -> None:
        """Original BehaviourEngine is untouched."""
        from drummer.behaviour import BehaviourEngine
        e = BehaviourEngine()
        d = e.evaluate(current_time=0.0, recent_events=[])
        assert d.intent == BehaviourIntent.LISTEN


# ============================================================================
# 9. Decision quality
# ============================================================================


class TestFeatureDecisionQuality:
    """Decisions have appropriate confidence and reason strings."""

    def test_enter_decision_has_scores(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            d = eng.evaluate(s)
            if i == 2:
                assert d.intent == BehaviourIntent.ENTER_SOFT
                assert "repetition_stability" in d.scores
                assert "player_certainty" in d.scores
                assert "confirmation_count" in d.scores
                assert d.confidence > 0.0

    def test_listen_decision_has_confidence(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        s = _snap(timestamp=0.0)
        d = eng.evaluate(s)
        assert d.intent == BehaviourIntent.LISTEN
        assert d.confidence == 0.3

    def test_anchor_decision_confidence_bounded(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        for i in range(3):
            s = _snap(
                timestamp=float(i) * 0.5,
                repetition_stability=0.75,
                player_certainty=0.70,
                phase_alignment=0.60,
            )
            eng.evaluate(s)
        weak = _snap(timestamp=2.0, player_certainty=0.25, repetition_stability=0.50)
        d = eng.evaluate(weak)
        assert d.intent == BehaviourIntent.ANCHOR
        assert 0.0 <= d.confidence <= 1.0

    def test_all_decisions_have_evaluated_at(self) -> None:
        eng = FeatureDrivenBehaviourEngine()
        s = _snap(timestamp=1.5)
        d = eng.evaluate(s)
        assert d.evaluated_at == 1.5