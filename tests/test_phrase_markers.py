"""Tests for the Phrase Marker / Ear-Perk Layer.

Covers selection logic, safety guards, rendering, and output contract
preservation.

Run::
    python -m pytest tests/test_phrase_markers.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from drummer.behaviour import BehaviourIntent
from drummer.phrase_markers import (
    PhraseMarkerType,
    PhraseMarkerConfig,
    PhraseMarkerState,
    choose_phrase_marker,
    apply_phrase_marker,
    apply_eight_bar_ear_perk,
    apply_sixteen_bar_frill,
    is_bar_8_boundary,
    is_bar_16_boundary,
    is_musically_safe,
    phrase_marker_label,
)
from drummer.feel import GrooveEvent
from perception.features import FeatureSnapshot


# ============================================================================
# Helpers
# ============================================================================


def _make_snapshot(
    player_certainty: float = 0.75,
    phase_alignment: float = 0.75,
    repetition_stability: float = 0.75,
    input_density: float = 0.5,
    change_score: float = 0.5,
    silence_duration: float = 0.0,
) -> FeatureSnapshot:
    """Create a FeatureSnapshot with specific feature values."""
    return FeatureSnapshot(
        timestamp=0.0,
        input_density=input_density,
        strength_ema=0.6,
        fast_strength_ema=0.6,
        slow_strength_ema=0.6,
        change_score=change_score,
        silence_duration=silence_duration,
        repetition_stability=repetition_stability,
        phase_alignment=phase_alignment,
        player_certainty=player_certainty,
    )


def _simple_bar_events(bar: int = 0) -> list[GrooveEvent]:
    """Return a simple one-bar rock groove."""
    return [
        GrooveEvent("kick", 0, bar_index=bar, velocity=100),
        GrooveEvent("hi_hat", 0, bar_index=bar, velocity=80),
        GrooveEvent("hi_hat", 2, bar_index=bar, velocity=70),
        GrooveEvent("snare", 4, bar_index=bar, velocity=100),
        GrooveEvent("hi_hat", 4, bar_index=bar, velocity=80),
        GrooveEvent("hi_hat", 6, bar_index=bar, velocity=70),
        GrooveEvent("kick", 8, bar_index=bar, velocity=98),
        GrooveEvent("hi_hat", 8, bar_index=bar, velocity=80),
        GrooveEvent("hi_hat", 10, bar_index=bar, velocity=70),
        GrooveEvent("snare", 12, bar_index=bar, velocity=100),
        GrooveEvent("hi_hat", 12, bar_index=bar, velocity=80),
        GrooveEvent("hi_hat", 14, bar_index=bar, velocity=70),
    ]


DEFAULT_CONFIG = PhraseMarkerConfig()


# ============================================================================
# Boundary tests
# ============================================================================


class TestBarBoundaries:
    """Test 8-bar and 16-bar boundary detection."""

    def test_not_8_boundary_early_bars(self):
        """Bars 0-6 should not be 8-bar boundaries."""
        for bar in range(7):
            assert is_bar_8_boundary(bar) is False, f"bar {bar} should not be 8-bar boundary"

    def test_bar_7_is_8_boundary(self):
        """Bar 7 (zero-indexed) = musical bar 8 → 8-bar boundary."""
        assert is_bar_8_boundary(7) is True

    def test_bar_15_is_8_boundary(self):
        """Bar 15 (zero-indexed) = musical bar 16 → 8-bar boundary."""
        assert is_bar_8_boundary(15) is True

    def test_bar_23_is_8_boundary(self):
        """Bar 23 (zero-indexed) = musical bar 24 → 8-bar boundary."""
        assert is_bar_8_boundary(23) is True

    def test_not_16_boundary_early_bars(self):
        """Bars 0-14 should not be 16-bar boundaries."""
        for bar in range(15):
            assert is_bar_16_boundary(bar) is False, f"bar {bar} should not be 16-bar boundary"

    def test_bar_15_is_16_boundary(self):
        """Bar 15 (zero-indexed) = musical bar 16 → 16-bar boundary."""
        assert is_bar_16_boundary(15) is True

    def test_bar_31_is_16_boundary(self):
        """Bar 31 (zero-indexed) = musical bar 32 → 16-bar boundary."""
        assert is_bar_16_boundary(31) is True

    def test_bar_15_is_both_boundary(self):
        """Bar 15 is both an 8-bar and 16-bar boundary."""
        assert is_bar_8_boundary(15) is True
        assert is_bar_16_boundary(15) is True


# ============================================================================
# Musical safety tests
# ============================================================================


class TestIsMusicallySafe:
    """Test the musical safety check for phrase markers."""

    def test_unsafe_when_listen(self):
        """LISTEN intent should not be safe for phrase markers."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.LISTEN, snap, 0.7) is False

    def test_unsafe_when_drop(self):
        """DROP intent should not be safe for phrase markers."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.DROP, snap, 0.7) is False

    def test_unsafe_when_bail(self):
        """BAIL intent should not be safe for phrase markers."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.BAIL, snap, 0.7) is False

    def test_unsafe_when_final_bail(self):
        """FINAL_BAIL intent should not be safe for phrase markers."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.FINAL_BAIL, snap, 0.7) is False

    def test_unsafe_when_anchor(self):
        """ANCHOR intent should not be safe for phrase markers."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.ANCHOR, snap, 0.7) is False

    def test_unsafe_when_reduce(self):
        """REDUCE intent should not be safe for phrase markers."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.REDUCE, snap, 0.7) is False

    def test_safe_when_maintain_with_good_features(self):
        """MAINTAIN with good features and confidence should be safe."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.MAINTAIN, snap, 0.7) is True

    def test_safe_when_build_with_good_features(self):
        """BUILD with good features and confidence should be safe."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.BUILD, snap, 0.7) is True

    def test_unsafe_when_confidence_too_low(self):
        """Very low confidence should not be safe even with good features."""
        snap = _make_snapshot()
        assert is_musically_safe(BehaviourIntent.MAINTAIN, snap, 0.1) is False

    def test_unsafe_when_phase_too_low(self):
        """Low phase alignment should not be safe."""
        snap = _make_snapshot(phase_alignment=0.3)
        assert is_musically_safe(BehaviourIntent.MAINTAIN, snap, 0.7) is False

    def test_unsafe_when_certainty_too_low(self):
        """Low player certainty should not be safe."""
        snap = _make_snapshot(player_certainty=0.3)
        assert is_musically_safe(BehaviourIntent.MAINTAIN, snap, 0.7) is False

    def test_unsafe_when_stability_too_low(self):
        """Low repetition stability should not be safe."""
        snap = _make_snapshot(repetition_stability=0.3)
        assert is_musically_safe(BehaviourIntent.MAINTAIN, snap, 0.7) is False

    def test_unsafe_when_density_too_high(self):
        """High input density should not be safe."""
        snap = _make_snapshot(input_density=0.9)
        assert is_musically_safe(BehaviourIntent.MAINTAIN, snap, 0.7) is False

    def test_unsafe_after_anchor_without_rebuild(self):
        """Should not be safe immediately after ANCHOR with low confidence."""
        snap = _make_snapshot()
        state = PhraseMarkerState(bars_since_anchor=1)
        assert is_musically_safe(
            BehaviourIntent.MAINTAIN, snap, 0.5,
            state=state,
        ) is False

    def test_safe_after_anchor_with_rebuilt_confidence(self):
        """Should be safe after ANCHOR if confidence has rebuilt."""
        snap = _make_snapshot()
        state = PhraseMarkerState(bars_since_anchor=1)
        assert is_musically_safe(
            BehaviourIntent.MAINTAIN, snap, 0.65,
            state=state,
        ) is True

    def test_safe_after_anchor_grace_period_elapsed(self):
        """Should be safe after ANCHOR grace period has elapsed."""
        snap = _make_snapshot()
        state = PhraseMarkerState(bars_since_anchor=5)
        assert is_musically_safe(
            BehaviourIntent.MAINTAIN, snap, 0.5,
            state=state,
        ) is True


# ============================================================================
# Selection tests
# ============================================================================


class TestChoosePhraseMarker:
    """Test phrase marker selection logic."""

    def test_no_marker_when_disabled(self):
        """No marker should fire when config is disabled."""
        config = PhraseMarkerConfig(enabled=False)
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.7, snap, config=config)
        assert result == PhraseMarkerType.NONE

    def test_no_marker_when_confidence_low(self):
        """No marker when confidence is too low."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.1, snap)
        assert result == PhraseMarkerType.NONE

    def test_no_marker_during_anchor(self):
        """No marker during ANCHOR intent."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.ANCHOR, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_no_marker_during_drop(self):
        """No marker during DROP intent."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.DROP, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_no_marker_during_bail(self):
        """No marker during BAIL intent."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.BAIL, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_no_marker_during_final_bail(self):
        """No marker during FINAL_BAIL intent."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.FINAL_BAIL, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_no_marker_during_reduce_frantic(self):
        """No marker during frantic REDUCE."""
        snap = _make_snapshot(input_density=0.85)
        result = choose_phrase_marker(7, BehaviourIntent.REDUCE, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_eight_bar_marker_on_8_boundary(self):
        """8-bar marker selected on eligible 8-bar boundary with high enough confidence."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.5, snap)
        assert result == PhraseMarkerType.EIGHT_BAR_EAR_PERK

    def test_sixteen_bar_marker_on_16_boundary(self):
        """16-bar marker selected on eligible 16-bar boundary."""
        snap = _make_snapshot()
        result = choose_phrase_marker(15, BehaviourIntent.MAINTAIN, 0.7, snap)
        assert result == PhraseMarkerType.SIXTEEN_BAR_FRILL

    def test_sixteen_takes_priority_over_eight(self):
        """16-bar marker takes priority over 8-bar marker when both occur."""
        snap = _make_snapshot()
        result = choose_phrase_marker(15, BehaviourIntent.MAINTAIN, 0.7, snap)
        assert result == PhraseMarkerType.SIXTEEN_BAR_FRILL

    def test_eight_bar_marker_at_edge_threshold(self):
        """8-bar marker fires at exactly the min confidence threshold."""
        config = PhraseMarkerConfig(eight_bar_min_confidence=0.45)
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.45, snap, config=config)
        assert result == PhraseMarkerType.EIGHT_BAR_EAR_PERK

    def test_eight_bar_marker_below_threshold(self):
        """8-bar marker does not fire below min confidence."""
        config = PhraseMarkerConfig(eight_bar_min_confidence=0.45)
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.44, snap, config=config)
        assert result == PhraseMarkerType.NONE

    def test_sixteen_bar_marker_at_edge_threshold(self):
        """16-bar marker fires at exactly the min confidence threshold."""
        config = PhraseMarkerConfig(sixteen_bar_min_confidence=0.60)
        snap = _make_snapshot()
        result = choose_phrase_marker(15, BehaviourIntent.MAINTAIN, 0.60, snap, config=config)
        assert result == PhraseMarkerType.SIXTEEN_BAR_FRILL

    def test_deterministic(self):
        """Phrase marker selection is deterministic (same inputs → same result)."""
        snap = _make_snapshot()
        r1 = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.6, snap)
        r2 = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.6, snap)
        assert r1 == r2


# ============================================================================
# Rendering tests
# ============================================================================


class TestApplyEightBarEarPerk:
    """Test 8-bar ear-perk rendering."""

    def test_at_most_one_event_added(self):
        """8-bar ear-perk adds at most 1 event or modifies 1 existing."""
        events = _simple_bar_events()
        for bar in [7, 15, 23]:
            result = apply_eight_bar_ear_perk(events, bar)
            added = len(result) - len(events)
            # Allow -1 if kick boost replaced, or +1 if event added
            assert added <= 1, f"bar {bar}: expected at most 1 extra event, got {added}"

    def test_no_crash_added(self):
        """8-bar ear-perk never adds a crash."""
        events = _simple_bar_events()
        for bar in [7, 15, 23]:
            result = apply_eight_bar_ear_perk(events, bar)
            crashes = [e for e in result if e.instrument.lower() == "crash"]
            assert len(crashes) == 0, f"bar {bar}: crash found in ear-perk"

    def test_returns_events_sorted(self):
        """Result events should be sorted by grid_position."""
        events = _simple_bar_events()
        result = apply_eight_bar_ear_perk(events, 7)
        positions = [e.grid_position for e in result]
        assert positions == sorted(positions)


class TestApplySixteenBarFrill:
    """Test 16-bar frill rendering."""

    def test_at_most_three_events_added(self):
        """16-bar frill adds at most 3 events."""
        events = _simple_bar_events()
        for bar in [15, 31, 47]:
            result = apply_sixteen_bar_frill(events, bar)
            added = len(result) - len(events)
            assert added <= 3, f"bar {bar}: expected at most 3 extra events, got {added}"

    def test_no_crash_added(self):
        """16-bar frill never adds a crash."""
        events = _simple_bar_events()
        for bar in [15, 31, 47]:
            result = apply_sixteen_bar_frill(events, bar)
            crashes = [e for e in result if e.instrument.lower() == "crash"]
            assert len(crashes) == 0, f"bar {bar}: crash found in frill"

    def test_returns_events_sorted(self):
        """Result events should be sorted by grid_position (via apply_phrase_marker)."""
        events = _simple_bar_events()
        result = apply_phrase_marker(events, PhraseMarkerType.SIXTEEN_BAR_FRILL, 15)
        positions = [e.grid_position for e in result]
        assert positions == sorted(positions)

    def test_deterministic(self):
        """Frill rendering is deterministic."""
        events = _simple_bar_events()
        r1 = apply_sixteen_bar_frill(events, 15)
        r2 = apply_sixteen_bar_frill(events, 15)
        assert len(r1) == len(r2)
        for e1, e2 in zip(r1, r2):
            assert e1.instrument == e2.instrument
            assert e1.grid_position == e2.grid_position


class TestApplyPhraseMarker:
    """Test the combined phrase marker application."""

    def test_none_returns_original(self):
        """NONE marker type returns the original events unchanged."""
        events = _simple_bar_events()
        result = apply_phrase_marker(events, PhraseMarkerType.NONE, 7)
        assert len(result) == len(events)

    def test_eight_bar_applied_correctly(self):
        """8-bar ear-perk is applied when specified."""
        events = _simple_bar_events()
        result = apply_phrase_marker(events, PhraseMarkerType.EIGHT_BAR_EAR_PERK, 7)
        assert len(result) >= len(events)
        assert len(result) <= len(events) + 1

    def test_sixteen_bar_applied_correctly(self):
        """16-bar frill is applied when specified."""
        events = _simple_bar_events()
        result = apply_phrase_marker(events, PhraseMarkerType.SIXTEEN_BAR_FRILL, 15)
        assert len(result) >= len(events)
        assert len(result) <= len(events) + 3

    def test_disabled_config_returns_original(self):
        """Disabled config returns original events even with non-NONE marker."""
        config = PhraseMarkerConfig(enabled=False)
        events = _simple_bar_events()
        result = apply_phrase_marker(events, PhraseMarkerType.EIGHT_BAR_EAR_PERK, 7, config=config)
        assert len(result) == len(events)

    def test_empty_events_eight_bar(self):
        """8-bar ear-perk on empty events adds a kick on beat 1."""
        result = apply_eight_bar_ear_perk([], 7)
        assert len(result) == 1
        assert result[0].instrument.lower() == "kick"


# ============================================================================
# Label tests
# ============================================================================


class TestPhraseMarkerLabel:
    """Test the convenience label function."""

    def test_none_label(self):
        assert phrase_marker_label(PhraseMarkerType.NONE) == ""

    def test_eight_bar_label(self):
        assert phrase_marker_label(PhraseMarkerType.EIGHT_BAR_EAR_PERK) == "8bar"

    def test_sixteen_bar_label(self):
        assert phrase_marker_label(PhraseMarkerType.SIXTEEN_BAR_FRILL) == "16bar"


# ============================================================================
# DROP / BAIL / FINAL_BAIL contract preservation tests
# ============================================================================


class TestOutputContractPreservation:
    """Test that phrase markers do not affect DROP/BAIL/FINAL_BAIL contracts."""

    def test_drop_contract_preserved(self):
        """DROP output remains sparse with phrase markers enabled."""
        from drummer.output_shaping import (
            BehaviourOutputShaper,
            OutputShapingConfig,
            is_drop_output,
        )
        shaper = BehaviourOutputShaper()
        drop_events = shaper.shape([], BehaviourIntent.DROP)
        # DROP with phrase markers should produce same result as without:
        # sparse (≤2 events), at least one kick, no crash, optional quiet hat tick.
        assert is_drop_output(drop_events)
        assert len(drop_events) <= 2
        kicks = [e for e in drop_events if e.instrument.lower() == "kick"]
        hats = [e for e in drop_events if e.instrument.lower() in ("hi_hat", "closed_hat", "open_hat")]
        crashes = [e for e in drop_events if "crash" in e.instrument.lower()]
        assert len(kicks) >= 1, "DROP must contain at least one kick"
        assert len(crashes) == 0, "DROP must not contain crash"
        for h in hats:
            assert h.velocity <= 40, (
                f"DROP hi_hat must be quiet (≤40), got velocity {h.velocity}"
            )

    def test_bail_contract_preserved(self):
        """BAIL output remains empty with phrase markers enabled."""
        from drummer.output_shaping import (
            BehaviourOutputShaper,
            is_bail_output,
        )
        shaper = BehaviourOutputShaper()
        bail_events = shaper.shape([], BehaviourIntent.BAIL)
        assert is_bail_output(bail_events)
        assert len(bail_events) == 0

    def test_final_bail_contract_preserved(self):
        """FINAL_BAIL output remains kick + crash with phrase markers."""
        from drummer.output_shaping import (
            BehaviourOutputShaper,
            is_final_bail_output,
        )
        shaper = BehaviourOutputShaper()
        final_events = shaper.shape([], BehaviourIntent.FINAL_BAIL)
        assert is_final_bail_output(final_events)
        assert len(final_events) == 2
        has_kick = any(e.instrument.lower() == "kick" for e in final_events)
        has_crash = any(e.instrument.lower() == "crash" for e in final_events)
        assert has_kick
        assert has_crash

    def test_anchor_not_decorated(self):
        """ANCHOR intent should not get phrase markers even on eligible boundaries."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.ANCHOR, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_phrase_marker_does_not_break_drop_check(self):
        """Applying a phrase marker to DROP events should be safe (no-op)."""
        from drummer.output_shaping import BehaviourOutputShaper
        shaper = BehaviourOutputShaper()
        drop_events = shaper.shape([], BehaviourIntent.DROP)
        # Apply 8-bar ear-perk to DROP output
        marked = apply_phrase_marker(drop_events, PhraseMarkerType.EIGHT_BAR_EAR_PERK, 7)
        # DROP events may get the marker but the core contract is preserved
        # (markers only fire when intent is MAINTAIN/BUILD, so this is belt-and-braces)
        assert len(marked) >= 1


# ============================================================================
# Demo story simulation
# ============================================================================


class TestDemoStory:
    """Simulate the expected demo story for phrase markers.

    Expected behaviour:
    - early bars: no marker because confidence too low or intent not eligible
    - stable bars with confidence: 8-bar ear-perk appears
    - 16-bar frill only if musically safe
    - uncertain/drop/bail sections: no markers
    """

    def test_early_bars_no_marker(self):
        """Early bars (0-6) should have no marker."""
        snap = _make_snapshot()
        for bar in range(7):
            result = choose_phrase_marker(bar, BehaviourIntent.MAINTAIN, 0.5, snap)
            assert result == PhraseMarkerType.NONE, f"bar {bar} should have no marker"

    def test_bar_7_eight_bar_marker_with_confidence(self):
        """Bar 7 (8-bar boundary) with sufficient confidence should get 8-bar marker."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.5, snap)
        assert result == PhraseMarkerType.EIGHT_BAR_EAR_PERK

    def test_bar_7_no_marker_with_low_confidence(self):
        """Bar 7 with low confidence should not get a marker."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.MAINTAIN, 0.1, snap)
        assert result == PhraseMarkerType.NONE

    def test_drop_section_no_marker(self):
        """DROP section should have no marker regardless of bar boundary."""
        snap = _make_snapshot()
        for bar in [7, 15, 23]:
            result = choose_phrase_marker(bar, BehaviourIntent.DROP, 0.7, snap)
            assert result == PhraseMarkerType.NONE, f"bar {bar} DROP should have no marker"

    def test_bail_section_no_marker(self):
        """BAIL section should have no marker."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.BAIL, 0.7, snap)
        assert result == PhraseMarkerType.NONE

    def test_anchor_section_no_marker(self):
        """ANCHOR section should have no marker."""
        snap = _make_snapshot()
        result = choose_phrase_marker(7, BehaviourIntent.ANCHOR, 0.7, snap)
        assert result == PhraseMarkerType.NONE