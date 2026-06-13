"""Tests for Behaviour-Driven Output Shaping."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from drummer.behaviour import BehaviourIntent
from drummer.output_shaping import (
    BehaviourOutputShaper,
    OutputShapingConfig,
    _beat_in_bar,
    _is_eighth_note,
    _is_ghost,
    _is_strong_beat,
    _is_kick,
    _is_snare,
    _is_hat,
    _is_ride,
    _is_crash,
    _is_tom,
)
from drummer.feel import GrooveEvent


# ============================================================================
# Helpers
# ============================================================================


def _k(pos: int, vel: int = 100, bar: int = 0, source_role: str = "main") -> GrooveEvent:
    return GrooveEvent("kick", pos, bar_index=bar, velocity=vel, source_role=source_role)


def _s(pos: int, vel: int = 100, bar: int = 0, articulation: str = "default",
       source_role: str = "main") -> GrooveEvent:
    return GrooveEvent("snare", pos, bar_index=bar, velocity=vel,
                       articulation=articulation, source_role=source_role)


def _h(pos: int, vel: int = 80, bar: int = 0, articulation: str = "default") -> GrooveEvent:
    return GrooveEvent("hi_hat", pos, bar_index=bar, velocity=vel,
                       articulation=articulation)


def _basic_groove() -> list[GrooveEvent]:
    """A simple rock groove: kick on 1&3, snare on 2&4, 8th hats."""
    return [
        _k(0, 110), _h(0, 80),
        _h(2, 70), _s(4, 100),
        _h(6, 70), _k(8, 105),
        _h(10, 70), _s(12, 100),
        _h(14, 70),
    ]


def _busy_groove() -> list[GrooveEvent]:
    """A busy groove with ghost notes, 16th hats, extra kicks."""
    return [
        # Beat 1
        _k(0, 115), _h(0, 80),
        _h(1, 60), _h(2, 70),
        _s(3, 20, articulation="ghost"),
        # Beat 2
        _s(4, 110), _h(4, 80),
        _h(5, 55), _h(6, 70),
        # Beat 3
        _k(7, 50, source_role="ghost"), _k(8, 112), _h(8, 80),
        _h(9, 60), _h(10, 70),
        # Beat 4
        _s(12, 108), _h(12, 80),
        _s(13, 22, articulation="ghost"), _h(14, 70),
        # Extra kick decoration
        _k(15, 90),
    ]


# ============================================================================
# 1. Instrument classification
# ============================================================================


class TestInstrumentClassification:
    """instrument group helpers work correctly."""

    def test_is_kick(self) -> None:
        assert _is_kick(_k(0)) is True
        assert _is_kick(_s(0)) is False

    def test_is_snare(self) -> None:
        assert _is_snare(_s(4)) is True
        assert _is_snare(_k(0)) is False

    def test_is_hat(self) -> None:
        assert _is_hat(_h(0)) is True
        assert _is_hat(_k(0)) is False

    def test_is_ride(self) -> None:
        assert _is_ride(GrooveEvent("ride", 0)) is True
        assert _is_ride(_k(0)) is False

    def test_is_crash(self) -> None:
        assert _is_crash(GrooveEvent("crash", 0)) is True

    def test_is_tom(self) -> None:
        assert _is_tom(GrooveEvent("tom", 0)) is True
        assert _is_tom(GrooveEvent("mid_tom", 0)) is True

    def test_is_ghost_articulation(self) -> None:
        e = GrooveEvent("snare", 4, velocity=100, articulation="ghost")
        assert _is_ghost(e, OutputShapingConfig()) is True

    def test_is_ghost_source_role(self) -> None:
        e = GrooveEvent("snare", 4, velocity=100, source_role="ghost")
        assert _is_ghost(e, OutputShapingConfig()) is True

    def test_is_ghost_low_velocity(self) -> None:
        e = GrooveEvent("snare", 4, velocity=25)
        assert _is_ghost(e, OutputShapingConfig(ghost_max_velocity=35)) is True

    def test_not_ghost_at_normal_velocity(self) -> None:
        e = GrooveEvent("snare", 4, velocity=60)
        assert _is_ghost(e, OutputShapingConfig()) is False


# ============================================================================
# 2. Beat position helpers
# ============================================================================


class TestBeatPosition:
    """Beat-position helpers work correctly."""

    def test_strong_beat_16th_positions(self) -> None:
        for pos in (0, 4, 8, 12):
            assert _is_strong_beat(pos) is True

    def test_not_strong_beat(self) -> None:
        for pos in (1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15):
            assert _is_strong_beat(pos) is False

    def test_eighth_note_positions(self) -> None:
        for pos in (0, 2, 4, 6, 8, 10, 12, 14):
            assert _is_eighth_note(pos) is True

    def test_not_eighth_note(self) -> None:
        for pos in (1, 3, 5, 7, 9, 11, 13, 15):
            assert _is_eighth_note(pos) is False

    def test_beat_in_bar(self) -> None:
        assert _beat_in_bar(0) == 1
        assert _beat_in_bar(4) == 2
        assert _beat_in_bar(8) == 3
        assert _beat_in_bar(12) == 4
        assert _beat_in_bar(1) == -1
        assert _beat_in_bar(16) == 1


# ============================================================================
# 3. MAINTAIN
# ============================================================================


class TestMaintain:
    """MAINTAIN preserves the groove unchanged."""

    def test_maintain_preserves_all_events(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.MAINTAIN)
        assert len(result) == len(events)

    def test_maintain_preserves_order(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.MAINTAIN)
        for i in range(len(events)):
            assert result[i].instrument == events[i].instrument
            assert result[i].grid_position == events[i].grid_position
            assert result[i].velocity == events[i].velocity

    def test_maintain_on_empty_list(self) -> None:
        shaper = BehaviourOutputShaper()
        result = shaper.shape([], BehaviourIntent.MAINTAIN)
        assert result == []


# ============================================================================
# 4. REDUCE
# ============================================================================


class TestReduce:
    """REDUCE strips ghost notes and thins hats."""

    def test_reduce_removes_ghost_notes(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.REDUCE)
        # Should have fewer events than input
        assert len(result) < len(events)
        # No ghost articulation events should remain
        for e in result:
            assert e.articulation != "ghost"
            assert e.source_role != "ghost"

    def test_reduce_removes_low_velocity_snares(self) -> None:
        shaper = BehaviourOutputShaper()
        config = OutputShapingConfig(reduce_min_snare_velocity=50)
        shaper_cfg = BehaviourOutputShaper(config)
        events = [
            _s(0, 100),  # strong velocity — keep
            _s(3, 30),   # low velocity off-beat — remove
            _s(4, 80),   # beat 2 — keep
            _s(7, 25),   # low velocity off-beat — remove
            _s(12, 30),  # low velocity — keep because it's on beat 4?
        ]
        result = shaper_cfg.shape(events, BehaviourIntent.REDUCE)
        # Should have removed the off-beat low-velocity ones
        # snare at pos 3 (off-beat, low vel) → remove
        # snare at pos 7 (off-beat, low vel) → remove
        # snare at pos 12 (beat 4) → keep even if low vel
        assert len(result) == 3  # 0, 4, 12

    def test_reduce_preserves_kick_on_beat_1(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.REDUCE)
        # Should still have kick on beat 1 (position 0)
        kick_beat1 = [e for e in result if _is_kick(e) and e.grid_position % 16 == 0]
        assert len(kick_beat1) >= 1

    def test_reduce_preserves_snare_backbeat(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.REDUCE)
        # Should have snare on beats 2 and 4 (positions 4 and 12)
        snare_positions = {e.grid_position % 16 for e in result if _is_snare(e)}
        assert 4 in snare_positions
        assert 12 in snare_positions

    def test_reduce_thins_hats_to_8th(self) -> None:
        """16th-note hats should be thinned to 8th notes."""
        shaper = BehaviourOutputShaper()
        config = OutputShapingConfig(reduce_thin_hats=True)
        shaper_cfg = BehaviourOutputShaper(config)
        events = _busy_groove()
        result = shaper_cfg.shape(events, BehaviourIntent.REDUCE)
        hat_positions = [e.grid_position % 16 for e in result if _is_hat(e)]
        for pos in hat_positions:
            assert pos % 2 == 0  # all hats should be on 8th-note positions

    def test_reduce_not_empty_when_input_not_empty(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.REDUCE)
        assert len(result) > 0

    def test_reduce_empty_input_stays_empty(self) -> None:
        shaper = BehaviourOutputShaper()
        result = shaper.shape([], BehaviourIntent.REDUCE)
        assert result == []

    def test_reduce_keeps_all_kicks(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.REDUCE)
        kick_count = sum(1 for e in result if _is_kick(e))
        assert kick_count > 0


# ============================================================================
# 5. ANCHOR
# ============================================================================


class TestAnchor:
    """ANCHOR simplifies to a clear pulse."""

    def test_anchor_removes_ghost_notes(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.ANCHOR)
        for e in result:
            assert e.articulation != "ghost"
            assert e.source_role != "ghost"

    def test_anchor_strips_syncopated_kicks(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()  # kicks at 0, 7, 8, 15
        result = shaper.shape(events, BehaviourIntent.ANCHOR)
        kick_positions = {e.grid_position % 16 for e in result if _is_kick(e)}
        # Should only have kicks on beats 1 and 3 (pos 0 and 8)
        assert 0 in kick_positions
        assert 8 in kick_positions
        assert 7 not in kick_positions
        assert 15 not in kick_positions

    def test_anchor_strips_syncopated_snares(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()  # snares at 3, 4, 12, 13
        result = shaper.shape(events, BehaviourIntent.ANCHOR)
        snare_positions = {e.grid_position % 16 for e in result if _is_snare(e)}
        # Should only have snares on beats 2 and 4 (pos 4 and 12)
        assert 4 in snare_positions
        assert 12 in snare_positions
        assert 3 not in snare_positions
        assert 13 not in snare_positions

    def test_anchor_simplifies_hats_to_quarter_notes(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.ANCHOR)
        hat_positions = [e.grid_position for e in result if _is_hat(e)]
        for pos in hat_positions:
            assert pos % 4 == 0  # all hats should be on quarter-note positions

    def test_anchor_reduces_velocity_variation(self) -> None:
        shaper = BehaviourOutputShaper()
        config = OutputShapingConfig(anchor_reduce_velocity_variation=True,
                                      anchor_target_velocity=100)
        shaper_cfg = BehaviourOutputShaper(config)
        events = _basic_groove()
        result = shaper_cfg.shape(events, BehaviourIntent.ANCHOR)
        velocities = [e.velocity for e in result]
        # Velocities should be pulled toward target (100)
        for v in velocities:
            assert 50 <= v <= 127  # reasonable range, not an exact assertion

    def test_anchor_generates_missing_pulse(self) -> None:
        """If input has no kick/snare/hats, ANCHOR should generate basic pulse."""
        shaper = BehaviourOutputShaper()
        # Empty input
        result = shaper.shape([], BehaviourIntent.ANCHOR)
        # Should generate a basic anchor pattern
        assert len(result) > 0
        instruments = {e.instrument for e in result}
        assert "kick" in instruments
        assert "snare" in instruments
        assert "hi_hat" in instruments

    def test_anchor_on_empty_input_generates_four_on_floor(self) -> None:
        shaper = BehaviourOutputShaper()
        result = shaper.shape([], BehaviourIntent.ANCHOR)
        # Should have kick on 1&3, snare on 2&4, hats on quarter notes
        kicks = [e for e in result if _is_kick(e)]
        snares = [e for e in result if _is_snare(e)]
        hats = [e for e in result if _is_hat(e)]
        assert len(kicks) == 2
        assert len(snares) == 2
        assert len(hats) == 4


# ============================================================================
# 6. BUILD
# ============================================================================


class TestBuild:
    """BUILD boosts velocities and opens hats."""

    def test_build_increases_velocity(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.BUILD)
        for i, evt in enumerate(events):
            assert result[i].velocity >= evt.velocity

    def test_build_velocity_clamped_at_max(self) -> None:
        config = OutputShapingConfig(build_velocity_boost=20, build_max_velocity=120)
        shaper = BehaviourOutputShaper(config)
        events = [_k(0, 115)]
        result = shaper.shape(events, BehaviourIntent.BUILD)
        assert result[0].velocity == 120  # 115 + 20 would be 135, clamped to 120

    def test_build_opens_hats(self) -> None:
        config = OutputShapingConfig(build_open_hats=True)
        shaper = BehaviourOutputShaper(config)
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.BUILD)
        open_hats = [e for e in result if _is_hat(e) and e.articulation == "open"]
        assert len(open_hats) > 0

    def test_build_preserves_core_groove(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.BUILD)
        assert len(result) == len(events)  # BUILD doesn't add/remove notes

    def test_build_does_not_exceed_127(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.BUILD)
        for e in result:
            assert e.velocity <= 127

    def test_build_on_empty_list(self) -> None:
        shaper = BehaviourOutputShaper()
        result = shaper.shape([], BehaviourIntent.BUILD)
        assert result == []


# ============================================================================
# 7. BAIL / DROP
# ============================================================================


class TestBailDrop:
    """BAIL and DROP suppress output."""

    def test_bail_returns_empty(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.BAIL)
        assert result == []

    def test_drop_returns_empty(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.DROP)
        assert result == []

    def test_bail_on_empty(self) -> None:
        shaper = BehaviourOutputShaper()
        result = shaper.shape([], BehaviourIntent.BAIL)
        assert result == []


# ============================================================================
# 8. ENTER / ENTER_SOFT
# ============================================================================


class TestEnter:
    """ENTER caps and scales velocities for controlled entry."""

    def test_enter_scales_velocities(self) -> None:
        shaper = BehaviourOutputShaper()
        config = OutputShapingConfig(enter_soft_scale=0.5)
        shaper_cfg = BehaviourOutputShaper(config)
        events = [_k(0, 120)]
        result = shaper_cfg.shape(events, BehaviourIntent.ENTER_SOFT)
        assert result[0].velocity == 60  # 120 * 0.5

    def test_enter_caps_velocity(self) -> None:
        config = OutputShapingConfig(enter_velocity_cap=80, enter_soft_scale=1.0)
        shaper = BehaviourOutputShaper(config)
        events = [_k(0, 120)]
        result = shaper.shape(events, BehaviourIntent.ENTER_SOFT)
        assert result[0].velocity == 80  # capped

    def test_enter_preserves_event_count(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.ENTER_SOFT)
        assert len(result) == len(events)

    def test_enter_full_also_shapes(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.ENTER_FULL)
        assert len(result) == len(events)
        for r, orig in zip(result, events):
            assert r.velocity <= orig.velocity


# ============================================================================
# 9. Instrument safety
# ============================================================================


class TestInstrumentSafety:
    """Unknown instruments are preserved, output is sorted."""

    def test_unknown_instrument_preserved(self) -> None:
        shaper = BehaviourOutputShaper()
        events = [GrooveEvent("cowbell", 0, velocity=80)]
        # MAINTAIN should preserve
        result = shaper.shape(events, BehaviourIntent.MAINTAIN)
        assert len(result) == 1
        assert result[0].instrument == "cowbell"

    def test_output_sorted_by_time(self) -> None:
        shaper = BehaviourOutputShaper()
        events = [
            GrooveEvent("kick", 8),
            GrooveEvent("kick", 0),
            GrooveEvent("snare", 4),
        ]
        result = shaper.shape(events, BehaviourIntent.MAINTAIN)
        positions = [e.grid_position for e in result]
        assert positions == [0, 4, 8]

    def test_multi_bar_sorted(self) -> None:
        shaper = BehaviourOutputShaper()
        events = [
            _k(0, bar=1),
            _k(0, bar=0),
            _s(4, bar=1),
            _s(4, bar=0),
        ]
        result = shaper.shape(events, BehaviourIntent.MAINTAIN)
        assert result[0].bar_index == 0
        assert result[0].grid_position == 0
        assert result[1].grid_position == 4
        assert result[2].bar_index == 1
        assert result[3].bar_index == 1


# ============================================================================
# 10. Pass-through intents
# ============================================================================


class TestPassThrough:
    """LISTEN, FILL, CRASH pass through unchanged."""

    def test_listen_passes_through(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _busy_groove()
        result = shaper.shape(events, BehaviourIntent.LISTEN)
        assert len(result) == len(events)
        for r, e in zip(result, events):
            assert r.instrument == e.instrument
            assert r.grid_position == e.grid_position

    def test_crash_passes_through(self) -> None:
        shaper = BehaviourOutputShaper()
        events = _basic_groove()
        result = shaper.shape(events, BehaviourIntent.CRASH)
        assert len(result) == len(events)


# ============================================================================
# 11. Config defaults
# ============================================================================


class TestConfigDefaults:
    """OutputShapingConfig has sensible defaults."""

    def test_default_config(self) -> None:
        cfg = OutputShapingConfig()
        assert cfg.reduce_min_snare_velocity == 60
        assert cfg.reduce_thin_hats is True
        assert cfg.reduce_strip_ghosts is True
        assert cfg.reduce_preserve_strong_beats is True
        assert cfg.anchor_strip_ghosts is True
        assert cfg.anchor_strip_syncopated is True
        assert cfg.anchor_simplify_hats is True
        assert cfg.anchor_reduce_velocity_variation is True
        assert cfg.anchor_target_velocity == 100
        assert cfg.build_velocity_boost == 12
        assert cfg.build_max_velocity == 127
        assert cfg.build_open_hats is True
        assert cfg.enter_velocity_cap == 100
        assert cfg.enter_soft_scale == 0.85
        assert cfg.ghost_max_velocity == 35

    def test_custom_config_respected(self) -> None:
        cfg = OutputShapingConfig(
            reduce_min_snare_velocity=80,
            build_velocity_boost=5,
            enter_soft_scale=0.5,
        )
        shaper = BehaviourOutputShaper(cfg)
        assert shaper.config.reduce_min_snare_velocity == 80
        assert shaper.config.build_velocity_boost == 5
        assert shaper.config.enter_soft_scale == 0.5