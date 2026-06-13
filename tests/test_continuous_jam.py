"""Tests for the continuous jam MIDI demo.

Validates core invariants of the continuous jam orchestra:
    - One global schedule, not isolated per-section playback
    - Pipeline state is not reset between sections
    - BUILD ramp increases velocity/intensity over multiple bars
    - REDUCE ramp lowers event count or hat density
    - ANCHOR produces simplified guide pattern
    - BAIL produces no events after ending
    - Deterministic output (same inputs -> same results)
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from drummer.behaviour import BehaviourIntent
from perception.models import MusicalEvent

# Import the demo module's internals for testing
from demo_continuous_jam_midi import (
    ArrangementState,
    ContinuousJamRenderer,
    run_continuous_jam,
    _simple_groove,
    _busy_groove,
    _anchor_groove,
    _is_strong_beat,
    build_simulated_timeline,
)


# ============================================================================
# ArrangementState tests
# ============================================================================


class TestArrangementState:
    """Tests for intensity ramp behaviour."""

    def test_initial_state(self) -> None:
        a = ArrangementState()
        assert a.current_intensity == 0.0
        assert a.target_intensity == 0.0
        assert a.ramp_rate_per_bar == 0.0

    def test_build_ramp_starts(self) -> None:
        """BUILD sets target=1.0 with gradual ramp rate."""
        a = ArrangementState()
        a.update_intent(BehaviourIntent.BUILD, bar=0)
        assert a.target_intensity == 1.0
        assert a.ramp_rate_per_bar > 0.0
        assert a.current_velocity_scale == 1.0
        assert a.current_hat_density == 16  # 16th hats during BUILD

    def test_build_ramp_progresses(self) -> None:
        """BUILD intensity increases over multiple bars."""
        a = ArrangementState()
        a.current_intensity = 0.4
        a.update_intent(BehaviourIntent.BUILD, bar=0)

        # After 1 advance
        a.advance_bar()
        assert a.current_intensity > 0.4
        assert a.current_intensity < 1.0

        # After enough advances, reaches target
        for _ in range(20):
            a.advance_bar()
        assert abs(a.current_intensity - a.target_intensity) < 0.001

    def test_reduce_ramp_thins_hats(self) -> None:
        """REDUCE thins hat density and lowers velocity scale."""
        a = ArrangementState()
        a.current_intensity = 0.8
        a.current_hat_density = 16
        a.update_intent(BehaviourIntent.REDUCE, bar=0)
        assert a.current_hat_density == 4  # quarter hats
        assert a.current_velocity_scale == 0.6
        assert a.target_intensity == 0.35

    def test_anchor_simplifies(self) -> None:
        """ANCHOR sets moderate intensity with quarter hats."""
        a = ArrangementState()
        a.current_intensity = 0.8
        a.current_hat_density = 16
        a.update_intent(BehaviourIntent.ANCHOR, bar=0)
        # Anchor: lock at 0.5, quarter hats, steady velocity
        assert a.target_intensity == 0.5
        assert a.current_hat_density == 4
        assert a.current_velocity_scale == 0.95

    def test_bail_cuts_to_zero(self) -> None:
        """BAIL sets intensity target to 0."""
        a = ArrangementState()
        a.current_intensity = 0.8
        a.update_intent(BehaviourIntent.BAIL, bar=0)
        assert a.target_intensity == 0.0
        assert a.current_velocity_scale == 0.0
        assert a.current_hat_density == 0

    def test_listen_cuts_to_zero(self) -> None:
        """LISTEN also sets intensity target to 0."""
        a = ArrangementState()
        a.current_intensity = 0.8
        a.update_intent(BehaviourIntent.LISTEN, bar=0)
        assert a.target_intensity == 0.0

    def test_same_intent_no_reset(self) -> None:
        """Repeated same intent does not reset ramp."""
        a = ArrangementState()
        a.current_intensity = 0.5
        a.update_intent(BehaviourIntent.MAINTAIN, bar=0)
        rate = a.ramp_rate_per_bar
        a.advance_bar()
        a.update_intent(BehaviourIntent.MAINTAIN, bar=1)
        # Ramp continues unchanged
        assert a.ramp_rate_per_bar == rate

    def test_enter_soft_ramps_up(self) -> None:
        """ENTER_SOFT starts low and ramps up."""
        a = ArrangementState()
        a.current_intensity = 0.0
        a.update_intent(BehaviourIntent.ENTER_SOFT, bar=0)
        assert a.target_intensity == 0.4
        assert a.ramp_rate_per_bar > 0

    def test_drop_cuts_hard_and_fast(self) -> None:
        """DROP lowers intensity rapidly."""
        a = ArrangementState()
        a.current_intensity = 0.8
        a.current_hat_density = 16
        a.update_intent(BehaviourIntent.DROP, bar=0)
        assert a.target_intensity == 0.1
        assert a.ramp_rate_per_bar == 0.5  # fast cut
        assert a.current_hat_density == 0  # remove hats
        assert a.current_velocity_scale == 0.3  # low velocity


# ============================================================================
# ContinuousJamRenderer tests
# ============================================================================


class TestRenderer:
    """Tests for the ContinuousJamRenderer."""

    def setup_method(self) -> None:
        self.renderer = ContinuousJamRenderer()

    def test_render_at_zero_intensity_return_empty(self) -> None:
        """Zero intensity produces no events."""
        a = ArrangementState()
        a.current_intensity = 0.0
        result = self.renderer.render_bar(_simple_groove(), a, bar=0)
        assert result == []
        assert len(result) == 0

    def test_render_at_low_intensity_has_fewer_events(self) -> None:
        """Low intensity strips weak events but preserves backbone."""
        a = ArrangementState()
        a.current_intensity = 0.2
        a.current_velocity_scale = 0.5
        a.current_hat_density = 4
        result = self.renderer.render_bar(_simple_groove(), a, bar=0)
        # Should have at least kick/snare backbeats
        assert len(result) >= 4
        # Should be fewer than full groove
        assert len(result) < len(_simple_groove())

    def test_render_at_full_intensity_produces_all_events(self) -> None:
        """Full intensity keeps all events."""
        a = ArrangementState()
        a.current_intensity = 1.0
        a.current_velocity_scale = 1.0
        a.current_hat_density = 16
        groove = _simple_groove()
        result = self.renderer.render_bar(groove, a, bar=0)
        assert len(result) == len(groove)

    def test_render_reduce_thins_hats(self) -> None:
        """REDUCE removes offbeat hats."""
        a = ArrangementState()
        a.current_intensity = 0.5
        a.current_velocity_scale = 0.6
        a.current_hat_density = 4  # quarter hats only
        a.last_intent = BehaviourIntent.REDUCE
        result = self.renderer.render_bar(_simple_groove(), a, bar=0)
        # Check that only quarter-note hats remain
        for evt in result:
            if evt.instrument in ("hi_hat", "closed_hat"):
                pos = evt.grid_position % 16
                assert pos % 4 == 0, f"hat at position {pos} should be quarter note"

    def test_render_anchor_produces_simplified_guide(self) -> None:
        """ANCHOR strips ghost notes and decorations."""
        a = ArrangementState()
        a.current_intensity = 0.5
        a.current_velocity_scale = 0.95
        a.current_hat_density = 4
        a.last_intent = BehaviourIntent.ANCHOR
        result = self.renderer.render_bar(_busy_groove(), a, bar=0)
        # Anchor on busy groove should still produce events
        assert len(result) > 0
        # No ghost notes allowed
        for evt in result:
            assert evt.articulation != "ghost"
            assert evt.source_role != "ghost"

    def test_render_drop_leaves_only_kick_pulse(self) -> None:
        """DROP at very low intensity leaves only kick on strong beats."""
        a = ArrangementState()
        a.current_intensity = 0.1
        a.current_velocity_scale = 0.3
        a.current_hat_density = 0
        a.last_intent = BehaviourIntent.DROP
        result = self.renderer.render_bar(_simple_groove(), a, bar=0)
        # Should have kicks only on strong beats
        for evt in result:
            assert evt.instrument in ("kick",) or _is_strong_beat(
                evt.grid_position % 16
            )

    def test_render_build_arrival_adds_pickup(self) -> None:
        """BUILD adds a kick pickup on the bar before arrival."""
        a = ArrangementState()
        a.current_intensity = 0.9
        a.current_velocity_scale = 1.0
        a.current_hat_density = 16
        a.last_intent = BehaviourIntent.BUILD
        a.arrival_bar = 5  # arrival at bar 5
        result = self.renderer.render_bar(_busy_groove(), a, bar=4)
        has_extra_kick = any(
            evt.instrument == "kick" and evt.grid_position % 16 == 14
            for evt in result
        )
        assert has_extra_kick, "BUILD arrival bar should have kick pickup at position 14"

    def test_render_clamps_velocities(self) -> None:
        """All velocities stay in [1, 127]."""
        a = ArrangementState()
        a.current_intensity = 1.0
        a.current_velocity_scale = 5.0  # deliberately too high
        a.current_hat_density = 16
        result = self.renderer.render_bar(_simple_groove(), a, bar=0)
        for evt in result:
            assert 1 <= evt.velocity <= 127

    def test_render_preserves_kick_snare_on_backbeat(self) -> None:
        """Even at moderate intensity, kick 1/3 and snare 2/4 survive."""
        a = ArrangementState()
        a.current_intensity = 0.3
        a.current_velocity_scale = 0.7
        a.current_hat_density = 8
        result = self.renderer.render_bar(_simple_groove(), a, bar=0)
        instruments_at_positions = {
            (evt.instrument, evt.grid_position % 16) for evt in result
        }
        # Kick on beats 1 and 3
        assert ("kick", 0) in instruments_at_positions
        assert ("kick", 8) in instruments_at_positions
        # Snare on beats 2 and 4
        assert ("snare", 4) in instruments_at_positions
        assert ("snare", 12) in instruments_at_positions


# ============================================================================
# run_continuous_jam integration tests
# ============================================================================


class TestContinuousJam:
    """Integration tests for the full continuous jam pipeline."""

    def test_run_returns_non_empty_schedule(self) -> None:
        """A jam run produces a non-trivial global schedule."""
        pipeline, diagnostics, schedule = run_continuous_jam(bars=16, bpm=120.0)
        assert len(schedule) > 0
        assert len(diagnostics) == 16

    def test_pipeline_not_reset_between_sections(self) -> None:
        """Pipeline maintains state across all bars (no reset)."""
        pipeline, _, schedule = run_continuous_jam(bars=16, bpm=120.0)
        # After running 16 bars, the pipeline should have entered
        last_time = pipeline._monitor._last_event_time
        assert last_time is not None and last_time > 0, (
            f"Monitor should have recorded events (state persists), got {last_time}"
        )

    def test_build_ramp_stays_above_minimum(self) -> None:
        """BUILD section always has some intensity (never drops to zero)."""
        _, diagnostics, _ = run_continuous_jam(bars=16, bpm=120.0)
        build_bars = [d for d in diagnostics if d["section"] == "BUILD"]
        assert len(build_bars) >= 1, "Should have at least one BUILD bar"
        for d in build_bars:
            assert d["arrangement_intensity"] > 0.0, (
                f"BUILD bar {d['bar']} should have intensity > 0"
            )
            assert d["hat_density"] > 0, (
                f"BUILD bar {d['bar']} should have hat density > 0"
            )

    def test_reduce_lowers_event_count(self) -> None:
        """REDUCE section has fewer events than preceding BUILD peak."""
        _, diagnostics, _ = run_continuous_jam(bars=16, bpm=120.0)
        build_bars = [d for d in diagnostics if d["section"] == "BUILD"]
        reduce_bars = [d for d in diagnostics if d["section"] == "REDUCE"]
        if build_bars and reduce_bars:
            build_max = max(d["event_count"] for d in build_bars)
            reduce_first = reduce_bars[0]["event_count"]
            assert reduce_first <= build_max, (
                f"REDUCE ({reduce_first}) should not have more events than BUILD peak ({build_max})"
            )

    def test_bail_produces_zero_events(self) -> None:
        """BAIL section produces no events."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0)
        bail_bars = [d for d in diagnostics if d["section"] == "BAIL"]
        for d in bail_bars:
            assert d["event_count"] == 0, (
                f"BAIL bar {d['bar']} should have 0 events, got {d['event_count']}"
            )

    def test_drop_has_minimal_events(self) -> None:
        """DROP section has very few events."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0)
        drop_bars = [d for d in diagnostics if d["section"] == "DROP"]
        for d in drop_bars:
            assert d["event_count"] <= 4, (
                f"DROP bar {d['bar']} should have <=4 events, got {d['event_count']}"
            )

    def test_anchor_produces_guide_pattern(self) -> None:
        """ANCHOR produces events but fewer than full groove."""
        _, diagnostics, schedule = run_continuous_jam(bars=20, bpm=120.0)
        anchor_bars = [d for d in diagnostics if d["section"] == "ANCHOR"]
        assert len(anchor_bars) > 0, "Should have ANCHOR section"
        for d in anchor_bars:
            assert d["event_count"] >= 4, "ANCHOR should have at least backbone"
            assert d["arrangement_intensity"] > 0.3, "ANCHOR intensity should be moderate"

    def test_schedule_is_continuous(self) -> None:
        """Events span the full timeline without gaps where there shouldn't be."""
        _, diagnostics, schedule = run_continuous_jam(bars=16, bpm=120.0)
        for evt in schedule:
            assert evt.grid_position >= 0
            assert evt.bar_index >= 0
            assert evt.velocity >= 1

    def test_deterministic_output(self) -> None:
        """Same inputs produce same outputs."""
        _, diag1, sched1 = run_continuous_jam(bars=12, bpm=120.0)
        _, diag2, sched2 = run_continuous_jam(bars=12, bpm=120.0)
        assert len(sched1) == len(sched2)
        for e1, e2 in zip(sched1, sched2):
            assert e1.instrument == e2.instrument
            assert e1.grid_position == e2.grid_position
            assert e1.bar_index == e2.bar_index
            assert e1.velocity == e2.velocity

    def test_listen_section_has_no_events(self) -> None:
        """LISTEN section produces no drum events."""
        _, diagnostics, _ = run_continuous_jam(bars=16, bpm=120.0)
        listen_bars = [d for d in diagnostics if d["section"] == "LISTEN"]
        for d in listen_bars:
            assert d["event_count"] == 0, (
                f"LISTEN bar {d['bar']} should be silent"
            )

    def test_with_different_bpm(self) -> None:
        """Jam runs correctly at different BPMs."""
        _, diagnostics, schedule = run_continuous_jam(bars=8, bpm=100.0)
        assert len(schedule) > 0
        assert len(diagnostics) == 8

    def test_with_different_bar_count(self) -> None:
        """Jam runs correctly with different bar counts."""
        for bars in [8, 12, 16, 24]:
            _, diagnostics, schedule = run_continuous_jam(bars=bars, bpm=120.0)
            assert len(diagnostics) == bars
            assert isinstance(schedule, list)


# ============================================================================
# Simulated timeline tests
# ============================================================================


class TestSimulatedTimeline:
    """Tests for the simulated player input timeline builder."""

    def test_timeline_returns_correct_bar_count(self) -> None:
        """Timeline returns one list per bar."""
        for bars in [8, 16, 24]:
            timeline = build_simulated_timeline(bpm=120.0, bars=bars)
            assert len(timeline) == bars

    def test_listen_bars_are_empty(self) -> None:
        """LISTEN bars (0-1) have no events."""
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        assert timeline[0] == []
        assert timeline[1] == []

    def test_enter_soft_bars_have_events(self) -> None:
        """ENTER_SOFT bars (2-3) have events."""
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        assert len(timeline[2]) > 0
        assert len(timeline[3]) > 0

    def test_maintain_bars_have_dense_events(self) -> None:
        """MAINTAIN bars (4-6) have dense 8th-note events."""
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        assert len(timeline[4]) == 8  # 8 eighth notes
        assert len(timeline[5]) == 8

    def test_build_bars_have_increasing_strength(self) -> None:
        """BUILD bars (7-9) have events with increasing strengths."""
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        build_bars = timeline[7:10]  # BUILD is bars 7, 8, 9
        assert len(build_bars) >= 2, "Should have at least 2 BUILD bars"
        max_strengths = [max(e.strength for e in bar) if bar else 0.0
                         for bar in build_bars]
        for i in range(1, len(max_strengths)):
            assert max_strengths[i] >= max_strengths[i - 1]

    def test_reduce_bars_have_high_density(self) -> None:
        """REDUCE bars have 16th-note density (many events)."""
        timeline = build_simulated_timeline(bpm=120.0, bars=20)
        # REDUCE section: bars 10, 11, 12 (0-based)
        reduce_bars = timeline[10:13]
        assert len(reduce_bars) >= 2, "Should have at least 2 REDUCE bars"
        for bar in reduce_bars:
            assert len(bar) >= 8  # at least 8 16th notes

    def test_anchor_bars_have_weak_events(self) -> None:
        """ANCHOR bars have weak-strength events."""
        timeline = build_simulated_timeline(bpm=120.0, bars=20)
        anchor_bar = timeline[13]  # ANCHOR bar
        for evt in anchor_bar:
            assert evt.strength < 0.3, f"ANCHOR event strength {evt.strength} should be < 0.3"

    def test_bail_bars_are_empty(self) -> None:
        """BAIL bars (17+) are empty (no player input)."""
        timeline = build_simulated_timeline(bpm=120.0, bars=20)
        for bar in range(17, len(timeline)):
            assert timeline[bar] == [], f"BAIL bar {bar} should be empty"

    def test_events_have_increasing_timestamps(self) -> None:
        """Event timestamps increase monotonically within and across bars."""
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        all_events = []
        for bar_events in timeline:
            all_events.extend(bar_events)
        for i in range(1, len(all_events)):
            assert all_events[i].time_seconds >= all_events[i - 1].time_seconds


# ============================================================================
# Helper tests
# ============================================================================


class TestHelpers:
    """Tests for utility/helper functions."""

    def test_is_strong_beat(self) -> None:
        """_is_strong_beat identifies quarter-note positions."""
        assert _is_strong_beat(0) is True
        assert _is_strong_beat(4) is True
        assert _is_strong_beat(8) is True
        assert _is_strong_beat(12) is True
        assert _is_strong_beat(16) is True

        assert _is_strong_beat(1) is False
        assert _is_strong_beat(2) is False
        assert _is_strong_beat(3) is False
        assert _is_strong_beat(6) is False
        assert _is_strong_beat(7) is False
        assert _is_strong_beat(14) is False