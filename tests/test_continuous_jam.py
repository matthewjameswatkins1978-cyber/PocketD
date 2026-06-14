"""Tests for the continuous jam MIDI demo.

Validates core invariants of both scripted and inferred modes.
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

from demo_continuous_jam_midi import (
    ArrangementState,
    ContinuousJamRenderer,
    run_continuous_jam,
    _simple_groove,
    _busy_groove,
    _anchor_groove,
    _is_strong_beat,
    build_simulated_timeline,
    _timeline_section_name,
    _phase_for_section,
)
from drummer.output_shaping import BehaviourOutputShaper


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
        a = ArrangementState()
        a.update_intent(BehaviourIntent.BUILD, bar=0)
        assert a.target_intensity == 1.0
        assert a.ramp_rate_per_bar > 0.0
        assert a.current_velocity_scale == 1.0
        assert a.current_hat_density == 16

    def test_build_ramp_progresses(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.4
        a.update_intent(BehaviourIntent.BUILD, bar=0)
        a.advance_bar()
        assert a.current_intensity > 0.4
        assert a.current_intensity < 1.0
        for _ in range(20):
            a.advance_bar()
        assert abs(a.current_intensity - a.target_intensity) < 0.001

    def test_reduce_ramp_thins_hats(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.8
        a.current_hat_density = 16
        a.update_intent(BehaviourIntent.REDUCE, bar=0)
        assert a.current_hat_density == 4
        assert a.current_velocity_scale == 0.6
        assert a.target_intensity == 0.35

    def test_anchor_simplifies(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.8
        a.current_hat_density = 16
        a.update_intent(BehaviourIntent.ANCHOR, bar=0)
        assert a.target_intensity == 0.5
        assert a.current_hat_density == 4
        assert a.current_velocity_scale == 0.95

    def test_bail_cuts_to_zero(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.8
        a.update_intent(BehaviourIntent.BAIL, bar=0)
        assert a.target_intensity == 0.0
        assert a.current_velocity_scale == 0.0
        assert a.current_hat_density == 0

    def test_listen_cuts_to_zero(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.8
        a.update_intent(BehaviourIntent.LISTEN, bar=0)
        assert a.target_intensity == 0.0

    def test_same_intent_no_reset(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.5
        a.update_intent(BehaviourIntent.MAINTAIN, bar=0)
        rate = a.ramp_rate_per_bar
        a.advance_bar()
        a.update_intent(BehaviourIntent.MAINTAIN, bar=1)
        assert a.ramp_rate_per_bar == rate

    def test_enter_soft_ramps_up(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.0
        a.update_intent(BehaviourIntent.ENTER_SOFT, bar=0)
        assert a.target_intensity == 0.4
        assert a.ramp_rate_per_bar > 0

    def test_drop_cuts_hard_and_fast(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.8
        a.current_hat_density = 16
        a.update_intent(BehaviourIntent.DROP, bar=0)
        assert a.target_intensity == 0.1
        assert a.ramp_rate_per_bar == 0.5
        assert a.current_hat_density == 0
        assert a.current_velocity_scale == 0.3


# ============================================================================
# ContinuousJamRenderer tests
# ============================================================================


class TestRenderer:
    """Tests for the ContinuousJamRenderer."""

    def setup_method(self) -> None:
        self.renderer = ContinuousJamRenderer()

    def test_render_at_zero_intensity_return_empty(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.0
        result = self.renderer.render_bar(_simple_groove(), a, bar=0, intent=BehaviourIntent.MAINTAIN)
        assert result == []

    def test_render_at_low_intensity_has_fewer_events(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.2
        a.current_velocity_scale = 0.5
        a.current_hat_density = 4
        result = self.renderer.render_bar(_simple_groove(), a, bar=0, intent=BehaviourIntent.MAINTAIN)
        assert len(result) >= 4
        assert len(result) < len(_simple_groove())

    def test_render_at_full_intensity_produces_all_events(self) -> None:
        a = ArrangementState()
        a.current_intensity = 1.0
        a.current_velocity_scale = 1.0
        a.current_hat_density = 16
        groove = _simple_groove()
        result = self.renderer.render_bar(groove, a, bar=0, intent=BehaviourIntent.MAINTAIN)
        assert len(result) == len(groove)

    def test_render_reduce_thins_hats(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.5
        a.current_velocity_scale = 0.6
        a.current_hat_density = 4
        a.last_intent = BehaviourIntent.REDUCE
        result = self.renderer.render_bar(_simple_groove(), a, bar=0, intent=BehaviourIntent.REDUCE)
        for evt in result:
            if evt.instrument in ("hi_hat", "closed_hat"):
                pos = evt.grid_position % 16
                assert pos % 4 == 0

    def test_render_anchor_produces_simplified_guide(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.5
        a.current_velocity_scale = 0.95
        a.current_hat_density = 4
        a.last_intent = BehaviourIntent.ANCHOR
        result = self.renderer.render_bar(_busy_groove(), a, bar=0, intent=BehaviourIntent.ANCHOR)
        assert len(result) > 0
        for evt in result:
            assert evt.articulation != "ghost"
            assert evt.source_role != "ghost"

    def test_render_drop_leaves_only_kick_pulse(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.1
        a.current_velocity_scale = 0.3
        a.current_hat_density = 0
        a.last_intent = BehaviourIntent.DROP
        result = self.renderer.render_bar(_simple_groove(), a, bar=0, intent=BehaviourIntent.DROP)
        for evt in result:
            assert evt.instrument in ("kick",) or _is_strong_beat(
                evt.grid_position % 16
            )

    def test_render_build_arrival_adds_pickup(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.9
        a.current_velocity_scale = 1.0
        a.current_hat_density = 16
        a.last_intent = BehaviourIntent.BUILD
        a.arrival_bar = 5
        result = self.renderer.render_bar(_busy_groove(), a, bar=4, intent=BehaviourIntent.BUILD)
        has_extra_kick = any(
            evt.instrument == "kick" and evt.grid_position % 16 == 14
            for evt in result
        )
        assert has_extra_kick

    def test_render_clamps_velocities(self) -> None:
        a = ArrangementState()
        a.current_intensity = 1.0
        a.current_velocity_scale = 5.0
        a.current_hat_density = 16
        result = self.renderer.render_bar(_simple_groove(), a, bar=0, intent=BehaviourIntent.MAINTAIN)
        for evt in result:
            assert 1 <= evt.velocity <= 127

    def test_render_preserves_kick_snare_on_backbeat(self) -> None:
        a = ArrangementState()
        a.current_intensity = 0.3
        a.current_velocity_scale = 0.7
        a.current_hat_density = 8
        result = self.renderer.render_bar(_simple_groove(), a, bar=0, intent=BehaviourIntent.MAINTAIN)
        instruments_at_positions = {
            (evt.instrument, evt.grid_position % 16) for evt in result
        }
        assert ("kick", 0) in instruments_at_positions
        assert ("kick", 8) in instruments_at_positions
        assert ("snare", 4) in instruments_at_positions
        assert ("snare", 12) in instruments_at_positions


# ============================================================================
# Scripted mode tests
# ============================================================================


class TestScriptedMode:
    """Scripted mode: DROP/ANCHOR/BAIL are forced for reliable arc."""

    def test_run_returns_non_empty_schedule(self) -> None:
        _, diagnostics, schedule = run_continuous_jam(bars=16, bpm=120.0, mode="scripted")
        assert len(schedule) > 0
        assert len(diagnostics) == 16

    def test_pipeline_not_reset_between_sections(self) -> None:
        pipeline, _, _ = run_continuous_jam(bars=16, bpm=120.0, mode="scripted")
        last_time = pipeline._monitor._last_event_time
        assert last_time is not None and last_time > 0

    def test_listen_section_has_no_events(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=16, bpm=120.0, mode="scripted")
        listen_bars = [d for d in diagnostics if d["section"] == "LISTEN"]
        for d in listen_bars:
            assert d["event_count"] == 0

    def test_bail_produces_zero_events(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="scripted")
        bail_bars = [d for d in diagnostics if d["section"] == "BAIL"]
        for d in bail_bars:
            assert d["event_count"] == 0

    def test_drop_has_minimal_events(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="scripted")
        drop_bars = [d for d in diagnostics if d["section"] == "DROP"]
        for d in drop_bars:
            assert d["event_count"] <= 4

    def test_anchor_produces_guide_pattern(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="scripted")
        anchor_bars = [d for d in diagnostics if d["section"] == "ANCHOR"]
        assert len(anchor_bars) > 0
        for d in anchor_bars:
            assert d["event_count"] >= 4
            assert d["arrangement_intensity"] >= 0.3

    def test_scripted_mode_overrides_are_present(self) -> None:
        """Scripted mode should show OVERRIDE for DROP section."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="scripted")
        mismatch_bars = [
            d for d in diagnostics
            if d["inferred_intent"] != d["intent"]
            and d["section"] in ("DROP",)
        ]
        assert len(mismatch_bars) >= 1, "DROP should be overridden in scripted mode"

    def test_build_is_inferred(self) -> None:
        """BUILD intents come from the pipeline, not forced.
        (The last BUILD bar may naturally transition to REDUCE
        via density inversion — that's still inferred.)"""
        _, diagnostics, _ = run_continuous_jam(bars=16, bpm=120.0, mode="scripted")
        build_bars = [d for d in diagnostics if d["section"] == "BUILD"]
        assert len(build_bars) >= 1
        build_intents = [d for d in build_bars if d["inferred_intent"] == "build"]
        assert len(build_intents) >= 1, "At least one BUILD bar should infer BUILD"

    def test_reduce_is_inferred(self) -> None:
        """REDUCE intents come from pipeline density detection."""
        _, diagnostics, _ = run_continuous_jam(bars=16, bpm=120.0, mode="scripted")
        reduce_bars = [d for d in diagnostics if d["section"] == "REDUCE"]
        if reduce_bars:
            for d in reduce_bars:
                assert d["inferred_intent"] == d["intent"]


# ============================================================================
# Inferred mode tests
# ============================================================================


class TestInferredMode:
    """Inferred mode: pipeline decides everything from feature input."""

    def test_run_returns_non_empty_schedule(self) -> None:
        _, diagnostics, schedule = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        assert len(schedule) > 0
        assert len(diagnostics) == 20

    def test_pipeline_not_reset_between_sections(self) -> None:
        pipeline, _, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        last_time = pipeline._monitor._last_event_time
        assert last_time is not None and last_time > 0

    def test_no_forced_overrides_for_unchanged_bars(self) -> None:
        """In inferred mode, INferred bars have inferred_intent == intent."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        # DROP, FINAL_BAIL, and BAIL are forced for output correctness.
        # Everything else should match.
        unchanged_sections = {"LISTEN", "ENTER_SOFT", "MAINTAIN", "BUILD", "REDUCE", "ANCHOR"}
        for d in diagnostics:
            if d["section"] in unchanged_sections:
                assert d["inferred_intent"] == d["intent"], (
                    f"Bar {d['bar']} ({d['section']}): "
                    f"inferred={d['inferred_intent']} != intent={d['intent']}"
                )

    def test_stable_input_reaches_enter_or_maintain(self) -> None:
        """Steady quarter/8th-note input reaches ENTER_SOFT then BUILD/MAINTAIN."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        # Bars 4-7 (MAINTAIN/BUILD sections) should not be LISTEN
        main_bars = [d for d in diagnostics if d["section"] in ("MAINTAIN", "ENTER_SOFT")]
        intents = [d["intent"] for d in main_bars if d["bar"] >= 4]
        assert "listen" not in intents, f"Should have entered by bar 4, got: {intents}"

    def test_controlled_rising_input_reaches_build(self) -> None:
        """Rising strength with 8th-note pattern → BUILD."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        build_bars = [d for d in diagnostics if d["section"] == "BUILD"]
        # At least one bar should have inferred BUILD
        build_intents = [d for d in build_bars if d["inferred_intent"] == "build"]
        assert len(build_intents) >= 1, "BUILD section should have at least one BUILD intent"

    def test_dense_input_reaches_reduce(self) -> None:
        """Frantic 16th-note input → REDUCE."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        reduce_bars = [d for d in diagnostics if d["section"] == "REDUCE"]
        if reduce_bars:
            reduce_intents = [d for d in reduce_bars if d["inferred_intent"] == "reduce"]
            assert len(reduce_intents) >= 1, "REDUCE section should have REDUCE intent"

    def test_weak_erratic_input_reaches_anchor(self) -> None:
        """Weak events + poor phase → ANCHOR."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        # ANCHOR section bars: 13-14
        # But ANCHOR may also fire at DROP bar (12) with phase=0.25
        anchor_intents = [
            d for d in diagnostics
            if d["inferred_intent"] == "anchor"
        ]
        assert len(anchor_intents) >= 1, "Should have at least one ANCHOR intent"

    def test_silence_after_playing_reaches_bail(self) -> None:
        """Silence after active playing → BAIL."""
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        bail_bars = [d for d in diagnostics if d["section"] == "BAIL"]
        for d in bail_bars:
            assert d["inferred_intent"] == "bail", (
                f"BAIL bar {d['bar']} should infer bail, got {d['inferred_intent']}"
            )
            assert d["event_count"] == 0

    def test_bail_produces_zero_events(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        bail_bars = [d for d in diagnostics if d["section"] == "BAIL"]
        for d in bail_bars:
            assert d["event_count"] == 0

    def test_listen_section_has_no_events(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        listen_bars = [d for d in diagnostics if d["section"] == "LISTEN"]
        for d in listen_bars:
            assert d["event_count"] == 0

    def test_build_ramp_stays_above_minimum(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        build_bars = [d for d in diagnostics if d["section"] == "BUILD"]
        for d in build_bars:
            assert d["arrangement_intensity"] > 0.0
            assert d["hat_density"] > 0

    def test_anchor_produces_guide_pattern(self) -> None:
        _, diagnostics, _ = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        anchor_bars = [d for d in diagnostics if d["intent"] == "anchor"]
        assert len(anchor_bars) > 0, "Should have at least one ANCHOR bar"
        for d in anchor_bars:
            assert d["arrangement_intensity"] > 0.3

    def test_schedule_is_continuous(self) -> None:
        _, _, schedule = run_continuous_jam(bars=20, bpm=120.0, mode="inferred")
        for evt in schedule:
            assert evt.grid_position >= 0
            assert evt.bar_index >= 0
            assert evt.velocity >= 1

    def test_deterministic_output(self) -> None:
        _, _, sched1 = run_continuous_jam(bars=12, bpm=120.0, mode="inferred")
        _, _, sched2 = run_continuous_jam(bars=12, bpm=120.0, mode="inferred")
        assert len(sched1) == len(sched2)
        for e1, e2 in zip(sched1, sched2):
            assert e1.instrument == e2.instrument
            assert e1.grid_position == e2.grid_position
            assert e1.bar_index == e2.bar_index
            assert e1.velocity == e2.velocity

    def test_with_different_bpm(self) -> None:
        _, diagnostics, schedule = run_continuous_jam(bars=8, bpm=100.0, mode="inferred")
        assert len(schedule) > 0
        assert len(diagnostics) == 8

    def test_with_different_bar_count(self) -> None:
        for bars in [8, 12, 16]:
            _, diagnostics, schedule = run_continuous_jam(bars=bars, bpm=120.0, mode="inferred")
            assert len(diagnostics) == bars
            assert isinstance(schedule, list)


# ============================================================================
# Simulated timeline tests
# ============================================================================


class TestSimulatedTimeline:
    """Tests for the simulated player input timeline builder."""

    def test_timeline_returns_correct_bar_count(self) -> None:
        for bars in [8, 16, 24]:
            timeline = build_simulated_timeline(bpm=120.0, bars=bars)
            assert len(timeline) == bars

    def test_listen_bars_are_empty(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        assert timeline[0] == []
        assert timeline[1] == []

    def test_enter_soft_bars_have_events(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        assert len(timeline[2]) > 0
        assert len(timeline[3]) > 0

    def test_maintain_bars_have_dense_events(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        assert len(timeline[4]) == 8
        assert len(timeline[5]) == 8

    def test_build_bars_have_increasing_strength(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        build_bars = timeline[7:10]
        assert len(build_bars) >= 2
        max_strengths = [max(e.strength for e in bar) if bar else 0.0
                         for bar in build_bars]
        for i in range(1, len(max_strengths)):
            assert max_strengths[i] >= max_strengths[i - 1]

    def test_reduce_bars_have_high_density(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=20)
        # REDUCE section: bars 10, 11 (bar 12 is DROP)
        reduce_bars = timeline[10:12]
        assert len(reduce_bars) >= 2
        for bar in reduce_bars:
            assert len(bar) >= 8

    def test_anchor_bars_have_weak_events(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=20)
        # ANCHOR section is now bar 16
        anchor_bar = timeline[16]
        for evt in anchor_bar:
            assert evt.strength < 0.3, f"ANCHOR events should be weak, got {evt.strength}"

    def test_bail_bars_are_empty(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=20)
        # BAIL section now starts at bar 19
        for bar in range(19, len(timeline)):
            assert timeline[bar] == []

    def test_events_have_increasing_timestamps(self) -> None:
        timeline = build_simulated_timeline(bpm=120.0, bars=16)
        all_events = []
        for bar_events in timeline:
            all_events.extend(bar_events)
        # Within each bar, check timestamps are increasing
        for bar_events in timeline:
            for i in range(1, len(bar_events)):
                assert bar_events[i].time_seconds >= bar_events[i - 1].time_seconds, \
                    f"Timestamps not increasing in bar: {bar_events}"


# ============================================================================
# Phase helper tests
# ============================================================================


class TestPhaseHelper:
    """Tests for phase alignment lookup."""

    def test_anchor_phase_is_poor(self) -> None:
        assert _phase_for_section("ANCHOR") < 0.45, "ANCHOR should have poor phase"

    def test_maintain_phase_is_good(self) -> None:
        assert _phase_for_section("MAINTAIN") > 0.5, "MAINTAIN should have good phase"

    def test_build_phase_is_controlled(self) -> None:
        assert _phase_for_section("BUILD") > 0.55, "BUILD should have controlled phase"

    def test_unknown_section_defaults(self) -> None:
        assert _phase_for_section("UNKNOWN") == 0.75


# ============================================================================
# Helper tests
# ============================================================================


class TestHelpers:
    """Tests for utility/helper functions."""

    def test_is_strong_beat(self) -> None:
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