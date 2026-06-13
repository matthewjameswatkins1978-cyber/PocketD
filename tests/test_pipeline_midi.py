"""Tests for pipeline MIDI conversion helpers — no MIDI hardware required."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest

from drummer.pipeline_midi import (
    resolve_note,
    groove_events_to_midi_messages,
    beat_to_seconds,
    grid_to_seconds,
    build_schedule,
    INSTRUMENT_TO_NOTE,
)
from drummer.feel import GrooveEvent


class TestResolveNote:
    """Note resolution from instrument names."""

    def test_kick_maps_to_36(self) -> None:
        assert resolve_note(GrooveEvent("kick", 0)) == 36

    def test_snare_maps_to_38(self) -> None:
        assert resolve_note(GrooveEvent("snare", 0)) == 38

    def test_hi_hat_maps_to_42(self) -> None:
        assert resolve_note(GrooveEvent("hi_hat", 0)) == 42

    def test_closed_hat_maps_to_42(self) -> None:
        assert resolve_note(GrooveEvent("closed_hat", 0)) == 42

    def test_open_hat_maps_to_46(self) -> None:
        assert resolve_note(GrooveEvent("open_hat", 0)) == 46

    def test_ride_maps_to_51(self) -> None:
        assert resolve_note(GrooveEvent("ride", 0)) == 51

    def test_crash_maps_to_49(self) -> None:
        assert resolve_note(GrooveEvent("crash", 0)) == 49

    def test_tom_maps_to_45(self) -> None:
        assert resolve_note(GrooveEvent("tom", 0)) == 45

    def test_unknown_instrument_returns_none(self) -> None:
        assert resolve_note(GrooveEvent("theremin", 0)) is None

    def test_group_resolution_fallback(self) -> None:
        # "rd" should resolve to ride via instrument group
        assert resolve_note(GrooveEvent("rd", 0)) == 51


class TestGrooveEventsToMidiMessages:
    """Pure conversion from GrooveEvents to (time, note, velocity) tuples."""

    def test_single_event_converts(self) -> None:
        events = [GrooveEvent("kick", 0, velocity=100)]
        msgs = groove_events_to_midi_messages(events, bpm=120)
        assert msgs == [(0.0, 36, 100)]

    def test_timing_at_120_bpm(self) -> None:
        """At 120 BPM, a 16th note = 0.125s."""
        events = [GrooveEvent("snare", 4, velocity=80)]
        msgs = groove_events_to_midi_messages(events, bpm=120)
        assert len(msgs) == 1
        assert msgs[0][0] == pytest.approx(0.5)  # 4 * 0.125

    def test_timing_at_60_bpm(self) -> None:
        """At 60 BPM, a 16th note = 0.25s."""
        events = [GrooveEvent("kick", 8, velocity=70)]
        msgs = groove_events_to_midi_messages(events, bpm=60)
        assert len(msgs) == 1
        assert msgs[0][0] == pytest.approx(2.0)  # 8 * 0.25

    def test_unknown_instruments_skipped(self) -> None:
        events = [
            GrooveEvent("kick", 0, velocity=100),
            GrooveEvent("theremin", 4, velocity=80),
            GrooveEvent("snare", 4, velocity=80),
        ]
        msgs = groove_events_to_midi_messages(events, bpm=120)
        assert len(msgs) == 2  # theremin skipped

    def test_empty_list_produces_empty(self) -> None:
        msgs = groove_events_to_midi_messages([], bpm=120)
        assert msgs == []

    def test_events_sorted_by_time(self) -> None:
        events = [
            GrooveEvent("snare", 12, velocity=80),
            GrooveEvent("kick", 0, velocity=100),
            GrooveEvent("hi_hat", 4, velocity=70),
        ]
        msgs = groove_events_to_midi_messages(events, bpm=120)
        times = [m[0] for m in msgs]
        assert times == sorted(times)

    def test_timing_offset_applied(self) -> None:
        """timing_offset_ms should affect the event time."""
        events = [GrooveEvent("kick", 0, velocity=100, timing_offset_ms=50)]
        msgs = groove_events_to_midi_messages(events, bpm=120)
        assert msgs[0][0] == pytest.approx(0.050)

    def test_all_expected_notes_in_mapping(self) -> None:
        """Verify the instrument-to-note mapping contains expected drums."""
        expected = {"kick", "snare", "hi_hat", "open_hat", "ride", "crash"}
        for inst in expected:
            assert inst in INSTRUMENT_TO_NOTE or inst in INSTRUMENT_TO_NOTE


# ============================================================================
# 3. Beat-to-seconds conversion
# ============================================================================


class TestBeatToSeconds:
    """Beat positions convert correctly to wall-clock seconds."""

    def test_120_bpm_beat_duration(self) -> None:
        assert beat_to_seconds(1.0, bpm=120) == pytest.approx(0.5)

    def test_120_bpm_beats(self) -> None:
        for beat, expected in [(0, 0.0), (1, 0.5), (2, 1.0), (3, 1.5)]:
            assert beat_to_seconds(float(beat), bpm=120) == pytest.approx(expected)

    def test_60_bpm_double_duration(self) -> None:
        assert beat_to_seconds(1.0, bpm=60) == pytest.approx(1.0)

    def test_2_bar_span(self) -> None:
        """8 beats = 2 bars of 4/4."""
        secs = beat_to_seconds(8.0, bpm=120)
        assert secs == pytest.approx(4.0)

    def test_grid_to_seconds_match(self) -> None:
        """grid_to_seconds(0) = 0s, grid_to_seconds(4) = 0.5s at 120 BPM."""
        assert grid_to_seconds(0, bpm=120) == pytest.approx(0.0)
        assert grid_to_seconds(4, bpm=120) == pytest.approx(0.5)
        assert grid_to_seconds(8, bpm=120) == pytest.approx(1.0)
        assert grid_to_seconds(12, bpm=120) == pytest.approx(1.5)

    def test_simultaneous_events_same_time(self) -> None:
        """Kick and hat on grid 0 have the same time."""
        t_kick = grid_to_seconds(0, bpm=120)
        t_hat = grid_to_seconds(0, bpm=120)
        assert t_kick == t_hat

    def test_8th_note_positions(self) -> None:
        """8th notes at 120 BPM: grid 0, 2, 4, 6 -> 0.0, 0.25, 0.5, 0.75s."""
        for grid, expected in [(0, 0.0), (2, 0.25), (4, 0.5), (6, 0.75)]:
            assert grid_to_seconds(grid, bpm=120) == pytest.approx(expected)


# ============================================================================
# 4. Scheduler message ordering
# ============================================================================


class TestBuildSchedule:
    """The schedule builder produces correct ordering."""

    def test_schedule_has_on_and_off(self) -> None:
        events = [GrooveEvent("kick", 0, velocity=100)]
        s = build_schedule(events, bpm=120, note_duration=0.1, repeats=1)
        assert len(s) == 2
        assert s[0] == (0.0, "on", 36, 100)
        assert s[1] == (0.1, "off", 36, 0)

    def test_note_off_before_note_on_at_same_time(self) -> None:
        """If a note_off and a new note_on happen at the exact same time,
        note_off should come first."""
        # Create two events: kick at 0.0s and snare at exactly 0.1s (when kick_off fires)
        events = [
            GrooveEvent("kick", 0, velocity=100),   # kick_on=0.0, kick_off=0.1
            GrooveEvent("snare", 4, velocity=80),   # snare_on=0.5, snare_off=0.6
        ]
        # Force note_duration so off of event 1 == on of event 2
        s = build_schedule(events, bpm=120, note_duration=0.5, repeats=1)
        # Find kick_off and snare_on
        kick_off_idx = next(i for i, e in enumerate(s) if e[1] == "off" and e[2] == 36)
        snare_on_idx = next(i for i, e in enumerate(s) if e[1] == "on" and e[2] == 38)
        # kick_off should come before snare_on if times are equal
        # They may not be exactly equal in this test, but verify ordering is correct
        for i in range(len(s) - 1):
            t1, typ1 = s[i][0], s[i][1]
            t2, typ2 = s[i + 1][0], s[i + 1][1]
            if t1 == t2:
                assert typ1 == "off" and typ2 == "on", \
                    f"At t={t1}, expected off before on, got {typ1} before {typ2}"

    def test_messages_sorted_by_time(self) -> None:
        events = [
            GrooveEvent("snare", 12, velocity=80),
            GrooveEvent("kick", 0, velocity=100),
            GrooveEvent("hi_hat", 4, velocity=70),
        ]
        s = build_schedule(events, bpm=120, note_duration=0.09, repeats=1)
        times = [e[0] for e in s]
        assert times == sorted(times)

    def test_note_off_does_not_delay_later_note_on(self) -> None:
        """note_off and note_on at different times are correctly interleaved."""
        events = [
            GrooveEvent("kick", 0, velocity=100),
            GrooveEvent("snare", 4, velocity=80),
        ]
        s = build_schedule(events, bpm=120, note_duration=0.09, repeats=1)
        # kick_on < kick_off < snare_on < snare_off
        types = [e[1] for e in s]
        assert types == ["on", "off", "on", "off"]

    def test_repeats_produce_multiple_bars(self) -> None:
        events = [GrooveEvent("kick", 0, velocity=100)]
        s = build_schedule(events, bpm=120, note_duration=0.09, repeats=2)
        # 2 bars * 2 events per bar = 4 events
        assert len(s) == 4
        # First bar: on at 0.0, off at 0.09
        # Second bar: on at 2.0, off at 2.09
        times = [e[0] for e in s]
        assert times[0] == pytest.approx(0.0)
        assert times[2] == pytest.approx(2.0)
