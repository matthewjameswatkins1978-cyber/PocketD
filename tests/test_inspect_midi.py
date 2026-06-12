"""Tests for tools/inspect_midi.py — MIDI file inspection and problem detection."""

from __future__ import annotations

import os
import tempfile

from mido import Message, MetaMessage, MidiFile, MidiTrack

from tools.inspect_midi import (
    check_tempo_present,
    detect_increasing_spacing,
    detect_large_gaps,
    extract_timing_data,
    run_all_checks,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_midi_file(
    ticks_per_beat: int = 480,
    tempo: int | None = 500000,
    note_events: list[tuple[int, int, int, int]] | None = None,
) -> str:
    """
    Build a tiny single-track MIDI file at a temporary path.

    Each element in *note_events* is (delta_ticks, note, velocity, channel).
    """
    mid = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    mid.tracks.append(track)

    # optional tempo
    if tempo is not None:
        track.append(MetaMessage("set_tempo", tempo=tempo))

    if note_events:
        for delta, note, velocity, channel in note_events:
            track.append(
                Message(
                    "note_on", note=note, velocity=velocity, channel=channel, time=delta
                )
            )

    fd, path = tempfile.mkstemp(suffix=".mid")
    os.close(fd)
    mid.save(path)
    return path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestExtractTimingData:
    """Verify absolute tick reconstruction and metadata extraction."""

    def test_single_note_at_zero(self) -> None:
        path = _make_midi_file(note_events=[(0, 42, 100, 9)])
        data = extract_timing_data(path)
        os.unlink(path)

        assert data["ticks_per_beat"] == 480
        assert data["tempo"] == 500000
        assert data["bpm"] is not None and abs(data["bpm"] - 120.0) < 0.01
        assert data["note_on_count"] == 1
        assert data["note_on_events"][0]["absolute_tick"] == 0.0
        assert data["note_on_events"][0]["delta_from_previous_note_on"] == 0.0

    def test_absolute_tick_reconstruction(self) -> None:
        """Three note_on events with known delta times."""
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (240, 42, 80, 9),
                (240, 36, 100, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        events = data["note_on_events"]
        assert len(events) == 3
        assert events[0]["absolute_tick"] == 0.0
        assert events[1]["absolute_tick"] == 240.0
        assert events[2]["absolute_tick"] == 480.0
        # deltas
        assert events[0]["delta_from_previous_note_on"] == 0.0
        assert events[1]["delta_from_previous_note_on"] == 240.0
        assert events[2]["delta_from_previous_note_on"] == 240.0

    def test_non_note_messages_do_not_affect_delta(self) -> None:
        """Only note_on messages should be considered for delta_from_previous_note_on."""
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (120, 42, 80, 9),  # half beat apart
                (240, 36, 100, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        events = data["note_on_events"]
        assert events[1]["delta_from_previous_note_on"] == 120.0

    def test_drum_name_resolution(self) -> None:
        path = _make_midi_file(
            note_events=[
                (0, 36, 100, 9),
                (0, 38, 100, 9),
                (0, 42, 100, 9),
                (0, 46, 100, 9),
                (0, 49, 100, 9),
                (0, 51, 100, 9),
                (0, 76, 100, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        names = {ev["note"]: ev["drum_name"] for ev in data["note_on_events"]}
        assert names[36] == "kick"
        assert names[38] == "snare"
        assert names[42] == "closed_hat"
        assert names[46] == "open_hat"
        assert names[49] == "crash"
        assert names[51] == "ride"
        assert names[76] == "metronome_click"

    def test_spacing_analysis(self) -> None:
        """Regular hi-hat (42) and metronome (76) spacing."""
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (240, 42, 80, 9),
                (240, 42, 80, 9),
                (10, 76, 60, 9),
                (230, 76, 60, 9),
                (240, 76, 60, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        # hi-hat (42) spacings: 240, 240
        assert data["hi_hat_spacings"] == [240.0, 240.0]
        # metronome (76) spacings: 230, 240
        assert data["metronome_spacings"] == [230.0, 240.0]
        # tick positions
        assert data["tick_positions"]["hi_hat_42"] == [0.0, 240.0, 480.0]
        assert data["tick_positions"]["metronome_76"] == [490.0, 720.0, 960.0]

    def test_no_tempo(self) -> None:
        path = _make_midi_file(tempo=None, note_events=[(0, 42, 100, 9)])
        data = extract_timing_data(path)
        os.unlink(path)

        assert data["tempo"] is None
        assert data["bpm"] is None


class TestDetectIncreasingSpacing:
    """Verify the increasing-spacing heuristic."""

    def test_regular_spacing_no_warning(self) -> None:
        spacings = [240.0, 240.0, 240.0, 240.0, 240.0, 240.0]
        warnings = detect_increasing_spacing(spacings)
        assert warnings == []

    def test_too_few_spacings_no_warning(self) -> None:
        spacings = [240.0, 240.0, 240.0]  # fewer than 4
        warnings = detect_increasing_spacing(spacings)
        assert warnings == []

    def test_increasing_spacing_triggers_warning(self) -> None:
        spacings = [240.0, 244.0, 400.0, 410.0, 500.0, 510.0]
        warnings = detect_increasing_spacing(spacings)
        assert len(warnings) == 1
        assert "WARNING" in warnings[0]
        assert "spacing appears to increase" in warnings[0]

    def test_slight_increase_below_threshold_no_warning(self) -> None:
        # under 15% increase is allowed
        spacings = [240.0, 240.0, 260.0, 265.0]
        warnings = detect_increasing_spacing(spacings)
        assert warnings == []

    def test_empty_spacings(self) -> None:
        assert detect_increasing_spacing([]) == []


class TestDetectLargeGaps:
    """Verify large-gap detection."""

    def test_no_gaps(self) -> None:
        events = [
            {"absolute_tick": 0.0, "delta_from_previous_note_on": 0.0},
            {"absolute_tick": 240.0, "delta_from_previous_note_on": 240.0},
        ]
        assert detect_large_gaps(events) == []

    def test_large_gap_detected(self) -> None:
        events = [
            {"absolute_tick": 0.0, "delta_from_previous_note_on": 0.0,
             "note": 42, "drum_name": "closed_hat", "velocity": 100, "channel": 9, "track": 0},
            {"absolute_tick": 200000.0, "delta_from_previous_note_on": 200000.0,
             "note": 42, "drum_name": "closed_hat", "velocity": 80, "channel": 9, "track": 0},
        ]
        warnings = detect_large_gaps(events)
        assert len(warnings) == 1
        assert "large timing gap" in warnings[0]


class TestCheckTempoPresent:
    """Verify tempo-present check."""

    def test_tempo_present(self) -> None:
        assert check_tempo_present(500000) == []

    def test_tempo_missing(self) -> None:
        warnings = check_tempo_present(None)
        assert len(warnings) == 1
        assert "no tempo meta message" in warnings[0]


class TestRunAllChecks:
    """Integration-level check for run_all_checks."""

    def test_clean_file(self) -> None:
        """A simple regular MIDI file should produce no warnings."""
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (240, 42, 80, 9),
                (240, 42, 80, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        warnings = run_all_checks(data)
        assert warnings == []

    def test_no_tempo_warning(self) -> None:
        path = _make_midi_file(tempo=None, note_events=[(0, 42, 100, 9)])
        data = extract_timing_data(path)
        os.unlink(path)

        warnings = run_all_checks(data)
        assert any("no tempo" in w for w in warnings)

    def test_increasing_spacing_warning(self) -> None:
        """
        Build a MIDI file where hi-hat spacing increases from ~240 to ~400.
        """
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (240, 42, 80, 9),
                (240, 42, 80, 9),
                (240, 42, 80, 9),
                (400, 42, 80, 9),
                (400, 42, 80, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        warnings = run_all_checks(data)
        assert any("spacing appears to increase" in w for w in warnings)


class TestEndToEndInspectFile:
    """Higher-level tests using actual MIDI file inspection."""

    def test_inspect_regular_hi_hat(self) -> None:
        """Regular hi-hat (note 42) every 240 ticks should have no increasing
        spacing warning and correct tick positions."""
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (240, 42, 80, 9),
                (240, 42, 85, 9),
                (240, 42, 80, 9),
                (240, 42, 90, 9),
                (240, 42, 80, 9),
                (240, 42, 85, 9),
                (240, 42, 80, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        assert data["note_on_count"] == 8
        # All hi-hat spacings should be exactly 240.0
        assert all(s == 240.0 for s in data["hi_hat_spacings"])

        # Check absolute ticks: 0, 240, 480, ...
        expected_ticks = [240.0 * i for i in range(8)]
        actual_ticks = [ev["absolute_tick"] for ev in data["note_on_events"]]
        assert actual_ticks == expected_ticks

    def test_inspect_increasing_spacing_detected(self) -> None:
        """When delta times increase, the tool should flag it."""
        path = _make_midi_file(
            note_events=[
                (0, 42, 100, 9),
                (240, 42, 80, 9),
                (240, 42, 80, 9),
                (240, 42, 80, 9),
                (500, 42, 80, 9),
                (500, 42, 80, 9),
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        warnings = run_all_checks(data)
        assert any("spacing appears to increase" in w for w in warnings)

    def test_drum_positions(self) -> None:
        """Verify kick and snare positions."""
        path = _make_midi_file(
            note_events=[
                (0, 36, 100, 9),  # kick
                (480, 38, 100, 9),  # snare
                (480, 36, 100, 9),  # kick
                (480, 38, 100, 9),  # snare
            ]
        )
        data = extract_timing_data(path)
        os.unlink(path)

        tick_pos = data["tick_positions"]
        assert tick_pos["kick_36"] == [0.0, 960.0]
        assert tick_pos["snare_38"] == [480.0, 1440.0]
