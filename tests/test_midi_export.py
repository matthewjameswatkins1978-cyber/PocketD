"""Tests for drummer/midi_export.py — MIDI file export of GrooveEvent sequences."""

from __future__ import annotations

import os
import tempfile

from mido import MidiFile

from drummer.feel import GrooveEvent
from drummer.midi_export import (
    GM_DRUM_MAP,
    export_groove_events_to_midi,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_event(
    instrument: str = "kick",
    grid_position: int = 0,
    velocity: int = 100,
    timing_offset_ms: float = 0.0,
    bar_index: int = 0,
    source_role: str = "main",
) -> GrooveEvent:
    """Helper to build a GrooveEvent with minimal boilerplate."""
    return GrooveEvent(
        instrument=instrument,
        grid_position=grid_position,
        bar_index=bar_index,
        velocity=velocity,
        probability=1.0,
        timing_offset_ms=timing_offset_ms,
        articulation="default",
        source_role=source_role,
    )


def _replay_midi_track(midi_path: str) -> list[dict]:
    """Read a MIDI file and return a list of (message_type, note, delta_time, velocity)
    for channel-10 (drum) messages, in order."""
    midi = MidiFile(midi_path)
    events: list[dict] = []
    for track in midi.tracks:
        for msg in track:
            if msg.type in ("note_on", "note_off") and msg.channel == 9:
                events.append({
                    "type": msg.type,
                    "note": msg.note,
                    "time": msg.time,
                    "velocity": msg.velocity,
                })
    return events


# ---------------------------------------------------------------------------
# Basic file creation
# ---------------------------------------------------------------------------


def test_export_creates_file() -> None:
    """export_groove_events_to_midi should create a non-empty MIDI file."""
    events = [
        _make_event("kick", 0, 100),
        _make_event("snare", 4, 100),
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        assert os.path.exists(tmp_path), "MIDI file was not created"
        assert os.path.getsize(tmp_path) > 0, "MIDI file is empty"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_export_non_empty() -> None:
    """A single event should produce a non-empty MIDI file."""
    events = [_make_event("kick", 0, 100)]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        file_size = os.path.getsize(tmp_path)
        assert file_size > 0, f"Expected file size > 0 bytes, got {file_size}"
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Negative timing offsets are clamped safely
# ---------------------------------------------------------------------------


def test_negative_timing_offset_clamped() -> None:
    """Exported MIDI file should not fail when events have large negative offsets."""
    events = [
        _make_event("kick", 0, 100, timing_offset_ms=-200.0),
        _make_event("snare", 4, 100, timing_offset_ms=-500.0),
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Unknown instruments map to safe default
# ---------------------------------------------------------------------------


def test_unknown_instrument_uses_default() -> None:
    """An instrument not in GM_DRUM_MAP should fall back to a sensible default."""
    events = [_make_event("completely_bogus_instrument", 0, 100)]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Known GM drum notes produce expected note numbers
# ---------------------------------------------------------------------------


def test_known_instruments_map_correctly() -> None:
    """All standard GM drum instruments in GM_DRUM_MAP should produce valid note_on events."""
    events = []
    for idx, inst in enumerate(GM_DRUM_MAP):
        events.append(_make_event(inst, idx * 4, 100))

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        midi = MidiFile(str(tmp_path))
        note_numbers: set[int] = set()
        for track in midi.tracks:
            for msg in track:
                if msg.type == "note_on":
                    note_numbers.add(msg.note)

        for expected_note in set(GM_DRUM_MAP.values()):
            assert expected_note in note_numbers, (
                f"Expected GM note {expected_note} not found in exported MIDI"
            )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Empty events list
# ---------------------------------------------------------------------------


def test_empty_events_creates_valid_file() -> None:
    """An empty events list should produce a valid MIDI file (no notes)."""
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi([], tmp_path, tempo_bpm=120)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tempo parameter
# ---------------------------------------------------------------------------


def test_tempo_parameter() -> None:
    """Different tempo should produce valid files for the same events."""
    events = [
        _make_event("kick", 0, 100),
        _make_event("snare", 4, 100),
    ]

    for bpm in (60, 120, 200):
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            export_groove_events_to_midi(events, tmp_path, tempo_bpm=bpm)
            assert os.path.getsize(tmp_path) > 0, f"File empty at {bpm} BPM"
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Many events
# ---------------------------------------------------------------------------


def test_many_events() -> None:
    """A full bar of 16th-note events across multiple bars should export cleanly."""
    events = []
    for bar in range(4):
        for pos in range(16):
            events.append(_make_event("kick", pos, 64, bar_index=bar))

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        assert os.path.getsize(tmp_path) > 0
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


# ===================================================================
# Timing correctness tests  (delta times, not absolute ticks)
# ===================================================================


def test_deltas_do_not_accumulate() -> None:
    """Delta times should represent gaps between consecutive events, not absolute ticks.

    For two events at ticks 0 and 120 on a clean grid, the deltas should be
    0 (first note_on), 60 (note_off at tick 60), 60 (note_on at tick 120),
    etc. — the sum of deltas between a note_on and its note_off should be
    the duration, and the next note_on delta should be the gap since the
    previous event, not an accumulated total.
    """
    events = [
        _make_event("kick", 0, 100),   # grid pos 0 -> tick 0
        _make_event("kick", 1, 100),   # grid pos 1 -> tick 120
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        replay = _replay_midi_track(tmp_path)

        # We should have 4 events: note_on(0), note_off(0), note_on(1), note_off(1)
        assert len(replay) == 4, f"Expected 4 events, got {len(replay)}"

        # Check deltas are reasonable, not summing to 240+ in one delta
        for entry in replay:
            assert entry["time"] >= 0

        # The sum of all deltas should be <= total span + duration
        total_span = (1 * 120) + 60  # last note_on at tick 120 + duration 60 = 180
        sum_deltas = sum(e["time"] for e in replay)
        assert sum_deltas <= total_span, (
            f"Delta sum {sum_deltas} exceeds total span {total_span} — "
            f"likely using absolute ticks instead of deltas"
        )

        # The note_on for the second kick should have delta near 60 ticks
        # (note_off of first note at tick 60, then note_on of second at tick 120 -> delta 60)
        note_on_events = [e for e in replay if e["type"] == "note_on"]
        assert len(note_on_events) == 2
        # delta for second note_on
        assert note_on_events[1]["time"] <= 120, (
            f"Second note_on delta {note_on_events[1]['time']} is too large, "
            f"expected ~60 (120 - 60)"
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_regular_16th_note_spacing() -> None:
    """A 16th-note hi-hat pattern should have regularly spaced note_on deltas.

    With 8 hi-hat hits at positions 0,2,4,6,8,10,12,14 and 120 ticks/16th,
    the note_on events should have alternating deltas of ~60 (gap from
    previous note_off to next note_on) and the note_off deltas should be ~60.
    The key check: no single delta should accumulate multiple 16th notes.
    """
    events = [
        _make_event("hi_hat", pos, 80)
        for pos in (0, 2, 4, 6, 8, 10, 12, 14)
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        replay = _replay_midi_track(tmp_path)

        note_on_deltas = [e["time"] for e in replay if e["type"] == "note_on"]
        # There should be 8 note_on events
        assert len(note_on_deltas) == 8, f"Expected 8 note_on events, got {len(note_on_deltas)}"

        # First note_on delta is always 0 (at start of track)
        assert note_on_deltas[0] == 0

        # For a perfect 16th-note grid at 480 TPQN:
        #   note_on at t=0,   note_off at t=60  (delta 60)
        #   note_on at t=240, note_off at t=300 (delta 180 from previous note_off)
        # Actually at positions 0,2,4,6,8,10,12,14:
        #   pos 0 -> abs tick 0
        #   pos 2 -> abs tick 240
        #   pos 4 -> abs tick 480
        #   pos 6 -> abs tick 720
        #   etc.
        # Events sorted: note_on(0), note_off(60), note_on(240), note_off(300),
        #                note_on(480), note_off(540), ...
        # Deltas: 0, 60, 180, 60, 180, 60, 180, ...
        # So note_on deltas (indices 0,2,4,6,8,10,12,14 of the sorted list):
        #   [0, 180, 180, 180, 180, 180, 180, 180]
        for i in range(1, len(note_on_deltas)):
            # Each subsequent note_on should have delta ~180 (not > 200)
            assert 160 <= note_on_deltas[i] <= 200, (
                f"note_on delta[{i}] = {note_on_deltas[i]}, expected ~180 for "
                f"16th-note spacing. Deltas: {note_on_deltas}"
            )

        # No single delta should exceed a 1-beat span (480 ticks) —
        # if absolute ticks were used, a delta would be the full 240.
        max_delta = max(e["time"] for e in replay)
        assert max_delta < 480, (
            f"Max delta {max_delta} exceeds 1 beat (480 ticks) — "
            f"likely using absolute ticks instead of deltas"
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_simultaneous_events_have_delta_zero() -> None:
    """When multiple events share the same absolute tick, the first gets the
    delta and subsequent events at the same tick get delta=0."""
    events = [
        _make_event("kick", 0, 100),   # abs tick 0
        _make_event("snare", 0, 90),   # abs tick 0 (same position)
        _make_event("hi_hat", 0, 80),  # abs tick 0 (same position)
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        replay = _replay_midi_track(tmp_path)

        # The track should start with note_ons at tick 0
        # After sorting: note_on(kick,0), note_on(snare,0), note_on(hi_hat,0),
        #                note_off(kick,60), note_off(snare,60), note_off(hi_hat,60)
        # Deltas: 0, 0, 0, 60, 0, 0
        expected_deltas = [0, 0, 0, 60, 0, 0]
        actual_deltas = [e["time"] for e in replay]

        assert len(actual_deltas) == 6, f"Expected 6 events, got {len(actual_deltas)}"
        for i, (actual, expected) in enumerate(zip(actual_deltas, expected_deltas)):
            assert actual == expected, (
                f"Event {i}: expected delta={expected}, got {actual}. "
                f"All deltas: {actual_deltas}"
            )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_note_off_does_not_delay_note_on() -> None:
    """A note_off event should not shift subsequent note_on timing."""
    events = [
        _make_event("kick", 0, 100),  # abs tick 0
        _make_event("snare", 4, 100),  # abs tick 480
    ]

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        replay = _replay_midi_track(tmp_path)

        # Events: note_on(k,0), note_off(k,60), note_on(s,480), note_off(s,540)
        # Deltas: 0, 60, 420, 60
        # So the note_on for snare should have time=420, not some larger value
        note_on_events = [e for e in replay if e["type"] == "note_on"]
        assert len(note_on_events) == 2

        assert note_on_events[0]["time"] == 0, "First note_on should have delta=0"
        assert note_on_events[1]["time"] == 420, (
            f"Snare note_on delta should be 420 (480 - 60), got {note_on_events[1]['time']}"
        )

        # Also verify the sum of all deltas equals the last absolute tick
        # last event: note_off at tick 540
        total_last_tick = sum(e["time"] for e in replay)
        assert total_last_tick == 540, f"Sum of deltas should be 540, got {total_last_tick}"

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def test_tempo_does_not_appear_to_slow() -> None:
    """A 4-bar quarter-note kick pattern should have consistent spacing.

    The kick plays on each quarter note (positions 0,4,8,12,16,...).
    At 120 BPM with 480 TPQN, adjacent kicks are 480 ticks apart.
    The delta times between note_on events should be ~480, not
    increasing due to accumulated note_off deltas.
    """
    events = []
    for bar in range(4):
        for quarter in range(4):
            pos = bar * 16 + quarter * 4
            events.append(_make_event("kick", pos, 100))

    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        export_groove_events_to_midi(events, tmp_path, tempo_bpm=120)
        replay = _replay_midi_track(tmp_path)

        # Extract note_on deltas (skip the first which is always 0)
        note_on_deltas = [
            e["time"] for e in replay if e["type"] == "note_on"
        ]

        assert len(note_on_deltas) == 16, f"Expected 16 note_on events, got {len(note_on_deltas)}"
        assert note_on_deltas[0] == 0

        # For a quarter-note pattern with 60-tick durations:
        #   note_on(t=0)    , note_off(t=60)
        #   note_on(t=480)  , note_off(t=540)
        #   note_on(t=960)  , note_off(t=1020)
        #   ...
        # Deltas for note_on: [0, 420, 420, 420, ...]
        # (480 - 60 = 420 from previous note_off's tick)
        for i in range(1, len(note_on_deltas)):
            actual = note_on_deltas[i]
            # Allow small rounding variance
            assert 410 <= actual <= 430, (
                f"note_on delta[{i}] = {actual}, expected ~420. "
                f"All note_on deltas: {note_on_deltas}"
            )

        # Also verify no delta exceeds 480 (one beat) — if absolute ticks
        # leaked in, a delta would be 480+ (e.g., 960, 1440...)
        all_deltas = [e["time"] for e in replay]
        assert max(all_deltas) <= 540, (
            f"Max delta {max(all_deltas)} exceeds max expected 540 — "
            f"likely absolute ticks leaking into deltas"
        )

    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)