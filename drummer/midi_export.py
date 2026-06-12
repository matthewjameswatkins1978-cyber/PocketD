"""MIDI file export for GrooveEvent sequences.

Provides a single convenience function ``export_groove_events_to_midi`` that
writes processed drum events to a Standard MIDI File (SMF Type 0) using the
``mido`` library.

Usage::

    from drummer.midi_export import export_groove_events_to_midi

    events: list[GrooveEvent] = ...
    export_groove_events_to_midi(events, "groove.mid", tempo_bpm=120)
"""

from __future__ import annotations

from pathlib import Path

from mido import MidiFile, MidiTrack, MetaMessage, Message

from drummer.feel import GrooveEvent

# ---------------------------------------------------------------------------
# General MIDI drum note mapping  (GM standard key numbers)
# ---------------------------------------------------------------------------
GM_DRUM_MAP: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hi_hat": 42,
    "closed_hat": 42,
    "open_hat": 46,
    "ride": 51,
    "crash": 49,
    "tom": 45,  # generic tom -> mid tom
    "toms": 45,
    "hi_tom": 48,
    "mid_tom": 45,
    "low_tom": 41,
    "rimshot": 37,
    "clap": 39,
    "cowbell": 56,
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_note(instrument: str) -> int:
    """Resolve a GM drum note number for an instrument name.

    Falls back through instrument group resolution, then to kick (36).
    """
    note = GM_DRUM_MAP.get(instrument)
    if note is not None:
        return note
    from drummer.feel import _instrument_group

    group = _instrument_group(instrument)
    note = GM_DRUM_MAP.get(group)
    if note is not None:
        return note
    return GM_DRUM_MAP["kick"]


# ---------------------------------------------------------------------------
# Export function
# ---------------------------------------------------------------------------


def export_groove_events_to_midi(
    events: list[GrooveEvent],
    output_path: str | Path,
    tempo_bpm: float = 120.0,
    ticks_per_beat: int = 480,
) -> None:
    """Write a sequence of ``GrooveEvent`` s to a MIDI file.

    Parameters
    ----------
    events : list[GrooveEvent]
        Processed drum events to export.
    output_path : str | Path
        Destination file path (e.g. ``"feel_machine.mid"``).
    tempo_bpm : float, optional
        Tempo in beats per minute (default 120).
    ticks_per_beat : int, optional
        MIDI ticks per quarter note (default 480).

    Notes
    -----
    * All events are placed on MIDI channel 10 (zero-indexed channel 9).
    * Negative tick positions (caused by early timing offsets) are clamped to 0.
    * Each note has a fixed duration of 60 ticks (~1/8th note at 480 TPQN).
    * Unknown instrument names map to a safe default note (kick, 36).
    * MIDI message times are written as delta times (not absolute ticks), which
      is what the SMF specification requires.
    """
    # --- Constants ---
    MIDI_DRUM_CHANNEL = 9  # zero-indexed; channel 10 in GM
    NOTE_DURATION_TICKS = 60
    SIXTEENTH_TICKS = ticks_per_beat // 4  # 120 at 480 TPQN

    # Precompute tick-per-ms factor at this tempo
    # 1 quarter note = ticks_per_beat
    # 1 quarter note = 60 000 / tempo_bpm  ms
    # -> 1 ms = ticks_per_beat / (60 000 / tempo_bpm)  ticks
    ms_to_ticks = (tempo_bpm * ticks_per_beat) / 60_000.0

    # ------------------------------------------------------------------
    # Step 1: build all MIDI events with absolute tick positions
    # ------------------------------------------------------------------
    midi_events: list[tuple[int, str, int, int]] = []
    # Each tuple: (absolute_tick, type, note, velocity)
    # type: "note_on" or "note_off"

    for ev in events:
        note = _resolve_note(ev.instrument)

        # Absolute tick from grid position + timing offset
        base_tick = ev.grid_position * SIXTEENTH_TICKS
        tick_offset = round(ev.timing_offset_ms * ms_to_ticks)
        abs_tick = base_tick + tick_offset
        abs_tick = max(0, abs_tick)  # clamp negative to 0

        midi_events.append((abs_tick, "note_on", note, ev.velocity))
        midi_events.append((abs_tick + NOTE_DURATION_TICKS, "note_off", note, 0))

    # ------------------------------------------------------------------
    # Step 2: sort all events by absolute tick
    # ------------------------------------------------------------------
    midi_events.sort(key=lambda x: x[0])

    # ------------------------------------------------------------------
    # Step 3: build the MIDI file with delta times
    # ------------------------------------------------------------------
    midi = MidiFile(ticks_per_beat=ticks_per_beat)
    track = MidiTrack()
    midi.tracks.append(track)

    # Set tempo meta event (microseconds per quarter note)
    us_per_qn = int(60_000_000.0 / tempo_bpm)
    track.append(MetaMessage("set_tempo", tempo=us_per_qn))

    previous_tick = 0
    for abs_tick, msg_type, note, velocity in midi_events:
        delta = abs_tick - previous_tick
        previous_tick = abs_tick

        if msg_type == "note_on":
            track.append(
                Message(
                    "note_on",
                    note=note,
                    velocity=velocity,
                    time=delta,
                    channel=MIDI_DRUM_CHANNEL,
                )
            )
        else:  # note_off
            track.append(
                Message(
                    "note_off",
                    note=note,
                    velocity=velocity,
                    time=delta,
                    channel=MIDI_DRUM_CHANNEL,
                )
            )

    # End-of-track meta event
    track.append(MetaMessage("end_of_track"))

    # Write file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(output_path))