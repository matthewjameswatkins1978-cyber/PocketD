"""Pipeline-to-MIDI playback helper.

Converts ``GrooveEvent`` lists to timed MIDI messages and sends them
through the existing ``midi_out.MidiOut`` infrastructure.

Timing units
------------
* ``GrooveEvent.grid_position`` is in **16th-note subdivisions**.
    * 0 = beat 1
    * 4 = beat 2
    * 8 = beat 3
    * 12 = beat 4
    * 16 = beat 1 of next bar
* ``GrooveEvent.timing_offset_ms`` is an **offset in milliseconds**
  added to the grid position (legacy feel support).
* ``groove_events_to_midi_messages`` converts grid positions to
  wall-clock seconds using ``grid_position * (60 / bpm / 4)``.
* At 120 BPM: 1 beat = 0.5s, one bar = 2.0s.

Design contract
---------------
* Pure conversion helpers are testable without MIDI hardware.
* Playback requires an open ``MidiOut`` port.
* ``play_events_absolute`` uses absolute wall-clock scheduling for
  machine-tight timing — no accumulated drift.
"""

from __future__ import annotations

import time
from collections import OrderedDict

from midi_out import MidiOut, DRUM_CHANNEL
from drummer.feel import GrooveEvent, _instrument_group

# ---------------------------------------------------------------------------
# Note resolution — mirrors drummer/midi_export.py GM_DRUM_MAP
# ---------------------------------------------------------------------------

INSTRUMENT_TO_NOTE: dict[str, int] = {
    "kick": 36,
    "snare": 38,
    "hi_hat": 42,
    "closed_hat": 42,
    "open_hat": 46,
    "ride": 51,
    "crash": 49,
    "tom": 45,
    "toms": 45,
    "hi_tom": 48,
    "mid_tom": 45,
    "low_tom": 41,
    "rimshot": 37,
    "clap": 39,
    "cowbell": 56,
}


def resolve_note(evt: GrooveEvent) -> int | None:
    """Return the GM MIDI note number for a GrooveEvent, or None for unknown instruments.

    Falls back through instrument group resolution.
    """
    inst = evt.instrument.lower().replace(" ", "_")
    note = INSTRUMENT_TO_NOTE.get(inst)
    if note is not None:
        return note
    group = _instrument_group(evt.instrument)
    note = INSTRUMENT_TO_NOTE.get(group)
    if note is not None:
        return note
    return None


def beat_to_seconds(beat_position: float, bpm: float = 120.0) -> float:
    """Convert a beat position to wall-clock seconds.

    Parameters
    ----------
    beat_position : float
        Position in quarter-note beats.  0 = beat 1 of bar 1.
    bpm : float
        Tempo in beats per minute.

    Returns
    -------
    float
        Wall-clock seconds from time 0.
    """
    return beat_position * (60.0 / bpm)


def grid_to_seconds(grid_position: int, bpm: float = 120.0,
                    timing_offset_ms: float = 0.0) -> float:
    """Convert a 16th-note grid position to wall-clock seconds.

    Parameters
    ----------
    grid_position : int
        Position in 16th-note subdivisions.  0 = beat 1.
    bpm : float
        Tempo in beats per minute.
    timing_offset_ms : float
        Additional timing offset in milliseconds.

    Returns
    -------
    float
        Wall-clock seconds from time 0.
    """
    beat_duration = 60.0 / bpm
    sixteenth = beat_duration / 4.0
    return grid_position * sixteenth + timing_offset_ms / 1000.0


def groove_events_to_midi_messages(
    events: list[GrooveEvent],
    bpm: float = 120.0,
) -> list[tuple[float, int, int]]:
    """Convert GrooveEvents to ``(seconds, note, velocity)`` tuples.

    Parameters
    ----------
    events : list[GrooveEvent]
        Input drum events with grid_position in 16th-note units.
    bpm : float
        Tempo in beats per minute.

    Returns
    -------
    list[tuple[float, int, int]]
        Sorted ``(time_seconds, midi_note, velocity)`` messages.
        Unknown instruments are skipped.
    """
    messages: list[tuple[float, int, int]] = []
    for evt in events:
        note = resolve_note(evt)
        if note is None:
            continue
        time_sec = grid_to_seconds(evt.grid_position, bpm, evt.timing_offset_ms)
        messages.append((time_sec, note, evt.velocity))

    messages.sort(key=lambda x: x[0])
    return messages


def build_schedule(
    events: list[GrooveEvent],
    bpm: float = 120.0,
    note_duration: float = 0.09,
    repeats: int = 1,
) -> list[tuple[float, str, int, int]]:
    """Build an absolute-time schedule of (seconds, type, note, velocity).

    ``type`` is ``"on"`` or ``"off"``.

    ``seconds`` is wall-clock time from the start of playback.

    Parameters
    ----------
    events : list[GrooveEvent]
        Drum events.
    bpm : float
        Tempo in BPM.
    note_duration : float
        How long each note rings (seconds).
    repeats : int
        Number of bar repeats.

    Returns
    -------
    list[tuple[float, str, int, int]]
        Sorted schedule of ``(abs_seconds, "on"|"off", note, velocity)``.
    """
    if not events:
        return []

    messages = groove_events_to_midi_messages(events, bpm=bpm)
    if not messages:
        return []

    one_bar_duration = (60.0 / bpm) * 4.0

    schedule: list[tuple[float, str, int, int]] = []
    for rep in range(repeats):
        bar_offset = rep * one_bar_duration
        for msg_time, note, velocity in messages:
            abs_on = bar_offset + msg_time
            abs_off = abs_on + note_duration
            # Note: if note_off and note_on happen at the exact same time,
            # note_off goes first (sort key: type "off" < "on")
            schedule.append((abs_on, "on", note, velocity))
            schedule.append((abs_off, "off", note, 0))

    # Sort by time; note_off ("off") sorts before note_on ("on") at same time
    schedule.sort(key=lambda x: (x[0], x[1]))
    return schedule


def play_events_absolute(
    midi: MidiOut,
    events: list[GrooveEvent],
    bpm: float = 120.0,
    repeats: int = 1,
    note_duration: float = 0.09,
) -> list[tuple[float, float, str, int, int]]:
    """Play GrooveEvents with absolute wall-clock scheduling.

    Builds the entire schedule upfront, then dispatches each event
    at its absolute target time using ``time.perf_counter()``.

    Uses ``time.sleep()`` for > 2ms waits, then busy-loops for
    sub-2ms precision.  This is suitable for demo playback — not
    a production real-time audio engine.

    Returns a list of ``(target_time, actual_time, type, note, velocity)``
    for diagnostic timing analysis.
    """
    if not events:
        return []

    schedule = build_schedule(events, bpm=bpm, note_duration=note_duration,
                              repeats=repeats)
    if not schedule:
        return []

    drum_channel = 9
    timing_log: list[tuple[float, float, str, int, int]] = []

    start_time = time.perf_counter()
    for target_abs, msg_type, note, velocity in schedule:
        # Compute absolute target from the master start clock
        elapsed = time.perf_counter() - start_time
        wait = target_abs - elapsed

        if wait > 0.002:
            # Sleep until ~2ms before target
            time.sleep(wait - 0.002)
        # Busy-wait for the last 2ms
        while time.perf_counter() - start_time < target_abs:
            pass

        # Dispatch
        if msg_type == "on":
            midi.note_on(note, velocity)
        else:
            midi.note_off(note)

        actual = time.perf_counter() - start_time
        timing_log.append((target_abs, actual, msg_type, note, velocity))

    return timing_log


def print_timing_report(
    timing_log: list[tuple[float, float, str, int, int]],
    events_src: list[GrooveEvent] | None = None,
) -> None:
    """Print a human-readable timing diagnostic report."""
    if not timing_log:
        print("  (no events)")
        return

    errors_ms: list[float] = []
    drum_channel = 9

    print(f"\n  Timing Diagnostic Report:")
    print(f"  {'Idx':>4s}  {'Target':>8s}  {'Actual':>8s}  "
          f"{'Err(ms)':>8s}  {'Type':>4s}  {'Note':>5s}  {'Vel':>4s}  {'Inst'}")
    print(f"  {'-' * 4}  {'-' * 8}  {'-' * 8}  "
          f"{'-' * 8}  {'-' * 4}  {'-' * 5}  {'-' * 4}  {'-' * 15}")

    for i, (target, actual, msg_type, note, vel) in enumerate(timing_log):
        error_ms = (actual - target) * 1000.0
        errors_ms.append(error_ms)

        # Resolve instrument name
        inst = [k for k, v in INSTRUMENT_TO_NOTE.items() if v == note]
        inst_label = inst[0] if inst else f"note{note}"
        if msg_type == "off":
            inst_label = inst_label[:10] + "_off"

        print(f"  {i:4d}  {target:8.4f}  {actual:8.4f}  "
              f"{error_ms:+7.2f}  {msg_type:>4s}  {note:5d}  {vel:4d}  ch={drum_channel} {inst_label}")

    if errors_ms:
        abs_errors = [abs(e) for e in errors_ms]
        print(f"\n  Summary:")
        print(f"    Events: {len(timing_log)}")
        print(f"    Mean abs error: {sum(abs_errors)/len(abs_errors):.2f} ms")
        print(f"    Max abs error:  {max(abs_errors):.2f} ms")


# ---------------------------------------------------------------------------
# Legacy API — maintained for backward compatibility
# ---------------------------------------------------------------------------


def play_events(
    midi: MidiOut,
    events: list[GrooveEvent],
    bpm: float = 120.0,
    repeats: int = 1,
    note_duration: float = 0.09,
) -> None:
    """Play GrooveEvents using absolute scheduling (delegates to
    ``play_events_absolute``).  Prints the timing diagnostics report.
    """
    timing = play_events_absolute(midi, events, bpm=bpm, repeats=repeats,
                                  note_duration=note_duration)
    print_timing_report(timing, events)


def play_events_with_diagnostics(
    midi: MidiOut,
    events: list[GrooveEvent],
    bpm: float = 120.0,
    repeats: int = 1,
    note_duration: float = 0.09,
) -> None:
    """Play with full diagnostic output."""
    drum_channel = 9
    messages = groove_events_to_midi_messages(events, bpm=bpm)

    print(f"    MIDI channel: {drum_channel}")
    print(f"    Total messages (after conversion): {len(messages)}")
    print(f"    BPM: {bpm}")
    print(f"    Note duration: {note_duration * 1000:.0f}ms")
    print(f"    Repeats: {repeats}")

    if messages:
        print(f"    Scheduled note_on events:")
        for t, n, v in messages:
            inst = [k for k, val in INSTRUMENT_TO_NOTE.items() if val == n]
            inst_label = inst[0] if inst else f"note {n}"
            beat = t / (60.0 / bpm)
            print(f"      t={t:.4f}s (beat {beat:.2f})  note={n:3d} "
                  f"({inst_label})  vel={v:3d}")

    play_events(midi, events, bpm=bpm, repeats=repeats, note_duration=note_duration)


def list_available_ports() -> list[str]:
    """Return a list of available MIDI output port names."""
    from midi_out import list_output_ports
    return list_output_ports()


def find_or_none(name_substring: str) -> str | None:
    """Find a MIDI output port by substring, returning None if unavailable."""
    from midi_out import list_output_ports
    needle = name_substring.lower()
    for port in list_output_ports():
        if needle in port.lower():
            return port
    return None