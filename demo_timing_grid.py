"""Machine-Tight Timing Grid Demo.

Bypasses all behaviour/pipeline logic and sends a dead-simple 2-bar
rock pattern directly to MIDI with absolute wall-clock scheduling.

Run:
    python demo_timing_grid.py [--port "PocketDrummer Out"] [--bpm 120]

Purpose:
    Prove the MIDI scheduler can play machine-tight.
    If this demo is tight but pipeline demo is sloppy, the problem is
    in GrooveEvent generation, output shaping, or humanization.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midi_out import MidiOut
from drummer.pipeline_midi import (
    list_available_ports,
    find_or_none,
    beat_to_seconds,
)
from drummer.feel import GrooveEvent

# General MIDI drum notes (channel 10 -> zero-indexed channel 9)
KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
DRUM_CHANNEL = 9


def _build_grid_pattern(bpm: float = 120.0) -> tuple[
    list[GrooveEvent],
    list[tuple[float, float, str, int]],
]:
    """Build a 2-bar machine-tight rock pattern.

    Returns
    -------
    events : list[GrooveEvent]
        The GrooveEvent pattern (for display/reference).
    expected_schedule : list[tuple[float, float, str, int]]
        Expected (beat_position, seconds, instrument, MIDI note).
    """
    beat_dur = 60.0 / bpm  # seconds per quarter note
    eighth_dur = beat_dur / 2.0

    events: list[GrooveEvent] = []
    expected: list[tuple[float, float, str, int]] = []

    def add(beat: float, instrument: str, note: int, vel: int = 100) -> None:
        """Add a note at a beat position.

        beat: quarter-note beat position (0 = beat 1)
        """
        # Convert beat position to 16th-note grid position
        grid_pos = int(round(beat * 4))
        sec = beat * beat_dur
        events.append(GrooveEvent(instrument, grid_pos, velocity=vel))
        expected.append((beat, sec, instrument, note))

    # Bar 1
    add(0.0, "kick", KICK, 110)        # Beat 1
    add(0.0, "hi_hat", CLOSED_HAT, 80)
    add(0.5, "hi_hat", CLOSED_HAT, 65)  # 8th off
    add(1.0, "snare", SNARE, 105)      # Beat 2
    add(1.0, "hi_hat", CLOSED_HAT, 80)
    add(1.5, "hi_hat", CLOSED_HAT, 65)
    add(2.0, "kick", KICK, 108)        # Beat 3
    add(2.0, "hi_hat", CLOSED_HAT, 80)
    add(2.5, "hi_hat", CLOSED_HAT, 65)
    add(3.0, "snare", SNARE, 107)      # Beat 4
    add(3.0, "hi_hat", CLOSED_HAT, 80)
    add(3.5, "hi_hat", OPEN_HAT, 75)

    # Bar 2
    add(4.0, "kick", KICK, 110)
    add(4.0, "hi_hat", CLOSED_HAT, 80)
    add(4.5, "hi_hat", CLOSED_HAT, 65)
    add(5.0, "snare", SNARE, 105)
    add(5.0, "hi_hat", CLOSED_HAT, 80)
    add(5.5, "hi_hat", CLOSED_HAT, 65)
    add(6.0, "kick", KICK, 108)
    add(6.0, "hi_hat", CLOSED_HAT, 80)
    add(6.5, "hi_hat", CLOSED_HAT, 65)
    add(7.0, "snare", SNARE, 107)
    add(7.0, "hi_hat", CLOSED_HAT, 80)
    add(7.5, "hi_hat", OPEN_HAT, 75)

    # Crash on first beat
    add(0.0, "crash", CRASH, 90)

    # Sort by grid_position
    events.sort(key=lambda e: e.grid_position)
    expected.sort(key=lambda x: (x[0], x[3]))

    return events, expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Machine-Tight Timing Grid Demo")
    parser.add_argument("--port", type=str, default="PocketDrummer Out")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--no-play", action="store_true",
                        help="Print schedule only, do not play")
    args = parser.parse_args()

    ports = list_available_ports()
    print(f"Target port: '{args.port}'")
    print(f"Available MIDI output ports:")
    if not ports:
        print("  (none)")
        if not args.no_play:
            print("\nNo MIDI output ports found.")
            return 1
    for p in ports:
        print(f"  - {p}")

    if not args.no_play:
        port_name = find_or_none(args.port)
        if port_name is None:
            print(f"\nPort '{args.port}' not found.")
            return 1
        print(f"\nOpening MIDI output: {port_name}")

    bpm = args.bpm
    beat_dur = 60.0 / bpm
    note_duration = 0.09  # 90ms drum hits
    events, expected = _build_grid_pattern(bpm)

    # Print schedule
    print(f"\nMachine-Tight Timing Grid ({bpm} BPM, 2 bars):")
    print(f"  Beat duration: {beat_dur:.4f}s")
    print(f"  Note duration: {note_duration * 1000:.0f}ms")
    print(f"  One bar: 0.0 to {beat_dur * 4:.3f}s")
    print()
    print(f"  {'Beat':>6s}  {'Sec':>7s}  {'Instrument':>10s}  MIDI Note")
    print(f"  {'-' * 6}  {'-' * 7}  {'-' * 10}  {'-' * 9}")
    for beat, sec, inst, note in expected:
        print(f"  {beat:6.2f}  {sec:7.4f}  {inst:>10s}  {note:4d}")

    if args.no_play:
        print("\n  (--no-play: schedule printed, exiting)")
        return 0

    # Playback
    midi = MidiOut(port_name)
    midi.open()

    print(f"\n  Playing 2 bars with absolute scheduling...")
    start = time.perf_counter()

    # Build the full schedule manually for direct control
    # Each note becomes one "on" event and one "off" event
    schedule: list[tuple[float, str, str, int, int]] = []
    for evt in events:
        sec = evt.grid_position * (beat_dur / 4.0)
        inst = evt.instrument
        note = {
            "kick": KICK, "snare": SNARE, "hi_hat": CLOSED_HAT,
            "closed_hat": CLOSED_HAT, "open_hat": OPEN_HAT,
            "crash": CRASH,
        }.get(inst, 0)
        if note == 0:
            continue
        schedule.append((sec, "on", inst, note, evt.velocity))
        schedule.append((sec + note_duration, "off", inst, note, 0))

    schedule.sort(key=lambda x: (x[0], 0 if x[1] == "off" else 1))

    errors_ms: list[float] = []
    for target_sec, event_type, inst, note, vel in schedule:
        elapsed = time.perf_counter() - start
        wait = target_sec - elapsed
        if wait > 0.002:
            time.sleep(wait - 0.002)
        while time.perf_counter() - start < target_sec:
            pass

        actual = time.perf_counter() - start
        error_ms = (actual - target_sec) * 1000.0
        errors_ms.append(abs(error_ms))

        if event_type == "on":
            midi.note_on(note, vel)
        else:
            midi.note_off(note)

        # Print every event with timing error
        marker = ""
        if abs(error_ms) > 5.0:
            marker = " <-- LARGE ERROR!"
        print(f"  t={target_sec:7.4f}s  {event_type:>4s}  {inst:>10s}  "
              f"note={note:4d}  vel={vel:3d}  err={error_ms:+6.2f}ms{marker}")

    time.sleep(0.3)  # let final notes ring
    midi.close()

    print(f"\n  Summary:")
    print(f"    Events: {len(schedule)}")
    print(f"    Mean abs error: {sum(errors_ms)/len(errors_ms):.2f} ms")
    print(f"    Max abs error:  {max(errors_ms):.2f} ms")
    print(f"    {'OK - machine-tight!' if max(errors_ms) < 10 else 'Room for improvement'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())