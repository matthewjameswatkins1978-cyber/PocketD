"""MIDI Smoke Test — sends a simple kick/snare/hat pattern to verify MIDI output.

Target: ``PocketDrummer Out``

Run:
    python demo_midi_smoke_test.py [--port "PocketDrummer Out"] [--bpm 120]

If no port name is given, defaults to ``"PocketDrummer Out"``.
If the port is not found, lists available ports and exits.
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
from drummer.pipeline_midi import list_available_ports, find_or_none


# General MIDI drum notes (channel 10 → zero-indexed channel 9)
KICK = 36
SNARE = 38
CLOSED_HAT = 42
OPEN_HAT = 46
CRASH = 49
DRUM_CHANNEL = 9


def _simple_pattern(bpm: float = 120.0) -> list[tuple[float, int, int, str]]:
    """Return a simple 2-bar rock pattern as (seconds, note, velocity, instrument)."""
    beat = 60.0 / bpm  # quarter-note duration
    eighth = beat / 2.0

    pattern: list[tuple[float, int, int, str]] = []

    def add_bar(bar_start: float) -> None:
        t = bar_start
        pattern.append((t, KICK, 110, "kick"))
        pattern.append((t, CLOSED_HAT, 80, "closed_hat"))
        pattern.append((t + eighth, CLOSED_HAT, 65, "closed_hat"))
        pattern.append((t + beat, SNARE, 105, "snare"))
        pattern.append((t + beat, CLOSED_HAT, 80, "closed_hat"))
        pattern.append((t + beat + eighth, CLOSED_HAT, 65, "closed_hat"))
        pattern.append((t + beat * 2, KICK, 108, "kick"))
        pattern.append((t + beat * 2, CLOSED_HAT, 80, "closed_hat"))
        pattern.append((t + beat * 2 + eighth, CLOSED_HAT, 65, "closed_hat"))
        pattern.append((t + beat * 3, SNARE, 107, "snare"))
        pattern.append((t + beat * 3, CLOSED_HAT, 80, "closed_hat"))
        pattern.append((t + beat * 3 + eighth, OPEN_HAT, 75, "open_hat"))

    add_bar(0.0)
    add_bar(beat * 4)
    pattern.append((beat * 8 - 0.01, CRASH, 100, "crash"))
    pattern.sort(key=lambda x: x[0])
    return pattern


def main() -> int:
    parser = argparse.ArgumentParser(description="MIDI Smoke Test")
    parser.add_argument("--port", type=str, default="PocketDrummer Out",
                        help="MIDI output port name (default: PocketDrummer Out)")
    parser.add_argument("--bpm", type=float, default=120.0)
    args = parser.parse_args()

    ports = list_available_ports()
    print(f"Target port: '{args.port}'")
    print(f"Available MIDI output ports:")
    if not ports:
        print("  (none)")
        print("\nNo MIDI output ports found. Please ensure EZDrummer or loopMIDI")
        print("is running and has the 'PocketDrummer Out' port configured.")
        return 1
    for p in ports:
        print(f"  - {p}")

    port_name = find_or_none(args.port)
    if port_name is None:
        print(f"\nPort '{args.port}' not found in available ports.")
        return 1

    print(f"\nOpening MIDI output: {port_name}")
    midi = MidiOut(port_name)
    midi.open()
    bpm = args.bpm
    note_duration = 0.09  # ~90ms

    pattern = _simple_pattern(bpm)

    print(f"\nSmoke test pattern ({bpm} BPM, {len(pattern)} hits):")
    print(f"  Channel: {DRUM_CHANNEL} (zero-indexed, GM drum channel 10)")
    print(f"  Note duration: {note_duration * 1000:.0f}ms")
    print(f"  {'Time':>7s}  {'Note':>5s}  {'Vel':>4s}  Instrument")
    print(f"  {'-' * 7}  {'-' * 5}  {'-' * 4}  {'-' * 15}")
    for t, note, vel, inst in pattern:
        print(f"  {t:7.3f}s  {note:5d}  {vel:4d}  {inst}")

    print(f"\nPlaying pattern (2 bars, {len(pattern)} hits)...")

    start_time = time.perf_counter()
    try:
        for t, note, vel, inst in pattern:
            # Sleep until this note's scheduled time
            elapsed = time.perf_counter() - start_time
            delay = t - elapsed
            if delay > 0:
                time.sleep(delay)

            midi.note_on(note, vel)
            print(f"  ON  note={note:3d}  vel={vel:3d}  ch={DRUM_CHANNEL}  "
                  f"t={t:.3f}s  [{inst}]")
            time.sleep(note_duration)
            midi.note_off(note)
    except KeyboardInterrupt:
        print("\n\nStopped by user.")

    # Let final notes ring
    time.sleep(0.3)
    midi.close()
    print("MIDI port closed.")
    print("\nSmoke test complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())