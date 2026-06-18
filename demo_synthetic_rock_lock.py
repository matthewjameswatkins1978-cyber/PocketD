"""Demo: synthetic pulse in, simple rock beat out.

This is the smallest proof that the current modules can work together:

1. Generate a synthetic 4/4 quarter-note pulse at a known BPM.
2. Detect those pulse events as onsets.
3. Estimate the BPM from the onset timings.
4. Load ``simple_rock`` from ``data/grooves.yaml``.
5. Schedule that groove at the estimated BPM, with optional MIDI playback.

Run dry:
    python demo_synthetic_rock_lock.py --bpm 118

Play over MIDI:
    python demo_synthetic_rock_lock.py --bpm 118 --port "PocketDrummer Out"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from drummer.feel import GrooveEvent
from drummer.pipeline_midi import (
    build_schedule,
    find_or_none,
    list_available_ports,
    play_events_with_diagnostics,
)
from groove_library import get_groove
from midi_out import MidiOut
from models import Groove
from pulse_tracker import estimate_tempo


def estimate_bpm_from_synthetic_pulse(
    input_bpm: float,
    duration_seconds: float = 8.0,
) -> tuple[float, int]:
    """Generate synthetic pulses and estimate tempo from detected onsets."""
    pulse_events = generate_pulse_events(
        bpm=input_bpm,
        duration_seconds=duration_seconds,
    )
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)
    estimated_bpm = estimate_tempo([event.time_seconds for event in onsets])
    return estimated_bpm, len(onsets)


def groove_to_events(groove: Groove) -> list[GrooveEvent]:
    """Convert a library Groove definition into playable GrooveEvents."""
    events: list[GrooveEvent] = []

    for step in groove.kick_steps:
        events.append(GrooveEvent("kick", step, velocity=104))

    for step in groove.snare_steps:
        events.append(GrooveEvent("snare", step, velocity=108))

    for step in groove.hat_steps:
        velocity = 78 if step % 8 == 0 else 68
        events.append(GrooveEvent("hi_hat", step, velocity=velocity))

    events.sort(key=lambda event: (event.bar_index, event.grid_position, event.instrument))
    return events


def print_dry_run_report(
    groove_id: str,
    input_bpm: float,
    estimated_bpm: float,
    onset_count: int,
    events: list[GrooveEvent],
    repeats: int,
) -> None:
    """Print the timing proof without requiring MIDI hardware."""
    bpm_error = estimated_bpm - input_bpm
    bar_seconds = (60.0 / estimated_bpm) * 4.0
    schedule = build_schedule(events, bpm=estimated_bpm, repeats=repeats)

    print("\nSynthetic rock lock demo")
    print("=" * 48)
    print(f"Input pulse:        {input_bpm:.1f} BPM, 4/4 quarter-note clicks")
    print(f"Detected onsets:    {onset_count}")
    print(f"Estimated tempo:    {estimated_bpm:.1f} BPM ({bpm_error:+.1f})")
    print(f"Selected groove:    {groove_id}")
    print(f"Bar duration:       {bar_seconds:.3f}s")
    print(f"Scheduled notes:    {len(schedule) // 2} note-ons over {repeats} bars")
    print("\nFirst-bar note-ons:")

    for seconds, message_type, note, velocity in schedule:
        if message_type != "on" or seconds >= bar_seconds:
            continue
        beat_position = seconds / (60.0 / estimated_bpm) + 1.0
        print(
            f"  t={seconds:6.3f}s  beat={beat_position:4.2f}  "
            f"note={note:2d}  velocity={velocity:3d}"
        )


def play_over_midi(
    port_query: str,
    events: list[GrooveEvent],
    estimated_bpm: float,
    repeats: int,
) -> int:
    """Play the generated rock groove to a matching MIDI output port."""
    ports = list_available_ports()
    if not ports:
        print("No MIDI output ports found.")
        return 1

    port_name = find_or_none(port_query)
    if port_name is None:
        print(f"No MIDI output port matching '{port_query}'.")
        print("Available ports:")
        for port in ports:
            print(f"  - {port}")
        return 1

    print(f"\nOpening MIDI output: {port_name}")
    with MidiOut(port_name) as midi:
        play_events_with_diagnostics(
            midi,
            events,
            bpm=estimated_bpm,
            repeats=repeats,
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate synthetic pulses, estimate BPM, and play simple rock.",
    )
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--duration", type=float, default=8.0)
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--groove", type=str, default="simple_rock")
    parser.add_argument(
        "--port",
        type=str,
        default=None,
        help="Optional MIDI output port name/substr. Omit for dry run.",
    )
    args = parser.parse_args()

    estimated_bpm, onset_count = estimate_bpm_from_synthetic_pulse(
        input_bpm=args.bpm,
        duration_seconds=args.duration,
    )
    groove = get_groove(args.groove)
    events = groove_to_events(groove)

    print_dry_run_report(
        groove_id=groove.id,
        input_bpm=args.bpm,
        estimated_bpm=estimated_bpm,
        onset_count=onset_count,
        events=events,
        repeats=args.repeats,
    )

    if args.port is None:
        print("\nDry run only. Add --port to hear it over MIDI.")
        return 0

    return play_over_midi(
        port_query=args.port,
        events=events,
        estimated_bpm=estimated_bpm,
        repeats=args.repeats,
    )


if __name__ == "__main__":
    raise SystemExit(main())
