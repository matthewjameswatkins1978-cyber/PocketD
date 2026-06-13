"""Demo: Pipeline MIDI Playback — hear the drummer brain.

Simulates the same behavioural paths as demo_pipeline.py but sends
shaped GrooveEvents to a MIDI output port for audible comparison.

Run:
    python demo_pipeline_midi.py [--port "PocketDrummer Out"] [--bpm 120]

Defaults to ``"PocketDrummer Out"``.  If no --port is given or the port
is not found, lists available ports and exits.
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
from drummer.pipeline import DrummerBrainPipeline
from drummer.pipeline_midi import (
    play_events_with_diagnostics,
    list_available_ports,
    find_or_none,
    groove_events_to_midi_messages,
)
from perception.models import MusicalEvent


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _enter_pipeline(pipeline: DrummerBrainPipeline) -> None:
    """Feed enough stable events to enter, processing each cycle."""
    for i in range(6):
        t = i * 0.5
        pipeline.feed_event(MusicalEvent(t, 0.7))
        pipeline.process(now=t, phase_alignment=0.75)


def _report(scenario: str, d) -> None:
    snap = d.feature_snapshot
    print(f"\n  [{scenario}]")
    print(f"    intent={d.behaviour_intent.value}"
          f"  density={snap.input_density:.2f}"
          f"  certainty={snap.player_certainty:.2f}"
          f"  change={snap.change_score:.2f}"
          f"  raw_events={len(d.raw_events)}"
          f"  shaped_events={len(d.shaped_events)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline MIDI Playback Demo")
    parser.add_argument("--port", type=str, default="PocketDrummer Out",
                        help="MIDI output port name (default: PocketDrummer Out)")
    parser.add_argument("--bpm", type=float, default=120.0,
                        help="Tempo in BPM (default: 120)")
    args = parser.parse_args()

    ports = list_available_ports()
    print(f"Target port: '{args.port}'")
    print(f"Available MIDI output ports:")
    if not ports:
        print("  (none)")
        print("\nNo MIDI output ports found. Please ensure EZDrummer or loopMIDI")
        print("is running with the 'PocketDrummer Out' port configured.")
        return 1
    for p in ports:
        print(f"  - {p}")

    port_name = find_or_none(args.port)
    if port_name is None:
        print(f"\nPort '{args.port}' not found in available ports.")
        print("Check your MIDI routing and try again.")
        return 1

    print(f"\nOpening MIDI output: {port_name}")
    midi = MidiOut(port_name)
    midi.open()
    bpm = args.bpm
    print(f"Tempo: {bpm} BPM")

    try:
        # ── Scenario 1: Stable Sparse → ENTER → MAINTAIN ──
        _header("SCENARIO 1: Stable Sparse -> ENTER -> MAINTAIN")
        p = DrummerBrainPipeline()
        _enter_pipeline(p)
        for i in range(2):
            t = 3.0 + i
            p.process(now=t, phase_alignment=0.75)
        d = p.process(now=5.0, phase_alignment=0.75)
        _report("MAINTAIN", d)
        if d.shaped_events:
            play_events_with_diagnostics(midi, d.shaped_events, bpm=bpm, repeats=2)
        time.sleep(0.5)

        # ── Scenario 2: Dense Frantic → REDUCE ──
        _header("SCENARIO 2: Dense Frantic -> REDUCE")
        p = DrummerBrainPipeline()
        _enter_pipeline(p)
        for i in range(20):
            t = 3.1 + i * 0.1
            p.feed_event(MusicalEvent(t, 0.7))
            p.process(now=t, phase_alignment=0.75)
        d = p.process(now=5.5, phase_alignment=0.75)
        _report("REDUCE", d)
        if d.shaped_events:
            play_events_with_diagnostics(midi, d.shaped_events, bpm=bpm, repeats=2)
        time.sleep(0.5)

        # ── Scenario 3: Weak Erratic → ANCHOR ──
        _header("SCENARIO 3: Weak Erratic -> ANCHOR")
        p = DrummerBrainPipeline()
        _enter_pipeline(p)
        for t in [3.1, 3.5, 3.8, 4.3, 4.7, 5.0]:
            p.feed_event(MusicalEvent(t, 0.12))
            p.process(now=t, phase_alignment=0.25)
        d = p.process(now=5.5, phase_alignment=0.25)
        _report("ANCHOR", d)
        if d.shaped_events:
            play_events_with_diagnostics(midi, d.shaped_events, bpm=bpm, repeats=2)
        time.sleep(0.5)

        # ── Scenario 4: Controlled Build → BUILD ──
        _header("SCENARIO 4: Controlled Build -> BUILD")
        p = DrummerBrainPipeline()
        _enter_pipeline(p)
        for i in range(8):
            t = 3.1 + i * 0.25
            strength = 0.5 + i * 0.05
            p.feed_event(MusicalEvent(t, min(strength, 0.85)))
            p.process(now=t, phase_alignment=0.7)
        d = p.process(now=5.5, phase_alignment=0.7)
        _report("BUILD", d)
        if d.shaped_events:
            play_events_with_diagnostics(midi, d.shaped_events, bpm=bpm, repeats=2)
        time.sleep(0.5)

        # ── Scenario 5: Long Silence → BAIL ──
        _header("SCENARIO 5: Long Silence -> BAIL")
        p = DrummerBrainPipeline()
        _enter_pipeline(p)
        d = p.process(now=10.0, phase_alignment=0.75)
        _report("BAIL", d)
        if d.shaped_events:
            play_events_with_diagnostics(midi, d.shaped_events, bpm=bpm, repeats=1)
        else:
            print("    (silence — no MIDI output)")

        print(f"\n{'=' * 60}")
        print("Pipeline MIDI Demo complete.")
        print(f"{'=' * 60}")

    finally:
        midi.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())