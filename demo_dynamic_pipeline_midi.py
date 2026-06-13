"""Dynamic Pipeline MIDI Scenario Demo.

Feeds simulated player input into ``DrummerBrainPipeline`` over time
and plays the resulting shaped MIDI output at each step so we can
hear the drummer transition between behaviours.

Run:
    python demo_dynamic_pipeline_midi.py                    # all sections, play steps
    python demo_dynamic_pipeline_midi.py --section stable   # single section
    python demo_dynamic_pipeline_midi.py --final-only       # play only final step per section
    python demo_dynamic_pipeline_midi.py --no-play          # print only

Sections:
    stable  — sparse stable playing → LISTEN → ENTER → MAINTAIN
    dense   — frantic playing → REDUCE (audibly simpler)
    anchor  — weak erratic playing → ANCHOR (clear supportive pulse)
    build   — controlled rising → BUILD
    bail    — long silence after entry → BAIL
    all     — (default) play all sections in sequence

Defaults: port="PocketDrummer Out", bpm=120, repeats=1, amount=0.25.
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
from drummer.pipeline import DrummerBrainPipeline, PipelineDecision
from drummer.pipeline_midi import (
    play_events_absolute,
    print_timing_report,
    groove_events_to_midi_messages,
    list_available_ports,
    find_or_none,
    INSTRUMENT_TO_NOTE,
)
from drummer.behaviour import BehaviourIntent
from perception.models import MusicalEvent
from drummer.feel import GrooveEvent


# ============================================================================
# Grooves
# ============================================================================


def _simple_groove() -> list[GrooveEvent]:
    """Simple rock groove: kick 1/3, snare 2/4, 8th hats."""
    return [
        GrooveEvent("kick", 0, velocity=100),
        GrooveEvent("hi_hat", 0, velocity=80),
        GrooveEvent("hi_hat", 2, velocity=70),
        GrooveEvent("snare", 4, velocity=100),
        GrooveEvent("hi_hat", 4, velocity=80),
        GrooveEvent("hi_hat", 6, velocity=70),
        GrooveEvent("kick", 8, velocity=98),
        GrooveEvent("hi_hat", 8, velocity=80),
        GrooveEvent("hi_hat", 10, velocity=70),
        GrooveEvent("snare", 12, velocity=100),
        GrooveEvent("hi_hat", 12, velocity=80),
        GrooveEvent("hi_hat", 14, velocity=70),
    ]


def _busy_groove() -> list[GrooveEvent]:
    """Busier groove with 16th hats, ghost snares, extra kicks."""
    return [
        GrooveEvent("kick", 0, velocity=115),
        GrooveEvent("hi_hat", 0, velocity=80),
        GrooveEvent("hi_hat", 1, velocity=55),
        GrooveEvent("hi_hat", 2, velocity=70),
        GrooveEvent("snare", 3, velocity=22, articulation="ghost"),
        GrooveEvent("snare", 4, velocity=110),
        GrooveEvent("hi_hat", 4, velocity=80),
        GrooveEvent("hi_hat", 5, velocity=55),
        GrooveEvent("hi_hat", 6, velocity=70),
        GrooveEvent("kick", 7, velocity=60, source_role="ghost"),
        GrooveEvent("kick", 8, velocity=112),
        GrooveEvent("hi_hat", 8, velocity=80),
        GrooveEvent("hi_hat", 9, velocity=55),
        GrooveEvent("hi_hat", 10, velocity=70),
        GrooveEvent("snare", 12, velocity=108),
        GrooveEvent("hi_hat", 12, velocity=80),
        GrooveEvent("snare", 13, velocity=20, articulation="ghost"),
        GrooveEvent("hi_hat", 14, velocity=70),
        GrooveEvent("kick", 15, velocity=85),
    ]


# ============================================================================
# Scenario builders
# ============================================================================


def _feed_and_process(
    pipeline: DrummerBrainPipeline,
    events: list[MusicalEvent],
    phase_alignment: float = 0.75,
) -> list[PipelineDecision]:
    decisions: list[PipelineDecision] = []
    for evt in events:
        pipeline.feed_event(evt)
        d = pipeline.process(now=evt.time_seconds,
                             phase_alignment=phase_alignment)
        decisions.append(d)
    return decisions


def scenario_stable(bpm: float = 120.0) -> list[PipelineDecision]:
    """Stable sparse playing → LISTEN → ENTER → MAINTAIN.

    Uses truly constant-strength events so EMAs settle quickly.
    After entry, feeds many more constant events to let change_score
    decay below BUILD threshold.
    """
    p = DrummerBrainPipeline()
    spacing = 60.0 / bpm  # quarter notes
    # Feed 40 quarter-note events at constant strength
    events = [MusicalEvent(time_seconds=i * spacing, strength=0.7)
              for i in range(40)]
    return _feed_and_process(p, events, phase_alignment=0.75)


def scenario_dense(bpm: float = 120.0) -> list[PipelineDecision]:
    """Frantic playing → REDUCE.

    Enters with stable events first, then floods with 16th-note burst.
    """
    p = DrummerBrainPipeline()
    # Enter first
    for i in range(6):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.7))
        p.process(now=t, phase_alignment=0.75)

    # Dense burst
    sixteenth = 60.0 / bpm / 4.0
    decisions: list[PipelineDecision] = []
    for i in range(30):
        t = 3.1 + i * sixteenth
        p.feed_event(MusicalEvent(t, 0.7))
        d = p.process(now=t, phase_alignment=0.75)
        decisions.append(d)

    # Process a few more times with no new events to let density drop
    for gap in [0.1, 0.2, 0.3]:
        t = 3.1 + 30 * sixteenth + gap
        decisions.append(p.process(now=t, phase_alignment=0.75))

    return decisions


def scenario_anchor(bpm: float = 120.0) -> list[PipelineDecision]:
    """Weak erratic playing → ANCHOR.

    Enters first, then feeds weak erratic events with poor phase.
    """
    p = DrummerBrainPipeline()
    # Enter
    for i in range(6):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.7))
        p.process(now=t, phase_alignment=0.75)

    # Weak erratic events with low phase
    decisions: list[PipelineDecision] = []
    for i, t in enumerate([3.1, 3.5, 3.8, 4.3, 4.7, 5.0, 5.3, 5.8]):
        p.feed_event(MusicalEvent(t, 0.12))
        d = p.process(now=t, phase_alignment=0.25)
        decisions.append(d)

    # Extra processes to push into ANCHOR
    for t in [6.0, 6.5, 7.0]:
        decisions.append(p.process(now=t, phase_alignment=0.25))

    return decisions


def scenario_build(bpm: float = 120.0) -> list[PipelineDecision]:
    """Controlled rising build → BUILD.

    Good repetition, good phase, controlled density, rising strength.
    """
    p = DrummerBrainPipeline()
    # Enter first with moderate stable events
    for i in range(8):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.5))
        p.process(now=t, phase_alignment=0.75)

    # Rising strength at 8th-note spacing
    decisions: list[PipelineDecision] = []
    spacing = 60.0 / bpm / 2.0  # 8th notes
    for i in range(20):
        t = 4.1 + i * spacing
        strength = 0.5 + min(i * 0.025, 0.4)  # 0.5 → 0.9
        p.feed_event(MusicalEvent(t, strength))
        d = p.process(now=t, phase_alignment=0.75)
        decisions.append(d)

    # Extra processes to push into BUILD
    for gap in [0.1, 0.2, 0.3]:
        t = 4.1 + 20 * spacing + gap
        decisions.append(p.process(now=t, phase_alignment=0.75))

    return decisions


def scenario_bail(bpm: float = 120.0) -> list[PipelineDecision]:
    """Long silence after entry → BAIL.

    Enters first, then processes with silence to trigger BAIL.
    Does NOT feed new events — just polls process() against silence.
    """
    p = DrummerBrainPipeline()
    # Enter
    for i in range(8):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.7))
        p.process(now=t, phase_alignment=0.75)

    # Now go silent — just poll process() at increasing times
    # to build up silence_duration beyond feature_bail_silence_seconds (1.50)
    decisions: list[PipelineDecision] = []
    for gap in [4.5, 5.5, 7.0, 10.0, 15.0]:
        d = p.process(now=gap, phase_alignment=0.75)
        decisions.append(d)

    return decisions


SCENARIOS: dict[str, callable] = {
    "stable": scenario_stable,
    "dense": scenario_dense,
    "anchor": scenario_anchor,
    "build": scenario_build,
    "bail": scenario_bail,
}


# ============================================================================
# Helpers
# ============================================================================


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_step(i: int, d: PipelineDecision) -> None:
    snap = d.feature_snapshot
    intent = d.behaviour_intent.value
    marker = "←" if intent != _print_step._prev_intent else " "
    _print_step._prev_intent = intent
    print(f"  step {i:3d}  t={d.timestamp:7.2f}s  "
          f"intent={intent:12s} {marker}"
          f"  density={snap.input_density:.2f}"
          f"  cert={snap.player_certainty:.2f}"
          f"  change={snap.change_score:.2f}"
          f"  stab={snap.repetition_stability:.2f}"
          f"  silence={snap.silence_duration:.1f}s"
          f"  raw={len(d.raw_events):2d}  shaped={len(d.shaped_events):2d}")

_print_step._prev_intent = ""


def _play_step(
    midi: MidiOut,
    step: int,
    d: PipelineDecision,
    bpm: float,
    note_duration: float = 0.09,
) -> None:
    events = d.shaped_events
    if not events:
        return
    # Play one bar only per step (to hear transitions)
    play_events_absolute(midi, events, bpm=bpm, repeats=1,
                         note_duration=note_duration)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Dynamic Pipeline MIDI Demo")
    parser.add_argument("--port", type=str, default="PocketDrummer Out")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--amount", type=float, default=0.25)
    parser.add_argument("--no-play", action="store_true",
                        help="Print only, do not send MIDI")
    parser.add_argument("--final-only", action="store_true",
                        help="Play only the final decision per section")
    parser.add_argument("--section", type=str, default="all",
                        help="Section: stable, dense, anchor, build, bail, or all")
    parser.add_argument("--repeats", type=int, default=1,
                        help="Repeats per playback step (default: 1)")
    args = parser.parse_args()

    do_play = not args.no_play
    final_only = args.final_only
    bpm = args.bpm
    section = args.section.lower()

    if section not in ("all", "stable", "dense", "anchor", "build", "bail"):
        print(f"Invalid section: '{args.section}'")
        return 1

    sections_to_run = list(SCENARIOS.keys()) if section == "all" else [section]

    # MIDI port
    midi: MidiOut | None = None
    if do_play:
        ports = list_available_ports()
        if not ports:
            print("No MIDI ports. Exiting.")
            return 1
        port_name = find_or_none(args.port)
        if port_name is None:
            print(f"Port '{args.port}' not found. Available: {ports}")
            return 1
        print(f"MIDI output: {port_name}")
        midi = MidiOut(port_name)
        midi.open()

    mode = ("final-only" if final_only else "play-steps") + (
        "" if do_play else " (print-only)")
    print(f"Dynamic Pipeline MIDI Demo — {mode}")
    print(f"  BPM: {bpm}  Repeats: {args.repeats}  Amount: {args.amount}")

    try:
        for sec_name in sections_to_run:
            scenario_fn = SCENARIOS[sec_name]
            _header(f"{sec_name.upper()}")

            decisions = scenario_fn(bpm)
            _print_step._prev_intent = ""

            for i, d in enumerate(decisions):
                # Print every step
                _print_step(i, d)

                # Play if this is a state change or final step
                is_new_intent = (
                    i == 0 or
                    d.behaviour_intent != decisions[i - 1].behaviour_intent
                )
                is_last = (i == len(decisions) - 1)
                should_play = (not final_only and is_new_intent) or (final_only and is_last)

                if do_play and midi and should_play and d.shaped_events:
                    print(f"           → playing ({d.behaviour_intent.value})")
                    _play_step(midi, i, d, bpm)
                    time.sleep(0.3)
                elif do_play and midi and should_play:
                    print(f"           → silence")

            # Final summary
            last = decisions[-1]
            print(f"\n  FINAL: intent={last.behaviour_intent.value}"
                  f"  raw={len(last.raw_events)}  shaped={len(last.shaped_events)}")

            # If final-only and not already played, play final now
            if final_only and do_play and midi:
                if last.shaped_events:
                    print(f"  Playing final bar × {args.repeats}...")
                    play_events_absolute(midi, last.shaped_events,
                                         bpm=bpm, repeats=args.repeats)
                else:
                    print(f"  (silence)")

            time.sleep(0.3)

        print(f"\n{'=' * 60}")
        print("Dynamic Pipeline Demo complete.")
        print(f"{'=' * 60}")

    finally:
        if midi is not None:
            midi.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())