"""Pipeline Comparison Demo — Raw vs Shaped output.

Run:
    python demo_pipeline_compare.py                                    # compare, default amount, play
    python demo_pipeline_compare.py --amount 0.0 --raw-only            # machine-tight raw only
    python demo_pipeline_compare.py --intent REDUCE --compare          # A/B one intent
    python demo_pipeline_compare.py --no-play                          # print events only
    python demo_pipeline_compare.py --amount 0.0 --shaped-only --play  # machine-tight shaped

Defaults: compare mode, play enabled, amount=0.25, port="PocketDrummer Out".
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
    play_events_absolute,
    print_timing_report,
    groove_events_to_midi_messages,
    list_available_ports,
    find_or_none,
    INSTRUMENT_TO_NOTE,
    grid_to_seconds,
    build_schedule,
)
from drummer.output_shaping import BehaviourOutputShaper, OutputShapingConfig
from drummer.behaviour import BehaviourIntent, parse_behaviour_intent
from drummer.feel import GrooveEvent


# ============================================================================
# Default groove
# ============================================================================


def _default_groove() -> list[GrooveEvent]:
    """Return the default one-bar rock groove (kick 1/3, snare 2/4, 8th hats)."""
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


# ============================================================================
# Helpers
# ============================================================================


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_events(label: str, events: list[GrooveEvent], bpm: float) -> None:
    print(f"\n  [{label}]  {len(events)} events")
    print(f"    {'Inst':>10s}  {'Grid':>4s}  {'Vel':>4s}  "
          f"{'Beat':>6s}  {'Sec':>7s}  {'Art':>8s}")
    print(f"    {'-' * 10}  {'-' * 4}  {'-' * 4}  "
          f"{'-' * 6}  {'-' * 7}  {'-' * 8}")
    for e in events:
        sec = grid_to_seconds(e.grid_position, bpm)
        beat = e.grid_position / 4.0
        print(f"    {e.instrument:>10s}  {e.grid_position:4d}  {e.velocity:4d}  "
              f"{beat:6.2f}  {sec:7.4f}  {e.articulation:>8s}")


def _print_changes(raw: list[GrooveEvent], shaped: list[GrooveEvent]) -> None:
    """Print per-event velocity/articulation diffs."""
    print(f"\n    Changes from raw:")
    raw_map = {(e.instrument, e.grid_position): e for e in raw}
    shaped_map = {(e.instrument, e.grid_position): e for e in shaped}
    all_keys = set(raw_map.keys()) | set(shaped_map.keys())

    for key in sorted(all_keys, key=lambda k: (k[1], k[0])):
        r = raw_map.get(key)
        s = shaped_map.get(key)
        inst, pos = key
        if r is None:
            print(f"      {inst:>10s} pos={pos:2d}  "
                  f"[ADDED]                   vel={s.velocity:3d}")
        elif s is None:
            print(f"      {inst:>10s} pos={pos:2d}  "
                  f"[REMOVED]  vel was {r.velocity:3d}")
        elif r.velocity != s.velocity or r.articulation != s.articulation:
            print(f"      {inst:>10s} pos={pos:2d}  "
                  f"vel: {r.velocity:3d}→{s.velocity:3d}  "
                  f"art: {r.articulation}→{s.articulation}")


def _play_events_set(
    midi: MidiOut,
    events: list[GrooveEvent],
    label: str,
    bpm: float,
    repeats: int,
    note_duration: float = 0.09,
) -> None:
    """Play a set of events with diagnostics."""
    msgs = groove_events_to_midi_messages(events, bpm=bpm)
    total_schedule = len(msgs) * 2  # on + off
    print(f"\n  [{label}]")
    print(f"    Events: {len(events)}  "
          f"MIDI messages: {len(msgs)} note_on + {len(msgs)} note_off "
          f"= {total_schedule} total")
    print(f"    Port: {midi.port_name}  BPM: {bpm}  "
          f"Note duration: {note_duration*1000:.0f}ms  Repeats: {repeats}")
    print(f"    Playing...")
    timing = play_events_absolute(midi, events, bpm=bpm, repeats=repeats,
                                  note_duration=note_duration)
    print_timing_report(timing)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(description="Pipeline Compare Demo")
    parser.add_argument("--port", type=str, default="PocketDrummer Out")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--amount", type=float, default=0.25,
                        help="Humanize amount: 0.0=tight, 1.0=full, 0.25=subtle")
    parser.add_argument("--raw-only", action="store_true",
                        help="Play/paint only raw (unshaped) output")
    parser.add_argument("--shaped-only", action="store_true",
                        help="Play/print only shaped output")
    parser.add_argument("--compare", action="store_true",
                        help="Play/print raw then shaped for comparison (default)")
    parser.add_argument("--no-play", action="store_true",
                        help="Print events only, do not send MIDI")
    parser.add_argument("--play", action="store_true", default=True,
                        help="Send MIDI output (default)")
    parser.add_argument("--intent", type=str, default=None,
                        help="Single intent: MAINTAIN, REDUCE, ANCHOR, BUILD, ENTER_SOFT")
    parser.add_argument("--repeats", type=int, default=2,
                        help="Number of bar repeats (default: 2)")
    args = parser.parse_args()

    # --no-play overrides
    do_play = not args.no_play

    # Determine compare mode
    raw_only = args.raw_only
    shaped_only = args.shaped_only
    compare = args.compare
    if not raw_only and not shaped_only and not compare:
        compare = True  # default

    bpm = args.bpm
    amount = args.amount
    repeats = args.repeats

    single_intent = parse_behaviour_intent(args.intent) if args.intent else None

    config = OutputShapingConfig(humanize_amount=amount)
    shaper = BehaviourOutputShaper(config)

    mode_str = ("raw-only" if raw_only else
                "shaped-only" if shaped_only else
                "compare")
    play_str = "PLAY enabled" if do_play else "PRINT-ONLY (no MIDI)"

    print(f"Pipeline Compare Demo")
    print(f"  Mode: {mode_str}  |  {play_str}")
    print(f"  Amount: {amount:.2f}  |  BPM: {bpm}  |  Repeats: {repeats}")

    # Build intents to test
    if single_intent:
        intents_to_test: list[BehaviourIntent] = [single_intent]
    else:
        intents_to_test = [
            BehaviourIntent.MAINTAIN,
            BehaviourIntent.REDUCE,
            BehaviourIntent.ANCHOR,
            BehaviourIntent.BUILD,
            BehaviourIntent.ENTER_SOFT,
        ]

    # MIDI port — only needed if playing
    midi: MidiOut | None = None
    if do_play:
        ports = list_available_ports()
        print(f"  Target port: '{args.port}'")
        print(f"  Available: {ports if ports else '(none)'}")
        if not ports:
            print("\nNo MIDI output ports available. Exiting.")
            return 1
        port_name = find_or_none(args.port)
        if port_name is None:
            print(f"\nPort '{args.port}' not found in available ports.")
            return 1
        print(f"  Opening MIDI output: {port_name}")
        midi = MidiOut(port_name)
        midi.open()

    try:
        groove = _default_groove()

        for intent in intents_to_test:
            _header(f"{intent.value.upper()} (humanize={amount:.2f})")

            # --- RAW ---
            if raw_only or compare:
                _print_events("RAW (unshaped)", groove, bpm)
                if do_play and midi:
                    _play_events_set(midi, groove, "RAW", bpm, repeats)
                elif do_play:
                    print(f"\n  [RAW] MIDI port not available — skipping playback")
                else:
                    msgs = groove_events_to_midi_messages(groove, bpm=bpm)
                    print(f"\n  [RAW] Print-only: {len(msgs)} note_on, "
                          f"{len(msgs)*2} total messages (not sent)")
                time.sleep(0.5)

            # --- SHAPED ---
            if shaped_only or compare:
                shaped = shaper.shape(groove, intent)
                _print_events(f"SHAPED ({intent.value})", shaped, bpm)

                if compare:
                    _print_changes(groove, shaped)

                if do_play and midi:
                    time.sleep(1.0)  # audible gap in compare mode
                    _play_events_set(midi, shaped, f"SHAPED ({intent.value})",
                                     bpm, repeats)
                elif do_play:
                    print(f"\n  [SHAPED] MIDI port not available — skipping playback")
                else:
                    msgs = groove_events_to_midi_messages(shaped, bpm=bpm)
                    print(f"\n  [SHAPED] Print-only: {len(msgs)} note_on, "
                          f"{len(msgs)*2} total messages (not sent)")
                time.sleep(0.5)

        print(f"\n{'=' * 60}")
        print("Pipeline Compare Demo complete.")
        print(f"{'=' * 60}")

    finally:
        if midi is not None:
            midi.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())