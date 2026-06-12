"""Demo: Bar / Downbeat Tracker — Module 3.

Demonstrates the BarTracker's ability to estimate bar position and downbeat
location from MusicalEvent streams and PulseTracker output.

Run:
    python demo_bar.py

Three demo patterns:
    1. Clear 4/4 groove at 120 BPM with strong downbeats
    2. Ambiguous half-bar accents
    3. Human-jittered groove that still stabilizes
"""

from __future__ import annotations

import logging
import sys

from perception.bar import BarTracker
from perception.models import MusicalEvent
from perception.pulse import PulseTracker, PulseHypothesis

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("demo")


def _event(t: float, strength: float = 0.8, energy: float = 0.5) -> MusicalEvent:
    return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=0.5)


def _print_bar_state(state, event_time: float) -> None:
    """Pretty-print a BarState."""
    if state.best_hypothesis is None:
        print(f"  [-- BPM | no bar hypothesis yet]")
        return

    best = state.best_hypothesis
    beat_pos = state.estimated_beat_in_bar
    if beat_pos is None:
        beat_pos = "?"

    print(f"  BPM: {best.bpm:.1f}  |  Downbeat: {best.downbeat_time:.2f}s  |  "
          f"Beat-in-bar: {beat_pos}  |  Confidence: {state.confidence:.0%}  |  "
          f"Hypotheses: {len(state.hypotheses)}")

    if len(state.hypotheses) > 1:
        for h in state.hypotheses[1:3]:
            print(f"    -> {h.bpm:.1f} BPM, downbeat={h.downbeat_time:.2f}s, conf={h.confidence:.0%}")


def demo_clear_four_four() -> None:
    """Pattern 1: Clear 4/4 groove at 120 BPM with strong downbeats."""
    print("\n" + "=" * 60)
    print("PATTERN 1: Clear 4/4 at 120 BPM (4 bars)")
    print("=" * 60)

    pulse_tracker = PulseTracker()
    bar_tracker = BarTracker()
    beat = 0.5  # 120 BPM

    for bar_num in range(4):
        base = bar_num * 2.0

        # Beat 1 — strong kick (downbeat)
        ev = _event(base + 0.0, strength=1.0, energy=0.9)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        # Beat 2 — soft hi-hat
        ev = _event(base + 0.5, strength=0.3, energy=0.2)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        # Beat 3 — strong snare
        ev = _event(base + 1.0, strength=0.9, energy=0.9)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        # Beat 4 — soft hi-hat
        ev = _event(base + 1.5, strength=0.3, energy=0.2)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        print(f"\n[Bar {bar_num + 1}] t={base + 2.0:.2f}s  "
              f"Pulse: {pulse_state.best_bpm:.0f} BPM ({pulse_state.confidence:.0%})")
        _print_bar_state(bar_state, base + 2.0)


def demo_ambiguous_half_bar() -> None:
    """Pattern 2: Ambiguous half-bar accents — strong hit every 2 beats."""
    print("\n" + "=" * 60)
    print("PATTERN 2: Ambiguous (strong beats every 1.0s)")
    print("=" * 60)

    pulse_tracker = PulseTracker()
    bar_tracker = BarTracker()

    for i in range(8):
        t = i * 1.0
        # Strong every 2 beats, weak otherwise
        strength = 1.0 if i % 2 == 0 else 0.4
        energy = 0.9 if i % 2 == 0 else 0.2

        ev = _event(t, strength=strength, energy=energy)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        if pulse_state.best_bpm is not None:
            print(f"\n[{t:.2f}s] Pulse: {pulse_state.best_bpm:.0f} BPM ({pulse_state.confidence:.0%})")
        else:
            print(f"\n[{t:.2f}s] Pulse: -- BPM")
        _print_bar_state(bar_state, t)


def demo_human_jitter() -> None:
    """Pattern 3: Human-jittered groove that still stabilizes."""
    print("\n" + "=" * 60)
    print("PATTERN 3: Human-jittered 4/4 (±25ms). Should still find bar.")
    print("=" * 60)

    pulse_tracker = PulseTracker()
    bar_tracker = BarTracker()
    beat = 0.5

    for bar_num in range(4):
        base = bar_num * 2.0

        # Add jitter (±25ms) to each hit
        j1 = (hash(bar_num * 10) % 51 - 25) / 1000.0
        j2 = (hash(bar_num * 10 + 1) % 51 - 25) / 1000.0
        j3 = (hash(bar_num * 10 + 2) % 51 - 25) / 1000.0
        j4 = (hash(bar_num * 10 + 3) % 51 - 25) / 1000.0

        ev = _event(base + j1, strength=1.0, energy=0.9)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        ev = _event(base + 0.5 + j2, strength=0.3, energy=0.2)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        ev = _event(base + 1.0 + j3, strength=0.9, energy=0.9)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        ev = _event(base + 1.5 + j4, strength=0.3, energy=0.2)
        pulse_state = pulse_tracker.process_event(ev)
        bar_state = bar_tracker.update(ev, pulse_state)

        # Show at bar end
        state = bar_tracker.get_state(current_time=base + 2.0)
        print(f"\n[Bar {bar_num + 1}] t={base + 2.0:.2f}s  "
              f"Pulse: {pulse_state.best_bpm:.0f} BPM")
        _print_bar_state(state, base + 2.0)


def main() -> int:
    print("=" * 60)
    print("BAR / DOWNBEAT TRACKER — MODULE 3: DEMO")
    print("=" * 60)
    print("\nDemonstrating bar position estimation from events + pulse.")
    print("The tracker maintains competing downbeat hypotheses.")
    print("Confidence rises when strong events repeat at bar boundaries.")

    demo_clear_four_four()
    demo_ambiguous_half_bar()
    demo_human_jitter()

    print("\n" + "=" * 60)
    print("Module 3 complete — the animal knows where the bars breathe.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())