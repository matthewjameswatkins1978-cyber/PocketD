"""Demo: Pulse Tracker — Module 2.

Demonstrates the PulseTracker's ability to maintain competing tempo/pulse
hypotheses from a stream of MusicalEvent objects.

Run:
    python demo_pulse.py

Three demo patterns:
    1. Steady 120 BPM — shows convergence on a clean pulse
    2. Human-feel 120 BPM — with timing jitter
    3. Half-time ambiguity — 60 vs 120 BPM ambiguity
"""

from __future__ import annotations

import logging
import sys

from perception.models import MusicalEvent
from perception.pulse import PulseTracker

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)
log = logging.getLogger("demo")


def _event(time: float, strength: float = 0.8, energy: float = 0.5) -> MusicalEvent:
    return MusicalEvent(
        time_seconds=time,
        strength=strength,
        energy=energy,
        density=0.5,
    )


def _print_state(state, label: str = "") -> None:
    """Pretty-print a PulseState."""
    if state.best_bpm is not None:
        print(f"\n  [{state.best_bpm:.1f} BPM | Confidence: {state.confidence:.0%} | "
              f"Stability: {state.stability}]")
    else:
        print(f"\n  [-- BPM | Confidence: {state.confidence:.0%} | "
              f"Stability: {state.stability}]")
    for hyp in state.hypotheses[:3]:  # show top 3
        print(f"    {hyp.bpm:.1f} BPM | Confidence: {hyp.confidence:.0%}")


def demo_steady_120() -> None:
    """Pattern 1: Steady 120 BPM."""
    print("\n" + "=" * 55)
    print("PATTERN 1: Steady 120 BPM")
    print("=" * 55)

    tracker = PulseTracker()
    interval = 60.0 / 120.0

    for i in range(12):
        t = i * interval
        tracker.process_event(_event(t))
        state = tracker.get_state()

        if i >= 2:  # Show from beat 2 onward
            print(f"\n[{t:.2f}s] PULSE STATE")
            _print_state(state)


def demo_human_feel() -> None:
    """Pattern 2: Human-feel 120 BPM with timing variation."""
    print("\n" + "=" * 55)
    print("PATTERN 2: Human-feel 120 BPM (with timing jitter)")
    print("=" * 55)

    tracker = PulseTracker()
    interval = 60.0 / 120.0

    for i in range(12):
        t = i * interval
        # Add human-like jitter (±15ms)
        jitter = (hash(i * 7 + 13) % 31 - 15) / 1000.0
        t = max(t + jitter, 0.0)

        # Vary strength slightly
        strength = 0.6 + (hash(i * 3 + 7) % 30) / 100.0
        tracker.process_event(_event(t, strength=strength))
        state = tracker.get_state()

        if i >= 2:
            print(f"\n[{t:.2f}s] PULSE STATE")
            _print_state(state)


def demo_half_time_ambiguity() -> None:
    """Pattern 3: Half-time ambiguity — events every 1.0s."""
    print("\n" + "=" * 55)
    print("PATTERN 3: Half-time ambiguity (events at 1.0s intervals)")
    print("=" * 55)

    tracker = PulseTracker()

    for i in range(8):
        t = i * 1.0  # 60 BPM rate
        # Every other event is an accent
        strength = 1.0 if i % 2 == 0 else 0.6
        tracker.process_event(_event(t, strength=strength))
        state = tracker.get_state()

        print(f"\n[{t:.2f}s] PULSE STATE")
        _print_state(state)


def main() -> int:
    print("=" * 55)
    print("PULSE TRACKER — MODULE 2: DEMO")
    print("=" * 55)
    print("\nDemonstrating competing tempo/pulse hypotheses from events.")
    print("The tracker maintains multiple interpretations simultaneously.")
    print("Confidence rises when evidence repeats and decays during silence.")

    demo_steady_120()
    demo_human_feel()
    demo_half_time_ambiguity()

    print("\n" + "=" * 55)
    print("Module 2 complete — the animal begins to nod its head.")
    print("Pulse perception established.")
    print("=" * 55)

    return 0


if __name__ == "__main__":
    sys.exit(main())