"""Demo: Groove Intent Engine — Module 4.

Demonstrates the GrooveIntentEngine converting perception state into
high-level drummer behaviour decisions.

Run:
    python demo_groove_intent.py

Shows:
    1. WAIT (low confidence)
    2. ENTER (confidence rises)
    3. HOLD (steady groove)
    4. BUILD (rising energy/density)
    5. REDUCE (falling energy)
    6. PREPARE_FILL (near bar end with confidence)
"""

from __future__ import annotations

import logging
import sys

from drummer.intent import GrooveIntentEngine, GrooveAction
from perception.bar import BarHypothesis, BarState
from perception.models import MusicalEvent
from perception.pulse import PulseHypothesis, PulseState

logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s — %(message)s",
)


def _event(t: float, strength: float = 0.8, energy: float = 0.5, density: float = 0.5) -> MusicalEvent:
    return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=density)


def _pulse(confidence: float) -> PulseState:
    return PulseState(
        hypotheses=[PulseHypothesis(bpm=120.0, confidence=confidence, matches=10)],
        best_bpm=120.0,
        confidence=confidence,
        stability="stable" if confidence > 0.5 else "rising",
    )


def _bar(confidence: float, beat: int) -> BarState:
    return BarState(
        hypotheses=[
            BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=4, confidence=confidence),
        ],
        best_hypothesis=BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=4, confidence=confidence),
        is_confident=confidence > 0.5,
        estimated_bar_position=float(beat),
        estimated_beat_in_bar=beat,
        confidence=confidence,
        timestamp=0.0,
    )


def _print_intent(intent, label: str = "") -> None:
    print(f"  Action: {intent.action.name:15s}  Play: {str(intent.should_play):5s}  "
          f"Fill: {str(intent.should_fill):5s}  Complex: {intent.suggested_complexity:.2f}  "
          f"Vel: {intent.suggested_velocity:.2f}")
    print(f"    Pulse conf: {intent.pulse_confidence:.0%}  Bar conf: {intent.bar_confidence:.0%}  "
          f"Energy: {intent.energy_level:.2f}  Density: {intent.density_level:.2f}")
    print(f"    Reason: {intent.reason}")


def main() -> int:
    print("=" * 65)
    print("GROOVE INTENT ENGINE — MODULE 4: DEMO")
    print("=" * 65)
    print("\nConverting perception state into drummer behaviour intentions.")
    print("No MIDI. No patterns. Just intent.\n")

    engine = GrooveIntentEngine()
    t = 0.0

    # ── Phase 1: WAIT (low confidence) ─────────────────────────
    print("── Phase 1: WAIT — listening for pulse and bar ──")
    for i in range(3):
        t += 0.5
        intent = engine.update(_event(t, energy=0.4), _pulse(0.2), _bar(0.15, beat=0))
        print(f"\n[{t:.1f}s]")
        _print_intent(intent)

    # ── Phase 2: ENTER (confidence rises above threshold) ──────
    print("\n── Phase 2: ENTER — confidence met, entering conservatively ──")
    for i in range(3):
        t += 0.5
        intent = engine.update(_event(t, energy=0.5), _pulse(0.50), _bar(0.40, beat=i % 4))
        print(f"\n[{t:.1f}s]")
        _print_intent(intent)

    # ── Phase 3: HOLD (steady groove) ──────────────────────────
    print("\n── Phase 3: HOLD — steady musical input ──")
    for i in range(4):
        t += 0.5
        intent = engine.update(_event(t, energy=0.55, density=0.5), _pulse(0.75), _bar(0.70, beat=i % 4))
        print(f"\n[{t:.1f}s]")
        _print_intent(intent)

    # ── Phase 4: BUILD (rising intensity) ──────────────────────
    print("\n── Phase 4: BUILD — energy and density rising ──")
    for i in range(4):
        t += 0.5
        e = 0.4 + i * 0.15
        d = 0.3 + i * 0.2
        intent = engine.update(_event(t, energy=e, density=d), _pulse(0.80), _bar(0.80, beat=i % 4))
        print(f"\n[{t:.1f}s]")
        _print_intent(intent)

    # ── Phase 5: REDUCE (falling energy) ───────────────────────
    print("\n── Phase 5: REDUCE — energy dropping ──")
    for i in range(4):
        t += 0.5
        e = 0.8 - i * 0.18
        d = 0.5 - i * 0.12
        intent = engine.update(_event(t, energy=e, density=d), _pulse(0.80), _bar(0.80, beat=i % 4))
        print(f"\n[{t:.1f}s]")
        _print_intent(intent)

    # ── Phase 6: PREPARE_FILL (bar end, high confidence) ───────
    print("\n── Phase 6: PREPARE_FILL — near bar end ──")
    # Feed several bars worth to satisfy MIN_BARS_BETWEEN_FILLS
    for bar in range(5):
        for beat in range(4):
            t += 0.5
            intent = engine.update(
                _event(t, energy=0.65, density=0.6),
                _pulse(0.85),
                _bar(0.85, beat=beat),
            )
    # Now at beat 3 — near bar end
    print(f"\n[{t:.1f}s] (beat 3 — approaching bar boundary)")
    _print_intent(intent)

    # One more at beat 3 with high energy
    t += 0.5
    intent = engine.update(
        _event(t, energy=0.75, density=0.7, strength=0.9),
        _pulse(0.90),
        _bar(0.90, beat=3),
    )
    print(f"\n[{t:.1f}s] (beat 3 — preparing fill)")
    _print_intent(intent)

    print("\n" + "=" * 65)
    print("Module 4 complete — groove intent established.")
    print("The drummer knows what kind of behaviour is appropriate.")
    print("=" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())