"""Synthetic groove scheduler demo using the existing groove logic."""

from __future__ import annotations

from groove_library import get_groove
from scheduler import GrooveScheduler, step_duration_seconds


class DummyMidi:
    """Minimal stand-in for MIDI output during synthetic demos."""

    def __init__(self) -> None:
        self.sent: list[tuple[int, int]] = []

    def send_note(self, note: int, velocity: int) -> None:
        self.sent.append((note, velocity))


def main() -> None:
    groove = get_groove("simple_rock")
    midi = DummyMidi()
    scheduler = GrooveScheduler(midi=midi, groove=groove, bpm=120.0, complexity_level=2)

    print("Synthetic groove scheduler demo")
    print(f"Step duration: {step_duration_seconds(120.0):.3f}s")
    print(f"Groove steps: {groove.steps}")

    for step in range(4):
        scheduler._fire_step(step)
        print(f"step={step} hits={midi.sent[-len(midi.sent) :] if midi.sent else []}")


if __name__ == "__main__":
    main()
