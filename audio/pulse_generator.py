"""Generate synthetic pulse events for the Bunny Deluxe diagnostic path."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PulseEvent:
    time_seconds: float
    strength: float
    label: str = "pulse"


def generate_pulse_events(
    bpm: float = 120.0,
    duration_seconds: float = 4.0,
    pulse_width: float = 0.02,
) -> list[PulseEvent]:
    """Generate a simple sequence of onsets at quarter-note spacing.

    The output is intentionally tiny and deterministic so it can be used as a
    fake input stage before real audio or microphone handling is added.
    """
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    if duration_seconds <= 0:
        raise ValueError("duration_seconds must be positive")

    quarter_note_seconds = 60.0 / bpm
    total_beats = int(duration_seconds / quarter_note_seconds) + 1
    events: list[PulseEvent] = []

    for beat_index in range(total_beats):
        time_seconds = beat_index * quarter_note_seconds
        if time_seconds > duration_seconds + 1e-9:
            break

        strength = 1.0 if beat_index % 4 == 0 else 0.75
        events.append(
            PulseEvent(
                time_seconds=time_seconds,
                strength=strength,
                label="kick" if beat_index % 4 == 0 else "pulse",
            )
        )

    return events


def main() -> None:
    events = generate_pulse_events()
    print(f"Generated {len(events)} synthetic pulse events at 120 BPM")
    for event in events:
        print(
            f"t={event.time_seconds:.3f}s strength={event.strength:.2f} label={event.label}"
        )


if __name__ == "__main__":
    main()
