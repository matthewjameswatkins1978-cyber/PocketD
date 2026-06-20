"""Fake MIDI sink for deterministic scheduler / controller tests.

Captures every sent note as a ``FakeMidiEvent`` with the monotonic
deadline and timestamps, plus diagnostic counters.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from drummer.live_models import MonotonicClock


@dataclass(frozen=True)
class FakeMidiEvent:
    """One captured MIDI note emission."""

    note: int
    velocity: int
    channel: int
    deadline: float
    """The monotonic deadline the scheduler intended."""
    fired_at: float
    """The monotonic time when the send actually occurred."""
    note_on: bool = True
    """True for note_on, False for note_off."""


class FakeMidiSink:
    """Records MIDI note-on / note-off calls for test inspection.

    Parameters
    ----------
    clock : MonotonicClock
        The same fake clock used by the controller / scheduler.
    """

    def __init__(self, clock: MonotonicClock) -> None:
        self._clock = clock
        self.events: list[FakeMidiEvent] = []
        self._open: bool = True

    # ── MIDI output interface ──

    def send_note(self, note: int, velocity: int, channel: int = 9) -> None:
        """Simulate a combined note-on + note-off."""
        if not self._open:
            raise RuntimeError("FakeMidiSink is closed")
        now = self._clock()
        self.events.append(
            FakeMidiEvent(
                note=note,
                velocity=velocity,
                channel=channel,
                deadline=now,
                fired_at=now,
                note_on=True,
            )
        )

    def note_on(self, note: int, velocity: int, channel: int = 9) -> None:
        if not self._open:
            raise RuntimeError("FakeMidiSink is closed")
        self.events.append(
            FakeMidiEvent(
                note=note,
                velocity=velocity,
                channel=channel,
                deadline=self._clock(),
                fired_at=self._clock(),
                note_on=True,
            )
        )

    def note_off(self, note: int, channel: int = 9) -> None:
        if not self._open:
            raise RuntimeError("FakeMidiSink is closed")
        self.events.append(
            FakeMidiEvent(
                note=note,
                velocity=0,
                channel=channel,
                deadline=self._clock(),
                fired_at=self._clock(),
                note_on=False,
            )
        )

    def send_kick(self, velocity: int = 100) -> None:
        self.send_note(36, velocity)

    def send_snare(self, velocity: int = 100) -> None:
        self.send_note(38, velocity)

    def send_hat(self, velocity: int = 80) -> None:
        self.send_note(42, velocity)

    # ── Scheduler compatibility ──

    def send_scheduled(
        self,
        note: int,
        velocity: int,
        channel: int,
        deadline: float,
    ) -> None:
        """Called by the live scheduler at a specific deadline."""
        if not self._open:
            raise RuntimeError("FakeMidiSink is closed")
        now = self._clock()
        self.events.append(
            FakeMidiEvent(
                note=note,
                velocity=velocity,
                channel=channel,
                deadline=deadline,
                fired_at=now,
                note_on=True,
            )
        )

    def panic(self) -> None:
        """Simulate an all-notes-off panic (records note_off for common drums)."""
        for note in (36, 38, 42, 46, 49, 51):
            self.note_off(note)

    def close(self) -> None:
        """Mark the sink as closed."""
        self._open = False

    @property
    def is_open(self) -> bool:
        return self._open

    # ── Test helpers ──

    def notes_at_step(self) -> list[int]:
        """Return note numbers in emission order."""
        return [e.note for e in self.events if e.note_on]

    def kick_count(self) -> int:
        return sum(1 for e in self.events if e.note == 36 and e.note_on)

    def snare_count(self) -> int:
        return sum(1 for e in self.events if e.note == 38 and e.note_on)

    def hat_count(self) -> int:
        return sum(1 for e in self.events if e.note == 42 and e.note_on)

    def clear(self) -> None:
        """Reset captured events."""
        self.events.clear()

    def deduplicated_notes(self) -> list[int]:
        """Return sorted unique notes that were emitted."""
        return sorted({e.note for e in self.events if e.note_on})