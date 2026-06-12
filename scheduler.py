"""16th-note groove scheduler — fires MIDI on step boundaries."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable

from groove_library import get_groove
from midi_out import MidiOut
from models import CLOSED_HAT, Groove, KICK, SNARE

logger = logging.getLogger(__name__)

NOTE_MAP = {
    "kick": KICK,
    "snare": SNARE,
    "hat": CLOSED_HAT,
}


def step_duration_seconds(bpm: float) -> float:
    """Duration of one 16th note at the given BPM."""
    return 60.0 / bpm / 4.0


def steps_for_complexity(groove: Groove, complexity_level: int) -> dict[str, list[int]]:
    """Reduce groove detail based on complexity ladder (5 = full, 1 = minimal)."""
    kick = list(groove.kick_steps)
    snare = list(groove.snare_steps)
    hats = list(groove.hat_steps)

    if complexity_level >= 5:
        return {"kick": kick, "snare": snare, "hat": hats}
    if complexity_level >= 4:
        return {"kick": kick, "snare": snare, "hat": hats}
    if complexity_level >= 3:
        return {"kick": kick, "snare": snare, "hat": hats[::2]}
    if complexity_level >= 2:
        return {"kick": kick, "snare": snare, "hat": []}
    # Level 1: pulse keeper — kick on downbeats only
    return {"kick": [0], "snare": [], "hat": []}


def hits_at_step(groove: Groove, step: int, complexity_level: int = 5) -> list[tuple[str, int]]:
    """Return (instrument, midi_note) pairs to fire at this step."""
    pattern = steps_for_complexity(groove, complexity_level)
    hits: list[tuple[str, int]] = []
    if step in pattern["kick"]:
        hits.append(("kick", KICK))
    if step in pattern["snare"]:
        hits.append(("snare", SNARE))
    if step in pattern["hat"]:
        hits.append(("hat", CLOSED_HAT))
    return hits


class GrooveScheduler:
    """Clock-driven scheduler that plays a groove loop via MIDI."""

    def __init__(
        self,
        midi: MidiOut,
        groove: Groove,
        bpm: float = 120.0,
        complexity_level: int = 5,
        on_step: Callable[[int], None] | None = None,
    ) -> None:
        self.midi = midi
        self.groove = groove
        self.bpm = bpm
        self.complexity_level = complexity_level
        self.on_step = on_step
        self._running = False
        self.current_step = 0

    def _fire_step(self, step: int) -> None:
        self.current_step = step
        if self.on_step:
            self.on_step(step)

        for instrument, note in hits_at_step(self.groove, step, self.complexity_level):
            velocity = 80 if instrument == "hat" else 100
            self.midi.send_note(note, velocity)
            logger.debug("step %2d  %s (%d)", step, instrument, note)

    def run(self, bars: int | None = None) -> None:
        """Run the scheduler loop until stopped or bars exhausted."""
        self._running = True
        step_dur = step_duration_seconds(self.bpm)
        total_steps = self.groove.steps * (bars if bars else 999_999)
        step_index = 0
        next_tick = time.perf_counter()

        logger.info(
            "Scheduler started: groove=%s bpm=%.1f step_dur=%.3fs",
            self.groove.id,
            self.bpm,
            step_dur,
        )

        try:
            while self._running and step_index < total_steps:
                step = step_index % self.groove.steps
                self._fire_step(step)
                step_index += 1
                next_tick += step_dur
                sleep_time = next_tick - time.perf_counter()
                if sleep_time > 0:
                    time.sleep(sleep_time)
                else:
                    logger.warning("Scheduler behind by %.1f ms", -sleep_time * 1000)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
        finally:
            self._running = False

    def stop(self) -> None:
        self._running = False


def run_hardcoded_groove(
    midi_port: str,
    groove_id: str = "simple_rock",
    bpm: float = 120.0,
    bars: int | None = None,
    complexity_level: int = 5,
) -> None:
    """Milestone 1 entry point: play a hard-coded groove to MIDI."""
    groove = get_groove(groove_id)
    with MidiOut(midi_port) as midi:
        scheduler = GrooveScheduler(
            midi=midi,
            groove=groove,
            bpm=bpm,
            complexity_level=complexity_level,
            on_step=lambda s: logger.info("beat step %2d/16", s) if s % 4 == 0 else None,
        )
        scheduler.run(bars=bars)
