"""Revisioned priority-queue MIDI scheduler for the Bunny V1 controller.

Receives immutable ``LedgerEvent`` records and a MIDI sink (real or fake).
Emits events on their deadlines, respects generation invalidation,
deduplicates, and reports timing diagnostics.

The scheduler owns the timing loop.  It makes NO musical decisions and
has NO access to raw listener/tracker state.
"""

from __future__ import annotations

import heapq
import logging
import time as _time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from drummer.live_models import LedgerEvent, LiveConfig, MonotonicClock

log = logging.getLogger(__name__)


def _default_clock() -> float:
    return _time.perf_counter()


# ── MIDI sink protocol ────────────────────────────────────────────────


class MidiSinkProtocol(Protocol):
    """Minimal protocol the scheduler needs from a MIDI output.

    The fake sink (``tests/fake_midi.FakeMidiSink``) satisfies this,
    as does a thin wrapper around ``midi_out.MidiOut``.
    """

    def send_scheduled(
        self, note: int, velocity: int, channel: int, deadline: float
    ) -> None: ...

    def send_note(self, note: int, velocity: int, channel: int) -> None: ...

    def close(self) -> None: ...

    @property
    def is_open(self) -> bool: ...


# ── Timing diagnostics ────────────────────────────────────────────────


@dataclass
class SchedulerDiagnostics:
    """Accumulated timing stats from the scheduler loop."""

    total_emitted: int = 0
    total_dropped: int = 0
    total_late: int = 0
    max_late_seconds: float = 0.0
    total_jitter_sum: float = 0.0
    """Sum of |actual - deadline| for on-time events."""
    queue_depth_max: int = 0
    cycles: int = 0
    """Number of planning/sleep cycles."""

    @property
    def mean_jitter(self) -> float:
        if self.total_emitted == 0:
            return 0.0
        return self.total_jitter_sum / self.total_emitted


# ── Live Scheduler ────────────────────────────────────────────────────


class LiveScheduler:
    """Priority-queue MIDI scheduler with generation-based cancellation.

    Parameters
    ----------
    config : LiveConfig
        Lookahead, late budget, etc.
    sink : MidiSinkProtocol
        MIDI output (real or fake).
    clock : MonotonicClock | None
        Injectable monotonic clock.
    """

    def __init__(
        self,
        config: LiveConfig,
        sink: MidiSinkProtocol,
        clock: MonotonicClock | None = None,
    ) -> None:
        self._config = config
        self._sink = sink
        self._clock = clock if clock is not None else _default_clock

        # Priority queue: list of (deadline, insertion_order, event)
        self._queue: list[tuple[float, int, LedgerEvent]] = []
        self._insertion_counter: int = 0

        # Deduplication is retained for the lifetime of a generation, not
        # merely while an event is queued.  A rolling planner may revisit a
        # horizon after an event has fired; that must never emit it twice.
        self._seen_event_keys: set[tuple[int, str]] = set()

        # Generation tracking
        self._current_generation: int = 0

        # Lifecycle
        self._running: bool = False
        self._shutdown: bool = False

        # Diagnostics
        self.diag = SchedulerDiagnostics()

        log.info("LiveScheduler initialised — lookahead=%.3fs late=%.3fs",
                  config.max_lookahead_seconds, config.late_budget_seconds)

    # ── Public API ────────────────────────────────────────────────────

    def enqueue(self, events: list[LedgerEvent]) -> None:
        """Add ledger events to the queue.

        Events from older generations are silently skipped.
        Events with duplicate ``(generation, event_id)`` keys are skipped,
        including events that have already fired or been dropped.
        """
        for ev in events:
            if ev.generation < self._current_generation:
                log.debug("Skip old-gen event %s (gen %d < %d)",
                           ev.event_id, ev.generation, self._current_generation)
                continue
            key = (ev.generation, ev.event_id)
            if key in self._seen_event_keys:
                log.debug("Skip duplicate event %s", ev.event_id)
                continue
            self._insertion_counter += 1
            heapq.heappush(
                self._queue,
                (ev.deadline, self._insertion_counter, ev),
            )
            self._seen_event_keys.add(key)
        self.diag.queue_depth_max = max(self.diag.queue_depth_max, len(self._queue))

    def invalidate_generation(self, new_generation: int) -> int:
        """Remove all events with generation < *new_generation*.

        Returns the count of removed events.
        """
        removed = 0
        self._current_generation = new_generation
        # Old generations can never be accepted again, so discard their
        # historical keys while retaining current/future-generation keys.
        self._seen_event_keys = {
            key for key in self._seen_event_keys if key[0] >= new_generation
        }
        new_queue: list[tuple[float, int, LedgerEvent]] = []
        for item in self._queue:
            if item[2].generation < new_generation:
                removed += 1
            else:
                new_queue.append(item)
        heapq.heapify(new_queue)
        self._queue = new_queue
        self.diag.queue_depth_max = max(self.diag.queue_depth_max, len(self._queue))
        if removed:
            log.info("Invalidated %d events (gen < %d)", removed, new_generation)
        return removed

    def fire_due_events(self) -> None:
        """Emit all queued events whose deadline has passed.

        Events within the late budget are emitted despite being late.
        Events beyond the late budget are dropped.
        """
        now = self._clock()
        late_budget = self._config.late_budget_seconds

        while self._queue and self._queue[0][0] <= now:
            deadline, _, event = heapq.heappop(self._queue)

            lateness = now - deadline
            if lateness > late_budget:
                self.diag.total_dropped += 1
                log.warning("Drop event %s (late by %.3fms > %.3fms budget)",
                             event.event_id, lateness * 1000, late_budget * 1000)
                continue

            if lateness > 0:
                self.diag.total_late += 1
                self.diag.max_late_seconds = max(self.diag.max_late_seconds, lateness)

            try:
                self._sink.send_scheduled(
                    event.note, event.velocity, event.channel, deadline,
                )
                self.diag.total_emitted += 1
                self.diag.total_jitter_sum += abs(lateness)
            except Exception:
                log.exception("Failed to emit MIDI event %s", event.event_id)

    def next_deadline(self) -> float | None:
        """Return the deadline of the next event, or None if queue is empty."""
        if not self._queue:
            return None
        return self._queue[0][0]

    def sleep_until_next(self) -> None:
        """Sleep until the next deadline, bounded by lookahead.

        If no events are queued, sleep for the lookahead duration.
        Uses the injected clock for sleep timing.
        """
        now = self._clock()
        nd = self.next_deadline()

        if nd is None:
            # No events — sleep for lookahead
            sleep_duration = self._config.max_lookahead_seconds
        else:
            sleep_duration = nd - now
            if sleep_duration <= 0:
                return  # already due or past
            # Cap to lookahead
            sleep_duration = min(sleep_duration, self._config.max_lookahead_seconds)

        # Bound and sleep
        sleep_duration = max(0.0, min(sleep_duration, 0.200))  # max 200ms per cycle
        if sleep_duration > 0:
            _time.sleep(sleep_duration)

    def run_loop(self) -> None:
        """Run the scheduling loop until ``stop()`` is called.

        This is a blocking call.  Each cycle:
        1. Fire any due events.
        2. Sleep until the next deadline (or bounded lookahead).
        3. Allow the caller to enqueue more events between cycles
           via calls from another thread (though for V1 single-threaded
           use is assumed).
        """
        self._running = True
        log.info("Scheduler loop started")

        try:
            while self._running and not self._shutdown:
                self.diag.cycles += 1
                self.diag.queue_depth_max = max(
                    self.diag.queue_depth_max, len(self._queue),
                )
                self.fire_due_events()
                self.sleep_until_next()
        finally:
            self._running = False
            log.info("Scheduler loop ended — emitted=%d dropped=%d late=%d",
                       self.diag.total_emitted, self.diag.total_dropped,
                       self.diag.total_late)

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False
        self._shutdown = True

    def shutdown(self) -> None:
        """Stop the loop and close the MIDI sink."""
        self.stop()
        if self._sink.is_open:
            self._sink.close()
            log.info("MIDI sink closed")

    @property
    def queue_depth(self) -> int:
        return len(self._queue)
