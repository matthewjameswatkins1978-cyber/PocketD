"""Dependency-injected orchestration for the Bunny V1 live slice.

The runtime joins adapters, controller, planners and scheduler without owning
audio capture.  A hardware runner only needs to feed the existing trackers and
call :meth:`LiveRuntime.tick` regularly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from drummer.live_controller import LiveController
from drummer.live_models import (
    BarAdapterState,
    ControllerSnapshot,
    LiveConfig,
    MonotonicClock,
    PulseAdapterState,
)
from drummer.live_scheduler import LiveScheduler
from drummer.straight_pocket import (
    KickMirrorObserver,
    SlotObservation,
    plan_anchor_ledger,
    plan_mirror_ledger,
    quantise_to_locked_grid,
)


class PulseAdapterProtocol(Protocol):
    def adapt(self) -> PulseAdapterState: ...


class BarAdapterProtocol(Protocol):
    def adapt(self) -> BarAdapterState: ...


@dataclass(frozen=True)
class RuntimeSnapshot:
    """One tick of controller and scheduler diagnostics."""

    controller: ControllerSnapshot
    queue_depth: int
    total_emitted: int
    total_dropped: int
    total_late: int


class LiveRuntime:
    """Coordinate one deterministic iteration of Bunny's live pipeline."""

    def __init__(
        self,
        config: LiveConfig,
        pulse_adapter: PulseAdapterProtocol,
        bar_adapter: BarAdapterProtocol,
        controller: LiveController,
        scheduler: LiveScheduler,
        clock: MonotonicClock,
        *,
        planning_horizon_bars: int = 2,
        mirror_observer: KickMirrorObserver | None = None,
    ) -> None:
        if planning_horizon_bars < 1:
            raise ValueError("planning_horizon_bars must be at least 1")
        self._config = config
        self._pulse_adapter = pulse_adapter
        self._bar_adapter = bar_adapter
        self._controller = controller
        self._scheduler = scheduler
        self._clock = clock
        self._planning_horizon_bars = planning_horizon_bars
        self._mirror_observer = mirror_observer or KickMirrorObserver(config)

        self._known_generation = controller.generation
        self._previous_state = controller.state
        self._last_completed_bar: int | None = None
        self._stopped = False

    def tick(self) -> RuntimeSnapshot:
        """Adapt tracker state, update control, plan, and fire due MIDI."""
        if self._stopped:
            return self._snapshot(self._controller.stop())

        pulse = self._pulse_adapter.adapt()
        bar = self._bar_adapter.adapt()
        control = self._controller.update(pulse, bar)

        self._synchronise_generation()
        self._handle_state_change(control.state)

        if control.state == "PLAYING":
            self._finish_crossed_bars(control.current_bar_index)
            self._plan_horizon(control.current_bar_index)

        self._scheduler.fire_due_events()
        self._previous_state = control.state
        return self._snapshot(control)

    def observe_player_hit(self, observed_at: float, strength: float) -> bool:
        """Quantise and retain one player hit for conservative kick mirroring.

        Returns ``True`` only when the hit lies on the locked grid and is
        accepted by the observer. Anchor slots are safely ignored downstream.
        """
        if (
            self._stopped
            or self._controller.state != "PLAYING"
            or self._controller.bar_epoch is None
            or self._controller.beat_period is None
        ):
            return False
        bar_index, slot, offset = quantise_to_locked_grid(
            observed_at,
            self._controller.bar_epoch,
            self._controller.beat_period,
            self._config,
        )
        if bar_index < 0:
            return False
        self._mirror_observer.observe(
            SlotObservation(
                bar_index=bar_index,
                slot=slot,
                offset_seconds=offset,
                strength=strength,
                observed_at=observed_at,
            )
        )
        return slot not in set(self._config.anchor_slots)

    def stop(self, *, close_midi: bool = True) -> RuntimeSnapshot:
        """Stop control, invalidate every pending note, and optionally close MIDI."""
        control = self._controller.stop()
        self._synchronise_generation()
        self._mirror_observer.reset()
        self._stopped = True
        if close_midi:
            self._scheduler.shutdown()
        else:
            self._scheduler.stop()
        return self._snapshot(control)

    def _synchronise_generation(self) -> None:
        generation = self._controller.generation
        if generation != self._known_generation:
            self._scheduler.invalidate_generation(generation)
            self._known_generation = generation

    def _handle_state_change(self, state: str) -> None:
        if state == "PLAYING" and self._previous_state != "PLAYING":
            # A new entry starts at bar zero. Recovery preserves the absolute
            # locked bar index, so begin completion tracking immediately before it.
            self._last_completed_bar = self._controller.bar_index - 1
        elif self._previous_state == "PLAYING" and state != "PLAYING":
            self._mirror_observer.reset()
            self._last_completed_bar = None

    def _finish_crossed_bars(self, current_bar: int) -> None:
        if self._last_completed_bar is None:
            self._last_completed_bar = current_bar - 1
        while self._last_completed_bar < current_bar - 1:
            self._controller.note_bar_completed()
            self._mirror_observer.finish_bar(self._controller)
            self._last_completed_bar += 1

    def _plan_horizon(self, current_bar: int) -> None:
        now = self._clock()
        for bar_index in range(
            current_bar, current_bar + self._planning_horizon_bars
        ):
            events = plan_anchor_ledger(
                self._controller, self._config, bar_index, now
            )
            events.extend(
                plan_mirror_ledger(
                    self._controller, self._config, bar_index, now
                )
            )
            # Never create a surprise mid-bar catch-up burst. Past events are
            # omitted; an event exactly on this tick remains eligible.
            self._scheduler.enqueue(
                [event for event in events if event.deadline >= now]
            )

    def _snapshot(self, control: ControllerSnapshot) -> RuntimeSnapshot:
        return RuntimeSnapshot(
            controller=control,
            queue_depth=self._scheduler.queue_depth,
            total_emitted=self._scheduler.diag.total_emitted,
            total_dropped=self._scheduler.diag.total_dropped,
            total_late=self._scheduler.diag.total_late,
        )


class ScheduledMidiSink:
    """Adapt the existing ``MidiOut`` interface to ``LiveScheduler``."""

    def __init__(self, midi_out: object) -> None:
        self._midi_out = midi_out
        self._open = True

    def send_scheduled(
        self,
        note: int,
        velocity: int,
        channel: int,
        deadline: float,
    ) -> None:
        del channel, deadline  # MidiOut already owns the GM drum channel.
        if not self._open:
            raise RuntimeError("MIDI sink is closed")
        self._midi_out.send_note(note, velocity)  # type: ignore[attr-defined]

    def send_note(self, note: int, velocity: int, channel: int = 9) -> None:
        del channel
        self.send_scheduled(note, velocity, 9, 0.0)

    def close(self) -> None:
        if self._open:
            self._midi_out.close()  # type: ignore[attr-defined]
            self._open = False

    @property
    def is_open(self) -> bool:
        return self._open
