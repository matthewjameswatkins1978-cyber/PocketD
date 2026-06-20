"""Hardware-free integration tests for the Bunny live runtime."""

from __future__ import annotations

from dataclasses import dataclass

from drummer.live_controller import LiveController
from drummer.live_models import BarAdapterState, LiveConfig, PulseAdapterState
from drummer.live_runtime import LiveRuntime, ScheduledMidiSink
from drummer.live_scheduler import LiveScheduler
from tests.fake_clock import FakeClock
from tests.fake_midi import FakeMidiSink


@dataclass
class _PulseAdapter:
    state: PulseAdapterState
    calls: int = 0

    def adapt(self) -> PulseAdapterState:
        self.calls += 1
        return self.state


@dataclass
class _BarAdapter:
    state: BarAdapterState
    calls: int = 0

    def adapt(self, reference_bpm: float | None = None) -> BarAdapterState:
        del reference_bpm
        self.calls += 1
        return self.state


def _config(**overrides: object) -> LiveConfig:
    values: dict[str, object] = {
        "entry_min_evidence_beats": 1.0,
        "degradation_dwell_beats": 0.5,
        "recovery_dwell_beats": 0.5,
        "mirror_min_stable_bars": 2,
        "mirror_min_sample_count": 2,
        "mirror_strength_percentile": 0.0,
    }
    values.update(overrides)
    return LiveConfig(**values)


def _pulse(clock: FakeClock, confidence: float = 0.8, age: float = 0.0) -> PulseAdapterState:
    return PulseAdapterState(
        observed_at=clock.now() - age,
        computed_at=clock.now(),
        winning_bpm=120.0,
        winning_confidence=confidence,
        runner_up_bpm=60.0,
        runner_up_confidence=0.1,
        ambiguity_margin=max(0.0, confidence - 0.1),
        hypothesis_count=2,
        support_count=16,
        evidence_age=age,
        predicted_next_beat=clock.now() + 0.5,
        beat_period=0.5,
        stability="locked",
    )


def _bar(clock: FakeClock, epoch: float, confidence: float = 0.8, age: float = 0.0) -> BarAdapterState:
    return BarAdapterState(
        observed_at=clock.now() - age,
        computed_at=clock.now(),
        winning_bpm=120.0,
        winning_confidence=confidence,
        runner_up_confidence=0.1,
        ambiguity_margin=max(0.0, confidence - 0.1),
        hypothesis_count=2,
        support_count=12,
        estimated_beat_in_bar=0,
        bar_position=0.0,
        downbeat_time=epoch,
        bar_duration=2.0,
        evidence_age=age,
        is_confident=confidence >= 0.4,
    )


def _rig(start: float = 100.0):
    clock = FakeClock(start)
    cfg = _config()
    pulse = _PulseAdapter(_pulse(clock))
    bar = _BarAdapter(_bar(clock, start))
    controller = LiveController(cfg, clock.now)
    sink = FakeMidiSink(clock.now)
    scheduler = LiveScheduler(cfg, sink, clock.now)
    runtime = LiveRuntime(cfg, pulse, bar, controller, scheduler, clock.now)
    return clock, pulse, bar, controller, sink, scheduler, runtime


def _enter(clock, pulse, bar, controller, runtime, epoch: float) -> None:
    for _ in range(20):
        pulse.state = _pulse(clock)
        bar.state = _bar(clock, epoch)
        runtime.tick()
        if controller.state == "PLAYING":
            return
        clock.advance(0.25)
    raise AssertionError("runtime did not enter PLAYING")


def test_tick_wires_adapters_and_stays_silent_before_entry() -> None:
    clock, pulse, bar, controller, sink, scheduler, runtime = _rig()
    pulse.state = _pulse(clock, confidence=0.2)
    bar.state = _bar(clock, 100.0, confidence=0.2)

    snapshot = runtime.tick()

    assert pulse.calls == bar.calls == 1
    assert snapshot.controller.state == "LISTENING"
    assert scheduler.queue_depth == 0
    assert sink.events == []


def test_entry_plans_two_bars_and_fires_downbeat_once() -> None:
    clock, pulse, bar, controller, sink, scheduler, runtime = _rig()
    _enter(clock, pulse, bar, controller, runtime, 100.0)

    # The entry tick fires kick + hat at slot zero and leaves the
    # remainder of the two-bar horizon queued.
    assert len(sink.events) == 2
    assert scheduler.queue_depth == 22
    runtime.tick()
    assert len(sink.events) == 2
    assert scheduler.queue_depth == 22


def test_degradation_replans_anchor_queue_without_stopping() -> None:
    clock, pulse, bar, controller, sink, scheduler, runtime = _rig(200.0)
    _enter(clock, pulse, bar, controller, runtime, 200.0)
    assert scheduler.queue_depth > 0

    clock.advance(0.5)
    pulse.state = _pulse(clock, confidence=0.05, age=1.0)
    bar.state = _bar(clock, 200.0, confidence=0.05, age=1.0)
    runtime.tick()

    assert controller.state == "DEGRADED"
    assert scheduler.queue_depth > 0


def test_runtime_activates_mirror_only_for_following_repeated_bar() -> None:
    clock, pulse, bar, controller, sink, scheduler, runtime = _rig(300.0)
    _enter(clock, pulse, bar, controller, runtime, 300.0)
    epoch = controller.bar_epoch
    assert epoch is not None

    # Slot 3 in bar zero.
    assert runtime.observe_player_hit(epoch + 3 * 0.125, 0.9)
    clock.set(epoch + 2.0)
    pulse.state = _pulse(clock)
    bar.state = _bar(clock, epoch)
    runtime.tick()
    assert not controller.mirror_active

    # Repeat slot 3 in bar one; activation happens as bar two begins.
    assert runtime.observe_player_hit(epoch + 2.0 + 3 * 0.125, 0.9)
    clock.set(epoch + 4.0)
    pulse.state = _pulse(clock)
    bar.state = _bar(clock, epoch)
    runtime.tick()
    assert controller.mirror_slot == 3


def test_stop_invalidates_notes_and_closes_sink() -> None:
    clock, pulse, bar, controller, sink, scheduler, runtime = _rig(400.0)
    _enter(clock, pulse, bar, controller, runtime, 400.0)
    assert scheduler.queue_depth > 0

    snapshot = runtime.stop()

    assert snapshot.controller.state == "STOPPED"
    assert scheduler.queue_depth == 0
    assert not sink.is_open


class _MidiOutStub:
    def __init__(self) -> None:
        self.notes: list[tuple[int, int]] = []
        self.closed = False

    def send_note(self, note: int, velocity: int) -> None:
        self.notes.append((note, velocity))

    def close(self) -> None:
        self.closed = True


def test_existing_midi_out_adapter() -> None:
    output = _MidiOutStub()
    sink = ScheduledMidiSink(output)
    sink.send_scheduled(36, 91, 9, 123.0)
    sink.close()

    assert output.notes == [(36, 91)]
    assert output.closed
    assert not sink.is_open
