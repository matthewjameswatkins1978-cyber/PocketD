"""Long-form, pre-hardware acceptance tests for the Bunny live slice.

These tests exercise musical timelines across controller, planners and
scheduler.  They deliberately assert externally useful invariants rather
than controller counters or diagnostic wording.
"""

from __future__ import annotations

import random

import pytest

from drummer.live_controller import LiveController
from drummer.live_models import BarAdapterState, LedgerEvent, LiveConfig, PulseAdapterState
from drummer.live_scheduler import LiveScheduler
from drummer.straight_pocket import (
    KickMirrorObserver,
    SlotObservation,
    plan_anchor_ledger,
    plan_mirror_ledger,
)
from tests.fake_clock import FakeClock
from tests.fake_midi import FakeMidiSink


def _config(**overrides: object) -> LiveConfig:
    values: dict[str, object] = {
        "entry_min_evidence_beats": 2.0,
        "degradation_dwell_beats": 1.0,
        "recovery_dwell_beats": 1.0,
        "silence_stop_timeout": 4.0,
        "silence_grace_beats": 8.0,
        "late_budget_seconds": 0.010,
    }
    values.update(overrides)
    return LiveConfig(**values)


def _pulse(
    clock: FakeClock,
    *,
    bpm: float | None = 120.0,
    confidence: float = 0.80,
    margin: float = 0.50,
    evidence_age: float = 0.0,
) -> PulseAdapterState:
    period = 60.0 / bpm if bpm else None
    return PulseAdapterState(
        observed_at=clock.now() - evidence_age,
        computed_at=clock.now(),
        winning_bpm=bpm,
        winning_confidence=confidence,
        runner_up_bpm=60.0 if bpm else None,
        runner_up_confidence=max(0.0, confidence - margin),
        ambiguity_margin=margin,
        hypothesis_count=2 if bpm else 0,
        support_count=16 if bpm else 0,
        evidence_age=evidence_age,
        predicted_next_beat=clock.now() + period if period else None,
        beat_period=period,
        stability="locked" if bpm else "unknown",
    )


def _bar(
    clock: FakeClock,
    *,
    epoch: float,
    bpm: float = 120.0,
    confidence: float = 0.80,
    margin: float = 0.50,
    evidence_age: float = 0.0,
    is_confident: bool = True,
) -> BarAdapterState:
    return BarAdapterState(
        observed_at=clock.now() - evidence_age,
        computed_at=clock.now(),
        winning_bpm=bpm,
        winning_confidence=confidence,
        runner_up_confidence=max(0.0, confidence - margin),
        ambiguity_margin=margin,
        hypothesis_count=2,
        support_count=12,
        estimated_beat_in_bar=0,
        bar_position=0.0,
        downbeat_time=epoch,
        bar_duration=4.0 * 60.0 / bpm,
        evidence_age=evidence_age,
        is_confident=is_confident,
    )


def _enter_playing(
    controller: LiveController,
    clock: FakeClock,
    *,
    epoch: float,
    bpm: float = 120.0,
) -> None:
    """Drive good evidence until the future downbeat lock is reached."""
    for _ in range(20):
        controller.update(_pulse(clock, bpm=bpm), _bar(clock, epoch=epoch, bpm=bpm))
        if controller.state == "PLAYING":
            return
        clock.advance(0.25)
    pytest.fail(f"controller failed to enter PLAYING; final state={controller.state}")


def _fire_bar(
    scheduler: LiveScheduler,
    clock: FakeClock,
    events: list[LedgerEvent],
) -> None:
    """Fire a bar exactly on its distinct deadlines."""
    scheduler.enqueue(events)
    scheduler.enqueue(events)  # every planning pass may overlap the previous one
    for deadline in sorted({event.deadline for event in events}):
        clock.set(deadline)
        scheduler.fire_due_events()


def test_silent_intro_then_eight_clean_bars_end_to_end() -> None:
    clock = FakeClock(100.0)
    cfg = _config()
    controller = LiveController(cfg, clock=clock.now)
    sink = FakeMidiSink(clock.now)
    scheduler = LiveScheduler(cfg, sink, clock=clock.now)
    epoch = 100.0

    # Uncertain intro must remain completely silent.
    for _ in range(12):
        snapshot = controller.update(
            _pulse(clock, confidence=0.20, margin=0.05),
            _bar(clock, epoch=epoch, confidence=0.20, margin=0.05, is_confident=False),
        )
        assert snapshot.state == "LISTENING"
        assert plan_anchor_ledger(controller, cfg, 0, clock.now()) == []
        clock.advance(0.25)
    assert sink.events == []

    _enter_playing(controller, clock, epoch=epoch)
    assert controller.bar_epoch is not None
    first_bar = controller.bar_index

    expected_events = 0
    for bar_index in range(first_bar, first_bar + 8):
        events = plan_anchor_ledger(controller, cfg, bar_index, clock.now())
        expected_events += len(events)
        _fire_bar(scheduler, clock, events)

    assert scheduler.diag.total_dropped == 0
    assert scheduler.diag.total_emitted == expected_events == 8 * 12
    assert len(sink.events) == expected_events
    assert all(event.fired_at == pytest.approx(event.deadline) for event in sink.events)


def test_tracker_wobble_cannot_move_the_playing_grid() -> None:
    clock = FakeClock(200.0)
    cfg = _config()
    controller = LiveController(cfg, clock=clock.now)
    _enter_playing(controller, clock, epoch=200.0)
    locked_bpm = controller.locked_bpm
    locked_epoch = controller.bar_epoch

    for index in range(64):
        clock.advance(0.125)
        bpm = 120.0 + ((index % 9) - 4) * 0.25
        snapshot = controller.update(
            _pulse(clock, bpm=bpm, confidence=0.72, margin=0.35),
            _bar(clock, epoch=200.0, bpm=bpm, confidence=0.72, margin=0.35),
        )
        assert snapshot.state == "PLAYING"
        assert controller.locked_bpm == locked_bpm
        assert controller.bar_epoch == locked_epoch


def test_long_confident_but_ambiguous_timeline_never_arms() -> None:
    clock = FakeClock(300.0)
    cfg = _config()
    controller = LiveController(cfg, clock=clock.now)

    for _ in range(256):
        snapshot = controller.update(
            _pulse(clock, confidence=0.95, margin=0.01),
            _bar(clock, epoch=300.0, confidence=0.95, margin=0.01),
        )
        assert snapshot.state == "LISTENING"
        clock.advance(0.0625)

    assert controller.generation == 0
    assert controller.locked_bpm is None


def test_dropout_invalidates_every_pending_note_and_recovery_keeps_grid() -> None:
    clock = FakeClock(400.0)
    cfg = _config()
    controller = LiveController(cfg, clock=clock.now)
    sink = FakeMidiSink(clock.now)
    scheduler = LiveScheduler(cfg, sink, clock=clock.now)
    _enter_playing(controller, clock, epoch=400.0)
    locked_grid = (controller.locked_bpm, controller.bar_epoch)

    future_bar = controller.bar_index + 2
    scheduler.enqueue(plan_anchor_ledger(controller, cfg, future_bar, clock.now()))
    assert scheduler.queue_depth == 12

    for _ in range(3):
        clock.advance(0.5)
        controller.update(
            _pulse(clock, confidence=0.05, evidence_age=1.0),
            _bar(clock, epoch=400.0, confidence=0.05, evidence_age=1.0, is_confident=False),
        )
        if controller.state == "DEGRADED":
            break
    assert controller.state == "DEGRADED"
    assert scheduler.invalidate_generation(controller.generation) == 12

    # Even after the old deadlines, invalidated notes cannot escape.
    clock.set(400.0 + (future_bar + 1) * 2.0)
    scheduler.fire_due_events()
    assert sink.events == []

    for _ in range(4):
        clock.advance(0.5)
        controller.update(_pulse(clock), _bar(clock, epoch=400.0))
        if controller.state == "PLAYING":
            break
    assert controller.state == "PLAYING"
    assert (controller.locked_bpm, controller.bar_epoch) == locked_grid


def test_replanning_after_emission_never_double_fires_an_event_id() -> None:
    clock = FakeClock(500.0)
    cfg = _config()
    sink = FakeMidiSink(clock.now)
    scheduler = LiveScheduler(cfg, sink, clock=clock.now)
    event = LedgerEvent("anchor:bar0:slot0", 500.0, 36, 100, 9, 0, 0, "anchor", 0)

    scheduler.enqueue([event])
    scheduler.fire_due_events()
    for _ in range(20):
        scheduler.enqueue([event])
        scheduler.fire_due_events()

    assert scheduler.diag.total_emitted == 1
    assert len(sink.events) == 1


def test_mirror_requires_the_same_non_anchor_slot_in_two_consecutive_bars() -> None:
    clock = FakeClock(600.0)
    cfg = _config(
        mirror_min_stable_bars=2,
        mirror_min_sample_count=2,
        mirror_strength_percentile=0.0,
    )
    controller = LiveController(cfg, clock=clock.now)
    observer = KickMirrorObserver(cfg)
    controller.note_bar_completed()
    observer.observe(SlotObservation(0, 3, 0.0, 0.9, 600.0))
    assert observer.finish_bar(controller) is None

    # A different syncopation in bar two must not activate anything.
    controller.note_bar_completed()
    observer.observe(SlotObservation(1, 5, 0.0, 0.9, 602.0))
    assert observer.finish_bar(controller) is None
    assert not controller.mirror_active

    # Slot 5 repeated in the following bar is now stable enough.
    controller.note_bar_completed()
    observer.observe(SlotObservation(2, 5, 0.0, 0.9, 604.0))
    assert observer.finish_bar(controller) == 5
    assert controller.mirror_slot == 5


def test_mirror_and_anchor_plans_coexist_without_duplicate_ids() -> None:
    clock = FakeClock(700.0)
    cfg = _config()
    controller = LiveController(cfg, clock=clock.now)
    _enter_playing(controller, clock, epoch=700.0)
    controller.set_mirror(3)

    events = plan_anchor_ledger(controller, cfg, controller.bar_index, clock.now())
    events += plan_mirror_ledger(controller, cfg, controller.bar_index, clock.now())
    ids = [event.event_id for event in events]
    assert len(events) == 13
    assert len(ids) == len(set(ids))
    assert sum(event.source == "mirror" for event in events) == 1


@pytest.mark.parametrize("seed", range(12))
def test_randomized_scheduler_never_double_fires_or_emits_invalid_generation(seed: int) -> None:
    rng = random.Random(seed)
    clock = FakeClock(800.0)
    cfg = _config(late_budget_seconds=0.020)
    sink = FakeMidiSink(clock.now)
    scheduler = LiveScheduler(cfg, sink, clock=clock.now)
    generation = 0
    submitted_keys: set[tuple[int, str]] = set()

    for step in range(300):
        if rng.random() < 0.08:
            generation += 1
            scheduler.invalidate_generation(generation)

        event_id = f"event-{rng.randrange(30)}"
        event = LedgerEvent(
            event_id=event_id,
            deadline=clock.now() + rng.choice((0.0, 0.005, 0.010)),
            note=36 + rng.randrange(3),
            velocity=80,
            channel=9,
            bar_index=step // 16,
            slot=step % 16,
            source="stress",
            generation=generation,
        )
        submitted_keys.add((generation, event_id))
        scheduler.enqueue([event, event])
        clock.advance(rng.choice((0.0, 0.005, 0.010)))
        scheduler.fire_due_events()

    # Every accepted key can emit at most once. Invalidated and not-yet-due
    # keys make this an upper bound rather than an equality.
    assert scheduler.diag.total_emitted <= len(submitted_keys)
    assert all(event.fired_at - event.deadline <= cfg.late_budget_seconds for event in sink.events)
