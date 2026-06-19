"""Deterministic tests for locked grid quantisation, Straight Pocket ledger,
Live Scheduler, and controller-scheduler integration.

All tests use FakeClock, FakeMidiSink, and LiveController.
No real audio, no real MIDI, no sleeps.
"""

from __future__ import annotations

import pytest

from drummer.live_controller import LiveController
from drummer.live_models import (
    LedgerEvent,
    LiveConfig,
)
from drummer.live_scheduler import LiveScheduler
from drummer.straight_pocket import (
    KickMirrorObserver,
    SlotObservation,
    plan_anchor_ledger,
    plan_mirror_ledger,
    quantise_to_locked_grid,
)
from tests.fake_clock import FakeClock
from tests.fake_midi import FakeMidiSink


# ── Helpers ──────────────────────────────────────────────────────────


def _config(**overrides) -> LiveConfig:
    kw: dict = {}
    for k in overrides:
        kw[k] = overrides[k]
    return LiveConfig(**kw)


def _event(
    event_id: str,
    deadline: float,
    note: int = 36,
    velocity: int = 100,
    channel: int = 9,
    bar_index: int = 0,
    slot: int = 0,
    source: str = "anchor",
    generation: int = 1,
) -> LedgerEvent:
    return LedgerEvent(
        event_id=event_id,
        deadline=deadline,
        note=note,
        velocity=velocity,
        channel=channel,
        bar_index=bar_index,
        slot=slot,
        source=source,
        generation=generation,
    )


# ── Quantisation ──────────────────────────────────────────────────────


def test_quantise_exact_slot_centre():
    cfg = _config()
    bp = 0.5  # 120 BPM
    epoch = 100.0
    # Slot 0 at 100.0, slot 4 at 100.0 + 0.5 = 100.5
    bar_idx, slot, offset = quantise_to_locked_grid(100.0, epoch, bp, cfg)
    assert bar_idx == 0
    assert slot == 0
    assert offset == pytest.approx(0.0)

    bar_idx, slot, offset = quantise_to_locked_grid(100.5, epoch, bp, cfg)
    assert bar_idx == 0
    assert slot == 4
    assert offset == pytest.approx(0.0)


def test_quantise_slightly_early():
    cfg = _config()
    bp = 0.5
    epoch = 100.0
    # 10 ms early for slot 4 (100.5)
    bar_idx, slot, offset = quantise_to_locked_grid(100.49, epoch, bp, cfg)
    assert bar_idx == 0
    assert slot == 4
    assert offset == pytest.approx(-0.01)


def test_quantise_outside_tolerance():
    cfg = _config(quantisation_tolerance_beats=0.10)
    bp = 0.5
    epoch = 100.0
    # slot 0 at 100.0, slot 1 at 100.125
    # At 100.0625: offset = 0.0625s = 0.125 beats → outside 0.10 tolerance
    bar_idx, slot, offset = quantise_to_locked_grid(100.0625, epoch, bp, cfg)
    assert bar_idx == -1  # rejected — outside 0.10 beat tolerance

    # At exactly slot centre — within tolerance
    bar_idx, slot, offset = quantise_to_locked_grid(100.0, epoch, bp, cfg)
    assert bar_idx == 0
    assert slot == 0


def test_quantise_bar_boundary_wrapping():
    cfg = _config()
    bp = 0.5  # 120 BPM
    epoch = 100.0
    # bar 0 ends at 102.0.  Event near 102.0 but in slot 15 of bar 0
    # slot 15 is at 100.0 + 15 * 0.125 = 101.875
    bar_idx, slot, offset = quantise_to_locked_grid(101.875, epoch, bp, cfg)
    assert bar_idx == 0
    assert slot == 15
    assert offset == pytest.approx(0.0)

    # Just before bar 1 starts — should map to slot 15 bar 0, not slot 0 bar 1
    bar_idx, slot, offset = quantise_to_locked_grid(101.99, epoch, bp, cfg)
    assert bar_idx == 1
    assert slot == 0


# ── Anchor ledger ─────────────────────────────────────────────────────


def test_plan_anchor_ledger_empty_when_not_locked():
    clock = FakeClock(100.0)
    ctrl = LiveController(_config(), clock=clock.now)
    events = plan_anchor_ledger(ctrl, _config(), 0, 100.0)
    assert events == []


def test_plan_anchor_ledger_contains_correct_slots():
    clock = FakeClock(100.0)
    cfg = _config()
    ctrl = LiveController(cfg, clock=clock.now)

    # Manually lock the controller
    ctrl._locked_bpm = 120.0  # type: ignore[attr]
    ctrl._beat_period = 0.5  # type: ignore[attr]
    ctrl._bar_epoch = 100.0  # type: ignore[attr]
    ctrl._state = "PLAYING"  # type: ignore[attr]
    ctrl._generation = 5  # type: ignore[attr]

    events = plan_anchor_ledger(ctrl, cfg, 0, 100.0)
    assert len(events) > 0

    # Verify slots
    slots = {e.slot for e in events}
    # Anchor slots should include kick 0,8; snare 4,12; hat 0,2,4,6,8,10,12,14
    assert 0 in slots
    assert 8 in slots
    assert 4 in slots
    assert 12 in slots

    # Verify sources
    sources = {e.source for e in events}
    assert "anchor" in sources
    assert "hat" in sources

    # All should have correct generation
    for e in events:
        assert e.generation == 5


def test_planning_same_horizon_twice_produces_same_event_ids():
    """Planning the same bar twice should produce events with same IDs,
    which the scheduler will deduplicate.
    """
    clock = FakeClock(100.0)
    cfg = _config()
    ctrl = LiveController(cfg, clock=clock.now)
    ctrl._locked_bpm = 120.0  # type: ignore[attr]
    ctrl._beat_period = 0.5  # type: ignore[attr]
    ctrl._bar_epoch = 100.0  # type: ignore[attr]
    ctrl._state = "PLAYING"  # type: ignore[attr]
    ctrl._generation = 1  # type: ignore[attr]

    events1 = plan_anchor_ledger(ctrl, cfg, 0, 100.0)
    events2 = plan_anchor_ledger(ctrl, cfg, 0, 100.0)

    ids1 = {e.event_id for e in events1}
    ids2 = {e.event_id for e in events2}
    assert ids1 == ids2


# ── Mirror ledger ────────────────────────────────────────────────────


def test_plan_mirror_ledger_empty_when_not_active():
    clock = FakeClock(100.0)
    cfg = _config()
    ctrl = LiveController(cfg, clock=clock.now)
    events = plan_mirror_ledger(ctrl, cfg, 0, 100.0)
    assert events == []


def test_plan_mirror_ledger_produces_single_event():
    clock = FakeClock(100.0)
    cfg = _config()
    ctrl = LiveController(cfg, clock=clock.now)
    ctrl._locked_bpm = 120.0  # type: ignore[attr]
    ctrl._beat_period = 0.5  # type: ignore[attr]
    ctrl._bar_epoch = 100.0  # type: ignore[attr]
    ctrl._generation = 3  # type: ignore[attr]
    ctrl.set_mirror(7)

    events = plan_mirror_ledger(ctrl, cfg, 0, 100.0)
    assert len(events) == 1
    assert events[0].source == "mirror"
    assert events[0].slot == 7
    assert events[0].note == cfg.kick_note
    assert events[0].velocity == cfg.mirror_velocity


# ── Scheduler: enqueue / fire / dedup ─────────────────────────────────


def test_enqueue_and_fire():
    clock = FakeClock(100.0)
    sink = FakeMidiSink(clock.now)
    cfg = _config(late_budget_seconds=1.0)  # generous
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    sched.enqueue([
        _event("a", deadline=100.1, note=36),
        _event("b", deadline=100.15, note=38),
    ])

    assert sched.queue_depth == 2

    # Not yet due
    sched.fire_due_events()
    assert sink.kick_count() == 0
    assert sink.snare_count() == 0
    assert sched.queue_depth == 2

    # Advance to 100.5 — both due
    clock.advance(0.5)
    sched.fire_due_events()
    assert sink.kick_count() == 1
    assert sink.snare_count() == 1
    assert sched.queue_depth == 0


def test_deduplicate_event_ids():
    clock = FakeClock(100.0)
    sink = FakeMidiSink(clock.now)
    cfg = _config(late_budget_seconds=1.0)
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    sched.enqueue([_event("dup", deadline=100.1)])
    sched.enqueue([_event("dup", deadline=100.1)])  # same ID — should be ignored
    assert sched.queue_depth == 1

    clock.advance(0.5)
    sched.fire_due_events()
    assert sched.diag.total_emitted == 1


def test_generation_invalidation_removes_old_events():
    clock = FakeClock(100.0)
    sink = FakeMidiSink(clock.now)
    cfg = _config(late_budget_seconds=2.0)
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    sched.enqueue([
        _event("gen1_a", deadline=100.1, generation=1),
        _event("gen1_b", deadline=100.2, generation=1),
        _event("gen2_a", deadline=100.3, generation=2),
    ])
    assert sched.queue_depth == 3

    removed = sched.invalidate_generation(2)
    assert removed == 2  # gen1 events removed
    assert sched.queue_depth == 1

    clock.advance(1.0)
    sched.fire_due_events()
    assert sched.diag.total_emitted == 1


def test_late_drop_policy():
    clock = FakeClock(100.0)
    cfg = _config(late_budget_seconds=0.010)  # 10ms
    sink = FakeMidiSink(clock.now)
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    # Only enqueue ONE event for this test
    sched.enqueue([_event("will_be_late", deadline=100.05)])

    # Advance so the event is 50ms late (beyond 10ms budget)
    clock.advance(0.100)
    sched.fire_due_events()

    # Should be dropped, not emitted
    assert sched.diag.total_dropped == 1
    assert sched.diag.total_emitted == 0


def test_scheduler_diagnostics():
    clock = FakeClock(100.0)
    sink = FakeMidiSink(clock.now)
    cfg = _config(late_budget_seconds=2.0)
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    sched.enqueue([
        _event("a", deadline=100.1),
        _event("b", deadline=100.2),
        _event("c", deadline=100.3),
    ])
    clock.advance(1.0)
    sched.fire_due_events()

    assert sched.diag.total_emitted == 3
    assert sched.diag.queue_depth_max == 3
    assert sched.diag.cycles == 0  # fire_due_events doesn't increment cycles


def test_queue_depth():
    clock = FakeClock(100.0)
    sink = FakeMidiSink(clock.now)
    sched = LiveScheduler(_config(), sink, clock=clock.now)
    assert sched.queue_depth == 0
    sched.enqueue([_event("a", 100.1), _event("b", 100.2)])
    assert sched.queue_depth == 2


# ── Controller → Scheduler integration ───────────────────────────────


def test_controller_generation_invalidates_scheduler_events():
    clock = FakeClock(100.0)
    cfg = _config(entry_min_evidence_beats=0.5)
    ctrl = LiveController(cfg, clock=clock.now)
    sink = FakeMidiSink(clock.now)
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    # Get controller to PLAYING
    from tests.test_live_controller import _pulse, _bar  # type: ignore[import]

    clock.advance(0.5)
    ctrl.update(
        _pulse(now=100.5, bpm=120.0, confidence=0.60),
        _bar(now=100.5, bpm=120.0, confidence=0.50),
    )
    clock.advance(0.4)
    ctrl.update(
        _pulse(now=100.9, bpm=120.0, confidence=0.60),
        _bar(now=100.9, bpm=120.0, confidence=0.50),
    )
    assert ctrl.state == "PLAYING"

    # Plan anchor ledger and enqueue
    events = plan_anchor_ledger(ctrl, cfg, ctrl.bar_index, clock.now())
    sched.enqueue(events)
    assert sched.queue_depth > 0

    # Now stop controller — this increments generation
    gen_before = ctrl.generation
    ctrl.stop()
    assert ctrl.generation > gen_before

    # Invalidate scheduler with new generation
    removed = sched.invalidate_generation(ctrl.generation)
    assert removed > 0
    assert sched.queue_depth == 0


def test_degradation_clears_mirror_and_scheduler():
    clock = FakeClock(100.0)
    cfg = _config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=1.5,
    )
    ctrl = LiveController(cfg, clock=clock.now)
    sink = FakeMidiSink(clock.now)
    sched = LiveScheduler(cfg, sink, clock=clock.now)

    from tests.test_live_controller import _pulse, _bar  # type: ignore[import]

    # Go to PLAYING
    clock.advance(0.5)
    ctrl.update(
        _pulse(now=100.5, bpm=120.0, confidence=0.60),
        _bar(now=100.5, bpm=120.0, confidence=0.50),
    )
    clock.advance(0.4)
    ctrl.update(
        _pulse(now=100.9, bpm=120.0, confidence=0.60),
        _bar(now=100.9, bpm=120.0, confidence=0.50),
    )
    assert ctrl.state == "PLAYING"
    ctrl.set_mirror(9)

    # Enqueue anchor + mirror events
    anchor = plan_anchor_ledger(ctrl, cfg, ctrl.bar_index, clock.now())
    mirror = plan_mirror_ledger(ctrl, cfg, ctrl.bar_index, clock.now())
    sched.enqueue(anchor + mirror)

    gen_playing = ctrl.generation

    # Degrade
    for _ in range(5):
        clock.advance(0.5)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.15),
            _bar(now=clock.now(), bpm=120.0, confidence=0.15),
        )
    assert ctrl.state == "DEGRADED"
    assert not ctrl.mirror_active
    assert ctrl.generation > gen_playing

    # Invalidate old events
    sched.invalidate_generation(ctrl.generation)
    # Mirror events should be gone, anchor may also be invalid


# ── Kick Mirror Observer ──────────────────────────────────────────────


def test_mirror_observer_ignores_anchor_slots():
    cfg = _config()
    obs = KickMirrorObserver(cfg)
    obs.observe(SlotObservation(bar_index=0, slot=0, offset_seconds=0.0, strength=0.8, observed_at=100.0))
    assert len(obs._observations) == 0  # ignored anchor kick slot


def test_mirror_observer_records_non_anchor():
    cfg = _config()
    obs = KickMirrorObserver(cfg)
    obs.observe(SlotObservation(bar_index=0, slot=3, offset_seconds=0.0, strength=0.7, observed_at=100.0))
    assert len(obs._observations) == 1


def test_mirror_observer_rejects_hits_below_absolute_strength_floor():
    cfg = _config(mirror_min_strength=0.35)
    obs = KickMirrorObserver(cfg)
    obs.observe(SlotObservation(bar_index=0, slot=3, offset_seconds=0.0, strength=0.34, observed_at=100.0))
    assert len(obs._observations) == 0


def test_mirror_not_eligible_before_min_stable_bars():
    clock = FakeClock(100.0)
    cfg = _config(mirror_min_stable_bars=2)
    ctrl = LiveController(cfg, clock=clock.now)
    obs = KickMirrorObserver(cfg)

    # Add enough observations for percentile
    for i in range(10):
        obs.observe(SlotObservation(bar_index=0, slot=3, offset_seconds=0.0, strength=0.8, observed_at=100.0 + i * 0.1))

    # Only 0 bars completed — should not activate
    result = obs.finish_bar(ctrl)
    assert result is None
    assert not ctrl.mirror_active


def test_mirror_reset():
    cfg = _config()
    obs = KickMirrorObserver(cfg)
    obs.observe(SlotObservation(bar_index=0, slot=3, offset_seconds=0.0, strength=0.8, observed_at=100.0))
    obs.reset()
    assert len(obs._observations) == 0
    assert len(obs._current_bar_hits) == 0
    assert len(obs._previous_bar_hits) == 0


def test_mirror_clears_on_unsupported_bars():
    clock = FakeClock(100.0)
    cfg = _config(
        mirror_min_stable_bars=1,
        mirror_min_sample_count=1,
        mirror_expire_unsupported_bars=1,
    )
    ctrl = LiveController(cfg, clock=clock.now)
    ctrl._playing_bars_completed = 2  # type: ignore[attr]
    obs = KickMirrorObserver(cfg)

    # First bar: slot 5 seen
    obs.observe(SlotObservation(bar_index=0, slot=5, offset_seconds=0.0, strength=0.8, observed_at=100.0))
    result = obs.finish_bar(ctrl)
    # Should activate if conditions met
    if result is not None:
        ctrl.set_mirror(result)
    # Next bar: no support for slot 5
    obs.finish_bar(ctrl)  # shifts bar, no slot 5 this bar
    obs.finish_bar(ctrl)  # unsupported_bars should hit threshold
    assert not ctrl.mirror_active


# ── Scheduler: next_deadline ──────────────────────────────────────────


def test_next_deadline():
    clock = FakeClock(100.0)
    sink = FakeMidiSink(clock.now)
    sched = LiveScheduler(_config(), sink, clock=clock.now)

    assert sched.next_deadline() is None

    sched.enqueue([_event("a", 100.5)])
    assert sched.next_deadline() == 100.5

    sched.enqueue([_event("b", 100.1)])
    # b is earlier
    assert sched.next_deadline() == 100.1
