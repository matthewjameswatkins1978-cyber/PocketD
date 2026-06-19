"""Deterministic state machine tests for LiveController.

All tests use FakeClock.  No audio, no MIDI, no real sleep.
"""

from __future__ import annotations

import pytest

from drummer.live_controller import LiveController
from drummer.live_models import (
    BarAdapterState,
    ControllerSnapshot,
    LiveConfig,
    PulseAdapterState,
)
from tests.fake_clock import FakeClock


# ── Helpers ──────────────────────────────────────────────────────────


def _make_config(**overrides) -> LiveConfig:
    kwargs = {}
    for field_name in (
        "entry_confidence_threshold",
        "entry_min_evidence_beats",
        "entry_ambiguity_margin",
        "exit_confidence_threshold",
        "degradation_dwell_beats",
        "recovery_confidence_threshold",
        "recovery_dwell_beats",
        "silence_stop_timeout",
        "silence_grace_beats",
        "max_evidence_age_beats",
        "tempo_drift_fraction",
        "tempo_drift_dwell_beats",
    ):
        if field_name in overrides:
            kwargs[field_name] = overrides[field_name]
    return LiveConfig(**kwargs)


def _pulse(
    *,
    now: float = 0.0,
    bpm: float = 120.0,
    confidence: float = 0.50,
    runner_confidence: float = 0.10,
    evidence_age: float = 0.1,
    support: int = 10,
    stability: str = "stable",
) -> PulseAdapterState:
    margin = confidence - runner_confidence
    beat_period = 60.0 / bpm
    next_beat = now + beat_period  # rough
    return PulseAdapterState(
        observed_at=now,
        computed_at=now,
        winning_bpm=bpm,
        winning_confidence=confidence,
        runner_up_bpm=bpm * 2.0 if runner_confidence > 0 else None,
        runner_up_confidence=runner_confidence,
        ambiguity_margin=margin,
        hypothesis_count=4,
        support_count=support,
        evidence_age=evidence_age,
        predicted_next_beat=next_beat,
        beat_period=beat_period,
        stability=stability,
    )


def _bar(
    *,
    now: float = 0.0,
    bpm: float = 120.0,
    confidence: float = 0.50,
    runner_confidence: float = 0.10,
    evidence_age: float = 0.1,
    is_confident: bool = True,
    downbeat_time: float | None = None,
    bar_position: float | None = None,
) -> BarAdapterState:
    margin = confidence - runner_confidence
    beat_period = 60.0 / bpm
    bar_duration = beat_period * 4
    # Default: next downbeat is imminent (~0.1s away), bar almost done
    if downbeat_time is None:
        downbeat_time = now - (bar_duration - 0.1)
    if bar_position is None:
        bar_position = 3.95  # near end of 4-beat bar
    return BarAdapterState(
        observed_at=now,
        computed_at=now,
        winning_bpm=bpm,
        winning_confidence=confidence,
        runner_up_confidence=runner_confidence,
        ambiguity_margin=margin,
        hypothesis_count=3,
        support_count=10,
        estimated_beat_in_bar=3,
        bar_position=bar_position,
        downbeat_time=downbeat_time,
        bar_duration=bar_duration,
        evidence_age=evidence_age,
        is_confident=is_confident,
    )


# ── Initial state ────────────────────────────────────────────────────


def test_initial_state_is_listening():
    clock = FakeClock(100.0)
    config = _make_config()
    ctrl = LiveController(config, clock=clock.now)
    assert ctrl.state == "LISTENING"
    assert ctrl.generation == 0
    assert ctrl.locked_bpm is None


# ── LISTENING → ARMED → PLAYING happy path ───────────────────────────


def test_listening_to_armed_to_playing():
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=1.0)
    ctrl = LiveController(config, clock=clock.now)

    # First update: confident pulse with low ambiguity, confident bar
    snap = ctrl.update(
        _pulse(now=100.0, bpm=120.0, confidence=0.60, runner_confidence=0.10),
        _bar(now=100.0, bpm=120.0, confidence=0.50, runner_confidence=0.10),
    )
    assert snap.state == "LISTENING"  # not enough evidence beats yet

    # Advance time to accumulate evidence beats
    clock.advance(1.0)
    snap = ctrl.update(
        _pulse(now=101.0, bpm=120.0, confidence=0.60, runner_confidence=0.10),
        _bar(now=101.0, bpm=120.0, confidence=0.50, runner_confidence=0.10),
    )
    assert snap.state == "ARMED"
    assert ctrl.locked_bpm is None  # not yet locked

    # Advance past the downbeat
    clock.advance(0.5)
    snap = ctrl.update(
        _pulse(now=101.5, bpm=120.0, confidence=0.60),
        _bar(now=101.5, bpm=120.0, confidence=0.50),
    )
    assert snap.state == "PLAYING"
    assert ctrl.locked_bpm == 120.0
    assert ctrl.beat_period == pytest.approx(0.5)
    assert ctrl.generation >= 1


# ── LISTENING: ambiguity gates entry ──────────────────────────────────


def test_high_confidence_but_ambiguous_tempo_stays_listening():
    """Entry requires both absolute confidence AND winner-runner margin."""
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=1.0)
    ctrl = LiveController(config, clock=clock.now)

    # Confidence is high but runner-up is nearly equal
    snap = ctrl.update(
        _pulse(now=100.0, bpm=120.0, confidence=0.80, runner_confidence=0.75),
        _bar(now=100.0, bpm=120.0, confidence=0.50, runner_confidence=0.10),
    )
    assert snap.state == "LISTENING"

    clock.advance(10.0)
    snap = ctrl.update(
        _pulse(now=110.0, bpm=120.0, confidence=0.80, runner_confidence=0.75),
        _bar(now=110.0, bpm=120.0, confidence=0.50),
    )
    assert snap.state == "LISTENING"  # still, margin too small


def test_ambiguous_bar_phase_blocks_entry():
    """High pulse confidence but bar has ambiguous runner-up → no entry."""
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=1.0, entry_ambiguity_margin=0.15)
    ctrl = LiveController(config, clock=clock.now)

    snap = ctrl.update(
        _pulse(now=100.0, bpm=120.0, confidence=0.80, runner_confidence=0.10),
        _bar(now=100.0, bpm=120.0, confidence=0.80, runner_confidence=0.75),
    )
    assert snap.state == "LISTENING"


# ── ARMED → LISTENING (cancellation) ──────────────────────────────────


def test_armed_cancels_when_confidence_drops():
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=0.5, exit_confidence_threshold=0.25)
    ctrl = LiveController(config, clock=clock.now)

    # Get to ARMED with a downbeat far in the future (2 bars away)
    dt_far = 100.5 + 4.0  # 2 bars at 120 BPM (4 seconds)
    clock.advance(0.5)
    snap = ctrl.update(
        _pulse(now=100.5, bpm=120.0, confidence=0.60),
        _bar(now=100.5, bpm=120.0, confidence=0.50, downbeat_time=dt_far),
    )
    assert snap.state == "ARMED"

    # Confidence collapses — should cancel before deadline
    clock.advance(0.1)
    snap = ctrl.update(
        _pulse(now=100.6, bpm=120.0, confidence=0.15),
        _bar(now=100.6, bpm=120.0, confidence=0.50),
    )
    assert snap.state == "LISTENING"


def test_armed_cancels_when_ambiguity_returns():
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=0.5, entry_ambiguity_margin=0.15)
    ctrl = LiveController(config, clock=clock.now)

    dt_far = 100.5 + 4.0  # 2 bars away
    clock.advance(0.5)
    ctrl.update(
        _pulse(now=100.5, bpm=120.0, confidence=0.60, runner_confidence=0.10),
        _bar(now=100.5, bpm=120.0, confidence=0.50, downbeat_time=dt_far),
    )
    assert ctrl.state == "ARMED"

    # Runner-up catches up — should cancel before deadline
    clock.advance(0.1)
    snap = ctrl.update(
        _pulse(now=100.6, bpm=120.0, confidence=0.60, runner_confidence=0.55),
        _bar(now=100.6, bpm=120.0, confidence=0.50),
    )
    assert snap.state == "LISTENING"


# ── PLAYING → DEGRADED ───────────────────────────────────────────────


def test_playing_to_degraded_on_sustained_low_confidence():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=2.0,
    )
    ctrl = LiveController(config, clock=clock.now)

    # Go to PLAYING
    clock.advance(0.5)
    ctrl.update(
        _pulse(now=100.5, bpm=120.0, confidence=0.60),
        _bar(now=100.5, bpm=120.0, confidence=0.50),
    )
    assert ctrl.state == "ARMED"
    clock.advance(0.4)
    ctrl.update(
        _pulse(now=100.9, bpm=120.0, confidence=0.60),
        _bar(now=100.9, bpm=120.0, confidence=0.50),
    )
    assert ctrl.state == "PLAYING"

    # Feed low-confidence updates for several beats
    for _ in range(6):
        clock.advance(0.4)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.15),
            _bar(now=clock.now(), bpm=120.0, confidence=0.15),
        )

    assert ctrl.state == "DEGRADED"


# ── DEGRADED → STOPPED (silence) ─────────────────────────────────────


def test_degraded_to_stopped_on_silence_timeout():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=1.0,
        silence_stop_timeout=2.0,
    )
    ctrl = LiveController(config, clock=clock.now)

    # Force to PLAYING then DEGRADED quickly
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
    for _ in range(4):
        clock.advance(0.5)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.15),
            _bar(now=clock.now(), bpm=120.0, confidence=0.15),
        )
    assert ctrl.state == "DEGRADED"

    # Now feed stale states (high evidence_age → silence)
    clock.advance(3.0)
    snap = ctrl.update(
        _pulse(now=clock.now(), bpm=120.0, confidence=0.15, evidence_age=5.0),
        _bar(now=clock.now(), bpm=120.0, confidence=0.15, evidence_age=5.0),
    )
    assert snap.state == "STOPPED"


# ── Explicit stop ────────────────────────────────────────────────────


def test_explicit_stop_from_any_state():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=4.0,
    )
    ctrl = LiveController(config, clock=clock.now)

    # Get to PLAYING
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

    gen_before = ctrl.generation
    snap = ctrl.stop()
    assert snap.state == "STOPPED"
    assert ctrl.generation > gen_before

    # Further updates should stay STOPPED
    clock.advance(1.0)
    snap = ctrl.update(
        _pulse(now=clock.now(), bpm=120.0, confidence=0.99),
        _bar(now=clock.now(), bpm=120.0, confidence=0.99),
    )
    assert snap.state == "STOPPED"


# ── DEGRADED → PLAYING (recovery) ────────────────────────────────────


def test_degraded_to_playing_on_recovery():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=1.0,
        recovery_confidence_threshold=0.45,
        recovery_dwell_beats=1.0,
    )
    ctrl = LiveController(config, clock=clock.now)

    # Go to PLAYING → DEGRADED
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
    for _ in range(3):
        clock.advance(0.5)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.15),
            _bar(now=clock.now(), bpm=120.0, confidence=0.15),
        )
    assert ctrl.state == "DEGRADED"

    # Feed enough high confidence to arm recovery, but do not resume mid-bar.
    for _ in range(2):
        clock.advance(0.5)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.70),
            _bar(now=clock.now(), bpm=120.0, confidence=0.50),
        )
    assert ctrl.state == "DEGRADED"

    assert ctrl.bar_epoch is not None
    bar_duration = 4 * (60.0 / 120.0)
    bars_elapsed = int((clock.now() - ctrl.bar_epoch) / bar_duration)
    recovery_boundary = ctrl.bar_epoch + (bars_elapsed + 1) * bar_duration
    clock.set(recovery_boundary)
    ctrl.update(
        _pulse(now=clock.now(), bpm=120.0, confidence=0.70),
        _bar(now=clock.now(), bpm=120.0, confidence=0.50),
    )
    assert ctrl.state == "PLAYING"


def test_sustained_tempo_drift_degrades_then_relocks_on_future_downbeat():
    clock = FakeClock(200.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        tempo_drift_fraction=0.05,
        tempo_drift_dwell_beats=1.0,
        recovery_confidence_threshold=0.45,
    )
    ctrl = LiveController(config, clock=clock.now)

    clock.advance(0.5)
    ctrl.update(
        _pulse(now=clock.now(), bpm=120.0, confidence=0.7),
        _bar(now=clock.now(), bpm=120.0, confidence=0.7),
    )
    clock.advance(0.25)
    ctrl.update(
        _pulse(now=clock.now(), bpm=120.0, confidence=0.7),
        _bar(now=clock.now(), bpm=120.0, confidence=0.7),
    )
    assert ctrl.state == "PLAYING"

    for _ in range(8):
        clock.advance(0.25)
        ctrl.update(
            _pulse(now=clock.now(), bpm=132.0, confidence=0.8),
            _bar(now=clock.now(), bpm=132.0, confidence=0.8),
        )
        if ctrl.state == "DEGRADED":
            break
    assert ctrl.state == "DEGRADED"

    clock.advance(0.1)
    ctrl.update(
        _pulse(now=clock.now(), bpm=132.0, confidence=0.8),
        _bar(now=clock.now(), bpm=132.0, confidence=0.8),
    )
    assert ctrl.state == "LISTENING"
    assert ctrl.locked_bpm is None

    for _ in range(12):
        clock.advance(0.25)
        ctrl.update(
            _pulse(now=clock.now(), bpm=132.0, confidence=0.8),
            _bar(now=clock.now(), bpm=132.0, confidence=0.8),
        )
        if ctrl.state == "PLAYING":
            break
    assert ctrl.state == "PLAYING"
    assert ctrl.locked_bpm == 132.0


# ── Locked BPM stability under flutter ────────────────────────────────


def test_locked_bpm_preserved_under_small_tracker_flutter():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=8.0,
        tempo_drift_fraction=0.05,
        tempo_drift_dwell_beats=4.0,
    )
    ctrl = LiveController(config, clock=clock.now)

    # Lock at 120 BPM
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
    assert ctrl.locked_bpm == 120.0
    assert ctrl.state == "PLAYING"

    # Simulate tracker fluttering between 118-122
    for bpm_flutter in [119.0, 121.0, 118.5, 122.0, 119.5]:
        clock.advance(0.4)
        ctrl.update(
            _pulse(now=clock.now(), bpm=bpm_flutter, confidence=0.55),
            _bar(now=clock.now(), bpm=bpm_flutter, confidence=0.50),
        )
    assert ctrl.locked_bpm == 120.0
    assert ctrl.state == "PLAYING"


# ── Generation increments on state changes ───────────────────────────


def test_generation_increments_on_armed_to_playing():
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=0.5)
    ctrl = LiveController(config, clock=clock.now)

    clock.advance(0.5)
    ctrl.update(
        _pulse(now=100.5, bpm=120.0, confidence=0.60),
        _bar(now=100.5, bpm=120.0, confidence=0.50),
    )
    gen_armed = ctrl.generation

    clock.advance(0.4)
    ctrl.update(
        _pulse(now=100.9, bpm=120.0, confidence=0.60),
        _bar(now=100.9, bpm=120.0, confidence=0.50),
    )
    assert ctrl.generation > gen_armed


def test_generation_increments_on_degraded():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=1.0,
    )
    ctrl = LiveController(config, clock=clock.now)

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
    gen_playing = ctrl.generation

    for _ in range(4):
        clock.advance(0.5)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.15),
            _bar(now=clock.now(), bpm=120.0, confidence=0.15),
        )
    assert ctrl.state == "DEGRADED"
    assert ctrl.generation > gen_playing


# ── STOPPED is terminal ──────────────────────────────────────────────


def test_stopped_ignores_all_updates():
    clock = FakeClock(100.0)
    config = _make_config()
    ctrl = LiveController(config, clock=clock.now)

    ctrl.stop()
    assert ctrl.state == "STOPPED"

    for _ in range(10):
        clock.advance(1.0)
        snap = ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.99),
            _bar(now=clock.now(), bpm=120.0, confidence=0.99),
        )
        assert snap.state == "STOPPED"


# ── Grid position tracking ───────────────────────────────────────────


def test_grid_position_advances_in_playing():
    clock = FakeClock(100.0)
    config = _make_config(entry_min_evidence_beats=0.5)
    ctrl = LiveController(config, clock=clock.now)

    # Lock at 120 BPM
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

    # After 0.5 seconds (1 beat), slot should advance ~4 slots
    clock.advance(0.5)
    snap = ctrl.update(
        _pulse(now=clock.now(), bpm=120.0, confidence=0.60),
        _bar(now=clock.now(), bpm=120.0, confidence=0.50),
    )
    # 1 beat at 120 BPM = 4 slots
    assert snap.current_slot >= 3  # at least some advancement


# ── Mirror methods ───────────────────────────────────────────────────


def test_mirror_activate_and_clear():
    clock = FakeClock(100.0)
    ctrl = LiveController(_make_config(), clock=clock.now)

    assert not ctrl.mirror_active
    assert ctrl.mirror_slot is None

    ctrl.set_mirror(7)
    assert ctrl.mirror_active
    assert ctrl.mirror_slot == 7

    ctrl.clear_mirror()
    assert not ctrl.mirror_active
    assert ctrl.mirror_slot is None


def test_mirror_cleared_on_stop():
    clock = FakeClock(100.0)
    ctrl = LiveController(_make_config(), clock=clock.now)

    ctrl.set_mirror(9)
    ctrl.stop()
    assert not ctrl.mirror_active


def test_mirror_cleared_on_degraded():
    clock = FakeClock(100.0)
    config = _make_config(
        entry_min_evidence_beats=0.5,
        exit_confidence_threshold=0.25,
        degradation_dwell_beats=1.0,
    )
    ctrl = LiveController(config, clock=clock.now)

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
    ctrl.set_mirror(5)
    assert ctrl.mirror_active

    for _ in range(4):
        clock.advance(0.5)
        ctrl.update(
            _pulse(now=clock.now(), bpm=120.0, confidence=0.15),
            _bar(now=clock.now(), bpm=120.0, confidence=0.15),
        )
    assert ctrl.state == "DEGRADED"
    assert not ctrl.mirror_active


# ── Bar completion counting ──────────────────────────────────────────


def test_note_bar_completed_increments_counter():
    clock = FakeClock(100.0)
    ctrl = LiveController(_make_config(), clock=clock.now)

    assert ctrl._playing_bars_completed == 0  # type: ignore[attr]
    ctrl.note_bar_completed()
    ctrl.note_bar_completed()
    assert ctrl._playing_bars_completed == 2  # type: ignore[attr]
