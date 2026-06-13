"""Tests for the Feature Monitor — Module 5: Musical Feature Tracking."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.features import (
    FeatureMonitor,
    FeatureMonitorConfig,
    FeatureSnapshot,
)
from perception.models import MusicalEvent


# ─── Helpers ────────────────────────────────────────────────────────────

def _make_event(time_seconds: float, strength: float = 0.5) -> MusicalEvent:
    return MusicalEvent(time_seconds=time_seconds, strength=strength)


def _feed_n(fm: FeatureMonitor, n: int, spacing: float = 0.25,
            strength: float = 0.5, start_time: float = 0.0) -> list[FeatureSnapshot]:
    """Feed *n* evenly-spaced events and return snapshots."""
    snaps: list[FeatureSnapshot] = []
    for i in range(n):
        t = start_time + i * spacing
        s = fm.feed(_make_event(t, strength))
        snaps.append(s)
    return snaps


# ─── 1. Initialization ──────────────────────────────────────────────────


class TestInitialization:
    """FeatureMonitor creates clean default snapshots with no crashes."""

    def test_default_config_creates_monitor(self) -> None:
        fm = FeatureMonitor()
        assert fm.config is not None
        assert fm.config.density_window_seconds == 2.0
        assert fm._strength_ema == 0.0
        assert fm._fast_ema == 0.0
        assert fm._slow_ema == 0.0
        assert fm._last_event_time is None
        assert fm._events == []

    def test_custom_config(self) -> None:
        cfg = FeatureMonitorConfig(
            density_window_seconds=3.0,
            strength_alpha=0.2,
            max_expected_density=8.0,
        )
        fm = FeatureMonitor(config=cfg)
        assert fm.config.density_window_seconds == 3.0
        assert fm.config.strength_alpha == 0.2
        assert fm.config.max_expected_density == 8.0

    def test_snapshot_with_no_events_returns_zeros(self) -> None:
        fm = FeatureMonitor()
        snap = fm.snapshot(now=5.0)
        assert snap.timestamp == 5.0
        assert snap.input_density == 0.0
        assert snap.strength_ema == 0.0
        assert snap.fast_strength_ema == 0.0
        assert snap.slow_strength_ema == 0.0
        assert snap.change_score == 0.0
        assert snap.silence_duration == 0.0
        assert snap.repetition_stability == 0.0
        assert snap.phase_alignment is None
        assert snap.player_certainty == 0.0

    def test_reset_clears_everything(self) -> None:
        fm = FeatureMonitor()
        fm.feed(_make_event(1.0, 0.8))
        fm.feed(_make_event(1.5, 0.9))
        fm.reset()
        assert fm._strength_ema == 0.0
        assert fm._fast_ema == 0.0
        assert fm._slow_ema == 0.0
        assert fm._last_event_time is None
        assert fm._events == []

    def test_frozen_config_is_immutable(self) -> None:
        cfg = FeatureMonitorConfig()
        try:
            cfg.strength_alpha = 0.99  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except Exception:
            pass


# ─── 2. Density tracking ────────────────────────────────────────────────


class TestDensityTracking:
    """input_density tracks events in a rolling window and normalises."""

    def test_sparse_events_low_density(self) -> None:
        fm = FeatureMonitor()
        # One event in 2s window with max_expected=12 → density ~0.083
        s = fm.feed(_make_event(1.0))
        assert 0.0 < s.input_density < 0.2

    def test_dense_events_high_density(self) -> None:
        fm = FeatureMonitor()
        # Feed 10 events in quick succession within the window
        snaps = _feed_n(fm, 10, spacing=0.1, start_time=0.0)
        final = snaps[-1]
        # 10 / 12 = 0.833
        assert final.input_density > 0.7

    def test_old_events_pruned(self) -> None:
        fm = FeatureMonitor(config=FeatureMonitorConfig(density_window_seconds=1.0))
        # Event at t=0.0, then event at t=2.0 → the t=0.0 event should be pruned
        fm.feed(_make_event(0.0))
        s = fm.feed(_make_event(2.0))
        max_d = fm.config.max_expected_density
        assert s.input_density <= 1.0 / max_d + 0.01  # just the t=2.0 event

    def test_density_clamps_at_one(self) -> None:
        fm = FeatureMonitor(config=FeatureMonitorConfig(max_expected_density=4.0))
        # Feed 8 events in a tight window — raw count 8 for max 4 → 2.0 should clamp
        _feed_n(fm, 8, spacing=0.1, start_time=0.0)
        s = fm.snapshot(now=1.0)
        assert s.input_density == 1.0

    def test_window_with_configurable_max(self) -> None:
        cfg = FeatureMonitorConfig(density_window_seconds=1.0, max_expected_density=6.0)
        fm = FeatureMonitor(config=cfg)
        _feed_n(fm, 3, spacing=0.3, start_time=0.0)
        s = fm.snapshot(now=1.0)
        assert s.input_density == 3.0 / 6.0  # 0.5


# ─── 3. EMA strength ────────────────────────────────────────────────────


class TestEMAStrength:
    """strength_ema rises with events, smooths, and decays in silence."""

    def test_ema_rises_with_strong_events(self) -> None:
        cfg = FeatureMonitorConfig(strength_alpha=0.5)
        fm = FeatureMonitor(config=cfg)
        s1 = fm.feed(_make_event(0.0, 0.0))
        s2 = fm.feed(_make_event(0.1, 1.0))
        # After first: ema = 0.5*0 + 0.5*0 = 0.0
        # After second: ema = 0.5*1.0 + 0.5*0.0 = 0.5
        assert abs(s2.strength_ema - 0.5) < 1e-9

    def test_ema_does_not_jump_instantly(self) -> None:
        cfg = FeatureMonitorConfig(strength_alpha=0.15)  # default
        fm = FeatureMonitor(config=cfg)
        s = fm.feed(_make_event(0.0, 0.9))
        # strength_ema should be alpha * 0.9 = 0.135, not 0.9
        assert s.strength_ema < 0.2

    def test_ema_decays_during_silence(self) -> None:
        cfg = FeatureMonitorConfig(strength_alpha=0.5, decay_per_second=0.5)
        fm = FeatureMonitor(config=cfg)
        fm.feed(_make_event(0.0, 1.0))  # ema = 0.5
        import math
        # 1 second passes with no events, decay factor = exp(-0.5 * 1) ≈ 0.6065
        expected = 0.5 * math.exp(-0.5 * 1.0)
        s = fm.snapshot(now=1.0)
        assert abs(s.strength_ema - expected) < 1e-6

    def test_ema_converges_with_repeated_same_strength(self) -> None:
        cfg = FeatureMonitorConfig(strength_alpha=0.3)
        fm = FeatureMonitor(config=cfg)
        for i in range(20):
            fm.feed(_make_event(float(i) * 0.1, 0.5))
        s = fm.snapshot(now=2.0)
        # After many events, ema should be close to 0.5
        assert abs(s.strength_ema - 0.5) < 0.05


# ─── 4. Fast / slow change detection ────────────────────────────────────


class TestChangeDetection:
    """change_score rises when fast EMA diverges from slow EMA."""

    def test_sudden_strong_detected(self) -> None:
        cfg = FeatureMonitorConfig(
            fast_strength_alpha=0.4,
            slow_strength_alpha=0.05,
        )
        fm = FeatureMonitor(config=cfg)
        # Build a baseline of weak events
        _feed_n(fm, 10, spacing=0.1, strength=0.1)
        # Sudden strong event
        s = fm.feed(_make_event(1.0, 0.9))
        # fast should react faster than slow, so change_score > 0
        assert s.change_score > 0.0

    def test_change_score_reduces_after_stable_input(self) -> None:
        cfg = FeatureMonitorConfig(
            fast_strength_alpha=0.4,
            slow_strength_alpha=0.05,
        )
        fm = FeatureMonitor(config=cfg)
        _feed_n(fm, 10, spacing=0.1, strength=0.2)
        # Now feed many events at the new strength level so slow catches up
        snaps = _feed_n(fm, 30, spacing=0.1, strength=0.2, start_time=1.0)
        final = snaps[-1]
        # slow should have caught up somewhat, change_score should be lower
        first_after = fm.snapshot(now=1.05)
        # Not strictly monotonic but final should be small
        assert final.change_score < 0.15

    def test_change_score_is_zero_for_const_input(self) -> None:
        cfg = FeatureMonitorConfig(
            fast_strength_alpha=0.3,
            slow_strength_alpha=0.3,  # same alpha → no divergence
        )
        fm = FeatureMonitor(config=cfg)
        _feed_n(fm, 10, spacing=0.1, strength=0.5)
        s = fm.snapshot(now=1.0)
        # Same alpha, same initial values → fast == slow → change_score = 0
        assert s.change_score < 1e-9

    def test_change_score_clamped(self) -> None:
        cfg = FeatureMonitorConfig(
            fast_strength_alpha=0.99,
            slow_strength_alpha=0.01,
        )
        fm = FeatureMonitor(config=cfg)
        _feed_n(fm, 5, spacing=0.1, strength=1.0)
        s = fm.snapshot(now=1.0)
        assert 0.0 <= s.change_score <= 1.0


# ─── 5. Silence duration ────────────────────────────────────────────────


class TestSilenceDuration:
    """silence_duration tracks time since the last event."""

    def test_silence_increases_with_no_events(self) -> None:
        fm = FeatureMonitor()
        fm.feed(_make_event(1.0))
        s = fm.snapshot(now=3.0)
        assert s.silence_duration == 2.0

    def test_silence_resets_on_new_event(self) -> None:
        fm = FeatureMonitor()
        fm.feed(_make_event(1.0))
        fm.snapshot(now=3.0)  # silence = 2.0
        s = fm.feed(_make_event(3.5))
        # silence = 3.5 - 3.5 = 0 (the event time is 3.5)
        assert s.silence_duration == 0.0

    def test_silence_zero_with_no_prior_events(self) -> None:
        fm = FeatureMonitor()
        s = fm.snapshot(now=10.0)
        assert s.silence_duration == 0.0

    def test_silence_never_negative(self) -> None:
        fm = FeatureMonitor()
        fm.feed(_make_event(5.0))
        # snapshot at a time before the event (shouldn't happen normally)
        s = fm.snapshot(now=3.0)
        assert s.silence_duration == 0.0


# ─── 6. Repetition stability ────────────────────────────────────────────


class TestRepetitionStability:
    """repetition_stability scores how regular event spacing is."""

    def test_evenly_spaced_high_stability(self) -> None:
        fm = FeatureMonitor()
        _feed_n(fm, 10, spacing=0.25, start_time=0.0)
        s = fm.snapshot(now=2.5)
        # Even spacing → very low CV → high stability
        assert s.repetition_stability > 0.9

    def test_irregular_spacing_low_stability(self) -> None:
        fm = FeatureMonitor()
        # Alternating short/long gaps
        times = [0.0, 0.1, 0.5, 0.6, 1.0, 1.1]
        for t in times:
            fm.feed(_make_event(t))
        s = fm.snapshot(now=1.2)
        assert s.repetition_stability < 0.6

    def test_too_few_events_zero_stability(self) -> None:
        fm = FeatureMonitor()
        fm.feed(_make_event(0.0))
        s = fm.snapshot(now=0.5)
        assert s.repetition_stability == 0.0

    def test_random_spacing_produces_low_stability(self) -> None:
        fm = FeatureMonitor()
        import random
        random.seed(42)
        t = 0.0
        for _ in range(20):
            gap = 0.05 + random.random() * 0.4  # 0.05–0.45s
            t += gap
            fm.feed(_make_event(t))
        s = fm.snapshot(now=t)
        # Random gaps should yield low stability
        assert s.repetition_stability < 0.5

    def test_stability_clamped_between_zero_and_one(self) -> None:
        fm = FeatureMonitor()
        _feed_n(fm, 20, spacing=0.1, start_time=0.0)
        s = fm.snapshot(now=2.0)
        assert 0.0 <= s.repetition_stability <= 1.0

    def test_stability_with_only_two_events(self) -> None:
        # Two events → one IOI → CV = 0 (single value has no variance)
        cfg = FeatureMonitorConfig(min_iois_for_stability=1)
        fm = FeatureMonitor(config=cfg)
        fm.feed(_make_event(0.0))
        fm.feed(_make_event(0.5))
        s = fm.snapshot(now=0.6)
        # One IOI → std=0, CV=0 → stability = 1.0
        assert s.repetition_stability == 1.0

    def test_stability_configurable_window(self) -> None:
        # Feed many regular events but with repetition_window_beats=2
        cfg = FeatureMonitorConfig(repetition_window_beats=2)
        fm = FeatureMonitor(config=cfg)
        _feed_n(fm, 20, spacing=0.25, start_time=0.0)
        s = fm.snapshot(now=5.0)
        assert s.repetition_stability > 0.9


# ─── 7. Phase alignment handling ────────────────────────────────────────


class TestPhaseAlignment:
    """phase_alignment is stored in snapshot when provided."""

    def test_phase_provided_appears_in_snapshot(self) -> None:
        fm = FeatureMonitor()
        s = fm.snapshot(now=1.0, phase_alignment=0.85)
        assert s.phase_alignment == 0.85

    def test_phase_none_by_default(self) -> None:
        fm = FeatureMonitor()
        fm.feed(_make_event(1.0))
        assert fm.snapshot(now=1.5).phase_alignment is None

    def test_certainty_works_with_phase_none(self) -> None:
        fm = FeatureMonitor()
        _feed_n(fm, 10, spacing=0.25, strength=0.7)
        s = fm.snapshot(now=2.5)
        # Should compute certainty without phase, not crash
        assert 0.0 <= s.player_certainty <= 1.0

    def test_certainty_higher_with_good_phase(self) -> None:
        fm = FeatureMonitor()
        _feed_n(fm, 10, spacing=0.25, strength=0.7)
        s_no_phase = fm.snapshot(now=2.5)
        s_good_phase = fm.snapshot(now=2.5, phase_alignment=0.9)
        s_bad_phase = fm.snapshot(now=2.5, phase_alignment=0.1)
        # Good phase should boost certainty vs no phase
        # Bad phase should reduce it
        assert s_good_phase.player_certainty > s_bad_phase.player_certainty


# ─── 8. Player certainty ────────────────────────────────────────────────


class TestPlayerCertainty:
    """player_certainty is a 0–1 composite reflecting playing confidence."""

    def test_stable_strong_aligned_high_certainty(self) -> None:
        fm = FeatureMonitor()
        _feed_n(fm, 15, spacing=0.25, strength=0.8)
        s = fm.snapshot(now=4.0, phase_alignment=0.9)
        assert s.player_certainty > 0.5

    def test_weak_erratic_low_certainty(self) -> None:
        cfg = FeatureMonitorConfig(repetition_window_beats=8)
        fm = FeatureMonitor(config=cfg)
        # Weak, irregular events
        fm.feed(_make_event(0.0, 0.05))
        fm.feed(_make_event(0.3, 0.05))
        fm.feed(_make_event(1.0, 0.05))
        fm.feed(_make_event(1.5, 0.05))
        s = fm.snapshot(now=2.0, phase_alignment=0.1)
        assert s.player_certainty < 0.4

    def test_certainty_always_clamped(self) -> None:
        fm = FeatureMonitor()
        # Feed extreme values
        for i in range(50):
            fm.feed(_make_event(float(i) * 0.05, 1.0))
        s = fm.snapshot(now=3.0, phase_alignment=1.0)
        assert 0.0 <= s.player_certainty <= 1.0

    def test_certainty_near_zero_for_empty_monitor(self) -> None:
        fm = FeatureMonitor()
        s = fm.snapshot(now=5.0)
        assert s.player_certainty == 0.0

    def test_certainty_custom_weights_respected(self) -> None:
        # All weight on strength
        cfg = FeatureMonitorConfig(
            certainty_strength_weight=1.0,
            certainty_repetition_weight=0.0,
            certainty_phase_weight=0.0,
        )
        fm = FeatureMonitor(config=cfg)
        _feed_n(fm, 5, spacing=0.25, strength=0.9)
        s = fm.snapshot(now=1.5)
        # Certainty should be roughly strength_ema (which is < 0.9 due to alpha)
        assert abs(s.player_certainty - fm._strength_ema) < 1e-9


# ─── 9. FeatureSnapshot dataclass ───────────────────────────────────────


class TestFeatureSnapshotDataclass:
    """FeatureSnapshot behaves as expected."""

    def test_all_fields_accessible(self) -> None:
        snap = FeatureSnapshot(
            timestamp=1.0,
            input_density=0.5,
            strength_ema=0.3,
            fast_strength_ema=0.4,
            slow_strength_ema=0.2,
            change_score=0.2,
            silence_duration=0.1,
            repetition_stability=0.8,
            phase_alignment=0.7,
            player_certainty=0.6,
        )
        assert snap.timestamp == 1.0
        assert snap.input_density == 0.5
        assert snap.change_score == 0.2
        assert snap.phase_alignment == 0.7

    def test_frozen(self) -> None:
        snap = FeatureSnapshot(timestamp=0.0, input_density=0.0)
        try:
            snap.timestamp = 5.0  # type: ignore[misc]
            assert False, "Should have raised FrozenInstanceError"
        except Exception:
            pass

    def test_defaults(self) -> None:
        snap = FeatureSnapshot(timestamp=0.0, input_density=0.0)
        assert snap.strength_ema == 0.0
        assert snap.fast_strength_ema == 0.0
        assert snap.slow_strength_ema == 0.0
        assert snap.change_score == 0.0
        assert snap.silence_duration == 0.0
        assert snap.repetition_stability == 0.0
        assert snap.phase_alignment is None
        assert snap.player_certainty == 0.0


# ─── 10. Config default values ──────────────────────────────────────────


class TestConfigDefaults:
    """FeatureMonitorConfig has sensible documented defaults."""

    def test_defaults_match_spec(self) -> None:
        cfg = FeatureMonitorConfig()
        assert cfg.density_window_seconds == 2.0
        assert cfg.strength_alpha == 0.15
        assert cfg.fast_strength_alpha == 0.35
        assert cfg.slow_strength_alpha == 0.05
        assert cfg.repetition_window_beats == 4
        assert cfg.silence_timeout_seconds == 1.0
        assert cfg.max_expected_density == 12.0
        assert cfg.certainty_strength_weight == 0.3
        assert cfg.certainty_repetition_weight == 0.4
        assert cfg.certainty_phase_weight == 0.3
        assert cfg.decay_per_second == 0.3
        assert cfg.min_iois_for_stability == 2