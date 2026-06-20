"""Immutable data types for the Bunny V1 live-play controller.

All timestamps crossing module boundaries use monotonic seconds from a
single injectable clock (default ``time.perf_counter``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# ── Configuration ───────────────────────────────────────────────────


@dataclass(frozen=True)
class LiveConfig:
    """Immutable configuration for the Bunny V1 controller.

    All duration thresholds are expressed in seconds.  Defaults are
    hypotheses for initial tuning, not product truth.
    """

    # ── BPM range ──
    min_bpm: float = 40.0
    max_bpm: float = 250.0

    # ── Entry ──
    entry_confidence_threshold: float = 0.40
    """Minimum winning pulse confidence to consider entry."""
    entry_min_evidence_beats: float = 4.0
    """Minimum beats of evidence before entry (beats, not seconds)."""
    entry_ambiguity_margin: float = 0.15
    """Winner must beat runner-up by at least this confidence margin."""

    # ── Exit / degradation ──
    exit_confidence_threshold: float = 0.25
    """Confidence below this triggers degradation (lower than entry)."""
    degradation_dwell_beats: float = 4.0
    """Beats of sub-threshold confidence before entering DEGRADED."""
    recovery_confidence_threshold: float = 0.45
    """Confidence above this for dwell triggers DEGRADED -> PLAYING."""
    recovery_dwell_beats: float = 2.0
    """Beats of sufficient confidence before recovery."""

    # ── Silence / stop ──
    silence_stop_timeout: float = 4.0
    """Seconds of silence (no events) before auto-STOPPED."""
    silence_grace_beats: float = 4.0
    """Beats of silence while keeping the clock in DEGRADED."""

    # ── Tempo / bar freshness ──
    max_evidence_age_beats: float = 2.0
    """Maximum age of evidence (beats) before it is considered stale."""
    tempo_drift_fraction: float = 0.10
    """Relative BPM change that can trigger re-lock (10%)."""
    tempo_drift_dwell_beats: float = 8.0
    """Beats of sustained drift before re-lock (two 4/4 bars)."""

    # ── Quantisation ──
    quantisation_tolerance_beats: float = 0.25
    """Max offset (fraction of a beat) to snap to the nearest slot."""

    # ── Lookahead / scheduling ──
    max_lookahead_seconds: float = 0.060
    """How far ahead to pre-schedule events (60 ms)."""
    late_budget_seconds: float = 0.010
    """Maximum lateness before dropping an event (10 ms)."""

    # ── Kick mirror ──
    mirror_min_stable_bars: int = 2
    """Completed stable playing bars before mirror is eligible."""
    mirror_strength_percentile: float = 50.0
    """Rolling strength percentile threshold."""
    mirror_min_strength: float = 0.35
    """Absolute strength floor; relative percentiles cannot promote weak hits."""
    mirror_min_sample_count: int = 8
    """Minimum observations before percentile filtering works."""
    mirror_max_observations: int = 64
    """Hard cap on rolling observation history."""
    mirror_velocity: int = 60
    """Moderate velocity for mirrored kicks."""
    mirror_expire_unsupported_bars: int = 1
    """Bars without support before clearing mirror."""

    # ── Grid ──
    beats_per_bar: int = 4
    slots_per_bar: int = 16

    # ── Anchor slots ──
    kick_anchor_slots: tuple[int, ...] = (0, 8)
    snare_anchor_slots: tuple[int, ...] = (4, 12)
    hat_anchor_slots: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)
    anchor_slots: tuple[int, ...] = (0, 2, 4, 6, 8, 10, 12, 14)
    """All anchor slots (kick + snare + hat)."""

    # ── MIDI ──
    kick_note: int = 36
    snare_note: int = 38
    closed_hat_note: int = 42
    drum_channel: int = 9
    anchor_velocity: int = 100
    hat_velocity: int = 80


# ── Adapter input types ─────────────────────────────────────────────


@dataclass(frozen=True)
class PulseAdapterState:
    """Distilled pulse tracker state in the shared monotonic clock domain.

    All timestamps are monotonic seconds from the injected clock.
    """

    observed_at: float
    """Monotonic time when the underlying audio evidence occurred."""
    computed_at: float
    """Monotonic time when this state was produced by the tracker."""

    winning_bpm: float | None
    winning_confidence: float  # in [0, 1]

    runner_up_bpm: float | None
    runner_up_confidence: float  # in [0, 1]

    ambiguity_margin: float
    """winning_confidence - runner_up_confidence (or 1.0 if solo)."""

    hypothesis_count: int
    support_count: int
    """Total events supporting the winning hypothesis."""

    evidence_age: float
    """Seconds since the last event supporting the winner."""

    predicted_next_beat: float | None
    """Monotonic deadline of the next predicted beat, if computable."""

    beat_period: float | None
    """Seconds per beat at the winning BPM, if computable."""

    stability: str  # "unknown" | "rising" | "stable" | "locked"

    def age_seconds(self, now: float) -> float:
        """Seconds since this state was computed."""
        return now - self.computed_at


@dataclass(frozen=True)
class BarAdapterState:
    """Distilled bar tracker state in the shared monotonic clock domain."""

    observed_at: float
    computed_at: float

    winning_bpm: float | None
    winning_confidence: float
    runner_up_confidence: float
    ambiguity_margin: float
    hypothesis_count: int
    support_count: int

    estimated_beat_in_bar: int | None
    """0-based beat index within the bar (0, 1, 2, 3 for 4/4)."""
    bar_position: float | None
    """Beat position within the bar (0.0 to beats_per_bar)."""

    downbeat_time: float | None
    """Monotonic timestamp of the reference downbeat (beat 1)."""
    bar_duration: float | None
    """Duration of one bar in seconds."""

    evidence_age: float
    is_confident: bool

    def age_seconds(self, now: float) -> float:
        return now - self.computed_at

    def next_downbeat(self, now: float) -> float | None:
        """Monotonic deadline of the next beat 1, at or after *now*."""
        if self.downbeat_time is None or self.bar_duration is None:
            return None
        elapsed = now - self.downbeat_time
        bars_passed = int(elapsed / self.bar_duration)
        candidate = self.downbeat_time + self.bar_duration * (bars_passed + 1)
        if candidate < now:
            candidate += self.bar_duration
        return candidate


# ── Ledger event record ──────────────────────────────────────────────


@dataclass(frozen=True)
class LedgerEvent:
    """An immutable scheduled MIDI event.

    Once created, never mutated.  Cancellation is by generation invalidation.
    """

    event_id: str
    """Unique identifier, stable across re-plans."""
    deadline: float
    """Absolute monotonic time to fire this event."""
    note: int
    velocity: int
    channel: int
    bar_index: int
    """Absolute bar number (0-based)."""
    slot: int
    """Slot within the bar (0..slots_per_bar-1)."""
    source: str
    """Where this event came from: 'anchor', 'hat', 'mirror'."""
    generation: int
    """Controller generation — events from older generations are invalid."""

    def with_deadline(self, new_deadline: float) -> LedgerEvent:
        """Return a copy with a different deadline (preserves event_id)."""
        return LedgerEvent(
            event_id=self.event_id,
            deadline=new_deadline,
            note=self.note,
            velocity=self.velocity,
            channel=self.channel,
            bar_index=self.bar_index,
            slot=self.slot,
            source=self.source,
            generation=self.generation,
        )


# ── Controller snapshot ──────────────────────────────────────────────


@dataclass(frozen=True)
class ControllerSnapshot:
    """Immutable summary of the controller for diagnostics / tests."""

    state: str
    generation: int
    locked_bpm: float | None
    bar_epoch: float | None
    beat_period: float | None
    current_bar_index: int
    current_slot: int
    mirror_active: bool
    mirror_slot: int | None
    anchor_count: int
    mirror_count: int
    queue_depth: int
    computed_at: float


# ── Type alias for clock injection ──────────────────────────────────

MonotonicClock = Callable[[], float]
