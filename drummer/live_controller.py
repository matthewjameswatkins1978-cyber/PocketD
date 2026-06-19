"""Bunny V1 live controller — state machine, locks, generation counter.

This is the top-level loop controller.  It receives adapted tracker
state and transitions through explicit states.  It owns the locked
playing grid and the generation counter used to invalidate stale
ledger events.

It does **not** plan individual MIDI events, run a scheduler loop,
or emit MIDI.  Those responsibilities belong to ``straight_pocket``
and ``live_scheduler``.
"""

from __future__ import annotations

import logging
import time as _time

from drummer.live_models import (
    BarAdapterState,
    ControllerSnapshot,
    LiveConfig,
    MonotonicClock,
    PulseAdapterState,
)

log = logging.getLogger(__name__)


def _default_clock() -> float:
    return _time.perf_counter()


# ── State enumeration ────────────────────────────────────────────────


class _State:
    LISTENING = "LISTENING"
    ARMED = "ARMED"
    PLAYING = "PLAYING"
    DEGRADED = "DEGRADED"
    STOPPED = "STOPPED"

    @staticmethod
    def all_states() -> list[str]:
        return [
            _State.LISTENING,
            _State.ARMED,
            _State.PLAYING,
            _State.DEGRADED,
            _State.STOPPED,
        ]


# ── LiveController ───────────────────────────────────────────────────


class LiveController:
    """Explicit-state controller for Bunny V1 live play.

    Parameters
    ----------
    config : LiveConfig
        Immutable configuration.
    clock : MonotonicClock | None
        Injectable monotonic clock.  Defaults to ``time.perf_counter``.
    """

    def __init__(self, config: LiveConfig, clock: MonotonicClock | None = None) -> None:
        self._config = config
        self._clock = clock if clock is not None else _default_clock

        # ── State ──
        self._state: str = _State.LISTENING
        self._generation: int = 0

        # ── Locked playing grid ──
        self._locked_bpm: float | None = None
        self._beat_period: float | None = None
        self._bar_epoch: float | None = None  # monotonic time of beat 1

        # ── Bar tracking (absolute) ──
        self._bar_index: int = 0
        """Absolute bar number since epoch (fraction truncated)."""
        self._current_slot: int = 0
        """Slot 0..15 within the current bar."""

        # ── Mirror state ──
        self._mirror_active: bool = False
        self._mirror_slot: int | None = None

        # ── Dwell / grace counters (monotonic seconds or beat counts) ──
        self._state_entered_at: float = self._clock()
        self._degraded_beats: float = 0.0
        self._recovery_beats: float = 0.0
        self._recovery_at: float | None = None
        self._silence_beats: float = 0.0
        self._tempo_drift_beats: float = 0.0
        self._requires_relock: bool = False
        self._playing_bars_completed: int = 0

        # ── Previous adapted states (for drift detection) ──
        self._last_pulse: PulseAdapterState | None = None
        self._last_bar: BarAdapterState | None = None

        # ── Arming data ──
        self._armed_downbeat: float | None = None
        self._armed_bpm: float | None = None

        # ── Evidence tracking for entry ──
        self._consecutive_evidence_beats: float = 0.0
        """Accumulated beats of sufficient evidence in LISTENING."""

        log.info("LiveController initialised — state=%s generation=%d", self._state, self._generation)
        self._log_config()

    # ── Public API ───────────────────────────────────────────────────

    def update(
        self,
        pulse: PulseAdapterState,
        bar: BarAdapterState,
    ) -> ControllerSnapshot:
        """Process one tick of adapted tracker state.

        Returns an immutable snapshot for tests / diagnostics.
        """
        now = self._clock()

        # ── Stop is terminal ──
        if self._state == _State.STOPPED:
            return self._snapshot(now)

        # ── Update bar/slot tracking if we have a locked grid ──
        if self._bar_epoch is not None and self._beat_period is not None:
            self._update_grid_position(now)

        # ── Dispatch to current state ──
        if self._state == _State.LISTENING:
            self._do_listening(pulse, bar, now)
        elif self._state == _State.ARMED:
            self._do_armed(pulse, bar, now)
        elif self._state == _State.PLAYING:
            self._do_playing(pulse, bar, now)
        elif self._state == _State.DEGRADED:
            self._do_degraded(pulse, bar, now)

        # Store for next tick
        self._last_pulse = pulse
        self._last_bar = bar

        return self._snapshot(now)

    def stop(self) -> ControllerSnapshot:
        """Explicit stop — invalidate future events and enter STOPPED."""
        now = self._clock()
        self._transition_to(_State.STOPPED, "explicit stop", now)
        return self._snapshot(now)

    @property
    def state(self) -> str:
        return self._state

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def locked_bpm(self) -> float | None:
        return self._locked_bpm

    @property
    def bar_epoch(self) -> float | None:
        return self._bar_epoch

    @property
    def beat_period(self) -> float | None:
        return self._beat_period

    @property
    def bar_index(self) -> int:
        return self._bar_index

    @property
    def mirror_active(self) -> bool:
        return self._mirror_active

    @property
    def mirror_slot(self) -> int | None:
        return self._mirror_slot

    def set_mirror(self, slot: int) -> None:
        """Activate the kick mirror on the given slot (called by pocket planner)."""
        self._mirror_active = True
        self._mirror_slot = slot
        log.info("Kick mirror activated on slot %d", slot)

    def clear_mirror(self) -> None:
        """Deactivate the kick mirror."""
        if self._mirror_active:
            log.info("Kick mirror cleared (was slot %s)", self._mirror_slot)
        self._mirror_active = False
        self._mirror_slot = None

    def note_bar_completed(self) -> None:
        """Called by the pocket planner each time a full bar completes."""
        self._playing_bars_completed += 1

    def increment_generation(self, reason: str = "") -> int:
        """Bump generation, invalidating all previously scheduled events."""
        self._generation += 1
        log.info("Generation %d — %s", self._generation, reason or "manual increment")
        return self._generation

    # ── State transitions ────────────────────────────────────────────

    def _transition_to(self, new_state: str, reason: str, now: float) -> None:
        old = self._state
        if old == new_state:
            return
        self._state = new_state
        self._state_entered_at = now
        log.info("State %s -> %s — %s", old, new_state, reason)

        # Side effects on state entry
        if new_state == _State.STOPPED:
            self._clear_locked_grid()
            self._clear_arming()
            self.clear_mirror()
            self.increment_generation("stop")
        elif new_state == _State.PLAYING:
            # If entering PLAYING from ARMED, lock the BPM and bar epoch
            if old == _State.ARMED and self._armed_bpm is not None and self._armed_downbeat is not None:
                self._locked_bpm = self._armed_bpm
                self._beat_period = 60.0 / self._armed_bpm
                self._bar_epoch = self._armed_downbeat
                self._bar_index = 0
                self._current_slot = 0
                self._playing_bars_completed = 0
                self._clear_arming()
                self.increment_generation("armed -> playing lock")
            self._degraded_beats = 0.0
            self._recovery_beats = 0.0
            self._recovery_at = None
            self._silence_beats = 0.0
            self._tempo_drift_beats = 0.0
            self._requires_relock = False
        elif new_state == _State.DEGRADED:
            self.clear_mirror()
            self.increment_generation("degraded")
            self._degraded_beats = 0.0
            self._recovery_beats = 0.0
            self._recovery_at = None
            self._silence_beats = 0.0
        elif new_state == _State.LISTENING:
            self._clear_locked_grid()
            self._clear_arming()
            self.clear_mirror()
            self.increment_generation("listening reset")
            self._consecutive_evidence_beats = 0.0
            self._tempo_drift_beats = 0.0
            self._requires_relock = False
        elif new_state == _State.ARMED:
            self._degraded_beats = 0.0
            self._recovery_beats = 0.0
            self._silence_beats = 0.0

    def _clear_locked_grid(self) -> None:
        self._locked_bpm = None
        self._beat_period = None
        self._bar_epoch = None
        self._bar_index = 0
        self._current_slot = 0
        self._playing_bars_completed = 0

    def _clear_arming(self) -> None:
        self._armed_bpm = None
        self._armed_downbeat = None

    # ── LISTENING ────────────────────────────────────────────────────

    def _do_listening(
        self,
        pulse: PulseAdapterState,
        bar: BarAdapterState,
        now: float,
    ) -> None:
        """Check entry conditions: confidence, ambiguity, freshness, downbeat available."""
        if pulse.winning_bpm is None or pulse.beat_period is None:
            self._consecutive_evidence_beats = 0.0
            return

        cfg = self._config

        # Check BPM range
        if not (cfg.min_bpm <= pulse.winning_bpm <= cfg.max_bpm):
            self._consecutive_evidence_beats = 0.0
            log.debug("BPM %.1f outside range [%.0f, %.0f]", pulse.winning_bpm, cfg.min_bpm, cfg.max_bpm)
            return

        # Check absolute confidence
        if pulse.winning_confidence < cfg.entry_confidence_threshold:
            self._consecutive_evidence_beats = 0.0
            return

        # Check ambiguity margin
        if pulse.ambiguity_margin < cfg.entry_ambiguity_margin:
            self._consecutive_evidence_beats = 0.0
            log.debug(
                "Ambiguity margin %.4f < threshold %.4f",
                pulse.ambiguity_margin,
                cfg.entry_ambiguity_margin,
            )
            return

        # Check evidence freshness
        if pulse.evidence_age > cfg.max_evidence_age_beats * pulse.beat_period:
            self._consecutive_evidence_beats = 0.0
            return

        # Check bar state: need a downbeat prediction
        if bar.downbeat_time is None or bar.bar_duration is None:
            self._consecutive_evidence_beats = 0.0
            return

        if not bar.is_confident:
            self._consecutive_evidence_beats = 0.0
            return

        # Bar ambiguity check
        if bar.ambiguity_margin < cfg.entry_ambiguity_margin:
            self._consecutive_evidence_beats = 0.0
            log.debug(
                "Bar ambiguity margin %.4f < threshold %.4f",
                bar.ambiguity_margin,
                cfg.entry_ambiguity_margin,
            )
            return

        # Accumulate evidence beats
        beat_period = pulse.beat_period
        if self._last_pulse is not None and self._last_pulse.beat_period is not None:
            # Approximate beats passed since last update
            elapsed = now - self._last_pulse.computed_at
            beats_passed = elapsed / max(self._last_pulse.beat_period, 0.001)
            self._consecutive_evidence_beats += beats_passed
        else:
            self._consecutive_evidence_beats = beat_period  # first good pulse

        # Limit accumulation
        self._consecutive_evidence_beats = min(
            self._consecutive_evidence_beats, cfg.entry_min_evidence_beats * 2,
        )

        # Check minimum evidence duration
        if self._consecutive_evidence_beats < cfg.entry_min_evidence_beats:
            return

        # ── Arm for the next downbeat ──
        next_db = bar.next_downbeat(now)
        if next_db is None:
            return

        # Ensure the downbeat is in the future with some margin
        if next_db <= now:
            return

        self._armed_bpm = pulse.winning_bpm
        self._armed_downbeat = next_db
        self._transition_to(
            _State.ARMED,
            f"entry: bpm={pulse.winning_bpm:.1f} conf={pulse.winning_confidence:.3f} "
            f"margin={pulse.ambiguity_margin:.3f} downbeat=+{(next_db - now):.3f}s",
            now,
        )

    # ── ARMED ─────────────────────────────────────────────────────────

    def _do_armed(
        self,
        pulse: PulseAdapterState,
        bar: BarAdapterState,
        now: float,
    ) -> None:
        """Wait for the armed downbeat deadline, or cancel if evidence goes stale."""
        if self._armed_downbeat is None:
            # Shouldn't happen, but safety
            self._transition_to(_State.LISTENING, "armed without deadline", now)
            return

        # Check if we've reached the deadline
        if now >= self._armed_downbeat:
            self._transition_to(_State.PLAYING, "armed deadline reached", now)
            return

        # Check if evidence is still valid — cancel if stale/unsafe
        cfg = self._config

        if pulse.winning_bpm is None:
            self._transition_to(_State.LISTENING, "armed cancel: no pulse", now)
            return

        beat_period = pulse.beat_period or (60.0 / pulse.winning_bpm if pulse.winning_bpm else None)
        if beat_period is None:
            self._transition_to(_State.LISTENING, "armed cancel: no beat period", now)
            return

        # Confidence dropped below exit threshold
        if pulse.winning_confidence < cfg.exit_confidence_threshold:
            self._transition_to(
                _State.LISTENING,
                f"armed cancel: confidence {pulse.winning_confidence:.3f} < "
                f"exit {cfg.exit_confidence_threshold:.3f}",
                now,
            )
            return

        # Ambiguity returned
        if pulse.ambiguity_margin < cfg.entry_ambiguity_margin:
            self._transition_to(
                _State.LISTENING,
                f"armed cancel: ambiguity margin {pulse.ambiguity_margin:.3f} < "
                f"threshold {cfg.entry_ambiguity_margin:.3f}",
                now,
            )
            return

        # Evidence stale
        if pulse.evidence_age > cfg.max_evidence_age_beats * beat_period:
            self._transition_to(
                _State.LISTENING,
                f"armed cancel: evidence age {pulse.evidence_age:.3f}s > "
                f"limit {cfg.max_evidence_age_beats * beat_period:.3f}s",
                now,
            )
            return

    # ── PLAYING ───────────────────────────────────────────────────────

    def _do_playing(
        self,
        pulse: PulseAdapterState,
        bar: BarAdapterState,
        now: float,
    ) -> None:
        """Maintain locked clock; detect degradation and silence."""
        cfg = self._config

        if pulse.winning_bpm is None or self._beat_period is None:
            self._degraded_beats += 0.25  # approximate
        else:
            if pulse.winning_confidence < cfg.exit_confidence_threshold:
                # Accumulate degraded beats
                elapsed = now - (self._last_pulse.computed_at if self._last_pulse else now)
                beats = elapsed / max(self._beat_period, 0.001)
                self._degraded_beats += min(beats, 0.5)  # cap per tick
            else:
                # Confidence is back — decay the degraded counter
                self._degraded_beats = max(0.0, self._degraded_beats - 0.25)

        # Check degradation dwell
        if self._degraded_beats >= cfg.degradation_dwell_beats:
            self._requires_relock = False
            self._transition_to(
                _State.DEGRADED,
                f"degraded: confidence below {cfg.exit_confidence_threshold:.3f} "
                f"for {self._degraded_beats:.1f} beats",
                now,
            )
            return

        # Check for silence (no events arriving)
        silence_duration = max(pulse.evidence_age, bar.evidence_age)
        if silence_duration > cfg.silence_stop_timeout:
            self._transition_to(
                _State.STOPPED,
                f"silence timeout: {silence_duration:.1f}s > {cfg.silence_stop_timeout:.1f}s",
                now,
            )
            return

        # Track beats of silence
        if silence_duration > 0 and self._beat_period is not None:
            self._silence_beats = silence_duration / self._beat_period
        else:
            self._silence_beats = 0.0

        # ── Drift detection ──
        if (
            pulse.winning_bpm is not None
            and self._locked_bpm is not None
            and self._last_pulse is not None
            and self._last_pulse.winning_bpm is not None
        ):
            bpm_diff = abs(pulse.winning_bpm - self._locked_bpm)
            if bpm_diff / self._locked_bpm > cfg.tempo_drift_fraction:
                elapsed = now - self._last_pulse.computed_at
                self._tempo_drift_beats += min(
                    elapsed / self._beat_period, 0.5
                )
                if self._tempo_drift_beats >= cfg.tempo_drift_dwell_beats:
                    self._requires_relock = True
                    self._transition_to(
                        _State.DEGRADED,
                        f"tempo drift: {pulse.winning_bpm:.1f} vs locked {self._locked_bpm:.1f}",
                        now,
                    )
                    return
            else:
                self._tempo_drift_beats = max(
                    0.0, self._tempo_drift_beats - 0.25
                )

    # ── DEGRADED ──────────────────────────────────────────────────────

    def _do_degraded(
        self,
        pulse: PulseAdapterState,
        bar: BarAdapterState,
        now: float,
    ) -> None:
        """Hold the locked clock, watch for recovery or silence timeout."""
        cfg = self._config

        # A confidently changed tempo must return through LISTENING/ARMED so
        # the new grid can only lock on a newly predicted future downbeat.
        if (
            self._requires_relock
            and pulse.winning_bpm is not None
            and pulse.winning_confidence >= cfg.recovery_confidence_threshold
        ):
            self._transition_to(_State.LISTENING, "tempo drift requires re-lock", now)
            return

        if pulse.winning_bpm is None or self._beat_period is None:
            self._silence_beats += 0.25
        else:
            if pulse.winning_confidence >= cfg.recovery_confidence_threshold:
                elapsed = now - (self._last_pulse.computed_at if self._last_pulse else now)
                beats = elapsed / max(self._beat_period, 0.001)
                self._recovery_beats += min(beats, 0.5)
            else:
                self._recovery_beats = max(0.0, self._recovery_beats - 0.25)
                self._recovery_at = None

            # Silence tracking
            silence_duration = max(pulse.evidence_age, bar.evidence_age)
            if self._beat_period > 0:
                self._silence_beats = silence_duration / self._beat_period

        # Check silence stop timeout
        silence_duration = max(pulse.evidence_age, bar.evidence_age)
        if silence_duration > cfg.silence_stop_timeout:
            self._transition_to(
                _State.STOPPED,
                f"degraded silence timeout: {silence_duration:.1f}s",
                now,
            )
            return

        # Check recovery. First arm a boundary on the controller's locked
        # grid, then remain DEGRADED until that exact boundary arrives.
        if self._recovery_beats >= cfg.recovery_dwell_beats:
            if self._recovery_at is None:
                self._recovery_at = self._next_locked_bar_boundary(now)
            elif now >= self._recovery_at:
                if now - self._recovery_at <= cfg.late_budget_seconds:
                    self._transition_to(
                        _State.PLAYING,
                        f"recovery boundary: confidence {pulse.winning_confidence:.3f} >= "
                        f"{cfg.recovery_confidence_threshold:.3f} for "
                        f"{self._recovery_beats:.1f} beats",
                        now,
                    )
                    return
                # A delayed tick missed the safe boundary; wait for the next.
                self._recovery_at = self._next_locked_bar_boundary(now)

        # Check grace timeout for silence (beats-based)
        if self._silence_beats > cfg.silence_grace_beats:
            self._transition_to(
                _State.STOPPED,
                f"degraded silence grace exceeded: {self._silence_beats:.1f} beats",
                now,
            )
            return

    # ── Grid helpers ──────────────────────────────────────────────────

    def _update_grid_position(self, now: float) -> None:
        """Compute current bar_index and slot from locked grid."""
        if self._bar_epoch is None or self._beat_period is None:
            return
        elapsed = now - self._bar_epoch
        if elapsed < 0:
            return
        total_beats = elapsed / self._beat_period
        bars_elapsed = int(total_beats / self._config.beats_per_bar)
        beat_in_bar = total_beats % self._config.beats_per_bar
        self._bar_index = bars_elapsed
        self._current_slot = int(beat_in_bar * (self._config.slots_per_bar // self._config.beats_per_bar))
        self._current_slot = min(self._current_slot, self._config.slots_per_bar - 1)

    def _next_locked_bar_boundary(self, now: float) -> float | None:
        if self._bar_epoch is None or self._beat_period is None:
            return None
        bar_duration = self._beat_period * self._config.beats_per_bar
        bars_elapsed = int(max(0.0, now - self._bar_epoch) / bar_duration)
        candidate = self._bar_epoch + (bars_elapsed + 1) * bar_duration
        if candidate <= now:
            candidate += bar_duration
        return candidate

    # ── Snapshot ──────────────────────────────────────────────────────

    def _snapshot(self, now: float) -> ControllerSnapshot:
        return ControllerSnapshot(
            state=self._state,
            generation=self._generation,
            locked_bpm=self._locked_bpm,
            bar_epoch=self._bar_epoch,
            beat_period=self._beat_period,
            current_bar_index=self._bar_index,
            current_slot=self._current_slot,
            mirror_active=self._mirror_active,
            mirror_slot=self._mirror_slot,
            anchor_count=0,  # filled by pocket planner if needed
            mirror_count=0,
            queue_depth=0,
            computed_at=now,
        )

    # ── Config logging ────────────────────────────────────────────────

    def _log_config(self) -> None:
        cfg = self._config
        log.info(
            "LiveConfig — bpm=[%.0f, %.0f] entry_conf=%.2f entry_beats=%.1f "
            "ambiguity=%.2f exit_conf=%.2f deg_dwell=%.1f "
            "rec_conf=%.2f rec_dwell=%.1f silence_stop=%.1fs "
            "silence_grace=%.1f evidence_age=%.1f drift=%.0f%% "
            "drift_dwell=%.1f lookahead=%.3fs late=%.3fs "
            "mirror_bars=%d mirror_pct=%.0f mirror_vel=%d",
            cfg.min_bpm,
            cfg.max_bpm,
            cfg.entry_confidence_threshold,
            cfg.entry_min_evidence_beats,
            cfg.entry_ambiguity_margin,
            cfg.exit_confidence_threshold,
            cfg.degradation_dwell_beats,
            cfg.recovery_confidence_threshold,
            cfg.recovery_dwell_beats,
            cfg.silence_stop_timeout,
            cfg.silence_grace_beats,
            cfg.max_evidence_age_beats,
            cfg.tempo_drift_fraction * 100,
            cfg.tempo_drift_dwell_beats,
            cfg.max_lookahead_seconds,
            cfg.late_budget_seconds,
            cfg.mirror_min_stable_bars,
            cfg.mirror_strength_percentile,
            cfg.mirror_velocity,
        )
