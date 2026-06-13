"""Feature Monitor — continuous musical feature summarisation.

This module observes ``MusicalEvent`` objects over time and produces
``FeatureSnapshot`` values that describe *what the player seems to be
doing*.  It does **not** make drum decisions — that is the Behaviour
Engine's job.

Design contract
---------------
* Read from MusicalEvent stream.
* Summarise density, strength, change, silence, repetition, and phase.
* Produce transparent, explainable features.
* Never decide BUILD / REDUCE / ENTER / BAIL / FILL.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.models import MusicalEvent


# ---------------------------------------------------------------------------
# FeatureSnapshot — the measurable musical state at a point in time
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureSnapshot:
    """A point-in-time summary of the player's musical behaviour.

    All fields are in [0, 1] unless documented otherwise.
    """

    timestamp: float
    """The time (seconds) this snapshot was computed for."""

    input_density: float = 0.0
    """Normalised density of recent events in [0, 1]."""

    strength_ema: float = 0.0
    """Exponentially-weighted moving average of event strength."""

    fast_strength_ema: float = 0.0
    """Fast-reacting EMA of event strength (sensitive to change)."""

    slow_strength_ema: float = 0.0
    """Slow-reacting EMA of event strength (stable baseline)."""

    change_score: float = 0.0
    """Divergence between fast and slow strength EMA, clamped to [0, 1].

    Higher values indicate the player is suddenly playing louder/softer
    than their recent norm.
    """

    silence_duration: float = 0.0
    """Seconds since the last event.  0.0 if no event has ever arrived."""

    repetition_stability: float = 0.0
    """How repetitive / even the recent event spacing is, in [0, 1].

    1.0 = perfectly regular spacing.  0.0 = highly erratic.
    """

    phase_alignment: Optional[float] = None
    """How well events align to a quantised pulse grid, if known.

    Supplied externally (e.g. from the pulse/bar tracker).  ``None``
    means no phase information is available.
    """

    player_certainty: float = 0.0
    """Composite confidence that the player is playing intentionally.

    0.0–1.0 bounded.  High when strength, repetition, and (if available)
    phase alignment are all healthy.
    """


# ---------------------------------------------------------------------------
# FeatureMonitorConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureMonitorConfig:
    """Tuning knobs for the Feature Monitor.

    All parameters have sensible defaults.  Tweak them to adjust how
    quickly / sensitively the monitor reacts to changes.
    """

    density_window_seconds: float = 2.0
    """Rolling window for counting recent events."""

    strength_alpha: float = 0.15
    """EMA smoothing factor for normal strength tracking."""

    fast_strength_alpha: float = 0.35
    """EMA smoothing factor for the fast (change-sensitive) tracker."""

    slow_strength_alpha: float = 0.05
    """EMA smoothing factor for the slow (baseline) tracker."""

    repetition_window_beats: int = 4
    """How many inter-onset intervals to consider for repetition stability."""

    silence_timeout_seconds: float = 1.0
    """After this much silence certainty starts to decay."""

    max_expected_density: float = 12.0
    """Events-per-window that maps to density = 1.0."""

    certainty_strength_weight: float = 0.3
    """Weight of strength_ema in player_certainty."""

    certainty_repetition_weight: float = 0.4
    """Weight of repetition_stability in player_certainty."""

    certainty_phase_weight: float = 0.3
    """Weight of phase_alignment in player_certainty (if available)."""

    decay_per_second: float = 0.3
    """How much strength EMA decays per second when no events arrive.

    A value of 0.3 means the EMA loses ~26 % of its value each second
    (applied multiplicatively as ``ema *= e^{-decay * dt}``).
    """

    min_iois_for_stability: int = 2
    """Minimum number of inter-onset intervals needed for a meaningful
    repetition stability estimate.  Fewer than this returns 0.0."""


# ---------------------------------------------------------------------------
# FeatureMonitor
# ---------------------------------------------------------------------------


@dataclass
class FeatureMonitor:
    """Observes MusicalEvents and continuously summarises musical features.

    Usage
    -----
    >>> fm = FeatureMonitor()
    >>> for event in event_stream:
    ...     snapshot = fm.feed(event)
    ...     print(snapshot.player_certainty)
    """

    config: FeatureMonitorConfig = field(default_factory=FeatureMonitorConfig)

    # -- internal state -------------------------------------------------------
    _events: list[MusicalEvent] = field(default_factory=list, repr=False)
    _strength_ema: float = 0.0
    _fast_ema: float = 0.0
    _slow_ema: float = 0.0
    _last_event_time: Optional[float] = None
    _last_snapshot_time: float = 0.0

    def feed(self, event: MusicalEvent) -> FeatureSnapshot:
        """Accept a single ``MusicalEvent`` and return an up-to-date snapshot.

        Parameters
        ----------
        event : MusicalEvent
            The latest detected musical event.

        Returns
        -------
        FeatureSnapshot
            Current feature summary including the new event.
        """
        # ---- bookkeeping ----
        self._events.append(event)
        self._last_event_time = event.time_seconds
        now = event.time_seconds

        # ---- prune old events outside density window ----
        cutoff = now - self.config.density_window_seconds
        while self._events and self._events[0].time_seconds < cutoff:
            self._events.pop(0)

        # ---- update strength EMAs ----
        dt = now - self._last_snapshot_time
        if dt > 0:
            self._decay_strength_emas(dt)

        self._update_ema(event.strength)

        self._last_snapshot_time = now

        # ---- compute features ----
        input_density = self._compute_density()
        change_score = self._compute_change_score()
        silence_dur = self._compute_silence_duration(now)
        repetition = self._compute_repetition_stability()
        phase = self._get_phase_alignment(event)
        certainty = self._compute_certainty(repetition, phase)

        return FeatureSnapshot(
            timestamp=now,
            input_density=input_density,
            strength_ema=self._strength_ema,
            fast_strength_ema=self._fast_ema,
            slow_strength_ema=self._slow_ema,
            change_score=change_score,
            silence_duration=silence_dur,
            repetition_stability=repetition,
            phase_alignment=phase,
            player_certainty=certainty,
        )

    def snapshot(self, now: float, phase_alignment: Optional[float] = None) -> FeatureSnapshot:
        """Return a snapshot at an arbitrary time (no new event).

        Use this to poll the monitor when the player is silent or to inject
        phase-alignment data from the pulse/bar tracker.

        Parameters
        ----------
        now : float
            Current time in seconds.
        phase_alignment : float | None
            Optional phase-alignment value to include in the snapshot.

        Returns
        -------
        FeatureSnapshot
        """
        dt = now - self._last_snapshot_time
        if dt > 0:
            self._decay_strength_emas(dt)

        self._last_snapshot_time = now

        # prune density window
        cutoff = now - self.config.density_window_seconds
        while self._events and self._events[0].time_seconds < cutoff:
            self._events.pop(0)

        input_density = self._compute_density()
        change_score = self._compute_change_score()
        silence_dur = self._compute_silence_duration(now)
        repetition = self._compute_repetition_stability()
        certainty = self._compute_certainty(repetition, phase_alignment)

        return FeatureSnapshot(
            timestamp=now,
            input_density=input_density,
            strength_ema=self._strength_ema,
            fast_strength_ema=self._fast_ema,
            slow_strength_ema=self._slow_ema,
            change_score=change_score,
            silence_duration=silence_dur,
            repetition_stability=repetition,
            phase_alignment=phase_alignment,
            player_certainty=certainty,
        )

    def reset(self) -> None:
        """Clear all internal state."""
        self._events.clear()
        self._strength_ema = 0.0
        self._fast_ema = 0.0
        self._slow_ema = 0.0
        self._last_event_time = None
        self._last_snapshot_time = 0.0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_ema(self, strength: float) -> None:
        """Blend a new strength value into all three EMAs."""
        a = self.config.strength_alpha
        fa = self.config.fast_strength_alpha
        sa = self.config.slow_strength_alpha
        self._strength_ema = a * strength + (1 - a) * self._strength_ema
        self._fast_ema = fa * strength + (1 - fa) * self._fast_ema
        self._slow_ema = sa * strength + (1 - sa) * self._slow_ema

    def _decay_strength_emas(self, dt: float) -> None:
        """Smoothly decay strength EMAs when no events are arriving.

        Uses exponential decay: ``ema *= exp(-decay * dt)``.
        """
        import math
        factor = math.exp(-self.config.decay_per_second * dt)
        self._strength_ema *= factor
        self._fast_ema *= factor
        self._slow_ema *= factor

    def _compute_density(self) -> float:
        """Count recent events within the density window and normalise."""
        count = len(self._events)
        max_d = self.config.max_expected_density
        if max_d <= 0:
            return 0.0
        raw = count / max_d
        return max(0.0, min(1.0, raw))

    def _compute_change_score(self) -> float:
        """Absolute divergence between fast and slow EMA, clamped to [0, 1]."""
        return max(0.0, min(1.0, abs(self._fast_ema - self._slow_ema)))

    def _compute_silence_duration(self, now: float) -> float:
        """Seconds since the last event (0.0 if none ever)."""
        if self._last_event_time is None:
            return 0.0
        return max(0.0, now - self._last_event_time)

    def _compute_repetition_stability(self) -> float:
        """Estimate how regular/spaced recent events are.

        Strategy (explainable, no ML)
        ------------------------------
        1. Extract inter-onset intervals (IOIs) from recent events.
        2. Compute the coefficient of variation (std / mean) of IOIs.
        3. Invert: ``stability = 1.0 - CV``, clamped to [0, 1].

        * High CV → erratic timing → low stability.
        * Low CV → regular timing → high stability.
        * Too few IOIs → neutral 0.0.
        """
        import math

        if len(self._events) < 2:
            return 0.0

        iois: list[float] = []
        for i in range(1, len(self._events)):
            gap = self._events[i].time_seconds - self._events[i - 1].time_seconds
            if gap > 0:
                iois.append(gap)

        # Cap to the repetition window (approximate — use recent N+1 events)
        n = self.config.repetition_window_beats + 1
        if len(iois) > n:
            iois = iois[-n:]

        if len(iois) < self.config.min_iois_for_stability:
            return 0.0

        mean_ioi = sum(iois) / len(iois)
        if mean_ioi <= 0:
            return 0.0

        variance = sum((ioi - mean_ioi) ** 2 for ioi in iois) / len(iois)
        stddev = math.sqrt(variance)
        cv = stddev / mean_ioi  # coefficient of variation

        # CV > 1.0 is very erratic — bottom out at 0.0
        stability = max(0.0, min(1.0, 1.0 - cv))
        return stability

    def _get_phase_alignment(self, event: MusicalEvent) -> Optional[float]:
        """Placeholder: phase alignment is supplied externally.

        MusicalEvent does not carry phase_alignment.  Callers should use
        ``snapshot(now, phase_alignment=...)`` to inject it.
        """
        return None

    def _compute_certainty(
        self,
        repetition_stability: float,
        phase_alignment: Optional[float],
    ) -> float:
        """Composite 0.0–1.0 confidence score.

        Weights are from config.
        If phase_alignment is None, its weight is redistributed proportionally
        to strength and repetition.
        """
        cfg = self.config
        s_weight = cfg.certainty_strength_weight
        r_weight = cfg.certainty_repetition_weight
        p_weight = cfg.certainty_phase_weight

        score = s_weight * self._strength_ema + r_weight * repetition_stability

        if phase_alignment is not None:
            # phase_alignment is expected to be in [0, 1]
            pa = max(0.0, min(1.0, phase_alignment))
            score += p_weight * pa
            total_weight = s_weight + r_weight + p_weight
        else:
            # Redistribute phase weight proportionally
            total_weight = s_weight + r_weight
            if total_weight > 0:
                scale = (s_weight + r_weight + p_weight) / total_weight
                score *= scale
            else:
                score = 1.0  # all weights are zero — default to full certainty

        # Clamp
        return max(0.0, min(1.0, score))