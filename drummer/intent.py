"""Groove Intent Engine — Module 4: perception-to-behaviour decisions.

Converts musical perception state (pulse, bar, events) into high-level
drummer behaviour decisions expressed as GrooveIntent objects.

This module does NOT generate MIDI.
This module does NOT choose specific kick/snare/hat patterns.

It answers: "Given what the system currently hears, what kind of drummer
behaviour is appropriate?"

The engine models a drummer who:
- Listens before playing (WAIT)
- Enters conservatively when confident (ENTER)
- Holds steady patterns (HOLD)
- Builds intensity when energy rises (BUILD)
- Makes space when energy falls (REDUCE / SIMPLIFY)
- Prepares fills near phrase boundaries (PREPARE_FILL)
- Marks downbeats with emphasis (MARK_DOWNBEAT)
- Resets when the music resets (RESET)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto

from perception.models import MusicalEvent
from perception.pulse import PulseState
from perception.bar import BarState

log = logging.getLogger(__name__)

# ── Thresholds ─────────────────────────────────────────────────────

MIN_PULSE_CONFIDENCE_TO_PLAY: float = 0.40
MIN_BAR_CONFIDENCE_TO_PLAY: float = 0.30
HIGH_CONFIDENCE: float = 0.70
ENERGY_RISE_THRESHOLD: float = 0.15
ENERGY_DROP_THRESHOLD: float = 0.15
DENSITY_RISE_THRESHOLD: float = 0.15
DENSITY_DROP_THRESHOLD: float = 0.15
MIN_BARS_BETWEEN_FILLS: int = 4
FILL_WINDOW_BEATS: float = 1.5  # beats before bar end where fill may trigger
MIN_CONSECUTIVE_STABLE: int = 3   # stable updates before declaring HOLD
ENERGY_HISTORY_SIZE: int = 8      # events to track for trend detection


# ── Enum ───────────────────────────────────────────────────────────


class GrooveAction(Enum):
    """High-level drummer behaviour decision."""
    WAIT = auto()           # Not confident enough to play yet
    ENTER = auto()          # First entry — conservative, listening
    HOLD = auto()           # Steady state — maintain current groove
    BUILD = auto()          # Increase intensity/complexity
    REDUCE = auto()         # Reduce intensity/complexity
    SIMPLIFY = auto()       # Strip back to essentials
    MARK_DOWNBEAT = auto()  # Emphasise this downbeat
    PREPARE_FILL = auto()   # Indicate a fill is appropriate
    RESET = auto()          # Regime change — start fresh


# ── Dataclass ──────────────────────────────────────────────────────


@dataclass
class GrooveIntent:
    """A high-level drummer behaviour intention.

    Not a drum pattern. Not MIDI. Just intent.

    Parameters
    ----------
    action : GrooveAction
        The primary behavioural action.
    confidence : float
        Overall confidence in this intent decision.
    energy_level : float
        Current estimated musical energy level [0, 1].
    density_level : float
        Current attack density level [0, 1].
    pulse_confidence : float
        Confidence in pulse estimation.
    bar_confidence : float
        Confidence in bar/downbeat estimation.
    suggested_complexity : float
        Suggested complexity level [0, 1] for the drummer.
    suggested_velocity : float
        Suggested velocity factor [0, 1] for the drummer.
    should_play : bool
        Whether the drummer should play at all.
    should_fill : bool
        Whether a fill is appropriate right now.
    reason : str
        Human-readable explanation for debugging/demos.
    timestamp : float
        Time of this intent decision.
    """
    action: GrooveAction = GrooveAction.WAIT
    confidence: float = 0.0
    energy_level: float = 0.0
    density_level: float = 0.0
    pulse_confidence: float = 0.0
    bar_confidence: float = 0.0
    suggested_complexity: float = 0.0
    suggested_velocity: float = 0.0
    should_play: bool = False
    should_fill: bool = False
    reason: str = ""
    timestamp: float = 0.0


# ── Engine ────────────────────────────────────────────────────────


class GrooveIntentEngine:
    """Converts perception state into drummer behaviour intentions.

    Parameters
    ----------
    energy_history_size : int
        Number of recent event energies to track for trend detection.
    """

    def __init__(
        self,
        energy_history_size: int = ENERGY_HISTORY_SIZE,
    ) -> None:
        self._energy_history: list[float] = []
        self._density_history: list[float] = []
        self._history_size = energy_history_size
        self._current_intent = GrooveIntent(action=GrooveAction.WAIT, reason="initialised")
        self._bars_since_fill: int = 999  # large number — allow fill after enough time
        self._stable_updates: int = 0
        self._has_played: bool = False
        self._previous_action: GrooveAction = GrooveAction.WAIT
        self._event_count: int = 0
        self._last_event_time: float = 0.0

        log.info(
            "GrooveIntentEngine initialised — history=%d events",
            energy_history_size,
        )

    # ── Public API ─────────────────────────────────────────────────

    def update(
        self,
        event: MusicalEvent | None,
        pulse_state: PulseState,
        bar_state: BarState,
    ) -> GrooveIntent:
        """Process perception state and produce an updated intent.

        Parameters
        ----------
        event : MusicalEvent | None
            The current musical event, or None if advancing time.
        pulse_state : PulseState
            Current pulse hypotheses.
        bar_state : BarState
            Current bar position hypotheses.

        Returns
        -------
        GrooveIntent
            The intent decision for this moment.
        """
        # Update energy/density history from event
        if event is not None:
            self._energy_history.append(event.energy)
            self._density_history.append(event.density)
            if len(self._energy_history) > self._history_size:
                self._energy_history = self._energy_history[-self._history_size:]
            if len(self._density_history) > self._history_size:
                self._density_history = self._density_history[-self._history_size:]
            self._last_event_time = event.time_seconds
            self._event_count += 1

        # Compute current levels
        energy_level = self._compute_energy_level()
        density_level = self._compute_density_level()
        pulse_conf = pulse_state.confidence
        bar_conf = bar_state.confidence

        # Detect energy/density trends
        energy_trend = self._compute_energy_trend()
        density_trend = self._compute_density_trend()

        # Determine action
        action, reason = self._decide_action(
            pulse_conf=pulse_conf,
            bar_conf=bar_conf,
            energy_level=energy_level,
            density_level=density_level,
            energy_trend=energy_trend,
            density_trend=density_trend,
            bar_state=bar_state,
            event=event,
        )

        # Compute suggested parameters
        should_play = self._should_play(action, pulse_conf, bar_conf)
        should_fill = self._should_fill(action, pulse_conf, bar_conf, bar_state)

        if action in (GrooveAction.ENTER, GrooveAction.HOLD, GrooveAction.BUILD, GrooveAction.MARK_DOWNBEAT):
            self._has_played = True

        # Track bars since fill
        if should_fill:
            self._bars_since_fill = 0
        elif bar_state.estimated_beat_in_bar == 0 and self._previous_action != GrooveAction.WAIT:
            self._bars_since_fill += 1

        suggested_complexity = self._compute_complexity(action, energy_level, density_level, pulse_conf, bar_conf)
        suggested_velocity = self._compute_velocity(energy_level, pulse_conf, bar_conf)

        self._previous_action = action

        timestamp = event.time_seconds if event else self._last_event_time

        self._current_intent = GrooveIntent(
            action=action,
            confidence=max(pulse_conf, bar_conf),
            energy_level=round(energy_level, 3),
            density_level=round(density_level, 3),
            pulse_confidence=round(pulse_conf, 3),
            bar_confidence=round(bar_conf, 3),
            suggested_complexity=round(suggested_complexity, 3),
            suggested_velocity=round(suggested_velocity, 3),
            should_play=should_play,
            should_fill=should_fill,
            reason=reason,
            timestamp=round(timestamp, 3),
        )
        return self._current_intent

    def get_current_intent(self) -> GrooveIntent:
        """Return the most recent intent without updating."""
        return self._current_intent

    def reset(self) -> None:
        """Clear all internal state to initial conditions."""
        self._energy_history.clear()
        self._density_history.clear()
        self._current_intent = GrooveIntent(action=GrooveAction.WAIT, reason="reset")
        self._bars_since_fill = 999
        self._stable_updates = 0
        self._has_played = False
        self._previous_action = GrooveAction.WAIT
        self._event_count = 0
        self._last_event_time = 0.0
        log.debug("GrooveIntentEngine reset")

    # ── Internal: Compute ─────────────────────────────────────────

    def _compute_energy_level(self) -> float:
        if not self._energy_history:
            return 0.0
        return sum(self._energy_history) / len(self._energy_history)

    def _compute_density_level(self) -> float:
        if not self._density_history:
            return 0.0
        return sum(self._density_history) / len(self._density_history)

    def _compute_energy_trend(self) -> str:
        """Return 'rising', 'falling', or 'steady'."""
        if len(self._energy_history) < 4:
            return "steady"
        mid = len(self._energy_history) // 2
        first_half = self._energy_history[:mid]
        second_half = self._energy_history[mid:]
        first_avg = sum(first_half) / len(first_half) + 0.001
        second_avg = sum(second_half) / len(second_half) + 0.001
        ratio = second_avg / first_avg
        if ratio > 1.0 + ENERGY_RISE_THRESHOLD:
            return "rising"
        elif ratio < 1.0 - ENERGY_DROP_THRESHOLD:
            return "falling"
        return "steady"

    def _compute_density_trend(self) -> str:
        """Return 'rising', 'falling', or 'steady'."""
        if len(self._density_history) < 4:
            return "steady"
        mid = len(self._density_history) // 2
        first_half = self._density_history[:mid]
        second_half = self._density_history[mid:]
        first_avg = sum(first_half) / len(first_half) + 0.001
        second_avg = sum(second_half) / len(second_half) + 0.001
        ratio = second_avg / first_avg
        if ratio > 1.0 + DENSITY_RISE_THRESHOLD:
            return "rising"
        elif ratio < 1.0 - DENSITY_DROP_THRESHOLD:
            return "falling"
        return "steady"

    def _compute_complexity(
        self,
        action: GrooveAction,
        energy_level: float,
        density_level: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> float:
        """Compute suggested complexity from current state."""
        if action == GrooveAction.WAIT:
            return 0.0

        if action == GrooveAction.SIMPLIFY:
            return 0.15

        if action == GrooveAction.REDUCE:
            return 0.3

        if action == GrooveAction.ENTER:
            # Conservative start — moderate complexity
            base = 0.3
            # Slightly higher if confident
            conf_factor = min(pulse_conf, bar_conf)
            return base + conf_factor * 0.15

        if action == GrooveAction.BUILD:
            # Complexity rises with energy and confidence but capped
            base = 0.4
            boost = energy_level * 0.3 + density_level * 0.2
            conf_boost = min(pulse_conf, bar_conf) * 0.1
            return min(1.0, base + boost + conf_boost)

        if action == GrooveAction.HOLD:
            base = 0.4
            conf_factor = min(pulse_conf, bar_conf)
            return base + conf_factor * 0.2

        # Default moderate
        return 0.4

    def _compute_velocity(
        self,
        energy_level: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> float:
        """Compute suggested velocity from current state."""
        # Velocity tracks energy, dampened by uncertainty
        conf_factor = min(pulse_conf, bar_conf)
        return energy_level * 0.7 + conf_factor * 0.3

    # ── Internal: Decision Logic ───────────────────────────────────

    def _decide_action(
        self,
        pulse_conf: float,
        bar_conf: float,
        energy_level: float,
        density_level: float,
        energy_trend: str,
        density_trend: str,
        bar_state: BarState,
        event: MusicalEvent | None,
    ) -> tuple[GrooveAction, str]:
        """Core decision logic — what should the drummer do?"""

        can_play = (
            pulse_conf >= MIN_PULSE_CONFIDENCE_TO_PLAY
            and bar_conf >= MIN_BAR_CONFIDENCE_TO_PLAY
        )

        is_confident = (
            pulse_conf >= HIGH_CONFIDENCE and bar_conf >= HIGH_CONFIDENCE
        )

        # ── WAIT ──────────────────────────────────────────────
        if not can_play:
            if pulse_conf < MIN_PULSE_CONFIDENCE_TO_PLAY:
                return GrooveAction.WAIT, "waiting: pulse confidence too low"
            return GrooveAction.WAIT, "waiting: bar confidence too low"

        # ── ENTER ─────────────────────────────────────────────
        if not self._has_played:
            self._stable_updates += 1
            return GrooveAction.ENTER, (
                "entering: first stable pulse and bar — entering conservatively"
            )

        # ── RESET (large bar gap) ─────────────────────────────
        if bar_state.estimated_beat_in_bar == 0 and self._event_count > 20 and (
            self._previous_action in (GrooveAction.HOLD, GrooveAction.BUILD)
        ):
            if energy_trend == "falling" and energy_level < 0.2:
                return GrooveAction.RESET, "reset: energy dropped sharply at bar boundary"

        # Trend-based actions require a minimum history to stabilise
        can_react_to_trend = self._event_count >= 8

        # ── BUILD ─────────────────────────────────────────────
        if can_react_to_trend and (energy_trend == "rising" or density_trend == "rising"):
            return GrooveAction.BUILD, (
                f"build: energy {energy_trend}, density {density_trend}"
            )

        # ── REDUCE / SIMPLIFY ─────────────────────────────────
        if can_react_to_trend and (energy_trend == "falling" or density_trend == "falling"):
            if energy_level < 0.25:
                return GrooveAction.SIMPLIFY, "simplify: energy very low"
            return GrooveAction.REDUCE, (
                f"reduce: energy {energy_trend}, level={energy_level:.2f}"
            )

        # ── PREPARE_FILL ─────────────────────────────────────
        if (
            is_confident
            and self._bars_since_fill >= MIN_BARS_BETWEEN_FILLS
            and bar_state.estimated_beat_in_bar is not None
            and bar_state.estimated_beat_in_bar >= 2
            and (
                bar_state.best_hypothesis is not None
                and bar_state.estimated_beat_in_bar
                >= bar_state.best_hypothesis.beats_per_bar - FILL_WINDOW_BEATS
            )
            and energy_trend in ("steady", "rising")
        ):
            return GrooveAction.PREPARE_FILL, (
                f"fill: near bar end (beat {bar_state.estimated_beat_in_bar}), "
                f"{self._bars_since_fill} bars since last fill"
            )

        # ── MARK_DOWNBEAT ────────────────────────────────────
        if (
            is_confident
            and bar_state.estimated_beat_in_bar == 0
            and event is not None
            and event.strength > 0.7
        ):
            return GrooveAction.MARK_DOWNBEAT, (
                "downbeat: strong event on beat 1 with high confidence"
            )

        # ── HOLD ──────────────────────────────────────────────
        self._stable_updates += 1
        return GrooveAction.HOLD, (
            f"hold: stable groove, pulse={pulse_conf:.0%}, bar={bar_conf:.0%}"
        )

    # ── Internal: Boolean Decisions ───────────────────────────────

    def _should_play(
        self,
        action: GrooveAction,
        pulse_conf: float,
        bar_conf: float,
    ) -> bool:
        """Determine whether the drummer should produce sound."""
        if action == GrooveAction.WAIT:
            return False
        if action == GrooveAction.RESET:
            return False
        if action == GrooveAction.SIMPLIFY:
            return True  # play, but minimally
        # All other actions imply playing
        return (
            pulse_conf >= MIN_PULSE_CONFIDENCE_TO_PLAY
            and bar_conf >= MIN_BAR_CONFIDENCE_TO_PLAY
        )

    def _should_fill(
        self,
        action: GrooveAction,
        pulse_conf: float,
        bar_conf: float,
        bar_state: BarState,
    ) -> bool:
        """Determine whether a fill is appropriate."""
        if action != GrooveAction.PREPARE_FILL:
            return False
        return (
            pulse_conf >= HIGH_CONFIDENCE
            and bar_conf >= HIGH_CONFIDENCE
            and self._bars_since_fill >= MIN_BARS_BETWEEN_FILLS
        )