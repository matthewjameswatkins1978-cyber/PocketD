"""Anchor ledger and kick-mirror observation/selection for the Straight Pocket.

This module plans ledger events (anchor + mirror) for the locked playing
grid.  It does **not** run a timing loop or emit MIDI.  It asks the
controller for the current state (generation, grid, mirror slot) and
returns immutable ``LedgerEvent`` records.

Quantisation: observations at time *t* are snapped to the nearest slot on
the locked grid, producing ``(absolute_bar_index, slot, offset_seconds)``.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass

from drummer.live_controller import LiveController
from drummer.live_models import (
    LedgerEvent,
    LiveConfig,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class SlotObservation:
    """A quantised observation of a player hit on the locked grid."""

    bar_index: int
    """Absolute bar number."""
    slot: int
    """Slot 0..15 within the bar."""
    offset_seconds: float
    """Signed offset from the ideal slot centre (negative = early)."""
    strength: float
    """Normalised strength [0, 1]."""
    observed_at: float
    """Monotonic timestamp of observation."""


def quantise_to_locked_grid(
    observed_at: float,
    bar_epoch: float,
    beat_period: float,
    config: LiveConfig,
) -> tuple[int, int, float]:
    """Quantise a monotonic observation time onto the locked grid.

    Parameters
    ----------
    observed_at : float
        Monotonic timestamp of the observation.
    bar_epoch : float
        Monotonic time of beat 1 of bar 0 on the locked grid.
    beat_period : float
        Seconds per beat at the locked BPM.
    config : LiveConfig
        Configuration with ``beats_per_bar``, ``slots_per_bar``,
        and ``quantisation_tolerance_beats``.

    Returns
    -------
    tuple[int, int, float]
        ``(absolute_bar_index, slot, offset_seconds)``.
        *offset_seconds* is positive if the observation was late, negative
        if early.  Returns ``(-1, -1, 0.0)`` if the observation is outside
        the quantisation tolerance.
    """
    elapsed = observed_at - bar_epoch
    if elapsed < 0:
        return (-1, -1, 0.0)

    total_slots = elapsed / (beat_period / (config.slots_per_bar // config.beats_per_bar))
    nearest_slot = round(total_slots)
    slot_float = total_slots

    # Compute the ideal centre of the nearest slot
    slot_duration = beat_period / (config.slots_per_bar // config.beats_per_bar)
    ideal_time = bar_epoch + nearest_slot * slot_duration
    offset = observed_at - ideal_time

    # Check tolerance
    beat_offset = abs(offset) / beat_period
    if beat_offset > config.quantisation_tolerance_beats:
        return (-1, -1, 0.0)

    slots_per_bar = config.slots_per_bar
    bar_index = nearest_slot // slots_per_bar
    slot = nearest_slot % slots_per_bar

    return (bar_index, slot, offset)


# ── Anchor planner ────────────────────────────────────────────────────


def plan_anchor_ledger(
    controller: LiveController,
    config: LiveConfig,
    bar_index: int,
    now: float,
) -> list[LedgerEvent]:
    """Generate anchor ledger events for the given bar.

    Called once per bar before the bar starts.  Produces kick, snare,
    and hat events for the anchor slots.

    Parameters
    ----------
    controller : LiveController
        State owner — read locked BPM, bar_epoch, generation.
    config : LiveConfig
        Immutable configuration.
    bar_index : int
        Absolute bar number to plan.
    now : float
        Current monotonic time (to compute deadlines).

    Returns
    -------
    list[LedgerEvent]
        Anchor events for this bar.  Mirrored kick is NOT included here;
        it is planned separately via ``plan_mirror_ledger``.
    """
    if controller.bar_epoch is None or controller.beat_period is None:
        return []

    gen = controller.generation
    bp = controller.beat_period
    epoch = controller.bar_epoch
    slots_per_bar = config.slots_per_bar

    slot_duration = bp / (slots_per_bar // config.beats_per_bar)

    events: list[LedgerEvent] = []

    def _add(note: int, slot: int, velocity: int, source: str) -> None:
        deadline = epoch + bar_index * bp * config.beats_per_bar + slot * slot_duration
        event_id = f"{source}:bar{bar_index}:slot{slot}"
        events.append(
            LedgerEvent(
                event_id=event_id,
                deadline=deadline,
                note=note,
                velocity=velocity,
                channel=config.drum_channel,
                bar_index=bar_index,
                slot=slot,
                source=source,
                generation=gen,
            )
        )

    # Kick anchors
    for s in config.kick_anchor_slots:
        _add(config.kick_note, s, config.anchor_velocity, "anchor")

    # Snare anchors
    for s in config.snare_anchor_slots:
        _add(config.snare_note, s, config.anchor_velocity, "anchor")

    # Hat anchors
    for s in config.hat_anchor_slots:
        _add(config.closed_hat_note, s, config.hat_velocity, "hat")

    return events


def plan_mirror_ledger(
    controller: LiveController,
    config: LiveConfig,
    bar_index: int,
    now: float,
) -> list[LedgerEvent]:
    """Generate the mirrored kick event for a bar, if mirror is active.

    Must only be called while ``controller.mirror_active`` is True and
    ``controller.mirror_slot`` is a valid non-anchor slot.
    """
    if not controller.mirror_active or controller.mirror_slot is None:
        return []

    if controller.bar_epoch is None or controller.beat_period is None:
        return []

    gen = controller.generation
    bp = controller.beat_period
    epoch = controller.bar_epoch
    slots_per_bar = config.slots_per_bar
    slot_duration = bp / (slots_per_bar // config.beats_per_bar)
    slot = controller.mirror_slot
    deadline = epoch + bar_index * bp * config.beats_per_bar + slot * slot_duration

    event_id = f"mirror:bar{bar_index}:slot{slot}"

    return [
        LedgerEvent(
            event_id=event_id,
            deadline=deadline,
            note=config.kick_note,
            velocity=config.mirror_velocity,
            channel=config.drum_channel,
            bar_index=bar_index,
            slot=slot,
            source="mirror",
            generation=gen,
        )
    ]


# ── Kick Mirror observer ──────────────────────────────────────────────


class KickMirrorObserver:
    """Observes quantised player hits and decides whether to activate a mirrored kick.

    Parameters
    ----------
    config : LiveConfig
        Thresholds for mirror eligibility.
    """

    def __init__(self, config: LiveConfig) -> None:
        self._config = config
        self._observations: deque[SlotObservation] = deque()
        self._slot_hits: dict[int, list[float]] = {}  # slot -> list of strengths

        # Per-bar tracking
        self._current_bar_hits: dict[int, float] = {}
        """slot -> max strength observed in the current bar."""
        self._previous_bar_hits: dict[int, float] = {}
        """slot -> max strength from the previous bar."""
        self._unsupported_bars: int = 0

        # Active state
        self._active_slot: int | None = None

    def observe(self, obs: SlotObservation) -> None:
        """Record a quantised observation for the current playing bar."""
        slot = obs.slot

        # Skip anchor slots and kick/snare positions
        anchor_set = set(self._config.anchor_slots)
        if slot in anchor_set or obs.strength < self._config.mirror_min_strength:
            return

        # Record strength for this slot
        current = self._current_bar_hits.get(slot, 0.0)
        self._current_bar_hits[slot] = max(current, obs.strength)

        # Keep a rolling window of all observations for percentile
        self._observations.append(obs)
        if len(self._observations) > self._config.mirror_max_observations:
            self._observations.popleft()

    def finish_bar(self, controller: LiveController) -> int | None:
        """Called when a bar completes.

        Returns the slot to mirror (if eligible), or None.
        Also updates the controller's mirror state directly.
        """
        cfg = self._config

        completed_bar_hits = dict(self._current_bar_hits)
        self._current_bar_hits.clear()

        # Mirror already active: check for expiration
        if self._active_slot is not None:
            if self._active_slot in completed_bar_hits:
                self._unsupported_bars = 0
            else:
                self._unsupported_bars += 1
                if self._unsupported_bars >= cfg.mirror_expire_unsupported_bars:
                    controller.clear_mirror()
                    self._active_slot = None
                    self._previous_bar_hits = completed_bar_hits
                    return None
            self._previous_bar_hits = completed_bar_hits
            return self._active_slot

        # Mirror not yet active: check eligibility
        if controller._playing_bars_completed < cfg.mirror_min_stable_bars:  # type: ignore[attr]
            self._previous_bar_hits = completed_bar_hits
            return None

        # Need minimum sample count for percentile
        if len(self._observations) < cfg.mirror_min_sample_count:
            self._previous_bar_hits = completed_bar_hits
            return None

        result = self._evaluate_mirror(controller, completed_bar_hits)
        self._previous_bar_hits = completed_bar_hits
        return result

    def _evaluate_mirror(
        self,
        controller: LiveController,
        completed_bar_hits: dict[int, float],
    ) -> int | None:
        """Evaluate whether any slot qualifies for mirror activation."""
        cfg = self._config

        # For V1 simplicity: use the percentile threshold on all observations
        strengths = [o.strength for o in self._observations]
        if not strengths:
            return None
        strengths.sort()
        percentile_index = int(
            len(strengths) * (cfg.mirror_strength_percentile / 100.0)
        )
        threshold = strengths[min(percentile_index, len(strengths) - 1)]

        # Require the same strong, non-anchor slot in both consecutive bars.
        candidates: list[tuple[float, int]] = []
        for slot in self._previous_bar_hits.keys() & completed_bar_hits.keys():
            repeated_strength = min(
                self._previous_bar_hits[slot], completed_bar_hits[slot]
            )
            if repeated_strength >= threshold and slot not in set(cfg.anchor_slots):
                candidates.append((repeated_strength, slot))

        if candidates:
            # Strongest repeated slot wins; lower slot is deterministic on ties.
            _, slot = max(candidates, key=lambda candidate: (candidate[0], -candidate[1]))
            controller.set_mirror(slot)
            self._active_slot = slot
            self._unsupported_bars = 0
            return slot

        return None

    def reset(self) -> None:
        """Clear all accumulated observations."""
        self._observations.clear()
        self._slot_hits.clear()
        self._current_bar_hits.clear()
        self._previous_bar_hits.clear()
        self._unsupported_bars = 0
        self._active_slot = None
