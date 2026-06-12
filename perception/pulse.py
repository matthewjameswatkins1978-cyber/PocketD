"""Pulse Tracker — Module 2: maintain competing tempo/pulse hypotheses from musical events.

The Pulse Tracker does NOT output a single locked BPM.

It maintains multiple possible interpretations simultaneously, each with its own
confidence level. Confidence rises when events support a hypothesis and decays
when they do not.

This mimics how a human drummer listens for the pulse:
- uncertain at first
- forming multiple possible interpretations
- increasing confidence when evidence repeats
- lowering confidence when evidence weakens
- avoiding premature certainty
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from perception.models import MusicalEvent

log = logging.getLogger(__name__)


@dataclass
class PulseHypothesis:
    """A single competing BPM hypothesis with confidence and stability tracking.

    Parameters
    ----------
    bpm : float
        The beats-per-minute value of this hypothesis.
    confidence : float
        Current confidence in [0, 1] that this hypothesis reflects the true pulse.
    matches : int
        How many events have supported this hypothesis.
    misses : int
        How many events have not supported this hypothesis.
    last_event_time : float
        The time of the last event that supported this hypothesis.
    stability : float
        A stability measure in [0, 1]. Higher means the hypothesis has been
        consistently supported over time.
    """
    bpm: float
    confidence: float = 0.0
    matches: int = 0
    misses: int = 0
    last_event_time: float = 0.0
    stability: float = 0.0


@dataclass
class PulseState:
    """The current belief state of the Pulse Tracker.

    Parameters
    ----------
    hypotheses : list[PulseHypothesis]
        Competing BPM hypotheses, sorted by confidence descending.
    best_bpm : float | None
        The BPM of the highest-confidence hypothesis, if any.
    confidence : float
        The confidence of the best hypothesis.
    stability : str
        A human-readable stability label: "unknown", "rising", "stable", "falling".
    """
    hypotheses: list[PulseHypothesis] = field(default_factory=list)
    best_bpm: float | None = None
    confidence: float = 0.0
    stability: str = "unknown"


# Default configuration
DEFAULT_MIN_BPM: float = 40.0
DEFAULT_MAX_BPM: float = 250.0
DEFAULT_TOLERANCE: float = 0.12  # 12% tolerance for matching beat intervals
DEFAULT_MAX_HYPOTHESES: int = 8
DEFAULT_CONFIDENCE_DECAY: float = 0.96  # multiplier per event miss
DEFAULT_HISTORY_SIZE: int = 16  # number of recent events to keep
DEFAULT_MERGE_TOLERANCE: float = 0.05  # 5% — merge candidates within this range


class PulseTracker:
    """Maintain competing pulse hypotheses from a stream of musical events.

    Parameters
    ----------
    min_bpm : float
        Minimum allowable BPM (default 40.0).
    max_bpm : float
        Maximum allowable BPM (default 250.0).
    tolerance : float
        Fractional tolerance for matching event intervals to beat predictions
        (default 0.12 = 12%).
    max_hypotheses : int
        Maximum number of competing hypotheses to maintain (default 8).
    confidence_decay : float
        Multiplier applied to hypothesis confidence when an event misses it
        (default 0.96).
    merge_tolerance : float
        Fractional tolerance for merging nearby BPM candidates into existing
        hypotheses (default 0.05 = 5%).
    history_size : int
        Number of recent events to retain for interval analysis (default 16).
    """

    def __init__(
        self,
        min_bpm: float = DEFAULT_MIN_BPM,
        max_bpm: float = DEFAULT_MAX_BPM,
        tolerance: float = DEFAULT_TOLERANCE,
        max_hypotheses: int = DEFAULT_MAX_HYPOTHESES,
        confidence_decay: float = DEFAULT_CONFIDENCE_DECAY,
        merge_tolerance: float = DEFAULT_MERGE_TOLERANCE,
        history_size: int = DEFAULT_HISTORY_SIZE,
    ) -> None:
        self._min_bpm = min_bpm
        self._max_bpm = max_bpm
        self._tolerance = tolerance
        self._max_hypotheses = max_hypotheses
        self._confidence_decay = confidence_decay
        self._merge_tolerance = merge_tolerance
        self._history_size = history_size

        self._events: list[MusicalEvent] = []
        self._hypotheses: list[PulseHypothesis] = []
        self._previous_state: PulseState = PulseState()

        log.info(
            "PulseTracker initialised — min=%s BPM, max=%s BPM, "
            "tolerance=%.0f%%, merge=%.0f%%",
            min_bpm, max_bpm, tolerance * 100, merge_tolerance * 100,
        )

    # ── Public API ─────────────────────────────────────────────────

    def process_event(self, event: MusicalEvent) -> PulseState:
        """Process a single musical event and update pulse hypotheses.

        Parameters
        ----------
        event : MusicalEvent
            The detected musical event (time, strength, energy, density).

        Returns
        -------
        PulseState
            The current state of all pulse hypotheses after processing.
        """
        self._events.append(event)

        # Keep event history bounded
        if len(self._events) > self._history_size:
            self._events = self._events[-self._history_size:]

        # Generate BPM candidates from intervals with previous events
        candidates = self._generate_candidates(event)

        # Merge candidates into existing hypotheses
        self._merge_candidates(candidates, event)

        # Score all hypotheses against the new event
        self._score_hypotheses(event)

        # Decay hypotheses that this event did not support
        self._decay_hypotheses(event)

        # Keep only top hypotheses
        self._prune_hypotheses()

        # Build and return the current state
        state = self._build_state()
        self._previous_state = state
        return state

    def get_state(self) -> PulseState:
        """Return the current pulse state without processing a new event."""
        return self._build_state()

    def advance_time(self, current_time: float) -> PulseState:
        """Advance time without a new event, decaying all hypotheses.

        Useful for simulating silence/gaps in the audio stream.

        Parameters
        ----------
        current_time : float
            The current time to advance to.

        Returns
        -------
        PulseState
            Updated pulse state after decay.
        """
        for hyp in self._hypotheses:
            # Decay confidence based on time since last event
            time_since = current_time - hyp.last_event_time
            if time_since > 1.0:
                # Apply decay multiple times for longer gaps
                decays = int(time_since / 0.5)
                for _ in range(min(decays, 20)):
                    hyp.confidence *= self._confidence_decay
                hyp.confidence = max(0.0, hyp.confidence)

        self._prune_hypotheses()
        return self._build_state()

    def reset(self) -> None:
        """Clear all events and hypotheses."""
        self._events.clear()
        self._hypotheses.clear()
        self._previous_state = PulseState()
        log.debug("PulseTracker reset")

    # ── Internal: Candidate Generation ─────────────────────────────

    def _generate_candidates(self, event: MusicalEvent) -> list[float]:
        """Generate BPM candidates from intervals with recent events."""
        candidates: list[float] = []

        if len(self._events) < 2:
            return candidates

        # Compare against recent events (skip self)
        for prev in self._events[:-1]:
            interval = event.time_seconds - prev.time_seconds
            if interval <= 0.01:  # skip near-zero intervals (simultaneous events)
                continue

            bpm = 60.0 / interval

            # Only keep candidates in range
            if self._min_bpm <= bpm <= self._max_bpm:
                candidates.append(bpm)

            # Add musically related multiples and divisions
            for factor in [0.5, 2.0]:
                related = bpm * factor
                if self._min_bpm <= related <= self._max_bpm:
                    candidates.append(related)

        return candidates

    # ── Internal: Merge Candidates ─────────────────────────────────

    def _merge_candidates(
        self,
        candidates: list[float],
        event: MusicalEvent,
    ) -> None:
        """Merge BPM candidates into existing hypotheses or create new ones."""
        if not candidates:
            return

        # Weight each candidate by (1 / interval_count) strength/energy
        for bpm in candidates:
            self._merge_or_create(bpm, event)

    def _merge_or_create(self, bpm: float, event: MusicalEvent) -> None:
        """Merge a candidate into a close existing hypothesis or create a new one."""
        # Check if this BPM is close to an existing hypothesis
        for hyp in self._hypotheses:
            if self._bpm_close(bpm, hyp.bpm):
                # Merge: nudge toward new candidate, weighted by event strength
                weight = self._event_weight(event)
                hyp.bpm = hyp.bpm * (1.0 - weight * 0.1) + bpm * (weight * 0.1)
                return

        # No close hypothesis found — create new one
        weight = self._event_weight(event)
        confidence = min(0.5, 0.05 + weight * 0.1)

        self._hypotheses.append(PulseHypothesis(
            bpm=round(bpm, 1),
            confidence=confidence,
            matches=1,
            last_event_time=event.time_seconds,
            stability=0.0,
        ))

    @staticmethod
    def _event_weight(event: MusicalEvent) -> float:
        """Compute a weight from event strength and energy for confidence adjustments."""
        return 0.5 + (event.strength * 0.4) + (event.energy * 0.1)

    @staticmethod
    def _bpm_close(bpm_a: float, bpm_b: float, tolerance: float | None = None) -> bool:
        """Check if two BPM values are close within a fractional tolerance."""
        if bpm_a <= 0 or bpm_b <= 0:
            return False
        ratio = max(bpm_a, bpm_b) / min(bpm_a, bpm_b)
        return ratio <= (1.0 + (tolerance or DEFAULT_MERGE_TOLERANCE))

    # ── Internal: Scoring ──────────────────────────────────────────

    def _score_hypotheses(self, event: MusicalEvent) -> None:
        """Score all hypotheses against the new event.

        A hypothesis is supported if the event lands close to an expected
        beat multiple (1x, 2x, 3x, 4x of the beat interval).
        """
        for hyp in self._hypotheses:
            if hyp.last_event_time <= 0:
                hyp.last_event_time = event.time_seconds
                continue

            beat_interval = 60.0 / hyp.bpm
            time_since = event.time_seconds - hyp.last_event_time

            # Check event against 1x, 2x, 3x, 4x beat multiples
            supported = False
            for multiple in range(1, 5):
                expected = multiple * beat_interval
                if abs(time_since - expected) / max(expected, 0.01) <= self._tolerance:
                    supported = True
                    break

            if supported:
                # Increase confidence
                weight = self._event_weight(event)
                confidence_boost = 0.08 + weight * 0.06
                hyp.confidence = min(1.0, hyp.confidence + confidence_boost)
                hyp.matches += 1
                hyp.last_event_time = event.time_seconds

                # Increase stability
                hyp.stability = min(1.0, hyp.stability + 0.05)
            else:
                # This event doesn't support this hypothesis
                hyp.misses += 1
                # Slight decay for a miss (but don't penalise too hard for non-beat events)
                hyp.confidence *= 0.98

    def _decay_hypotheses(self, event: MusicalEvent) -> None:
        """Apply decay to hypotheses not supported by the current event.

        A decay is applied per hypothesis. Hypotheses that *were* supported
        by this event already had their confidence boosted, so this is a
        gentle decay for those that were not supported.
        """
        for hyp in self._hypotheses:
            # Only decay if this event could have reasonably supported it
            # but didn't
            time_since = event.time_seconds - hyp.last_event_time
            if time_since > 2.0 * (60.0 / max(hyp.bpm, 1.0)):
                # Event occurred well past when this hypothesis expected one
                hyp.confidence *= self._confidence_decay
                hyp.stability = max(0.0, hyp.stability - 0.02)

    def _prune_hypotheses(self) -> None:
        """Remove low-confidence hypotheses and keep only the top ones."""
        # Sort by confidence descending
        self._hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        # Remove very low confidence hypotheses (but keep at least one)
        if len(self._hypotheses) > 1:
            self._hypotheses = [
                h for h in self._hypotheses
                if h.confidence > 0.01
            ]

        # Keep only top N
        if len(self._hypotheses) > self._max_hypotheses:
            self._hypotheses = self._hypotheses[:self._max_hypotheses]

    # ── Internal: State Building ───────────────────────────────────

    def _build_state(self) -> PulseState:
        """Build the current PulseState from hypotheses."""
        self._hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        if not self._hypotheses:
            return PulseState()

        best = self._hypotheses[0]
        best_bpm = round(best.bpm, 1)
        confidence = round(best.confidence, 4)

        # Determine stability label
        if best.stability < 0.2:
            stability_label = "unknown"
        elif best.stability < 0.5:
            stability_label = "rising"
        elif best.stability < 0.8:
            stability_label = "stable"
        else:
            stability_label = "locked"

        # Normalise confidences for display — proportional to best
        total_conf = sum(h.confidence for h in self._hypotheses)
        if total_conf > 0:
            displayed_hypotheses = [
                PulseHypothesis(
                    bpm=round(h.bpm, 1),
                    confidence=round(h.confidence / total_conf, 4),
                    matches=h.matches,
                    misses=h.misses,
                    last_event_time=h.last_event_time,
                    stability=round(h.stability, 4),
                )
                for h in self._hypotheses[:self._max_hypotheses]
            ]
        else:
            displayed_hypotheses = list(self._hypotheses)

        return PulseState(
            hypotheses=displayed_hypotheses,
            best_bpm=best_bpm,
            confidence=confidence,
            stability=stability_label,
        )