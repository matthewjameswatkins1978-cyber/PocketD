"""Bar Tracker — Module 3: estimate likely bar position and downbeat location.

The Bar Tracker does NOT output a single locked bar position.

It maintains competing bar-phase hypotheses, each with confidence, similar to
how PulseTracker maintains competing tempo hypotheses.

Philosophy:
    "Where does the musical cycle begin?"

The tracker uses:
- Pulse hypotheses from Module 2 (BPM, beat interval)
- Event strength and energy (strong accents = likely downbeats)
- Repetition over bar-length cycles (consistency builds confidence)

It preserves uncertainty when evidence is ambiguous.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from perception.models import MusicalEvent
from perception.pulse import PulseHypothesis, PulseState

log = logging.getLogger(__name__)

# ── Configuration ───────────────────────────────────────────────────

DEFAULT_BEATS_PER_BAR: int = 4
DEFAULT_MAX_HYPOTHESES: int = 8
DEFAULT_PHASE_TOLERANCE: float = 0.20  # 20% of beat interval — tolerance for beat alignment
DEFAULT_CONFIDENCE_DECAY: float = 0.95  # per event miss
DEFAULT_MIN_PULSE_CONFIDENCE: float = 0.08  # pulse hypothesis must exceed this to spawn bars
DEFAULT_MERGE_PHASE_TOLERANCE: float = 0.15  # 15% of beat interval — merge nearby phases
DEFAULT_SILENCE_DECAY_SECONDS: float = 2.0  # start decaying after this many seconds


# ── Data Models ─────────────────────────────────────────────────────


@dataclass
class BarHypothesis:
    """A single competing interpretation of where the bar cycle begins.

    Parameters
    ----------
    bpm : float
        The pulse BPM this bar hypothesis is derived from.
    beat_interval : float
        Seconds per beat (60 / bpm).
    beats_per_bar : int
        Number of beats in a bar (default 4 for 4/4).
    downbeat_time : float
        Absolute time of a reference beat 1 (the phase anchor).
    confidence : float
        Confidence in [0, 1] that this bar phase is correct.
    last_updated : float
        Time of the last event that matched this hypothesis.
    supporting_events : int
        Total events that have supported this bar phase.
    accent_score : float
        Accumulated event strength/energy weighted toward downbeat position.
    regularity_score : float
        Consistency of support over time (rises with repeated bar cycles).
    """
    bpm: float
    beat_interval: float
    beats_per_bar: int = DEFAULT_BEATS_PER_BAR
    downbeat_time: float = 0.0
    confidence: float = 0.0
    last_updated: float = 0.0
    supporting_events: int = 0
    accent_score: float = 0.0
    regularity_score: float = 0.0

    @property
    def bar_duration(self) -> float:
        """Duration of one bar in seconds."""
        return self.beat_interval * self.beats_per_bar


@dataclass
class BarState:
    """Current belief state of the Bar Tracker.

    Parameters
    ----------
    hypotheses : list[BarHypothesis]
        Competing bar-phase hypotheses, sorted by confidence descending.
    best_hypothesis : BarHypothesis | None
        The highest-confidence hypothesis, if any.
    is_confident : bool
        Whether the system is confident enough in the bar position.
    estimated_bar_position : float | None
        Phase within the bar (0.0 to beats_per_bar) at the current time.
    estimated_beat_in_bar : int | None
        Which beat (0-based) we are currently on.
    confidence : float
        Confidence of the best hypothesis.
    timestamp : float
        Time of the last update.
    """
    hypotheses: list[BarHypothesis] = field(default_factory=list)
    best_hypothesis: BarHypothesis | None = None
    is_confident: bool = False
    estimated_bar_position: float | None = None
    estimated_beat_in_bar: int | None = None
    confidence: float = 0.0
    timestamp: float = 0.0


# ── BarTracker ──────────────────────────────────────────────────────


class BarTracker:
    """Maintain competing bar-position hypotheses from events and pulse state.

    Parameters
    ----------
    beats_per_bar : int
        Default number of beats per bar (default 4 for 4/4).
    max_hypotheses : int
        Maximum number of competing bar hypotheses (default 8).
    phase_tolerance : float
        Fractional tolerance for matching events to beat positions.
        Default 0.20 = 20% of a beat interval.
    confidence_decay : float
        Multiplier for confidence decay when events miss.
    silence_decay_seconds : float
        Seconds without events before confidence starts decaying.
    """

    def __init__(
        self,
        beats_per_bar: int = DEFAULT_BEATS_PER_BAR,
        max_hypotheses: int = DEFAULT_MAX_HYPOTHESES,
        phase_tolerance: float = DEFAULT_PHASE_TOLERANCE,
        confidence_decay: float = DEFAULT_CONFIDENCE_DECAY,
        silence_decay_seconds: float = DEFAULT_SILENCE_DECAY_SECONDS,
    ) -> None:
        if beats_per_bar < 1:
            raise ValueError("beats_per_bar must be at least 1")
        self._beats_per_bar = beats_per_bar
        self._max_hypotheses = max_hypotheses
        self._phase_tolerance = phase_tolerance
        self._confidence_decay = confidence_decay
        self._silence_decay_seconds = silence_decay_seconds

        self._hypotheses: list[BarHypothesis] = []
        self._last_event_time: float | None = None
        self._current_time: float = 0.0

        log.info(
            "BarTracker initialised — %d/%d, tolerance=%.0f%%",
            beats_per_bar, beats_per_bar, phase_tolerance * 100,
        )

    # ── Public API ─────────────────────────────────────────────────

    def update(self, event: MusicalEvent, pulse_state: PulseState) -> BarState:
        """Process a musical event with current pulse hypotheses.

        Parameters
        ----------
        event : MusicalEvent
            The detected musical event.
        pulse_state : PulseState
            Current pulse hypotheses from PulseTracker.

        Returns
        -------
        BarState
            Updated bar-position belief state.
        """
        self._current_time = event.time_seconds

        # Derive bar hypotheses from pulse hypotheses
        self._derive_hypotheses_from_pulse(pulse_state, event)

        # Score existing bar hypotheses against this event
        self._score_hypotheses(event)

        # Decay hypotheses not recently supported
        self._decay_hypotheses()

        # Prune weak hypotheses
        self._prune_hypotheses()

        # Merge nearby hypotheses
        self._merge_hypotheses()

        self._last_event_time = event.time_seconds

        return self._build_state()

    def get_state(self, current_time: float | None = None) -> BarState:
        """Return current bar state without processing an event.

        Parameters
        ----------
        current_time : float | None
            If provided, update the timestamp and estimate beat position
            at this time.
        """
        if current_time is not None:
            self._current_time = current_time
            self._decay_hypotheses()
            self._prune_hypotheses()
        return self._build_state()

    def reset(self) -> None:
        """Clear all hypotheses and state."""
        self._hypotheses.clear()
        self._last_event_time = None
        self._current_time = 0.0
        log.debug("BarTracker reset")

    # ── Internal: Hypothesis Derivation ────────────────────────────

    def _derive_hypotheses_from_pulse(
        self, pulse_state: PulseState, event: MusicalEvent,
    ) -> None:
        """Create new bar hypotheses from strong pulse hypotheses.

        For each pulse hypothesis with sufficient confidence, create
        a bar hypothesis if none exists for that BPM, aligned to this
        event as a potential downbeat reference.
        """
        for ph in pulse_state.hypotheses:
            if ph.confidence < DEFAULT_MIN_PULSE_CONFIDENCE:
                continue

            beat_interval = 60.0 / ph.bpm
            bar_duration = beat_interval * self._beats_per_bar

            # Check if we already have hypotheses for this BPM
            existing = [
                h for h in self._hypotheses
                if self._bpm_close(h.bpm, ph.bpm)
            ]

            if not existing:
                # Create a new bar hypothesis aligned to this event
                # as a candidate downbeat
                self._hypotheses.append(BarHypothesis(
                    bpm=round(ph.bpm, 1),
                    beat_interval=beat_interval,
                    beats_per_bar=self._beats_per_bar,
                    downbeat_time=round(event.time_seconds, 3),
                    confidence=0.05 + ph.confidence * 0.1,
                    last_updated=event.time_seconds,
                    supporting_events=1,
                    accent_score=self._event_accent_weight(event),
                ))

                # Also create a hypothesis at the opposite phase
                # (shifted by half a bar) to maintain ambiguity
                half_bar = bar_duration / 2.0
                self._hypotheses.append(BarHypothesis(
                    bpm=round(ph.bpm, 1),
                    beat_interval=beat_interval,
                    beats_per_bar=self._beats_per_bar,
                    downbeat_time=round(event.time_seconds + half_bar, 3),
                    confidence=0.03 + ph.confidence * 0.05,
                    last_updated=event.time_seconds,
                    supporting_events=1,
                    accent_score=self._event_accent_weight(event) * 0.5,
                ))

    # ── Internal: Scoring ──────────────────────────────────────────

    def _score_hypotheses(self, event: MusicalEvent) -> None:
        """Score all bar hypotheses against the current event.

        An event supports a bar hypothesis if it lands near a beat
        position within that bar. Events near beat 0 (downbeat) get
        higher weight.
        """
        for hyp in self._hypotheses:
            # Calculate phase within this bar's cycle
            bar_duration = hyp.bar_duration
            if bar_duration <= 0:
                continue

            phase = (event.time_seconds - hyp.downbeat_time) % bar_duration
            beat_position = phase / max(hyp.beat_interval, 0.001)

            # Check if event is near a beat position (0, 1, 2, 3 for 4/4)
            nearest_beat = round(beat_position)
            nearest_beat = nearest_beat % hyp.beats_per_bar  # wrap within bar
            distance = abs(beat_position - nearest_beat)

            # Normalise distance by looking at fractional offset within beat
            # beat_position could be like 3.9, nearest_beat 4 which wraps to 0
            # Better: check all integer positions
            best_distance = float("inf")
            best_beat = -1
            for b in range(-1, hyp.beats_per_bar + 2):
                # Check b and b+beats_per_bar to handle wraparound
                test_dist = abs(beat_position - b)
                if test_dist < best_distance:
                    best_distance = test_dist
                    best_beat = b % hyp.beats_per_bar

            in_tolerance = best_distance <= self._phase_tolerance

            if in_tolerance:
                # Event aligns with a beat in this bar hypothesis
                weight = self._event_accent_weight(event)
                base_boost = 0.04 + weight * 0.04

                # Extra boost for downbeat alignment
                if best_beat == 0:
                    base_boost *= 1.5
                    hyp.accent_score += weight * 1.5

                hyp.confidence = min(1.0, hyp.confidence + base_boost)
                hyp.supporting_events += 1
                hyp.last_updated = event.time_seconds

                # Regularity: check if this bar cycle is repeating
                # How many bar cycles have passed since downbeat_time?
                cycles = (event.time_seconds - hyp.downbeat_time) / bar_duration
                if cycles >= 1.0:
                    # At least one full bar has passed — evidence of regularity
                    hyp.regularity_score = min(1.0, hyp.regularity_score + 0.03)
            else:
                # Event doesn't align — slight penalty
                hyp.confidence *= 0.99

            # Nudge downbeat_time toward alignment if very close
            if best_distance <= self._phase_tolerance * 0.5 and best_distance > 0.001:
                # Slightly adjust phase to track drift
                correction = (nearest_beat - beat_position) * hyp.beat_interval
                # Limit correction per event
                correction = max(-0.01, min(0.01, correction * 0.05))
                hyp.downbeat_time += correction

    def _decay_hypotheses(self) -> None:
        """Decay confidence of hypotheses not recently supported."""
        if self._current_time <= 0:
            return

        for hyp in self._hypotheses:
            time_since = self._current_time - hyp.last_updated
            if time_since > self._silence_decay_seconds:
                decays = int((time_since - self._silence_decay_seconds) / 1.0) + 1
                for _ in range(min(decays, 20)):
                    hyp.confidence *= self._confidence_decay
                hyp.regularity_score = max(0.0, hyp.regularity_score - 0.02)

    def _prune_hypotheses(self) -> None:
        """Remove low-confidence hypotheses, keep only top N."""
        self._hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        if len(self._hypotheses) > 1:
            self._hypotheses = [
                h for h in self._hypotheses
                if h.confidence > 0.005
            ]

        if len(self._hypotheses) > self._max_hypotheses:
            self._hypotheses = self._hypotheses[:self._max_hypotheses]

    def _merge_hypotheses(self) -> None:
        """Merge hypotheses with similar BPM and downbeat phase.

        Two hypotheses are mergeable if:
        - Their BPM values are close
        - Their downbeat times differ by less than half a beat interval
          (modulo the bar duration)
        """
        if len(self._hypotheses) < 2:
            return

        merged: list[BarHypothesis] = []
        used = [False] * len(self._hypotheses)

        for i, h1 in enumerate(self._hypotheses):
            if used[i]:
                continue
            combined = BarHypothesis(
                bpm=h1.bpm,
                beat_interval=h1.beat_interval,
                beats_per_bar=h1.beats_per_bar,
                downbeat_time=h1.downbeat_time,
                confidence=h1.confidence,
                last_updated=h1.last_updated,
                supporting_events=h1.supporting_events,
                accent_score=h1.accent_score,
                regularity_score=h1.regularity_score,
            )
            active_count = 1

            for j, h2 in enumerate(self._hypotheses):
                if j <= i or used[j]:
                    continue
                if not self._bpm_close(h1.bpm, h2.bpm):
                    continue

                # Check if phases are close (within half a beat)
                bar_dur = h1.bar_duration
                phase_diff = abs(
                    (h1.downbeat_time - h2.downbeat_time) % bar_dur
                )
                phase_diff = min(phase_diff, bar_dur - phase_diff)

                merge_max = h1.beat_interval * DEFAULT_MERGE_PHASE_TOLERANCE
                if phase_diff <= merge_max:
                    # Merge them: weighted average
                    w1 = combined.confidence
                    w2 = h2.confidence
                    total_w = w1 + w2 + 0.001

                    combined.downbeat_time = (
                        combined.downbeat_time * w1 + h2.downbeat_time * w2
                    ) / total_w
                    combined.confidence = min(
                        1.0, w1 + w2 - w1 * w2 * 0.5
                    )
                    combined.supporting_events += h2.supporting_events
                    combined.accent_score += h2.accent_score
                    combined.regularity_score = max(
                        combined.regularity_score, h2.regularity_score,
                    )
                    combined.last_updated = max(
                        combined.last_updated, h2.last_updated,
                    )
                    active_count += 1
                    used[j] = True

            merged.append(combined)
            used[i] = True

        self._hypotheses = merged

    # ── Internal: State Building ────────────────────────────────────

    def _build_state(self) -> BarState:
        """Build the current BarState from hypotheses."""
        self._hypotheses.sort(key=lambda h: h.confidence, reverse=True)

        if not self._hypotheses:
            return BarState(timestamp=self._current_time)

        best = self._hypotheses[0]
        is_confident = best.confidence > 0.4 and best.supporting_events >= 4

        # Normalise confidences for display
        total_conf = sum(h.confidence for h in self._hypotheses)
        if total_conf > 0:
            displayed = [
                BarHypothesis(
                    bpm=h.bpm,
                    beat_interval=h.beat_interval,
                    beats_per_bar=h.beats_per_bar,
                    downbeat_time=h.downbeat_time,
                    confidence=round(h.confidence / total_conf, 4),
                    last_updated=h.last_updated,
                    supporting_events=h.supporting_events,
                    accent_score=round(h.accent_score, 3),
                    regularity_score=round(h.regularity_score, 3),
                )
                for h in self._hypotheses[:self._max_hypotheses]
            ]
        else:
            displayed = list(self._hypotheses)

        # Estimate beat-in-bar
        beat_in_bar: int | None = None
        bar_position: float | None = None

        if best.bar_duration > 0:
            phase = (self._current_time - best.downbeat_time) % best.bar_duration
            bar_position = phase / max(best.beat_interval, 0.001)
            bar_position = bar_position % best.beats_per_bar
            beat_in_bar = int(bar_position) % best.beats_per_bar

        return BarState(
            hypotheses=displayed,
            best_hypothesis=best,
            is_confident=is_confident,
            estimated_bar_position=round(bar_position, 2) if bar_position is not None else None,
            estimated_beat_in_bar=beat_in_bar,
            confidence=round(best.confidence, 4),
            timestamp=self._current_time,
        )

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _event_accent_weight(event: MusicalEvent) -> float:
        """Compute accent weight from event properties."""
        return event.strength * 0.6 + event.energy * 0.3 + event.density * 0.1 + 0.1

    @staticmethod
    def _bpm_close(a: float, b: float, ratio: float = 1.06) -> bool:
        """Check if two BPM values are within ~6% of each other."""
        if a <= 0 or b <= 0:
            return False
        r = max(a, b) / min(a, b)
        return r <= ratio