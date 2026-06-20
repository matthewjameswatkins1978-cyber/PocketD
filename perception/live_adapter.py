"""Live adapter — wraps existing PulseState/BarState into the monotonic clock
domain required by the Bunny V1 controller.

This module does **not** rebuild or replace the trackers.  It reads
their current public state and produces immutable ``PulseAdapterState``
and ``BarAdapterState`` records with injectable clock timestamps.
"""

from __future__ import annotations

import time as _time
from typing import Callable

from drummer.live_models import (
    BarAdapterState,
    MonotonicClock,
    PulseAdapterState,
)
from perception.bar import BarState, BarTracker
from perception.pulse import PulseState, PulseTracker


_SAME_TEMPO_RATIO = 1.12
_BAR_PULSE_RATIO = 1.12


def _same_tempo_family(first: float, second: float) -> bool:
    if first <= 0.0 or second <= 0.0:
        return False
    ratio = max(first, second) / min(first, second)
    # Half/double-time readings are octave aliases of the same physical pulse,
    # not independent evidence that should deadlock entry forever.
    return ratio <= _SAME_TEMPO_RATIO or 2.0 / _SAME_TEMPO_RATIO <= ratio <= 2.0 * _SAME_TEMPO_RATIO


def _matches_pulse_family(first: float, second: float) -> bool:
    if first <= 0.0 or second <= 0.0:
        return False
    return max(first, second) / min(first, second) <= _BAR_PULSE_RATIO


def _relative_runner_strength(
    best_confidence: float,
    best_support: int,
    runner_confidence: float,
    runner_support: int,
) -> float:
    """Return runner strength relative to the winner on a 0..1 scale.

    Tracker hypothesis confidences exposed in state lists are normalised for
    display.  Combining their ratio with supporting-event ratio preserves
    meaningful ambiguity without mistaking those display weights for the
    tracker's absolute confidence.
    """
    if best_confidence <= 0.0:
        return 0.0
    confidence_ratio = min(1.0, runner_confidence / best_confidence)
    support_ratio = min(1.0, runner_support / max(best_support, 1))
    return confidence_ratio * support_ratio


def _default_clock() -> float:
    """Default monotonic clock: ``time.perf_counter``."""
    return _time.perf_counter()


class LivePulseAdapter:
    """Wraps a ``PulseTracker`` and converts its output to the monotonic domain.

    Parameters
    ----------
    tracker : PulseTracker
        The existing pulse tracker instance.
    clock : MonotonicClock | None
        Inject a callable returning monotonic seconds.  Defaults to
        ``time.perf_counter``.
    """

    def __init__(
        self,
        tracker: PulseTracker,
        clock: MonotonicClock | None = None,
    ) -> None:
        self._tracker = tracker
        self._clock = clock if clock is not None else _default_clock

    def adapt(self) -> PulseAdapterState:
        """Read the tracker's current ``PulseState`` and return an adapted record.

        All timestamps are captured from the injected clock.  Audio-relative
        times from the tracker are left in their native domain; the
        monotonic ``observed_at`` / ``computed_at`` are the *adaptation*
        wall timestamps, not the audio-stream times.
        """
        now = self._clock()
        ps: PulseState = self._tracker.get_state()
        hyps = ps.hypotheses

        if not hyps:
            return PulseAdapterState(
                observed_at=now,
                computed_at=now,
                winning_bpm=None,
                winning_confidence=0.0,
                runner_up_bpm=None,
                runner_up_confidence=0.0,
                ambiguity_margin=1.0,
                hypothesis_count=0,
                support_count=0,
                evidence_age=0.0,
                predicted_next_beat=None,
                beat_period=None,
                stability=ps.stability,
            )

        best = hyps[0]
        runner = next(
            (
                hypothesis
                for hypothesis in hyps[1:]
                if not _same_tempo_family(best.bpm, hypothesis.bpm)
            ),
            None,
        )

        winning_bpm = best.bpm
        winning_conf = ps.confidence
        runner_bpm = runner.bpm if runner else None
        runner_strength = (
            _relative_runner_strength(
                best.confidence,
                best.matches,
                runner.confidence,
                runner.matches,
            )
            if runner
            else 0.0
        )
        runner_conf = winning_conf * runner_strength

        margin = winning_conf - runner_conf

        beat_period: float | None = 60.0 / winning_bpm if winning_bpm > 0 else None

        # Best-effort next beat prediction in monotonic domain
        predicted_next_beat: float | None = None
        if beat_period is not None and best.last_event_time > 0:
            # Audio time of last evidence → how many beats since then
            beats_elapsed = (now - best.last_event_time) / beat_period
            beats_ahead = 1.0 - (beats_elapsed % 1.0)
            predicted_next_beat = now + beats_ahead * beat_period

        evidence_age = now - best.last_event_time if best.last_event_time > 0 else float("inf")

        return PulseAdapterState(
            observed_at=now,
            computed_at=now,
            winning_bpm=round(winning_bpm, 1),
            winning_confidence=round(winning_conf, 4),
            runner_up_bpm=round(runner_bpm, 1) if runner_bpm else None,
            runner_up_confidence=round(runner_conf, 4),
            ambiguity_margin=round(margin, 4),
            hypothesis_count=len(hyps),
            support_count=best.matches,
            evidence_age=evidence_age,
            predicted_next_beat=predicted_next_beat,
            beat_period=round(beat_period, 6) if beat_period else None,
            stability=ps.stability,
        )


class LiveBarAdapter:
    """Wraps a ``BarTracker`` and converts its output to the monotonic domain.

    Parameters
    ----------
    tracker : BarTracker
        The existing bar tracker instance.
    clock : MonotonicClock | None
        Inject a callable returning monotonic seconds.
    """

    def __init__(
        self,
        tracker: BarTracker,
        clock: MonotonicClock | None = None,
    ) -> None:
        self._tracker = tracker
        self._clock = clock if clock is not None else _default_clock

    def adapt(self, reference_bpm: float | None = None) -> BarAdapterState:
        """Read the tracker's current ``BarState`` and return an adapted record.

        The bar tracker's ``timestamp`` field is an audio-stream time.
        We capture monotonic ``observed_at`` / ``computed_at`` separately.
        The ``downbeat_time`` is converted from audio time by shifting
        into the monotonic domain using the difference between the
        tracker timestamp and the current monotonic time.  This is an
        approximation; for precise alignment the caller should use the
        listener adapter to bridge the domains.

        Returns
        -------
        BarAdapterState
        """
        now = self._clock()
        bs: BarState = self._tracker.get_state(current_time=now)

        if not bs.hypotheses:
            return BarAdapterState(
                observed_at=now,
                computed_at=now,
                winning_bpm=None,
                winning_confidence=0.0,
                runner_up_confidence=0.0,
                ambiguity_margin=1.0,
                hypothesis_count=0,
                support_count=0,
                estimated_beat_in_bar=None,
                bar_position=None,
                downbeat_time=None,
                bar_duration=None,
                evidence_age=0.0,
                is_confident=False,
            )

        hyps = bs.hypotheses
        candidates = (
            [
                hypothesis
                for hypothesis in hyps
                if reference_bpm is not None
                and _matches_pulse_family(reference_bpm, hypothesis.bpm)
            ]
            if reference_bpm is not None
            else list(hyps)
        )
        if not candidates:
            return BarAdapterState(
                observed_at=now,
                computed_at=now,
                winning_bpm=None,
                winning_confidence=0.0,
                runner_up_confidence=0.0,
                ambiguity_margin=0.0,
                hypothesis_count=len(hyps),
                support_count=0,
                estimated_beat_in_bar=None,
                bar_position=None,
                downbeat_time=None,
                bar_duration=None,
                evidence_age=float("inf"),
                is_confident=False,
            )

        global_displayed_best = hyps[0]
        displayed_best = candidates[0]
        best = (
            bs.best_hypothesis
            if reference_bpm is None and bs.best_hypothesis is not None
            else displayed_best
        )
        runner = candidates[1] if len(candidates) > 1 else None

        relative_to_global = (
            min(1.0, displayed_best.confidence / global_displayed_best.confidence)
            if global_displayed_best.confidence > 0.0
            else 0.0
        )
        winning_conf = bs.confidence * relative_to_global
        runner_strength = (
            _relative_runner_strength(
                displayed_best.confidence,
                displayed_best.supporting_events,
                runner.confidence,
                runner.supporting_events,
            )
            if runner
            else 0.0
        )
        runner_conf = winning_conf * runner_strength
        margin = winning_conf - runner_conf

        evidence_age = now - best.last_updated if best.last_updated > 0 else float("inf")

        # Convert downbeat_time from audio domain to monotonic domain.
        # The bar tracker stores its last updated audio time in
        # BarState.timestamp and best.last_updated.  We cannot directly
        # translate audio time → monotonic without the listener's
        # translation, so we store the *relative* offset and let the
        # caller provide the domain bridge.
        #
        # For the approximate case, if we assume audio time ≈ monotonic
        # time (same process, no scheduling delay), we can use the raw
        # downbeat_time directly.

        winning_bpm = best.bpm
        beat_interval = best.beat_interval
        bar_duration = best.bar_duration if best.bar_duration > 0 else None

        # Best-effort: use the downbeat_time directly as monotonic.
        # A proper domain bridge would be needed for hardware audio.
        downbeat_time = best.downbeat_time if best.downbeat_time > 0 else None

        # estimated_beat_in_bar and bar_position are from BarState already
        beat_in_bar = bs.estimated_beat_in_bar
        bar_pos = bs.estimated_bar_position

        return BarAdapterState(
            observed_at=now,
            computed_at=now,
            winning_bpm=round(winning_bpm, 1),
            winning_confidence=round(winning_conf, 4),
            runner_up_confidence=round(runner_conf, 4),
            ambiguity_margin=round(margin, 4),
            hypothesis_count=len(hyps),
            support_count=best.supporting_events,
            estimated_beat_in_bar=beat_in_bar,
            bar_position=round(bar_pos, 4) if bar_pos is not None else None,
            downbeat_time=downbeat_time,
            bar_duration=round(bar_duration, 6) if bar_duration else None,
            evidence_age=evidence_age,
            is_confident=(winning_conf > 0.4 and best.supporting_events >= 4),
        )
