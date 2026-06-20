"""Regression tests for tracker-to-live-controller confidence adaptation."""

from __future__ import annotations

import pytest

from perception.bar import BarHypothesis, BarState
from perception.live_adapter import LiveBarAdapter, LivePulseAdapter
from perception.models import MusicalEvent
from perception.pulse import PulseTracker
from tests.fake_clock import FakeClock


def test_pulse_adapter_uses_absolute_tracker_confidence() -> None:
    tracker = PulseTracker()
    for index in range(24):
        tracker.process_event(
            MusicalEvent(
                time_seconds=100.0 + index * 0.45,
                strength=1.0,
                energy=0.2,
            )
        )
    raw_state = tracker.get_state()
    adapter = LivePulseAdapter(tracker, FakeClock(111.0).now)

    adapted = adapter.adapt()

    assert raw_state.confidence >= 0.4
    assert adapted.winning_confidence == pytest.approx(raw_state.confidence)
    assert adapted.winning_confidence > raw_state.hypotheses[0].confidence
    assert adapted.ambiguity_margin >= 0.15


def test_pulse_adapter_treats_double_time_as_an_alias_not_a_rival() -> None:
    tracker = PulseTracker()
    for index in range(24):
        tracker.process_event(
            MusicalEvent(
                time_seconds=100.0 + index * 0.7,
                strength=1.0,
                energy=0.2,
            )
        )

    adapted = LivePulseAdapter(tracker, FakeClock(117.0).now).adapt()

    assert adapted.winning_bpm is not None
    assert adapted.ambiguity_margin >= 0.15


class _BarTrackerStub:
    def __init__(self, state: BarState) -> None:
        self.state = state

    def get_state(self, current_time: float | None = None) -> BarState:
        del current_time
        return self.state


def test_bar_adapter_uses_absolute_confidence_but_preserves_phase_ambiguity() -> None:
    best_raw = BarHypothesis(
        bpm=132.0,
        beat_interval=60.0 / 132.0,
        downbeat_time=100.0,
        confidence=0.9,
        last_updated=108.0,
        supporting_events=20,
    )
    displayed_best = BarHypothesis(
        bpm=132.0,
        beat_interval=60.0 / 132.0,
        downbeat_time=100.0,
        confidence=0.5,
        last_updated=108.0,
        supporting_events=20,
    )
    displayed_runner = BarHypothesis(
        bpm=132.0,
        beat_interval=60.0 / 132.0,
        downbeat_time=100.0 + 2 * (60.0 / 132.0),
        confidence=0.5,
        last_updated=108.0,
        supporting_events=19,
    )
    state = BarState(
        hypotheses=[displayed_best, displayed_runner],
        best_hypothesis=best_raw,
        is_confident=True,
        confidence=0.9,
        timestamp=108.0,
    )
    adapter = LiveBarAdapter(_BarTrackerStub(state), FakeClock(108.0).now)  # type: ignore[arg-type]

    adapted = adapter.adapt()

    assert adapted.winning_confidence == pytest.approx(0.9)
    assert adapted.runner_up_confidence == pytest.approx(0.855)
    assert adapted.ambiguity_margin == pytest.approx(0.045)


def test_bar_adapter_selects_hypotheses_matching_the_pulse_tempo() -> None:
    half_time = BarHypothesis(
        bpm=66.0,
        beat_interval=60.0 / 66.0,
        downbeat_time=100.0,
        confidence=0.6,
        last_updated=108.0,
        supporting_events=20,
    )
    pulse_match = BarHypothesis(
        bpm=132.0,
        beat_interval=60.0 / 132.0,
        downbeat_time=101.0,
        confidence=0.4,
        last_updated=108.0,
        supporting_events=12,
    )
    state = BarState(
        hypotheses=[half_time, pulse_match],
        best_hypothesis=half_time,
        is_confident=True,
        confidence=0.9,
        timestamp=108.0,
    )
    adapter = LiveBarAdapter(_BarTrackerStub(state), FakeClock(108.0).now)  # type: ignore[arg-type]

    adapted = adapter.adapt(reference_bpm=132.0)

    assert adapted.winning_bpm == pytest.approx(132.0)
    assert adapted.winning_confidence == pytest.approx(0.6)
    assert adapted.downbeat_time == pytest.approx(101.0)
    assert adapted.is_confident is True
