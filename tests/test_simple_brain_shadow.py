"""Tests for Simple Brain shadow-mode helpers.

Verifies that the shadow runner produces valid trace rows
over real FeatureMonitor snapshots without crashing, and
that SimpleBrain behaves reasonably on at least one real scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# -- Reuse existing timeline builder --
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_continuous_jam_midi import (  # noqa: E402
    _SECTION_PHASE,
    build_scenario_timeline,
)

from drummer.simple_brain import (  # noqa: E402
    LOCK_SNAPSHOTS,
    LOCK_THRESHOLD,
    BrainAction,
    SimpleBrain,
)
from perception.features import FeatureMonitor  # noqa: E402
from perception.models import MusicalEvent  # noqa: E402


# ---------------------------------------------------------------------------
# Helper — run one scenario bar by bar
# ---------------------------------------------------------------------------


def _run_scenario(
    scenario: str,
    variation: str,
    bpm: float = 120.0,
    bars: int = 16,
) -> tuple[list[dict], SimpleBrain]:
    """Build a timeline, feed through FeatureMonitor+SimpleBrain.

    Returns trace rows and the final brain state.
    """
    timeline_bars = build_scenario_timeline(
        scenario=scenario,
        playtest_variation=variation,
        bpm=bpm,
        bars=bars,
    )

    monitor = FeatureMonitor()
    brain = SimpleBrain()

    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = 4.0 * seconds_per_beat

    rows: list[dict] = []
    current_time: float = 0.0

    for bar_idx, events in enumerate(timeline_bars):
        bar_end = current_time + seconds_per_bar
        for evt in events:
            monitor.feed(evt)

        n_events = len(events)
        if n_events == 0:
            section = "silent"
        elif n_events <= 2:
            section = "sparse"
        elif n_events <= 6:
            section = "medium"
        else:
            section = "dense"

        phase = _SECTION_PHASE.get(section.upper(), None)
        snapshot = monitor.snapshot(bar_end, phase_alignment=phase)
        decision = brain.decide(snapshot)

        rows.append(
            {
                "bar": bar_idx,
                "section": section,
                "action": decision.action.value,
                "beat": decision.beat_name,
                "confidence": decision.confidence,
                "input_density": snapshot.input_density,
                "player_certainty": snapshot.player_certainty,
                "stability": snapshot.repetition_stability,
                "change_score": snapshot.change_score,
                "silence": snapshot.silence_duration,
                "reason": decision.reason,
                "scores": decision.scores,
            }
        )
        current_time = bar_end

    return rows, brain


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_produces_trace_rows_for_stable_enter() -> None:
    """Shadow runner produces trace rows for enter/stable_input."""
    rows, brain = _run_scenario("enter", "stable_input")
    assert len(rows) == 16
    for row in rows:
        assert isinstance(row["bar"], int)
        assert row["action"] in {"LISTEN", "CHOOSE", "HOLD"}
        assert isinstance(row["reason"], str)
        assert len(row["reason"]) > 0
        assert isinstance(row["confidence"], float)
        assert 0.0 <= row["confidence"] <= 1.0


def test_every_row_has_reason_action_features() -> None:
    """Every trace row has reason, action, and feature fields."""
    rows, _ = _run_scenario("enter", "stable_input")
    for row in rows:
        assert "reason" in row
        assert "action" in row
        assert "input_density" in row
        assert "player_certainty" in row
        assert "stability" in row
        assert "change_score" in row
        assert "silence" in row
        assert isinstance(row["reason"], str)
        assert len(row["reason"]) > 0


def test_no_crash_on_empty_intro_bars() -> None:
    """Silent intro bars produce LISTEN without exceptions."""
    rows, _ = _run_scenario("enter", "stable_input")
    # Bar 0 and 1 are silent in the enter scenario.
    silent_rows = [r for r in rows if r["section"] == "silent"]
    assert len(silent_rows) >= 1
    for row in silent_rows:
        assert row["action"] == "LISTEN"
        assert row["beat"] is None


def test_simple_brain_eventually_chooses_in_stable_input() -> None:
    """After enough confident bars, SimpleBrain chooses a beat."""
    rows, brain = _run_scenario("enter", "stable_input")
    choices = [r for r in rows if r["action"] == "CHOOSE"]
    assert len(choices) >= 1
    assert choices[0]["beat"] is not None
    assert brain.current_beat is not None


def test_simple_brain_does_not_choose_during_uncertain_intro() -> None:
    """During the fully uncertain/noisy intro, brain stays in LISTEN."""
    rows, _ = _run_scenario("enter", "uncertain_input")
    # Bars 0-3 are LISTEN/silent/sparse — brain should not choose yet.
    early = rows[:4]
    for row in early:
        assert row["action"] == "LISTEN"
        assert row["beat"] is None


def test_no_dependency_on_behaviour_engine() -> None:
    """``drummer/simple_brain.py`` does not import the old behaviour engine."""
    simple_brain_source = (
        Path(__file__).parent.parent / "drummer" / "simple_brain.py"
    ).read_text()
    assert "drummer.behaviour" not in simple_brain_source
    assert "FeatureDrivenBehaviourEngine" not in simple_brain_source
    assert "BehaviourEngine" not in simple_brain_source


def test_shadow_runner_produces_scores_after_choose() -> None:
    """CHOOSE and HOLD rows include non-empty scores from the brain."""
    rows, _ = _run_scenario("build", "strong_build")

    choices = [r for r in rows if r["action"] == "CHOOSE"]
    holds = [r for r in rows if r["action"] == "HOLD"]

    assert len(choices) >= 1
    assert len(holds) >= 1

    # First CHOOSE row must have populated scores.
    first_choose = choices[0]
    assert isinstance(first_choose.get("scores"), dict)
    assert len(first_choose["scores"]) > 0

    # Every HOLD row after the first CHOOSE must have scores.
    for hold_row in holds:
        assert isinstance(hold_row.get("scores"), dict)
        assert len(hold_row["scores"]) > 0, (
            f"HOLD row at bar {hold_row['bar']} has empty scores"
        )
