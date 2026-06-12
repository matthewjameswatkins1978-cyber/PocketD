"""Tests for reason-coded groove decisions.

Verifies that ``select_groove_decision`` returns a ``GrooveDecision`` with the
correct reason code, selected groove, and changed/held status for each scenario.
"""

from __future__ import annotations

import sys
from pathlib import Path

from groove_matcher import (
    GrooveDecision,
    select_groove_by_tempo,
    select_groove_decision,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# High-confidence selections
# ---------------------------------------------------------------------------


def test_decision_120_bpm_returns_motorik_with_reason() -> None:
    """120 BPM high confidence returns motorik with reason high_confidence_motorik."""
    d = select_groove_decision(bpm=120.0, confidence=0.85)
    assert isinstance(d, GrooveDecision)
    assert d.selected_groove_id == "motorik"
    assert d.reason == "high_confidence_motorik"
    assert d.tempo == 120.0
    assert d.confidence == 0.85
    assert d.previous_groove_id is None
    assert d.changed is None


def test_decision_90_bpm_returns_half_time_with_reason() -> None:
    """90 BPM high confidence returns half_time with reason high_confidence_half_time."""
    d = select_groove_decision(bpm=90.0, confidence=0.85)
    assert isinstance(d, GrooveDecision)
    assert d.selected_groove_id == "half_time"
    assert d.reason == "high_confidence_half_time"
    assert d.tempo == 90.0
    assert d.confidence == 0.85


# ---------------------------------------------------------------------------
# Low-confidence memory
# ---------------------------------------------------------------------------


def test_decision_low_confidence_keeps_motorik() -> None:
    """Low confidence with previous motorik returns motorik + low_confidence_keep_previous."""
    d = select_groove_decision(
        bpm=120.0,
        confidence=0.2,
        previous_groove_id="motorik",
    )
    assert d.selected_groove_id == "motorik"
    assert d.reason == "low_confidence_keep_previous"
    assert d.previous_groove_id == "motorik"
    assert d.changed is False


def test_decision_low_confidence_keeps_half_time() -> None:
    """Low confidence with previous half_time returns half_time + low_confidence_keep_previous."""
    d = select_groove_decision(
        bpm=90.0,
        confidence=0.2,
        previous_groove_id="half_time",
    )
    assert d.selected_groove_id == "half_time"
    assert d.reason == "low_confidence_keep_previous"
    assert d.previous_groove_id == "half_time"
    assert d.changed is False


def test_decision_low_confidence_no_previous_falls_back() -> None:
    """Low confidence no previous returns simple_rock + low_confidence_default."""
    d = select_groove_decision(bpm=120.0, confidence=0.2)
    assert d.selected_groove_id == "simple_rock"
    assert d.reason == "low_confidence_default"
    assert d.previous_groove_id is None
    assert d.changed is None


# ---------------------------------------------------------------------------
# Changed / held logic
# ---------------------------------------------------------------------------


def test_decision_changed_true_when_switching_grooves() -> None:
    """changed is True when high confidence switches from motorik to half_time."""
    d = select_groove_decision(
        bpm=90.0,
        confidence=0.85,
        previous_groove_id="motorik",
    )
    assert d.selected_groove_id == "half_time"
    assert d.reason == "high_confidence_half_time"
    assert d.changed is True


def test_decision_changed_false_when_keeping_previous() -> None:
    """changed is False when low confidence keeps the previous groove."""
    d = select_groove_decision(
        bpm=120.0,
        confidence=0.2,
        previous_groove_id="motorik",
    )
    assert d.selected_groove_id == "motorik"
    assert d.changed is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_decision_deterministic() -> None:
    """Same inputs always produce the same GrooveDecision."""
    r1 = select_groove_decision(bpm=120.0, confidence=0.85)
    r2 = select_groove_decision(bpm=120.0, confidence=0.85)
    r3 = select_groove_decision(bpm=120.0, confidence=0.85)

    for r in (r1, r2, r3):
        assert r.selected_groove_id == "motorik"
        assert r.reason == "high_confidence_motorik"
        assert r.tempo == 120.0
        assert r.confidence == 0.85
        assert r.changed is None