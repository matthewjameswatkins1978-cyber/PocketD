"""Tests for the previous-groove memory behaviour.

Verifies that ``select_groove_by_tempo`` correctly remembers and holds the
previous groove when confidence is low, and only falls back to ``simple_rock``
when there is no previous groove to remember.
"""

from __future__ import annotations

import sys
from pathlib import Path

from groove_library import load_grooves
from groove_matcher import select_groove_by_tempo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

grooves = load_grooves()


# ---------------------------------------------------------------------------
# High-confidence selections (same as original groove-change tests)
# ---------------------------------------------------------------------------


def test_high_confidence_120_bpm_selects_motorik() -> None:
    """High-confidence 120 BPM should select motorik."""
    groove = select_groove_by_tempo(bpm=120.0, confidence=0.85)
    assert groove.id == "motorik", (
        f"Expected motorik for 120 BPM @ 0.85 confidence, got {groove.id}"
    )


def test_high_confidence_90_bpm_selects_half_time() -> None:
    """High-confidence 90 BPM should select half_time."""
    groove = select_groove_by_tempo(bpm=90.0, confidence=0.85)
    assert groove.id == "half_time", (
        f"Expected half_time for 90 BPM @ 0.85 confidence, got {groove.id}"
    )


# ---------------------------------------------------------------------------
# Low-confidence memory: keep previous groove
# ---------------------------------------------------------------------------


def test_low_confidence_keeps_motorik() -> None:
    """Low-confidence input with previous groove ``motorik`` keeps motorik."""
    groove = select_groove_by_tempo(
        bpm=120.0,
        confidence=0.2,
        previous_groove_id="motorik",
    )
    assert groove.id == "motorik", (
        f"Expected motorik (kept from previous), got {groove.id}"
    )


def test_low_confidence_keeps_half_time() -> None:
    """Low-confidence input with previous groove ``half_time`` keeps half_time."""
    groove = select_groove_by_tempo(
        bpm=90.0,
        confidence=0.2,
        previous_groove_id="half_time",
    )
    assert groove.id == "half_time", (
        f"Expected half_time (kept from previous), got {groove.id}"
    )


# ---------------------------------------------------------------------------
# Low-confidence fallback: no previous groove → simple_rock
# ---------------------------------------------------------------------------


def test_low_confidence_no_previous_falls_back_to_simple_rock() -> None:
    """Low-confidence input with no previous groove falls back to simple_rock."""
    groove = select_groove_by_tempo(bpm=120.0, confidence=0.2)
    assert groove.id == "simple_rock", (
        f"Expected simple_rock fallback, got {groove.id}"
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_groove_memory_deterministic() -> None:
    """Same inputs always produce the same groove (memory is deterministic)."""
    r1 = select_groove_by_tempo(
        bpm=100.0,
        confidence=0.2,
        previous_groove_id="motorik",
    )
    r2 = select_groove_by_tempo(
        bpm=100.0,
        confidence=0.2,
        previous_groove_id="motorik",
    )
    r3 = select_groove_by_tempo(
        bpm=100.0,
        confidence=0.2,
        previous_groove_id="motorik",
    )
    assert r1.id == r2.id == r3.id == "motorik", (
        "Groove memory must be deterministic for identical inputs"
    )