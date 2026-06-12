"""Tests for model-aware humanization (humanize_events with HumanizeRules).

Verifies backward compatibility with the legacy API and correctness of
model-based humanization for all three preset models.
"""

from __future__ import annotations

import sys
import math
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drummer.humanize import humanize_events  # noqa: E402
from drummer.models import (  # noqa: E402
    MOTORIK_TIGHT_MODEL,
    SIMPLE_ROCK_SAFE_MODEL,
    SPARSE_POSTPUNK_MODEL,
)
from drummer.rules import HumanizeRules  # noqa: E402
from models import KICK, SNARE, CLOSED_HAT  # noqa: E402

# ---- helpers ----------------------------------------------------------------


def _make_fake_events(count: int = 8) -> list[dict]:
    """Build a repeatable set of synthetic scheduler events."""
    return [
        {"timestamp": t, "step": i, "instrument": inst, "note": note, "velocity": vel}
        for i, (t, inst, note, vel) in enumerate(
            [
                (0.000, "kick", KICK, 100),
                (0.125, "hat", CLOSED_HAT, 80),
                (0.250, "snare", SNARE, 100),
                (0.375, "hat", CLOSED_HAT, 80),
                (0.500, "kick", KICK, 100),
                (0.625, "hat", CLOSED_HAT, 80),
                (0.750, "snare", SNARE, 100),
                (0.875, "hat", CLOSED_HAT, 80),
            ]
        )
    ]


def _all_fields_match(
    original: list[dict], humanized: list[dict], key: str
) -> bool:
    return all(
        orig[key] == hum[key]
        for orig, hum in zip(original, humanized)
    )


# ===================================================================
# 1.  Legacy API still works
# ===================================================================


class TestLegacyApi:
    """humanize_events() without humanize_rules must behave as before."""

    def test_legacy_preserves_event_count(self) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, seed=0)
        assert len(result) == len(events)

    def test_legacy_preserves_instruments(self) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, seed=0)
        assert _all_fields_match(events, result, "instrument")

    def test_legacy_preserves_midi_notes(self) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, seed=0)
        assert _all_fields_match(events, result, "note")

    def test_legacy_velocities_stay_in_range(self) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, seed=0)
        for ev in result:
            assert 0 <= ev["velocity"] <= 127

    def test_legacy_timestamps_stay_non_negative(self) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, seed=0)
        for ev in result:
            assert ev["timestamp"] >= 0.0

    def test_legacy_deterministic_same_seed(self) -> None:
        events = _make_fake_events(8)
        r1 = humanize_events(events, seed=42)
        r2 = humanize_events(events, seed=42)
        for e1, e2 in zip(r1, r2):
            assert e1["timestamp"] == e2["timestamp"]
            assert e1["velocity"] == e2["velocity"]

    def test_legacy_event_order_sensible(self) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, seed=0)
        timestamps = [ev["timestamp"] for ev in result]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]


# ===================================================================
# 2.  Model-aware API works (all three models)
# ===================================================================

MODEL_RULES = [
    ("motorik_tight", MOTORIK_TIGHT_MODEL.humanize),
    ("sparse_postpunk", SPARSE_POSTPUNK_MODEL.humanize),
    ("simple_rock_safe", SIMPLE_ROCK_SAFE_MODEL.humanize),
]


class TestModelApi:
    """humanize_events() with humanize_rules must accept all preset models."""

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_accepts_model_rules(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        assert len(result) == len(events)

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_preserves_event_count(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        assert len(result) == len(events)

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_preserves_instruments(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        assert _all_fields_match(events, result, "instrument")

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_preserves_midi_notes(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        assert _all_fields_match(events, result, "note")

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_velocities_stay_in_range(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        for ev in result:
            assert 0 <= ev["velocity"] <= 127

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_timestamps_stay_non_negative(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        for ev in result:
            assert ev["timestamp"] >= 0.0

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_event_order_sensible(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        result = humanize_events(events, humanize_rules=rules, seed=42)
        timestamps = [ev["timestamp"] for ev in result]
        for i in range(1, len(timestamps)):
            assert timestamps[i] >= timestamps[i - 1]

    @pytest.mark.parametrize("name,rules", MODEL_RULES, ids=[m[0] for m in MODEL_RULES])
    def test_model_deterministic_same_seed(self, name: str, rules: HumanizeRules) -> None:
        events = _make_fake_events(8)
        r1 = humanize_events(events, humanize_rules=rules, seed=42)
        r2 = humanize_events(events, humanize_rules=rules, seed=42)
        for e1, e2 in zip(r1, r2):
            assert e1["timestamp"] == e2["timestamp"]
            assert e1["velocity"] == e2["velocity"]

    def test_model_changes_something(self) -> None:
        """At least one timing or velocity value should differ from the original."""
        events = _make_fake_events(16)
        result = humanize_events(
            events, humanize_rules=SIMPLE_ROCK_SAFE_MODEL.humanize, seed=7
        )
        timing_changed = any(
            orig["timestamp"] != hum["timestamp"]
            for orig, hum in zip(events, result)
        )
        velocity_changed = any(
            orig["velocity"] != hum["velocity"]
            for orig, hum in zip(events, result)
        )
        assert timing_changed or velocity_changed


# ===================================================================
# 3.  Different model rules produce different output with same seed
# ===================================================================


class TestModelComparison:
    """Different models should produce measurably different humanization."""

    def test_motorik_and_postpunk_produce_different_output(self) -> None:
        events = _make_fake_events(16)
        motorik = humanize_events(
            events, humanize_rules=MOTORIK_TIGHT_MODEL.humanize, seed=42
        )
        postpunk = humanize_events(
            events, humanize_rules=SPARSE_POSTPUNK_MODEL.humanize, seed=42
        )
        # Expect at least one timestamp to differ between models
        timestamps_differ = any(
            m["timestamp"] != p["timestamp"]
            for m, p in zip(motorik, postpunk)
        )
        assert timestamps_differ, (
            "Motorik and post-punk should produce different timestamps with same seed"
        )

    def test_motorik_timing_spread_not_larger_than_postpunk(self) -> None:
        """Motorik timing jitter should be <= post-punk timing jitter per instrument."""
        motorik = MOTORIK_TIGHT_MODEL.humanize
        postpunk = SPARSE_POSTPUNK_MODEL.humanize
        for inst in ("kick", "snare", "hat"):
            assert motorik.timing_jitter_ms[inst] <= postpunk.timing_jitter_ms[inst], (
                f"Motorik {inst} jitter ({motorik.timing_jitter_ms[inst]}) "
                f"should not exceed post-punk ({postpunk.timing_jitter_ms[inst]})"
            )

    def test_motorik_velocity_spread_not_larger_than_postpunk(self) -> None:
        motorik = MOTORIK_TIGHT_MODEL.humanize
        postpunk = SPARSE_POSTPUNK_MODEL.humanize
        for inst in ("kick", "snare", "hat"):
            assert motorik.velocity_jitter[inst] <= postpunk.velocity_jitter[inst], (
                f"Motorik {inst} velocity spread ({motorik.velocity_jitter[inst]}) "
                f"should not exceed post-punk ({postpunk.velocity_jitter[inst]})"
            )

    def test_motorik_actual_timing_spread_smaller(self) -> None:
        """With a fixed seed, motorik's actual RMS timing deviation should be
        smaller than sparse post-punk's."""
        events = _make_fake_events(32)
        raw_times = [e["timestamp"] for e in events]

        motorik_result = humanize_events(
            events, humanize_rules=MOTORIK_TIGHT_MODEL.humanize, seed=1
        )
        postpunk_result = humanize_events(
            events, humanize_rules=SPARSE_POSTPUNK_MODEL.humanize, seed=1
        )

        def rms_deviation(humanized: list[dict]) -> float:
            deviations = [
                (h["timestamp"] - r) ** 2
                for h, r in zip(humanized, raw_times)
            ]
            return math.sqrt(sum(deviations) / len(deviations))

        motorik_rms = rms_deviation(motorik_result)
        postpunk_rms = rms_deviation(postpunk_result)
        assert motorik_rms <= postpunk_rms, (
            f"Motorik RMS deviation ({motorik_rms:.6f}s) should not exceed "
            f"post-punk RMS deviation ({postpunk_rms:.6f}s)"
        )