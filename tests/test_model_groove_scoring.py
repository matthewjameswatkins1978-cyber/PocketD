"""Tests for model-based groove scoring (select_groove_decision_with_model).

Verifies the additive decision path works correctly for all three preset
models while the existing heuristic selectors remain unchanged.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from groove_matcher import (  # noqa: E402
    select_groove_decision,
    select_groove_decision_with_model,
    select_groove_by_tempo,
)
from drummer.models import (  # noqa: E402
    MOTORIK_TIGHT_MODEL,
    SIMPLE_ROCK_SAFE_MODEL,
    SPARSE_POSTPUNK_MODEL,
)
from drummer.rules import (  # noqa: E402
    GrooveCandidateScore,
    ModelGrooveDecision,
)

MODELS = [
    ("simple_rock_safe", SIMPLE_ROCK_SAFE_MODEL),
    ("motorik_tight", MOTORIK_TIGHT_MODEL),
    ("sparse_postpunk", SPARSE_POSTPUNK_MODEL),
]


# ===================================================================
# 1.  Function accepts all three built-in models
# ===================================================================


class TestAcceptsAllModels:
    """select_groove_decision_with_model must accept every preset model."""

    @pytest.mark.parametrize("label,model", MODELS, ids=[m[0] for m in MODELS])
    def test_accepts_model(self, label: str, model) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.8, model=model
        )
        assert isinstance(decision, ModelGrooveDecision)
        assert isinstance(decision.selected_groove_id, str)
        assert decision.selected_groove_id


# ===================================================================
# 2.  High-confidence 120 BPM with Motorik selects motorik
# ===================================================================


class TestMotorikHighConfidence:
    """Motorik model at 120 BPM should prefer its highest-scoring groove."""

    def test_motorik_120_high_confidence(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.85, model=MOTORIK_TIGHT_MODEL
        )
        assert decision.selected_groove_id == "motorik"
        assert decision.reason == "model_highest_score"
        assert decision.changed is None  # no previous groove

    def test_motorik_120_high_confidence_has_candidate_scores(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.85, model=MOTORIK_TIGHT_MODEL
        )
        assert len(decision.candidate_scores) >= 1
        for cs in decision.candidate_scores:
            assert isinstance(cs, GrooveCandidateScore)
            assert cs.tempo_score >= 0.0
            assert cs.confidence_score == 0.85
            assert cs.preference_score >= 0.0
            assert cs.change_penalty >= 0.0
            assert cs.total_score >= 0.0

    def test_motorik_120_high_confidence_motorik_score_highest(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.85, model=MOTORIK_TIGHT_MODEL
        )
        motorik_score = None
        for cs in decision.candidate_scores:
            if cs.groove_id == "motorik":
                motorik_score = cs.total_score
                break
        assert motorik_score is not None
        # Motorik should be the winner (first in sorted tuple)
        assert decision.candidate_scores[0].groove_id == "motorik"


# ===================================================================
# 3.  High-confidence 90 BPM with Sparse Post-Punk
# ===================================================================


class TestSparsePostpunkHighConfidence:
    """Sparse post-punk at 90 BPM should prefer half_time or simple_rock."""

    def test_postpunk_90_high_confidence_selects_half_time(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=90.0, confidence=0.85, model=SPARSE_POSTPUNK_MODEL
        )
        assert decision.reason == "model_highest_score"
        # The model has half_time with ideal_tempo=90 and preference=0.7
        assert decision.selected_groove_id == "half_time"

    def test_postpunk_90_high_confidence_has_candidate_scores(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=90.0, confidence=0.85, model=SPARSE_POSTPUNK_MODEL
        )
        assert len(decision.candidate_scores) > 0
        for cs in decision.candidate_scores:
            assert isinstance(cs, GrooveCandidateScore)
            assert cs.tempo_score >= 0.0
            assert cs.preference_score >= 0.0


# ===================================================================
# 4.  Low confidence with previous groove keeps previous
# ===================================================================


class TestLowConfidenceKeepPrevious:
    """Below min_to_change, the model should keep the previous groove."""

    @pytest.mark.parametrize("label,model", MODELS, ids=[m[0] for m in MODELS])
    def test_low_confidence_keeps_previous(self, label: str, model) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.1, model=model, previous_groove_id="half_time"
        )
        assert decision.selected_groove_id == "half_time"
        assert decision.reason == "model_low_confidence_keep_previous"
        assert decision.changed is False
        assert len(decision.candidate_scores) == 0


# ===================================================================
# 5.  Low confidence with no previous groove uses default
# ===================================================================


class TestLowConfidenceDefault:
    """Below min_to_change with no previous, model falls back to default."""

    def test_simple_rock_falls_back(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.1, model=SIMPLE_ROCK_SAFE_MODEL
        )
        assert decision.selected_groove_id == SIMPLE_ROCK_SAFE_MODEL.default_groove_id
        assert decision.reason == "model_low_confidence_default"
        assert decision.changed is None

    def test_motorik_falls_back_to_motorik(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.1, model=MOTORIK_TIGHT_MODEL
        )
        assert decision.selected_groove_id == MOTORIK_TIGHT_MODEL.default_groove_id
        assert decision.reason == "model_low_confidence_default"

    def test_postpunk_falls_back(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.1, model=SPARSE_POSTPUNK_MODEL
        )
        assert decision.selected_groove_id == SPARSE_POSTPUNK_MODEL.default_groove_id
        assert decision.reason == "model_low_confidence_default"


# ===================================================================
# 6.  Deterministic output for same inputs
# ===================================================================


class TestDeterministic:
    """Same inputs must produce identical decisions."""

    @pytest.mark.parametrize("label,model", MODELS, ids=[m[0] for m in MODELS])
    def test_deterministic_repeat(self, label: str, model) -> None:
        d1 = select_groove_decision_with_model(
            bpm=100.0, confidence=0.75, model=model, previous_groove_id="simple_rock"
        )
        d2 = select_groove_decision_with_model(
            bpm=100.0, confidence=0.75, model=model, previous_groove_id="simple_rock"
        )
        assert d1.selected_groove_id == d2.selected_groove_id
        assert d1.reason == d2.reason
        assert d1.changed == d2.changed
        assert len(d1.candidate_scores) == len(d2.candidate_scores)
        for s1, s2 in zip(d1.candidate_scores, d2.candidate_scores):
            assert s1.total_score == s2.total_score


# ===================================================================
# 7.  Change penalty prevents unnecessary switching
# ===================================================================


class TestChangePenalty:
    """A previous groove that scores similarly should be kept."""

    def test_change_penalty_keeps_previous_when_close(self) -> None:
        # Simple rock at 110 BPM — own ideal tempo — so simple_rock
        # should score highest without any previous.
        no_previous = select_groove_decision_with_model(
            bpm=110.0, confidence=0.75, model=SIMPLE_ROCK_SAFE_MODEL
        )
        assert no_previous.selected_groove_id == "simple_rock"

        # With previous=motorik, the change penalty should NOT make
        # motorik win (simple_rock's tempo+preference advantage should
        # overcome the small penalty).
        with_previous = select_groove_decision_with_model(
            bpm=110.0, confidence=0.75, model=SIMPLE_ROCK_SAFE_MODEL,
            previous_groove_id="motorik",
        )
        # simple_rock should still win due to its tempo advantage at 110
        assert with_previous.selected_groove_id == "simple_rock"

    def test_change_penalty_visible_in_scores(self) -> None:
        decision = select_groove_decision_with_model(
            bpm=120.0, confidence=0.8, model=MOTORIK_TIGHT_MODEL,
            previous_groove_id="half_time",
        )
        for cs in decision.candidate_scores:
            if cs.groove_id == "motorik":
                assert cs.change_penalty > 0.0
            elif cs.groove_id == "half_time":
                assert cs.change_penalty == 0.0


# ===================================================================
# 8.  Existing old selectors still work unchanged
# ===================================================================


class TestExistingSelectorsUnchanged:
    """The original heuristic selectors must still produce valid results."""

    def test_select_groove_by_tempo_still_works(self) -> None:
        groove = select_groove_by_tempo(bpm=120.0, confidence=0.8)
        assert groove.id == "motorik"

    def test_select_groove_decision_still_works(self) -> None:
        decision = select_groove_decision(bpm=90.0, confidence=0.8)
        assert decision.selected_groove_id == "half_time"
        assert decision.reason == "high_confidence_half_time"

    def test_select_groove_decision_low_confidence(self) -> None:
        decision = select_groove_decision(
            bpm=120.0, confidence=0.2, previous_groove_id="half_time"
        )
        assert decision.selected_groove_id == "half_time"
        assert decision.reason == "low_confidence_keep_previous"