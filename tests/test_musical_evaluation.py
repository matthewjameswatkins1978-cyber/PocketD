"""Tests for the musical evaluation harness."""

from __future__ import annotations

import json

import pytest

from drummer.musical_evaluation import (
    Deduction,
    MusicalEvaluationReport,
    evaluate_musical_usefulness,
)
from drummer.musical_sanity import MusicalSanityReport, MusicalSanityIssue


def _diag(bar: int, section: str, intent: str, event_count: int,
          density: float = 0.5, phrase_marker: str = "none",
          notes_summary: str = "", **kw) -> dict:
    """Helper to create a diagnostic dict."""
    d = {
        "bar": bar,
        "section": section,
        "intent": intent,
        "event_count": event_count,
        "inferred_intent": kw.get("inferred_intent", intent),
        "density": density,
        "certainty": 0.5,
        "stability": 0.5,
        "phase": 0.5,
        "confidence": 0.5,
        "notes_summary": notes_summary,
        "phrase_marker": phrase_marker,
    }
    return d


# ---------------------------------------------------------------------------
# Report properties
# ---------------------------------------------------------------------------


class TestReportProperties:
    def test_empty_report(self) -> None:
        r = evaluate_musical_usefulness([])
        assert r.score == 100
        assert r.grade == "excellent"
        assert r.safe_for_ear_testing is True
        assert r.deductions == []
        assert r.direct_deduction_count == 0

    def test_report_serializes(self) -> None:
        r = evaluate_musical_usefulness([])
        d = r.to_dict()
        assert d["score"] == 100
        assert d["grade"] == "excellent"
        assert json.dumps(d)

    def test_format_text_empty(self) -> None:
        r = evaluate_musical_usefulness([])
        text = r.format_text()
        assert "100/100" in text
        assert "YES" in text

    def test_format_text_with_deductions(self) -> None:
        ms = MusicalSanityReport()
        ms.issues.append(MusicalSanityIssue("error", "drop", 13, "DROP_SILENT",
            "DROP produced zero events"))
        r = evaluate_musical_usefulness([], musical_sanity_report=ms)
        text = r.format_text()
        assert "MUSICAL_SANITY_ERRORS" in text or "50/100" in text


# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------


class TestGradeThresholds:
    def test_score_100_is_excellent(self) -> None:
        r = evaluate_musical_usefulness([])
        assert r.grade == "excellent"
        assert r.safe_for_ear_testing is True

    def test_score_94_is_excellent(self) -> None:
        # small deduction
        ms = MusicalSanityReport()
        ms.issues.append(MusicalSanityIssue("warning", "arrangement", 0, "MINOR", "minor issue"))
        # warning only, no error → score stays high
        r = evaluate_musical_usefulness([], musical_sanity_report=ms)
        assert r.score >= 90
        assert r.safe_for_ear_testing is True

    def test_score_82_is_usable(self) -> None:
        diags = [
            _diag(6, "BUILD", "build", 16, density=0.7),
            _diag(7, "BUILD", "build", 18, density=0.7),
            _diag(8, "BUILD", "build", 19, density=0.8),
            _diag(9, "DROP", "drop", 2, density=0.2),
            _diag(10, "DROP", "drop", 2, density=0.2),
            _diag(11, "DROP", "drop", 2, density=0.2),
            _diag(12, "DROP", "drop", 2, density=0.2),
            _diag(13, "DROP", "drop", 2, density=0.2),
        ]
        r = evaluate_musical_usefulness(diags)
        assert 60 <= r.score <= 89

    def test_score_65_is_needs_review(self) -> None:
        diags = [
            _diag(6, "BUILD", "build", 16, density=0.7),
            _diag(7, "BUILD", "build", 18, density=0.7),
            _diag(8, "BUILD", "build", 19, density=0.8),
            _diag(9, "BUILD", "build", 19, density=0.8),
            _diag(10, "REDUCE", "reduce", 2, density=0.2),
            _diag(11, "REDUCE", "reduce", 2, density=0.2),
            _diag(12, "REDUCE", "reduce", 2, density=0.2),
            _diag(13, "REDUCE", "reduce", 2, density=0.2),
        ]
        r = evaluate_musical_usefulness(diags)
        assert r.score < 90
        assert r.safe_for_ear_testing is False or r.grade in ("needs_review", "usable")

    def test_score_40_is_do_not_ear_test(self) -> None:
        ms = MusicalSanityReport()
        ms.issues.append(MusicalSanityIssue("error", "final_bail", 14, "REPEATED_ENDING_CUE",
            "repeated ending"))
        r = evaluate_musical_usefulness([], musical_sanity_report=ms)
        assert r.score < 60
        assert r.grade == "do_not_ear_test"
        assert r.safe_for_ear_testing is False


# ---------------------------------------------------------------------------
# Direct pattern analysis
# ---------------------------------------------------------------------------


class TestDirectPatternAnalysis:
    def test_clean_reports_but_repeated_bars_scores_below_90(self) -> None:
        """Even with clean sanity reports, 8 repeated same-pattern bars should score low."""
        diags = [
            _diag(6, "BUILD", "build", 16, density=0.7),
            _diag(7, "BUILD", "build", 18, density=0.8),
            _diag(8, "BUILD", "build", 19, density=0.8),
            _diag(9, "BUILD", "build", 19, density=0.8),
            _diag(10, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(11, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(12, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(13, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(14, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(15, "MAINTAIN", "maintain", 3, density=0.15),
        ]
        r = evaluate_musical_usefulness(diags)
        assert r.score < 90, f"Expected score < 90, got {r.score}"
        assert r.direct_deduction_count >= 1

    def test_build_with_no_arrival_scores_below_90(self) -> None:
        """Build 4+ bars then collapse into low activity → direct deduction."""
        diags = [
            _diag(6, "BUILD", "build", 16, density=0.7),
            _diag(7, "BUILD", "build", 18, density=0.8),
            _diag(8, "BUILD", "build", 19, density=0.8),
            _diag(9, "BUILD", "build", 19, density=0.8),
            _diag(10, "DROP", "drop", 2, density=0.2),
            _diag(11, "DROP", "drop", 2, density=0.2),
            _diag(12, "DROP", "drop", 2, density=0.2),
            _diag(13, "DROP", "drop", 2, density=0.2),
        ]
        r = evaluate_musical_usefulness(diags)
        direct_codes = {d.code for d in r.deductions if d.source == "direct_evaluation"}
        assert "BUILD_WITH_NO_ARRIVAL" in direct_codes or "STATIC_SECTION_6_PLUS" in direct_codes
        assert r.score < 90

    def test_final_cue_then_resumed_build_scores_below_75(self) -> None:
        """Final cue then scenario resumes → heavy deduction."""
        diags = [
            _diag(12, "MAINTAIN", "maintain", 12, density=0.5),
            _diag(13, "DROP", "drop", 2, density=0.2),
            _diag(14, "FINAL_BAIL", "final_bail", 2, density=0.9),
            _diag(15, "MAINTAIN", "maintain", 12, density=0.5),  # restarted!
            _diag(16, "BUILD", "build", 18, density=0.7),
        ]
        r = evaluate_musical_usefulness(diags)
        assert r.score < 75, f"Expected score < 75, got {r.score}"
        codes = {d.code for d in r.deductions}
        assert "SCENARIO_LOOP_RESTART" in codes

    def test_enter_scenario_with_final_cue_scores_below_threshold(self) -> None:
        """ENTER scenario containing final cue/restart → heavily penalized."""
        diags = [
            _diag(0, "LISTEN", "listen", 0),
            _diag(2, "ENTER_SOFT", "enter_soft", 4, density=0.3),
            _diag(3, "ENTER_SOFT", "enter_soft", 4, density=0.3),
            _diag(4, "MAINTAIN", "maintain", 8, density=0.5),
            _diag(14, "FINAL_BAIL", "final_bail", 2, density=0.9),
            _diag(17, "MAINTAIN", "maintain", 8, density=0.5),  # resumed after final
        ]
        r = evaluate_musical_usefulness(diags)
        assert r.score < 90
        codes = {d.code for d in r.deductions}
        assert "SCENARIO_LOOP_RESTART" in codes

    def test_simple_maintain_with_phrase_movement_scores_high(self) -> None:
        """Stable maintain with occasional phrase markers should score well."""
        diags = [
            _diag(4, "MAINTAIN", "maintain", 8, density=0.5),
            _diag(5, "MAINTAIN", "maintain", 8, density=0.5),
            _diag(6, "MAINTAIN", "maintain", 10, density=0.5, phrase_marker="8bar"),
            _diag(7, "MAINTAIN", "maintain", 8, density=0.5),
            _diag(8, "MAINTAIN", "maintain", 8, density=0.5),
        ]
        r = evaluate_musical_usefulness(diags)
        assert r.score >= 90, f"Expected score >= 90, got {r.score}"

    def test_direct_deductions_appear_with_clean_sanity(self) -> None:
        """Direct deductions appear even when sanity reports are empty."""
        diags = [
            _diag(6, "BUILD", "build", 16, density=0.7),
            _diag(7, "BUILD", "build", 18, density=0.8),
            _diag(8, "BUILD", "build", 19, density=0.8),
            _diag(9, "BUILD", "build", 19, density=0.8),
            _diag(10, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(11, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(12, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(13, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(14, "MAINTAIN", "maintain", 3, density=0.15),
            _diag(15, "MAINTAIN", "maintain", 3, density=0.15),
        ]
        # Explicitly pass empty sanity reports
        ms = MusicalSanityReport()
        arr = MusicalSanityReport()
        r = evaluate_musical_usefulness(diags, musical_sanity_report=ms, arrangement_sanity_report=arr)
        assert r.direct_deduction_count >= 1
        assert r.contract_error_count == 0
        assert r.arrangement_error_count == 0


# ---------------------------------------------------------------------------
# Sanity-to-deduction mapping
# ---------------------------------------------------------------------------


class TestSanityDeductionMapping:
    def test_musical_sanity_error_deducts_50(self) -> None:
        ms = MusicalSanityReport()
        ms.issues.append(MusicalSanityIssue("error", "enter_soft", 4, "ENTER_SOFT_CRASH",
            "crash detected"))
        r = evaluate_musical_usefulness([], musical_sanity_report=ms)
        assert r.score <= 50
        assert r.contract_error_count == 1

    def test_arrangement_error_deducts_35(self) -> None:
        arr = MusicalSanityReport()
        arr.issues.append(MusicalSanityIssue("error", "arrangement", 0, "ARR_ERROR",
            "arrangement error"))
        r = evaluate_musical_usefulness([], arrangement_sanity_report=arr)
        assert r.score <= 65
        assert r.arrangement_error_count == 1

    def test_repeated_final_cue_deducts_40(self) -> None:
        ms = MusicalSanityReport()
        ms.issues.append(MusicalSanityIssue("error", "final_bail", 14, "DOUBLE_FINAL_CRASH",
            "FINAL_BAIL in two consecutive bars"))
        r = evaluate_musical_usefulness([], musical_sanity_report=ms)
        assert r.score <= 60
        codes = {d.code for d in r.deductions}
        assert "DOUBLE_FINAL_CRASH" in codes

    def test_build_with_no_payoff_deducts_20(self) -> None:
        arr = MusicalSanityReport()
        arr.issues.append(MusicalSanityIssue("warning", "arrangement", 6, "BUILD_WITH_NO_PAYOFF",
            "build bars 6-9 followed by 3 low-activity bars"))
        r = evaluate_musical_usefulness([], arrangement_sanity_report=arr)
        codes = {d.code for d in r.deductions}
        assert "BUILD_WITH_NO_PAYOFF" in codes

    def test_isolated_kick_deducts_20(self) -> None:
        arr = MusicalSanityReport()
        arr.issues.append(MusicalSanityIssue("warning", "arrangement", 14, "ISOLATED_KICK_AFTER_DROP",
            "single event after drop"))
        r = evaluate_musical_usefulness([], arrangement_sanity_report=arr)
        codes = {d.code for d in r.deductions}
        assert "ISOLATED_KICK_AFTER_DROP" in codes


# ---------------------------------------------------------------------------
# Self-play integration
# ---------------------------------------------------------------------------


class TestSelfPlayIntegration:
    def test_self_play_run_includes_score(self) -> None:
        """Verify that SelfPlayRun can hold evaluation fields."""
        from demo_self_play_sanity import SelfPlayRun
        r = SelfPlayRun(
            run_index=0, seed=42, scenario="enter", variation="stable_input",
            preset="normal", passed=True, error_count=0, warning_count=0,
            evaluation_score=95,
            evaluation_grade="excellent",
            safe_for_ear_testing=True,
        )
        assert r.evaluation_score == 95
        assert r.evaluation_grade == "excellent"

    def test_self_play_run_to_dict_includes_evaluation(self) -> None:
        from demo_self_play_sanity import SelfPlayRun
        r = SelfPlayRun(
            run_index=0, seed=42, scenario="enter", variation="stable_input",
            preset="normal", passed=True, error_count=0, warning_count=0,
            evaluation_score=72,
            evaluation_grade="needs_review",
            safe_for_ear_testing=False,
        )
        d = r.to_dict()
        assert d["evaluation_score"] == 72
        assert d["evaluation_grade"] == "needs_review"
        assert d["safe_for_ear_testing"] is False