"""Tests for the musical sanity checker."""

from __future__ import annotations

import json

import pytest

from drummer.musical_sanity import (
    MusicalSanityIssue,
    MusicalSanityReport,
    check_musical_sanity,
    check_scenario_sanity,
)
from drummer.feel import GrooveEvent


# ---------------------------------------------------------------------------
# Helpers for constructing test events
# ---------------------------------------------------------------------------


def _event(instrument: str, grid: int = 0, velocity: int = 80, bar_index: int = 0) -> GrooveEvent:
    return GrooveEvent(instrument=instrument, grid_position=grid, velocity=velocity, bar_index=bar_index)


def _kick(grid: int = 0, velocity: int = 80, bar_index: int = 0) -> GrooveEvent:
    return _event("kick", grid, velocity, bar_index)


def _snare(grid: int = 4, velocity: int = 80, bar_index: int = 0) -> GrooveEvent:
    return _event("snare", grid, velocity, bar_index)


def _hat(grid: int = 0, velocity: int = 70, bar_index: int = 0) -> GrooveEvent:
    return _event("hi_hat", grid, velocity, bar_index)


def _crash(grid: int = 0, velocity: int = 100, bar_index: int = 0) -> GrooveEvent:
    return _event("crash", grid, velocity, bar_index)


# ---------------------------------------------------------------------------
# Report tests
# ---------------------------------------------------------------------------


class TestReportProperties:
    def test_empty_report_passes(self) -> None:
        r = MusicalSanityReport()
        assert r.passed is True
        assert r.error_count == 0
        assert r.warning_count == 0

    def test_report_with_warnings_passes(self) -> None:
        r = MusicalSanityReport()
        r.issues.append(MusicalSanityIssue("warning", "enter_soft", 0, "TEST", "test"))
        assert r.passed is True  # warnings don't fail
        assert r.error_count == 0
        assert r.warning_count == 1

    def test_report_with_errors_fails(self) -> None:
        r = MusicalSanityReport()
        r.issues.append(MusicalSanityIssue("error", "enter_soft", 0, "TEST", "test"))
        assert r.passed is False
        assert r.error_count == 1

    def test_to_dict_serializes(self) -> None:
        r = MusicalSanityReport()
        r.issues.append(MusicalSanityIssue("error", "enter_soft", 4, "ENTER_SOFT_CRASH",
            "ENTER_SOFT contains crash"))
        d = r.to_dict()
        assert d["passed"] is False
        assert d["error_count"] == 1
        assert len(d["issues"]) == 1
        assert d["issues"][0]["code"] == "ENTER_SOFT_CRASH"

    def test_to_dict_round_trips_through_json(self) -> None:
        r = MusicalSanityReport()
        r.issues.append(MusicalSanityIssue("warning", "build", 7, "BUILD_SILENT",
            "BUILD is silent"))
        d = r.to_dict()
        serialized = json.dumps(d)
        loaded = json.loads(serialized)
        assert loaded["passed"] is True
        assert loaded["warning_count"] == 1

    def test_format_text_empty(self) -> None:
        r = MusicalSanityReport()
        assert "No musical sanity issues" in r.format_text()

    def test_format_text_with_issues(self) -> None:
        r = MusicalSanityReport()
        r.issues.append(MusicalSanityIssue("error", "enter_soft", 4, "E1", "crash present"))
        r.issues.append(MusicalSanityIssue("warning", "maintain", 5, "W1", "crash present"))
        text = r.format_text()
        assert "FAILED" in text
        assert "1 errors" in text
        assert "crash present" in text


# ---------------------------------------------------------------------------
# ENTER_SOFT tests
# ---------------------------------------------------------------------------


class TestEnterSoftSanity:
    def test_crash_fails(self) -> None:
        events = [_kick(0, 80), _crash(0, 90)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is False
        assert any(i.code == "ENTER_SOFT_CRASH" for i in report.issues)

    def test_final_bail_pattern_fails(self) -> None:
        events = [_kick(0, 100), _crash(0, 100)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is False
        assert any(i.code == "ENTER_SOFT_RESEMBLES_FINAL_BAIL" for i in report.issues)

    def test_isolated_kick_fails(self) -> None:
        events = [_kick(0, 115)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is False
        assert any(i.code == "ENTER_SOFT_ISOLATED_KICK" for i in report.issues)

    def test_too_loud_fails(self) -> None:
        events = [_kick(0, 110), _snare(4, 110), _hat(0, 70), _hat(4, 70)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is False
        assert any(i.code == "ENTER_SOFT_TOO_LOUD" for i in report.issues)

    def test_loud_announcement_fails(self) -> None:
        events = [_kick(0, 105), _hat(2, 70), _snare(4, 80)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is False
        assert any(i.code == "ENTER_SOFT_LOUD_ANNOUNCEMENT" for i in report.issues)

    def test_modest_groove_passes(self) -> None:
        events = [
            _kick(0, 75), _hat(0, 65),
            _hat(2, 60), _snare(4, 80), _hat(4, 65),
            _hat(6, 60), _kick(8, 72), _hat(8, 65),
        ]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is True

    def test_hat_led_entry_passes(self) -> None:
        events = [
            _hat(0, 55), _hat(2, 55), _hat(4, 60),
            _kick(4, 65), _hat(6, 55), _snare(8, 70),
            _hat(8, 60), _hat(10, 55), _hat(12, 60),
            _kick(12, 65), _hat(14, 55),
        ]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is True

    def test_too_busy_warning(self) -> None:
        events = [_kick(i % 16, 70) for i in range(16)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.warning_count >= 1
        assert any(i.code == "ENTER_SOFT_TOO_BUSY" for i in report.issues)

    def test_too_busy_error_at_22(self) -> None:
        events = [_kick(i % 4, 70) for i in range(22)]
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.passed is False
        assert any(i.code == "ENTER_SOFT_TOO_BUSY" and i.severity == "error"
                   for i in report.issues)

    def test_silent_warns(self) -> None:
        events: list[GrooveEvent] = []
        report = check_musical_sanity("enter_soft", events, 4)
        assert report.warning_count >= 1
        assert any(i.code == "ENTER_SOFT_SILENT" for i in report.issues)


# ---------------------------------------------------------------------------
# DROP tests
# ---------------------------------------------------------------------------


class TestDropSanity:
    def test_zero_events_fails(self) -> None:
        report = check_musical_sanity("drop", [], 13)
        assert report.passed is False
        assert any(i.code == "DROP_SILENT" for i in report.issues)

    def test_crash_fails(self) -> None:
        events = [_kick(0, 60), _crash(4, 90)]
        report = check_musical_sanity("drop", events, 13)
        assert report.passed is False
        assert any(i.code == "DROP_CRASH" for i in report.issues)

    def test_final_bail_pattern_fails(self) -> None:
        events = [_kick(0, 100), _crash(0, 100)]
        report = check_musical_sanity("drop", events, 13)
        assert report.passed is False
        assert any(i.code == "DROP_RESEMBLES_FINAL_BAIL" for i in report.issues)

    def test_sparse_kick_passes(self) -> None:
        events = [_kick(0, 60), _kick(8, 55)]
        report = check_musical_sanity("drop", events, 13)
        assert report.passed is True

    def test_too_busy_warns(self) -> None:
        events = [_kick(i, 60) for i in range(6)]
        report = check_musical_sanity("drop", events, 13)
        assert report.warning_count >= 1
        assert any(i.code == "DROP_TOO_BUSY" for i in report.issues)


# ---------------------------------------------------------------------------
# BAIL tests
# ---------------------------------------------------------------------------


class TestBailSanity:
    def test_any_events_fail(self) -> None:
        events = [_kick(0, 80)]
        report = check_musical_sanity("bail", events, 19)
        assert report.passed is False
        assert any(i.code == "BAIL_NOT_SILENT" for i in report.issues)

    def test_zero_events_passes(self) -> None:
        report = check_musical_sanity("bail", [], 19)
        assert report.passed is True

    def test_multiple_events_reports_count(self) -> None:
        events = [_kick(0, 80), _snare(4, 80), _hat(8, 70)]
        report = check_musical_sanity("bail", events, 19)
        assert report.error_count >= 1
        msg = report.issues[0].message
        assert "3" in msg


# ---------------------------------------------------------------------------
# FINAL_BAIL tests
# ---------------------------------------------------------------------------


class TestFinalBailSanity:
    def test_exact_kick_crash_passes(self) -> None:
        events = [_kick(0, 100), _crash(0, 90)]
        report = check_musical_sanity("final_bail", events, 14)
        assert report.passed is True

    def test_extra_event_fails(self) -> None:
        events = [_kick(0, 100), _crash(0, 90), _hat(4, 60)]
        report = check_musical_sanity("final_bail", events, 14)
        assert report.passed is False
        assert any(i.code == "FINAL_BAIL_WRONG_COUNT" for i in report.issues)

    def test_wrong_notes_fails(self) -> None:
        events = [_kick(0, 100), _snare(0, 90)]
        report = check_musical_sanity("final_bail", events, 14)
        assert report.passed is False
        assert any(i.code == "FINAL_BAIL_WRONG_NOTES" for i in report.issues)

    def test_not_on_beat1_fails(self) -> None:
        events = [_kick(4, 100), _crash(4, 90)]
        report = check_musical_sanity("final_bail", events, 14)
        assert report.passed is False
        assert any(i.code == "FINAL_BAIL_BEAT1" for i in report.issues)

    def test_single_event_fails(self) -> None:
        events = [_kick(0, 100)]
        report = check_musical_sanity("final_bail", events, 14)
        assert report.passed is False
        assert any(i.code == "FINAL_BAIL_WRONG_COUNT" for i in report.issues)

    def test_zero_events_fails(self) -> None:
        report = check_musical_sanity("final_bail", [], 14)
        assert report.passed is False
        assert any(i.code == "FINAL_BAIL_WRONG_COUNT" for i in report.issues)


# ---------------------------------------------------------------------------
# ANCHOR tests
# ---------------------------------------------------------------------------


class TestAnchorSanity:
    def test_crash_fails(self) -> None:
        events = [_kick(0, 80), _crash(4, 90)]
        report = check_musical_sanity("anchor", events, 16)
        assert report.passed is False
        assert any(i.code == "ANCHOR_CRASH" for i in report.issues)

    def test_normal_groove_passes(self) -> None:
        events = [_kick(0, 80), _hat(0, 70), _snare(4, 85), _hat(4, 70)]
        report = check_musical_sanity("anchor", events, 16)
        assert report.passed is True

    def test_silent_warns(self) -> None:
        report = check_musical_sanity("anchor", [], 16)
        assert report.warning_count >= 1
        assert any(i.code == "ANCHOR_SILENT" for i in report.issues)

    def test_too_busy_warns(self) -> None:
        events = [_kick(i % 16, 70) for i in range(14)]
        report = check_musical_sanity("anchor", events, 16)
        assert report.warning_count >= 1
        assert any(i.code == "ANCHOR_TOO_BUSY" for i in report.issues)


# ---------------------------------------------------------------------------
# BUILD tests
# ---------------------------------------------------------------------------


class TestBuildSanity:
    def test_silent_warns(self) -> None:
        report = check_musical_sanity("build", [], 7)
        assert report.warning_count >= 1
        assert any(i.code == "BUILD_SILENT" for i in report.issues)

    def test_early_crash_warns(self) -> None:
        events = [_crash(0, 100)]
        report = check_musical_sanity("build", events, 7)
        assert report.warning_count >= 1
        assert any(i.code == "BUILD_CRASH_EARLY" for i in report.issues)

    def test_normal_build_passes(self) -> None:
        events = [
            _kick(0, 90), _hat(0, 75),
            _hat(2, 70), _snare(4, 92), _hat(4, 75),
            _hat(6, 70), _kick(8, 88), _hat(8, 75),
        ]
        report = check_musical_sanity("build", events, 7)
        assert report.passed is True


# ---------------------------------------------------------------------------
# MAINTAIN / REDUCE / LISTEN tests
# ---------------------------------------------------------------------------


class TestOtherIntents:
    def test_maintain_silent_warns(self) -> None:
        report = check_musical_sanity("maintain", [], 5)
        assert report.warning_count >= 1
        assert any(i.code == "MAINTAIN_SILENT" for i in report.issues)

    def test_maintain_crash_warns(self) -> None:
        events = [_kick(0, 80), _crash(4, 100)]
        report = check_musical_sanity("maintain", events, 5)
        assert report.warning_count >= 1
        assert any(i.code == "MAINTAIN_CRASH" for i in report.issues)

    def test_reduce_crash_warns(self) -> None:
        events = [_kick(0, 60), _crash(8, 90)]
        report = check_musical_sanity("reduce", events, 10)
        assert report.warning_count >= 1
        assert any(i.code == "REDUCE_CRASH" for i in report.issues)

    def test_listen_with_events_warns(self) -> None:
        events = [_kick(0, 50)]
        report = check_musical_sanity("listen", events, 0)
        assert report.warning_count >= 1
        assert any(i.code == "LISTEN_NOT_SILENT" for i in report.issues)


# ---------------------------------------------------------------------------
# Scenario-level sanity
# ---------------------------------------------------------------------------


class TestScenarioSanity:
    def test_full_scenario_sanity_integration(self) -> None:
        """Run scenario sanity on a simple bar_events/diagnostics pair."""
        bar_events = {
            0: [],
            4: [_kick(0, 75), _hat(0, 65), _hat(2, 60)],
            13: [_kick(0, 60), _kick(8, 55)],
            14: [_kick(0, 100, bar_index=14), _crash(0, 90, bar_index=14)],
            19: [],
        }
        per_bar_diagnostics = [
            {"bar": 0, "intent": "listen"},
            {"bar": 4, "intent": "enter_soft"},
            {"bar": 13, "intent": "drop"},
            {"bar": 14, "intent": "final_bail"},
            {"bar": 19, "intent": "bail"},
        ]
        report = check_scenario_sanity(bar_events, per_bar_diagnostics)
        # LISTEN bar 0 has no events → passes silently
        # ENTER_SOFT bar 4 has modest groove → passes
        # DROP bar 13 has sparse kick → passes
        # FINAL_BAIL bar 14 has kick+crash → passes
        # BAIL bar 19 has no events → passes
        assert report.passed is True

    def test_scenario_with_failures(self) -> None:
        bar_events = {
            0: [_crash(0, 100)],      # LISTEN with crash → warning
            4: [_crash(0, 100)],      # ENTER_SOFT with crash → error
            13: [],                   # DROP silent → error
            19: [_kick(0, 80)],       # BAIL with event → error
        }
        per_bar_diagnostics = [
            {"bar": 0, "intent": "listen"},
            {"bar": 4, "intent": "enter_soft"},
            {"bar": 13, "intent": "drop"},
            {"bar": 19, "intent": "bail"},
        ]
        report = check_scenario_sanity(bar_events, per_bar_diagnostics)
        assert report.passed is False
        assert report.error_count >= 3

    def test_unknown_intent_returns_empty_report(self) -> None:
        report = check_musical_sanity("fill", [_kick(0, 100)], 0)
        assert report.passed is True
        assert len(report.issues) == 0

    def test_empty_events_with_explicit_intent(self) -> None:
        """All intents should handle empty event lists gracefully."""
        for intent in ("enter_soft", "drop", "bail", "final_bail", "anchor", "build",
                        "maintain", "reduce", "listen"):
            report = check_musical_sanity(intent, [], 0)
            # Should not raise
            assert isinstance(report, MusicalSanityReport)


# ---------------------------------------------------------------------------
# Playtest diagnostics sanity fields
# ---------------------------------------------------------------------------


class TestPlaytestSanityIntegration:
    def test_diagnostics_summary_has_sanity_fields(self) -> None:
        """PlaytestDiagnosticsSummary has musical sanity fields."""
        from drummer.playtest_feedback import PlaytestDiagnosticsSummary
        s = PlaytestDiagnosticsSummary(
            total_events=42, first_enter_bar=3, first_build_bar=None,
            confidence_peak=0.85, phrase_marker_count=2,
            inferred_intents={"enter": 1},
            output_contracts_passed=True,
            drop_event_count=1, final_bail_event_count=2, bail_event_count=0,
        )
        assert s.musical_sanity_passed is True
        assert s.musical_sanity_errors == 0
        assert s.musical_sanity_warnings == 0
        assert s.musical_sanity_issues == []

    def test_diagnostics_summary_sanity_in_to_dict(self) -> None:
        from drummer.playtest_feedback import PlaytestDiagnosticsSummary
        s = PlaytestDiagnosticsSummary(
            total_events=10, first_enter_bar=2, first_build_bar=5,
            confidence_peak=0.5, phrase_marker_count=1,
            inferred_intents={"enter": 1},
            output_contracts_passed=True,
            drop_event_count=1, final_bail_event_count=2, bail_event_count=0,
            musical_sanity_passed=False,
            musical_sanity_errors=2,
            musical_sanity_warnings=1,
            musical_sanity_issues=[{"severity": "error", "code": "TEST"}],
        )
        d = s.to_dict()
        assert d["musical_sanity_passed"] is False
        assert d["musical_sanity_errors"] == 2
        assert d["musical_sanity_warnings"] == 1
        assert len(d["musical_sanity_issues"]) == 1