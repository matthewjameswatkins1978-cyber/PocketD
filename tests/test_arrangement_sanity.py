"""Tests for the arrangement sanity checker."""

from __future__ import annotations

from drummer.arrangement_sanity import (
    check_arrangement_sanity,
    _check_build_with_no_payoff,
    _check_static_drop_too_long,
    _check_isolated_kick_after_drop,
    _check_double_final_crash,
    _check_repeated_ending_cue,
    _check_same_beat_after_change,
    _check_late_enter_then_immediate_build,
    _check_build_too_abrupt_to_drop,
)


def _diag(bar: int, section: str, intent: str, event_count: int,
          **kw) -> dict:
    """Helper to create a diagnostic dict."""
    d = {
        "bar": bar,
        "section": section,
        "intent": intent,
        "event_count": event_count,
        "inferred_intent": kw.get("inferred_intent", intent),
        "density": 0.5,
        "certainty": 0.5,
        "stability": 0.5,
        "phase": 0.5,
        "confidence": 0.5,
        "notes_summary": kw.get("notes_summary", ""),
    }
    return d


# ---------------------------------------------------------------------------
# BUILD_WITH_NO_PAYOFF
# ---------------------------------------------------------------------------


class TestBuildWithNoPayoff:
    def test_normal_build_into_maintain_does_not_warn(self) -> None:
        diags = [
            _diag(0, "LISTEN", "listen", 0),
            _diag(4, "MAINTAIN", "maintain", 12),
            _diag(5, "MAINTAIN", "maintain", 12),
            _diag(6, "MAINTAIN", "build", 16),   # single bar build
            _diag(7, "MAINTAIN", "maintain", 12),
        ]
        issues = _check_build_with_no_payoff(diags)
        assert len(issues) == 0

    def test_long_build_into_flat_output_triggers_warning(self) -> None:
        diags = [
            _diag(5, "MAINTAIN", "maintain", 12),
            _diag(6, "BUILD", "build", 16),
            _diag(7, "BUILD", "build", 18),
            _diag(8, "BUILD", "build", 19),
            _diag(9, "DROP", "drop", 2),
            _diag(10, "REDUCE", "reduce", 2),
            _diag(11, "REDUCE", "reduce", 2),
            _diag(12, "REDUCE", "reduce", 2),
        ]
        issues = _check_build_with_no_payoff(diags)
        assert len(issues) >= 1
        assert any(i.code == "BUILD_WITH_NO_PAYOFF" for i in issues)


# ---------------------------------------------------------------------------
# STATIC_DROP_TOO_LONG
# ---------------------------------------------------------------------------


class TestStaticDropTooLong:
    def test_short_drop_does_not_warn(self) -> None:
        diags = [
            _diag(12, "DROP", "drop", 2),
            _diag(13, "MAINTAIN", "maintain", 8),
        ]
        issues = _check_static_drop_too_long(diags)
        assert len(issues) == 0

    def test_long_static_drop_triggers_warning(self) -> None:
        diags = [
            _diag(9, "BUILD", "build", 18),
            _diag(10, "DROP", "drop", 2),
            _diag(11, "DROP", "drop", 2),
            _diag(12, "DROP", "drop", 2),
            _diag(13, "DROP", "drop", 2),
            _diag(14, "MAINTAIN", "maintain", 8),
        ]
        issues = _check_static_drop_too_long(diags)
        assert len(issues) >= 1
        assert any(i.code == "STATIC_DROP_TOO_LONG" for i in issues)

    def test_drop_same_signature_for_4_bars_triggers(self) -> None:
        diags = [
            _diag(8, "BUILD", "build", 16),
            _diag(9, "DROP", "drop", 1),
            _diag(10, "DROP", "drop", 1),
            _diag(11, "DROP", "drop", 1),
            _diag(12, "DROP", "drop", 1),
            _diag(13, "MAINTAIN", "maintain", 8),
        ]
        issues = _check_static_drop_too_long(diags)
        assert len(issues) >= 1


# ---------------------------------------------------------------------------
# ISOLATED_KICK_AFTER_DROP
# ---------------------------------------------------------------------------


class TestIsolatedKickAfterDrop:
    def test_single_event_after_drop_triggers(self) -> None:
        diags = [
            _diag(12, "DROP", "drop", 1),
            _diag(13, "MAINTAIN", "maintain", 1),
        ]
        issues = _check_isolated_kick_after_drop(diags)
        assert len(issues) >= 1
        assert any(i.code == "ISOLATED_KICK_AFTER_DROP" for i in issues)

    def test_multiple_events_after_drop_does_not_trigger(self) -> None:
        diags = [
            _diag(12, "DROP", "drop", 2),
            _diag(13, "MAINTAIN", "maintain", 8),
        ]
        issues = _check_isolated_kick_after_drop(diags)
        assert len(issues) == 0


# ---------------------------------------------------------------------------
# DOUBLE_FINAL_CRASH
# ---------------------------------------------------------------------------


class TestDoubleFinalCrash:
    def test_single_final_bail_does_not_trigger(self) -> None:
        diags = [
            _diag(14, "FINAL_BAIL", "final_bail", 2),
        ]
        issues = _check_double_final_crash(diags)
        assert len(issues) == 0

    def test_consecutive_final_bail_triggers_error(self) -> None:
        diags = [
            _diag(14, "FINAL_BAIL", "final_bail", 2),
            _diag(15, "FINAL_BAIL", "final_bail", 2),
        ]
        issues = _check_double_final_crash(diags)
        assert len(issues) >= 1
        assert issues[0].severity == "error"
        assert any(i.code == "DOUBLE_FINAL_CRASH" for i in issues)


# ---------------------------------------------------------------------------
# REPEATED_ENDING_CUE
# ---------------------------------------------------------------------------


class TestRepeatedEndingCue:
    def test_single_final_bail_section_passes(self) -> None:
        diags = [
            _diag(14, "FINAL_BAIL", "final_bail", 2),
            _diag(15, "BAIL", "bail", 0),
        ]
        issues = _check_repeated_ending_cue(diags)
        assert len(issues) == 0

    def test_final_bail_section_3_bars_triggers(self) -> None:
        """FINBAIL section appears in more than one span."""
        diags = [
            _diag(14, "FINAL_BAIL", "final_bail", 2),
            _diag(15, "FINAL_BAIL", "final_bail", 2),
        ]
        # Note: 2 bars of FINAL_BAIL = more than one section span
        issues = _check_repeated_ending_cue(diags)
        # Section name FINAL_BAIL appears 2 times
        assert len(issues) >= 1
        assert issues[0].severity == "error"
        assert any(i.code == "REPEATED_ENDING_CUE" for i in issues)


# ---------------------------------------------------------------------------
# SAME_BEAT_AFTER_CHANGE
# ---------------------------------------------------------------------------


class TestSameBeatAfterChange:
    def test_state_change_with_different_output_does_not_warn(self) -> None:
        diags = [
            _diag(4, "ENTER_SOFT", "enter_soft", 4),   # bucket L
            _diag(5, "MAINTAIN", "maintain", 12),       # bucket H
            _diag(6, "MAINTAIN", "maintain", 12),
            _diag(7, "MAINTAIN", "maintain", 12),
        ]
        issues = _check_same_beat_after_change(diags)
        assert len(issues) == 0

    def test_identical_output_after_change_triggers(self) -> None:
        diags = [
            _diag(4, "MAINTAIN", "maintain", 4),
            _diag(5, "MAINTAIN", "enter_soft", 4),  # change but same sig
            _diag(6, "MAINTAIN", "enter_soft", 4),   # third bar same too
        ]
        issues = _check_same_beat_after_change(diags)
        assert len(issues) >= 1
        assert any(i.code == "SAME_BEAT_AFTER_CHANGE" for i in issues)


# ---------------------------------------------------------------------------
# LATE_ENTER_THEN_IMMEDIATE_BUILD
# ---------------------------------------------------------------------------


class TestLateEnterThenImmediateBuild:
    def test_early_enter_no_warn(self) -> None:
        diags = [
            _diag(2, "ENTER_SOFT", "enter_soft", 4),
            _diag(3, "ENTER_SOFT", "enter_soft", 4),
            _diag(4, "BUILD", "build", 12),
        ]
        issues = _check_late_enter_then_immediate_build(diags)
        assert len(issues) == 0

    def test_late_enter_then_build_triggers(self) -> None:
        diags = [
            _diag(2, "ENTER_SOFT", "enter_soft", 2),
            _diag(3, "ENTER_SOFT", "enter_soft", 2),
            _diag(4, "ENTER_SOFT", "enter_soft", 3),
            _diag(5, "ENTER_SOFT", "enter_soft", 3),
            _diag(6, "ENTER_SOFT", "enter_soft", 4),
            _diag(7, "ENTER_SOFT", "enter_soft", 4),
            _diag(8, "ENTER_SOFT", "enter_soft", 4),
            _diag(9, "ENTER_SOFT", "enter_soft", 4),
            _diag(10, "BUILD", "build", 12),
        ]
        issues = _check_late_enter_then_immediate_build(diags)
        assert len(issues) >= 1
        assert any(i.code == "LATE_ENTER_THEN_IMMEDIATE_BUILD" for i in issues)


# ---------------------------------------------------------------------------
# BUILD_TOO_ABRUPT_TO_DROP
# ---------------------------------------------------------------------------


class TestBuildTooAbruptToDrop:
    def test_normal_transition_passes(self) -> None:
        diags = [
            _diag(7, "BUILD", "build", 16),
            _diag(8, "MAINTAIN", "maintain", 10),
            _diag(9, "REDUCE", "reduce", 6),
            _diag(10, "DROP", "drop", 2),
        ]
        issues = _check_build_too_abrupt_to_drop(diags)
        assert len(issues) == 0

    def test_build_to_drop_without_reduce_triggers(self) -> None:
        diags = [
            _diag(7, "BUILD", "build", 16),
            _diag(8, "BUILD", "build", 18),
            _diag(9, "DROP", "drop", 2),
        ]
        issues = _check_build_too_abrupt_to_drop(diags)
        assert len(issues) >= 1
        assert any(i.code == "BUILD_TOO_ABRUPT_TO_DROP" for i in issues)


# ---------------------------------------------------------------------------
# Full arrangement sanity integration
# ---------------------------------------------------------------------------


class TestFullArrangementSanity:
    def test_empty_diagnostics_returns_empty_report(self) -> None:
        report = check_arrangement_sanity([])
        assert report.passed is True
        assert len(report.issues) == 0

    def test_report_serializes(self) -> None:
        diags = [
            _diag(14, "FINAL_BAIL", "final_bail", 2),
            _diag(15, "FINAL_BAIL", "final_bail", 2),
        ]
        report = check_arrangement_sanity(diags)
        d = report.to_dict()
        assert "passed" in d
        assert "issues" in d
        assert d["error_count"] >= 1

    def test_user_scenario_triggers_multiple_issues(self) -> None:
        """Simulate the user's described scenario:
        builds over 4 bars, drops to same beat for 4 bars,
        bass drum by itself, then two crashes.
        """
        diags = [
            _diag(6, "BUILD", "build", 16),
            _diag(7, "BUILD", "build", 18),
            _diag(8, "BUILD", "build", 19),
            _diag(9, "BUILD", "build", 19),
            _diag(10, "DROP", "drop", 2),
            _diag(11, "DROP", "drop", 1),
            _diag(12, "DROP", "drop", 1),
            _diag(13, "DROP", "drop", 1),
            _diag(14, "MAINTAIN", "maintain", 1),  # isolated single event
            _diag(15, "FINAL_BAIL", "final_bail", 2),
            _diag(16, "FINAL_BAIL", "final_bail", 2),  # second crash
        ]
        report = check_arrangement_sanity(diags)
        codes = {i.code for i in report.issues}
        # Should catch at least build with no payoff, double crash, isolated kick, and repeated ending
        assert "BUILD_WITH_NO_PAYOFF" in codes, f"Expected BUILD_WITH_NO_PAYOFF in {codes}"
        assert "DOUBLE_FINAL_CRASH" in codes, f"Expected DOUBLE_FINAL_CRASH in {codes}"
        assert "ISOLATED_KICK_AFTER_DROP" in codes, f"Expected ISOLATED_KICK_AFTER_DROP in {codes}"
        assert "REPEATED_ENDING_CUE" in codes, f"Expected REPEATED_ENDING_CUE in {codes}"
        # At least 4 issues caught (the exact combination may vary)
        assert len(codes) >= 4, f"Expected at least 4 issue codes, got {codes}"
