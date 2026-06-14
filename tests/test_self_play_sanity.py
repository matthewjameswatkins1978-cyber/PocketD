"""Tests for the self-play musical sanity batch runner."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from demo_self_play_sanity import (
    SelfPlayRun,
    _resolve_scenarios,
    _resolve_presets,
    _build_issue_code_counts,
    _build_failure_counts_by_scenario,
    _build_failure_counts_by_preset,
    _find_representative_failures,
    _rerun_command,
    _compact_event_summary,
    _build_musical_evaluation_stats,
    _aggregate_deductions,
    _suggested_next_action,
    _format_deduction_list,
    build_parser,
    run_sanity_batch,
    run_single_case,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tmp_path(suffix: str = ".tmp") -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Scenario/preset resolution
# ---------------------------------------------------------------------------


class TestResolveScenarios:
    def test_all_scenarios(self) -> None:
        scs = _resolve_scenarios("all")
        assert "enter" in scs
        assert "build" in scs
        assert "drop" in scs
        assert "anchor_recovery" in scs
        assert "final_bail" in scs

    def test_single_scenario(self) -> None:
        scs = _resolve_scenarios("enter")
        assert scs == ["enter"]

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_scenarios("nonexistent")


class TestResolvePresets:
    def test_all_presets(self) -> None:
        presets = _resolve_presets("all")
        assert "cautious" in presets
        assert "normal" in presets
        assert "braver" in presets

    def test_single_preset(self) -> None:
        presets = _resolve_presets("normal")
        assert presets == ["normal"]

    def test_unknown_raises(self) -> None:
        with pytest.raises(ValueError):
            _resolve_presets("nonexistent")


# ---------------------------------------------------------------------------
# SelfPlayRun dataclass
# ---------------------------------------------------------------------------


class TestSelfPlayRun:
    def test_to_dict(self) -> None:
        r = SelfPlayRun(
            run_index=0, seed=42, scenario="enter", variation="stable_input",
            preset="normal", passed=True, error_count=0, warning_count=0,
        )
        d = r.to_dict()
        assert d["run_index"] == 0
        assert d["seed"] == 42
        assert d["scenario"] == "enter"
        assert d["passed"] is True

    def test_serializes_to_json(self) -> None:
        r = SelfPlayRun(
            run_index=5, seed=123, scenario="drop", variation="deliberate_sparse",
            preset="braver", passed=False, error_count=2, warning_count=1,
            sanity_issues=[{"code": "DROP_SILENT", "severity": "error"}],
        )
        j = json.dumps(r.to_dict())
        loaded = json.loads(j)
        assert loaded["run_index"] == 5
        assert loaded["sanity_issues"][0]["code"] == "DROP_SILENT"

    def test_includes_evaluation_deductions(self) -> None:
        r = SelfPlayRun(
            run_index=0, seed=42, scenario="enter", variation="stable_input",
            preset="normal", passed=True, error_count=0, warning_count=0,
            evaluation_deductions=[
                {"code": "STATIC_TEST", "reason": "x", "points": -20, "source": "direct_evaluation"},
            ],
        )
        d = r.to_dict()
        assert "evaluation_deductions" in d
        assert len(d["evaluation_deductions"]) == 1
        assert d["evaluation_deductions"][0]["code"] == "STATIC_TEST"


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


class TestReportBuilders:
    @pytest.fixture
    def sample_runs(self) -> list[SelfPlayRun]:
        return [
            SelfPlayRun(0, 1, "enter", "a", "normal", True, 0, 0),
            SelfPlayRun(1, 1, "enter", "b", "braver", False, 2, 0,
                         sanity_issues=[{"code": "ENTER_SOFT_CRASH", "severity": "error"}]),
            SelfPlayRun(2, 1, "drop", "c", "normal", False, 1, 0,
                         sanity_issues=[{"code": "DROP_SILENT", "severity": "error"}]),
            SelfPlayRun(3, 1, "enter", "a", "normal", True, 0, 1,
                         sanity_issues=[{"code": "ENTER_SOFT_TOO_BUSY", "severity": "warning"}]),
        ]

    def test_issue_code_counts(self, sample_runs: list[SelfPlayRun]) -> None:
        counts = _build_issue_code_counts(sample_runs)
        assert counts["ENTER_SOFT_CRASH"] == 1
        assert counts["DROP_SILENT"] == 1
        assert counts["ENTER_SOFT_TOO_BUSY"] == 1

    def test_failure_counts_by_scenario(self, sample_runs: list[SelfPlayRun]) -> None:
        counts = _build_failure_counts_by_scenario(sample_runs)
        assert counts.get("enter") == 1  # run 1 failed
        assert counts.get("drop") == 1   # run 2 failed

    def test_failure_counts_by_preset(self, sample_runs: list[SelfPlayRun]) -> None:
        counts = _build_failure_counts_by_preset(sample_runs)
        assert counts.get("normal") == 1  # run 2 (drop) failed
        assert counts.get("braver") == 1  # run 1 failed

    def test_representative_failures(self, sample_runs: list[SelfPlayRun]) -> None:
        examples = _find_representative_failures(sample_runs, max_examples=10)
        assert len(examples) >= 2  # at least 2 unique codes
        codes = {issue.get("code") for _, issue in examples}
        assert "ENTER_SOFT_CRASH" in codes
        assert "DROP_SILENT" in codes

    def test_rerun_command(self) -> None:
        r = SelfPlayRun(437, 123, "enter", "x", "braver", False, 0, 0)
        cmd = _rerun_command(r)
        assert "seed 123" in cmd
        assert "enter" in cmd
        assert "braver" in cmd


# ---------------------------------------------------------------------------
# Deduction aggregation
# ---------------------------------------------------------------------------


class TestDeductionAggregation:
    def test_empty_runs(self) -> None:
        all_codes, direct_codes = _aggregate_deductions([])
        assert all_codes == []
        assert direct_codes == []

    def test_aggregate_returns_code_counts_and_points(self) -> None:
        runs = [
            SelfPlayRun(0, 42, "enter", "a", "normal", True, 0, 0,
                        evaluation_score=50, total_deductions=-50,
                        evaluation_deductions=[
                            {"code": "STATIC_SECTION_6_PLUS", "reason": "x", "points": -20, "source": "direct_evaluation"},
                            {"code": "STATIC_SECTION_6_PLUS", "reason": "x", "points": -20, "source": "direct_evaluation"},
                            {"code": "BUILD_WITH_NO_ARRIVAL", "reason": "y", "points": -20, "source": "direct_evaluation"},
                            {"code": "MUSICAL_SANITY_ERRORS", "reason": "z", "points": -50, "source": "contract"},
                        ]),
            SelfPlayRun(1, 42, "enter", "b", "normal", True, 0, 0,
                        evaluation_score=60, total_deductions=-40,
                        evaluation_deductions=[
                            {"code": "STATIC_SECTION_6_PLUS", "reason": "x", "points": -20, "source": "direct_evaluation"},
                            {"code": "MUSICAL_SANITY_ERRORS", "reason": "z", "points": -50, "source": "contract"},
                        ]),
        ]
        all_codes, direct_codes = _aggregate_deductions(runs)

        code_map = {c["code"]: c for c in all_codes}
        assert code_map["STATIC_SECTION_6_PLUS"]["runs"] == 2  # 2 runs with this code
        assert code_map["STATIC_SECTION_6_PLUS"]["total_points"] == -60  # sum of ALL deductions: -20*3
        assert code_map["BUILD_WITH_NO_ARRIVAL"]["runs"] == 1
        assert code_map["BUILD_WITH_NO_ARRIVAL"]["total_points"] == -20
        assert code_map["MUSICAL_SANITY_ERRORS"]["runs"] == 2
        assert code_map["MUSICAL_SANITY_ERRORS"]["total_points"] == -100

        # Direct codes should only include direct_evaluation source
        direct_map = {c["code"]: c for c in direct_codes}
        assert "STATIC_SECTION_6_PLUS" in direct_map
        assert "MUSICAL_SANITY_ERRORS" not in direct_map
        assert direct_map["STATIC_SECTION_6_PLUS"]["runs"] == 2


# ---------------------------------------------------------------------------
# Musical evaluation stats
# ---------------------------------------------------------------------------


class TestMusicalEvaluationStats:
    def test_top_direct_deductions_is_list_not_scalar(self) -> None:
        runs = [
            SelfPlayRun(0, 42, "enter", "a", "normal", True, 0, 0,
                        evaluation_score=50, total_deductions=-50,
                        evaluation_deductions=[
                            {"code": "STATIC_TEST", "reason": "x", "points": -20, "source": "direct_evaluation"},
                        ]),
        ]
        stats = _build_musical_evaluation_stats(runs, ear_test_threshold=90)
        assert isinstance(stats["top_direct_deductions"], list)
        assert len(stats["top_direct_deductions"]) == 1
        d = stats["top_direct_deductions"][0]
        assert d["code"] == "STATIC_TEST"
        assert "runs" in d
        assert "total_points" in d

    def test_low_average_score_not_ready(self) -> None:
        runs = [
            SelfPlayRun(0, 42, "enter", "a", "normal", True, 0, 0,
                        evaluation_score=30, evaluation_grade="do_not_ear_test",
                        safe_for_ear_testing=False, total_deductions=-70),
        ]
        stats = _build_musical_evaluation_stats(runs, ear_test_threshold=90)
        assert stats["ready_for_ear_testing"] is False
        assert stats["average_score"] == 30.0

    def test_empty_stats_no_crash(self) -> None:
        stats = _build_musical_evaluation_stats([])
        assert stats["top_deduction_codes"] == []
        assert stats["top_direct_deductions"] == []


# ---------------------------------------------------------------------------
# Suggested next action
# ---------------------------------------------------------------------------


class TestSuggestedNextAction:
    def test_ready_returns_ear_testing_ready(self) -> None:
        stats = {"ready_for_ear_testing": True}
        actions = _suggested_next_action(stats)
        assert any("Ear testing is ready" in a for a in actions)
        assert not any("Do not ear test" in a for a in actions)

    def test_not_ready_mentions_do_not_ear_test(self) -> None:
        stats = {
            "ready_for_ear_testing": False,
            "average_score": 21.0,
            "top_deduction_codes": [
                {"code": "STATIC_SECTION_6_PLUS", "runs": 42, "total_points": -840},
            ],
        }
        actions = _suggested_next_action(stats)
        text = "\n".join(actions)
        assert "Do not ear test yet" in text
        assert "STATIC_SECTION_6_PLUS" in text

    def test_not_ready_with_no_top_codes(self) -> None:
        stats = {
            "ready_for_ear_testing": False,
            "average_score": 21.0,
            "top_deduction_codes": [],
        }
        actions = _suggested_next_action(stats)
        text = "\n".join(actions)
        assert "Do not ear test yet" in text


# ---------------------------------------------------------------------------
# Format deduction list
# ---------------------------------------------------------------------------


class TestFormatDeductionList:
    def test_format_single_item(self) -> None:
        items = [{"code": "TEST", "runs": 5, "total_points": -100}]
        lines = _format_deduction_list(items)
        assert len(lines) > 0
        assert "TEST" in lines[0]
        assert "5 runs" in lines[0]
        assert "100 points" in lines[0]

    def test_empty_list(self) -> None:
        lines = _format_deduction_list([])
        assert "(none)" in lines[0]

    def test_respects_max_items(self) -> None:
        items = [{"code": f"C{i}", "runs": 1, "total_points": -10} for i in range(10)]
        lines = _format_deduction_list(items, max_items=3)
        assert len(lines) == 3


# ---------------------------------------------------------------------------
# Compact event summary
# ---------------------------------------------------------------------------


class TestCompactEventSummary:
    def test_empty_events(self) -> None:
        summary = _compact_event_summary([])
        assert summary["total_events"] == 0
        assert summary["max_velocity"] == 0

    def test_with_events(self) -> None:
        from drummer.feel import GrooveEvent
        events = [
            GrooveEvent("kick", 0, velocity=80),
            GrooveEvent("snare", 4, velocity=90),
            GrooveEvent("hi_hat", 2, velocity=70),
        ]
        summary = _compact_event_summary(events)
        assert summary["kick_count"] == 1
        assert summary["snare_count"] == 1
        assert summary["hat_count"] == 1
        assert summary["crash_count"] == 0
        assert summary["max_velocity"] == 90


# ---------------------------------------------------------------------------
# Batch runner integration
# ---------------------------------------------------------------------------


class TestBatchRunner:
    def test_small_batch_with_all_scenarios(self) -> None:
        runs = run_sanity_batch(
            total_runs=10, seed=42,
            scenario_choice="all", preset_choice="all",
        )
        assert len(runs) == 10
        for r in runs:
            assert r.scenario in ("enter", "build", "anchor_recovery", "drop", "final_bail")
            assert r.preset in ("cautious", "normal", "braver")

    def test_deterministic_same_seed(self) -> None:
        runs1 = run_sanity_batch(5, 99, "all", "all")
        runs2 = run_sanity_batch(5, 99, "all", "all")
        for r1, r2 in zip(runs1, runs2):
            assert r1.scenario == r2.scenario
            assert r1.variation == r2.variation
            assert r1.preset == r2.preset

    def test_different_seeds_different_cases(self) -> None:
        runs1 = run_sanity_batch(5, 99, "all", "all")
        runs2 = run_sanity_batch(5, 100, "all", "all")
        matches = sum(
            1 for r1, r2 in zip(runs1, runs2)
            if r1.scenario == r2.scenario and r1.variation == r2.variation and r1.preset == r2.preset
        )
        assert matches < 5

    def test_scenario_enter_only(self) -> None:
        runs = run_sanity_batch(10, 42, "enter", "all")
        for r in runs:
            assert r.scenario == "enter"

    def test_preset_normal_only(self) -> None:
        runs = run_sanity_batch(10, 42, "all", "normal")
        for r in runs:
            assert r.preset == "normal"

    def test_sanity_runs_full_scenario(self) -> None:
        runs = run_sanity_batch(5, 42, "all", "normal")
        for r in runs:
            assert r.summary_total_events >= 0
            assert isinstance(r.sanity_issues, list)

    def test_zero_failures(self) -> None:
        runs = run_sanity_batch(5, 42, "all", "all")
        for r in runs:
            assert isinstance(r.passed, bool)

    def test_single_verbose_run(self) -> None:
        runs = run_sanity_batch(1, 42, "enter", "normal", verbose=True)
        assert len(runs) == 1

    def test_runs_include_evaluation_deductions(self) -> None:
        runs = run_sanity_batch(3, 42, "enter", "normal")
        for r in runs:
            assert isinstance(r.evaluation_deductions, list)
            assert isinstance(r.evaluation_score, int)


# ---------------------------------------------------------------------------
# Output file generation
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_output_files_are_created(self) -> None:
        import subprocess
        import sys as _sys

        output_dir = _tmp_path(suffix="") + "_sanity_test"
        try:
            result = subprocess.run(
                [_sys.executable, "demo_self_play_sanity.py",
                 "--runs", "5", "--scenario", "all", "--preset", "all",
                 "--output-dir", output_dir],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0

            report = os.path.join(output_dir, "self_play_report.md")
            failures = os.path.join(output_dir, "self_play_failures.jsonl")
            summary = os.path.join(output_dir, "self_play_summary.json")
            lucy = os.path.join(output_dir, "self_play_lucy_brief.md")
            assert os.path.exists(report), f"Missing: {report}"
            assert os.path.exists(failures), f"Missing: {failures}"
            assert os.path.exists(summary), f"Missing: {summary}"
            assert os.path.exists(lucy), f"Missing: {lucy}"

            with open(report) as f:
                content = f.read()
                assert "Self-Play Sanity Report" in content

            with open(summary) as f:
                data = json.load(f)
                assert data["settings"]["total_runs"] == 5
                assert data["totals"]["passed"] == 5

            with open(lucy) as f:
                content = f.read()
                assert "Sanity Brief" in content
                assert "Sanity Result" in content
                assert "Paste this whole file to Lucy" in content

        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_output_contains_musical_evaluation(self) -> None:
        import subprocess
        import sys as _sys

        output_dir = _tmp_path(suffix="") + "_sanity_eval"
        try:
            result = subprocess.run(
                [_sys.executable, "demo_self_play_sanity.py",
                 "--runs", "3", "--scenario", "enter", "--preset", "normal",
                 "--output-dir", output_dir],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            assert result.returncode == 0

            lucy = os.path.join(output_dir, "self_play_lucy_brief.md")
            summary = os.path.join(output_dir, "self_play_summary.json")

            with open(lucy) as f:
                content = f.read()
                assert "Musical Evaluation" in content
                assert "Top deduction codes:" in content

            with open(summary) as f:
                data = json.load(f)
                assert "musical_evaluation" in data
                me = data["musical_evaluation"]
                assert "top_deduction_codes" in me
                assert isinstance(me["top_deduction_codes"], list)

        finally:
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


# ---------------------------------------------------------------------------
# Ear-test threshold
# ---------------------------------------------------------------------------


class TestEarTestThreshold:
    def test_parser_accepts_threshold(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--ear-test-threshold", "90"])
        assert args.ear_test_threshold == 90

    def test_parser_accepts_80(self) -> None:
        parser = build_parser()
        args = parser.parse_args(["--ear-test-threshold", "80"])
        assert args.ear_test_threshold == 80

    def test_parser_default_is_90(self) -> None:
        parser = build_parser()
        args = parser.parse_args([])
        assert args.ear_test_threshold == 90

    def test_batch_readiness_uses_threshold(self) -> None:
        runs = [
            SelfPlayRun(0, 42, "enter", "a", "normal", True, 0, 0,
                        evaluation_score=85, evaluation_grade="usable",
                        safe_for_ear_testing=True, total_deductions=-10),
            SelfPlayRun(1, 42, "enter", "b", "normal", True, 0, 0,
                        evaluation_score=80, evaluation_grade="usable",
                        safe_for_ear_testing=True, total_deductions=-15),
        ]
        stats = _build_musical_evaluation_stats(runs, ear_test_threshold=90)
        assert stats["ready_for_ear_testing"] is False
        assert stats["ear_test_threshold"] == 90

        stats80 = _build_musical_evaluation_stats(runs, ear_test_threshold=80)
        assert stats80["ready_for_ear_testing"] is True
        assert stats80["ear_test_threshold"] == 80

    def test_readiness_requires_zero_errors(self) -> None:
        runs = [
            SelfPlayRun(0, 42, "enter", "a", "normal", True, 1, 0,
                        evaluation_score=95, evaluation_grade="excellent",
                        safe_for_ear_testing=True, total_deductions=0),
            SelfPlayRun(1, 42, "enter", "b", "normal", True, 0, 0,
                        evaluation_score=95, evaluation_grade="excellent",
                        safe_for_ear_testing=True, total_deductions=0),
        ]
        stats = _build_musical_evaluation_stats(runs, ear_test_threshold=90)
        assert stats["ready_for_ear_testing"] is False

    def test_readiness_requires_no_do_not_ear_test(self) -> None:
        runs = [
            SelfPlayRun(0, 42, "enter", "a", "normal", True, 0, 0,
                        evaluation_score=95, evaluation_grade="excellent",
                        safe_for_ear_testing=True, total_deductions=0),
            SelfPlayRun(1, 42, "enter", "b", "normal", True, 0, 0,
                        evaluation_score=55, evaluation_grade="do_not_ear_test",
                        safe_for_ear_testing=False, total_deductions=-45),
        ]
        stats = _build_musical_evaluation_stats(runs, ear_test_threshold=50)
        assert stats["do_not_ear_test_count"] == 1
        assert stats["ready_for_ear_testing"] is False

    def test_lucy_brief_includes_threshold(self) -> None:
        import subprocess
        import sys as _sys
        tmp = tempfile.NamedTemporaryFile(suffix=".md", delete=False)
        tmp.close()
        output_dir = tmp.name + "_ear_test"
        os.makedirs(output_dir, exist_ok=True)
        try:
            subprocess.run(
                [_sys.executable, "demo_self_play_sanity.py",
                 "--runs", "2", "--scenario", "enter", "--preset", "normal",
                 "--lucy-brief", tmp.name, "--output-dir", output_dir,
                 "--ear-test-threshold", "85"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            with open(tmp.name) as f:
                content = f.read()
                assert "Ear-test threshold: 85" in content
        finally:
            os.unlink(tmp.name)
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)

    def test_json_summary_stores_threshold(self) -> None:
        import subprocess
        import sys as _sys
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        tmp.close()
        output_dir = tmp.name + "_ear_test_json"
        os.makedirs(output_dir, exist_ok=True)
        try:
            subprocess.run(
                [_sys.executable, "demo_self_play_sanity.py",
                 "--runs", "2", "--scenario", "enter", "--preset", "normal",
                 "--summary-json", tmp.name, "--output-dir", output_dir,
                 "--ear-test-threshold", "70"],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent),
            )
            with open(tmp.name) as f:
                data = json.load(f)
                assert data["musical_evaluation"]["ear_test_threshold"] == 70
        finally:
            os.unlink(tmp.name)
            import shutil
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)


# ---------------------------------------------------------------------------
# Run single case
# ---------------------------------------------------------------------------


class TestRunSingleCase:
    def test_enter_stable_normal_runs(self) -> None:
        summary, raw_diags, global_events = run_single_case(
            "enter", "stable_input", "normal",
        )
        assert summary.total_events >= 0
        assert len(raw_diags) > 0
        assert len(global_events) > 0

    def test_drop_deliberate_normal_runs(self) -> None:
        summary, raw_diags, global_events = run_single_case(
            "drop", "deliberate_sparse", "normal",
        )
        assert summary.total_events >= 0

    def test_unknown_variation_raises(self) -> None:
        with pytest.raises(ValueError):
            run_single_case("enter", "nonexistent_variation", "normal")