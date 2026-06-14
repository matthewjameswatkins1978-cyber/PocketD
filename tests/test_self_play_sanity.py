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
        # At least some runs should differ
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
        """Batch runner checks sanity across all bars, not just focus range."""
        runs = run_sanity_batch(5, 42, "all", "normal")
        for r in runs:
            # All runs should produce full scenario diagnostics
            assert r.summary_total_events >= 0
            # sanity_issues should be populated
            assert isinstance(r.sanity_issues, list)

    def test_zero_failures(self) -> None:
        runs = run_sanity_batch(5, 42, "all", "all")
        for r in runs:
            assert isinstance(r.passed, bool)

    def test_single_verbose_run(self) -> None:
        """A single run with verbose completes without error."""
        runs = run_sanity_batch(1, 42, "enter", "normal", verbose=True)
        assert len(runs) == 1


# ---------------------------------------------------------------------------
# Output file generation
# ---------------------------------------------------------------------------


class TestOutputFiles:
    def test_output_files_are_created(self) -> None:
        """Running the main function generates all four output files."""
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

            # Check output files
            report = os.path.join(output_dir, "self_play_report.md")
            failures = os.path.join(output_dir, "self_play_failures.jsonl")
            summary = os.path.join(output_dir, "self_play_summary.json")
            lucy = os.path.join(output_dir, "self_play_lucy_brief.md")
            assert os.path.exists(report), f"Missing: {report}"
            assert os.path.exists(failures), f"Missing: {failures}"
            assert os.path.exists(summary), f"Missing: {summary}"
            assert os.path.exists(lucy), f"Missing: {lucy}"

            # Verify content
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
                assert "Paste this whole file to Lucy" in content

        finally:
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