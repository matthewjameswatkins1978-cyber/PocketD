#!/usr/bin/env python3
"""Self-play Musical Sanity Batch Runner.

Generates many simulated musical situations, runs Pocket Drummer's
pipeline, applies the Musical Sanity Checker, and reports failures
clearly so the user can fix obvious problems before ear testing.

Usage:
    python demo_self_play_sanity.py --runs 100 --preset all --scenario all
    python demo_self_play_sanity.py --runs 1000 --preset all --scenario all --output-dir artifacts/sanity
    python demo_self_play_sanity.py --seed 123 --runs 1 --scenario enter --preset normal --verbose
    python demo_self_play_sanity.py --seed 123 --run-index 437 --runs 1 --verbose
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from drummer.playtest_feedback import (
    list_playtest_scenarios,
    get_scenario_variations,
    run_playtest_scenario,
    PlaytestScenario,
    PlaytestDiagnosticsSummary,
    PlaytestFeedbackEntry,
)
from drummer.musical_sanity import (
    check_musical_sanity,
    check_scenario_sanity,
    MusicalSanityReport,
    MusicalSanityIssue,
)
from drummer.presets import DrummerPreset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PRESET_NAMES = ["cautious", "normal", "braver"]
_TOP_N_REPRESENTATIVE = 10  # max examples in Lucy brief

# ---------------------------------------------------------------------------
# Run record
# ---------------------------------------------------------------------------


@dataclass
class SelfPlayRun:
    """Record of one self-play run."""

    run_index: int
    seed: int
    scenario: str
    variation: str
    preset: str
    passed: bool
    error_count: int
    warning_count: int
    sanity_issues: list[dict] = field(default_factory=list)
    summary_total_events: int = 0
    summary_first_enter_bar: int | None = None
    summary_confidence_peak: float = 0.0
    summary_contracts_passed: bool = True

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Case generator — deterministic selection from seed
# ---------------------------------------------------------------------------


class DeterministicCaseGenerator:
    """Generates reproducible (scenario, variation, preset) tuples from a seed."""

    def __init__(self, seed: int, scenarios: list[str], presets: list[str]) -> None:
        self._seed = seed
        self._scenarios = scenarios
        self._presets = presets

    def get_case(self, run_index: int) -> tuple[str, str, str]:
        """Return (scenario, variation, preset) for *run_index*.

        Deterministic: same seed + run_index → same case every time.
        """
        # Use a simple deterministic hash
        combined = self._seed * 1000003 + run_index * 7919
        rng_state = (combined * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

        # Pick scenario
        sc_idx = (rng_state % len(self._scenarios))
        scenario = self._scenarios[sc_idx]
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

        # Pick preset
        preset_idx = rng_state % len(self._presets)
        preset = self._presets[preset_idx]
        rng_state = (rng_state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF

        # Pick variation — need to know the variations for the scenario
        # This method is called after scenarios are resolved.
        return scenario, "", preset


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_scenarios(scenario_choice: str) -> list[str]:
    """Resolve --scenario argument to a list of scenario names."""
    if scenario_choice == "all":
        all_scenarios = list_playtest_scenarios()
        # Deterministic order
        return sorted(all_scenarios)
    if scenario_choice in list_playtest_scenarios():
        return [scenario_choice]
    raise ValueError(f"Unknown scenario {scenario_choice!r}")


def _resolve_presets(preset_choice: str) -> list[str]:
    """Resolve --preset argument to a list of preset names."""
    if preset_choice == "all":
        return list(_PRESET_NAMES)
    if preset_choice in _PRESET_NAMES:
        return [preset_choice]
    raise ValueError(f"Unknown preset {preset_choice!r}")


def _compact_event_summary(events: list) -> dict:
    """Build a compact event summary dict from a GrooveEvent list."""
    from drummer.musical_sanity import _resolve_note

    kick_count = 0
    snare_count = 0
    hat_count = 0
    crash_count = 0
    velocities: list[int] = []
    grid_positions: list[int] = []
    instruments: list[str] = []
    notes_present: list[str] = []

    for evt in events:
        note = _resolve_note(evt)
        if note == 36:
            kick_count += 1
            notes_present.append("kick")
        elif note == 38:
            snare_count += 1
            notes_present.append("snare")
        elif note == 42:
            hat_count += 1
            notes_present.append("hat")
        elif note == 49:
            crash_count += 1
            notes_present.append("crash")
        else:
            notes_present.append(f"note_{note}")
        velocities.append(evt.velocity)
        grid_positions.append(evt.grid_position)
        instruments.append(evt.instrument)

    note_set = list(set(notes_present))
    max_vel = max(velocities) if velocities else 0

    return {
        "kick_count": kick_count,
        "snare_count": snare_count,
        "hat_count": hat_count,
        "crash_count": crash_count,
        "max_velocity": max_vel,
        "notes_present": note_set,
        "grid_positions": grid_positions[:8],  # first 8 only for brevity
        "instruments_present": list(set(instruments)),
        "total_events": len(events),
    }


def _build_failure_entry(
    run: SelfPlayRun,
    diag: dict,
    issue: dict,
    bar_events: list,
) -> dict:
    """Build a single failure JSONL entry."""
    return {
        "run_index": run.run_index,
        "seed": run.seed,
        "scenario": run.scenario,
        "variation": run.variation,
        "preset": run.preset,
        "bar_index": diag.get("bar"),
        "section": diag.get("section"),
        "intent": diag.get("intent"),
        "inferred_intent": diag.get("inferred_intent"),
        "event_count": diag.get("event_count"),
        "issue_severity": issue.get("severity"),
        "issue_code": issue.get("code"),
        "issue_message": issue.get("message"),
        "issue_details": issue.get("details"),
        "summary_total_events": run.summary_total_events,
        "summary_confidence_peak": run.summary_confidence_peak,
        "summary_contracts_passed": run.summary_contracts_passed,
        "compact_event_summary": _compact_event_summary(bar_events),
    }


# ---------------------------------------------------------------------------
# Report builders
# ---------------------------------------------------------------------------


def _build_issue_code_counts(runs: list[SelfPlayRun]) -> Counter:
    counter: Counter = Counter()
    for r in runs:
        for issue in r.sanity_issues:
            counter[issue.get("code", "UNKNOWN")] += 1
    return counter


def _build_failure_counts_by_scenario(runs: list[SelfPlayRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in runs:
        if not r.passed:
            counts[r.scenario] = counts.get(r.scenario, 0) + 1
    return counts


def _build_failure_counts_by_preset(runs: list[SelfPlayRun]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in runs:
        if not r.passed:
            counts[r.preset] = counts.get(r.preset, 0) + 1
    return counts


def _find_representative_failures(
    runs: list[SelfPlayRun], max_examples: int = 10,
) -> list[tuple[SelfPlayRun, dict]]:
    """Find representative failure examples, one per unique issue code."""
    seen_codes: set[str] = set()
    examples: list[tuple[SelfPlayRun, dict]] = []
    for r in runs:
        for issue in r.sanity_issues:
            code = issue.get("code", "")
            if code not in seen_codes and len(examples) < max_examples:
                seen_codes.add(code)
                examples.append((r, issue))
    return examples


def _rerun_command(run: SelfPlayRun) -> str:
    return (
        f"python demo_self_play_sanity.py --seed {run.seed} --runs 1 "
        f"--scenario {run.scenario} --preset {run.preset} --verbose"
    )


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------


def write_markdown_report(
    path: str,
    runs: list[SelfPlayRun],
    command: str,
    seed: int,
    scenario_choice: str,
    preset_choice: str,
    output_dir: str,
    total_runs: int,
) -> None:
    total = len(runs)
    passed = sum(1 for r in runs if r.passed)
    failed = total - passed
    total_errors = sum(r.error_count for r in runs)
    total_warnings = sum(r.warning_count for r in runs)
    pass_rate = (passed / total * 100) if total > 0 else 0.0
    issue_counts = _build_issue_code_counts(runs)
    scenario_fails = _build_failure_counts_by_scenario(runs)
    preset_fails = _build_failure_counts_by_preset(runs)
    examples = _find_representative_failures(runs)

    lines: list[str] = []
    lines.append("# Pocket Drummer Self-Play Sanity Report\n")
    lines.append(f"## Run Settings\n")
    lines.append(f"- Command: `{command}`")
    lines.append(f"- Seed: `{seed}`")
    lines.append(f"- Runs: `{total}`")
    lines.append(f"- Scenario: `{scenario_choice}`")
    lines.append(f"- Preset: `{preset_choice}`")
    lines.append(f"- Output: `{output_dir}`")
    lines.append("")

    lines.append("## Results\n")
    lines.append(f"- Total runs: {total}")
    lines.append(f"- Passed: {passed}")
    lines.append(f"- Failed: {failed}")
    lines.append(f"- Pass rate: {pass_rate:.1f}%")
    lines.append(f"- Total errors: {total_errors}")
    lines.append(f"- Total warnings: {total_warnings}")
    lines.append("")

    lines.append("## Top Issue Codes\n")
    if issue_counts:
        for code, count in issue_counts.most_common():
            lines.append(f"- {code}: {count}")
    else:
        lines.append("(no issues)")

    lines.append("")
    lines.append("## Failures by Scenario\n")
    if scenario_fails:
        for sc, count in sorted(scenario_fails.items(), key=lambda x: -x[1]):
            lines.append(f"- {sc}: {count}")
    else:
        lines.append("(no failures)")

    lines.append("")
    lines.append("## Failures by Preset\n")
    if preset_fails:
        for preset, count in sorted(preset_fails.items(), key=lambda x: -x[1]):
            lines.append(f"- {preset}: {count}")
    else:
        lines.append("(no failures)")

    if examples:
        lines.append("")
        lines.append("## Representative Failures\n")
        for run, issue in examples:
            lines.append(f"### {issue.get('code', '?')}")
            lines.append(f"- run_index: {run.run_index}")
            lines.append(f"- seed: {run.seed}")
            lines.append(f"- scenario: {run.scenario}")
            lines.append(f"- variation: {run.variation}")
            lines.append(f"- preset: {run.preset}")
            lines.append(f"- bar: {issue.get('bar_index', '?')}")
            lines.append(f"- intent: {issue.get('intent', '?')}")
            lines.append(f"- message: {issue.get('message', '')}")
            lines.append(f"- severity: {issue.get('severity', '')}")
            lines.append("")
            lines.append(f"  Rerun: `{_rerun_command(run)}`")
            lines.append("")

    lines.append("## Failure Log\n")
    lines.append(f"- JSONL: `{os.path.join(output_dir, 'self_play_failures.jsonl')}`")
    lines.append("")

    lines.append("## Reproduction\n")
    lines.append(f"To reproduce a specific run:")
    lines.append(f"```")
    lines.append(f"python demo_self_play_sanity.py --seed {seed} --runs {total}")
    lines.append(f"```")
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(lines))
        f.write("\n")


# ---------------------------------------------------------------------------
# JSON summary
# ---------------------------------------------------------------------------


def write_json_summary(
    path: str,
    runs: list[SelfPlayRun],
    command: str,
    seed: int,
    scenario_choice: str,
    preset_choice: str,
    output_dir: str,
    output_files: dict[str, str],
) -> None:
    total = len(runs)
    passed = sum(1 for r in runs if r.passed)
    failed = total - passed
    total_errors = sum(r.error_count for r in runs)
    total_warnings = sum(r.warning_count for r in runs)
    pass_rate = (passed / total * 100) if total > 0 else 0.0
    issue_counts = _build_issue_code_counts(runs)
    scenario_fails = _build_failure_counts_by_scenario(runs)
    preset_fails = _build_failure_counts_by_preset(runs)
    examples = _find_representative_failures(runs)

    summary = {
        "command": command,
        "settings": {
            "seed": seed,
            "total_runs": total,
            "scenario": scenario_choice,
            "preset": preset_choice,
            "output_dir": output_dir,
        },
        "totals": {
            "passed": passed,
            "failed": failed,
            "pass_rate": round(pass_rate, 1),
            "total_errors": total_errors,
            "total_warnings": total_warnings,
        },
        "issue_code_counts": dict(issue_counts.most_common()),
        "failures_by_scenario": scenario_fails,
        "failures_by_preset": preset_fails,
        "representative_failures": [
            {
                "run_index": r.run_index,
                "seed": r.seed,
                "scenario": r.scenario,
                "preset": r.preset,
                "code": issue.get("code"),
                "message": issue.get("message"),
                "severity": issue.get("severity"),
            }
            for r, issue in examples
        ],
        "output_files": output_files,
    }

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")


# ---------------------------------------------------------------------------
# Lucy-readable brief
# ---------------------------------------------------------------------------


def write_lucy_brief(
    path: str,
    runs: list[SelfPlayRun],
    command: str,
    seed: int,
    scenario_choice: str,
    preset_choice: str,
    output_dir: str,
) -> None:
    total = len(runs)
    passed = sum(1 for r in runs if r.passed)
    failed = total - passed
    total_errors = sum(r.error_count for r in runs)
    total_warnings = sum(r.warning_count for r in runs)
    pass_rate = (passed / total * 100) if total > 0 else 0.0
    issue_counts = _build_issue_code_counts(runs)
    scenario_fails = _build_failure_counts_by_scenario(runs)
    preset_fails = _build_failure_counts_by_preset(runs)
    examples = _find_representative_failures(runs, _TOP_N_REPRESENTATIVE)

    lines: list[str] = []
    lines.append("# Pocket Drummer Self-Play Sanity Brief\n")
    lines.append("Paste this whole file to Lucy before doing ear testing.\n")
    lines.append("---\n")
    lines.append("")

    lines.append("## Command Run\n")
    lines.append(f"```")
    lines.append(f"{command}")
    lines.append(f"```")
    lines.append("")

    lines.append("## Overall Result\n")
    lines.append(f"- Total runs: {total}")
    lines.append(f"- Passed: {passed}")
    lines.append(f"- Failed: {failed}")
    lines.append(f"- Pass rate: {pass_rate:.1f}%")
    lines.append(f"- Total errors: {total_errors}")
    lines.append(f"- Total warnings: {total_warnings}")
    lines.append("")

    lines.append("## Top Issue Codes\n")
    if issue_counts:
        for code, count in issue_counts.most_common(10):
            lines.append(f"- {code}: {count}")
    else:
        lines.append("(no issues)")

    lines.append("")
    lines.append("## Failures by Scenario\n")
    if scenario_fails:
        for sc, count in sorted(scenario_fails.items(), key=lambda x: -x[1]):
            lines.append(f"- {sc}: {count}")
    else:
        lines.append("(no failures)")

    lines.append("")
    lines.append("## Failures by Preset\n")
    if preset_fails:
        for preset, count in sorted(preset_fails.items(), key=lambda x: -x[1]):
            lines.append(f"- {preset}: {count}")
    else:
        lines.append("(no failures)")

    if examples:
        lines.append("")
        lines.append("## Representative Failures\n")
        for run, issue in examples:
            lines.append(f"### {issue.get('code', '?')}")
            lines.append(f"- run_index: {run.run_index}")
            lines.append(f"- seed: {run.seed}")
            lines.append(f"- scenario: {run.scenario}")
            lines.append(f"- variation: {run.variation}")
            lines.append(f"- preset: {run.preset}")
            bar_idx = issue.get("bar_index", "?")
            lines.append(f"- bar: {bar_idx}")
            lines.append(f"- section: {issue.get('intent', '?')}")
            lines.append(f"- issue message: {issue.get('message', '')}")
            lines.append(f"- severity: {issue.get('severity', '')}")

            # Include compact event summary if available
            details = issue.get("details", {})
            lines.append(f"- compact event summary: {json.dumps(details)}")
            lines.append("")
            lines.append(f"  Rerun: `{_rerun_command(run)}`")
            lines.append("")

    lines.append("## Suggested Next Action\n")
    lines.append("*Generated suggestion, not final judgement.*\n")
    lines.append("")
    top_issues = issue_counts.most_common(3)
    if top_issues:
        for code, _ in top_issues:
            if "ENTER_SOFT" in code:
                lines.append(
                    "- Likely next fix: tighten ENTER_SOFT shaping and/or sanity contract."
                )
            elif "DROP" in code:
                lines.append(
                    "- Likely next fix: separate DROP from BAIL more clearly "
                    "or adjust DROP event generation."
                )
            elif "ANCHOR" in code:
                lines.append(
                    "- Likely next fix: reduce ANCHOR decoration and busyness."
                )
            elif "BAIL" in code:
                lines.append(
                    "- Likely next fix: ensure BAIL section produces exactly zero events."
                )
            elif "FINAL_BAIL" in code:
                lines.append(
                    "- Likely next fix: ensure FINAL_BAIL generates exactly "
                    "kick+crash on beat 1 with no extras."
                )
            elif "MAINTAIN" in code:
                lines.append(
                    "- Likely next fix: ensure MAINTAIN sections always generate groove events."
                )
            elif "LISTEN" in code:
                lines.append(
                    "- Likely next fix: ensure LISTEN sections are always silent."
                )
    else:
        lines.append("- No issues found. Ear testing is ready.")
    lines.append("")
    lines.append("---\n")
    lines.append("")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.writelines("\n".join(lines))
        f.write("\n")


# ---------------------------------------------------------------------------
# Core runner
# ---------------------------------------------------------------------------


def run_single_case(
    scenario_name: str,
    variation_name: str,
    preset_name: str,
) -> tuple[PlaytestDiagnosticsSummary, list[dict], list]:
    """Run one playtest scenario variation and return diagnostics + events."""
    from drummer.playtest_feedback import get_scenario_variations, PlaytestScenario

    # Build a scenario from the requested name/variation/preset
    sc_list = get_scenario_variations(scenario_name, preset=preset_name)
    sc = None
    for s in sc_list:
        if s.variation_name == variation_name:
            sc = s
            break
    if sc is None:
        raise ValueError(
            f"Variation {variation_name!r} not found for scenario {scenario_name!r}"
        )

    return run_playtest_scenario(sc, no_play=True)


def run_sanity_batch(
    total_runs: int,
    seed: int,
    scenario_choice: str,
    preset_choice: str,
    verbose: bool = False,
    failures_only: bool = False,
    stop_on_fail: bool = False,
) -> list[SelfPlayRun]:
    """Run *total_runs* self-play sanity checks and return records."""
    scenarios = _resolve_scenarios(scenario_choice)
    presets = _resolve_presets(preset_choice)
    import random as _random
    rng = _random.Random(seed)

    # Build a list of all (scenario, variation, preset) tuples
    all_cases: list[tuple[str, str, str]] = []
    for sc in scenarios:
        variations = get_scenario_variations(sc, preset=presets[0])
        for v in variations:
            for preset in presets:
                all_cases.append((sc, v.variation_name, preset))

    if not all_cases:
        raise RuntimeError("No cases to run. Check scenario/preset names.")

    runs: list[SelfPlayRun] = []
    for run_index in range(total_runs):
        # Deterministic case selection
        idx = rng.randint(0, len(all_cases) - 1)
        scenario, variation, preset = all_cases[idx]

        if verbose:
            print(f"\n--- Run {run_index + 1}/{total_runs} ---")
            print(f"  scenario={scenario} variation={variation} preset={preset}")

        try:
            summary, raw_diags, global_events = run_single_case(
                scenario, variation, preset,
            )

            # Build bar_events
            bar_events: dict[int, list] = {}
            for evt in global_events:
                bar_events.setdefault(evt.bar_index, []).append(evt)

            # Run on all bars for full-scenario sanity
            from drummer.musical_sanity import check_musical_sanity
            sanity_report = MusicalSanityReport()
            for diag in raw_diags:
                bar = diag.get("bar", 0)
                intent = diag.get("intent", "listen")
                events = bar_events.get(bar, [])
                br = check_musical_sanity(intent, events, bar_index=bar)
                sanity_report.issues.extend(br.issues)

            run = SelfPlayRun(
                run_index=run_index,
                seed=seed,
                scenario=scenario,
                variation=variation,
                preset=preset,
                passed=sanity_report.passed,
                error_count=sanity_report.error_count,
                warning_count=sanity_report.warning_count,
                sanity_issues=[i.to_dict() for i in sanity_report.issues],
                summary_total_events=summary.total_events,
                summary_first_enter_bar=summary.first_enter_bar,
                summary_confidence_peak=summary.confidence_peak,
                summary_contracts_passed=summary.output_contracts_passed,
            )
        except Exception as e:
            run = SelfPlayRun(
                run_index=run_index,
                seed=seed,
                scenario=scenario,
                variation=variation,
                preset=preset,
                passed=False,
                error_count=1,
                warning_count=0,
                sanity_issues=[{
                    "severity": "error",
                    "code": "RUN_EXCEPTION",
                    "message": str(e),
                    "bar_index": None,
                }],
            )

        runs.append(run)

        if verbose:
            status = "PASS" if run.passed else f"FAIL ({run.error_count} errors)"
            print(f"  Result: {status}")

        if not run.passed and not failures_only:
            for issue in run.sanity_issues:
                if issue.get("severity") == "error":
                    bar = issue.get("bar_index", "?")
                    print(f"  FAIL: bar {bar} {issue.get('code', '')}: {issue.get('message', '')}")

        if stop_on_fail and not run.passed:
            print(f"\n  --stop-on-fail triggered at run {run_index}")
            print(f"  Rerun: {_rerun_command(run)}")
            break

    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Self-play musical sanity batch runner."
    )
    parser.add_argument("--runs", type=int, default=100,
                        help="Number of runs (default: 100)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--scenario", type=str, default="all",
                        choices=list_playtest_scenarios() + ["all"],
                        help="Scenario to run (default: all)")
    parser.add_argument("--preset", type=str, default="all",
                        choices=["cautious", "normal", "braver", "all"],
                        help="Preset to run (default: all)")
    parser.add_argument("--output-dir", type=str, default="artifacts/sanity",
                        help="Output directory (default: artifacts/sanity)")
    parser.add_argument("--markdown-report", type=str, default=None,
                        help="Path to markdown report (default: output-dir/self_play_report.md)")
    parser.add_argument("--jsonl-failures", type=str, default=None,
                        help="Path to JSONL failure log (default: output-dir/self_play_failures.jsonl)")
    parser.add_argument("--summary-json", type=str, default=None,
                        help="Path to JSON summary (default: output-dir/self_play_summary.json)")
    parser.add_argument("--lucy-brief", type=str, default=None,
                        help="Path to Lucy-readable brief (default: output-dir/self_play_lucy_brief.md)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print per-run details")
    parser.add_argument("--failures-only", action="store_true",
                        help="Suppress passing run output")
    parser.add_argument("--stop-on-fail", action="store_true",
                        help="Stop at first failure")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Resolve output paths
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    markdown_path = args.markdown_report or os.path.join(output_dir, "self_play_report.md")
    jsonl_path = args.jsonl_failures or os.path.join(output_dir, "self_play_failures.jsonl")
    summary_json_path = args.summary_json or os.path.join(output_dir, "self_play_summary.json")
    lucy_path = args.lucy_brief or os.path.join(output_dir, "self_play_lucy_brief.md")

    output_files = {
        "markdown_report": markdown_path,
        "jsonl_failures": jsonl_path,
        "summary_json": summary_json_path,
        "lucy_brief": lucy_path,
    }

    command = f"python {' '.join(sys.argv)}"

    # Run batch
    runs = run_sanity_batch(
        total_runs=args.runs,
        seed=args.seed,
        scenario_choice=args.scenario,
        preset_choice=args.preset,
        verbose=args.verbose,
        failures_only=args.failures_only,
        stop_on_fail=args.stop_on_fail,
    )

    total = len(runs)
    passed = sum(1 for r in runs if r.passed)
    failed = total - passed
    pass_rate = (passed / total * 100) if total > 0 else 0.0
    total_errors = sum(r.error_count for r in runs)
    total_warnings = sum(r.warning_count for r in runs)
    issue_counts = _build_issue_code_counts(runs)
    scenario_fails = _build_failure_counts_by_scenario(runs)
    preset_fails = _build_failure_counts_by_preset(runs)

    # Print terminal summary
    sep = "=" * 60
    print(f"\n{sep}")
    print("  Self-play Sanity Report")
    print(sep)
    print(f"  Runs:      {total}")
    print(f"  Passed:    {passed}")
    print(f"  Failed:    {failed}")
    print(f"  Pass rate: {pass_rate:.1f}%")
    print(f"  Errors:    {total_errors}")
    print(f"  Warnings:  {total_warnings}")
    print()

    if issue_counts:
        print("  Top issue codes:")
        for code, count in issue_counts.most_common(10):
            print(f"    {code}: {count}")
        print()

    if scenario_fails:
        print("  Failures by scenario:")
        for sc, count in sorted(scenario_fails.items(), key=lambda x: -x[1]):
            print(f"    {sc}: {count}")
        print()

    if preset_fails:
        print("  Failures by preset:")
        for preset, count in sorted(preset_fails.items(), key=lambda x: -x[1]):
            print(f"    {preset}: {count}")
        print()

    # Write JSONL failures
    failures_written = 0
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for run in runs:
            if not run.passed:
                f.write(json.dumps(run.to_dict()) + "\n")
                failures_written += 1

    # Write markdown report
    write_markdown_report(
        markdown_path, runs, command, args.seed,
        args.scenario, args.preset, output_dir, total,
    )

    # Write JSON summary
    write_json_summary(
        summary_json_path, runs, command, args.seed,
        args.scenario, args.preset, output_dir, output_files,
    )

    # Write Lucy brief
    write_lucy_brief(
        lucy_path, runs, command, args.seed,
        args.scenario, args.preset, output_dir,
    )

    print(f"  Report written to:   {markdown_path}")
    print(f"  Failures written to: {jsonl_path} ({failures_written} failures)")
    print(f"  Summary written to:  {summary_json_path}")
    print(f"  Lucy brief written to: {lucy_path}")
    print(f"  Paste that file to Lucy before ear testing.")
    print(sep)
    print()

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())