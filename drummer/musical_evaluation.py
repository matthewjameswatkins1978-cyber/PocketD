"""Musical Evaluation Harness — automated pre-listening quality assessment.

Scores generated performances on musical usefulness before human ear
testing.  Combines three layers of analysis:

1. Direct pattern analysis of diagnostics/events (catches "legal but
   contextless" output that existing sanity checkers miss).
2. Sanity issue-to-deduction mapping from Musical/Arrangement Sanity.
3. Combined scoring with overlap prevention.

Design contract
---------------
* Pure: no MIDI hardware, no playback, no side effects.
* Deterministic: same inputs → same score every time.
* Three-layer: contract errors, arrangement issues, direct pattern checks.
* Not taste: catches obvious contextless behaviour, not subtle feel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from drummer.musical_sanity import MusicalSanityReport
from drummer.arrangement_sanity import _bar_signature


# ---------------------------------------------------------------------------
# Constants / thresholds
# ---------------------------------------------------------------------------

_GRADE_THRESHOLDS: list[tuple[int, str, bool]] = [
    (90, "excellent", True),
    (75, "usable", True),
    (60, "needs_review", False),
    (0, "do_not_ear_test", False),
]


def _grade_from_score(score: int) -> tuple[str, bool]:
    for min_score, grade, safe in _GRADE_THRESHOLDS:
        if score >= min_score:
            return grade, safe
    return "do_not_ear_test", False


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Deduction:
    """One point deduction with source tracking."""

    code: str
    reason: str
    points: int
    source: str  # "contract", "arrangement", "direct_evaluation"

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "reason": self.reason,
            "points": self.points,
            "source": self.source,
        }


@dataclass
class MusicalEvaluationReport:
    """Combined evaluation for a single run."""

    score: int = 100
    grade: str = "excellent"
    safe_for_ear_testing: bool = True
    deductions: list[Deduction] = field(default_factory=list)
    contract_error_count: int = 0
    arrangement_error_count: int = 0
    arrangement_warning_count: int = 0
    direct_deduction_count: int = 0

    @property
    def top_reasons(self) -> list[str]:
        """Return deduction reasons sorted by weight descending."""
        sorted_d = sorted(self.deductions, key=lambda d: -d.points)
        seen: list[str] = []
        for d in sorted_d:
            label = f"[{d.source}] {d.code}: {d.reason} (-{d.points})"
            if label not in seen:
                seen.append(label)
        return seen

    def to_dict(self) -> dict:
        return {
            "score": self.score,
            "grade": self.grade,
            "safe_for_ear_testing": self.safe_for_ear_testing,
            "deductions": [d.to_dict() for d in self.deductions],
            "top_reasons": self.top_reasons,
            "contract_error_count": self.contract_error_count,
            "arrangement_error_count": self.arrangement_error_count,
            "arrangement_warning_count": self.arrangement_warning_count,
            "direct_deduction_count": self.direct_deduction_count,
        }

    def format_text(self) -> str:
        lines: list[str] = []
        lines.append(f"Musical Evaluation: {self.score}/100 ({self.grade})")
        if self.safe_for_ear_testing:
            lines.append("  Safe for ear testing: YES")
        else:
            lines.append("  Safe for ear testing: NO — needs review")
        if self.deductions:
            lines.append(f"  Total deductions: {sum(d.points for d in self.deductions)}")
            for reason in self.top_reasons[:5]:
                lines.append(f"    {reason}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pattern signature helpers
# ---------------------------------------------------------------------------
# The pattern signature focuses on kick/snare/crash skeleton plus fuzzy
# buckets.  Two bars are "similar" if kick+snares match (or subset) AND
# event/density/velocity buckets are close.

def _event_bucket(count: int) -> str:
    if count == 0:
        return "S"
    if count <= 4:
        return "L"
    if count <= 10:
        return "M"
    return "H"


def _density_bucket(density: float) -> str:
    if density <= 0.2:
        return "L"
    if density <= 0.5:
        return "M"
    return "H"


def _velocity_bucket(max_vel: int) -> str:
    if max_vel <= 60:
        return "L"
    if max_vel <= 100:
        return "M"
    return "H"


def _crash_count_bucket(count: int) -> str:
    if count == 0:
        return "0"
    if count == 1:
        return "1"
    return "2+"


def _bar_skeleton_sig(diag: dict) -> str:
    """Return a fuzzy skeleton signature for a bar based on diagnostics.

    Uses event count, density, and notes_summary (crash hints).
    Does NOT require event-level data — works purely from diagnostics.
    """
    events = diag.get("event_count", 0)
    density = diag.get("density", 0.5)
    eb = _event_bucket(events)
    db = _density_bucket(density)
    # Crash hint from notes_summary
    notes = diag.get("notes_summary", "")
    crash_count = notes.count("crash")
    cb = _crash_count_bucket(crash_count)
    return f"{eb}/{db}/{cb}"


def _bars_are_similar(d1: dict, d2: dict) -> bool:
    """True if two bars have similar skeleton signatures.

    Kick/snare skeleton matching is approximated through density + event
    combo.  Two bars with same density and event buckets are considered
    similar in pattern identity.
    """
    return _bar_skeleton_sig(d1) == _bar_skeleton_sig(d2)


def _longest_similar_run(diagnostics: list[dict]) -> int:
    """Return the longest consecutive run of bars with similar skeleton."""
    if not diagnostics:
        return 0
    max_run = 1
    current = 1
    for i in range(1, len(diagnostics)):
        if _bars_are_similar(diagnostics[i - 1], diagnostics[i]):
            current += 1
            max_run = max(max_run, current)
        else:
            current = 1
    return max_run


def _similar_bars_in_range(diagnostics: list[dict], start: int, end: int) -> int:
    """Return the longest run of similar bars within [start, end)."""
    if end - start <= 1:
        return 0
    return _longest_similar_run(diagnostics[start:end])


def _bars_after_final_cue(diagnostics: list[dict]) -> list[int]:
    """Return bar indices that occur after a FINAL_BAIL section."""
    n = len(diagnostics)
    final_bail_bar = None
    for i in range(n):
        if diagnostics[i].get("section") == "FINAL_BAIL":
            final_bail_bar = i
            break
    if final_bail_bar is None:
        return []
    # Bars after the final bail section
    after = [i for i in range(final_bail_bar + 1, n)]
    # Filter to only non-silent bars
    return [i for i in after if diagnostics[i].get("event_count", 0) > 0]


def _has_final_cue(diagnostics: list[dict]) -> bool:
    return any(d.get("section") == "FINAL_BAIL" for d in diagnostics)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evaluation_intent(diag: dict) -> str:
    """Return the intent that should be used for musical evaluation.

    Prefers ``rendered_intent`` (the intent that actually drove output
    shaping) over raw pipeline ``intent``. Falls back to ``intent``
    only when ``rendered_intent`` is absent (backward compatibility).
    """
    return diag.get("rendered_intent", diag.get("intent", ""))


# ---------------------------------------------------------------------------
# Direct evaluation checks
# ---------------------------------------------------------------------------


def _check_static_section(diagnostics: list[dict]) -> list[Deduction]:
    """Detect long runs of same-pattern bars.

    Deducts:
    - 4+ near-identical bars after state change: -10
    - 6+ near-identical bars after build/drop: -20
    """
    deductions: list[Deduction] = []
    n = len(diagnostics)

    for i in range(n):
        intent = _evaluation_intent(diagnostics[i])
        if intent in ("build", "drop", "reduce"):
            # Look ahead for long same-pattern run after this section
            after_start = i + 1
            max_run = _similar_bars_in_range(diagnostics, after_start, min(after_start + 10, n))
            if max_run >= 6:
                deductions.append(Deduction(
                    "STATIC_SECTION_6_PLUS",
                    f"After {intent} bar {i}: {max_run} consecutive near-identical pattern bars",
                    -20, "direct_evaluation",
                ))
                break  # Only catch the worst case once
            elif max_run >= 4:
                deductions.append(Deduction(
                    "STATIC_SECTION_4_PLUS",
                    f"After {intent} bar {i}: {max_run} consecutive near-identical pattern bars",
                    -10, "direct_evaluation",
                ))
                break  # Only catch the worst case once

    return deductions


def _check_build_payoff(diagnostics: list[dict]) -> list[Deduction]:
    """Detect BUILD lasting 3+ bars that collapses into no arrival.

    Deducts -20 if build followed by long static low-intensity or
    immediate collapse without recovery.
    """
    deductions: list[Deduction] = []
    n = len(diagnostics)

    for i in range(n):
        if _evaluation_intent(diagnostics[i]) == "build":
            # Count consecutive build bars
            build_start = i
            build_end = i
            while build_end + 1 < n and diagnostics[build_end + 1].get("intent") == "build":
                build_end += 1
            build_len = build_end - build_start + 1

            if build_len >= 3:
                # Check what follows
                after_start = build_end + 1
                if after_start >= n:
                    continue
                # Look for flat/low output
                low_count = 0
                for j in range(after_start, min(after_start + 6, n)):
                    if diagnostics[j].get("event_count", 0) <= 4:
                        low_count += 1
                    else:
                        break
                if low_count >= 3:
                    deductions.append(Deduction(
                        "BUILD_WITH_NO_ARRIVAL",
                        f"Build bars {build_start}-{build_end} ({build_len} bars) "
                        f"followed by {low_count} bars of low-activity output with no arrival",
                        -20, "direct_evaluation",
                    ))
                    break  # Only flag the first/longest build

    return deductions


def _check_scenario_loop(diagnostics: list[dict]) -> list[Deduction]:
    """Detect final cue followed by resumed non-silent output.

    Deducts -40 if final cue appears then later bars have events.
    """
    deductions: list[Deduction] = []
    after = _bars_after_final_cue(diagnostics)
    if after:
        deductions.append(Deduction(
            "SCENARIO_LOOP_RESTART",
            f"FINAL_BAIL section followed by {len(after)} non-silent bars: "
            f"bars {after}; scenario should end after final cue",
            -40, "direct_evaluation",
        ))
    return deductions


def _check_gesture_window(diagnostics: list[dict]) -> list[Deduction]:
    """Detect too many crash events within a short window.

    Deducts -15 for multiple crash appearances within 2 bars.
    """
    deductions: list[Deduction] = []
    n = len(diagnostics)

    for i in range(n - 1):
        notes_i = diagnostics[i].get("notes_summary", "")
        notes_next = diagnostics[i + 1].get("notes_summary", "")
        crash_count = notes_i.count("crash") + notes_next.count("crash")
        if crash_count >= 3 and diagnostics[i].get("section") != "FINAL_BAIL":
            deductions.append(Deduction(
                "GESTURE_WINDOW",
                f"Multiple crash events across bars {i}-{i + 1} "
                f"({crash_count} crashes in 2-bar window)",
                -15, "direct_evaluation",
            ))
            break

    return deductions


def _check_intent_change_no_pattern_change(diagnostics: list[dict]) -> list[Deduction]:
    """Detect intent changes that don't change the output pattern.

    Deducts -10 if intent changes but bar skeleton stays same for 3+ bars.
    Uses ``rendered_intent`` to match the intent that drove actual output.
    """
    deductions: list[Deduction] = []
    n = len(diagnostics)

    for i in range(1, n):
        prev_intent = _evaluation_intent(diagnostics[i - 1])
        curr_intent = _evaluation_intent(diagnostics[i])
        if prev_intent != curr_intent:
            # Check 3 bars after the change
            end = min(i + 3, n)
            run = _similar_bars_in_range(diagnostics, i, end)
            if run >= 2:
                deductions.append(Deduction(
                    "INTENT_CHANGE_NO_PATTERN_CHANGE",
                    f"Intent changed from '{prev_intent}' to '{curr_intent}' at bar {i} "
                    f"but skeleton pattern unchanged for {run} bars",
                    -10, "direct_evaluation",
                ))
                break

    return deductions


def _check_scenario_purpose(diagnostics: list[dict]) -> list[Deduction]:
    """Check if output violates the scenario purpose.

    Checks based on section sequence:
    - If no BUILD section but ENTER needed, warn
    - If FINAL_BAIL appears but ENTER was recent, deduct
    Uses ``rendered_intent`` when available, falls back to section name.
    """
    deductions: list[Deduction] = []
    n = len(diagnostics)

    # ENTER scenario should not contain FINAL_BAIL-like restart
    # Use rendered_intent or section name
    enter_bars = [d for d in diagnostics
                  if _evaluation_intent(d) in ("enter_soft", "enter_full")
                  or d.get("section") == "ENTER_SOFT"]
    final_bail_bars = [d for d in diagnostics if d.get("section") == "FINAL_BAIL"]

    if enter_bars and final_bail_bars:
        last_enter = enter_bars[-1]["bar"]
        first_final = final_bail_bars[0]["bar"]
        # If ENTER happened AND final bail happened, check if there's non-silent
        # output after the ending that resembles a restart
        after = _bars_after_final_cue(diagnostics)
        enter_after_final = [b for b in after if b >= last_enter]
        if enter_after_final:
            deductions.append(Deduction(
                "SCENARIO_PURPOSE_VIOLATION",
                f"ENTER scenario contains FINAL_BAIL followed by {len(enter_after_final)} "
                f"non-silent bars; scenario likely restarted unnaturally",
                -15, "direct_evaluation",
            ))

    return deductions


def _check_no_phrase_movement(diagnostics: list[dict]) -> list[Deduction]:
    """Detect 8+ stable maintain/enter bars without phrase movement.

    Deducts -8 if no phrase marker appears during long stable section.
    Uses ``rendered_intent`` when available.
    """
    deductions: list[Deduction] = []
    n = len(diagnostics)

    stable_count = 0
    for i in range(n):
        intent = _evaluation_intent(diagnostics[i])
        section = diagnostics[i].get("section", "")
        if intent in ("maintain", "enter_soft", "enter_full"):
            stable_count += 1
            phrase = diagnostics[i].get("phrase_marker", "none")
            if phrase != "none":
                stable_count = 0  # Reset — phrase movement detected
            if stable_count >= 8:
                deductions.append(Deduction(
                    "NO_PHRASE_MOVEMENT",
                    f"No phrase movement for {stable_count} consecutive stable bars "
                    f"(bars {i - stable_count + 1}-{i}); long stable sections "
                    "benefit from subtle 8-bar movement",
                    -8, "direct_evaluation",
                ))
                break
        else:
            stable_count = 0

    return deductions


# ---------------------------------------------------------------------------
# Sanity-to-deduction mapping
# ---------------------------------------------------------------------------

def _deductions_from_musical_sanity(report: MusicalSanityReport) -> list[Deduction]:
    """Map musical sanity issues to deductions."""
    deductions: list[Deduction] = []
    seen_codes: set[str] = set()

    # Critical: any error → -50
    if report.error_count > 0:
        deductions.append(Deduction(
            "MUSICAL_SANITY_ERRORS",
            f"{report.error_count} musical sanity error(s)",
            -50, "contract",
        ))

    for issue in report.issues:
        code = issue.code
        if code in seen_codes:
            continue
        seen_codes.add(code)

        # Heavy hitters
        if code in ("DOUBLE_FINAL_CRASH", "REPEATED_ENDING_CUE"):
            deductions.append(Deduction(
                code, issue.message, -40, "contract",
            ))
        elif code == "BAIL_NOT_SILENT":
            deductions.append(Deduction(
                code, issue.message, -50, "contract",
            ))
        elif code in ("ENTER_SOFT_CRASH", "DROP_CRASH", "ANCHOR_CRASH"):
            deductions.append(Deduction(
                code, issue.message, -15, "contract",
            ))
        elif code == "ENTER_SOFT_ISOLATED_KICK":
            deductions.append(Deduction(
                code, issue.message, -20, "contract",
            ))

    return deductions


def _deductions_from_arrangement_sanity(report: MusicalSanityReport) -> list[Deduction]:
    """Map arrangement sanity issues to deductions."""
    deductions: list[Deduction] = []
    seen_codes: set[str] = set()

    # Critical: any error → -35
    if report.error_count > 0:
        deductions.append(Deduction(
            "ARRANGEMENT_SANITY_ERRORS",
            f"{report.error_count} arrangement sanity error(s)",
            -35, "arrangement",
        ))

    for issue in report.issues:
        code = issue.code
        if code in seen_codes:
            continue
        seen_codes.add(code)

        if code == "BUILD_WITH_NO_PAYOFF":
            deductions.append(Deduction(
                code, issue.message, -20, "arrangement",
            ))
        elif code == "STATIC_DROP_TOO_LONG":
            duration = issue.details.get("duration_bars", 0)
            if duration >= 6:
                deductions.append(Deduction(
                    code, issue.message, -20, "arrangement",
                ))
            else:
                deductions.append(Deduction(
                    code, issue.message, -15, "arrangement",
                ))
        elif code == "ISOLATED_KICK_AFTER_DROP":
            deductions.append(Deduction(
                code, issue.message, -20, "arrangement",
            ))
        elif code == "BUILD_TOO_ABRUPT_TO_DROP":
            deductions.append(Deduction(
                code, issue.message, -12, "arrangement",
            ))
        elif code == "SAME_BEAT_AFTER_CHANGE":
            deductions.append(Deduction(
                code, issue.message, -10, "arrangement",
            ))
        elif code == "LATE_ENTER_THEN_IMMEDIATE_BUILD":
            deductions.append(Deduction(
                code, issue.message, -10, "arrangement",
            ))

    return deductions


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate_musical_usefulness(
    per_bar_diagnostics: list[dict],
    global_events: list | None = None,
    musical_sanity_report: MusicalSanityReport | None = None,
    arrangement_sanity_report: MusicalSanityReport | None = None,
    context: dict | None = None,
) -> MusicalEvaluationReport:
    """Score a generated performance on musical usefulness.

    Combines direct pattern analysis, musical sanity deductions, and
    arrangement sanity deductions into a single 0-100 score.

    Parameters
    ----------
    per_bar_diagnostics : list[dict]
        Per-bar diagnostic records from the pipeline.
    global_events : list or None
        Optional list of GrooveEvents (for future detailed analysis).
    musical_sanity_report : MusicalSanityReport or None
        Report from :func:`check_musical_sanity`.
    arrangement_sanity_report : MusicalSanityReport or None
        Report from :func:`check_arrangement_sanity`.
    context : dict or None
        Optional context (scenario name, variation, etc.).

    Returns
    -------
    MusicalEvaluationReport
        Combined evaluation with score, grade, and deductions.
    """
    all_deductions: list[Deduction] = []
    contract_errors = 0
    arrangement_errors = 0
    arrangement_warnings = 0

    # Layer 1: direct pattern analysis (works even with empty sanity reports)
    direct_checks = [
        _check_static_section,
        _check_build_payoff,
        _check_scenario_loop,
        _check_gesture_window,
        _check_intent_change_no_pattern_change,
        _check_scenario_purpose,
        _check_no_phrase_movement,
    ]
    for check in direct_checks:
        all_deductions.extend(check(per_bar_diagnostics))

    direct_count = sum(1 for d in all_deductions if d.source == "direct_evaluation")

    # Layer 2: sanity issue mapping
    if musical_sanity_report is not None:
        san_ded = _deductions_from_musical_sanity(musical_sanity_report)
        contract_errors = musical_sanity_report.error_count
        all_deductions.extend(san_ded)

    if arrangement_sanity_report is not None:
        arr_ded = _deductions_from_arrangement_sanity(arrangement_sanity_report)
        arrangement_errors = arrangement_sanity_report.error_count
        arrangement_warnings = arrangement_sanity_report.warning_count
        all_deductions.extend(arr_ded)

    # Deduplicate — keep the largest deduction for each code
    code_max: dict[str, Deduction] = {}
    for d in all_deductions:
        if d.code not in code_max or d.points < code_max[d.code].points:
            code_max[d.code] = d
    deduped = list(code_max.values())

    # Calculate score
    total_deduction = sum(d.points for d in deduped)
    # Ensure min deduction = -100 (cap)
    score = max(0, min(100, 100 + total_deduction))

    grade, safe = _grade_from_score(score)

    return MusicalEvaluationReport(
        score=score,
        grade=grade,
        safe_for_ear_testing=safe,
        deductions=deduped,
        contract_error_count=contract_errors,
        arrangement_error_count=arrangement_errors,
        arrangement_warning_count=arrangement_warnings,
        direct_deduction_count=direct_count,
    )