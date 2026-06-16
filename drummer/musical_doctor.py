"""Musical Doctor - lightweight automated diagnosis from bar transcripts.

Reads BarTranscript data and produces a short musical diagnosis plus
suggested fix direction.  Inspection only - no MIDI, no grooves,
no auto-fixing.

Design contract
---------------
* Pure: bar transcript -> diagnosis (no side effects, no playback).
* Deterministic: same transcript -> same diagnosis every time.
* Not a fixer: suggests directions, does not edit code.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from drummer.bar_transcript import BarTranscript, BarLine


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class DoctorProblem:
    """One detected musical-shape issue."""

    rule: str                    # e.g. "DROP_REPEATED_NAKED_KICK"
    diagnosis: str               # human-readable description
    suggested_fix: str           # actionable direction
    confidence: str              # "low" | "medium" | "high"
    affected_bars: list[int]     # bar indices


@dataclass
class DoctorReport:
    """Complete musical doctor diagnosis for a playtest run."""

    scenario: str = ""
    preset: str = ""
    variation: str = ""
    examined_bars: int = 0
    problems: list[DoctorProblem] = field(default_factory=list)

    @property
    def high_confidence_problems(self) -> list[DoctorProblem]:
        return [p for p in self.problems if p.confidence == "high"]

    @property
    def problem_count(self) -> int:
        return len(self.problems)


# ---------------------------------------------------------------------------
# Rule 1 - DROP repeated naked kick
# ---------------------------------------------------------------------------


def _check_drop_repeated_naked_kick(bars: list[BarLine]) -> list[DoctorProblem]:
    """Detect consecutive DROP bars with bare loud kicks, no hats/snare.

    Trigger: two consecutive bars where both have rendered_intent == "drop",
    at least one kick, zero snare/hat/ride, and max_velocity > 90.
    """
    problems: list[DoctorProblem] = []
    n = len(bars)

    for i in range(n - 1):
        a = bars[i]
        b = bars[i + 1]
        if not (_is_naked_drop_kick(a) and _is_naked_drop_kick(b)):
            continue

        problems.append(DoctorProblem(
            rule="DROP_REPEATED_NAKED_KICK",
            diagnosis=f"DROP bars {a.bar}-{b.bar} have repeated naked kick stamps "
                       f"(no hats, no snare, velocity > 90).",
            suggested_fix="Vary second drop bar: reduce velocity, add ghost pulse "
                          "or quiet hat tick to avoid identical isolated kicks.",
            confidence="high",
            affected_bars=[a.bar, b.bar],
        ))
        break  # Only flag the first pair

    return problems


def _is_naked_drop_kick(bl: BarLine) -> bool:
    """True if this DROP bar is a loud kick with zero hats/snare/ride."""
    return (
        bl.rendered_intent == "drop"
        and bl.kick_count >= 1
        and bl.snare_count == 0
        and bl.hat_count == 0
        and bl.ride_count == 0
        and bl.max_velocity > 90
    )


# ---------------------------------------------------------------------------
# Rule 2 - Recovery backwards
# ---------------------------------------------------------------------------


def _check_recovery_backwards(bars: list[BarLine]) -> list[DoctorProblem]:
    """Detect recovery that goes empty -> busy -> thinner instead of climbing.

    Pattern: a RECOVER bar with very low events/hats, followed by a
    RECOVER bar with 8th hats or busy flag, followed by a SETTLE bar
    that drops to quarter hats or lower event count.
    """
    problems: list[DoctorProblem] = []
    n = len(bars)

    for i in range(n - 2):
        r1 = bars[i]
        r2 = bars[i + 1]
        s = bars[i + 2]

        # r1: RECOVER, very sparse (event_count <= 2 AND hat_count == 0)
        # A bar with 3+ events (e.g. kick+snare) without hats is a
        # deliberate recovery start, not empty/pathological.
        if not (_is_recover_or_enter(r1) and r1.event_count <= 2 and r1.hat_count == 0):
            continue

        # r2: RECOVER, busy (event_count >= 8 or HATS_8THS or BUSY_BAR)
        if not (_is_recover_or_enter(r2) and (r2.event_count >= 8 or "HATS_8THS" in r2.flags or "BUSY_BAR" in r2.flags)):
            continue

        # s: SETTLE or MAINTAIN, thins back (HATS_QUARTERS or HAT_DENSITY_DROPPED or event_count < r2.event_count - 2)
        if not (_is_maintain_or_settle(s)):
            continue
        if not ("HATS_QUARTERS" in s.flags or "HAT_DENSITY_DROPPED" in s.flags or s.event_count < r2.event_count - 2):
            continue

        problems.append(DoctorProblem(
            rule="RECOVERY_BACKWARDS",
            diagnosis=f"Bars {r1.bar}-{s.bar}: recovery shape goes "
                       f"sparse/no-hats -> busy 8ths -> thinner quarter hats, "
                       f"which feels backwards.",
            suggested_fix="Make recovery climb: gentle quarter pulse -> "
                          "fuller 8th recovery -> settle. Avoid empty -> "
                          "sudden-busy -> drop.",
            confidence="high",
            affected_bars=[r1.bar, r2.bar, s.bar],
        ))
        break

    return problems


def _is_recover_or_enter(bl: BarLine) -> bool:
    return bl.section in ("RECOVER", "RECOVER_1", "RECOVER_2") or bl.rendered_intent in ("recover", "enter_soft", "enter_full", "maintain")


def _is_maintain_or_settle(bl: BarLine) -> bool:
    return bl.section in ("SETTLE", "MAINTAIN", "MAINTAIN_1", "MAINTAIN_2") or bl.rendered_intent in ("maintain", "settle")


# ---------------------------------------------------------------------------
# Rule 3 - Hat density collapse
# ---------------------------------------------------------------------------


def _check_hat_density_collapse(bars: list[BarLine]) -> list[DoctorProblem]:
    """Detect sharp hat drops after an energetic bar, excluding DROP/BAIL.

    Trigger: hat_count drops by >= 4 within 1 bar after a bar with
    event_count >= 8, and the destination section is not DROP or BAIL.
    """
    problems: list[DoctorProblem] = []
    n = len(bars)

    for i in range(n - 1):
        prev = bars[i]
        curr = bars[i + 1]

        # Skip DROP/BAIL transitions - those are intentional
        if curr.rendered_intent in ("drop", "bail", "final_bail"):
            continue

        if prev.event_count >= 8 and prev.hat_count >= 4 and curr.hat_count <= prev.hat_count - 4:
            problems.append(DoctorProblem(
                rule="HAT_DENSITY_COLLAPSE",
                diagnosis=f"Bar {prev.bar}->{curr.bar}: hat density dropped sharply "
                           f"({prev.hat_count} -> {curr.hat_count}) after an energetic bar, "
                           f"which may feel like a collapse.",
                suggested_fix="Smooth the transition: keep a small pulse or "
                              "mark it as an intentional settle.",
                confidence="medium",
                affected_bars=[prev.bar, curr.bar],
            ))
            break  # Only report the first occurrence

    return problems


# ---------------------------------------------------------------------------
# Rule 4 - Too-samey tail
# ---------------------------------------------------------------------------


def _check_too_samey_tail(bars: list[BarLine]) -> list[DoctorProblem]:
    """Detect 4+ consecutive identical maintain/settle bars.

    Two bars are "samey" if they have the same rendered_intent,
    same event_count bucket (within +/-1), and same kick/snare/hat presence.
    """
    problems: list[DoctorProblem] = []
    n = len(bars)

    start = None
    for i in range(n):
        bl = bars[i]
        if not _is_maintain_or_settle(bl):
            start = None
            continue

        if start is None:
            start = i
            continue

        prev = bars[i - 1]
        if _bars_are_samey(prev, bl):
            run_len = i - start + 1
            if run_len >= 4:
                problems.append(DoctorProblem(
                    rule="TOO_SAMEY_TAIL",
                    diagnosis=f"Bars {bars[start].bar}-{bl.bar}: "
                               f"{run_len} very similar maintain/settle bars - "
                               f"the ending/tail is very static.",
                    suggested_fix="Add small safe variation every 4 or 8 bars "
                                  "(ghost note, hat pattern shift, kick placement change).",
                    confidence="medium",
                    affected_bars=list(range(bars[start].bar, bl.bar + 1)),
                ))
                break
        else:
            start = i

    return problems


def _bars_are_samey(a: BarLine, b: BarLine) -> bool:
    """True if two bars are musically similar for samey-tail detection."""
    if a.rendered_intent != b.rendered_intent:
        return False
    # Event count within +/-1
    if abs(a.event_count - b.event_count) > 1:
        return False
    # Same kick/snare/hat presence
    if (a.kick_count > 0) != (b.kick_count > 0):
        return False
    if (a.snare_count > 0) != (b.snare_count > 0):
        return False
    if (a.hat_count > 0) != (b.hat_count > 0):
        return False
    return True


# ---------------------------------------------------------------------------
# Rule 5 - Over-busy build
# ---------------------------------------------------------------------------


def _check_over_busy_build(bars: list[BarLine]) -> list[DoctorProblem]:
    """Detect 2+ consecutive BUILD bars with BUSY_BAR/POSSIBLE_FILL flags.

    This is a warning - not automatically bad, just worth checking.
    """
    problems: list[DoctorProblem] = []
    n = len(bars)

    for i in range(n - 1):
        a = bars[i]
        b = bars[i + 1]

        if a.rendered_intent != "build" or b.rendered_intent != "build":
            continue

        a_busy = "BUSY_BAR" in a.flags or "POSSIBLE_FILL" in a.flags
        b_busy = "BUSY_BAR" in b.flags or "POSSIBLE_FILL" in b.flags

        if a_busy and b_busy:
            problems.append(DoctorProblem(
                rule="OVER_BUSY_BUILD",
                diagnosis=f"BUILD bars {a.bar}-{b.bar} are both flagged as "
                           f"BUSY_BAR/POSSIBLE_FILL - build may be too "
                           f"fill-like rather than a rising groove.",
                suggested_fix="Reduce snare/kick fills during build or make "
                             "only the final build bar busy.",
                confidence="medium",
                affected_bars=[a.bar, b.bar],
            ))
            break

    return problems


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def diagnose_bar_transcript(transcript: BarTranscript) -> DoctorReport:
    """Run all musical doctor rules on a bar transcript.

    Parameters
    ----------
    transcript : BarTranscript
        The bar transcript from :func:`drummer.bar_transcript.build_bar_transcript`.

    Returns
    -------
    DoctorReport
        Report containing any musical-shape problems found.
    """
    bars = transcript.bar_lines
    problems: list[DoctorProblem] = []

    problems.extend(_check_drop_repeated_naked_kick(bars))
    problems.extend(_check_recovery_backwards(bars))
    problems.extend(_check_hat_density_collapse(bars))
    problems.extend(_check_too_samey_tail(bars))
    problems.extend(_check_over_busy_build(bars))

    return DoctorReport(
        scenario=transcript.scenario,
        preset=transcript.preset,
        variation=transcript.variation,
        examined_bars=transcript.bars,
        problems=problems,
    )


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------


def render_doctor_report_text(report: DoctorReport) -> str:
    """Render a DoctorReport as a human-readable text report."""
    lines: list[str] = []

    lines.append(f"Doctor Report: {report.scenario}/{report.preset}/{report.variation}")

    if report.problem_count == 0:
        lines.append("Doctor: no obvious musical-shape problems detected.")
        return "\n".join(lines)

    lines.append(f"Assessment: {report.problem_count} problem(s) found in {report.examined_bars} bars.")
    lines.append("")

    for i, p in enumerate(report.problems, 1):
        lines.append(f"Problem {i}: {p.rule}  bars {_format_bar_list(p.affected_bars)}  [{p.confidence}]")
        lines.append(f"  Diagnosis: {p.diagnosis}")
        lines.append(f"  Suggested fix: {p.suggested_fix}")
        lines.append("")

    return "\n".join(lines)


def _format_bar_list(bars: list[int]) -> str:
    """Format a list of bar indices compactly: '9-10' or '11,12,13'."""
    if not bars:
        return "none"
    if len(bars) <= 2:
        return ",".join(str(b) for b in bars)
    # Check if consecutive
    if bars == list(range(bars[0], bars[-1] + 1)):
        return f"{bars[0]}-{bars[-1]}"
    return ",".join(str(b) for b in bars)


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


def render_doctor_report_json(report: DoctorReport) -> str:
    """Render a DoctorReport as a JSON string."""
    data = {
        "scenario": report.scenario,
        "preset": report.preset,
        "variation": report.variation,
        "examined_bars": report.examined_bars,
        "problem_count": report.problem_count,
        "problems": [
            {
                "rule": p.rule,
                "diagnosis": p.diagnosis,
                "suggested_fix": p.suggested_fix,
                "confidence": p.confidence,
                "affected_bars": p.affected_bars,
            }
            for p in report.problems
        ],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# Terminal summary
# ---------------------------------------------------------------------------


def print_doctor_summary(report: DoctorReport) -> None:
    """Print a compact terminal diagnosis."""
    if report.problem_count == 0:
        print("  Doctor: no obvious musical-shape problems detected.")
        return

    print(f"  ==== Musical Doctor ({report.examined_bars} bars) ====")
    for p in report.problems:
        bars_str = _format_bar_list(p.affected_bars)
        print(f"  [{p.confidence}] {p.rule} bars {bars_str}")
        # Print condensed diagnosis (first sentence)
        first_sentence = p.diagnosis.split(".")[0] + "."
        print(f"    {first_sentence}")


# ---------------------------------------------------------------------------
# File output
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def save_doctor_report(
    report: DoctorReport,
    txt_path: str,
    json_path: str,
) -> None:
    """Save doctor report as both text and JSON files."""
    _ensure_dir(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_doctor_report_text(report))
        f.write("\n")

    _ensure_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(render_doctor_report_json(report))
        f.write("\n")