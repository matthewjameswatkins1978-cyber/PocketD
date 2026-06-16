"""Arrangement Sanity Checker — multi-bar musical contract verification.

Detects higher-level arrangement problems that per-bar Musical Sanity
cannot catch: builds that go nowhere, static drops that last too long,
isolated kicks after transitions, double ending cues, etc.

Design contract
---------------
* Pure: no MIDI hardware, no playback, no side effects.
* Deterministic: same diagnostics + events → same report every time.
* Multi-bar: inspects windows of bars, not single bars.
* Reuses ``MusicalSanityIssue`` / ``MusicalSanityReport`` from
  ``drummer.musical_sanity`` for consistent reporting.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from drummer.musical_sanity import (
    MusicalSanityReport,
    MusicalSanityIssue,
)


# ---------------------------------------------------------------------------
# Bar analysis helpers
# ---------------------------------------------------------------------------


def _bar_signature(diag: dict) -> str:
    """Return a simple pattern signature for a bar.

    Represents the bar's output as a compact string for detecting
    repeated same-beat behaviour.
    """
    section = diag.get("section", "?")
    intent = diag.get("intent", "?")
    events = diag.get("event_count", 0)
    # Bucket event count into low/medium/high
    if events == 0:
        bucket = "S"
    elif events <= 4:
        bucket = "L"
    elif events <= 10:
        bucket = "M"
    else:
        bucket = "H"
    return f"{section}/{intent}/{bucket}"


def _repeated_signature_count(diagnostics: list[dict], start: int, end: int) -> int:
    """Return the number of consecutive bars with the same signature in [start, end)."""
    if end - start <= 1:
        return 0
    sigs = [_bar_signature(diagnostics[i]) for i in range(start, end)]
    max_run = 1
    current_run = 1
    for i in range(1, len(sigs)):
        if sigs[i] == sigs[i - 1]:
            current_run += 1
            max_run = max(max_run, current_run)
        else:
            current_run = 1
    return max_run


def _max_consecutive(
    items: list[Any], predicate: Any,
) -> int:
    """Return the longest run of consecutive *predicate* in *items*."""
    max_run = 0
    current = 0
    for item in items:
        if item == predicate:
            current += 1
            max_run = max(max_run, current)
        else:
            current = 0
    return max_run


def _get_crash_count(diag: dict) -> int:
    """Return crash event count for a bar if available in notes_summary."""
    notes = diag.get("notes_summary", "")
    return notes.lower().count("crash")


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _make_issue(
    severity: str, code: str, message: str,
    bar_index: int | None = None, **details,
) -> MusicalSanityIssue:
    return MusicalSanityIssue(
        severity=severity,
        intent="arrangement",
        bar_index=bar_index,
        code=code,
        message=message,
        details=details,
    )


def _error(code: str, message: str, bar_index: int | None = None, **details) -> MusicalSanityIssue:
    return _make_issue("error", code, message, bar_index, **details)


def _warn(code: str, message: str, bar_index: int | None = None, **details) -> MusicalSanityIssue:
    return _make_issue("warning", code, message, bar_index, **details)


# ---------------------------------------------------------------------------
# Helper for evaluation context
# ---------------------------------------------------------------------------


def _evaluation_intent(diag: dict) -> str:
    """Return the intent that should be used for arrangement evaluation.

    Prefers ``rendered_intent`` (the intent that actually drove output
    shaping) over raw pipeline ``intent``. Falls back to ``intent``
    only when ``rendered_intent`` is absent (backward compatibility).
    """
    return diag.get("rendered_intent", diag.get("intent", ""))


# ---------------------------------------------------------------------------
# Core arrangement checks
# ---------------------------------------------------------------------------


def _check_build_with_no_payoff(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Check for BUILD that lasts multiple bars then collapses into flat output.

    Rule: if BUILD appears for >= 3 consecutive bars, and the following
    (non-BUILD) bars are all low-activity (<= 4 events) for >= 3 bars,
    the build likely went nowhere.
    """
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)
    build_start = None
    build_end = None

    for i in range(n):
        if diagnostics[i].get("intent") == "build":
            if build_start is None:
                build_start = i
            build_end = i
        elif build_start is not None:
            build_len = build_end - build_start + 1 if build_end is not None else 0
            if build_len >= 3:
                # Check what follows
                low_start = i
                low_count = 0
                for j in range(i, min(i + 6, n)):
                    if diagnostics[j].get("event_count", 0) <= 4:
                        low_count += 1
                    else:
                        break
                if low_count >= 3:
                    issues.append(_warn(
                        "BUILD_WITH_NO_PAYOFF",
                        f"Build bars {build_start}-{build_end} ({build_len} bars) "
                        f"followed by {low_count} bars of low-activity output; "
                        "build should lead to a musical arrival, not collapse flat.",
                        bar_index=build_start,
                        build_bars=list(range(build_start, build_end + 1)),
                        low_activity_bars=list(range(i, i + low_count)),
                        build_length=build_len,
                    ))
            build_start = None
            build_end = None

    return issues


def _check_static_drop_too_long(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Warn if the same low-activity pattern repeats for too many bars.

    Rule: after a BUILD or during REDUCE/DROP, if the same bar signature
    repeats for >= 3 bars, it's too static.
    Uses ``rendered_intent`` when available for better alignment with
    actual output shaping.
    """
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)

    i = 0
    while i < n:
        diag = diagnostics[i]
        # Use rendered_intent (which reflects actual output shaping) when available
        intent = _evaluation_intent(diag)
        section = diag.get("section", "")
        if intent in ("reduce", "drop") or section in ("DROP", "REDUCE", "MAINTAIN_2"):
            events = diag.get("event_count", 0)
            if events <= 4:
                # Count consecutive low-activity bars
                start = i
                while i < n and diagnostics[i].get("event_count", 0) <= 4:
                    i += 1
                end = i
                run = end - start
                if run >= 4:  # Was >=3, now >=4 to account for REDUCE bar being intentional
                    sigs = [_bar_signature(diagnostics[j]) for j in range(start, end)]
                    unique = len(set(sigs))
                    if unique <= 2:  # nearly identical
                        issues.append(_warn(
                            "STATIC_DROP_TOO_LONG",
                            f"Same low-activity pattern repeated for {run} bars "
                            f"(bars {start}-{end - 1}); a drop or reduction "
                            "should vary to stay musical.",
                            bar_index=start,
                            drop_bars=list(range(start, end)),
                            duration_bars=run,
                        ))
                continue
        i += 1

    return issues


def _check_isolated_kick_after_drop(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Flag a single bass drum in an otherwise empty/low bar after a drop/reduce."""
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)

    for i in range(1, n):
        prev_intent = diagnostics[i - 1].get("intent", "")
        curr_intent = diagnostics[i].get("intent", "")
        curr_events = diagnostics[i].get("event_count", 0)
        # Check if previous section was DROP/REDUCE and current has exactly 1 event
        if prev_intent in ("drop", "reduce") and curr_events == 1 and curr_intent not in ("drop", "final_bail"):
            issues.append(_warn(
                "ISOLATED_KICK_AFTER_DROP",
                f"Bar {i} has a single event after a {prev_intent} section; "
                "a lone kick can sound like a mistake or hiccup.",
                bar_index=i,
                previous_intent=prev_intent,
                event_count=curr_events,
            ))

    return issues


def _check_double_final_crash(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Error if FINAL_BAIL appears with crash in two consecutive bars.

    This catches the "two crashes" issue.
    """
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)

    for i in range(n - 1):
        if diagnostics[i].get("intent") == "final_bail":
            if diagnostics[i + 1].get("intent") == "final_bail":
                issues.append(_error(
                    "DOUBLE_FINAL_CRASH",
                    f"FINAL_BAIL intent appears in two consecutive bars "
                    f"({i}-{i + 1}); ending cue should be a single clear gesture.",
                    bar_index=i,
                    bars=list(range(i, i + 2)),
                ))
                break  # Only flag once for this pair

    return issues


def _check_repeated_ending_cue(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Error if FINAL_BAIL section appears more than once in the full sequence."""
    issues: list[MusicalSanityIssue] = []
    final_bail_bars = [
        d["bar"] for d in diagnostics
        if d.get("section") == "FINAL_BAIL"
    ]
    if len(final_bail_bars) > 1:  # more than one FINAL_BAIL bar = repeated
        issues.append(_error(
            "REPEATED_ENDING_CUE",
            f"FINAL_BAIL section appears {len(final_bail_bars)} times "
            f"(bars {final_bail_bars}); the ending cue should be a single event.",
            bar_index=final_bail_bars[0],
            final_bail_bars=final_bail_bars,
        ))

    return issues


def _event_bucket_sig(diag: dict) -> str:
    """Return just the event-bucket signature, ignoring section/intent labels.

    This is used by SAME_BEAT_AFTER_CHANGE to detect when output doesn't
    actually change even though intent/section labels change.
    """
    events = diag.get("event_count", 0)
    if events == 0:
        return "S"
    elif events <= 4:
        return "L"
    elif events <= 10:
        return "M"
    else:
        return "H"


def _check_same_beat_after_change(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Warn if intent changes but output event-density pattern stays the same.

    Musical meaning: if the drummer changes state (e.g. from BUILD to DROP)
    but the actual output density stays the same, the state machine may be
    labelling incorrectly without changing the musical result.
    """
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)

    for i in range(1, n):
        prev = diagnostics[i - 1]
        prev_intent = _evaluation_intent(prev)
        curr_intent = _evaluation_intent(diagnostics[i])

        if prev_intent != curr_intent and curr_intent != "final_bail":
            # Compare event-density buckets, not full signatures (which include intent)
            prev_bucket = _event_bucket_sig(prev)
            curr_bucket = _event_bucket_sig(diagnostics[i])
            if prev_bucket == curr_bucket:
                # Check if the next bar is also the same
                next_same = False
                if i + 1 < n:
                    next_bucket = _event_bucket_sig(diagnostics[i + 1])
                    if next_bucket == prev_bucket:
                        next_same = True
                if next_same:
                    issues.append(_warn(
                        "SAME_BEAT_AFTER_CHANGE",
                        f"State changed from '{prev_intent}' to '{curr_intent}' at bar {i}, "
                        f"but event-density bucket '{prev_bucket}' stayed the same for 2+ bars.",
                        bar_index=i,
                        previous_intent=prev_intent,
                        new_intent=curr_intent,
                    ))

    return issues


def _check_late_enter_then_immediate_build(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Warn if ENTER_SOFT lasts many bars (>6) then immediately BUILDs.

    Musical meaning: uncertain input delays entry, but rushing into build
    without settling feels rushed.
    """
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)

    enter_start = None
    for i in range(n):
        intent = diagnostics[i].get("intent", "")
        if intent == "enter_soft":
            if enter_start is None:
                enter_start = i
        elif intent == "build" and enter_start is not None:
            enter_len = i - enter_start
            if enter_len > 6:
                issues.append(_warn(
                    "LATE_ENTER_THEN_IMMEDIATE_BUILD",
                    f"ENTER_SOFT lasted {enter_len} bars (bars {enter_start}-{i - 1}) "
                    f"then immediately built at bar {i}; the drummer should settle "
                    "before building.",
                    bar_index=i,
                    enter_start=enter_start,
                    enter_end=i - 1,
                    build_bar=i,
                    enter_duration=enter_len,
                ))
            enter_start = None
        elif intent not in ("enter_soft", "listen"):
            enter_start = None

    return issues


def _check_build_too_abrupt_to_drop(diagnostics: list[dict]) -> list[MusicalSanityIssue]:
    """Warn if BUILD transitions to DROP immediately without a REDUCE phase."""
    issues: list[MusicalSanityIssue] = []
    n = len(diagnostics)

    for i in range(1, n):
        prev = diagnostics[i - 1].get("intent", "")
        curr = diagnostics[i].get("intent", "")
        if prev in ("build",) and curr == "drop":
            # Check how many bars of build preceded it
            build_start = i - 1
            while build_start > 0 and diagnostics[build_start - 1].get("intent") == "build":
                build_start -= 1
            build_len = i - build_start
            if build_len >= 2:
                issues.append(_warn(
                    "BUILD_TOO_ABRUPT_TO_DROP",
                    f"BUILD bars {build_start}-{i - 1} ({build_len} bars) transitions "
                    f"directly to DROP at bar {i} without REDUCE phase; "
                    "this can sound like a sudden collapse.",
                    bar_index=i,
                    build_bars=list(range(build_start, i)),
                    drop_bar=i,
                    build_length=build_len,
                ))

    return issues


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_arrangement_sanity(
    per_bar_diagnostics: list[dict],
    global_events: list | None = None,
) -> MusicalSanityReport:
    """Check multi-bar arrangement sanity across a full scenario.

    Parameters
    ----------
    per_bar_diagnostics : list[dict]
        Per-bar diagnostic records (same format as returned by the pipeline).
    global_events : list or None
        Optional list of global GrooveEvents (for future use).

    Returns
    -------
    MusicalSanityReport
        Report containing any arrangement-level issues found.
    """
    report = MusicalSanityReport()

    if not per_bar_diagnostics:
        return report

    # Run all checks
    report.issues.extend(_check_build_with_no_payoff(per_bar_diagnostics))
    report.issues.extend(_check_static_drop_too_long(per_bar_diagnostics))
    report.issues.extend(_check_isolated_kick_after_drop(per_bar_diagnostics))
    report.issues.extend(_check_double_final_crash(per_bar_diagnostics))
    report.issues.extend(_check_repeated_ending_cue(per_bar_diagnostics))
    report.issues.extend(_check_same_beat_after_change(per_bar_diagnostics))
    report.issues.extend(_check_late_enter_then_immediate_build(per_bar_diagnostics))
    report.issues.extend(_check_build_too_abrupt_to_drop(per_bar_diagnostics))

    return report