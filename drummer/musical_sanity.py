"""Musical Sanity Checker — deterministic linting for drum event output.

Catches obvious musical contract violations (crash in ENTER_SOFT,
silent DROP, etc.) before human ear testing.  This is NOT AI; it is
deterministic musical linting.

Design contract
---------------
* Pure: no MIDI hardware, no playback, no side effects.
* Deterministic: same events + intent → same report every time.
* Instrument-aware: uses internal note constants (no dependency on
  ``pipeline_midi`` or ``midi_out``).
* Intent-agnostic helpers: per-intent checkers dispatch from a single
  public entry point.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from drummer.behaviour import BehaviourIntent
from drummer.feel import GrooveEvent

# ---------------------------------------------------------------------------
# Self-contained instrument note numbers (GM standard)
# ---------------------------------------------------------------------------
# We intentionally do NOT import from drummer.pipeline_midi to avoid
# pulling in MIDI backend dependencies.

_KICK_NOTE = 36
_SNARE_NOTE = 38
_HI_HAT_NOTE = 42
_CRASH_NOTE = 49
# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class MusicalSanityIssue:
    """One musical sanity problem detected in a bar's events."""

    severity: str  # "warning" | "error"
    intent: str
    bar_index: int | None
    code: str
    message: str
    details: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "severity": self.severity,
            "intent": self.intent,
            "bar_index": self.bar_index,
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


@dataclass
class MusicalSanityReport:
    """Aggregated musical sanity report for a bar or scenario."""

    issues: list[MusicalSanityIssue] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Summary properties
    # ------------------------------------------------------------------

    @property
    def passed(self) -> bool:
        """True if there are zero errors (warnings alone do not fail)."""
        return self.error_count == 0

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "error")

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == "warning")

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "error_count": self.error_count,
            "warning_count": self.warning_count,
            "issues": [i.to_dict() for i in self.issues],
        }

    def format_text(self) -> str:
        """Return a compact human-readable report."""
        if not self.issues:
            return "No musical sanity issues."

        lines: list[str] = []
        lines.append(
            f"Musical Sanity: {'PASSED' if self.passed else 'FAILED'} "
            f"({self.error_count} errors, {self.warning_count} warnings)"
        )
        for issue in self.issues:
            prefix = "ERROR" if issue.severity == "error" else "WARN "
            bar = f"bar {issue.bar_index}" if issue.bar_index is not None else ""
            lines.append(
                f"  [{prefix}] {issue.intent} {bar}: {issue.message}"
            )
        return "\n".join(lines)


def _merge_reports(reports: list[MusicalSanityReport]) -> MusicalSanityReport:
    """Combine multiple per-bar reports into one."""
    merged = MusicalSanityReport()
    for r in reports:
        merged.issues.extend(r.issues)
    return merged


# ---------------------------------------------------------------------------
# Event-analysis helpers (no MIDI imports)
# ---------------------------------------------------------------------------


def _has_note(events: list[GrooveEvent], note: int) -> bool:
    """True if any event maps to *note*."""
    return any(
        _resolve_note(e) == note
        for e in events
    )


def _has_crash(events: list[GrooveEvent]) -> bool:
    return _has_note(events, _CRASH_NOTE)


def _has_final_bail_pattern(events: list[GrooveEvent]) -> bool:
    """True if events look like kick+crash on beat 1 — the FINAL_BAIL gesture.

    Checks: exactly 2 events, one kick (36), one crash (49), both at grid 0.
    """
    if len(events) != 2:
        return False
    resolved = [_resolve_note(e) for e in events]
    notes = sorted(n for n in resolved if n is not None)
    if notes != [_KICK_NOTE, _CRASH_NOTE]:
        return False
    # Both must be at grid_position 0 (beat 1).
    # Events may carry global positions (bar_index * 16), so normalise.
    return all(e.grid_position % 16 == 0 for e in events)


def _is_kick_only(events: list[GrooveEvent]) -> bool:
    """True if every event is kick (note 36)."""
    if not events:
        return False
    return all(_resolve_note(e) == _KICK_NOTE for e in events)


def _max_velocity(events: list[GrooveEvent]) -> int:
    if not events:
        return 0
    return max(e.velocity for e in events)


def _event_count(events: list[GrooveEvent]) -> int:
    return len(events)


def _has_loud_announcement(events: list[GrooveEvent]) -> bool:
    """True if there is a single loud event on beat 1 that resembles a stomp/announcement.

    Pattern: only one event (or only kick with vel > 90) at grid 0.
    """
    if not events:
        return False
    # Check for single loud kick at beat 1 (normalise for global schedules)
    beat1 = [e for e in events if e.grid_position % 16 == 0]
    if len(beat1) == 1:
        e = beat1[0]
        if _resolve_note(e) == _KICK_NOTE and e.velocity > 90:
            return True
    return False


def _note_names(events: list[GrooveEvent]) -> list[str]:
    """Return human-readable instrument names for each event."""
    return [_instrument_name(_resolve_note(e)) for e in events]


def _instrument_name(note: int | None) -> str:
    if note is None:
        return "?"
    return {
        _KICK_NOTE: "kick",
        _SNARE_NOTE: "snare",
        _HI_HAT_NOTE: "hi_hat",
        _CRASH_NOTE: "crash",
    }.get(note, f"note_{note}")


def _resolve_note(evt: GrooveEvent) -> int | None:
    """Map a GrooveEvent instrument string to a GM note number.

    Uses internal mapping only — no hardware dependency.
    """
    inst = evt.instrument.lower()
    # Precise matches
    if inst == "kick":
        return _KICK_NOTE
    if inst in ("snare", "rimshot"):
        return _SNARE_NOTE
    if inst in ("hi_hat", "closed_hat", "open_hat", "ride"):
        return _HI_HAT_NOTE  # grouped as hat family for crash detection
    if inst == "crash":
        return _CRASH_NOTE
    # Generic instrument family fallback
    if "kick" in inst:
        return _KICK_NOTE
    if "snare" in inst:
        return _SNARE_NOTE
    if "crash" in inst:
        return _CRASH_NOTE
    if "hat" in inst or "ride" in inst:
        return _HI_HAT_NOTE
    return None


# ---------------------------------------------------------------------------
# Issue factory helpers
# ---------------------------------------------------------------------------


def _issue(
    severity: str, intent: str, code: str, message: str,
    bar_index: int | None = None, **details,
) -> MusicalSanityIssue:
    return MusicalSanityIssue(
        severity=severity,
        intent=intent,
        bar_index=bar_index,
        code=code,
        message=message,
        details=details,
    )


def _error(intent: str, code: str, message: str,
           bar_index: int | None = None, **details) -> MusicalSanityIssue:
    return _issue("error", intent, code, message, bar_index, **details)


def _warn(intent: str, code: str, message: str,
          bar_index: int | None = None, **details) -> MusicalSanityIssue:
    return _issue("warning", intent, code, message, bar_index, **details)


# ---------------------------------------------------------------------------
# Per-intent checkers
# ---------------------------------------------------------------------------


def _check_enter_soft(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    vel = _max_velocity(events)
    count = _event_count(events)

    # ENTER_SOFT must not contain crash
    if _has_crash(events):
        report.issues.append(_error(intent, "ENTER_SOFT_CRASH",
            "ENTER_SOFT contains crash; a soft entry should not use crash cymbal.",
            bar_index, velocity=vel))

    # ENTER_SOFT must not resemble FINAL_BAIL pattern
    if _has_final_bail_pattern(events):
        report.issues.append(_error(intent, "ENTER_SOFT_RESEMBLES_FINAL_BAIL",
            "ENTER_SOFT resembles FINAL_BAIL; this sounds like an ending cue, not an entry.",
            bar_index))

    # ENTER_SOFT must not be kick-only at high velocity
    if _is_kick_only(events) and vel > 90:
        report.issues.append(_error(intent, "ENTER_SOFT_ISOLATED_KICK",
            f"ENTER_SOFT is kick-only at velocity {vel}; "
            "this sounds like a stomp rather than joining.",
            bar_index, velocity=vel))

    # ENTER_SOFT max velocity must be <= 100
    if vel > 100:
        report.issues.append(_error(intent, "ENTER_SOFT_TOO_LOUD",
            f"ENTER_SOFT max velocity {vel} exceeds 100; "
            "a soft entry should not be aggressive.",
            bar_index, velocity=vel))

    # ENTER_SOFT must not open with a loud beat-1 announcement
    if _has_loud_announcement(events):
        report.issues.append(_error(intent, "ENTER_SOFT_LOUD_ANNOUNCEMENT",
            "ENTER_SOFT opens with a loud beat-1 hit; "
            "sounds like an announcement, not joining.",
            bar_index))

    # Event count: warning at >14, error at >20
    if count > 20:
        report.issues.append(_error(intent, "ENTER_SOFT_TOO_BUSY",
            f"ENTER_SOFT has {count} events; a soft entry should not be explosive.",
            bar_index, event_count=count))
    elif count > 14:
        report.issues.append(_warn(intent, "ENTER_SOFT_TOO_BUSY",
            f"ENTER_SOFT has {count} events; a soft entry is usually sparser than this.",
            bar_index, event_count=count))

    # Empty ENTER_SOFT is questionable but not an error — pipeline may just
    # have started entering but hasn't produced events yet.
    if count == 0:
        report.issues.append(_warn(intent, "ENTER_SOFT_SILENT",
            "ENTER_SOFT produced zero events; entry may be too hesitant.",
            bar_index))

    return report


def _check_drop(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    # DROP must have > 0 events
    if count == 0:
        report.issues.append(_error(intent, "DROP_SILENT",
            "DROP produced zero events; DROP should stay alive (sparse), BAIL should stop.",
            bar_index))

    # DROP must not contain crash
    if _has_crash(events):
        report.issues.append(_error(intent, "DROP_CRASH",
            "DROP contains crash; a pulled-back section should not use crash cymbal.",
            bar_index))

    # DROP must not resemble FINAL_BAIL
    if _has_final_bail_pattern(events):
        report.issues.append(_error(intent, "DROP_RESEMBLES_FINAL_BAIL",
            "DROP resembles FINAL_BAIL; this sounds like an ending, not a pullback.",
            bar_index))

    # DROP should be very sparse (warning if > 4 events)
    if count > 4:
        report.issues.append(_warn(intent, "DROP_TOO_BUSY",
            f"DROP has {count} events; a pulled-back section is usually sparser.",
            bar_index, event_count=count))

    return report


def _check_bail(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    if count != 0:
        report.issues.append(_error(intent, "BAIL_NOT_SILENT",
            f"BAIL must produce exactly 0 events, got {count}. BAIL means stop.",
            bar_index, event_count=count))

    return report


def _check_final_bail(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    # FINAL_BAIL must have exactly 2 events
    if count != 2:
        report.issues.append(_error(intent, "FINAL_BAIL_WRONG_COUNT",
            f"FINAL_BAIL has {count} events, expected exactly 2 (kick + crash on beat 1).",
            bar_index, event_count=count))
        return report

    # Must be kick (36) + crash (49)
    notes = sorted(_resolve_note(e) for e in events)
    if notes != [_KICK_NOTE, _CRASH_NOTE]:
        report.issues.append(_error(intent, "FINAL_BAIL_WRONG_NOTES",
            f"FINAL_BAIL notes: {_note_names(events)}, expected kick + crash.",
            bar_index, notes=notes))

    # Both must be at grid_position 0 (beat 1).
    # Normalise to bar-local positions for global schedules.
    beat1 = all(e.grid_position % 16 == 0 for e in events)
    if not beat1:
        positions = [e.grid_position for e in events]
        report.issues.append(_error(intent, "FINAL_BAIL_BEAT1",
            f"FINAL_BAIL events at grid positions {positions}, expected both on beat 1.",
            bar_index, grid_positions=positions))

    return report


def _check_anchor(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    # ANCHOR must not contain crash
    if _has_crash(events):
        report.issues.append(_error(intent, "ANCHOR_CRASH",
            "ANCHOR contains crash; anchoring should not use crash cymbal.",
            bar_index))

    # ANCHOR must have some events (should not be silent)
    if count == 0:
        report.issues.append(_warn(intent, "ANCHOR_SILENT",
            "ANCHOR produced zero events; anchoring usually has a pulse.",
            bar_index))

    # ANCHOR should not be too busy (>12 events is unusual)
    if count > 12:
        report.issues.append(_warn(intent, "ANCHOR_TOO_BUSY",
            f"ANCHOR has {count} events; anchoring is usually simpler than this.",
            bar_index, event_count=count))

    return report


def _check_build(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    # BUILD must have > 0 events
    if count == 0:
        report.issues.append(_warn(intent, "BUILD_SILENT",
            "BUILD produced zero events; building should not be silent.",
            bar_index))

    # BUILD opening with a loud isolated crash from silence is concerning
    if _has_crash(events) and count <= 3:
        report.issues.append(_warn(intent, "BUILD_CRASH_EARLY",
            "BUILD contains crash with very few events; crash is usually saved for peaks.",
            bar_index, event_count=count))

    return report


def _check_maintain(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    if count == 0:
        report.issues.append(_warn(intent, "MAINTAIN_SILENT",
            "MAINTAIN produced zero events; groove should be present.",
            bar_index))

    if _has_crash(events):
        report.issues.append(_warn(intent, "MAINTAIN_CRASH",
            "MAINTAIN contains crash; crash is usually reserved for transitions or peaks.",
            bar_index))

    return report


def _check_reduce(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()

    # REDUCE with crash is odd — reduction shouldn't feature crash
    if _has_crash(events):
        report.issues.append(_warn(intent, "REDUCE_CRASH",
            "REDUCE contains crash; reducing intensity should not introduce crash.",
            bar_index))

    return report


def _check_listen(
    intent: str, events: list[GrooveEvent], bar_index: int | None,
) -> MusicalSanityReport:
    report = MusicalSanityReport()
    count = _event_count(events)

    # LISTEN should be silent (or very nearly so)
    if count > 0:
        report.issues.append(_warn(intent, "LISTEN_NOT_SILENT",
            f"LISTEN produced {count} events; listening phase should be silent.",
            bar_index, event_count=count))

    return report


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH: dict[BehaviourIntent, callable] = {
    BehaviourIntent.ENTER_SOFT: _check_enter_soft,
    BehaviourIntent.ENTER_FULL: _check_enter_soft,  # same rules as ENTER_SOFT
    BehaviourIntent.DROP: _check_drop,
    BehaviourIntent.BAIL: _check_bail,
    BehaviourIntent.FINAL_BAIL: _check_final_bail,
    BehaviourIntent.ANCHOR: _check_anchor,
    BehaviourIntent.BUILD: _check_build,
    BehaviourIntent.MAINTAIN: _check_maintain,
    BehaviourIntent.REDUCE: _check_reduce,
    BehaviourIntent.LISTEN: _check_listen,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def check_musical_sanity(
    intent: str,
    events: list[GrooveEvent],
    bar_index: int | None = None,
    context: dict | None = None,
) -> MusicalSanityReport:
    """Run musical sanity checks on *events* for a given *intent*.

    Parameters
    ----------
    intent : str
        ``BehaviourIntent`` value (e.g. ``"enter_soft"``, ``"drop"``).
    events : list[GrooveEvent]
        The generated groove events for this bar.
    bar_index : int | None
        Optional bar index for diagnostic messages.
    context : dict | None
        Reserved for future use (e.g. previous-bar state, tempo).

    Returns
    -------
    MusicalSanityReport
        Report containing any issues found.
    """
    # Resolve BehaviourIntent from string
    try:
        bi = BehaviourIntent(intent)
    except (ValueError, TypeError):
        # Unknown intent — no checks, return empty report
        return MusicalSanityReport()

    checker = _DISPATCH.get(bi)
    if checker is None:
        # No checker registered for this intent (FILL, CRASH, etc.) — pass
        return MusicalSanityReport()

    return checker(intent, events, bar_index)


def check_scenario_sanity(
    bar_events: dict[int, list[GrooveEvent]],
    per_bar_diagnostics: list[dict],
) -> MusicalSanityReport:
    """Check musical sanity across a full scenario.

    Parameters
    ----------
    bar_events : dict[int, list[GrooveEvent]]
        Per-bar index → list of GrooveEvents.
    per_bar_diagnostics : list[dict]
        Per-bar diagnostic records containing ``intent`` and ``bar`` keys.

    Returns
    -------
    MusicalSanityReport
        Combined report for the entire scenario.
    """
    reports: list[MusicalSanityReport] = []

    for diag in per_bar_diagnostics:
        bar = diag.get("bar", 0)
        intent = diag.get("intent", "listen")
        events = bar_events.get(bar, [])

        report = check_musical_sanity(intent, events, bar_index=bar)
        reports.append(report)

    return _merge_reports(reports)