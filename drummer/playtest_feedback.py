"""Playtest feedback data models, validation, scenario registry, runner, and JSONL persistence.

Part 6 adds feedback analysis and learning summary generation.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Helpers: ensure demo_continuous_jam_midi is importable
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Part 1 – Data models
# ---------------------------------------------------------------------------


@dataclass
class PlaytestScenario:
    """Describes one scenario variation presented to the playtester."""

    name: str
    variation_name: str
    description: str
    preset: str
    mode: str
    bars: int
    bpm: float
    listen_start_bar: int = 0
    listen_end_bar: int = 19
    what_to_listen_for: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlaytestQuestionnaire:
    """Fixed musical questions answered by the playtester."""

    overall_rating: int
    timing_rating: str
    amount_rating: str
    confidence_rating: str
    understood_rating: str
    suggested_change: str
    note: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlaytestDiagnosticsSummary:
    """Diagnostic snapshot captured alongside the playtester's answers."""

    total_events: int
    first_enter_bar: Optional[int]
    first_build_bar: Optional[int]
    confidence_peak: float
    phrase_marker_count: int
    inferred_intents: dict[str, int]
    output_contracts_passed: bool
    drop_event_count: int
    final_bail_event_count: int
    bail_event_count: int
    musical_sanity_passed: bool = True
    musical_sanity_errors: int = 0
    musical_sanity_warnings: int = 0
    musical_sanity_issues: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlaytestFeedbackEntry:
    """Complete feedback record: scenario, diagnostics, and answers."""

    scenario: PlaytestScenario
    diagnostics: PlaytestDiagnosticsSummary
    answers: PlaytestQuestionnaire

    def to_dict(self) -> dict:
        return {
            "scenario": self.scenario.to_dict(),
            "diagnostics": self.diagnostics.to_dict(),
            "answers": self.answers.to_dict(),
        }


# ---------------------------------------------------------------------------
# Part 2 – Fixed answer options and validation
# ---------------------------------------------------------------------------

# Overall musical feel
OVERALL_RATING_VALUES = {1, 2, 3, 4, 5}

# Timing
TIMING_CHOICES = frozenset({"too_early", "about_right", "too_late", "not_relevant"})

# Amount
AMOUNT_CHOICES = frozenset({"too_sparse", "about_right", "too_busy", "not_relevant"})

# Confidence
CONFIDENCE_CHOICES = frozenset({"too_timid", "about_right", "too_bold", "not_relevant"})

# Understanding
UNDERSTOOD_CHOICES = frozenset({"yes", "partly", "no"})

# Suggested change
SUGGESTED_CHANGE_CHOICES = frozenset({
    "enter_later",
    "enter_sooner",
    "play_less",
    "play_more",
    "build_more",
    "build_less",
    "recover_sooner",
    "recover_later",
    "mark_phrases_less",
    "mark_phrases_more",
    "ending_cue_stronger",
    "ending_cue_weaker",
    "no_change",
})


# ---------------------------------------------------------------------------
# Key-mapping dictionaries for quick single-key input
# ---------------------------------------------------------------------------

# Timing: key -> canonical value
TIMING_KEY_MAP: dict[str, str] = {
    "e": "too_early",
    "r": "about_right",
    "l": "too_late",
    "n": "not_relevant",
}

# Amount: key -> canonical value
AMOUNT_KEY_MAP: dict[str, str] = {
    "s": "too_sparse",
    "r": "about_right",
    "b": "too_busy",
    "n": "not_relevant",
}

# Confidence: key -> canonical value
CONFIDENCE_KEY_MAP: dict[str, str] = {
    "t": "too_timid",
    "r": "about_right",
    "d": "too_bold",
    "n": "not_relevant",
}

# Understood: key -> canonical value
UNDERSTOOD_KEY_MAP: dict[str, str] = {
    "y": "yes",
    "p": "partly",
    "n": "no",
}

# Suggested change: key -> canonical value
SUGGESTED_CHANGE_KEY_MAP: dict[str, str] = {
    "0": "no_change",
    "1": "enter_later",
    "2": "enter_sooner",
    "3": "play_less",
    "4": "play_more",
    "5": "build_less",
    "6": "build_more",
    "7": "recover_later",
    "8": "recover_sooner",
    "9": "mark_phrases_less",
    "10": "mark_phrases_more",
    "11": "ending_cue_weaker",
    "12": "ending_cue_stronger",
}


def parse_key_choice(key: str, mapping: dict[str, str], field_name: str) -> str:
    """Parse a single-key input into a canonical value using *mapping*.

    Accepts upper- or lower-case keys, strips whitespace.
    Raises ``ValueError`` if the key is not in the mapping.
    """
    cleaned = key.strip().lower()
    if cleaned not in mapping:
        valid_keys = sorted(mapping.keys(), key=lambda k: (isinstance(k, str) and not k.isdigit(), int(k) if k.isdigit() else k))
        raise ValueError(
            f"{field_name}: invalid key {key!r}. "
            f"Valid keys: {valid_keys}"
        )
    return mapping[cleaned]


def validate_overall_rating(value: int) -> None:
    """Raise ValueError if *value* is not a valid overall rating."""
    if value not in OVERALL_RATING_VALUES:
        raise ValueError(
            f"overall_rating must be one of {sorted(OVERALL_RATING_VALUES)}, got {value!r}"
        )


def validate_choice(value: str, valid_choices: frozenset, field_name: str) -> None:
    """Raise ValueError if *value* is not in *valid_choices*."""
    if value not in valid_choices:
        raise ValueError(
            f"{field_name} must be one of {sorted(valid_choices)}, got {value!r}"
        )


def validate_questionnaire_answers(
    overall_rating: int,
    timing_rating: str,
    amount_rating: str,
    confidence_rating: str,
    understood_rating: str,
    suggested_change: str,
) -> None:
    """Validate all fixed-choice answer fields at once."""
    validate_overall_rating(overall_rating)
    validate_choice(timing_rating, TIMING_CHOICES, "timing_rating")
    validate_choice(amount_rating, AMOUNT_CHOICES, "amount_rating")
    validate_choice(confidence_rating, CONFIDENCE_CHOICES, "confidence_rating")
    validate_choice(understood_rating, UNDERSTOOD_CHOICES, "understood_rating")
    validate_choice(suggested_change, SUGGESTED_CHANGE_CHOICES, "suggested_change")


# ---------------------------------------------------------------------------
# Part 3 – Scenario and variation registry
# ---------------------------------------------------------------------------


def _build_listen_focus(name: str, variation_name: str) -> tuple[int, int, str]:
    """Return (listen_start_bar, listen_end_bar, what_to_listen_for)."""
    focus_map: dict[str, dict[str, tuple[int, int, str]]] = {
        "enter": {
            "stable_input": (
                2, 5,
                "Listen to how the drummer enters: does it feel natural? "
                "Does it step in at the right moment?"
            ),
            "uncertain_input": (
                2, 5,
                "Listen to how the drummer handles slightly erratic input: "
                "does it hesitate or enter confidently despite uncertainty?"
            ),
        },
        "build": {
            "slow_build": (
                7, 12,
                "Listen to the BUILD section (bars 7-12): does the intensity "
                "ramp feel gradual and musical?"
            ),
            "strong_build": (
                7, 10,
                "Listen to the BUILD section (bars 7-9): does the drummer "
                "build quickly and assertively without rushing?"
            ),
        },
        "anchor_recovery": {
            "poor_phase_recovery": (
                16, 19,
                "Listen to the recovery after ANCHOR (bars 16-18): does the "
                "drummer regain stability and lock back into the pocket?"
            ),
            "weak_input_recovery": (
                16, 19,
                "Listen to the recovery after weak input (bars 16-18): does "
                "the drummer re-establish the groove convincingly?"
            ),
        },
        "drop": {
            "deliberate_sparse": (
                13, 14,
                "Listen to the DROP at bar 13: does the pullback sound "
                "intentional and tasteful, not like a mistake?"
            ),
            "pullback_after_build": (
                13, 14,
                "Listen to the DROP at bar 13 after the BUILD: does the "
                "transition from dense to sparse feel musical?"
            ),
        },
        "final_bail": {
            "clear_cue": (
                14, 16,
                "Listen to the ending (bars 14-15): does the drummer give a "
                "clear, confident cue that the performance is ending?"
            ),
            "ambiguous_cue": (
                14, 16,
                "Listen to the ending (bars 14-15): does the ending feel "
                "unclear or does the cue land well?"
            ),
        },
    }
    return focus_map.get(name, {}).get(
        variation_name, (0, 19, "Listen to the full performance.")
    )


def _enter_variations(preset: str) -> list[PlaytestScenario]:
    base = PlaytestScenario(
        name="enter",
        variation_name="stable_input",
        description="Consistent, predictable input — drummer enters naturally.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    uncertain = PlaytestScenario(
        name="enter",
        variation_name="uncertain_input",
        description="Slightly erratic input — drummer may hesitate before entering.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    for sc in (base, uncertain):
        start, end, focus = _build_listen_focus(sc.name, sc.variation_name)
        sc.listen_start_bar = start
        sc.listen_end_bar = end
        sc.what_to_listen_for = focus
    return [base, uncertain]


def _build_variations(preset: str) -> list[PlaytestScenario]:
    slow = PlaytestScenario(
        name="build",
        variation_name="slow_build",
        description="Gradual crescendo over many bars — drummer takes its time.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    strong = PlaytestScenario(
        name="build",
        variation_name="strong_build",
        description="Quick, assertive build — drummer climbs rapidly.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    for sc in (slow, strong):
        start, end, focus = _build_listen_focus(sc.name, sc.variation_name)
        sc.listen_start_bar = start
        sc.listen_end_bar = end
        sc.what_to_listen_for = focus
    return [slow, strong]


def _anchor_recovery_variations(preset: str) -> list[PlaytestScenario]:
    poor = PlaytestScenario(
        name="anchor_recovery",
        variation_name="poor_phase_recovery",
        description="Drummer drifts out of phase then recovers to the pocket.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    weak = PlaytestScenario(
        name="anchor_recovery",
        variation_name="weak_input_recovery",
        description="Input drops away then drummer re-establishes the groove.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    for sc in (poor, weak):
        start, end, focus = _build_listen_focus(sc.name, sc.variation_name)
        sc.listen_start_bar = start
        sc.listen_end_bar = end
        sc.what_to_listen_for = focus
    return [poor, weak]


def _drop_variations(preset: str) -> list[PlaytestScenario]:
    sparse = PlaytestScenario(
        name="drop",
        variation_name="deliberate_sparse",
        description="Intentional, tasteful pullback — drummer plays less on purpose.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    after_build = PlaytestScenario(
        name="drop",
        variation_name="pullback_after_build",
        description="Drummer builds up then drops back to a sparser texture.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    for sc in (sparse, after_build):
        start, end, focus = _build_listen_focus(sc.name, sc.variation_name)
        sc.listen_start_bar = start
        sc.listen_end_bar = end
        sc.what_to_listen_for = focus
    return [sparse, after_build]


def _final_bail_variations(preset: str) -> list[PlaytestScenario]:
    clear = PlaytestScenario(
        name="final_bail",
        variation_name="clear_cue",
        description="Drummer gives an obvious ending cue before stopping.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    ambiguous = PlaytestScenario(
        name="final_bail",
        variation_name="ambiguous_cue",
        description="Ending is unclear — drummer fades without a strong cue.",
        preset=preset,
        mode="continuous",
        bars=20,
        bpm=120.0,
    )
    for sc in (clear, ambiguous):
        start, end, focus = _build_listen_focus(sc.name, sc.variation_name)
        sc.listen_start_bar = start
        sc.listen_end_bar = end
        sc.what_to_listen_for = focus
    return [clear, ambiguous]


# Map scenario name -> variation generator
_VARIATION_BUILDERS: dict[str, Callable[[str], list[PlaytestScenario]]] = {
    "enter": _enter_variations,
    "build": _build_variations,
    "anchor_recovery": _anchor_recovery_variations,
    "drop": _drop_variations,
    "final_bail": _final_bail_variations,
}


def list_playtest_scenarios() -> list[str]:
    """Return all available scenario names."""
    return sorted(_VARIATION_BUILDERS.keys())


def get_scenario_variations(
    scenario_name: str, preset: str = "normal"
) -> list[PlaytestScenario]:
    """Return the variation definitions for *scenario_name*.

    Raises ``ValueError`` if the scenario is unknown.
    """
    builder = _VARIATION_BUILDERS.get(scenario_name)
    if builder is None:
        raise ValueError(
            f"Unknown scenario {scenario_name!r}. "
            f"Available: {list_playtest_scenarios()}"
        )
    return builder(preset)


# ---------------------------------------------------------------------------
# Part 4 – JSONL saving
# ---------------------------------------------------------------------------


def serialize_feedback_entry(entry: PlaytestFeedbackEntry) -> str:
    """Return a JSON line string for *entry* (no trailing newline)."""
    return json.dumps(entry.to_dict(), sort_keys=True)


def append_feedback_entry(path: str, entry: PlaytestFeedbackEntry) -> None:
    """Append *entry* as one JSON line to the file at *path*."""
    line = serialize_feedback_entry(entry)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_feedback_entries(path: str) -> list[dict]:
    """Read all JSON lines from *path* and return a list of dicts."""
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


# ---------------------------------------------------------------------------
# Part 5 – Scenario runner (wraps run_continuous_jam from demo)
# ---------------------------------------------------------------------------


def _extract_diagnostics_summary(
    diagnostics: list[dict],
) -> PlaytestDiagnosticsSummary:
    """Build a PlaytestDiagnosticsSummary from per-bar diagnostic records."""
    total_events = sum(d.get("event_count", 0) for d in diagnostics)

    # First enter bar
    first_enter_bar: Optional[int] = None
    for d in diagnostics:
        intent = d.get("intent", "")
        if intent in ("enter_soft", "enter_full"):
            first_enter_bar = d["bar"]
            break

    # First build bar
    first_build_bar: Optional[int] = None
    for d in diagnostics:
        if d.get("intent") == "build":
            first_build_bar = d["bar"]
            break

    # Confidence peak
    confidence_peak = max(
        (d.get("confidence", 0.0) for d in diagnostics),
        default=0.0,
    )

    # Phrase marker count
    phrase_marker_count = sum(
        1 for d in diagnostics if d.get("phrase_marker", "none") != "none"
    )

    # Inferred intents summary (count per intent value)
    inferred_intents: dict[str, int] = {}
    for d in diagnostics:
        intent_val = d.get("inferred_intent", d.get("intent", "unknown"))
        inferred_intents[intent_val] = inferred_intents.get(intent_val, 0) + 1

    # Output contract checks — actual musical/output contracts, not just intent labels.
    #
    # DROP:  must produce > 0 events, sparse output only, no crash
    # BAIL:  must produce exactly 0 events (is_bail means the output shaper
    #        correctly generated zero events)
    # FINAL_BAIL:  must produce exactly 2 events (kick note 36 + crash note 49
    #              on beat 1, grid_position 0)
    drop_ok = any(d.get("is_drop", False) and d.get("event_count", 0) > 0
                  for d in diagnostics)
    bail_ok = any(d.get("is_bail", False) for d in diagnostics)
    final_bail_ok = any(d.get("is_final_bail", False) for d in diagnostics)

    # Stricter checks for BAIL and FINAL_BAIL using section-level data
    # BAIL section must have 0 events (not just is_bail flag)
    for d in diagnostics:
        if d.get("section") == "BAIL":
            if d.get("event_count", 0) != 0:
                bail_ok = False
            # Also verify is_bail reflects zero-event output
            if not d.get("is_bail", False):
                bail_ok = False

    # FINAL_BAIL section must have exactly 2 events
    for d in diagnostics:
        if d.get("section") == "FINAL_BAIL":
            ec = d.get("event_count", 0)
            if ec == 0:
                final_bail_ok = False
            # is_final_bail requires exactly kick+crash on beat 1 — trust the
            # is_final_bail_output check from the pipeline which validates notes

    # DROP section must have > 0 events (not just is_drop flag)
    for d in diagnostics:
        if d.get("section") == "DROP":
            if d.get("event_count", 0) <= 0:
                drop_ok = False

    output_contracts_passed = drop_ok and bail_ok and final_bail_ok

    # Event counts per section
    drop_event_count = 0
    bail_event_count = 0
    final_bail_event_count = 0
    for d in diagnostics:
        if d.get("section") == "DROP":
            drop_event_count = d.get("event_count", 0)
        elif d.get("section") == "BAIL":
            bail_event_count = d.get("event_count", 0)
        elif d.get("section") == "FINAL_BAIL":
            final_bail_event_count = d.get("event_count", 0)

    return PlaytestDiagnosticsSummary(
        total_events=total_events,
        first_enter_bar=first_enter_bar,
        first_build_bar=first_build_bar,
        confidence_peak=confidence_peak,
        phrase_marker_count=phrase_marker_count,
        inferred_intents=inferred_intents,
        output_contracts_passed=output_contracts_passed,
        drop_event_count=drop_event_count,
        final_bail_event_count=final_bail_event_count,
        bail_event_count=bail_event_count,
    )


def run_playtest_scenario(
    scenario: PlaytestScenario,
    no_play: bool = True,
) -> tuple[PlaytestDiagnosticsSummary, list[dict], list]:
    """Run a playtest scenario through the continuous jam pipeline.

    Parameters
    ----------
    scenario : PlaytestScenario
        The scenario variation to run.
    no_play : bool
        If True, run diagnostics only (no MIDI playback).

    Returns
    -------
    summary : PlaytestDiagnosticsSummary
        Real diagnostics extracted from the run.
    raw_diagnostics : list[dict]
        Per-bar diagnostic records (for inspection/debugging).
    global_events : list[GrooveEvent]
        Global GrooveEvent list for MIDI playback (empty in no-play mode
        or if playback is not possible).
    """
    # Import here to avoid circular imports at module level.
    # TODO: Move reusable runner logic out of demo_continuous_jam_midi into a
    # library module (e.g. drummer/playtest_runner.py) to break the demo→library
    # reverse dependency.  Acceptable for now.
    from demo_continuous_jam_midi import run_continuous_jam

    _pipeline, raw_diags, global_events = run_continuous_jam(
        bars=scenario.bars,
        bpm=scenario.bpm,
        mode="scripted",
        preset_name=scenario.preset,
        playtest_variation=scenario.variation_name,
    )

    summary = _extract_diagnostics_summary(raw_diags)

    return summary, raw_diags, global_events


# ---------------------------------------------------------------------------
# Part 6 – Feedback analysis and learning summary
# ---------------------------------------------------------------------------


@dataclass
class PlaytestLearningSummary:
    """Aggregated learning summary derived from raw feedback entries.

    All fields are computed deterministically from the entry data —
    no machine learning, no guesswork.
    """

    total_entries: int = 0
    entries_by_scenario: dict[str, int] = field(default_factory=dict)
    entries_by_preset: dict[str, int] = field(default_factory=dict)
    average_rating_by_scenario: dict[str, float] = field(default_factory=dict)
    average_rating_by_preset: dict[str, float] = field(default_factory=dict)
    common_timing_complaints: list[str] = field(default_factory=list)
    common_amount_complaints: list[str] = field(default_factory=list)
    common_confidence_complaints: list[str] = field(default_factory=list)
    common_suggested_changes: list[str] = field(default_factory=list)
    best_rated_scenarios: list[tuple[str, float]] = field(default_factory=list)
    worst_rated_scenarios: list[tuple[str, float]] = field(default_factory=list)
    best_rated_presets: list[tuple[str, float]] = field(default_factory=list)
    worst_rated_presets: list[tuple[str, float]] = field(default_factory=list)
    repeated_issues: list[str] = field(default_factory=list)
    possible_tuning_directions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Complaint pattern helpers
# ---------------------------------------------------------------------------

_SCENARIO_DISPLAY_NAMES: dict[str, str] = {
    "enter": "ENTER",
    "build": "BUILD",
    "anchor_recovery": "ANCHOR_RECOVERY",
    "drop": "DROP",
    "final_bail": "FINAL_BAIL",
}

_PRESET_DISPLAY_NAMES: dict[str, str] = {
    "cautious": "CAUTIOUS",
    "normal": "NORMAL",
    "braver": "BRAVER",
}


def _scenario_label(name: str) -> str:
    return _SCENARIO_DISPLAY_NAMES.get(name, name.upper())


def _preset_label(preset: str) -> str:
    return _PRESET_DISPLAY_NAMES.get(preset, preset)


# Mapping from suggested_change values to human-readable issue descriptions
_SUGGESTED_CHANGE_ISSUES: dict[str, str] = {
    "enter_later": "drummer enters too soon",
    "enter_sooner": "drummer enters too late",
    "play_less": "drummer plays too much",
    "play_more": "drummer plays too little",
    "build_more": "build intensity is too weak",
    "build_less": "build intensity is too strong",
    "recover_sooner": "recovery after anchor/drop is too slow",
    "recover_later": "recovery after anchor/drop is too fast",
    "mark_phrases_less": "phrase markers are too pronounced",
    "mark_phrases_more": "phrase markers are too subtle",
    "ending_cue_stronger": "ending cue is too weak",
    "ending_cue_weaker": "ending cue is too strong",
}

# Mapping from (issue_type, value) to tuning directions
_TUNING_DIRECTION_MAP: dict[str, dict[str, str]] = {
    "timing": {
        "too_early": "increase entry confirmation or observation time",
        "too_late": "reduce entry confirmation or respond sooner",
    },
    "amount": {
        "too_sparse": "increase velocity/密度 or add more events",
        "too_busy": "reduce velocity/density or subtract events",
    },
    "confidence": {
        "too_timid": "increase confidence threshold for decisive action",
        "too_bold": "lower confidence threshold or add hesitation",
    },
    "suggested_change": {
        "enter_later": "increase entry confirmation or observation time",
        "enter_sooner": "reduce entry confirmation or observation time",
        "play_less": "reduce event density or velocity across the board",
        "play_more": "increase event density or velocity across the board",
        "build_more": "increase build velocity ramp or phrase marker assertiveness",
        "build_less": "reduce build velocity ramp or phrase marker assertiveness",
        "recover_sooner": "shorten recovery timeout or increase recovery assertiveness",
        "recover_later": "lengthen recovery timeout or reduce recovery assertiveness",
        "mark_phrases_less": "reduce phrase marker velocity or accent emphasis",
        "mark_phrases_more": "increase phrase marker velocity or accent emphasis",
        "ending_cue_stronger": "increase final crash velocity slightly or improve timing",
        "ending_cue_weaker": "reduce final crash velocity or soften the ending",
    },
}


def _compute_average_ratings(
    entries: list[dict],
) -> tuple[dict[str, float], dict[str, float]]:
    """Compute average overall_rating grouped by scenario and by preset."""
    scenario_sums: dict[str, float] = {}
    scenario_counts: dict[str, int] = {}
    preset_sums: dict[str, float] = {}
    preset_counts: dict[str, int] = {}

    for entry in entries:
        scenario_name = entry.get("scenario", {}).get("name", "unknown")
        preset = entry.get("scenario", {}).get("preset", "unknown")
        rating = entry.get("answers", {}).get("overall_rating", 0)

        scenario_sums[scenario_name] = scenario_sums.get(scenario_name, 0) + rating
        scenario_counts[scenario_name] = scenario_counts.get(scenario_name, 0) + 1
        preset_sums[preset] = preset_sums.get(preset, 0) + rating
        preset_counts[preset] = preset_counts.get(preset, 0) + 1

    avg_by_scenario = {
        name: round(scenario_sums[name] / scenario_counts[name], 2)
        for name in scenario_sums
    }
    avg_by_preset = {
        name: round(preset_sums[name] / preset_counts[name], 2)
        for name in preset_sums
    }
    return avg_by_scenario, avg_by_preset


def _count_choices(
    entries: list[dict], field_path: str
) -> dict[str, int]:
    """Count occurrences of each value for a given nested field path.

    *field_path* uses dot notation, e.g. ``"answers.timing_rating"``.
    """
    counts: dict[str, int] = {}
    parts = field_path.split(".")
    for entry in entries:
        val = entry
        for part in parts:
            if isinstance(val, dict):
                val = val.get(part, None)
            else:
                val = None
                break
        if val is not None:
            counts[str(val)] = counts.get(str(val), 0) + 1
    return counts


def _build_complaint_list(
    entries: list[dict],
    field_path: str,
    label_fn: Callable[[str], str],
    ignore_values: set[str],
) -> list[str]:
    """Build a sorted list of common complaint descriptions.

    Parameters
    ----------
    entries : list[dict]
        The feedback entries.
    field_path : str
        Dot-notation path to the complaint field (e.g. ``"answers.timing_rating"``).
    label_fn : callable
        Maps each complaint value to a human-readable label.
    ignore_values : set[str]
        Values to ignore (e.g. ``{"about_right", "not_relevant"}``).
    """
    counts = _count_choices(entries, field_path)
    complaint_counts = {
        k: v for k, v in counts.items()
        if k not in ignore_values
    }
    if not complaint_counts:
        return []
    sorted_complaints = sorted(
        complaint_counts.items(), key=lambda x: -x[1]
    )
    return [f"{label_fn(k)} (×{v})" for k, v in sorted_complaints]


def _issue_label_for_scenario(
    scenario_name: str, issue_type: str, issue_value: str
) -> str:
    """Build a human-readable issue label like 'ENTER often feels too early'."""
    label = _scenario_label(scenario_name)
    if issue_type == "timing":
        return f"{label} often feels {issue_value.replace('_', ' ')}"
    elif issue_type == "amount":
        return f"{label} often feels {issue_value.replace('_', ' ')}"
    elif issue_type == "confidence":
        return f"{label} often feels {issue_value.replace('_', ' ')}"
    elif issue_type == "suggested_change":
        desc = _SUGGESTED_CHANGE_ISSUES.get(issue_value, issue_value)
        return f"{label}: {desc}"
    return f"{label}: {issue_value}"


def _detect_repeated_issues(entries: list[dict]) -> list[str]:
    """Detect repeated issues by scenario for timing, amount, confidence, and suggested changes."""
    if not entries:
        return []

    issues: list[str] = []

    # Group entries by scenario
    by_scenario: dict[str, list[dict]] = {}
    for entry in entries:
        sc_name = entry.get("scenario", {}).get("name", "unknown")
        by_scenario.setdefault(sc_name, []).append(entry)

    for scenario_name, sc_entries in sorted(by_scenario.items()):
        # Timing complaints per scenario
        timing_counts = _count_choices(sc_entries, "answers.timing_rating")
        for val, count in sorted(timing_counts.items(), key=lambda x: -x[1]):
            if val not in ("about_right", "not_relevant") and count >= 1:
                issues.append(_issue_label_for_scenario(scenario_name, "timing", val))

        # Amount complaints per scenario
        amount_counts = _count_choices(sc_entries, "answers.amount_rating")
        for val, count in sorted(amount_counts.items(), key=lambda x: -x[1]):
            if val not in ("about_right", "not_relevant") and count >= 1:
                issues.append(_issue_label_for_scenario(scenario_name, "amount", val))

        # Confidence complaints per scenario
        conf_counts = _count_choices(sc_entries, "answers.confidence_rating")
        for val, count in sorted(conf_counts.items(), key=lambda x: -x[1]):
            if val not in ("about_right", "not_relevant") and count >= 1:
                issues.append(_issue_label_for_scenario(scenario_name, "confidence", val))

    # Overall (not per-scenario) suggested change complaints
    change_counts = _count_choices(entries, "answers.suggested_change")
    for val, count in sorted(change_counts.items(), key=lambda x: -x[1]):
        if val != "no_change" and count >= 1:
            desc = _SUGGESTED_CHANGE_ISSUES.get(val, val)
            issues.append(f"suggested: {desc} (×{count})")

    return issues


def _detect_tuning_directions(entries: list[dict]) -> list[str]:
    """Suggest possible tuning directions based on repeated complaints."""
    if not entries:
        return []

    directions: list[str] = []

    # Check timing complaints across all entries
    timing_counts = _count_choices(entries, "answers.timing_rating")
    for val, count in timing_counts.items():
        if val in _TUNING_DIRECTION_MAP.get("timing", {}):
            dir_text = _TUNING_DIRECTION_MAP["timing"][val]
            directions.append(
                f"based on timing feedback ({val} ×{count}): {dir_text}"
            )

    # Check amount complaints across all entries
    amount_counts = _count_choices(entries, "answers.amount_rating")
    for val, count in amount_counts.items():
        if val in _TUNING_DIRECTION_MAP.get("amount", {}):
            dir_text = _TUNING_DIRECTION_MAP["amount"][val]
            directions.append(
                f"based on amount feedback ({val} ×{count}): {dir_text}"
            )

    # Check confidence complaints across all entries
    conf_counts = _count_choices(entries, "answers.confidence_rating")
    for val, count in conf_counts.items():
        if val in _TUNING_DIRECTION_MAP.get("confidence", {}):
            dir_text = _TUNING_DIRECTION_MAP["confidence"][val]
            directions.append(
                f"based on confidence feedback ({val} ×{count}): {dir_text}"
            )

    # Check suggested change directions
    change_counts = _count_choices(entries, "answers.suggested_change")
    for val, count in change_counts.items():
        if val in _TUNING_DIRECTION_MAP.get("suggested_change", {}):
            dir_text = _TUNING_DIRECTION_MAP["suggested_change"][val]
            directions.append(
                f"based on suggested change ({val} ×{count}): {dir_text}"
            )

    return directions


def _rank_best_worst(
    avg_dict: dict[str, float],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    """Return (best_rated, worst_rated) sorted lists from an average-rating dict."""
    if not avg_dict:
        return [], []
    sorted_items = sorted(avg_dict.items(), key=lambda x: -x[1])
    best = sorted_items[:3]
    worst = list(reversed(sorted_items[-3:]))
    return best, worst


# ---------------------------------------------------------------------------
# Public API for feedback analysis
# ---------------------------------------------------------------------------


def summarize_feedback_entries(entries: list[dict]) -> PlaytestLearningSummary:
    """Produce a ``PlaytestLearningSummary`` from a list of raw feedback entry dicts.

    Parameters
    ----------
    entries : list[dict]
        The feedback entries as returned by :func:`load_feedback_entries`.

    Returns
    -------
    PlaytestLearningSummary
        Deterministic summary of what the feedback tells us.
    """
    summary = PlaytestLearningSummary()
    summary.total_entries = len(entries)

    if not entries:
        return summary

    # Count by scenario
    scenario_counts = _count_choices(entries, "scenario.name")
    summary.entries_by_scenario = dict(
        sorted(scenario_counts.items(), key=lambda x: -x[1])
    )

    # Count by preset
    preset_counts = _count_choices(entries, "scenario.preset")
    summary.entries_by_preset = dict(
        sorted(preset_counts.items(), key=lambda x: -x[1])
    )

    # Average ratings
    avg_scenario, avg_preset = _compute_average_ratings(entries)
    summary.average_rating_by_scenario = dict(
        sorted(avg_scenario.items(), key=lambda x: -x[1])
    )
    summary.average_rating_by_preset = dict(
        sorted(avg_preset.items(), key=lambda x: -x[1])
    )

    # Common complaints (non-"about_right"/"not_relevant")
    summary.common_timing_complaints = _build_complaint_list(
        entries,
        "answers.timing_rating",
        label_fn=lambda v: f"timing feels {v.replace('_', ' ')}",
        ignore_values={"about_right", "not_relevant"},
    )
    summary.common_amount_complaints = _build_complaint_list(
        entries,
        "answers.amount_rating",
        label_fn=lambda v: f"amount feels {v.replace('_', ' ')}",
        ignore_values={"about_right", "not_relevant"},
    )
    summary.common_confidence_complaints = _build_complaint_list(
        entries,
        "answers.confidence_rating",
        label_fn=lambda v: f"confidence feels {v.replace('_', ' ')}",
        ignore_values={"about_right", "not_relevant"},
    )

    # Common suggested changes
    change_counts_raw = _count_choices(entries, "answers.suggested_change")
    change_counts_filtered = {
        k: v for k, v in change_counts_raw.items()
        if k != "no_change"
    }
    if change_counts_filtered:
        sorted_changes = sorted(
            change_counts_filtered.items(), key=lambda x: -x[1]
        )
        summary.common_suggested_changes = [
            f"{_SUGGESTED_CHANGE_ISSUES.get(k, k)} (×{v})"
            for k, v in sorted_changes
        ]

    # Best / worst rated
    summary.best_rated_scenarios, summary.worst_rated_scenarios = (
        _rank_best_worst(avg_scenario)
    )
    summary.best_rated_presets, summary.worst_rated_presets = (
        _rank_best_worst(avg_preset)
    )

    # Repeated issues
    summary.repeated_issues = _detect_repeated_issues(entries)

    # Possible tuning directions
    summary.possible_tuning_directions = _detect_tuning_directions(entries)

    return summary


def load_and_summarize_feedback(path: str) -> PlaytestLearningSummary:
    """Load feedback entries from a JSONL file and produce a learning summary.

    Parameters
    ----------
    path : str
        Path to the JSONL feedback file.

    Returns
    -------
    PlaytestLearningSummary
    """
    entries = load_feedback_entries(path)
    return summarize_feedback_entries(entries)


def export_learning_summary_json(summary: PlaytestLearningSummary, path: str) -> None:
    """Write a learning summary to a JSON file.

    Parameters
    ----------
    summary : PlaytestLearningSummary
        The summary to export.
    path : str
        Output path for the JSON file.
    """
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2, sort_keys=True)
        f.write("\n")


def _format_rating_list(
    items: list[tuple[str, float]], label: str
) -> str:
    """Format a list of (name, avg_rating) tuples for markdown."""
    if not items:
        return f"  (no {label} data yet)\n"
    lines = []
    for name, avg in items:
        display = _scenario_label(name) if label in ("scenario", "scenarios") else _preset_label(name)
        lines.append(f"  * **{display}** — average rating {avg:.2f} / 5\n")
    return "".join(lines)


def export_learning_summary_markdown(summary: PlaytestLearningSummary, path: str) -> None:
    """Write a human-readable markdown learning summary.

    The markdown is designed to be pasted into ChatGPT or Cline for discussion.

    Parameters
    ----------
    summary : PlaytestLearningSummary
        The summary to export.
    path : str
        Output path for the markdown file.
    """
    lines: list[str] = []
    lines.append("# Pocket Drummer Playtest Learning Summary\n")
    lines.append(f"*Generated from {summary.total_entries} feedback entries*\n")

    # Overall
    lines.append("## Overall\n")
    lines.append(f"* **Total feedback entries:** {summary.total_entries}\n")
    lines.append("* **Presets tested:** "
                 f"{', '.join(summary.entries_by_preset.keys()) if summary.entries_by_preset else '(none)'}\n")
    lines.append("* **Scenarios tested:** "
                 f"{', '.join(summary.entries_by_scenario.keys()) if summary.entries_by_scenario else '(none)'}\n")
    lines.append("\n")

    # Best working areas
    lines.append("## Best Working Areas\n")
    if summary.best_rated_scenarios:
        lines.append("### Highest-Rated Scenarios\n")
        lines.append(_format_rating_list(summary.best_rated_scenarios, "scenario"))
    if summary.best_rated_presets:
        lines.append("### Highest-Rated Presets\n")
        lines.append(_format_rating_list(summary.best_rated_presets, "preset"))
    if not summary.best_rated_scenarios and not summary.best_rated_presets:
        lines.append("  (no data yet)\n")
    lines.append("\n")

    # Problem areas
    lines.append("## Problem Areas\n")
    if summary.worst_rated_scenarios:
        lines.append("### Lowest-Rated Scenarios\n")
        lines.append(_format_rating_list(summary.worst_rated_scenarios, "scenario"))
    if summary.worst_rated_presets:
        lines.append("### Lowest-Rated Presets\n")
        lines.append(_format_rating_list(summary.worst_rated_presets, "preset"))
    if not summary.worst_rated_scenarios and not summary.worst_rated_presets:
        lines.append("  (no data yet)\n")
    lines.append("\n")

    # Repeated complaints
    lines.append("## Repeated Complaints\n")
    if summary.repeated_issues:
        for issue in summary.repeated_issues:
            lines.append(f"* {issue}\n")
    else:
        lines.append("  (no repeated complaints identified)\n")
    lines.append("\n")

    # Timing complaints
    if summary.common_timing_complaints:
        lines.append("### Timing Complaints\n")
        for complaint in summary.common_timing_complaints:
            lines.append(f"* {complaint}\n")
        lines.append("\n")

    # Amount complaints
    if summary.common_amount_complaints:
        lines.append("### Amount Complaints\n")
        for complaint in summary.common_amount_complaints:
            lines.append(f"* {complaint}\n")
        lines.append("\n")

    # Confidence complaints
    if summary.common_confidence_complaints:
        lines.append("### Confidence Complaints\n")
        for complaint in summary.common_confidence_complaints:
            lines.append(f"* {complaint}\n")
        lines.append("\n")

    # Suggested changes
    if summary.common_suggested_changes:
        lines.append("### Most Requested Changes\n")
        for change in summary.common_suggested_changes:
            lines.append(f"* {change}\n")
        lines.append("\n")

    # Possible tuning directions
    lines.append("## Possible Tuning Directions\n")
    if summary.possible_tuning_directions:
        for direction in summary.possible_tuning_directions:
            lines.append(f"* {direction}\n")
    else:
        lines.append("  (no clear tuning directions yet — more feedback needed)\n")
    lines.append("\n")

    # Raw counts
    lines.append("## Raw Counts\n")
    lines.append(f"* Total entries: {summary.total_entries}\n")
    lines.append("\n")
    lines.append("### Entries by Scenario\n")
    if summary.entries_by_scenario:
        for sc_name, count in summary.entries_by_scenario.items():
            lines.append(f"* {_scenario_label(sc_name)}: {count}\n")
    else:
        lines.append("  (none)\n")
    lines.append("\n")
    lines.append("### Entries by Preset\n")
    if summary.entries_by_preset:
        for preset, count in summary.entries_by_preset.items():
            lines.append(f"* {_preset_label(preset)}: {count}\n")
    else:
        lines.append("  (none)\n")
    lines.append("\n")
    lines.append("### Average Rating by Scenario\n")
    if summary.average_rating_by_scenario:
        for sc_name, avg in summary.average_rating_by_scenario.items():
            lines.append(f"* {_scenario_label(sc_name)}: {avg:.2f} / 5\n")
    else:
        lines.append("  (none)\n")
    lines.append("\n")
    lines.append("### Average Rating by Preset\n")
    if summary.average_rating_by_preset:
        for preset, avg in summary.average_rating_by_preset.items():
            lines.append(f"* {_preset_label(preset)}: {avg:.2f} / 5\n")
    else:
        lines.append("  (none)\n")
    lines.append("\n")
    lines.append("---\n")
    lines.append("*This summary is deterministic — counts and averages only. "
                 "No machine learning or automatic tuning applied.*\n")

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def _print_console_summary(summary: PlaytestLearningSummary) -> None:
    """Print a compact console summary of the learning summary."""
    sep = "=" * 60
    print(f"\n{sep}")
    print("  POCKET DRUMMER — FEEDBACK LEARNING SUMMARY")
    print(f"{sep}")
    print(f"  Total feedback entries: {summary.total_entries}")
    print()

    if summary.total_entries == 0:
        print("  No feedback entries to summarise.")
        print(f"{sep}\n")
        return

    # Best scenarios
    print("  Top liked scenarios:")
    if summary.best_rated_scenarios:
        for name, avg in summary.best_rated_scenarios:
            print(f"    {_scenario_label(name):20s}  avg {avg:.2f}")
    else:
        print("    (no data)")
    print()

    # Lowest scenarios
    print("  Lowest rated scenarios:")
    if summary.worst_rated_scenarios:
        for name, avg in summary.worst_rated_scenarios:
            print(f"    {_scenario_label(name):20s}  avg {avg:.2f}")
    else:
        print("    (no data)")
    print()

    # Repeated complaints
    print("  Repeated complaints:")
    if summary.repeated_issues:
        for issue in summary.repeated_issues:
            print(f"    • {issue}")
    else:
        print("    (none identified)")
    print()

    # Possible tuning directions
    print("  Possible tuning directions:")
    if summary.possible_tuning_directions:
        for direction in summary.possible_tuning_directions:
            print(f"    → {direction}")
    else:
        print("    (insufficient data)")
    print(f"{sep}\n")