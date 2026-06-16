"""Tests for the playtest feedback module."""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

from drummer.playtest_feedback import (
    PlaytestScenario,
    PlaytestQuestionnaire,
    PlaytestDiagnosticsSummary,
    PlaytestFeedbackEntry,
    list_playtest_scenarios,
    get_scenario_variations,
    serialize_feedback_entry,
    append_feedback_entry,
    load_feedback_entries,
    validate_overall_rating,
    validate_choice,
    validate_questionnaire_answers,
    run_playtest_scenario,
    _extract_diagnostics_summary,
    OVERALL_RATING_VALUES,
    TIMING_CHOICES,
    AMOUNT_CHOICES,
    CONFIDENCE_CHOICES,
    UNDERSTOOD_CHOICES,
    SUGGESTED_CHANGE_CHOICES,
)


def _tmp_path(suffix: str = ".tmp") -> str:
    """Create a safe temporary path using NamedTemporaryFile."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_scenario() -> PlaytestScenario:
    return PlaytestScenario(
        name="enter",
        variation_name="stable_input",
        description="Consistent input.",
        preset="normal",
        mode="continuous",
        bars=8,
        bpm=120.0,
    )


@pytest.fixture
def sample_diagnostics() -> PlaytestDiagnosticsSummary:
    return PlaytestDiagnosticsSummary(
        total_events=42,
        first_enter_bar=3,
        first_build_bar=None,
        confidence_peak=0.85,
        phrase_marker_count=2,
        inferred_intents={"enter": 1, "groove": 3},
        output_contracts_passed=True,
        drop_event_count=0,
        final_bail_event_count=0,
        bail_event_count=0,
    )


@pytest.fixture
def sample_answers() -> PlaytestQuestionnaire:
    return PlaytestQuestionnaire(
        overall_rating=4,
        timing_rating="about_right",
        amount_rating="about_right",
        confidence_rating="about_right",
        understood_rating="yes",
        suggested_change="no_change",
        note="Sounds great!",
    )


@pytest.fixture
def sample_entry(
    sample_scenario: PlaytestScenario,
    sample_diagnostics: PlaytestDiagnosticsSummary,
    sample_answers: PlaytestQuestionnaire,
) -> PlaytestFeedbackEntry:
    return PlaytestFeedbackEntry(
        scenario=sample_scenario,
        diagnostics=sample_diagnostics,
        answers=sample_answers,
    )


# ---------------------------------------------------------------------------
# Part 3 – Scenario registry tests
# ---------------------------------------------------------------------------


class TestScenarioRegistry:
    def test_list_playtest_scenarios_includes_required_names(self) -> None:
        names = list_playtest_scenarios()
        for required in ("enter", "build", "anchor_recovery", "drop", "final_bail"):
            assert required in names, f"Missing required scenario: {required}"

    def test_each_scenario_has_at_least_two_variations(self) -> None:
        for name in list_playtest_scenarios():
            variations = get_scenario_variations(name)
            assert len(variations) >= 2, (
                f"Scenario {name!r} has fewer than 2 variations"
            )

    def test_unknown_scenario_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown scenario"):
            get_scenario_variations("nonexistent")

    def test_scenario_variations_use_preset(self) -> None:
        for name in ("enter", "build"):
            for preset in ("cautious", "normal", "braver"):
                variations = get_scenario_variations(name, preset=preset)
                for v in variations:
                    assert v.preset == preset

    def test_all_variations_have_correct_name(self) -> None:
        for name in list_playtest_scenarios():
            variations = get_scenario_variations(name)
            for v in variations:
                assert v.name == name

    def test_variations_have_listen_focus_fields(self) -> None:
        """Each variation must set listen_start_bar, listen_end_bar, what_to_listen_for."""
        for name in list_playtest_scenarios():
            variations = get_scenario_variations(name)
            for v in variations:
                assert v.listen_start_bar >= 0
                assert v.listen_end_bar > v.listen_start_bar
                assert len(v.what_to_listen_for) > 0

    def test_listen_fields_serialize(self) -> None:
        """listen_* fields appear in to_dict output."""
        sc = get_scenario_variations("enter")[0]
        d = sc.to_dict()
        assert "listen_start_bar" in d
        assert "listen_end_bar" in d
        assert "what_to_listen_for" in d
        assert isinstance(d["listen_start_bar"], int)
        assert isinstance(d["listen_end_bar"], int)

    def test_scenario_default_makes_20_bars_with_120_bpm(self) -> None:
        """Newly-built scenarios use 20 bars and 120 bpm by default."""
        variations = get_scenario_variations("build")
        for v in variations:
            assert v.bars == 20
            assert v.bpm == 120.0


# ---------------------------------------------------------------------------
# Key-parsing tests (single-key input maps to canonical values)
# ---------------------------------------------------------------------------


class TestKeyParsing:
    """Test that single-key input maps to correct canonical values."""

    @pytest.mark.parametrize("key, expected", [
        ("e", "too_early"), ("E", "too_early"), (" e ", "too_early"),
        ("r", "about_right"), ("R", "about_right"),
        ("l", "too_late"), ("L", "too_late"),
        ("n", "not_relevant"), ("N", "not_relevant"),
    ])
    def test_timing_key_map(self, key: str, expected: str) -> None:
        from drummer.playtest_feedback import parse_key_choice, TIMING_KEY_MAP
        assert parse_key_choice(key, TIMING_KEY_MAP, "timing") == expected

    @pytest.mark.parametrize("key, expected", [
        ("s", "too_sparse"), ("S", "too_sparse"),
        ("r", "about_right"), ("R", "about_right"),
        ("b", "too_busy"), ("B", "too_busy"),
        ("n", "not_relevant"), ("N", "not_relevant"),
    ])
    def test_amount_key_map(self, key: str, expected: str) -> None:
        from drummer.playtest_feedback import parse_key_choice, AMOUNT_KEY_MAP
        assert parse_key_choice(key, AMOUNT_KEY_MAP, "amount") == expected

    @pytest.mark.parametrize("key, expected", [
        ("t", "too_timid"), ("T", "too_timid"),
        ("r", "about_right"), ("R", "about_right"),
        ("d", "too_bold"), ("D", "too_bold"),
        ("n", "not_relevant"), ("N", "not_relevant"),
    ])
    def test_confidence_key_map(self, key: str, expected: str) -> None:
        from drummer.playtest_feedback import parse_key_choice, CONFIDENCE_KEY_MAP
        assert parse_key_choice(key, CONFIDENCE_KEY_MAP, "confidence") == expected

    @pytest.mark.parametrize("key, expected", [
        ("y", "yes"), ("Y", "yes"),
        ("p", "partly"), ("P", "partly"),
        ("n", "no"), ("N", "no"),
    ])
    def test_understood_key_map(self, key: str, expected: str) -> None:
        from drummer.playtest_feedback import parse_key_choice, UNDERSTOOD_KEY_MAP
        assert parse_key_choice(key, UNDERSTOOD_KEY_MAP, "understood") == expected

    @pytest.mark.parametrize("key, expected", [
        ("0", "no_change"),
        ("1", "enter_later"),
        ("2", "enter_sooner"),
        ("3", "play_less"),
        ("4", "play_more"),
        ("5", "build_less"),
        ("6", "build_more"),
        ("7", "recover_later"),
        ("8", "recover_sooner"),
        ("9", "mark_phrases_less"),
        ("10", "mark_phrases_more"),
        ("11", "ending_cue_weaker"),
        ("12", "ending_cue_stronger"),
    ])
    def test_suggested_change_key_map(self, key: str, expected: str) -> None:
        from drummer.playtest_feedback import parse_key_choice, SUGGESTED_CHANGE_KEY_MAP
        assert parse_key_choice(key, SUGGESTED_CHANGE_KEY_MAP, "suggested_change") == expected

    def test_whitespace_is_stripped(self) -> None:
        from drummer.playtest_feedback import parse_key_choice, TIMING_KEY_MAP
        assert parse_key_choice("  e  ", TIMING_KEY_MAP, "timing") == "too_early"
        assert parse_key_choice("\tr\n", TIMING_KEY_MAP, "timing") == "about_right"

    def test_invalid_key_raises_value_error(self) -> None:
        from drummer.playtest_feedback import parse_key_choice, TIMING_KEY_MAP
        with pytest.raises(ValueError, match="timing"):
            parse_key_choice("x", TIMING_KEY_MAP, "timing")

    def test_invalid_key_for_understood_raises(self) -> None:
        from drummer.playtest_feedback import parse_key_choice, UNDERSTOOD_KEY_MAP
        with pytest.raises(ValueError, match="understood"):
            parse_key_choice("z", UNDERSTOOD_KEY_MAP, "understood")

    def test_invalid_suggested_change_key_raises(self) -> None:
        from drummer.playtest_feedback import parse_key_choice, SUGGESTED_CHANGE_KEY_MAP
        with pytest.raises(ValueError, match="suggested_change"):
            parse_key_choice("13", SUGGESTED_CHANGE_KEY_MAP, "suggested_change")

    def test_stored_questionnaire_contains_canonical_values(self, sample_answers) -> None:
        """Verify the sample questionnaire uses canonical values (not keys)."""
        d = sample_answers.to_dict()
        assert d["timing_rating"] == "about_right"
        assert d["amount_rating"] == "about_right"
        assert d["confidence_rating"] == "about_right"
        assert d["understood_rating"] == "yes"
        assert d["suggested_change"] == "no_change"
        assert d["overall_rating"] == 4

    def test_full_canonical_validation_still_works(self) -> None:
        """Existing validate_questionnaire_answers still accepts canonical values."""
        validate_questionnaire_answers(
            overall_rating=5,
            timing_rating="too_early",
            amount_rating="too_busy",
            confidence_rating="about_right",
            understood_rating="partly",
            suggested_change="enter_sooner",
        )  # should not raise

    def test_full_pipeline_key_to_canonical_to_validated(self) -> None:
        """Key -> canonical -> validation pipeline works end-to-end."""
        from drummer.playtest_feedback import (
            parse_key_choice, TIMING_KEY_MAP, AMOUNT_KEY_MAP,
            CONFIDENCE_KEY_MAP, UNDERSTOOD_KEY_MAP, SUGGESTED_CHANGE_KEY_MAP,
            validate_questionnaire_answers,
        )
        timing = parse_key_choice("e", TIMING_KEY_MAP, "timing")
        amount = parse_key_choice("b", AMOUNT_KEY_MAP, "amount")
        confidence = parse_key_choice("r", CONFIDENCE_KEY_MAP, "confidence")
        understood = parse_key_choice("y", UNDERSTOOD_KEY_MAP, "understood")
        suggested = parse_key_choice("0", SUGGESTED_CHANGE_KEY_MAP, "suggested")
        validate_questionnaire_answers(
            overall_rating=3,
            timing_rating=timing,
            amount_rating=amount,
            confidence_rating=confidence,
            understood_rating=understood,
            suggested_change=suggested,
        )  # should not raise


# ---------------------------------------------------------------------------
# Part 2 – Validation tests
# ---------------------------------------------------------------------------


class TestValidation:
    def test_valid_overall_rating_passes(self) -> None:
        for r in OVERALL_RATING_VALUES:
            validate_overall_rating(r)  # should not raise

    @pytest.mark.parametrize("bad", [0, 6, -1, 999])
    def test_invalid_overall_rating_fails(self, bad: int) -> None:
        with pytest.raises(ValueError, match="overall_rating"):
            validate_overall_rating(bad)

    @pytest.mark.parametrize(
        "choices_set, field_name",
        [
            (TIMING_CHOICES, "timing_rating"),
            (AMOUNT_CHOICES, "amount_rating"),
            (CONFIDENCE_CHOICES, "confidence_rating"),
            (UNDERSTOOD_CHOICES, "understood_rating"),
            (SUGGESTED_CHANGE_CHOICES, "suggested_change"),
        ],
    )
    def test_valid_choices_pass(
        self, choices_set: frozenset, field_name: str
    ) -> None:
        for choice in choices_set:
            validate_choice(choice, choices_set, field_name)  # should not raise

    @pytest.mark.parametrize(
        "choices_set, field_name, bad",
        [
            (TIMING_CHOICES, "timing_rating", "invalid_timing"),
            (AMOUNT_CHOICES, "amount_rating", "invalid_amount"),
            (CONFIDENCE_CHOICES, "confidence_rating", "invalid_confidence"),
            (UNDERSTOOD_CHOICES, "understood_rating", "maybe"),
            (SUGGESTED_CHANGE_CHOICES, "suggested_change", "invalid_change"),
        ],
    )
    def test_invalid_choice_fails(
        self, choices_set: frozenset, field_name: str, bad: str
    ) -> None:
        with pytest.raises(ValueError, match=field_name):
            validate_choice(bad, choices_set, field_name)

    def test_valid_questionnaire_passes(self) -> None:
        validate_questionnaire_answers(
            overall_rating=3,
            timing_rating="about_right",
            amount_rating="too_busy",
            confidence_rating="too_timid",
            understood_rating="partly",
            suggested_change="play_less",
        )  # should not raise

    def test_invalid_questionnaire_fails(self) -> None:
        with pytest.raises(ValueError, match="overall_rating"):
            validate_questionnaire_answers(
                overall_rating=0,
                timing_rating="about_right",
                amount_rating="about_right",
                confidence_rating="about_right",
                understood_rating="yes",
                suggested_change="no_change",
            )


# ---------------------------------------------------------------------------
# Part 1 – Data model tests
# ---------------------------------------------------------------------------


class TestDataModels:
    def test_scenario_to_dict(self, sample_scenario: PlaytestScenario) -> None:
        d = sample_scenario.to_dict()
        assert d["name"] == "enter"
        assert d["variation_name"] == "stable_input"
        assert d["bars"] == 8
        assert isinstance(d["bpm"], float)

    def test_diagnostics_to_dict(
        self, sample_diagnostics: PlaytestDiagnosticsSummary
    ) -> None:
        d = sample_diagnostics.to_dict()
        assert d["total_events"] == 42
        assert d["first_build_bar"] is None
        assert d["confidence_peak"] == 0.85

    def test_questionnaire_to_dict(
        self, sample_answers: PlaytestQuestionnaire
    ) -> None:
        d = sample_answers.to_dict()
        assert d["overall_rating"] == 4
        assert d["timing_rating"] == "about_right"
        assert d["note"] == "Sounds great!"

    def test_entry_to_dict(self, sample_entry: PlaytestFeedbackEntry) -> None:
        d = sample_entry.to_dict()
        assert "scenario" in d
        assert "diagnostics" in d
        assert "answers" in d
        assert d["scenario"]["name"] == "enter"
        assert d["answers"]["overall_rating"] == 4


# ---------------------------------------------------------------------------
# Part 4 – JSONL serialization tests
# ---------------------------------------------------------------------------


class TestJsonlPersistence:
    def test_serialize_feedback_entry(self, sample_entry: PlaytestFeedbackEntry) -> None:
        line = serialize_feedback_entry(sample_entry)
        assert isinstance(line, str)
        parsed = json.loads(line)
        assert parsed["scenario"]["name"] == "enter"
        assert parsed["answers"]["overall_rating"] == 4
        assert parsed["diagnostics"]["total_events"] == 42

    def test_deterministic_serialization(
        self, sample_entry: PlaytestFeedbackEntry
    ) -> None:
        line1 = serialize_feedback_entry(sample_entry)
        line2 = serialize_feedback_entry(sample_entry)
        assert line1 == line2

    def test_append_and_load_one_entry(
        self, sample_entry: PlaytestFeedbackEntry
    ) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            append_feedback_entry(tmp, sample_entry)
            entries = load_feedback_entries(tmp)
            assert len(entries) == 1
            assert entries[0]["scenario"]["name"] == "enter"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_append_and_load_multiple_entries(
        self, sample_entry: PlaytestFeedbackEntry
    ) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            for _ in range(3):
                append_feedback_entry(tmp, sample_entry)
            entries = load_feedback_entries(tmp)
            assert len(entries) == 3
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_load_from_empty_file(self) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("")
            entries = load_feedback_entries(tmp)
            assert entries == []
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_append_creates_file(
        self, sample_entry: PlaytestFeedbackEntry
    ) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            append_feedback_entry(tmp, sample_entry)
            assert os.path.exists(tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 1
            line = lines[0].strip()
            assert json.loads(line)["scenario"]["name"] == "enter"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_entries_are_json_lines(
        self, sample_entry: PlaytestFeedbackEntry
    ) -> None:
        """Each entry should be exactly one JSON object per line."""
        tmp = _tmp_path(suffix=".jsonl")
        try:
            for _ in range(2):
                append_feedback_entry(tmp, sample_entry)
            with open(tmp, "r", encoding="utf-8") as f:
                lines = f.readlines()
            assert len(lines) == 2
            for line in lines:
                obj = json.loads(line.strip())
                assert "scenario" in obj
                assert "diagnostics" in obj
                assert "answers" in obj
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


# ---------------------------------------------------------------------------
# Part 5 – Scenario runner tests
# ---------------------------------------------------------------------------


class TestScenarioRunner:
    def test_run_playtest_scenario_works_in_no_play_mode(self) -> None:
        sc = get_scenario_variations("enter")[0]
        summary, raw_diags, global_events = run_playtest_scenario(sc, no_play=True)
        assert isinstance(summary, PlaytestDiagnosticsSummary)
        assert summary.total_events >= 0
        assert len(raw_diags) > 0
        # no-play still returns events (needed for sanity checking)
        assert len(global_events) > 0

    def test_each_scenario_produces_real_diagnostics(self) -> None:
        """Run first variation of each scenario name and verify diagnostics."""
        for name in list_playtest_scenarios():
            sc = get_scenario_variations(name)[0]
            summary, raw_diags, _ = run_playtest_scenario(sc, no_play=True)
            assert summary.total_events >= 0, (
                f"Scenario {name}: total_events should be >= 0"
            )
            assert len(raw_diags) > 0, (
                f"Scenario {name}: should produce raw diagnostics"
            )

    def test_diagnostics_include_inferred_intent_summary(self) -> None:
        sc = get_scenario_variations("drop")[0]
        summary, _raw, _ = run_playtest_scenario(sc, no_play=True)
        assert len(summary.inferred_intents) > 0
        # Should include known intents like listen, enter, build, drop, bail
        all_keys = set(summary.inferred_intents.keys())
        assert "listen" in all_keys or "bail" in all_keys

    def test_diagnostics_include_contracts(self) -> None:
        sc = get_scenario_variations("enter")[0]
        summary, _raw, _ = run_playtest_scenario(sc, no_play=True)
        # output_contracts_passed should be a bool
        assert isinstance(summary.output_contracts_passed, bool)

    def test_first_enter_bar_is_present(self) -> None:
        sc = get_scenario_variations("enter")[0]
        summary, _raw, _ = run_playtest_scenario(sc, no_play=True)
        assert summary.first_enter_bar is not None
        assert summary.first_enter_bar >= 0

    def test_confidence_peak_is_non_negative(self) -> None:
        sc = get_scenario_variations("build")[0]
        summary, _raw, _ = run_playtest_scenario(sc, no_play=True)
        assert summary.confidence_peak >= 0.0


class TestExtractDiagnosticsSummary:
    def test_empty_diagnostics(self) -> None:
        summary = _extract_diagnostics_summary([])
        assert summary.total_events == 0
        assert summary.first_enter_bar is None
        assert summary.first_build_bar is None
        assert summary.confidence_peak == 0.0
        assert summary.phrase_marker_count == 0
        assert summary.inferred_intents == {}
        assert summary.output_contracts_passed is False

    def test_extracts_first_enter_bar(self) -> None:
        raw = [
            {"bar": 0, "intent": "listen", "inferred_intent": "listen",
             "event_count": 0, "section": "LISTEN",
             "confidence": 0.0, "phrase_marker": "none",
             "is_drop": False, "is_bail": False, "is_final_bail": False},
            {"bar": 1, "intent": "listen", "inferred_intent": "listen",
             "event_count": 0, "section": "LISTEN",
             "confidence": 0.0, "phrase_marker": "none",
             "is_drop": False, "is_bail": False, "is_final_bail": False},
            {"bar": 2, "intent": "enter_soft", "inferred_intent": "enter_soft",
             "event_count": 5, "section": "ENTER_SOFT",
             "confidence": 0.5, "phrase_marker": "none",
             "is_drop": False, "is_bail": False, "is_final_bail": False},
        ]
        summary = _extract_diagnostics_summary(raw)
        assert summary.first_enter_bar == 2
        assert summary.total_events == 5


# ---------------------------------------------------------------------------
# Part 6 – Demo importability test
# ---------------------------------------------------------------------------


class TestDemoImportability:
    def test_demo_module_can_be_imported(self) -> None:
        """Import demo module without side effects (__name__ != '__main__')."""
        import importlib as _il
        mod = _il.import_module("demo_playtest_interview")
        assert hasattr(mod, "main")
        assert hasattr(mod, "build_parser")


# ---------------------------------------------------------------------------
# Part 7 – Feedback analysis / learning summary tests
# ---------------------------------------------------------------------------


class TestSummarizeFeedbackEmpty:
    def test_empty_feedback_summary_has_zero_entries(self) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries([])
        assert summary.total_entries == 0
        assert summary.entries_by_scenario == {}
        assert summary.entries_by_preset == {}
        assert summary.average_rating_by_scenario == {}
        assert summary.average_rating_by_preset == {}
        assert summary.common_timing_complaints == []
        assert summary.common_amount_complaints == []
        assert summary.common_confidence_complaints == []
        assert summary.common_suggested_changes == []
        assert summary.best_rated_scenarios == []
        assert summary.worst_rated_scenarios == []
        assert summary.best_rated_presets == []
        assert summary.worst_rated_presets == []
        assert summary.repeated_issues == []
        assert summary.possible_tuning_directions == []

    def test_load_and_summarize_empty_file(self) -> None:
        from drummer.playtest_feedback import load_and_summarize_feedback
        tmp = _tmp_path(suffix=".jsonl")
        try:
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("")
            summary = load_and_summarize_feedback(tmp)
            assert summary.total_entries == 0
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_load_and_summarize_non_existent_file_raises(self) -> None:
        from drummer.playtest_feedback import load_and_summarize_feedback
        with pytest.raises(FileNotFoundError):
            load_and_summarize_feedback("/no/such/file.jsonl")


class TestSummarizeFeedbackCounts:
    @pytest.fixture
    def entries(self) -> list[dict]:
        """Three feedback entries across different scenarios and presets."""
        return [
            {
                "scenario": {"name": "enter", "preset": "normal"},
                "answers": {
                    "overall_rating": 4,
                    "timing_rating": "about_right",
                    "amount_rating": "about_right",
                    "confidence_rating": "about_right",
                    "suggested_change": "no_change",
                    "note": "Good.",
                },
            },
            {
                "scenario": {"name": "enter", "preset": "braver"},
                "answers": {
                    "overall_rating": 3,
                    "timing_rating": "too_early",
                    "amount_rating": "about_right",
                    "confidence_rating": "too_bold",
                    "suggested_change": "enter_later",
                    "note": "A bit early.",
                },
            },
            {
                "scenario": {"name": "build", "preset": "normal"},
                "answers": {
                    "overall_rating": 5,
                    "timing_rating": "about_right",
                    "amount_rating": "too_busy",
                    "confidence_rating": "about_right",
                    "suggested_change": "build_less",
                    "note": "Build too intense.",
                },
            },
        ]

    def test_summary_counts_total_entries(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        assert summary.total_entries == 3

    def test_summary_counts_by_scenario(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        assert summary.entries_by_scenario.get("enter") == 2
        assert summary.entries_by_scenario.get("build") == 1

    def test_summary_counts_by_preset(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        assert summary.entries_by_preset.get("normal") == 2
        assert summary.entries_by_preset.get("braver") == 1

    def test_average_rating_by_scenario(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        # enter: (4 + 3) / 2 = 3.5, build: 5 / 1 = 5.0
        assert summary.average_rating_by_scenario.get("enter") == 3.5
        assert summary.average_rating_by_scenario.get("build") == 5.0

    def test_average_rating_by_preset(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        # normal: (4 + 5) / 2 = 4.5, braver: 3 / 1 = 3.0
        assert summary.average_rating_by_preset.get("normal") == 4.5
        assert summary.average_rating_by_preset.get("braver") == 3.0

    def test_common_timing_complaints_detected(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        # The label converts "too_early" to "too early" (spaces)
        timing_complaints = [c for c in summary.common_timing_complaints
                             if "too early" in c]
        assert len(timing_complaints) >= 1

    def test_common_amount_complaints_detected(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        # The label converts "too_busy" to "too busy" (spaces)
        amount_complaints = [c for c in summary.common_amount_complaints
                             if "too busy" in c]
        assert len(amount_complaints) >= 1

    def test_common_confidence_complaints_detected(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        # The label converts "too_bold" to "too bold" (spaces)
        conf_complaints = [c for c in summary.common_confidence_complaints
                           if "too bold" in c]
        assert len(conf_complaints) >= 1

    def test_common_suggested_changes(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        # We have enter_later and build_less suggested changes
        suggested_texts = " ".join(summary.common_suggested_changes)
        assert "enter_later" in suggested_texts or "later" in suggested_texts or \
               "soon" in suggested_texts

    def test_repeated_issues_detected(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        assert len(summary.repeated_issues) >= 1

    def test_possible_tuning_directions(self, entries: list[dict]) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(entries)
        assert len(summary.possible_tuning_directions) >= 1


class TestSummarizeFeedbackBestWorst:
    @pytest.fixture
    def varied_entries(self) -> list[dict]:
        """Entries with varied ratings for best/worst ranking."""
        return [
            {"scenario": {"name": "enter", "preset": "normal"},
             "answers": {"overall_rating": 5, "timing_rating": "about_right",
                         "amount_rating": "about_right",
                         "confidence_rating": "about_right",
                         "suggested_change": "no_change", "note": ""}},
            {"scenario": {"name": "build", "preset": "normal"},
             "answers": {"overall_rating": 4, "timing_rating": "about_right",
                         "amount_rating": "about_right",
                         "confidence_rating": "about_right",
                         "suggested_change": "no_change", "note": ""}},
            {"scenario": {"name": "drop", "preset": "normal"},
             "answers": {"overall_rating": 2, "timing_rating": "about_right",
                         "amount_rating": "about_right",
                         "confidence_rating": "about_right",
                         "suggested_change": "no_change", "note": ""}},
            {"scenario": {"name": "final_bail", "preset": "normal"},
             "answers": {"overall_rating": 3, "timing_rating": "about_right",
                         "amount_rating": "about_right",
                         "confidence_rating": "about_right",
                         "suggested_change": "no_change", "note": ""}},
        ]

    def test_best_rated_scenarios(self, varied_entries) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(varied_entries)
        # Best should be enter (5.0)
        best_names = [name for name, _ in summary.best_rated_scenarios]
        assert "enter" in best_names

    def test_worst_rated_scenarios(self, varied_entries) -> None:
        from drummer.playtest_feedback import summarize_feedback_entries
        summary = summarize_feedback_entries(varied_entries)
        # Worst should be drop (2.0)
        worst_names = [name for name, _ in summary.worst_rated_scenarios]
        assert "drop" in worst_names


class TestExportFormats:
    @pytest.fixture
    def sample_summary(self):
        from drummer.playtest_feedback import PlaytestLearningSummary
        return PlaytestLearningSummary(
            total_entries=2,
            entries_by_scenario={"enter": 1, "build": 1},
            entries_by_preset={"normal": 2},
            average_rating_by_scenario={"enter": 4.5, "build": 3.0},
            average_rating_by_preset={"normal": 3.75},
            common_timing_complaints=["timing feels too early (×1)"],
            common_amount_complaints=["amount feels too busy (×1)"],
            common_confidence_complaints=["confidence feels too timid (×1)"],
            common_suggested_changes=["drummer enters too soon (×1)"],
            best_rated_scenarios=[("enter", 4.5)],
            worst_rated_scenarios=[("build", 3.0)],
            best_rated_presets=[("normal", 3.75)],
            worst_rated_presets=[("normal", 3.75)],
            repeated_issues=["ENTER often feels too early"],
            possible_tuning_directions=[
                "based on timing feedback (too_early ×1): "
                "increase entry confirmation or observation time"
            ],
        )

    def test_json_export_contains_key_fields(self, sample_summary) -> None:
        from drummer.playtest_feedback import export_learning_summary_json
        tmp = _tmp_path(suffix=".json")
        try:
            export_learning_summary_json(sample_summary, tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                data = json.load(f)
            assert data["total_entries"] == 2
            assert "entries_by_scenario" in data
            assert "entries_by_preset" in data
            assert "average_rating_by_scenario" in data
            assert "average_rating_by_preset" in data
            assert "common_timing_complaints" in data
            assert "common_amount_complaints" in data
            assert "common_confidence_complaints" in data
            assert "common_suggested_changes" in data
            assert "best_rated_scenarios" in data
            assert "worst_rated_scenarios" in data
            assert "best_rated_presets" in data
            assert "worst_rated_presets" in data
            assert "repeated_issues" in data
            assert "possible_tuning_directions" in data
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_markdown_export_contains_key_sections(self, sample_summary) -> None:
        from drummer.playtest_feedback import export_learning_summary_markdown
        tmp = _tmp_path(suffix=".md")
        try:
            export_learning_summary_markdown(sample_summary, tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                content = f.read()
            assert "# Pocket Drummer Playtest Learning Summary" in content
            assert "## Overall" in content
            assert "## Best Working Areas" in content
            assert "## Problem Areas" in content
            assert "## Repeated Complaints" in content
            assert "## Possible Tuning Directions" in content
            assert "## Raw Counts" in content
            assert "**Total feedback entries:** 2" in content
            assert "This summary is deterministic" in content
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)


class TestCLISummaryMode:
    def test_cli_summary_mode_runs_on_small_test_file(self) -> None:
        """Create a small test feedback file, then run summarise via the module."""
        import subprocess
        from drummer.playtest_feedback import (
            PlaytestScenario, PlaytestQuestionnaire,
            PlaytestDiagnosticsSummary, PlaytestFeedbackEntry,
            append_feedback_entry,
        )

        tmp_feedback = _tmp_path(suffix=".jsonl")
        tmp_json = _tmp_path(suffix=".json")
        tmp_md = _tmp_path(suffix=".md")
        try:
            # Write a dummy feedback entry
            sc = PlaytestScenario(
                name="enter", variation_name="stable_input",
                description="Test", preset="normal",
                mode="continuous", bars=8, bpm=120.0,
            )
            diag = PlaytestDiagnosticsSummary(
                total_events=10, first_enter_bar=2, first_build_bar=None,
                confidence_peak=0.9, phrase_marker_count=1,
                inferred_intents={"enter": 1},
                output_contracts_passed=True, drop_event_count=0,
                final_bail_event_count=0, bail_event_count=0,
            )
            answers = PlaytestQuestionnaire(
                overall_rating=4, timing_rating="about_right",
                amount_rating="about_right", confidence_rating="about_right",
                understood_rating="yes", suggested_change="no_change",
                note="CLI test.",
            )
            entry = PlaytestFeedbackEntry(scenario=sc, diagnostics=diag, answers=answers)
            append_feedback_entry(tmp_feedback, entry)

            # Run CLI with explicit --feedback-file
            result = subprocess.run(
                [
                    sys.executable, "demo_playtest_interview.py",
                    "--summarize-feedback",
                    "--feedback-file", tmp_feedback,
                    "--summary-json", tmp_json,
                    "--summary-md", tmp_md,
                ],
                capture_output=True, text=True, cwd=_PROJECT_ROOT,
            )
            assert result.returncode == 0, f"CLI failed: {result.stderr}"
            assert "FEEDBACK LEARNING SUMMARY" in result.stdout
            assert "Total feedback entries: 1" in result.stdout
            assert f"Reading feedback from: {tmp_feedback}" in result.stdout

            # Check JSON export was written
            assert os.path.exists(tmp_json)
            with open(tmp_json, "r") as f:
                data = json.load(f)
            assert data["total_entries"] == 1

            # Check MD export was written
            assert os.path.exists(tmp_md)
        finally:
            for p in (tmp_feedback, tmp_json, tmp_md):
                if os.path.exists(p):
                    os.remove(p)

    def test_summarize_defaults_to_playtest_feedback(self) -> None:
        """--summarize-feedback without --feedback-file uses default path."""
        import subprocess
        # Use a temp feedback file at the default path in the temp directory approach
        # Instead, test that the parser no longer requires --feedback-file
        import importlib as _il
        mod = _il.import_module("demo_playtest_interview")
        parser = mod.build_parser()
        # --summarize-feedback alone (no --feedback-file) should parse
        args = parser.parse_args(["--summarize-feedback"])
        assert args.summarize_feedback is True
        assert args.feedback_file is None  # resolves to default later

    def test_summarize_with_missing_file_is_friendly(self) -> None:
        """Missing feedback file prints friendly message, no traceback."""
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "demo_playtest_interview.py",
                "--summarize-feedback",
                "--feedback-file", "/no/such/file_xyz_999.jsonl",
            ],
            capture_output=True, text=True, cwd=_PROJECT_ROOT,
        )
        assert result.returncode == 0, f"CLI failed: {result.stderr}"
        assert "No feedback file found" in result.stdout
        assert "Run a playtest interview first" in result.stdout


# ---------------------------------------------------------------------------
# Output contract validation tests
# ---------------------------------------------------------------------------


class TestOutputContractValidation:
    """Tests that output_contracts_passed validates actual outputs, not just labels."""

    def test_drop_with_zero_events_fails_contract(self) -> None:
        """DROP section with 0 events should make output_contracts_passed False."""
        raw = [
            {"bar": 0, "section": "LISTEN", "event_count": 0, "is_drop": False,
             "is_bail": False, "is_final_bail": False, "intent": "listen",
             "inferred_intent": "listen", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 13, "section": "DROP", "event_count": 0, "is_drop": True,
             "is_bail": False, "is_final_bail": False, "intent": "drop",
             "inferred_intent": "drop", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 14, "section": "FINAL_BAIL", "event_count": 2, "is_drop": False,
             "is_bail": False, "is_final_bail": True, "intent": "final_bail",
             "inferred_intent": "final_bail", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 19, "section": "BAIL", "event_count": 0, "is_drop": False,
             "is_bail": True, "is_final_bail": False, "intent": "bail",
             "inferred_intent": "bail", "confidence": 0.0, "phrase_marker": "none"},
        ]
        summary = _extract_diagnostics_summary(raw)
        # DROP has 0 events — should fail
        assert summary.output_contracts_passed is False, (
            "DROP with 0 events should make output_contracts_passed=False"
        )

    def test_bail_with_events_fails_contract(self) -> None:
        """BAIL section with >0 events should make output_contracts_passed False."""
        raw = [
            {"bar": 0, "section": "LISTEN", "event_count": 0, "is_drop": False,
             "is_bail": False, "is_final_bail": False, "intent": "listen",
             "inferred_intent": "listen", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 13, "section": "DROP", "event_count": 2, "is_drop": True,
             "is_bail": False, "is_final_bail": False, "intent": "drop",
             "inferred_intent": "drop", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 14, "section": "FINAL_BAIL", "event_count": 2, "is_drop": False,
             "is_bail": False, "is_final_bail": True, "intent": "final_bail",
             "inferred_intent": "final_bail", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 19, "section": "BAIL", "event_count": 3, "is_drop": False,
             "is_bail": True, "is_final_bail": False, "intent": "bail",
             "inferred_intent": "bail", "confidence": 0.0, "phrase_marker": "none"},
        ]
        summary = _extract_diagnostics_summary(raw)
        assert summary.output_contracts_passed is False, (
            "BAIL with 3 events should make output_contracts_passed=False"
        )

    def test_final_bail_with_zero_events_fails_contract(self) -> None:
        """FINAL_BAIL section with 0 events should fail."""
        raw = [
            {"bar": 0, "section": "LISTEN", "event_count": 0, "is_drop": False,
             "is_bail": False, "is_final_bail": False, "intent": "listen",
             "inferred_intent": "listen", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 13, "section": "DROP", "event_count": 2, "is_drop": True,
             "is_bail": False, "is_final_bail": False, "intent": "drop",
             "inferred_intent": "drop", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 14, "section": "FINAL_BAIL", "event_count": 0, "is_drop": False,
             "is_bail": False, "is_final_bail": True, "intent": "final_bail",
             "inferred_intent": "final_bail", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 19, "section": "BAIL", "event_count": 0, "is_drop": False,
             "is_bail": True, "is_final_bail": False, "intent": "bail",
             "inferred_intent": "bail", "confidence": 0.0, "phrase_marker": "none"},
        ]
        summary = _extract_diagnostics_summary(raw)
        assert summary.output_contracts_passed is False, (
            "FINAL_BAIL with 0 events should make output_contracts_passed=False"
        )

    def test_all_contracts_valid_passes(self) -> None:
        """All three contracts valid should make output_contracts_passed True."""
        raw = [
            {"bar": 0, "section": "LISTEN", "event_count": 0, "is_drop": False,
             "is_bail": False, "is_final_bail": False, "intent": "listen",
             "inferred_intent": "listen", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 13, "section": "DROP", "event_count": 2, "is_drop": True,
             "is_bail": False, "is_final_bail": False, "intent": "drop",
             "inferred_intent": "drop", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 14, "section": "FINAL_BAIL", "event_count": 2, "is_drop": False,
             "is_bail": False, "is_final_bail": True, "intent": "final_bail",
             "inferred_intent": "final_bail", "confidence": 0.0, "phrase_marker": "none"},
            {"bar": 19, "section": "BAIL", "event_count": 0, "is_drop": False,
             "is_bail": True, "is_final_bail": False, "intent": "bail",
             "inferred_intent": "bail", "confidence": 0.0, "phrase_marker": "none"},
        ]
        summary = _extract_diagnostics_summary(raw)
        assert summary.output_contracts_passed is True, (
            "All contracts valid should make output_contracts_passed=True"
        )


# ---------------------------------------------------------------------------
# Simple ear-test feedback mode tests
# ---------------------------------------------------------------------------


class Test2QFeedbackMode:
    """Tests for the 2-question ear-test feedback mode (_build_2q_answers).

    These tests verify Q1 (good/bad/mixed/unsure), Q2 (score 1-5),
    and Q3 (optional obvious issue) produce correct PlaytestQuestionnaire.
    """

    def _mod(self):
        import importlib as _il
        return _il.import_module("demo_playtest_interview")

    # -- Q1 + score combinations --

    def test_q1_good_score_4(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("g", 4)
        assert answers.overall_rating == 4
        assert answers.understood_rating == "yes"
        assert answers.note == "Matthew judged the overall performance as good."

    def test_q1_bad_score_2(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("b", 2)
        assert answers.overall_rating == 2
        assert answers.understood_rating == "partly"
        assert answers.note == "Matthew judged the overall performance as bad."

    def test_q1_mixed_score_3(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("m", 3)
        assert answers.overall_rating == 3
        assert answers.understood_rating == "partly"
        assert answers.note == "Matthew judged the overall performance as mixed."

    def test_q1_unsure_score_3(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("u", 3)
        assert answers.overall_rating == 3
        assert answers.understood_rating == "partly"
        assert answers.note == "Matthew was unsure how to judge the overall performance."

    def test_q1_good_score_5_can_be_golden(self) -> None:
        """Golden eligibility: Q1=good + score=5."""
        mod = self._mod()
        answers = mod._build_2q_answers("g", 5)
        assert answers.overall_rating == 5
        assert answers.understood_rating == "yes"
        assert "good" in answers.note

    # -- Q3 issue appending --

    def test_q3_timing_appends_issue(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("g", 4, q3="t")
        assert "Obvious issue: timing felt off." in answers.note

    def test_q3_samey_appends_issue(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("m", 3, q3="s")
        assert "Obvious issue: too samey or boring." in answers.note

    def test_q3_decision_appends_issue(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("b", 2, q3="d")
        assert "Obvious issue: drummer decision felt wrong for the input." in answers.note

    def test_q3_none_appends_none(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("g", 4, q3="n")
        assert "Obvious issue: none." in answers.note

    def test_q3_empty_appends_nothing(self) -> None:
        mod = self._mod()
        answers = mod._build_2q_answers("g", 4, q3="")
        assert "Obvious issue:" not in answers.note

    # -- Aux fields are set sensibly --

    def test_timing_rating_always_about_right(self) -> None:
        for q1 in ("g", "b", "m", "u"):
            answers = self._mod()._build_2q_answers(q1, 3)
            assert answers.timing_rating == "about_right"

    def test_amount_rating_always_about_right(self) -> None:
        for q1 in ("g", "b", "m", "u"):
            answers = self._mod()._build_2q_answers(q1, 3)
            assert answers.amount_rating == "about_right"

    def test_confidence_rating_always_about_right(self) -> None:
        for q1 in ("g", "b", "m", "u"):
            answers = self._mod()._build_2q_answers(q1, 3)
            assert answers.confidence_rating == "about_right"

    # -- Suggested change mapping --

    def test_good_maps_to_no_change(self) -> None:
        answers = self._mod()._build_2q_answers("g", 4)
        assert answers.suggested_change == "no_change"

    def test_bad_maps_to_no_change(self) -> None:
        # sentiment captured in note, not suggested_change
        answers = self._mod()._build_2q_answers("b", 2)
        assert answers.suggested_change == "no_change"

    def test_mixed_maps_to_no_change(self) -> None:
        answers = self._mod()._build_2q_answers("m", 3)
        assert answers.suggested_change == "no_change"

    def test_unsure_maps_to_no_change(self) -> None:
        answers = self._mod()._build_2q_answers("u", 3)
        assert answers.suggested_change == "no_change"


class Test2QFeedbackCLI:
    """Tests for the CLI with 2-question feedback mode."""

    def _mod(self):
        import importlib as _il
        return _il.import_module("demo_playtest_interview")

    def test_detailed_feedback_flag_preserves_old_flow(self) -> None:
        mod = self._mod()
        parser = mod.build_parser()
        args = parser.parse_args(["--detailed-feedback", "--no-play", "--scenario", "enter"])
        assert args.detailed_feedback is True

    def test_default_mode_is_not_detailed(self) -> None:
        mod = self._mod()
        parser = mod.build_parser()
        args = parser.parse_args(["--no-play", "--scenario", "enter"])
        assert args.detailed_feedback is False

    def test_2q_helpers_exist(self) -> None:
        """_build_2q_answers, _prompt_q1/_q2/_q3 exist."""
        mod = self._mod()
        assert hasattr(mod, "_build_2q_answers")
        assert hasattr(mod, "_prompt_q1")
        assert hasattr(mod, "_prompt_q2")
        assert hasattr(mod, "_prompt_q3")
        assert hasattr(mod, "_print_compact_diagnostics")

    def test_compact_diagnostics_accepts_summary(self) -> None:
        mod = self._mod()
        from drummer.playtest_feedback import PlaytestDiagnosticsSummary
        summary = PlaytestDiagnosticsSummary(
            total_events=42,
            first_enter_bar=2,
            first_build_bar=None,
            confidence_peak=0.85,
            phrase_marker_count=2,
            inferred_intents={"listen": 3, "enter_soft": 1},
            output_contracts_passed=True,
            drop_event_count=0,
            final_bail_event_count=0,
            bail_event_count=0,
        )
        mod._print_compact_diagnostics(summary, sanity_passed=True)


# We need _PROJECT_ROOT for the CLI test
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
