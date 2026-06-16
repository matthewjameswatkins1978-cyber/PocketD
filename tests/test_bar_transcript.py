"""Tests for bar transcript module."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from drummer.bar_transcript import (
    BarLine,
    BarTranscript,
    build_bar_transcript,
    render_bar_transcript_text,
    render_bar_transcript_json,
    save_bar_transcript,
    _abbrev,
    _count_instrument,
    _build_note_positions,
    _compute_flags,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeEvent:
    """Minimal GrooveEvent stand-in for testing."""

    def __init__(self, instrument, grid_position, bar_index=0, velocity=100, timing_offset_ms=0.0):
        self.instrument = instrument
        self.grid_position = grid_position
        self.bar_index = bar_index
        self.velocity = velocity
        self.timing_offset_ms = timing_offset_ms


def _diags(n: int, sections=None, intents=None) -> list[dict]:
    """Build a list of per-bar diagnostic dicts."""
    result = []
    for i in range(n):
        section = sections[i] if sections and i < len(sections) else "MAINTAIN_1"
        intent = intents[i] if intents and i < len(intents) else "maintain"
        result.append({
            "bar": i,
            "section": section,
            "intent": intent,
            "rendered_intent": intent,
        })
    return result


def _tmp_path(suffix: str = ".tmp") -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestInstrumentAbbreviation:
    def test_kick_to_k(self) -> None:
        assert _abbrev("kick") == "K"

    def test_snare_to_s(self) -> None:
        assert _abbrev("snare") == "S"

    def test_hi_hat_to_h(self) -> None:
        assert _abbrev("hi_hat") == "H"

    def test_closed_hat_to_h(self) -> None:
        assert _abbrev("closed_hat") == "H"

    def test_ride_to_r(self) -> None:
        assert _abbrev("ride") == "R"

    def test_crash_to_c(self) -> None:
        assert _abbrev("crash") == "C"


class TestInstrumentCounting:
    def test_count_kick(self) -> None:
        events = [_FakeEvent("kick", 0), _FakeEvent("snare", 4), _FakeEvent("kick", 8)]
        assert _count_instrument(events, "kick") == 2

    def test_count_snare(self) -> None:
        events = [_FakeEvent("kick", 0), _FakeEvent("snare", 4), _FakeEvent("snare", 12)]
        assert _count_instrument(events, "snare") == 2

    def test_count_hat(self) -> None:
        events = [
            _FakeEvent("hi_hat", 0), _FakeEvent("closed_hat", 2),
            _FakeEvent("open_hat", 4), _FakeEvent("hi_hat", 6),
        ]
        assert _count_instrument(events, "hat") == 4

    def test_count_crash(self) -> None:
        events = [_FakeEvent("crash", 0)]
        assert _count_instrument(events, "crash") == 1

    def test_count_ride(self) -> None:
        events = [_FakeEvent("ride", 0), _FakeEvent("ride", 8)]
        assert _count_instrument(events, "ride") == 2


class TestNotePositionFormatting:
    def test_single_event(self) -> None:
        events = [_FakeEvent("kick", 0, velocity=118)]
        result = _build_note_positions(events)
        assert result == "K@0:118"

    def test_multiple_same_instrument_same_velocity(self) -> None:
        events = [
            _FakeEvent("hi_hat", 0, velocity=52),
            _FakeEvent("hi_hat", 2, velocity=52),
            _FakeEvent("hi_hat", 4, velocity=52),
            _FakeEvent("hi_hat", 6, velocity=52),
        ]
        result = _build_note_positions(events)
        assert result == "H@0&2&4&6:52"

    def test_mixed_instruments(self) -> None:
        events = [
            _FakeEvent("kick", 0, velocity=118),
            _FakeEvent("snare", 4, velocity=78),
            _FakeEvent("hi_hat", 0, velocity=52),
            _FakeEvent("hi_hat", 2, velocity=52),
            _FakeEvent("hi_hat", 4, velocity=52),
            _FakeEvent("hi_hat", 6, velocity=52),
        ]
        result = _build_note_positions(events)
        # H comes before K alphabetically
        assert "H@0&2&4&6:52" in result
        assert "K@0:118" in result
        assert "S@4:78" in result

    def test_empty_events(self) -> None:
        assert _build_note_positions([]) == "—"


class TestFlags:
    def _bar_data(self, **kw) -> dict:
        defaults = {"event_count": 0, "kick": 0, "snare": 0,
                     "hat": 0, "ride": 0, "crash": 0, "max_velocity": 0}
        defaults.update(kw)
        return defaults

    def test_loud_isolated_kick(self) -> None:
        events = [_FakeEvent("kick", 0, velocity=118)]
        all_bars = [self._bar_data(event_count=1, kick=1, max_velocity=118)]
        flags = _compute_flags(events, 0, all_bars)
        assert "LOUD_ISOLATED_KICK" in flags

    def test_repeated_loud_isolated_kick(self) -> None:
        events = [_FakeEvent("kick", 0, velocity=118)]
        all_bars = [
            self._bar_data(event_count=1, kick=1, max_velocity=118),
            self._bar_data(event_count=1, kick=1, max_velocity=118),
        ]
        flags = _compute_flags(events, 1, all_bars)
        assert "LOUD_ISOLATED_KICK" in flags
        assert "REPEATED_LOUD_ISOLATED_KICK" in flags

    def test_hat_8ths(self) -> None:
        events = [_FakeEvent("hi_hat", i * 2, velocity=60) for i in range(8)]
        all_bars = [self._bar_data(event_count=8, hat=8)]
        flags = _compute_flags(events, 0, all_bars)
        assert "HATS_8THS" in flags

    def test_hat_quarters(self) -> None:
        events = [_FakeEvent("hi_hat", i * 4, velocity=60) for i in range(4)]
        all_bars = [self._bar_data(event_count=4, hat=4)]
        flags = _compute_flags(events, 0, all_bars)
        assert "HATS_QUARTERS" in flags

    def test_no_snare(self) -> None:
        events = [_FakeEvent("hi_hat", 0), _FakeEvent("kick", 4)]
        all_bars = [self._bar_data(event_count=2, kick=1, hat=1)]
        flags = _compute_flags(events, 0, all_bars)
        assert "NO_SNARE" in flags

    def test_no_hats(self) -> None:
        events = [_FakeEvent("kick", 0), _FakeEvent("snare", 4)]
        all_bars = [self._bar_data(event_count=2, kick=1, snare=1)]
        flags = _compute_flags(events, 0, all_bars)
        assert "NO_HATS" in flags

    def test_very_sparse(self) -> None:
        events = [_FakeEvent("kick", 0)]
        all_bars = [self._bar_data(event_count=1, kick=1)]
        flags = _compute_flags(events, 0, all_bars)
        assert "VERY_SPARSE" in flags

    def test_busy_bar(self) -> None:
        events = [_FakeEvent("hi_hat", i, velocity=60) for i in range(12)]
        all_bars = [self._bar_data(event_count=12, hat=12)]
        flags = _compute_flags(events, 0, all_bars)
        assert "BUSY_BAR" in flags

    def test_crash_present(self) -> None:
        events = [_FakeEvent("crash", 0), _FakeEvent("kick", 0)]
        all_bars = [self._bar_data(event_count=2, kick=1, crash=1)]
        flags = _compute_flags(events, 0, all_bars)
        assert "CRASH_PRESENT" in flags

    def test_possible_fill(self) -> None:
        events = (
            [_FakeEvent("kick", i * 2, velocity=100) for i in range(4)]
            + [_FakeEvent("snare", i * 4 + 1, velocity=90) for i in range(3)]
            + [_FakeEvent("hi_hat", i, velocity=60) for i in range(3)]
        )
        all_bars = [self._bar_data(event_count=10, kick=4, snare=3, hat=3)]
        flags = _compute_flags(events, 0, all_bars)
        assert "POSSIBLE_FILL" in flags

    def test_hat_density_dropped(self) -> None:
        events = [_FakeEvent("hi_hat", i * 4, velocity=60) for i in range(4)]
        all_bars = [
            self._bar_data(event_count=8, hat=8),
            self._bar_data(event_count=4, hat=4),
        ]
        flags = _compute_flags(events, 1, all_bars)
        assert "HAT_DENSITY_DROPPED" in flags


class TestBuildBarTranscript:
    def test_groups_events_by_bar(self) -> None:
        events = [
            _FakeEvent("kick", 0, bar_index=0),
            _FakeEvent("snare", 4, bar_index=0),
            _FakeEvent("hi_hat", 0, bar_index=1),
            _FakeEvent("hi_hat", 2, bar_index=1),
        ]
        diags = _diags(2)
        transcript = build_bar_transcript(events, diags, "test", "normal", "test_var")
        assert transcript.bars == 2
        assert transcript.total_events == 4
        assert len(transcript.bar_lines) == 2
        assert transcript.bar_lines[0].event_count == 2
        assert transcript.bar_lines[1].event_count == 2

    def test_instrument_counts_per_bar(self) -> None:
        events = [
            _FakeEvent("kick", 0, bar_index=0),
            _FakeEvent("snare", 4, bar_index=0),
            _FakeEvent("hi_hat", 0, bar_index=0),
            _FakeEvent("hi_hat", 2, bar_index=0),
            _FakeEvent("hi_hat", 4, bar_index=0),
            _FakeEvent("hi_hat", 6, bar_index=0),
        ]
        diags = _diags(1)
        transcript = build_bar_transcript(events, diags, "test", "normal", "test_var")
        bl = transcript.bar_lines[0]
        assert bl.kick_count == 1
        assert bl.snare_count == 1
        assert bl.hat_count == 4
        assert bl.ride_count == 0
        assert bl.crash_count == 0

    def test_suspicious_bars_listed_in_summary(self) -> None:
        events = [
            _FakeEvent("kick", 0, bar_index=0, velocity=118),
        ]
        diags = _diags(1)
        transcript = build_bar_transcript(events, diags, "test", "normal", "test_var")
        assert len(transcript.suspicious) >= 1
        assert any("loud isolated kick" in s for s in transcript.suspicious)

    def test_scenario_preset_variation_stored(self) -> None:
        events: list = []
        diags = _diags(1)
        transcript = build_bar_transcript(events, diags, "drop", "cautious", "deliberate_sparse")
        assert transcript.scenario == "drop"
        assert transcript.preset == "cautious"
        assert transcript.variation == "deliberate_sparse"


class TestTextRenderer:
    def test_produces_output(self) -> None:
        transcript = BarTranscript(
            scenario="drop", preset="cautious", variation="test",
            bars=1, total_events=1,
            suspicious=["bar 0 loud isolated kick"],
            bar_lines=[
                BarLine(bar=0, section="DROP", rendered_intent="drop",
                        event_count=1, kick_count=1, snare_count=0, hat_count=0,
                        ride_count=0, crash_count=0, max_velocity=118, avg_velocity=118,
                        note_positions="K@0:118",
                        flags=["LOUD_ISOLATED_KICK"]),
            ],
        )
        text = render_bar_transcript_text(transcript)
        assert "BAR TRANSCRIPT:" in text
        assert "SUMMARY:" in text
        assert "suspicious=1" in text
        assert "SUSPICIOUS:" in text
        assert "LOUD_ISOLATED_KICK" in text

    def test_includes_header_row(self) -> None:
        transcript = BarTranscript(bars=0, total_events=0)
        text = render_bar_transcript_text(transcript)
        assert "Bar" in text
        assert "Section" in text

    def test_empty_transcript_has_summary(self) -> None:
        transcript = BarTranscript(bars=0, total_events=0)
        text = render_bar_transcript_text(transcript)
        assert "SUMMARY: bars=0 total_events=0 suspicious=0" in text


class TestJsonRenderer:
    def test_produces_valid_json(self) -> None:
        transcript = BarTranscript(
            scenario="drop", preset="cautious", variation="test",
            bars=1, total_events=1,
            bar_lines=[
                BarLine(bar=0, section="DROP", rendered_intent="drop",
                        event_count=1, kick_count=1, snare_count=0, hat_count=0,
                        ride_count=0, crash_count=0, max_velocity=118, avg_velocity=118,
                        note_positions="K@0:118",
                        flags=["LOUD_ISOLATED_KICK"]),
            ],
        )
        json_str = render_bar_transcript_json(transcript)
        data = json.loads(json_str)
        assert data["scenario"] == "drop"
        assert data["total_events"] == 1
        assert len(data["bar_lines"]) == 1
        assert data["bar_lines"][0]["flags"] == ["LOUD_ISOLATED_KICK"]

    def test_json_includes_bar_line_fields(self) -> None:
        transcript = BarTranscript(
            bars=1, total_events=2,
            bar_lines=[
                BarLine(bar=0, section="TEST", rendered_intent="test",
                        event_count=2, kick_count=1, snare_count=1, hat_count=0,
                        ride_count=0, crash_count=0, max_velocity=100, avg_velocity=90,
                        note_positions="K@0:100,S@4:80"),
            ],
        )
        json_str = render_bar_transcript_json(transcript)
        data = json.loads(json_str)
        bl = data["bar_lines"][0]
        assert bl["section"] == "TEST"
        assert bl["note_positions"] == "K@0:100,S@4:80"
        assert "flags" in bl


class TestFileOutput:
    def test_save_creates_both_files(self) -> None:
        transcript = BarTranscript(
            scenario="drop", preset="cautious", bars=1, total_events=0,
            bar_lines=[BarLine(bar=0, section="TEST", rendered_intent="test",
                        event_count=0, kick_count=0, snare_count=0, hat_count=0,
                        ride_count=0, crash_count=0, max_velocity=0, avg_velocity=0,
                        note_positions="—")],
        )
        txt_path = _tmp_path(suffix=".txt")
        json_path = _tmp_path(suffix=".json")
        try:
            save_bar_transcript(transcript, txt_path, json_path)
            assert os.path.exists(txt_path)
            assert os.path.exists(json_path)
            with open(txt_path) as f:
                assert "BAR TRANSCRIPT:" in f.read()
            with open(json_path) as f:
                data = json.load(f)
                assert data["scenario"] == "drop"
        finally:
            for p in (txt_path, json_path):
                if os.path.exists(p):
                    os.remove(p)

    def test_save_creates_parent_dirs(self) -> None:
        tmp_dir = _tmp_path(suffix="")
        os.remove(tmp_dir)
        txt_path = os.path.join(tmp_dir, "nested", "test.txt")
        json_path = os.path.join(tmp_dir, "nested", "test.json")
        transcript = BarTranscript(bars=0, total_events=0)
        try:
            save_bar_transcript(transcript, txt_path, json_path)
            assert os.path.exists(txt_path)
            assert os.path.exists(json_path)
        finally:
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)