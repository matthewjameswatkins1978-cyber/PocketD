"""Tests for drummer.engine_trace — Engine Decision Trace table renderer."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from drummer.engine_trace import render_engine_trace_table


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------


def _make_trace_entry(
    bar: int = 0,
    section: str = "LISTEN",
    selected: str = "listen",
    rendered: str = "listen",
    reason: str = "Feature LISTEN: repetition_stability below threshold",
    confidence: float = 0.3,
) -> dict:
    """Build a minimal trace entry for testing."""
    return {
        "bar": bar,
        "time": bar * 2.0,
        "section": section,
        "previous_intent": "listen",
        "selected_intent": selected,
        "rendered_intent": rendered,
        "arrangement_intent": rendered,
        "decision_confidence": confidence,
        "decision_reason": reason,
        "decision_scores": {"repetition_stability": 0.1, "player_certainty": 0.0},
        "input_density": 0.0,
        "strength_ema": 0.0,
        "fast_strength_ema": 0.0,
        "slow_strength_ema": 0.0,
        "change_score": 0.0,
        "silence_duration": 20.0,
        "player_certainty": 0.22,
        "repetition_stability": 0.0,
        "phase_alignment": 0.75,
        "has_entered": False,
        "anchor_bar_count": 0,
        "confidence_state_confidence": 0.0,
        "confidence_state_stable_bars": 0,
        "confidence_state_unstable_bars": 0,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderEngineTraceTable:
    """Tests for render_engine_trace_table()."""

    def test_renders_header_for_empty_trace(self):
        """Empty trace returns a simple placeholder."""
        result = render_engine_trace_table([])
        assert "ENGINE TRACE" in result
        assert "(empty)" in result

    def test_renders_column_headers(self):
        """Table includes expected column headers."""
        trace = [_make_trace_entry(bar=0)]
        result = render_engine_trace_table(trace)
        assert "bar" in result
        assert "section" in result
        assert "selected" in result
        assert "rendered" in result
        assert "conf" in result
        assert "dens" in result
        assert "cert" in result
        assert "stab" in result
        assert "chg" in result
        assert "sil" in result
        assert "reason" in result

    def test_renders_bar_data(self):
        """Table includes values from trace entries."""
        trace = [_make_trace_entry(bar=0, section="LISTEN")]
        result = render_engine_trace_table(trace)
        assert "  0  " in result  # bar number
        assert "LISTEN" in result  # section name

    def test_renders_multiple_bars(self):
        """Table includes all bars in order."""
        trace = [
            _make_trace_entry(bar=0, section="LISTEN"),
            _make_trace_entry(bar=1, section="LISTEN"),
            _make_trace_entry(bar=2, section="MAINTAIN", rendered="maintain"),
        ]
        result = render_engine_trace_table(trace)
        assert "  0  " in result
        assert "  1  " in result
        assert "  2  " in result
        assert "LISTEN" in result
        assert "MAINTAIN" in result


class TestOverrideDetection:
    """Tests for override display in the trace table."""

    def test_override_marked_when_selected_differs_from_rendered(self):
        """If selected != rendered, the row shows OVERRIDE."""
        trace = [
            _make_trace_entry(
                bar=2,
                section="MAINTAIN",
                selected="enter_soft",
                rendered="maintain",
                reason="Feature ENTER: sustained repetition stability",
            ),
        ]
        result = render_engine_trace_table(trace)
        assert "OVERRIDE" in result
        assert "engine selected enter_soft" in result

    def test_no_override_when_selected_equals_rendered(self):
        """If selected == rendered, the row shows the reason, not OVERRIDE."""
        trace = [
            _make_trace_entry(
                bar=2,
                section="MAINTAIN",
                selected="maintain",
                rendered="maintain",
                reason="Feature MAINTAIN: holding the pocket",
            ),
        ]
        result = render_engine_trace_table(trace)
        assert "OVERRIDE" not in result
        assert "holding the pocket" in result

    def test_override_detected_with_long_names_before_truncation(self):
        """OVERRIDE is detected using raw values, not truncated display names.

        Two long intent names that differ only after the 12-char
        truncation window still correctly trigger an OVERRIDE.
        """
        trace = [
            _make_trace_entry(
                bar=2,
                section="MAINTAIN",
                selected="enter_soft_super_long_name",
                rendered="enter_full_super_long_name",
                reason="Feature ENTER: something",
            ),
        ]
        result = render_engine_trace_table(trace)
        # Raw values differ → OVERRIDE must appear
        assert "OVERRIDE" in result
        # The OVERRIDE message uses the raw selected value
        assert "engine selected enter_soft_super_long_name" in result

    def test_override_not_spuriously_triggered_by_truncation_match(self):
        """When raw values are identical but long, no OVERRIDE appears.

        Both truncate to the same 12-char display but the raw values
        are identical, so there is no override.
        """
        trace = [
            _make_trace_entry(
                bar=2,
                section="MAINTAIN",
                selected="maintain_long_name_that_truncates",
                rendered="maintain_long_name_that_truncates",
                reason="Feature MAINTAIN: holding the pocket",
            ),
        ]
        result = render_engine_trace_table(trace)
        assert "OVERRIDE" not in result
        assert "holding the pocket" in result


class TestReasonShortening:
    """Tests for reason text shortening."""

    def test_strips_feature_prefix(self):
        """'Feature LISTEN: ' prefix is stripped."""
        trace = [
            _make_trace_entry(
                reason="Feature LISTEN: repetition_stability below threshold",
            ),
        ]
        result = render_engine_trace_table(trace)
        assert "Feature LISTEN:" not in result
        assert "repetition_stability below threshold" in result

    def test_truncates_long_reason(self):
        """Long reasons get truncated with '...'."""
        long_reason = "Feature MAINTAIN: " + "x" * 80
        trace = [_make_trace_entry(reason=long_reason)]
        result = render_engine_trace_table(trace)
        # After stripping prefix, remaining text truncated to 60 chars + "..."
        assert "..." in result

    def test_uses_ascii_truncation(self):
        """Truncation uses ASCII '...' (three dots), not Unicode ellipsis."""
        long_reason = "Feature MAINTAIN: " + "x" * 80
        trace = [_make_trace_entry(reason=long_reason)]
        result = render_engine_trace_table(trace)
        assert "..." in result
        assert "\u2026" not in result  # no Unicode ellipsis character


class TestNoOutputChange:
    """Verify trace collection does not change musical output."""

    def test_trace_collection_does_not_alter_events(self):
        """Running with engine_trace=[] produces same global_events as without."""
        from demo_continuous_jam_midi import run_continuous_jam

        # Run without trace
        _, diags_no_trace, events_no_trace = run_continuous_jam(
            bars=8, bpm=120.0, mode="scripted", preset_name="cautious",
            scenario="enter",
        )

        # Run with trace
        trace: list[dict] = []
        _, diags_with_trace, events_with_trace = run_continuous_jam(
            bars=8, bpm=120.0, mode="scripted", preset_name="cautious",
            scenario="enter", engine_trace=trace,
        )

        # Trace should have 8 entries
        assert len(trace) == 8

        # Global events should be identical
        assert len(events_no_trace) == len(events_with_trace)
        for a, b in zip(events_no_trace, events_with_trace):
            assert a.instrument == b.instrument
            assert a.grid_position == b.grid_position
            assert a.velocity == b.velocity
            assert a.bar_index == b.bar_index

        # Diagnostics should be identical
        assert len(diags_no_trace) == len(diags_with_trace)
        for a, b in zip(diags_no_trace, diags_with_trace):
            assert a["bar"] == b["bar"]
            assert a["event_count"] == b["event_count"]
            assert a["intent"] == b["intent"]


class TestCliPrintEngineTrace:
    """Integration tests for --print-engine-trace CLI flag."""

    def test_cli_flag_prints_table(self):
        """--print-engine-trace produces 'ENGINE TRACE' in stdout."""
        result = subprocess.run(
            [sys.executable, "demo_continuous_jam_midi.py",
             "--scenario", "drop", "--preset", "cautious",
             "--bars", "8", "--no-play", "--print-engine-trace"],
            capture_output=True, text=True,
            cwd=Path(__file__).resolve().parent.parent,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "ENGINE TRACE" in result.stdout
        assert "bar" in result.stdout
        assert "section" in result.stdout
        assert "selected" in result.stdout

    def test_cli_with_engine_trace_and_print(self):
        """--engine-trace PATH with --print-engine-trace exports and prints."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tf:
            trace_path = tf.name

        try:
            result = subprocess.run(
                [sys.executable, "demo_continuous_jam_midi.py",
                 "--scenario", "drop", "--preset", "cautious",
                 "--bars", "8", "--no-play",
                 "--engine-trace", trace_path,
                 "--print-engine-trace"],
                capture_output=True, text=True,
                cwd=Path(__file__).resolve().parent.parent,
            )
            assert result.returncode == 0, f"stderr: {result.stderr}"
            assert "ENGINE TRACE" in result.stdout
            with open(trace_path, "r") as f:
                trace_data = json.load(f)
            assert isinstance(trace_data, list)
            assert len(trace_data) == 8
        finally:
            Path(trace_path).unlink(missing_ok=True)