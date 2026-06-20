"""Tests for Simple Brain trace renderer.

Verifies the pure renderer produces correct output for:
- Empty trace
- Single-row trace
- Multi-row trace with all columns
- Long reason truncation
- Determinism (same input → same output)
"""

from __future__ import annotations

from drummer.simple_brain_trace import render_simple_brain_trace_table


# ---------------------------------------------------------------------------
# Empty trace
# ---------------------------------------------------------------------------


def test_empty_trace_renders_cleanly() -> None:
    """Empty trace produces a short header-only message."""
    result = render_simple_brain_trace_table([])
    assert "SIMPLE BRAIN TRACE" in result
    assert "(empty)" in result
    assert len(result.splitlines()) <= 2


# ---------------------------------------------------------------------------
# Single row
# ---------------------------------------------------------------------------


def test_single_row_includes_all_columns() -> None:
    """One trace row renders with action, beat, confidence, feature cols, reason."""
    trace = [
        {
            "bar": 0,
            "section": "intro",
            "action": "LISTEN",
            "beat": None,
            "confidence": 0.15,
            "input_density": 0.05,
            "player_certainty": 0.15,
            "stability": 0.10,
            "change_score": 0.02,
            "silence": 1.5,
            "reason": "listening: confidence dipped, lock reset",
        }
    ]
    result = render_simple_brain_trace_table(trace)
    lines = result.splitlines()

    # Header lines.
    assert "SIMPLE BRAIN TRACE" in lines[0]
    assert "bar" in lines[1]
    assert "section" in lines[1]
    assert "action" in lines[1]
    assert "beat" in lines[1]
    assert "conf" in lines[1]
    assert "dens" in lines[1]
    assert "cert" in lines[1]
    assert "stab" in lines[1]
    assert "chg" in lines[1]
    assert "sil" in lines[1]
    assert "reason" in lines[1]

    # Data row.
    data_line = lines[3]
    assert "LISTEN" in data_line
    assert "none" in data_line
    assert "0.15" in data_line  # confidence
    assert "0.05" in data_line
    assert "1.5" in data_line
    # Reason truncated if needed.
    assert "confidence dipped" in data_line or "listening" in data_line


# ---------------------------------------------------------------------------
# Multi-row trace
# ---------------------------------------------------------------------------


def test_multi_row_trace() -> None:
    """Three rows — each with different action and beat."""
    trace = [
        {
            "bar": 0,
            "section": "intro",
            "action": "LISTEN",
            "beat": None,
            "confidence": 0.10,
            "reason": "listening: 1/4 confident snapshots",
        },
        {
            "bar": 1,
            "section": "verse",
            "action": "CHOOSE",
            "beat": "simple_rock",
            "confidence": 0.93,
            "reason": "choosing: locked; best match is simple_rock",
        },
        {
            "bar": 2,
            "section": "verse",
            "action": "HOLD",
            "beat": "simple_rock",
            "confidence": 0.92,
            "reason": "holding: best candidate is still simple_rock",
        },
    ]
    result = render_simple_brain_trace_table(trace)
    lines = result.splitlines()

    # Should have 6 lines: title, header, separator, then 3 data rows.
    assert len(lines) >= 6
    assert any("LISTEN" in line for line in lines[3:])
    assert any("CHOOSE" in line for line in lines[3:])
    assert any("HOLD" in line for line in lines[3:])
    assert any("simple_rock" in line for line in lines[3:])


# ---------------------------------------------------------------------------
# Missing keys — doesn't crash
# ---------------------------------------------------------------------------


def test_missing_keys_default_gracefully() -> None:
    """Empty dict produces a row of zeros without exceptions."""
    trace = [{}]
    result = render_simple_brain_trace_table(trace)
    lines = result.splitlines()
    # Data row exists and contains default zeros.
    assert len(lines) >= 4
    assert "0.00" in lines[3]


# ---------------------------------------------------------------------------
# Long reason truncation
# ---------------------------------------------------------------------------


def test_long_reason_is_truncated() -> None:
    """A reason over 60 characters is truncated with '...'."""
    long_reason = (
        "holding: change_score 0.05 < threshold 0.30; "
        "confidence 0.08 < switch minimum 0.30; "
        "score delta 0.26 < switch threshold 0.35; "
        "best candidate is still half_time; "
        "and many more details follow"
    )
    trace = [
        {
            "bar": 0,
            "section": "drop",
            "action": "HOLD",
            "beat": "half_time",
            "confidence": 0.47,
            "reason": long_reason,
        }
    ]
    result = render_simple_brain_trace_table(trace)
    lines = result.splitlines()
    data_line = lines[3]
    # The reason column in the data line must end with "..." or be shorter.
    reasons = [part for part in data_line.split("  ") if len(part) > 30]
    if reasons:
        rendered_reason = reasons[-1].strip()
        assert len(rendered_reason) <= 60
        # If the original reason was longer than 60 chars, truncation happened.
        if len(long_reason) > 60:
            assert rendered_reason.endswith("...")


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_renderer_is_deterministic() -> None:
    """Same trace produces identical output on every call."""
    trace = [
        {
            "bar": 0,
            "section": "verse",
            "action": "HOLD",
            "beat": "simple_rock",
            "confidence": 0.930,
            "input_density": 0.80,
            "player_certainty": 0.70,
            "stability": 0.65,
            "change_score": 0.05,
            "silence": 0.0,
            "reason": "holding: best candidate is still simple_rock",
        }
    ]
    result1 = render_simple_brain_trace_table(trace)
    result2 = render_simple_brain_trace_table(trace)
    result3 = render_simple_brain_trace_table(trace)
    assert result1 == result2 == result3


# ---------------------------------------------------------------------------
# None beat renders as "none"
# ---------------------------------------------------------------------------


def test_none_beat_renders_as_none() -> None:
    """A None beat_name renders as the string 'none'."""
    trace = [
        {
            "bar": 0,
            "action": "LISTEN",
            "beat": None,
            "confidence": 0.10,
            "reason": "listening",
        }
    ]
    result = render_simple_brain_trace_table(trace)
    assert "none" in result