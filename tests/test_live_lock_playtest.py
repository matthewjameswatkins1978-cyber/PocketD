"""Tests for the pure helpers in demo_live_lock_playtest.py.

No sounddevice or MIDI hardware is required — all helpers are tested
with synthetic data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_live_lock_playtest import (
    build_artifact,
    compute_bpm_error,
    compute_intervals,
    compute_median_interval,
    compute_timing_diagnostics,
    save_artifact,
)


# ── compute_intervals ────────────────────────────────────────────────────────


def test_compute_intervals_empty() -> None:
    assert compute_intervals([]) == []
    assert compute_intervals([1.0]) == []


def test_compute_intervals_normal() -> None:
    result = compute_intervals([0.0, 0.5, 1.0, 1.5])
    assert result == [0.5, 0.5, 0.5]


def test_compute_intervals_irregular() -> None:
    result = compute_intervals([0.0, 0.48, 1.02, 1.49])
    assert len(result) == 3
    assert result[0] == 0.48
    assert result[1] == 0.54
    assert result[2] == 0.47


# ── compute_median_interval ──────────────────────────────────────────────────


def test_median_interval_empty() -> None:
    assert compute_median_interval([]) is None


def test_median_interval_single() -> None:
    assert compute_median_interval([0.5]) == 0.5


def test_median_interval_odd() -> None:
    assert compute_median_interval([0.5, 0.6, 0.4]) == 0.5


def test_median_interval_even() -> None:
    result = compute_median_interval([0.4, 0.5, 0.6, 0.7])
    assert result == 0.55


# ── compute_bpm_error ────────────────────────────────────────────────────────


def test_bpm_error_both_none() -> None:
    assert compute_bpm_error(None, None) is None


def test_bpm_error_target_none() -> None:
    assert compute_bpm_error(None, 118.5) is None


def test_bpm_error_detected_none() -> None:
    assert compute_bpm_error(120.0, None) is None


def test_bpm_error_signed_positive() -> None:
    # detected faster than target
    assert compute_bpm_error(120.0, 122.3) == 2.3


def test_bpm_error_signed_negative() -> None:
    # detected slower than target
    assert compute_bpm_error(70.0, 69.4) == -0.6


def test_bpm_error_exact_match() -> None:
    assert compute_bpm_error(100.0, 100.0) == 0.0


# ── compute_timing_diagnostics ───────────────────────────────────────────────


def _make_timing_log(
    pairs: list[tuple[float, float]],
) -> list[tuple[float, float, str, int, int]]:
    """Build a minimal timing log: each pair is (target, actual) for a note-on."""
    log: list[tuple[float, float, str, int, int]] = []
    for target, actual in pairs:
        log.append((target, actual, "on", 36, 100))
        log.append((target + 0.09, actual + 0.09, "off", 36, 0))
    return log


def test_timing_diagnostics_none() -> None:
    total, note_on_count, mean_abs, max_abs = compute_timing_diagnostics(None)
    assert total == 0
    assert note_on_count == 0
    assert mean_abs is None
    assert max_abs is None


def test_timing_diagnostics_empty() -> None:
    total, note_on_count, mean_abs, max_abs = compute_timing_diagnostics([])
    assert total == 0
    assert note_on_count == 0
    assert mean_abs is None
    assert max_abs is None


def test_timing_diagnostics_perfect_timing() -> None:
    log = _make_timing_log([(0.0, 0.0), (0.5, 0.5), (1.0, 1.0)])
    total, note_on_count, mean_abs, max_abs = compute_timing_diagnostics(log)
    assert total == 6  # 3 on + 3 off
    assert note_on_count == 3
    assert mean_abs == 0.0
    assert max_abs == 0.0


def test_timing_diagnostics_with_errors() -> None:
    log = _make_timing_log(
        [
            (0.0, 0.001),   # 1 ms late
            (0.5, 0.497),   # 3 ms early
            (1.0, 1.005),   # 5 ms late
        ]
    )
    total, note_on_count, mean_abs, max_abs = compute_timing_diagnostics(log)
    assert total == 6
    assert note_on_count == 3
    # errors: 1ms, 3ms, 5ms => mean = 3.0, max = 5.0
    assert mean_abs == 3.0
    assert max_abs == 5.0


def test_timing_diagnostics_only_off_events() -> None:
    """Edge case: timing log with only note-off events — no note-ons."""
    log: list[tuple[float, float, str, int, int]] = [
        (0.09, 0.09, "off", 36, 0),
        (0.59, 0.59, "off", 38, 0),
    ]
    total, note_on_count, mean_abs, max_abs = compute_timing_diagnostics(log)
    assert total == 2
    assert note_on_count == 0
    assert mean_abs is None
    assert max_abs is None


# ── build_artifact (no playback) ─────────────────────────────────────────────


def test_build_artifact_no_play() -> None:
    artifact = build_artifact(
        timestamp="2026-06-19T00:00:00+00:00",
        device_name="AG06/AG03",
        device_index=1,
        channel=1,
        duration=12.0,
        sample_rate=44100,
        threshold=0.3,
        min_signal_peak=0.001,
        target_bpm=120.0,
        detected_bpm=118.5,
        onset_times=[0.0, 0.506, 1.012, 1.518],
        notes="test take",
        played=False,
        midi_port=None,
        repeats=4,
        timing_log=None,
    )

    assert artifact["timestamp"] == "2026-06-19T00:00:00+00:00"
    assert artifact["device_name"] == "AG06/AG03"
    assert artifact["device_index"] == 1
    assert artifact["channel"] == 1
    assert artifact["duration"] == 12.0
    assert artifact["sample_rate"] == 44100
    assert artifact["threshold"] == 0.3
    assert artifact["min_signal_peak"] == 0.001
    assert artifact["target_bpm"] == 120.0
    assert artifact["detected_bpm"] == 118.5
    assert artifact["bpm_error"] == -1.5
    assert artifact["clap_count"] == 4
    assert artifact["clap_onset_times"] == [0.0, 0.506, 1.012, 1.518]
    assert len(artifact["clap_intervals"]) == 3  # type: ignore[arg-type]
    assert artifact["median_interval"] > 0.5  # type: ignore[operator]
    assert artifact["notes"] == "test take"
    assert artifact["played"] is False
    assert artifact["midi_port"] is None
    assert artifact["repeats"] == 4

    # No MIDI keys when played=False
    assert "midi_event_count" not in artifact
    assert "midi_note_on_count" not in artifact
    assert "midi_note_on_mean_abs_error_ms" not in artifact
    assert "midi_note_on_max_abs_error_ms" not in artifact


def test_build_artifact_no_target_bpm() -> None:
    artifact = build_artifact(
        timestamp="2026-06-19T00:00:00+00:00",
        device_name="Test Device",
        device_index=None,
        channel=2,
        duration=8.0,
        sample_rate=48000,
        threshold=0.5,
        min_signal_peak=0.01,
        target_bpm=None,
        detected_bpm=105.0,
        onset_times=[0.0, 0.571, 1.143],
        notes="",
        played=False,
        midi_port=None,
        repeats=2,
        timing_log=None,
    )

    assert artifact["target_bpm"] is None
    assert artifact["bpm_error"] is None
    assert artifact["device_index"] is None


def test_build_artifact_no_detected_bpm() -> None:
    artifact = build_artifact(
        timestamp="2026-06-19T00:00:00+00:00",
        device_name="AG06/AG03",
        device_index=1,
        channel=1,
        duration=4.0,
        sample_rate=44100,
        threshold=0.3,
        min_signal_peak=0.001,
        target_bpm=120.0,
        detected_bpm=None,
        onset_times=[0.5],
        notes="single clap",
        played=False,
        midi_port=None,
        repeats=1,
        timing_log=None,
    )

    assert artifact["detected_bpm"] is None
    assert artifact["bpm_error"] is None
    assert artifact["clap_count"] == 1
    assert artifact["clap_intervals"] == []
    assert artifact["median_interval"] is None


# ── build_artifact (with playback) ───────────────────────────────────────────


def test_build_artifact_with_play() -> None:
    timing_log = _make_timing_log(
        [
            (0.0, 0.000),
            (0.5, 0.501),
            (1.0, 0.998),
        ]
    )
    artifact = build_artifact(
        timestamp="2026-06-19T00:00:00+00:00",
        device_name="AG06/AG03",
        device_index=1,
        channel=1,
        duration=8.0,
        sample_rate=44100,
        threshold=0.3,
        min_signal_peak=0.001,
        target_bpm=120.0,
        detected_bpm=120.0,
        onset_times=[0.0, 0.5, 1.0, 1.5],
        notes="with MIDI",
        played=True,
        midi_port="PocketDrummer Out",
        repeats=4,
        timing_log=timing_log,
    )

    assert artifact["played"] is True
    assert artifact["midi_port"] == "PocketDrummer Out"
    assert artifact["midi_event_count"] == 6  # 3 on + 3 off
    assert artifact["midi_note_on_count"] == 3
    assert isinstance(artifact["midi_note_on_mean_abs_error_ms"], float)
    assert isinstance(artifact["midi_note_on_max_abs_error_ms"], float)


# ── save_artifact ────────────────────────────────────────────────────────────


def test_save_artifact_creates_directory_and_file(tmp_path: Path) -> None:
    custom_dir = tmp_path / "custom_artifacts"
    artifact = {
        "timestamp": "2026-06-19T00-00-00",
        "test": True,
    }

    result_path = save_artifact(artifact, path=custom_dir / "test_artifact.json")

    assert result_path.exists()
    assert result_path.suffix == ".json"

    with open(result_path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["timestamp"] == "2026-06-19T00-00-00"
    assert loaded["test"] is True


def test_save_artifact_with_default_path(tmp_path: Path, monkeypatch) -> None:
    """Use a temporary ARTIFACT_DIR so we don't write into the real project."""
    import demo_live_lock_playtest as mod

    monkeypatch.setattr(mod, "ARTIFACT_DIR", tmp_path / "live_lock_tests")

    artifact = {
        "timestamp": "2026-06-19T00-00-00",
        "value": 42,
    }
    result_path = save_artifact(artifact, path=None)

    assert result_path.exists()
    assert result_path.parent == tmp_path / "live_lock_tests"

    with open(result_path, encoding="utf-8") as fh:
        loaded = json.load(fh)
    assert loaded["value"] == 42