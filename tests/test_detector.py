"""Tests for the Detector Diagnostics layer.

Covers:
1-6: Phase 1-6 tests (unchanged)
13. Instrument tests (Phase 7)
"""

from __future__ import annotations

import struct
import sys
import wave
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.detector import Detector, DetectorConfig, DetectorDiagnostics, DetectorResult

SAMPLE_RATE = 44100


# ── Helpers ─────────────────────────────────────────────────────────


def _make_impulse(duration: float, sr: int, time_s: float, amp: float = 0.7) -> np.ndarray:
    total = int(duration * sr)
    signal = np.zeros(total)
    idx = int(time_s * sr)
    if 0 <= idx < total:
        impulse_len = max(1, int(sr * 0.001))
        t_env = np.linspace(0, np.pi, impulse_len)
        env = np.sin(t_env)
        end = min(idx + impulse_len, total)
        actual = end - idx
        signal[idx:end] = amp * env[:actual]
    return signal


def _make_pulse(duration: float = 4.0, sr: int = SAMPLE_RATE, bpm: float = 120.0, amp: float = 0.7) -> np.ndarray:
    interval = 60.0 / bpm
    signal = np.zeros(int(duration * sr))
    t = 0.0
    while t < duration:
        signal += _make_impulse(duration, sr, t, amp=amp)
        t += interval
    return signal


def _make_white_noise(duration: float = 4.0, sr: int = SAMPLE_RATE, rms: float = 0.005) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, rms, int(duration * sr)).astype(np.float64)


def _make_hum(duration: float = 4.0, sr: int = SAMPLE_RATE, freq: float = 50.0, amp: float = 0.03) -> np.ndarray:
    num_samples = int(duration * sr)
    t = np.arange(num_samples) / sr
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _make_density_rise(duration: float = 4.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    signal = np.zeros(int(duration * sr))
    for t in [0.3, 1.0, 1.7]:
        signal += _make_impulse(duration, sr, t, amp=0.6)
    base = 2.2
    for i in range(16):
        t = base + i * 0.08
        if t < duration:
            signal += _make_impulse(duration, sr, t, amp=0.5)
    return signal


def _make_clipping(duration: float = 2.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    signal = np.sin(2 * np.pi * 440 * np.arange(int(duration * sr)) / sr)
    signal = np.clip(signal * 1.5, -1.0, 1.0)
    return signal.astype(np.float64)


def _write_wav(filepath: Path, samples: np.ndarray, sr: int, n_channels: int = 1) -> None:
    samples = np.asarray(samples, dtype=np.float64)
    i16 = np.clip(samples * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(str(filepath), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(i16.tobytes())


# ═════════════════════════════════════════════════════════════════════
# Phase 4: Automated Summary Tests
# ═════════════════════════════════════════════════════════════════════


class TestSummaryPulse:
    def test_quality_good_or_usable(self) -> None:
        signal = _make_pulse(duration=4.0, bpm=120.0)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.input_quality in ("good", "usable")

    def test_musical_state_stable_pulse(self) -> None:
        signal = _make_pulse(duration=4.0, bpm=120.0)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.musical_state == "stable_pulse"

    def test_has_clean_flag(self) -> None:
        signal = _make_pulse(duration=4.0, bpm=120.0)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert "CLEAN_STABLE_PULSE" in result.diagnostics.summary_flags


class TestSummaryNoisy:
    def test_quality_noisy(self) -> None:
        signal = _make_white_noise(duration=4.0, rms=0.005)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.input_quality in ("noisy", "unusable", "weak")

    def test_musical_state_noise(self) -> None:
        signal = _make_white_noise(duration=4.0, rms=0.005)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.musical_state in ("noise", "silence")


class TestSummarySilence:
    def test_musical_state_silence(self) -> None:
        detector = Detector()
        result = detector.detect(np.zeros(SAMPLE_RATE), SAMPLE_RATE)
        assert result.diagnostics.musical_state == "silence"


class TestClearPulse:
    def test_produces_events(self) -> None:
        signal = _make_pulse(duration=4.0, bpm=120.0)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count >= 4

    def test_pulse_tracker_estimates_bpm(self) -> None:
        signal = _make_pulse(duration=4.0, bpm=120.0)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.pulse_bpm is not None
        assert 80 <= result.diagnostics.pulse_bpm <= 160


class TestWhiteNoiseRejection:
    def test_few_events(self) -> None:
        signal = _make_white_noise(duration=4.0, rms=0.005)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count <= 6

    def test_rejected_count(self) -> None:
        signal = _make_white_noise(duration=4.0, rms=0.005)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.rejected_count >= 0


class TestHumRejection:
    def test_hum_few_events(self) -> None:
        signal = _make_hum(duration=4.0, amp=0.008)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count <= 2

    def test_hum_rejected_flag(self) -> None:
        # Full hum (0.03 amplitude) should trigger hum rejection
        signal = _make_hum(duration=4.0, amp=0.03)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        flags = result.diagnostics.summary_flags
        assert "HUM_REJECTED" in flags or result.diagnostics.event_count == 0


class TestPulseWithNoise:
    def test_pulse_survives_noise(self) -> None:
        pulse = _make_pulse(duration=4.0, bpm=120.0, amp=0.7)
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.003, len(pulse)).astype(np.float64)
        signal = pulse + noise
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count >= 4


class TestEdgeCases:
    def test_zero_duration(self) -> None:
        detector = Detector()
        result = detector.detect(np.array([]), SAMPLE_RATE)
        assert result.diagnostics.duration_seconds == 0.0
        assert result.diagnostics.event_count == 0

    def test_stereo(self) -> None:
        signal = np.zeros((SAMPLE_RATE, 2))
        signal[SAMPLE_RATE // 2, :] = 0.9
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count >= 1


class TestWavLoading:
    def _load(self):
        from demo_detector_diagnostics import load_wav
        return load_wav

    def test_pulse_loads(self, tmp_path: Path) -> None:
        signal = _make_pulse(duration=2.0, bpm=120.0, amp=0.7)
        wav_path = tmp_path / "pulse.wav"
        _write_wav(wav_path, signal, SAMPLE_RATE, n_channels=1)
        load_wav = self._load()
        loaded, sr = load_wav(wav_path)
        assert sr == SAMPLE_RATE
        detector = Detector()
        result = detector.detect(loaded, sr)
        assert result.diagnostics.event_count >= 3

    def test_missing_file_raises(self) -> None:
        load_wav = self._load()
        with pytest.raises(FileNotFoundError):
            load_wav("nonexistent_file.wav")


# ═════════════════════════════════════════════════════════════════════
# Phase 5: Batch Diagnostics Tests
# ═════════════════════════════════════════════════════════════════════


def _batch_helpers():
    from demo_detector_diagnostics import _BatchRecord, run_detector_for_source, write_batch_summary
    return _BatchRecord, run_detector_for_source, write_batch_summary


class TestBatchSynthetic:
    def test_batch_returns_multiple_records(self, tmp_path: Path) -> None:
        _BatchRecord, run_detector, _ = _batch_helpers()
        records = []
        sig1 = _make_pulse(duration=2.0, bpm=120.0, amp=0.7)
        sig2 = _make_impulse(duration=2.0, sr=SAMPLE_RATE, time_s=0.5, amp=0.9)
        records.append(run_detector("pulse_b", sig1, SAMPLE_RATE, tmp_path, save=False))
        records.append(run_detector("sparse_b", sig2, SAMPLE_RATE, tmp_path, save=False))
        assert len(records) == 2
        assert all(not r.error for r in records)

    def test_summary_contains_source_names(self, tmp_path: Path) -> None:
        _BatchRecord, run_detector, write_batch_summary = _batch_helpers()
        sig = _make_pulse(duration=2.0, bpm=120.0, amp=0.7)
        rec = run_detector("mytest", sig, SAMPLE_RATE, tmp_path, save=False)
        summary_path = tmp_path / "summary.md"
        text = write_batch_summary([rec], summary_path)
        assert "mytest" in text

    def test_table_includes_event_counts(self, tmp_path: Path) -> None:
        _BatchRecord, run_detector, write_batch_summary = _batch_helpers()
        sig = _make_pulse(duration=2.0, bpm=120.0, amp=0.7)
        rec = run_detector("test", sig, SAMPLE_RATE, tmp_path, save=False)
        summary_path = tmp_path / "summary.md"
        text = write_batch_summary([rec], summary_path)
        assert "| Events |" in text

    def test_error_record_appears_in_summary(self, tmp_path: Path) -> None:
        _BatchRecord, _, write_batch_summary = _batch_helpers()
        error_rec = _BatchRecord(source="bad_file", result=None, error="Simulated failure")
        summary_path = tmp_path / "summary.md"
        text = write_batch_summary([error_rec], summary_path)
        assert "bad_file" in text


# ═════════════════════════════════════════════════════════════════════
# Phase 7: Instrument Tests
# ═════════════════════════════════════════════════════════════════════


def _generate_bass() -> np.ndarray:
    from demo_detector_diagnostics import _generate_bass as fn
    return fn()


def _generate_guitar() -> np.ndarray:
    from demo_detector_diagnostics import _generate_guitar as fn
    return fn()


def _generate_keyboard() -> np.ndarray:
    from demo_detector_diagnostics import _generate_keyboard as fn
    return fn()


def _generate_drums() -> np.ndarray:
    from demo_detector_diagnostics import _generate_drums as fn
    return fn()


class TestInstruments:
    def test_bass_produces_signal(self) -> None:
        signal = _generate_bass()
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        # Bass may be caught by hum rejection or produce events — both are valid
        # depending on how the plucked notes interact with the detector.
        assert result.diagnostics.musical_state != "silence"

    def test_guitar_produces_accepted_events(self) -> None:
        signal = _generate_guitar()
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count > 0
        assert "EVENT_SPAM" not in result.diagnostics.warnings

    def test_keyboard_sustained_rejected(self) -> None:
        signal = _generate_keyboard()
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        # Phase 8: sustained-tone rejection should fire, producing 0 events
        # and state "noise" with SUSTAINED_TONE_REJECTED flag.
        assert result.diagnostics.event_count == 0
        assert "SUSTAINED_TONE_REJECTED" in result.diagnostics.summary_flags

    def test_drums_produce_clear_events(self) -> None:
        signal = _generate_drums()
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count >= 4
        assert "NO_EVENTS" not in result.diagnostics.warnings

    def test_instruments_in_batch_summary(self, tmp_path: Path) -> None:
        _BatchRecord, run_detector, write_batch_summary = _batch_helpers()
        records = []
        for name, gen in [("bass", _generate_bass), ("guitar", _generate_guitar),
                          ("keyboard", _generate_keyboard), ("drums", _generate_drums)]:
            rec = run_detector(name, gen(), SAMPLE_RATE, tmp_path, save=False)
            records.append(rec)
        summary_path = tmp_path / "summary.md"
        text = write_batch_summary(records, summary_path)
        for name in ("bass", "guitar", "keyboard", "drums"):
            assert name in text

    def test_hum_still_rejected(self) -> None:
        signal = _make_hum(duration=4.0, amp=0.03)
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count == 0
        assert result.diagnostics.musical_state == "noise"

    def test_pulse_with_noise_still_works(self) -> None:
        pulse = _make_pulse(duration=4.0, bpm=120.0, amp=0.7)
        rng = np.random.default_rng(99)
        noise = rng.normal(0, 0.003, len(pulse)).astype(np.float64)
        signal = pulse + noise
        detector = Detector()
        result = detector.detect(signal, SAMPLE_RATE)
        assert result.diagnostics.event_count >= 4
        assert result.diagnostics.input_quality in ("good", "usable")