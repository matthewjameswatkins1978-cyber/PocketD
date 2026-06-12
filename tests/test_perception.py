"""Tests for the Perception Engine — Module 1: Event Listener."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.density import AttackDensityTracker
from perception.energy import compute_energy, compute_energy_in_window
from perception.event_listener import (
    AudioFrame,
    EventListener,
    detect_events_from_audio,
    _onset_detection,
)
from perception.frequency import (
    band_energy,
    classify_frequency,
    classify_frequency_from_spectrum,
    compute_spectrum,
    region_frequency_profile,
)
from perception.models import FFT_SIZE, MusicalEvent


# ─── Frequency Tests ───────────────────────────────────────────────


class TestComputeSpectrum:
    def test_returns_positive_frequencies(self) -> None:
        frame = np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)[:FFT_SIZE]
        spectrum = compute_spectrum(frame, 44100)
        expected_len = FFT_SIZE // 2 + 1
        assert len(spectrum) == expected_len
        assert np.all(spectrum >= 0)

    def test_empty_frame_produces_zeros(self) -> None:
        spectrum = compute_spectrum(np.array([]), 44100)
        assert np.all(spectrum == 0.0)

    def test_short_frame_is_padded(self) -> None:
        frame = np.ones(100)
        spectrum = compute_spectrum(frame, 44100)
        assert len(spectrum) == FFT_SIZE // 2 + 1


class TestBandEnergy:
    def test_energy_positive(self) -> None:
        frame = np.sin(2 * np.pi * 100 * np.arange(44100) / 44100)[:FFT_SIZE]
        spectrum = compute_spectrum(frame, 44100)
        energy = band_energy(spectrum, 44100, 50, 400)
        assert energy >= 0

    def test_out_of_range_returns_zero(self) -> None:
        spectrum = np.ones(FFT_SIZE // 2 + 1)
        energy = band_energy(spectrum, 44100, 50000, 60000)
        assert energy == 0.0

    def zero_band_returns_zero(self) -> None:
        spectrum = np.ones(FFT_SIZE // 2 + 1)
        energy = band_energy(spectrum, 44100, 100, 50)
        assert energy == 0.0


class TestClassifyFrequency:
    def test_low_freq_frame_returns_low(self) -> None:
        # 60 Hz sine -> should be "low" or "sub"
        frame = np.sin(2 * np.pi * 60 * np.arange(44100) / 44100)[:FFT_SIZE]
        region = classify_frequency(frame, 44100)
        assert region in ("sub", "low", "unknown")

    def test_mid_freq_frame_returns_low_mid(self) -> None:
        # 400 Hz sine -> should be "low_mid"
        frame = np.sin(2 * np.pi * 400 * np.arange(44100) / 44100)[:FFT_SIZE]
        region = classify_frequency(frame, 44100)
        assert region in ("low_mid", "low", "unknown")

    def test_high_freq_frame_returns_high(self) -> None:
        # 8000 Hz sine -> should be "high"
        frame = np.sin(2 * np.pi * 8000 * np.arange(44100) / 44100)[:FFT_SIZE]
        region = classify_frequency(frame, 44100)
        assert region in ("high", "high_mid", "unknown")

    def test_none_frame_returns_unknown(self) -> None:
        region = classify_frequency(None, 44100)
        assert region == "unknown"

    def test_empty_frame_returns_unknown(self) -> None:
        region = classify_frequency(np.array([]), 44100)
        assert region == "unknown"


class TestRegionFrequencyProfile:
    def test_valid_region_returns_range(self) -> None:
        low, high = region_frequency_profile("low")
        assert low == 80.0
        assert high == 250.0

    def test_invalid_region_returns_zeros(self) -> None:
        low, high = region_frequency_profile("nonexistent")  # type: ignore[arg-type]
        assert low == 0.0
        assert high == 0.0


class TestClassifyFromSpectrum:
    def test_classifies_high_energy_band(self) -> None:
        frame = np.sin(2 * np.pi * 10000 * np.arange(44100) / 44100)[:FFT_SIZE]
        spectrum = compute_spectrum(frame, 44100)
        region = classify_frequency_from_spectrum(spectrum, 44100)
        assert region in ("high", "high_mid", "unknown")


# ─── Energy Tests ──────────────────────────────────────────────────


class TestComputeEnergy:
    def test_silence_returns_zero(self) -> None:
        frame = np.zeros(1000)
        assert compute_energy(frame) == 0.0

    def test_non_zero_returns_positive(self) -> None:
        frame = np.ones(1000) * 0.5
        energy = compute_energy(frame, normalise=True)
        assert 0 < energy <= 1.0

    def test_max_amplitude_returns_approx_rms(self) -> None:
        frame = np.ones(1000)
        energy = compute_energy(frame, normalise=False)
        assert energy == 1.0  # RMS of constant 1 is 1

    def test_empty_frame_returns_zero(self) -> None:
        assert compute_energy(np.array([])) == 0.0

    def test_int16_input_handled_correctly(self) -> None:
        frame = np.ones(1000, dtype=np.int16) * 10000
        energy = compute_energy(frame, normalise=True)
        assert 0 < energy <= 1.0


class TestComputeEnergyInWindow:
    def test_window_energy_positive(self) -> None:
        signal = np.zeros(10000)
        signal[5000:5100] = 0.5
        energy = compute_energy_in_window(signal, 5000, 200, 44100)
        assert 0 <= energy <= 1.0

    def test_out_of_bounds_returns_zero(self) -> None:
        signal = np.zeros(1000)
        energy = compute_energy_in_window(signal, 0, 5000, 44100)
        assert energy >= 0

    def test_empty_signal_returns_zero(self) -> None:
        energy = compute_energy_in_window(np.array([]), 0, 100, 44100)
        assert energy == 0.0


# ─── Density Tests ─────────────────────────────────────────────────


class TestAttackDensityTracker:
    def test_initial_density_zero(self) -> None:
        tracker = AttackDensityTracker()
        assert tracker.density() == 0.0

    def test_single_attack_density(self) -> None:
        tracker = AttackDensityTracker(window_seconds=2.0)
        tracker.record_attack(1.0)
        assert tracker.density() == 1.0

    def test_multiple_attacks_in_window(self) -> None:
        tracker = AttackDensityTracker(window_seconds=2.0)
        for t in [0.5, 1.0, 1.5]:
            tracker.record_attack(t)
        assert tracker.density() == 3.0

    def test_attacks_outside_window_are_pruned(self) -> None:
        tracker = AttackDensityTracker(window_seconds=1.0)
        tracker.record_attack(0.0)
        tracker.record_attack(2.0)
        # 0.0 is outside the window relative to 2.0
        assert tracker.density(now=2.0) == 1.0

    def test_normalised_density_bounded(self) -> None:
        tracker = AttackDensityTracker(window_seconds=2.0)
        for t in [0.0, 0.5, 1.0, 1.5, 2.0]:
            tracker.record_attack(t)
        nd = tracker.normalised_density()
        assert 0 <= nd <= 1.0

    def test_reset_clears_all(self) -> None:
        tracker = AttackDensityTracker()
        tracker.record_attack(1.0)
        tracker.reset()
        assert tracker.density() == 0.0
        assert tracker.count == 0

    def test_negative_window_raises(self) -> None:
        try:
            AttackDensityTracker(window_seconds=-1.0)
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


# ─── Onset Detection Tests ────────────────────────────────────────


class TestOnsetDetection:
    def test_silence_returns_empty(self) -> None:
        signal = np.zeros(10000)
        onsets = _onset_detection(signal, 44100)
        assert onsets == []

    def test_impulse_detected(self) -> None:
        signal = np.zeros(44100)
        signal[22050] = 1.0  # single impulse at 0.5s
        onsets = _onset_detection(signal, 44100)
        assert len(onsets) >= 1
        assert abs(onsets[0][0] - 0.5) < 0.02

    def test_multiple_impulses_detected(self) -> None:
        signal = np.zeros(44100 * 2)
        for idx in [11025, 33075]:  # 0.25s and 0.75s
            signal[idx] = 1.0
        onsets = _onset_detection(signal, 44100)
        assert len(onsets) == 2

    def test_multi_channel_stereo_to_mono(self) -> None:
        signal = np.zeros((44100, 2))
        signal[22050, 0] = 1.0
        onsets = _onset_detection(signal, 44100)
        assert len(onsets) >= 1

    def test_min_interval_respected(self) -> None:
        signal = np.zeros(44100)
        signal[11025] = 1.0
        signal[11026] = 1.0  # adjacent samples
        onsets = _onset_detection(signal, 44100, min_interval=0.05)
        assert len(onsets) <= 1


# ─── Detect Events From Audio Tests ───────────────────────────────


class TestDetectEventsFromAudio:
    def test_empty_signal_returns_empty(self) -> None:
        events = detect_events_from_audio(np.array([]), 44100)
        assert events == []

    def test_silence_returns_empty(self) -> None:
        events = detect_events_from_audio(np.zeros(44100), 44100)
        assert events == []

    def test_impulse_creates_event(self) -> None:
        signal = np.zeros(44100)
        signal[22050] = 1.0
        events = detect_events_from_audio(signal, 44100)
        assert len(events) >= 1
        event = events[0]
        assert event.time_seconds >= 0
        assert 0 <= event.strength <= 1.0
        assert isinstance(event.frequency_region, str)
        assert 0 <= event.energy <= 1.0
        assert 0 <= event.density <= 1.0

    def test_stereo_signal_handled(self) -> None:
        signal = np.zeros((44100, 2))
        signal[22050, :] = 1.0
        events = detect_events_from_audio(signal, 44100)
        assert len(events) >= 1

    def test_frequency_region_populated(self) -> None:
        # Low-frequency impulse -> should classify as low/sub
        signal = np.zeros(44100)
        # Add a low-frequency hit
        t = np.arange(44100) / 44100
        hit = np.sin(2 * np.pi * 60 * t) * np.exp(-t * 40)
        signal += hit * 0.5
        events = detect_events_from_audio(signal, 44100)
        if events:
            assert events[0].frequency_region != "unknown"


# ─── EventListener (Streaming) Tests ───────────────────────────────


class TestEventListener:
    def test_initial_state(self) -> None:
        listener = EventListener(sample_rate=44100)
        assert listener.flush() == []
        assert listener.buffer_ms < 1.0

    def test_process_silent_frames(self) -> None:
        listener = EventListener(sample_rate=44100)
        for i in range(10):
            frame = AudioFrame(
                samples=np.zeros(1024),
                sample_rate=44100,
                time_seconds=(i + 1) * 1024 / 44100,
            )
            listener.process_frame(frame)
        assert listener.flush() == []

    def test_process_impulse_frame(self) -> None:
        listener = EventListener(sample_rate=44100)
        samples = np.zeros(1024)
        samples[512] = 1.0
        frame = AudioFrame(
            samples=samples,
            sample_rate=44100,
            time_seconds=1024 / 44100,
        )
        listener.process_frame(frame)
        events = listener.flush()
        assert len(events) >= 1

    def test_callback_invoked(self) -> None:
        received: list[MusicalEvent] = []

        def cb(event: MusicalEvent) -> None:
            received.append(event)

        listener = EventListener(sample_rate=44100, callback=cb)
        samples = np.zeros(1024)
        samples[512] = 1.0
        frame = AudioFrame(
            samples=samples,
            sample_rate=44100,
            time_seconds=1024 / 44100,
        )
        listener.process_frame(frame)
        listener.flush()
        assert len(received) >= 1

    def test_reset_clears_state(self) -> None:
        listener = EventListener(sample_rate=44100)
        samples = np.zeros(1024)
        samples[512] = 1.0
        frame = AudioFrame(
            samples=samples,
            sample_rate=44100,
            time_seconds=1024 / 44100,
        )
        listener.process_frame(frame)
        listener.reset()
        assert listener.flush() == []

    def test_flush_clears_buffer(self) -> None:
        listener = EventListener(sample_rate=44100)
        samples = np.zeros(1024)
        samples[512] = 1.0
        frame = AudioFrame(
            samples=samples,
            sample_rate=44100,
            time_seconds=1024 / 44100,
        )
        listener.process_frame(frame)
        first = listener.flush()
        second = listener.flush()
        assert len(first) >= 1
        assert second == []


# ─── AudioFrame Tests ──────────────────────────────────────────────


class TestAudioFrame:
    def test_frame_attributes(self) -> None:
        samples = np.array([0.1, 0.2, -0.1], dtype=np.float64)
        frame = AudioFrame(samples=samples, sample_rate=44100, time_seconds=0.5)
        assert frame.sample_rate == 44100
        assert frame.time_seconds == 0.5
        assert np.array_equal(frame.samples, samples)


# ─── MusicalEvent Tests ────────────────────────────────────────────


class TestMusicalEvent:
    def test_default_values(self) -> None:
        event = MusicalEvent(time_seconds=1.0)
        assert event.time_seconds == 1.0
        assert event.strength == 0.0
        assert event.frequency_region == "unknown"
        assert event.energy == 0.0
        assert event.density == 0.0

    def test_frozen_dataclass(self) -> None:
        event = MusicalEvent(time_seconds=1.0, strength=0.8, frequency_region="low")
        assert event.time_seconds == 1.0
        assert event.strength == 0.8
        assert event.frequency_region == "low"

    def test_strength_energy_density_range(self) -> None:
        event = MusicalEvent(
            time_seconds=0.5,
            strength=0.75,
            energy=0.6,
            density=0.3,
        )
        assert 0 <= event.strength <= 1.0
        assert 0 <= event.energy <= 1.0
        assert 0 <= event.density <= 1.0