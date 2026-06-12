"""Event Listener — the core of Module 1.

Transforms raw audio into a stream of structured MusicalEvent objects.
Supports both offline (full buffer) and streaming (frame-by-frame) modes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from perception.density import AttackDensityTracker
from perception.energy import compute_energy, compute_energy_in_window
from perception.frequency import classify_frequency, compute_spectrum
from perception.models import FFT_SIZE, HOP_SIZE, FrequencyRegion, MusicalEvent

log = logging.getLogger(__name__)


@dataclass
class AudioFrame:
    """A single frame of audio for streaming processing.

    Parameters
    ----------
    samples : np.ndarray
        Audio samples (1-D float array in [-1, 1]).
    sample_rate : int
        Sample rate of the audio.
    time_seconds : float
        The timestamp of the *end* of this frame.
    """
    samples: np.ndarray
    sample_rate: int
    time_seconds: float


# Default onset detection parameters
DEFAULT_THRESHOLD_MULTIPLIER: float = 0.75
DEFAULT_STD_MULTIPLIER: float = 0.35
DEFAULT_MIN_INTERVAL: float = 0.05  # 50 ms minimum between events
DEFAULT_ENERGY_WINDOW: float = 0.05  # 50 ms window for energy computation


def _onset_detection(
    signal: np.ndarray,
    sample_rate: int,
    threshold_mult: float = DEFAULT_THRESHOLD_MULTIPLIER,
    std_mult: float = DEFAULT_STD_MULTIPLIER,
    min_interval: float = DEFAULT_MIN_INTERVAL,
) -> list[tuple[float, float]]:
    """Detect onset positions using spectral flux / energy rise.

    Returns a list of (time_seconds, strength) pairs.
    This is an enhanced version of the original onset_detector.py logic,
    operating on float arrays in [-1, 1].
    """
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    if signal.size < 4:
        return []

    diff = np.abs(np.diff(signal))
    window = max(3, min(25, int(sample_rate * 0.005)))
    smoothed = np.convolve(diff, np.ones(window) / window, mode="same")

    mean_val = float(np.mean(smoothed))
    std_val = float(np.std(smoothed))
    threshold = max(mean_val * threshold_mult + std_val * std_mult, 1e-6)

    onsets: list[tuple[float, float]] = []
    for idx in range(1, len(smoothed) - 1):
        if (
            smoothed[idx] > smoothed[idx - 1]
            and smoothed[idx] >= smoothed[idx + 1]
            and smoothed[idx] > threshold
        ):
            timestamp = idx / sample_rate
            if not onsets or timestamp - onsets[-1][0] >= min_interval:
                strength = float(min(1.0, smoothed[idx] / max(threshold, 1e-6)))
                onsets.append((timestamp, strength))

    return onsets


def _extract_attack_frame(
    signal: np.ndarray,
    sample_rate: int,
    onset_sample: int,
    frame_size: int = FFT_SIZE,
) -> np.ndarray:
    """Extract a short frame around an onset for spectral analysis."""
    half = frame_size // 2
    start = max(0, onset_sample - half)
    end = min(len(signal), onset_sample + half)
    frame = signal[start:end]
    if len(frame) < frame_size:
        padded = np.zeros(frame_size)
        padded[: len(frame)] = frame
        return padded
    return frame[:frame_size]


def detect_events_from_audio(
    signal: np.ndarray,
    sample_rate: int,
    threshold_mult: float = DEFAULT_THRESHOLD_MULTIPLIER,
    std_mult: float = DEFAULT_STD_MULTIPLIER,
    min_interval: float = DEFAULT_MIN_INTERVAL,
    energy_window_samples: int | None = None,
) -> list[MusicalEvent]:
    """Detect musical events from a full audio buffer (offline mode).

    This is the primary entry point for offline analysis. It processes a
    complete audio signal and returns a list of detected MusicalEvent objects.

    Parameters
    ----------
    signal : np.ndarray
        Audio samples (1-D or 2-D with shape (samples, channels)).
    sample_rate : int
        Sample rate in Hz.
    threshold_mult : float
        Multiplier for mean energy in onset threshold (default 0.75).
    std_mult : float
        Multiplier for std energy in onset threshold (default 0.35).
    min_interval : float
        Minimum seconds between onsets (default 0.05).
    energy_window_samples : int | None
        Window size for local energy computation (default = sample_rate * 0.05).

    Returns
    -------
    list[MusicalEvent]
        Detected musical events, in chronological order.
    """
    if signal.size == 0:
        return []

    # Convert to float mono
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if data.ndim > 1:
        data = data.flatten()

    # Detect onset positions
    onsets = _onset_detection(
        data,
        sample_rate,
        threshold_mult=threshold_mult,
        std_mult=std_mult,
        min_interval=min_interval,
    )

    if energy_window_samples is None:
        energy_window_samples = int(sample_rate * DEFAULT_ENERGY_WINDOW)
    energy_window_samples = max(1, energy_window_samples)

    density_tracker = AttackDensityTracker(window_seconds=2.0)

    events: list[MusicalEvent] = []
    for timestamp, strength in onsets:
        # Compute frequency region from a small frame around the onset
        onset_sample = int(timestamp * sample_rate)
        attack_frame = _extract_attack_frame(data, sample_rate, onset_sample)
        region = classify_frequency(attack_frame, sample_rate)

        # Compute local RMS energy
        local_energy = compute_energy_in_window(
            data, onset_sample, energy_window_samples, sample_rate,
        )

        # Track density
        density_tracker.record_attack(timestamp)
        norm_density = density_tracker.normalised_density()

        # Convert strength to percentage scale (0-100) for output
        strength_pct = round(strength * 100)

        event = MusicalEvent(
            time_seconds=timestamp,
            strength=strength_pct / 100.0,
            frequency_region=region,
            energy=local_energy,
            density=norm_density,
        )
        events.append(event)

    return events


class EventListener:
    """Streaming event listener for real-time audio processing.

    Processes audio frame-by-frame and invokes a callback for each
    detected musical event.

    Parameters
    ----------
    sample_rate : int
        Sample rate of the incoming audio.
    callback : Callable[[MusicalEvent], None] | None
        Called for each detected event. If None, events are buffered
        and can be retrieved via ``flush()``.
    threshold_mult : float
        Onset detection threshold multiplier (default 0.75).
    std_mult : float
        Onset detection std multiplier (default 0.35).
    min_interval : float
        Minimum interval between onsets in seconds (default 0.05).
    """

    def __init__(
        self,
        sample_rate: int,
        callback: Callable[[MusicalEvent], None] | None = None,
        threshold_mult: float = DEFAULT_THRESHOLD_MULTIPLIER,
        std_mult: float = DEFAULT_STD_MULTIPLIER,
        min_interval: float = DEFAULT_MIN_INTERVAL,
    ) -> None:
        self._sample_rate = sample_rate
        self._callback = callback
        self._threshold_mult = threshold_mult
        self._std_mult = std_mult
        self._min_interval = min_interval

        self._buffer: list[float] = []
        self._events: list[MusicalEvent] = []
        self._density_tracker = AttackDensityTracker(window_seconds=2.0)
        self._last_onset_time: float | None = None
        # Energy window samples
        self._energy_window = max(1, int(sample_rate * DEFAULT_ENERGY_WINDOW))

        # Ring buffer for spectral analysis — keep last FFT_SIZE samples
        self._spectral_buffer: list[float] = []

        log.info(
            "EventListener initialised — %d Hz, min_interval=%.3fs",
            sample_rate,
            min_interval,
        )

    def process_frame(self, frame: AudioFrame) -> None:
        """Process a single audio frame and detect any onset events."""
        samples = frame.samples.flatten() if frame.samples.ndim > 1 else frame.samples
        # Convert to float if needed
        samples_f = samples.astype(np.float64)

        # Apply a simple high-pass filter to reduce DC offset
        # (first-order difference)
        diff_samples = np.abs(np.diff(samples_f, prepend=samples_f[0:1]))

        # Update buffers
        self._buffer.extend(diff_samples.tolist())
        self._spectral_buffer.extend(samples_f.tolist())

        # Keep spectral buffer manageable
        max_spectral = FFT_SIZE * 4
        if len(self._spectral_buffer) > max_spectral:
            self._spectral_buffer = self._spectral_buffer[-max_spectral:]

        # Process diff buffer in chunks
        buffer_array = np.array(self._buffer, dtype=np.float64)
        if len(buffer_array) < 3:
            return

        # Compute threshold
        mean_val = float(np.mean(buffer_array))
        std_val = float(np.std(buffer_array))
        threshold = max(
            mean_val * self._threshold_mult + std_val * self._std_mult,
            1e-6,
        )

        # Smooth and find peaks
        window_size = max(3, min(25, int(self._sample_rate * 0.005)))
        if len(buffer_array) >= window_size:
            smoothed = np.convolve(
                buffer_array,
                np.ones(window_size) / window_size,
                mode="same",
            )

            # Check the most recent few samples for a peak
            check_start = max(len(smoothed) - len(samples_f) - 2, 1)
            for idx in range(check_start, len(smoothed) - 1):
                if (
                    smoothed[idx] > smoothed[idx - 1]
                    and smoothed[idx] >= smoothed[idx + 1]
                    and smoothed[idx] > threshold
                ):
                    timestamp = (idx / self._sample_rate) + (
                        frame.time_seconds
                        - len(samples_f) / self._sample_rate
                    )

                    # Enforce min interval
                    if (
                        self._last_onset_time is None
                        or timestamp - self._last_onset_time >= self._min_interval
                    ):
                        strength = float(
                            min(1.0, smoothed[idx] / max(threshold, 1e-6))
                        )

                        # Determine frequency region from spectral buffer
                        onset_offset = len(self._spectral_buffer) - (
                            len(buffer_array) - idx
                        )
                        onset_offset = max(0, min(onset_offset, len(self._spectral_buffer) - 1))

                        # Grab a frame around the onset
                        onset_signal = self._spectral_buffer[
                            max(0, onset_offset - FFT_SIZE // 2):
                            onset_offset + FFT_SIZE // 2
                        ]
                        if len(onset_signal) < FFT_SIZE:
                            onset_signal = self._spectral_buffer[-FFT_SIZE:]

                        region = classify_frequency(
                            np.array(onset_signal, dtype=np.float64),
                            self._sample_rate,
                        )

                        # Local energy from spectral buffer
                        local_energy = 0.0
                        if len(self._spectral_buffer) >= self._energy_window:
                            half = self._energy_window // 2
                            e_start = max(0, onset_offset - half)
                            e_end = min(len(self._spectral_buffer), onset_offset + half)
                            if e_end > e_start:
                                local_energy = compute_energy(
                                    np.array(
                                        self._spectral_buffer[e_start:e_end],
                                        dtype=np.float64,
                                    ),
                                    normalise=True,
                                )

                        # Density
                        self._density_tracker.record_attack(timestamp)
                        norm_density = self._density_tracker.normalised_density()

                        event = MusicalEvent(
                            time_seconds=timestamp,
                            strength=strength,
                            frequency_region=region,
                            energy=local_energy,
                            density=norm_density,
                        )

                        self._last_onset_time = timestamp

                        if self._callback:
                            self._callback(event)
                        self._events.append(event)

        # Trim buffer — keep last ~0.5 seconds
        max_buffer = int(self._sample_rate * 0.5)
        if len(self._buffer) > max_buffer:
            self._buffer = self._buffer[-max_buffer:]

    def flush(self) -> list[MusicalEvent]:
        """Return all detected events and clear the internal buffer.

        Returns
        -------
        list[MusicalEvent]
            All events detected since the last flush.
        """
        events = list(self._events)
        self._events.clear()
        return events

    def reset(self) -> None:
        """Reset all internal state."""
        self._buffer.clear()
        self._spectral_buffer.clear()
        self._events.clear()
        self._density_tracker.reset()
        self._last_onset_time = None

    @property
    def buffer_ms(self) -> float:
        """Approximate duration of the internal buffer in milliseconds."""
        return (len(self._buffer) / self._sample_rate) * 1000.0