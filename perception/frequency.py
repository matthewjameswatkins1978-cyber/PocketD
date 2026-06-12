"""Frequency region classification from spectral analysis."""

from __future__ import annotations

import numpy as np

from perception.models import FREQUENCY_REGIONS, FFT_SIZE, FrequencyRegion


def compute_spectrum(
    frame: np.ndarray,
    sample_rate: int,
    fft_size: int = FFT_SIZE,
) -> np.ndarray:
    """Compute the magnitude spectrum of a single audio frame.

    Returns a 1-D array of magnitude values (positive frequencies only).
    """
    window = np.hanning(min(fft_size, len(frame)))
    if len(frame) < fft_size:
        padded = np.zeros(fft_size)
        padded[: len(frame)] = frame[:fft_size]
        padded[: len(window)] = padded[: len(window)] * window
    else:
        padded = frame[:fft_size] * window

    spectrum = np.fft.rfft(padded)
    return np.abs(spectrum)


def band_energy(
    spectrum: np.ndarray,
    sample_rate: int,
    low_hz: float,
    high_hz: float,
    fft_size: int = FFT_SIZE,
) -> float:
    """Sum magnitude energy in a frequency band."""
    nyquist = sample_rate / 2.0
    if high_hz > nyquist:
        high_hz = nyquist
    if low_hz >= high_hz:
        return 0.0

    bin_low = int(low_hz * fft_size / sample_rate)
    bin_high = int(high_hz * fft_size / sample_rate)
    bin_high = min(bin_high, len(spectrum) - 1)

    if bin_low >= len(spectrum) or bin_low >= bin_high:
        return 0.0

    return float(np.sum(spectrum[bin_low:bin_high]))


def classify_frequency(
    frame: np.ndarray | None,
    sample_rate: int,
    fft_size: int = FFT_SIZE,
) -> FrequencyRegion:
    """Determine which frequency region dominates the given audio frame.

    Splits the spectrum into standard musical bands and returns the region
    with the highest energy concentration.

    Parameters
    ----------
    frame : np.ndarray | None
        A short audio segment (will be padded/truncated to fft_size).
        If None, returns "unknown".
    sample_rate : int
        Sample rate of the audio.
    fft_size : int
        FFT size (default 2048).

    Returns
    -------
    FrequencyRegion
        The dominant frequency region label.
    """
    if frame is None or len(frame) == 0:
        return "unknown"

    spectrum = compute_spectrum(frame, sample_rate, fft_size)

    best_region: FrequencyRegion = "unknown"
    best_energy = -1.0

    for name, low_hz, high_hz in FREQUENCY_REGIONS:
        energy = band_energy(spectrum, sample_rate, low_hz, high_hz, fft_size)
        if energy > best_energy:
            best_energy = energy
            best_region = name  # type: ignore[assignment]

    return best_region


def classify_frequency_from_spectrum(
    spectrum: np.ndarray,
    sample_rate: int,
    fft_size: int = FFT_SIZE,
) -> FrequencyRegion:
    """Classify frequency region from a pre-computed spectrum."""
    best_region: FrequencyRegion = "unknown"
    best_energy = -1.0

    for name, low_hz, high_hz in FREQUENCY_REGIONS:
        energy = band_energy(spectrum, sample_rate, low_hz, high_hz, fft_size)
        if energy > best_energy:
            best_energy = energy
            best_region = name  # type: ignore[assignment]

    return best_region


def region_frequency_profile(
    region: FrequencyRegion,
) -> tuple[float, float]:
    """Return the (low_hz, high_hz) range for a given region name."""
    for name, low_hz, high_hz in FREQUENCY_REGIONS:
        if name == region:
            return (low_hz, high_hz)
    return (0.0, 0.0)