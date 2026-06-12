"""Energy computation for audio frames."""

from __future__ import annotations

import numpy as np


def compute_energy(
    frame: np.ndarray,
    normalise: bool = True,
) -> float:
    """Compute RMS energy of an audio frame.

    Parameters
    ----------
    frame : np.ndarray
        Audio samples (1-D).
    normalise : bool
        If True, normalise output to [0, 1] by dividing by max possible RMS
        for the data type. For float [-1, 1] input this effectively returns
        the RMS as-is (max RMS ≈ 0.707). For int16 input, it scales by 32767.

    Returns
    -------
    float
        RMS energy value.
    """
    if frame.size == 0:
        return 0.0

    frame_float = frame.astype(np.float64)

    # Handle integer types by scaling to [-1, 1]
    if np.issubdtype(frame.dtype, np.integer):
        iinfo = np.iinfo(frame.dtype)
        frame_float = frame_float / max(abs(iinfo.min), abs(iinfo.max))

    rms = float(np.sqrt(np.mean(frame_float**2)))

    if normalise:
        # For float in [-1, 1], max RMS is 1.0 (square wave)
        return min(1.0, rms)
    return rms


def compute_energy_in_window(
    signal: np.ndarray,
    center_sample: int,
    window_samples: int,
    sample_rate: int,
) -> float:
    """Compute RMS energy in a window centred around a sample index.

    Parameters
    ----------
    signal : np.ndarray
        Full audio signal.
    center_sample : int
        Sample index to centre the window on.
    window_samples : int
        Total width of the window in samples.
    sample_rate : int
        Sample rate (used only for bounds checking, optional).

    Returns
    -------
    float
        Normalised RMS energy in [0, 1].
    """
    half = window_samples // 2
    start = max(0, center_sample - half)
    end = min(len(signal), center_sample + half)

    if end <= start:
        return 0.0

    window = signal[start:end]
    return compute_energy(window, normalise=True)