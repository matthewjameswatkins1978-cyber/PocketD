import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_live_clap_lock import analyse_claps


def _synthetic_clap_track(
    bpm: float,
    bars: int = 4,
    sample_rate: int = 44100,
) -> np.ndarray:
    duration = bars * 4 * (60.0 / bpm) + 1.0
    samples = np.zeros(int(duration * sample_rate), dtype=np.float32)
    clap_length = int(0.025 * sample_rate)
    envelope = np.linspace(1.0, 0.0, clap_length, dtype=np.float32)

    for beat in range(bars * 4):
        start = int(beat * (60.0 / bpm) * sample_rate)
        samples[start : start + clap_length] += envelope

    return samples


def test_analyse_claps_estimates_bpm_from_audio_impulses() -> None:
    samples = _synthetic_clap_track(bpm=118.0)

    analysis = analyse_claps(samples, sample_rate=44100, onset_threshold=0.5)

    assert len(analysis.onset_times) >= 15
    assert abs(analysis.bpm - 118.0) < 2.0


def test_analyse_claps_handles_empty_audio() -> None:
    analysis = analyse_claps(np.zeros(0, dtype=np.float32), sample_rate=44100)

    assert analysis.bpm == 120.0
    assert analysis.onset_times == []


def test_analyse_claps_ignores_quiet_interface_noise() -> None:
    rng = np.random.default_rng(42)
    samples = rng.normal(0.0, 0.00005, 44100).astype(np.float32)

    analysis = analyse_claps(samples, sample_rate=44100)

    assert analysis.bpm == 120.0
    assert analysis.onset_times == []
