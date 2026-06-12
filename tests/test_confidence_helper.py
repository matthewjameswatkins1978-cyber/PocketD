import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from confidence_engine import estimate_timing_confidence


def test_estimate_timing_confidence_returns_high_value_for_regular_pulse() -> None:
    onset_times = [0.0, 0.5, 1.0, 1.5, 2.0]

    confidence = estimate_timing_confidence(onset_times, expected_bpm=120.0)

    assert 0.6 <= confidence <= 1.0
