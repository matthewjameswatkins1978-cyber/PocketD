import sys
from pathlib import Path

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from pulse_tracker import estimate_tempo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_tempo_tracker_estimates_120_bpm_from_pulse_events() -> None:
    pulse_events = generate_pulse_events(bpm=120.0, duration_seconds=1.0)
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)

    bpm = estimate_tempo([event.time_seconds for event in onsets])

    assert bpm >= 110.0
    assert bpm <= 130.0
