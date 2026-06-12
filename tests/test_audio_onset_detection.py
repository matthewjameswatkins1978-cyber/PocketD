import sys
from pathlib import Path

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_detect_onsets_from_events_finds_pulse_hits() -> None:
    events = generate_pulse_events(bpm=120.0, duration_seconds=1.0)

    detected = detect_onsets_from_events(events, sample_rate=16000, min_interval=0.05)

    assert detected
    assert any(abs(event.time_seconds - 0.0) < 0.02 for event in detected)
    assert any(abs(event.time_seconds - 0.5) < 0.03 for event in detected)
