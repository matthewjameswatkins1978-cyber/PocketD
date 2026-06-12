import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio.pulse_generator import generate_pulse_events


def test_generate_pulse_events_returns_expected_count() -> None:
    events = generate_pulse_events(bpm=120.0, duration_seconds=1.0, pulse_width=0.02)

    assert len(events) >= 3
    assert events[0].time_seconds == 0.0
    assert events[1].time_seconds == 0.5
    assert events[2].time_seconds == 1.0
#
