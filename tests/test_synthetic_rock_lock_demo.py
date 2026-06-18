import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_synthetic_rock_lock import (
    estimate_bpm_from_synthetic_pulse,
    groove_to_events,
)
from drummer.pipeline_midi import build_schedule
from groove_library import get_groove


def test_estimates_bpm_from_synthetic_pulse() -> None:
    estimated_bpm, onset_count = estimate_bpm_from_synthetic_pulse(
        input_bpm=118.0,
        duration_seconds=8.0,
    )

    assert onset_count >= 15
    assert estimated_bpm == 118.0


def test_simple_rock_groove_schedules_at_estimated_bpm() -> None:
    estimated_bpm, _ = estimate_bpm_from_synthetic_pulse(
        input_bpm=120.0,
        duration_seconds=4.0,
    )
    events = groove_to_events(get_groove("simple_rock"))
    schedule = build_schedule(events, bpm=estimated_bpm, repeats=1)

    note_ons = [entry for entry in schedule if entry[1] == "on"]
    note_on_times = [entry[0] for entry in note_ons]

    assert estimated_bpm == 120.0
    assert len(events) == 12
    assert len(note_ons) == 12
    assert 0.0 in note_on_times
    assert 0.5 in note_on_times
    assert 1.0 in note_on_times
    assert 1.5 in note_on_times
