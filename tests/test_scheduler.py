import sys
from pathlib import Path

from groove_library import get_groove
from scheduler import hits_at_step, step_duration_seconds

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_step_duration_seconds_is_reasonable() -> None:
    duration = step_duration_seconds(120.0)

    assert duration == 0.125


def test_hits_at_step_returns_expected_groove_hits() -> None:
    groove = get_groove("simple_rock")

    hits = hits_at_step(groove, step=0, complexity_level=5)

    assert hits
    assert hits[0][0] == "kick"
