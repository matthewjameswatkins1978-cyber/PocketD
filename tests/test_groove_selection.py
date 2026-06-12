import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from groove_matcher import select_groove


def test_select_groove_returns_a_stable_choice() -> None:
    fingerprint = {
        "density": 0.6,
        "syncopation": 0.2,
        "strong_beats": [1, 3],
    }

    groove = select_groove(fingerprint, confidence=0.8, personality="Anchor")

    assert groove is not None
    assert groove.id in {
        "simple_rock",
        "motorik",
        "half_time",
        "shuffle",
        "funk_pocket",
    }
