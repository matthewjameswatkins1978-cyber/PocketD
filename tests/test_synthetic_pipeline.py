import sys
from pathlib import Path

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from confidence_engine import estimate_timing_confidence
from groove_matcher import select_groove
from pulse_tracker import estimate_tempo
from scheduler import hits_at_step

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_synthetic_pipeline_produces_stable_groove_choice() -> None:
    pulse_events = generate_pulse_events(bpm=120.0, duration_seconds=2.0)
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)
    bpm = estimate_tempo([event.time_seconds for event in onsets])
    confidence = estimate_timing_confidence(
        [event.time_seconds for event in onsets], expected_bpm=120.0
    )

    groove = select_groove(
        {"density": 0.6, "syncopation": 0.2, "strong_beats": [1, 3]},
        confidence=confidence,
        personality="Anchor",
    )

    assert bpm >= 110.0
    assert bpm <= 130.0
    assert 0.0 <= confidence <= 1.0
    assert groove is not None
    assert hits_at_step(groove, 0, complexity_level=2)
