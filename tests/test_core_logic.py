import math

from confidence_engine import calculate_confidence
from groove_matcher import select_groove
from memory import SessionMemory
from onset_detector import detect_onsets
from pulse_tracker import estimate_tempo


def test_detect_onsets_finds_expected_peaks():
    sample_rate = 16000
    duration = 0.4
    t = [i / sample_rate for i in range(int(sample_rate * duration))]
    signal = []
    for value in t:
        pulse = 0.8 if (value * 4) % 0.25 < 0.02 else 0.1
        signal.append(pulse)

    events = detect_onsets(signal, sample_rate=sample_rate, min_interval=0.05)

    assert events
    assert all(event.strength >= 0.0 for event in events)
    assert events[0].time_seconds >= 0.0


def test_estimate_tempo_returns_reasonable_bpm():
    onset_times = [0.0, 0.25, 0.50, 0.75, 1.00]

    bpm = estimate_tempo(onset_times)

    assert 110.0 <= bpm <= 130.0


def test_calculate_confidence_clamps_range():
    confidence = calculate_confidence(tempo_stability=0.95, accent_consistency=0.9, fingerprint_repeat=0.8, downbeat_confidence=0.7)

    assert 0.0 <= confidence <= 1.0
    assert confidence > 0.75


def test_select_groove_prefers_stable_choice():
    fingerprint = {"density": 0.6, "strong_beats": [1, 3], "syncopation": 0.2}

    groove = select_groove(fingerprint, confidence=0.8, personality="Anchor")

    assert groove is not None
    assert groove.id in {"simple_rock", "motorik", "half_time", "shuffle", "funk_pocket", "punk_drive"}


def test_session_memory_recalls_similar_profile():
    memory = SessionMemory()
    memory.store("section_a", fingerprint={"density": 0.6, "strong_beats": [1, 3]}, groove="simple_rock", feel="Anchor", energy=0.6)

    recall = memory.recall({"density": 0.58, "strong_beats": [1, 3]})

    assert recall is not None
    assert recall["groove"] == "simple_rock"
