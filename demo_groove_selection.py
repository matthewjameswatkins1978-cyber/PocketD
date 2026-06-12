"""Simple synthetic demo for the groove selection stage."""

from __future__ import annotations

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from confidence_engine import estimate_timing_confidence
from groove_matcher import select_groove
from pulse_tracker import estimate_tempo


def main() -> None:
    pulse_events = generate_pulse_events(bpm=120.0, duration_seconds=2.0)
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)

    bpm = estimate_tempo([event.time_seconds for event in onsets])
    confidence = estimate_timing_confidence(
        [event.time_seconds for event in onsets], expected_bpm=120.0
    )

    fingerprint = {
        "density": 0.6,
        "syncopation": 0.2,
        "strong_beats": [1, 3],
    }
    groove = select_groove(fingerprint, confidence=confidence, personality="Anchor")

    print("Synthetic pulse → groove selection demo")
    print(f"Estimated tempo: {bpm:.1f} BPM")
    print(f"Timing confidence: {confidence:.2f}")
    print(f"Selected groove: {groove.id} ({groove.name})")
    print(f"Groove density: {groove.density} energy: {groove.energy}")


if __name__ == "__main__":
    main()
