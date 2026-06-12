"""Synthetic end-to-end demo:
   pulse → onset → tempo → confidence → groove → scheduler events
   → humanized scheduler events."""

from __future__ import annotations

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from confidence_engine import estimate_timing_confidence
from drummer.humanize import humanize_events
from groove_matcher import select_groove
from models import KICK, SNARE, CLOSED_HAT
from pulse_tracker import estimate_tempo
from scheduler import hits_at_step, step_duration_seconds


def scheduled_events_table(
    groove, bpm: float, bars: int = 2, complexity: int = 3
) -> list[dict]:
    """Build a list of scheduled drum events (synthetic, no real-time clock)."""
    step_dur = step_duration_seconds(bpm)
    total_steps = groove.steps * bars
    events: list[dict] = []

    for step_idx in range(total_steps):
        step = step_idx % groove.steps
        timestamp = step_idx * step_dur
        hits = hits_at_step(groove, step, complexity_level=complexity)
        for instrument, note in hits:
            velocity = 80 if instrument == "hat" else 100
            events.append(
                {
                    "timestamp": round(timestamp, 3),
                    "step": step,
                    "instrument": instrument,
                    "note": note,
                    "velocity": velocity,
                }
            )

    return events


def main() -> None:
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

    # ---- scheduler stage ----
    events = scheduled_events_table(groove, bpm, bars=2, complexity=3)

    # ---- humanization stage ----
    humanized = humanize_events(events, seed=42)

    print("Synthetic pipeline demo")
    print(f"Pulse events: {len(pulse_events)}")
    print(f"Detected onsets: {len(onsets)}")
    print(f"Estimated tempo: {bpm:.1f} BPM")
    print(f"Timing confidence: {confidence:.2f}")
    print(f"Selected groove: {groove.id} ({groove.name})")
    print(f"Groove density: {groove.density}, energy: {groove.energy}")
    print(f"\nScheduled drum events (2 bars, complexity={3}):")
    print(f"  {'Timestamp':>9s}  {'Step':>4s}  {'Instrument':>10s}  {'Note':>4s}  {'Vel':>3s}")
    for ev in events:
        print(
            f"  {ev['timestamp']:9.3f}s  {ev['step']:4d}"
            f"  {ev['instrument']:>10s}  {ev['note']:4d}  {ev['velocity']:3d}"
        )

    print(f"\nHumanized drum events (seed=42):")
    print(f"  {'Timestamp':>9s}  {'Step':>4s}  {'Instrument':>10s}  {'Note':>4s}  {'Vel':>3s}")
    for ev in humanized:
        print(
            f"  {ev['timestamp']:9.3f}s  {ev['step']:4d}"
            f"  {ev['instrument']:>10s}  {ev['note']:4d}  {ev['velocity']:3d}"
        )


if __name__ == "__main__":
    main()