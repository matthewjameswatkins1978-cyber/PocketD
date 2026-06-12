"""Demo: Perception Engine — Module 1: Event Listener.

Demonstrates the perception engine's ability to detect musical events
(strength, frequency region, energy, density) from synthetic audio.

Run:
    python demo_perception.py

The demo generates synthetic audio that simulates:
    - A kick drum pattern (low frequency hits)
    - A snare pattern (mid frequency hits)
    - Hi-hats (high frequency)
    - A busy fill section (high density)
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np

from perception.event_listener import EventListener, AudioFrame, detect_events_from_audio
from perception.models import MusicalEvent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("demo")


def _print_event(event: MusicalEvent, index: int) -> None:
    """Pretty-print a MusicalEvent."""
    strength_pct = round(event.strength * 100)
    energy_pct = round(event.energy * 100)
    density_pct = round(event.density * 100)

    print(
        f"\n[{event.time_seconds:.2f}s]"
        f"\n  Attack detected"
        f"\n  Strength: {strength_pct}"
        f"\n  Frequency region: {event.frequency_region.capitalize()}"
        f"\n  Energy: {energy_pct}%"
        f"\n  Density: {density_pct}%"
    )


def generate_synthetic_signal(
    sample_rate: int = 44100,
    duration: float = 8.0,
    bpm: float = 120.0,
) -> np.ndarray:
    """Generate a synthetic drum-like audio signal for testing.

    Creates a mix of kick, snare, and hi-hat-like sounds with varying
    dynamics across the duration.
    """
    total_samples = int(sample_rate * duration)
    t = np.arange(total_samples) / sample_rate
    signal = np.zeros(total_samples, dtype=np.float64)

    beat_duration = 60.0 / bpm

    # Helper to add a drum hit
    def add_hit(time_sec: float, freq: float, decay: float, amp: float) -> None:
        hit_start = int(time_sec * sample_rate)
        if hit_start >= total_samples:
            return
        hit_len = int(decay * sample_rate)
        hit_end = min(hit_start + hit_len, total_samples)
        hit_t = np.arange(hit_end - hit_start) / sample_rate
        if len(hit_t) == 0:
            return
        hit = amp * np.sin(2 * np.pi * freq * hit_t) * np.exp(-hit_t * 20.0 / decay)
        signal[hit_start:hit_end] += hit[: len(hit_t)]

    # Add kick drum hits (low frequency, on 1 and 3)
    kick_pattern = [0, 2, 4, 6]
    for beat in kick_pattern:
        offset = beat * beat_duration
        add_hit(offset, freq=60.0, decay=0.2, amp=0.8)
        # Extra attack transient
        add_hit(offset, freq=150.0, decay=0.05, amp=0.6)

    # Add snare hits (mid frequency, on 2 and 4)
    snare_pattern = [1, 3, 5, 7]
    for beat in snare_pattern:
        offset = beat * beat_duration
        add_hit(offset, freq=200.0, decay=0.15, amp=0.7)
        add_hit(offset, freq=400.0, decay=0.08, amp=0.5)

    # Add hi-hats (high frequency, 8th notes)
    for beat in range(int(8 * 2)):  # 8th notes for 8 beats
        offset = (beat * beat_duration / 2)
        if beat % 2 == 0:
            amp = 0.3  # on beat
        else:
            amp = 0.2  # off beat
        add_hit(offset, freq=8000.0, decay=0.04, amp=amp)

    # Add a fill section with higher density at bars 4-5
    fill_start = 4.0
    fill_duration = 2.0
    fill_hits = int(32)  # 32nd notes
    for i in range(fill_hits):
        offset = fill_start + (i / fill_hits) * fill_duration
        freq = 100.0 + (i % 3) * 300.0
        amp = 0.4 + (i % 4) * 0.1
        add_hit(offset, freq=freq, decay=0.06, amp=min(amp, 0.9))

    # Normalise to [-1, 1]
    max_val = np.max(np.abs(signal))
    if max_val > 0:
        signal = signal / max_val * 0.95

    return signal


def demo_offline() -> None:
    """Demo: offline mode — process a full audio buffer at once."""
    print("=" * 50)
    print("PERCEPTION ENGINE — MODULE 1: EVENT LISTENER")
    print("=" * 50)
    print("\n[OFFLINE MODE] Processing full audio buffer...")

    sample_rate = 44100
    audio = generate_synthetic_signal(sample_rate=sample_rate, duration=8.0, bpm=120)

    events = detect_events_from_audio(audio, sample_rate)

    print(f"\nDetected {len(events)} musical events:")
    for i, event in enumerate(events):
        _print_event(event, i)

    # Summary statistics
    print("\n" + "-" * 50)
    print("SUMMARY")
    print("-" * 50)
    print(f"Total events detected: {len(events)}")
    if events:
        regions: dict[str, int] = {}
        for e in events:
            regions[e.frequency_region] = regions.get(e.frequency_region, 0) + 1
        for region, count in sorted(regions.items(), key=lambda x: -x[1]):
            print(f"  {region.capitalize():10s}: {count:3d} events")

        avg_strength = np.mean([e.strength for e in events])
        avg_energy = np.mean([e.energy for e in events])
        avg_density = np.mean([e.density for e in events])
        print(f"  Avg strength : {avg_strength:.2f}")
        print(f"  Avg energy   : {avg_energy:.2f}")
        print(f"  Avg density  : {avg_density:.2f}")


def event_callback(event: MusicalEvent) -> None:
    """Callback for streaming mode."""
    _print_event(event, 0)


def demo_streaming() -> None:
    """Demo: streaming mode — process audio frame by frame."""
    print("\n" + "=" * 50)
    print("[STREAMING MODE] Processing audio frame by frame...")
    print("=" * 50)

    sample_rate = 44100
    audio = generate_synthetic_signal(sample_rate=sample_rate, duration=4.0, bpm=120)

    listener = EventListener(sample_rate=sample_rate, callback=event_callback)

    frame_size = 1024
    total_frames = len(audio) // frame_size

    for i in range(total_frames):
        start = i * frame_size
        end = start + frame_size
        frame_samples = audio[start:end]
        time_sec = end / sample_rate

        frame = AudioFrame(
            samples=frame_samples,
            sample_rate=sample_rate,
            time_seconds=time_sec,
        )
        listener.process_frame(frame)

    remaining = listener.flush()
    if remaining:
        print(f"\n  [{len(remaining)} unflushed events retrieved]")


def main() -> int:
    demo_offline()
    demo_streaming()

    print("\n" + "=" * 50)
    print("Perception Engine Module 1 ready.")
    print("Success condition: A bass note, guitar hit, mute, accent, or strum")
    print("consistently creates meaningful event data.")
    print("=" * 50)
    return 0


if __name__ == "__main__":
    sys.exit(main())
