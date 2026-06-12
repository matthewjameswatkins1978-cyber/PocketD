"""Generate a simple 30-second WAV click track for onset diagnostics."""

from __future__ import annotations

import argparse
import math
import wave
from pathlib import Path

import numpy as np

DEFAULT_BPM = 120
DEFAULT_DURATION_SECONDS = 30.0
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_OUTPUT = "click_120bpm.wav"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 30-second click track for WAV onset testing"
    )
    parser.add_argument("--bpm", type=float, default=DEFAULT_BPM, help="Tempo in beats per minute")
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help="Length of the generated WAV file in seconds",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help="Output WAV sample rate in Hz",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help="Output WAV filename",
    )
    return parser.parse_args()


def generate_click_track(
    bpm: float,
    duration_seconds: float,
    sample_rate: int,
    output_path: Path,
) -> None:
    if bpm <= 0:
        raise ValueError("BPM must be positive")
    if duration_seconds <= 0:
        raise ValueError("Duration must be positive")

    quarter_note_seconds = 60.0 / bpm
    total_samples = int(duration_seconds * sample_rate)
    waveform = np.zeros(total_samples, dtype=np.float32)

    rng = np.random.default_rng(7)
    click_length_samples = max(1, int(0.02 * sample_rate))
    click_frequency = 1800.0

    for beat_index in range(int(math.ceil(duration_seconds / quarter_note_seconds))):
        onset_sample = int(beat_index * quarter_note_seconds * sample_rate)
        if onset_sample >= total_samples:
            break

        end_sample = min(total_samples, onset_sample + click_length_samples)
        local_samples = end_sample - onset_sample
        if local_samples <= 0:
            continue

        t = np.arange(local_samples, dtype=np.float32) / sample_rate
        envelope = np.exp(-t / 0.004)
        tone = np.sin(2.0 * math.pi * click_frequency * t)
        noise = rng.standard_normal(local_samples) * 0.05
        click = (0.65 * tone + 0.35 * noise) * envelope
        waveform[onset_sample:end_sample] += click.astype(np.float32)

    peak = float(np.max(np.abs(waveform)))
    if peak > 0:
        waveform *= 0.9 / peak

    waveform = np.clip(waveform, -1.0, 1.0)
    pcm = np.asarray(waveform * 32767.0, dtype=np.int16)

    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm.tobytes())


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser().resolve()

    generate_click_track(
        bpm=args.bpm,
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        output_path=output_path,
    )

    print(f"Generated click track: {output_path}")
    print(f"Tempo: {args.bpm:.0f} BPM")
    print(f"Length: {args.duration:.1f} seconds")


if __name__ == "__main__":
    main()
