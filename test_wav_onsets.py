"""Diagnostic WAV-file onset test.

Loads a WAV file from disk, runs the existing onset detector on the decoded
samples, prints each detected onset, and sends MIDI kick hits to the
configured MIDI output for each strong onset.

This is a diagnostic helper, not the final app.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from midi_out import MidiOut
from onset_detector import detect_onsets

DEFAULT_MIDI_PORT = "PocketDrummer Out"
DEFAULT_THRESHOLD = 0.35
MIDI_NOTE_KICK = 36


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAV onset detector → MIDI kick test")
    parser.add_argument("wav_file", type=str, help="Path to the input WAV file")
    parser.add_argument("--midi-port", type=str, default=DEFAULT_MIDI_PORT, help="MIDI output port")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD, help="Onset strength threshold")
    parser.add_argument("--min-interval", type=float, default=0.03, help="Minimum gap between onset detections")
    return parser.parse_args()


def load_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frames = wav_file.readframes(wav_file.getnframes())

    if sample_width == 1:
        dtype = np.uint8
        offset = 128
    elif sample_width == 2:
        dtype = np.int16
        offset = 0
    elif sample_width == 4:
        dtype = np.int32
        offset = 0
    else:
        raise ValueError(f"Unsupported WAV sample width: {sample_width}")

    data = np.frombuffer(frames, dtype=dtype)
    if sample_width == 1:
        data = data.astype(np.float32) - offset
    else:
        data = data.astype(np.float32)

    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)

    return data / (2 ** (8 * sample_width - 1)), sample_rate


def main() -> None:
    args = parse_args()
    wav_path = Path(args.wav_file).expanduser().resolve()

    if not wav_path.exists():
        raise FileNotFoundError(f"WAV file not found: {wav_path}")

    samples, sample_rate = load_wav(wav_path)
    events = detect_onsets(samples, sample_rate=sample_rate, min_interval=args.min_interval)

    print(f"Loaded WAV file: {wav_path}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Samples: {len(samples)}")
    print(f"Detected onsets: {len(events)}")
    for event in events:
        print(f"  time={event.time_seconds:.3f}s strength={event.strength:.3f}")

    strong_events = [event for event in events if event.strength >= args.threshold]
    print(f"Strong onsets above threshold {args.threshold:.2f}: {len(strong_events)}")

    midi = MidiOut(args.midi_port)
    midi.open()
    try:
        for event in strong_events:
            print(f"Sending kick at {event.time_seconds:.3f}s")
            midi.send_note(MIDI_NOTE_KICK, velocity=110)
    finally:
        midi.close()


if __name__ == "__main__":
    main()
