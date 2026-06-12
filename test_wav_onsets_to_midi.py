"""WAV onset diagnostic that sends MIDI kick notes to the configured output port."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

from midi_out import MidiOut
from onset_detector import detect_onsets

DEFAULT_MIDI_PORT = "PocketDrummer Out"
DEFAULT_THRESHOLD = 0.18
DEFAULT_MIN_INTERVAL = 0.03
MIDI_NOTE_KICK = 36


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WAV file onset detection → MIDI kick test")
    parser.add_argument("wav_file", type=str, help="Path to the WAV file to analyze")
    parser.add_argument(
        "--midi-port",
        type=str,
        default=DEFAULT_MIDI_PORT,
        help="MIDI output port name to use for kick hits",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Minimum onset strength to send a MIDI kick note",
    )
    parser.add_argument(
        "--min-interval",
        type=float,
        default=DEFAULT_MIN_INTERVAL,
        help="Minimum time between accepted onset detections in seconds",
    )
    parser.add_argument(
        "--velocity",
        type=int,
        default=110,
        help="MIDI note velocity for kick hits",
    )
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

    print("=== WAV onset → MIDI diagnostic ===")
    print(f"Input file: {wav_path}")
    print(f"MIDI output port: {args.midi_port}")
    print(f"Sensitivity threshold: {args.threshold:.3f}")
    print(f"Minimum onset interval: {args.min_interval:.3f}s")

    samples, sample_rate = load_wav(wav_path)
    events = detect_onsets(samples, sample_rate=sample_rate, min_interval=args.min_interval)

    print(f"Loaded {len(samples)} samples at {sample_rate} Hz")
    print(f"Detected onsets: {len(events)}")

    for event in events:
        confidence_text = "n/a"
        print(
            f"Onset t={event.time_seconds:.4f}s "
            f"strength={event.strength:.4f} "
            f"confidence={confidence_text}"
        )

    strong_events = [event for event in events if event.strength >= args.threshold]

    print(f"Strong onsets above threshold: {len(strong_events)}")

    midi = MidiOut(args.midi_port)
    midi.open()
    try:
        for index, event in enumerate(strong_events, start=1):
            print(
                f"[{index}/{len(strong_events)}] Sending MIDI kick note 36 "
                f"at t={event.time_seconds:.4f}s (velocity={args.velocity})"
            )
            midi.send_note(MIDI_NOTE_KICK, velocity=args.velocity)
    finally:
        midi.close()

    if not strong_events:
        print("No onset exceeded the sensitivity threshold. Try lowering --threshold.")


if __name__ == "__main__":
    main()
