"""Diagnostic live-audio onset test.

Listens to a chosen input device, detects onset peaks with the existing
onset_detector.py heuristic, and sends a kick drum MIDI note to the
configured loopMIDI output.

This is intentionally a diagnostic probe only: no groove logic, no tempo
tracking, and no GUI.
"""

from __future__ import annotations

import argparse
import time
from collections import deque

import numpy as np
import sounddevice as sd

from midi_out import MidiOut
from onset_detector import detect_onsets

# Adjustable sensitivity for the diagnostic test.
SENSITIVITY_THRESHOLD = 0.75
SAMPLE_RATE = 16000
BLOCKSIZE = 1024
HISTORY_SECONDS = 0.75
MIN_HIT_INTERVAL_SECONDS = 0.18
MIDI_NOTE_KICK = 36
DEFAULT_MIDI_PORT = "PocketDrummer Out"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live onset test → MIDI kick")
    parser.add_argument("--device", type=int, default=None, help="Input device index to use")
    parser.add_argument("--device-name", type=str, default=None, help="Input device name substring")
    parser.add_argument("--threshold", type=float, default=SENSITIVITY_THRESHOLD, help="Onset strength threshold")
    parser.add_argument("--midi-port", type=str, default=DEFAULT_MIDI_PORT, help="MIDI output port name")
    parser.add_argument("--sample-rate", type=int, default=SAMPLE_RATE, help="Input sample rate")
    parser.add_argument("--blocksize", type=int, default=BLOCKSIZE, help="Input block size")
    return parser.parse_args()


def find_input_device(device_index: int | None, device_name: str | None) -> object:
    devices = sd.query_devices()
    if not isinstance(devices, (list, tuple)):
        devices = [devices]

    if device_index is not None:
        try:
            return devices[device_index]
        except (IndexError, TypeError) as exc:
            raise ValueError(f"Input device index {device_index} is out of range") from exc

    if device_name:
        lower_name = device_name.lower()
        for device in devices:
            name = device.get("name", "").lower()
            if lower_name in name:
                return device
        raise ValueError(f"No input device matched '{device_name}'")

    for device in devices:
        if int(device.get("max_input_channels", 0)) > 0:
            return device
    raise ValueError("No input device with available inputs was found")


class LiveOnsetDrumProbe:
    def __init__(self, midi_port: str, sensitivity_threshold: float, sample_rate: int, blocksize: int) -> None:
        self.sensitivity_threshold = sensitivity_threshold
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.midi = MidiOut(midi_port)
        self.history: deque[float] = deque(maxlen=int(sample_rate * HISTORY_SECONDS))
        self.last_hit_time = 0.0

    def _mono(self, frame_buffer: np.ndarray) -> np.ndarray:
        data = np.asarray(frame_buffer, dtype=np.float32)
        if data.ndim == 2:
            return data.mean(axis=1)
        return data.reshape(-1)

    def callback(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        if status:
            print(f"Audio status: {status}")

        samples = self._mono(indata)
        scale = 1.0 / max(np.max(np.abs(samples)), 1.0)
        normalized = samples * scale
        self.history.extend(normalized.tolist())

        events = detect_onsets(list(self.history), sample_rate=self.sample_rate, min_interval=0.03)
        strong_events = [event for event in events if event.strength >= self.sensitivity_threshold]

        if strong_events:
            event = strong_events[0]
            now = time.perf_counter()
            if now - self.last_hit_time >= MIN_HIT_INTERVAL_SECONDS:
                self.last_hit_time = now
                print(
                    "Onset detected: "
                    f"time={event.time_seconds:.3f}s strength={event.strength:.3f} -> MIDI kick"
                )
                try:
                    self.midi.send_note(MIDI_NOTE_KICK, velocity=110)
                except Exception as exc:  # pragma: no cover - diagnostic path
                    print(f"MIDI send failed: {exc}")


def main() -> None:
    args = parse_args()

    try:
        device = find_input_device(args.device, args.device_name)
    except ValueError as exc:
        print(f"Input device setup failed: {exc}")
        print("Run 'python list_audio_devices.py' to see available inputs.")
        return

    device_index = int(device.get("index", 0)) if isinstance(device, dict) else None
    device_name = device.get("name", "") if isinstance(device, dict) else str(device)
    print("Using input device:")
    print(f"  index={device_index}")
    print(f"  name={device_name}")
    print(f"  threshold={args.threshold:.2f}")
    print("Press Ctrl+C to stop.\n")

    probe = LiveOnsetDrumProbe(
        midi_port=args.midi_port,
        sensitivity_threshold=args.threshold,
        sample_rate=args.sample_rate,
        blocksize=args.blocksize,
    )

    try:
        probe.midi.open()
        with sd.InputStream(
            device=device_index,
            channels=1,
            samplerate=args.sample_rate,
            blocksize=args.blocksize,
            callback=probe.callback,
            dtype="float32",
        ):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        probe.midi.close()


if __name__ == "__main__":
    main()
