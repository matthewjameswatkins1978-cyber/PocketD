"""Demo: clap into audio interface, estimate BPM, optionally play simple rock.

This is the real-input companion to the synthetic demos. It records a short
take from an audio input device, reads one channel, detects clap onsets,
estimates BPM, and prints a timing report.

Typical run for Yamaha AG03/AG06 channel 1:
    .venv\\Scripts\\python.exe demo_live_clap_lock.py --device-name "AG06/AG03" --channel 1 --duration 12

To play the detected rock beat over MIDI:
    .venv\\Scripts\\python.exe demo_live_clap_lock.py --device-name "AG06/AG03" --channel 1 --port "PocketDrummer Out"
"""

from __future__ import annotations

import argparse
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd
except ImportError:  # pragma: no cover - environment guard
    sd = None

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_synthetic_rock_lock import groove_to_events
from drummer.pipeline_midi import play_events_with_diagnostics
from groove_library import get_groove
from midi_out import MidiOut
from onset_detector import detect_onsets
from pulse_tracker import estimate_tempo


DEFAULT_DEVICE_NAME = "AG06/AG03"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_DURATION_SECONDS = 12.0
DEFAULT_MIN_INTERVAL_SECONDS = 0.18
DEFAULT_COUNTDOWN_SECONDS = 3


@dataclass(frozen=True)
class ClapAnalysis:
    bpm: float
    onset_times: list[float]
    onset_strengths: list[float]
    intervals: list[float]


def list_input_devices() -> list[tuple[int, str, int]]:
    """Return available input devices as (index, name, input_channels)."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; use the project .venv")

    devices = sd.query_devices()
    result: list[tuple[int, str, int]] = []
    for index, device in enumerate(devices):
        channels = int(device.get("max_input_channels", 0))
        if channels > 0:
            result.append((index, str(device.get("name", "")), channels))
    return result


def find_input_device(
    device_index: int | None = None,
    device_name: str | None = DEFAULT_DEVICE_NAME,
) -> int:
    """Resolve an input device index by explicit index or name substring."""
    devices = list_input_devices()
    if device_index is not None:
        indexes = {index for index, _, _ in devices}
        if device_index not in indexes:
            raise ValueError(f"Input device index {device_index} is not available")
        return device_index

    if device_name:
        needle = device_name.lower()
        for index, name, _ in devices:
            if needle in name.lower():
                return index
        available = ", ".join(f"[{index}] {name}" for index, name, _ in devices)
        raise ValueError(f"No input device matched '{device_name}'. Available: {available}")

    if not devices:
        raise ValueError("No input devices found")
    return devices[0][0]


def record_audio(
    device_index: int,
    duration_seconds: float,
    sample_rate: int,
    channel_number: int,
) -> np.ndarray:
    """Record audio and return the selected 1-based channel as mono samples."""
    if sd is None:
        raise RuntimeError("sounddevice is not installed; use the project .venv")

    info = sd.query_devices(device_index, "input")
    input_channels = int(info.get("max_input_channels", 1))
    if channel_number < 1 or channel_number > input_channels:
        raise ValueError(
            f"Channel {channel_number} is out of range for device {device_index} "
            f"({input_channels} input channel(s))"
        )

    frames = int(duration_seconds * sample_rate)
    data = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=input_channels,
        device=device_index,
        dtype="float32",
    )
    sd.wait()

    if data.ndim == 1:
        return np.asarray(data, dtype=np.float32)
    return np.asarray(data[:, channel_number - 1], dtype=np.float32)


def analyse_claps(
    samples: np.ndarray,
    sample_rate: int,
    min_interval_seconds: float = DEFAULT_MIN_INTERVAL_SECONDS,
    onset_threshold: float = 0.55,
    min_signal_peak: float = 0.01,
) -> ClapAnalysis:
    """Detect clap onsets and estimate BPM from the selected audio channel."""
    if samples.size == 0:
        return ClapAnalysis(bpm=120.0, onset_times=[], onset_strengths=[], intervals=[])

    peak = float(np.max(np.abs(samples)))
    if peak < min_signal_peak:
        return ClapAnalysis(bpm=120.0, onset_times=[], onset_strengths=[], intervals=[])

    normalised = samples / peak if peak > 0 else samples
    events = detect_onsets(
        normalised,
        sample_rate=sample_rate,
        min_interval=min_interval_seconds,
    )
    strong_events = [event for event in events if event.strength >= onset_threshold]
    onset_times = [event.time_seconds for event in strong_events]
    intervals = [
        b - a
        for a, b in zip(onset_times, onset_times[1:])
        if b > a
    ]
    bpm = estimate_tempo(onset_times)

    return ClapAnalysis(
        bpm=bpm,
        onset_times=onset_times,
        onset_strengths=[event.strength for event in strong_events],
        intervals=intervals,
    )


def save_wav(path: Path, samples: np.ndarray, sample_rate: int) -> None:
    """Save a mono float recording as 16-bit PCM WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())


def print_analysis(analysis: ClapAnalysis, duration_seconds: float) -> None:
    """Print a compact clap timing report."""
    print("\nLive clap lock result")
    print("=" * 56)
    print(f"Recording length:   {duration_seconds:.1f}s")
    print(f"Detected claps:     {len(analysis.onset_times)}")
    print(f"Estimated tempo:    {analysis.bpm:.1f} BPM")

    if len(analysis.onset_times) < 2:
        print("\nNot enough claps to estimate a reliable tempo. Try louder quarter-note claps.")
        return

    print("\nDetected clap timings:")
    for index, timestamp in enumerate(analysis.onset_times, start=1):
        interval = ""
        if index > 1:
            interval = f"  interval={analysis.intervals[index - 2]:.3f}s"
        print(f"  {index:02d}. t={timestamp:6.3f}s{interval}")


def countdown(seconds: int) -> None:
    """Print a simple countdown before recording."""
    for remaining in range(seconds, 0, -1):
        print(f"Recording starts in {remaining}...")
        time.sleep(1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record claps and estimate BPM")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--device-name", type=str, default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--channel", type=int, default=1, help="1-based input channel")
    parser.add_argument("--duration", type=float, default=DEFAULT_DURATION_SECONDS)
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--min-interval", type=float, default=DEFAULT_MIN_INTERVAL_SECONDS)
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--min-signal-peak", type=float, default=0.01)
    parser.add_argument("--countdown", type=int, default=DEFAULT_COUNTDOWN_SECONDS)
    parser.add_argument("--save-wav", type=Path, default=None)
    parser.add_argument("--port", type=str, default=None, help="Optional MIDI output port")
    parser.add_argument("--repeats", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.list_devices:
        print("Available input devices:")
        for index, name, channels in list_input_devices():
            print(f"[{index}] {name} | inputs={channels}")
        return 0

    device_index = find_input_device(args.device, args.device_name)
    device_info = sd.query_devices(device_index, "input")
    print("Using input:")
    print(f"  device=[{device_index}] {device_info.get('name')}")
    print(f"  channel={args.channel}")
    print(f"  duration={args.duration:.1f}s")
    print("\nClap steady quarter notes for the whole take.")

    if args.countdown > 0:
        countdown(args.countdown)

    print("Recording now...")
    samples = record_audio(
        device_index=device_index,
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        channel_number=args.channel,
    )
    print("Recording complete.")

    if args.save_wav is not None:
        save_wav(args.save_wav, samples, args.sample_rate)
        print(f"Saved recording: {args.save_wav}")

    analysis = analyse_claps(
        samples=samples,
        sample_rate=args.sample_rate,
        min_interval_seconds=args.min_interval,
        onset_threshold=args.threshold,
        min_signal_peak=args.min_signal_peak,
    )
    print_analysis(analysis, args.duration)

    if args.port is not None and len(analysis.onset_times) >= 2:
        events = groove_to_events(get_groove("simple_rock"))
        print(f"\nPlaying simple_rock at {analysis.bpm:.1f} BPM via {args.port}...")
        with MidiOut(args.port) as midi:
            play_events_with_diagnostics(
                midi,
                events,
                bpm=analysis.bpm,
                repeats=args.repeats,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
