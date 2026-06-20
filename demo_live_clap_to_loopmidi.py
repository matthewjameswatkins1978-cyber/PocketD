"""Opt-in Bunny V1 live clap-to-loopMIDI integration runner.

Nothing opens unless ``--run-live`` is supplied.  The PortAudio callback only
queues copied audio; all perception, control, planning and MIDI work happens on
the main thread.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Sequence

try:
    import sounddevice as sd  # type: ignore[import-untyped]
except (ImportError, OSError):  # pragma: no cover - host dependent
    sd = None

from drummer.live_audio import LiveAudioIngress
from drummer.live_controller import LiveController
from drummer.live_models import LiveConfig
from drummer.live_runtime import LiveRuntime, RuntimeSnapshot, ScheduledMidiSink
from drummer.live_scheduler import LiveScheduler
from midi_out import MidiOut, find_output_port, list_output_ports
from perception.bar import BarTracker
from perception.event_listener import AudioFrame, EventListener
from perception.live_adapter import LiveBarAdapter, LivePulseAdapter
from perception.models import MusicalEvent
from perception.pulse import PulseTracker


DEFAULT_DEVICE_NAME = "AG06/AG03"
DEFAULT_MIDI_PORT = "PocketDrummer Out"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_BLOCK_SIZE = 256
TRACE_DIR = Path(__file__).resolve().parent / "artifacts" / "bunny_live"
COUNT_IN_NOTE = 42


class _NullMidiSink:
    def __init__(self) -> None:
        self.is_open = True

    def send_scheduled(self, note: int, velocity: int, channel: int, deadline: float) -> None:
        del note, velocity, channel, deadline

    def send_note(self, note: int, velocity: int, channel: int = 9) -> None:
        del note, velocity, channel

    def close(self) -> None:
        self.is_open = False


def play_count_in(
    sink: object,
    count: int,
    interval_seconds: float,
    sleep: Callable[[float], None] = time.sleep,
) -> None:
    """Play an audible closed-hat count-in before live evidence is consumed."""
    for beat in range(count):
        velocity = 110 if beat == count - 1 else 85
        sink.send_note(COUNT_IN_NOTE, velocity, 9)  # type: ignore[attr-defined]
        sleep(interval_seconds)


def list_input_devices() -> list[tuple[int, str, int]]:
    if sd is None:
        raise RuntimeError("sounddevice is unavailable")
    return [
        (index, str(device.get("name", "")), int(device.get("max_input_channels", 0)))
        for index, device in enumerate(sd.query_devices())
        if int(device.get("max_input_channels", 0)) > 0
    ]


def resolve_input_device(index: int | None, name: str | None) -> int:
    devices = list_input_devices()
    if index is not None:
        if index not in {item[0] for item in devices}:
            raise ValueError(f"input device {index} is unavailable")
        return index
    needle = (name or "").lower()
    for device_index, device_name, _ in devices:
        if needle in device_name.lower():
            return device_index
    available = ", ".join(f"[{i}] {n}" for i, n, _ in devices) or "(none)"
    raise ValueError(f"no input device matched {name!r}; available: {available}")


def build_trace_record(
    now: float,
    snapshot: RuntimeSnapshot,
    *,
    event_count: int,
    audio_callbacks: int,
    audio_queue_depth: int,
    audio_dropped_blocks: int,
) -> dict[str, object]:
    control = snapshot.controller
    return {
        "at": round(now, 6),
        "state": control.state,
        "generation": control.generation,
        "locked_bpm": control.locked_bpm,
        "bar_epoch": control.bar_epoch,
        "bar_index": control.current_bar_index,
        "slot": control.current_slot,
        "mirror_slot": control.mirror_slot,
        "queue_depth": snapshot.queue_depth,
        "emitted": snapshot.total_emitted,
        "late": snapshot.total_late,
        "dropped": snapshot.total_dropped,
        "detected_events": event_count,
        "audio_callbacks": audio_callbacks,
        "audio_queue_depth": audio_queue_depth,
        "audio_dropped_blocks": audio_dropped_blocks,
    }


def save_trace(payload: dict[str, object], path: Path | None = None) -> Path:
    if path is None:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = TRACE_DIR / f"bunny_live_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-live", action="store_true", help="explicitly open audio and MIDI")
    parser.add_argument("--list-audio", action="store_true")
    parser.add_argument("--list-midi", action="store_true")
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--device-name", default=DEFAULT_DEVICE_NAME)
    parser.add_argument("--channel", type=int, default=1, help="1-based input channel")
    parser.add_argument("--port", default=DEFAULT_MIDI_PORT)
    parser.add_argument("--no-midi", action="store_true", help="run perception/control without output")
    parser.add_argument("--duration", type=float, default=30.0, help="seconds; 0 runs until Ctrl+C")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--block-size", type=int, default=DEFAULT_BLOCK_SIZE)
    parser.add_argument("--tick-ms", type=float, default=1.0)
    parser.add_argument("--min-interval", type=float, default=0.08)
    parser.add_argument("--count-in", type=int, default=4, help="audible MIDI clicks before capture")
    parser.add_argument("--count-in-bpm", type=float, default=120.0)
    parser.add_argument("--trace", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    if args.list_audio:
        try:
            for index, name, channels in list_input_devices():
                print(f"[{index}] {name} | inputs={channels}")
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1
    if args.list_midi:
        try:
            for name in list_output_ports():
                print(name)
        except Exception as exc:
            print(f"MIDI discovery failed: {exc}", file=sys.stderr)
            return 1
    if args.list_audio or args.list_midi:
        return 0

    if not args.run_live:
        print("Hardware remains closed. Re-run with --run-live after checking --list-audio and --list-midi.")
        return 0
    if sd is None:
        print("sounddevice is unavailable; install the project requirements", file=sys.stderr)
        return 1
    if args.channel < 1 or args.sample_rate <= 0 or args.block_size <= 0:
        print("channel, sample rate, and block size must be positive", file=sys.stderr)
        return 2
    if args.duration < 0 or args.tick_ms <= 0 or args.count_in < 0 or args.count_in_bpm <= 0:
        print("duration/count-in cannot be negative and timing values must be positive", file=sys.stderr)
        return 2

    try:
        device_index = resolve_input_device(args.device, args.device_name)
        device_info = sd.query_devices(device_index, "input")
        input_channels = int(device_info.get("max_input_channels", 0))
        if args.channel > input_channels:
            raise ValueError(
                f"channel {args.channel} requested but device has {input_channels} input(s)"
            )
    except (RuntimeError, ValueError) as exc:
        print(f"Audio device error: {exc}", file=sys.stderr)
        return 1

    midi_out: MidiOut | None = None
    if args.no_midi:
        sink = _NullMidiSink()
        resolved_port: str | None = None
    else:
        try:
            resolved_port = find_output_port(args.port)
            midi_out = MidiOut(resolved_port)
            midi_out.open()
            sink = ScheduledMidiSink(midi_out)
        except Exception as exc:
            print(f"MIDI setup failed: {exc}", file=sys.stderr)
            return 1

    clock = time.perf_counter
    config = LiveConfig()
    pulse_tracker = PulseTracker(min_bpm=config.min_bpm, max_bpm=config.max_bpm)
    bar_tracker = BarTracker(beats_per_bar=config.beats_per_bar)
    pulse_adapter = LivePulseAdapter(pulse_tracker, clock)
    bar_adapter = LiveBarAdapter(bar_tracker, clock)
    controller = LiveController(config, clock)
    scheduler = LiveScheduler(config, sink, clock)
    runtime = LiveRuntime(
        config, pulse_adapter, bar_adapter, controller, scheduler, clock
    )
    ingress = LiveAudioIngress(
        args.sample_rate, args.channel - 1, clock, max_blocks=256
    )
    if args.count_in:
        ingress.pause()

    event_records: list[dict[str, object]] = []

    def on_event(event: MusicalEvent) -> None:
        pulse_state = pulse_tracker.process_event(event)
        bar_tracker.update(event, pulse_state)
        runtime.observe_player_hit(event.time_seconds, event.strength)
        event_records.append(
            {
                "at": round(event.time_seconds, 6),
                "strength": round(event.strength, 4),
                "region": event.frequency_region,
                "energy": round(event.energy, 4),
            }
        )

    listener = EventListener(
        sample_rate=args.sample_rate,
        callback=on_event,
        min_interval=args.min_interval,
    )
    records: list[dict[str, object]] = []
    started = clock()
    last_trace_at = float("-inf")
    last_state: str | None = None
    interrupted = False
    error: str | None = None

    print("Bunny live runner")
    print(f"  input: [{device_index}] {device_info.get('name')} channel {args.channel}")
    print(f"  MIDI:  {resolved_port or '(disabled)'}")
    print("  clap a steady 4/4 pulse; Ctrl+C stops safely")

    try:
        with sd.InputStream(
            device=device_index,
            samplerate=args.sample_rate,
            channels=input_channels,
            dtype="float32",
            blocksize=args.block_size,
            callback=ingress.callback,
        ):
            if args.count_in:
                print(f"  GET READY: {args.count_in} audible clicks, then start clapping", flush=True)
                play_count_in(sink, args.count_in, 60.0 / args.count_in_bpm)
                # Do not let speaker bleed from the count-in become musical evidence.
                ingress.resume()
            started = clock()
            print("  GO: clap now", flush=True)
            while args.duration == 0 or clock() - started < args.duration:
                for block in ingress.drain():
                    listener.process_frame(
                        AudioFrame(block.samples, args.sample_rate, block.frame_end)
                    )
                snapshot = runtime.tick()
                now = clock()
                if snapshot.controller.state != last_state or now - last_trace_at >= 0.25:
                    records.append(
                        build_trace_record(
                            now,
                            snapshot,
                            event_count=len(event_records),
                            audio_callbacks=ingress.diag.callbacks,
                            audio_queue_depth=ingress.queue_depth,
                            audio_dropped_blocks=ingress.diag.dropped_blocks,
                        )
                    )
                    if snapshot.controller.state != last_state:
                        print(
                            f"  {snapshot.controller.state} | "
                            f"bpm={snapshot.controller.locked_bpm} "
                            f"gen={snapshot.controller.generation}"
                        )
                    last_state = snapshot.controller.state
                    last_trace_at = now

                delay = args.tick_ms / 1000.0
                next_deadline = scheduler.next_deadline()
                if next_deadline is not None:
                    delay = min(delay, max(0.0, next_deadline - clock()))
                if delay > 0:
                    time.sleep(delay)
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping...")
    except Exception as exc:  # hardware path: preserve diagnostics on failure
        error = f"{type(exc).__name__}: {exc}"
        print(f"Live runner failed: {error}", file=sys.stderr)
    finally:
        final = runtime.stop(close_midi=True)

    payload: dict[str, object] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": round(clock() - started, 3),
        "interrupted": interrupted,
        "error": error,
        "device": {"index": device_index, "name": str(device_info.get("name")), "channel": args.channel},
        "midi_port": resolved_port,
        "config": asdict(config),
        "audio": asdict(ingress.diag),
        "scheduler": asdict(scheduler.diag),
        "final": build_trace_record(
            clock(),
            final,
            event_count=len(event_records),
            audio_callbacks=ingress.diag.callbacks,
            audio_queue_depth=ingress.queue_depth,
            audio_dropped_blocks=ingress.diag.dropped_blocks,
        ),
        "timeline": records,
        "events": event_records,
    }
    trace_path = save_trace(payload, args.trace)
    print(f"Trace saved: {trace_path}")
    print(
        f"events={len(event_records)} emitted={scheduler.diag.total_emitted} "
        f"late={scheduler.diag.total_late} dropped={scheduler.diag.total_dropped} "
        f"audio_dropped={ingress.diag.dropped_blocks}"
    )
    return 1 if error else 0


if __name__ == "__main__":
    raise SystemExit(main())
