"""Repeatable live lock playtest harness for Pocket Drummer.

Records claps from an audio interface (default: AG06/AG03, channel 1),
detects clap onsets, estimates BPM, optionally plays simple_rock over
MIDI, and saves a timestamped JSON artifact to
``artifacts/live_lock_tests/`` after every take.

Typical usage::

    .venv\\Scripts\\python.exe demo_live_lock_playtest.py
    .venv\\Scripts\\python.exe demo_live_lock_playtest.py --target-bpm 120 --duration 16
    .venv\\Scripts\\python.exe demo_live_lock_playtest.py --no-play --notes "dry run"
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

try:
    import sounddevice as sd  # type: ignore[import-untyped]
except ModuleNotFoundError as exc:  # pragma: no cover – laptop lacks ASIO
    print(
        "sounddevice is required for live-audio.  "
        "Install it with:  pip install sounddevice",
        file=sys.stderr,
    )
    raise SystemExit(1) from exc

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_live_clap_lock import (  # noqa: E402  – path setup above
    analyse_claps,
    find_input_device,
)
from drummer.feel import GrooveEvent  # noqa: E402
from drummer.pipeline_midi import play_events_absolute  # noqa: E402
from groove_library import get_groove  # noqa: E402
from midi_out import MidiOut, list_output_ports  # noqa: E402

# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_DEVICE_NAME = "AG06/AG03"
DEFAULT_CHANNEL = 1
DEFAULT_THRESHOLD = 0.3
DEFAULT_MIN_SIGNAL_PEAK = 0.001
DEFAULT_DURATION_S = 12.0
DEFAULT_REPEATS = 4
DEFAULT_PORT = "PocketDrummer Out"
DEFAULT_SAMPLE_RATE = 44100
DEFAULT_MIN_INTERVAL_S = 0.18

ARTIFACT_DIR = PROJECT_ROOT / "artifacts" / "live_lock_tests"


# ── Pure helpers (testable without sounddevice or MIDI hardware) ─────────────


def compute_intervals(onset_times: list[float]) -> list[float]:
    """Return pairwise differences between consecutive onset times."""
    if len(onset_times) < 2:
        return []
    return [b - a for a, b in zip(onset_times, onset_times[1:])]


def compute_median_interval(intervals: list[float]) -> float | None:
    """Return the median of *intervals*, or ``None`` if the list is empty."""
    if not intervals:
        return None
    return float(statistics.median(intervals))


def compute_bpm_error(
    target_bpm: float | None, detected_bpm: float | None
) -> float | None:
    """Return signed BPM error ``detected - target``, or ``None``.

    A negative value means the estimate was slower than the target;
    a positive value means it was faster.
    """
    if target_bpm is None or detected_bpm is None:
        return None
    return round(detected_bpm - target_bpm, 1)


def compute_timing_diagnostics(
    timing_log: list[tuple[float, float, str, int, int]] | None,
) -> tuple[int, int, float | None, float | None]:
    """Extract ``(total_count, note_on_count, mean_abs_error_ms, max_abs_error_ms)``.

    *timing_log* is the return value of
    :func:`drummer.pipeline_midi.play_events_absolute` —
    each entry is ``(target_time, actual_time, "on"|"off", note, velocity)``.

    ``mean`` and ``max`` are computed from **note-on events only**.
    Returns ``(0, 0, None, None)`` for an empty / ``None`` log.
    """
    if not timing_log:
        return 0, 0, None, None

    total_count = len(timing_log)
    note_on_errors_ms: list[float] = []
    for target, actual, msg_type, _note, _velocity in timing_log:
        if msg_type == "on":
            error_ms = abs(actual - target) * 1000.0
            note_on_errors_ms.append(error_ms)

    note_on_count = len(note_on_errors_ms)
    if note_on_count == 0:
        return total_count, 0, None, None

    mean_abs = float(statistics.mean(note_on_errors_ms))
    max_abs = float(max(note_on_errors_ms))
    return total_count, note_on_count, round(mean_abs, 2), round(max_abs, 2)


def build_artifact(
    *,
    timestamp: str,
    device_name: str,
    device_index: int | None,
    channel: int,
    duration: float,
    sample_rate: int,
    threshold: float,
    min_signal_peak: float,
    target_bpm: float | None,
    detected_bpm: float | None,
    onset_times: list[float],
    notes: str,
    played: bool,
    midi_port: str | None,
    repeats: int,
    timing_log: list[tuple[float, float, str, int, int]] | None,
) -> dict[str, object]:
    """Assemble the full JSON artifact dictionary.

    All MIDI diagnostic fields are omitted when *played* is ``False``.
    """
    intervals = compute_intervals(onset_times)
    bpm_error = compute_bpm_error(target_bpm, detected_bpm)
    median_interval = compute_median_interval(intervals)

    artifact: dict[str, object] = {
        "timestamp": timestamp,
        "device_name": device_name,
        "device_index": device_index,
        "channel": channel,
        "duration": duration,
        "sample_rate": sample_rate,
        "threshold": threshold,
        "min_signal_peak": min_signal_peak,
        "target_bpm": target_bpm,
        "detected_bpm": detected_bpm,
        "bpm_error": bpm_error,
        "clap_count": len(onset_times),
        "clap_onset_times": [round(t, 4) for t in onset_times],
        "clap_intervals": [round(i, 4) for i in intervals],
        "median_interval": round(median_interval, 4) if median_interval is not None else None,
        "notes": notes,
        "played": played,
        "midi_port": midi_port if played else None,
        "repeats": repeats,
    }

    if played and timing_log is not None:
        total, note_on_count, mean_abs, max_abs = compute_timing_diagnostics(
            timing_log
        )
        artifact["midi_event_count"] = total
        artifact["midi_note_on_count"] = note_on_count
        artifact["midi_note_on_mean_abs_error_ms"] = mean_abs
        artifact["midi_note_on_max_abs_error_ms"] = max_abs

    return artifact


def save_artifact(artifact: dict[str, object], path: Path | None = None) -> Path:
    """Write *artifact* as JSON, creating parent directories.

    When *path* is ``None`` the file is written to
    ``artifacts/live_lock_tests/live_lock_<timestamp>.json``.
    """
    if path is None:
        ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
        # Sanitise timestamp for a safe filename
        ts = (
            str(artifact["timestamp"])
            .replace(":", "-")
            .replace("T", "_")
            .replace("+00-00", "Z")  # collapse UTC offset when present
            .rstrip("Z")
        )
        # Remove any trailing timezone suffix for a clean filename
        if ts.endswith("+00:00"):
            ts = ts[:-6]
        path = ARTIFACT_DIR / f"live_lock_{ts}.json"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, default=str)

    return path


# ── Groove conversion ────────────────────────────────────────────────────────


def _groove_to_events(groove: object) -> list[GrooveEvent]:
    """Convert a ``Groove`` model to a sorted ``list[GrooveEvent]``."""
    # We use getattr to avoid a hard import of the Groove dataclass so the
    # helper stays loosely coupled, but in practice *groove* will always be
    # the Groove model from groove_library / models.py.
    events: list[GrooveEvent] = []
    for step in getattr(groove, "kick_steps", []):
        events.append(GrooveEvent("kick", int(step), velocity=104))
    for step in getattr(groove, "snare_steps", []):
        events.append(GrooveEvent("snare", int(step), velocity=108))
    for step in getattr(groove, "hat_steps", []):
        velocity = 78 if int(step) % 8 == 0 else 68
        events.append(GrooveEvent("hi_hat", int(step), velocity=velocity))
    events.sort(key=lambda e: (e.bar_index, e.grid_position, e.instrument))
    return events


# ── Audio recording with progress ────────────────────────────────────────────


def _record_with_progress(
    device_index: int,
    duration_seconds: float,
    sample_rate: int,
    channel_number: int,
) -> np.ndarray:
    """Record audio with per-second console progress.

    Returns the selected 1‑based *channel_number* as a mono ``float32``
    array.
    """
    info = sd.query_devices(device_index, "input")
    input_channels = int(info.get("max_input_channels", 1))
    if channel_number < 1 or channel_number > input_channels:
        raise ValueError(
            f"Channel {channel_number} is out of range "
            f"(device has {input_channels} input channels)"
        )

    frames = int(duration_seconds * sample_rate)
    data = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=input_channels,
        device=device_index,
        dtype="float32",
    )

    progress_interval = 1.0
    next_progress = progress_interval
    start_time = time.monotonic()
    while time.monotonic() - start_time < duration_seconds:
        elapsed = time.monotonic() - start_time
        if elapsed >= next_progress:
            print(
                f"  listening... "
                f"{min(int(elapsed), int(duration_seconds))}/"
                f"{int(duration_seconds)}s",
                flush=True,
            )
            next_progress += progress_interval
        time.sleep(0.05)

    sd.wait()

    if data.ndim == 1:
        return np.asarray(data, dtype=np.float32)
    return np.asarray(data[:, channel_number - 1], dtype=np.float32)


# ── MIDI port resolution ─────────────────────────────────────────────────────


def _resolve_midi_port(query: str) -> str | None:
    """Return the first MIDI output port whose name contains *query*."""
    needle = query.lower()
    for name in list_output_ports():
        if needle in name.lower():
            return name
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Live clap‑to‑drummer lock playtest with artifact saving",
    )
    parser.add_argument(
        "--target-bpm",
        type=float,
        default=None,
        help="Known reference BPM (for error calculation in artifact).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=DEFAULT_DURATION_S,
        help=f"Recording duration in seconds (default: {DEFAULT_DURATION_S}).",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Free‑form notes stored in the artifact.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help=f"Onset strength threshold (default: {DEFAULT_THRESHOLD}).",
    )
    parser.add_argument(
        "--min-signal-peak",
        type=float,
        default=DEFAULT_MIN_SIGNAL_PEAK,
        help=f"Noise‑gate peak floor (default: {DEFAULT_MIN_SIGNAL_PEAK}).",
    )
    parser.add_argument(
        "--device-name",
        default=DEFAULT_DEVICE_NAME,
        help=f"Substring to match input device (default: {DEFAULT_DEVICE_NAME!r}).",
    )
    parser.add_argument(
        "--channel",
        type=int,
        default=DEFAULT_CHANNEL,
        help=f"1‑based input channel (default: {DEFAULT_CHANNEL}).",
    )
    parser.add_argument(
        "--port",
        default=DEFAULT_PORT,
        help=f"MIDI output port substring (default: {DEFAULT_PORT!r}).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help=f"Number of bar repeats for MIDI playback (default: {DEFAULT_REPEATS}).",
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Skip MIDI playback – only detect claps and estimate BPM.",
    )
    parser.add_argument(
        "--no-wait",
        action="store_true",
        help="Skip the 'Press Enter' prompt and start immediately.",
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=DEFAULT_SAMPLE_RATE,
        help=f"Audio sample rate in Hz (default: {DEFAULT_SAMPLE_RATE}).",
    )
    return parser.parse_args()


def main() -> int:  # noqa: C901  – main orchestrator; complexity is acceptable
    args = parse_args()

    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Header ───────────────────────────────────────────────────────────
    print("Pocket Drummer – Live Lock Playtest", flush=True)
    print(f"Input device filter : {args.device_name!r}", flush=True)
    print(f"Input channel       : {args.channel}", flush=True)
    if args.target_bpm is not None:
        print(f"Target BPM          : {args.target_bpm}", flush=True)
    print()

    # ── Resolve audio device ─────────────────────────────────────────────
    try:
        device_index = find_input_device(None, args.device_name)
    except ValueError as exc:
        print(f"Device error: {exc}", file=sys.stderr)
        return 1

    device_info = sd.query_devices(device_index, "input")
    print(
        f"Resolved device     : {device_index} – "
        f"{device_info.get('name', 'unknown')!r}",
        flush=True,
    )
    print(f"Sample rate         : {args.sample_rate} Hz", flush=True)
    print()

    if args.duration < 2.0:
        print("Duration must be at least 2 seconds.", file=sys.stderr)
        return 1

    # ── Performer prompt & countdown ─────────────────────────────────────
    if not args.no_wait:
        input("Press Enter when ready...")

    if not args.no_wait:
        for i in range(3, 0, -1):
            print(f"{i}...", flush=True)
            time.sleep(1)

    # ── Record ───────────────────────────────────────────────────────────
    print("LISTENING NOW – clap steady quarter notes", flush=True)
    audio = _record_with_progress(
        device_index=device_index,
        duration_seconds=args.duration,
        sample_rate=args.sample_rate,
        channel_number=args.channel,
    )
    print("Recording complete.", flush=True)

    # ── Analyse claps ────────────────────────────────────────────────────
    analysis = analyse_claps(
        samples=audio,
        sample_rate=args.sample_rate,
        min_interval_seconds=DEFAULT_MIN_INTERVAL_S,
        onset_threshold=args.threshold,
        min_signal_peak=args.min_signal_peak,
    )

    onset_times = analysis.onset_times
    # When no onsets are found, analyse_claps returns bpm=120.0 as a
    # safe default.  We treat zero or one onset as "no usable tempo".
    if len(onset_times) >= 2:
        detected_bpm: float | None = analysis.bpm
    else:
        detected_bpm = None

    intervals = compute_intervals(onset_times)

    print(f"\nDetected claps : {len(onset_times)}", flush=True)
    if onset_times:
        preview = [f"{t:.3f}" for t in onset_times[:16]]
        print(
            f"Onset times (s): {', '.join(preview)}"
            f"{'...' if len(onset_times) > 16 else ''}",
            flush=True,
        )

    if detected_bpm is not None:
        print(f"Estimated BPM   : {detected_bpm:.1f}", flush=True)
        if args.target_bpm is not None:
            error = detected_bpm - args.target_bpm
            print(f"BPM error       : {error:+.1f} BPM", flush=True)
    else:
        print(
            "Could not estimate BPM – need at least 2 claps.",
            file=sys.stderr,
        )

    # ── MIDI playback ────────────────────────────────────────────────────
    timing_log: list[tuple[float, float, str, int, int]] | None = None
    played = False

    if not args.no_play and detected_bpm is not None:
        port_name = _resolve_midi_port(args.port)
        if port_name is None:
            print(
                f"\nMIDI port {args.port!r} not found.\n"
                f"Available MIDI output ports:",
                flush=True,
            )
            for p in list_output_ports():
                print(f"  {p!r}", flush=True)
            # Continue to save artifact even without playback
        else:
            groove = get_groove("simple_rock")
            events = _groove_to_events(groove)

            print(
                f"\nPlaying drummer now... (port: {port_name!r})", flush=True
            )
            try:
                with MidiOut(port_name) as midi:
                    timing_log = play_events_absolute(
                        midi,
                        events,
                        bpm=detected_bpm,
                        repeats=args.repeats,
                    )
                played = True
                print("Playback complete.", flush=True)
            except Exception as exc:
                print(f"MIDI playback error: {exc}", file=sys.stderr)
                # Still save artifact – played stays False
    elif args.no_play:
        print("\n--no-play set: skipping MIDI playback.", flush=True)
    elif detected_bpm is None:
        print("\nSkipping playback – no BPM estimate.", flush=True)

    # ── Build & save artifact ────────────────────────────────────────────
    artifact = build_artifact(
        timestamp=timestamp,
        device_name=args.device_name,
        device_index=device_index,
        channel=args.channel,
        duration=args.duration,
        sample_rate=args.sample_rate,
        threshold=args.threshold,
        min_signal_peak=args.min_signal_peak,
        target_bpm=args.target_bpm,
        detected_bpm=detected_bpm,
        onset_times=onset_times,
        notes=args.notes,
        played=played,
        midi_port=args.port if played else None,
        repeats=args.repeats,
        timing_log=timing_log,
    )

    artifact_path = save_artifact(artifact)
    print(f"\nArtifact saved: {artifact_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())