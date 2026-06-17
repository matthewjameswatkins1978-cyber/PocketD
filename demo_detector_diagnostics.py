#!/usr/bin/env python3
"""Demo script for the Detector Diagnostics layer.

Generates synthetic audio or loads a WAV file, runs the Detector,
prints a readable report, and saves artifacts under artifacts/detector/.

Usage:
    # Single modes
    python demo_detector_diagnostics.py --synthetic pulse
    python demo_detector_diagnostics.py --synthetic noisy
    python demo_detector_diagnostics.py --file path/to/audio.wav

    # Batch modes
    python demo_detector_diagnostics.py --batch-synthetic
    python demo_detector_diagnostics.py --batch-dir path/to/wavs
"""

from __future__ import annotations

import argparse
import json
import logging
import struct
import sys
import wave
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.detector import Detector, DetectorConfig, DetectorDiagnostics, DetectorResult

log = logging.getLogger(__name__)

# ── Synthetic Audio Generators ─────────────────────────────────────

SAMPLE_RATE: int = 44100
DEFAULT_DURATION: float = 4.0


def _make_impulse(sr: int, time_s: float, amp: float = 0.7) -> np.ndarray:
    """Return a short impulse signal at a given time (seconds)."""
    total_samples = int(DEFAULT_DURATION * sr)
    signal = np.zeros(total_samples)
    sample_idx = int(time_s * sr)
    if 0 <= sample_idx < total_samples:
        impulse_len = max(1, int(sr * 0.001))
        t_env = np.linspace(0, np.pi, impulse_len)
        env = np.sin(t_env)
        end_idx = min(sample_idx + impulse_len, total_samples)
        actual_len = end_idx - sample_idx
        signal[sample_idx:end_idx] = amp * env[:actual_len]
    return signal


def _generate_pulse(sr: int = SAMPLE_RATE, bpm: float = 120.0) -> np.ndarray:
    interval = 60.0 / bpm
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    t = 0.0
    while t < DEFAULT_DURATION:
        imp = _make_impulse(sr, t, amp=0.7)
        signal += imp
        t += interval
    return signal


def _generate_noisy(sr: int = SAMPLE_RATE) -> np.ndarray:
    rng = np.random.default_rng(42)
    return rng.normal(0, 0.005, int(DEFAULT_DURATION * sr)).astype(np.float64)


def _generate_hum(sr: int = SAMPLE_RATE) -> np.ndarray:
    num_samples = int(DEFAULT_DURATION * sr)
    t = np.arange(num_samples) / sr
    return (0.03 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float64)


def _generate_sparse(sr: int = SAMPLE_RATE) -> np.ndarray:
    signal = _make_impulse(sr, 1.0, amp=0.9)
    signal += _make_impulse(sr, 3.0, amp=0.85)
    return signal


def _generate_density_rise(sr: int = SAMPLE_RATE) -> np.ndarray:
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    for t in [0.3, 1.0, 1.7]:
        signal += _make_impulse(sr, t, amp=0.6)
    base = 2.2
    for i in range(16):
        t = base + i * 0.08
        if t < DEFAULT_DURATION:
            signal += _make_impulse(sr, t, amp=0.5)
    return signal


def _generate_pulse_with_noise(sr: int = SAMPLE_RATE, bpm: float = 120.0) -> np.ndarray:
    rng = np.random.default_rng(99)
    noise = rng.normal(0, 0.003, int(DEFAULT_DURATION * sr)).astype(np.float64)
    return _generate_pulse(sr, bpm) + noise


def _generate_bass(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Plucked bass-like notes — very short, fast-decaying.

    Minimal sustain to avoid triggering sustained-tone false onsets.
    Each note is a brief burst (~100 ms) with a short percussive transient.
    """
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    notes = [
        (0.0, 55.0, 0.8),
        (0.5, 55.0, 0.75),
        (1.0, 82.4, 0.75),
        (1.5, 55.0, 0.7),
        (2.0, 110.0, 0.8),
        (2.5, 82.4, 0.7),
        (3.0, 55.0, 0.75),
        (3.5, 55.0, 0.7),
    ]
    for start_t, freq, amp in notes:
        start_idx = int(start_t * sr)
        note_len = int(0.10 * sr)  # 100 ms — very short
        t = np.arange(note_len) / sr
        # Fundamental + short high-freq noise burst at attack
        wave = amp * (
            0.7 * np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
        )
        # Noise burst for transient (string slap)
        rng = np.random.default_rng(int(start_t * 10000))
        noise_burst_len = int(0.005 * sr)
        actual_noise = min(noise_burst_len, note_len)
        wave[:actual_noise] += 0.3 * rng.normal(0, 1.0, actual_noise)
        # Very fast exponential decay
        decay = np.exp(-t * 20.0)
        wave *= decay
        end_idx = min(start_idx + note_len, len(signal))
        actual_len = end_idx - start_idx
        signal[start_idx:end_idx] += wave[:actual_len]
    return signal


def _generate_guitar(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Strummed guitar-like chords with harmonic content.

    Mid-frequency chords (~ 200-600 Hz), short strums, fast decay.
    """
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    # Chord strum timings
    strum_times = [0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6]
    for start_t in strum_times:
        start_idx = int(start_t * sr)
        chord_len = int(0.25 * sr)  # 250 ms chord
        t = np.arange(chord_len) / sr
        # Multi-frequency chord (E minor)
        wave = 0.6 * (
            0.25 * np.sin(2 * np.pi * 164.8 * t)  # E3
            + 0.20 * np.sin(2 * np.pi * 196.0 * t)  # G3
            + 0.20 * np.sin(2 * np.pi * 246.9 * t)  # B3
            + 0.15 * np.sin(2 * np.pi * 329.6 * t)  # E4
            + 0.10 * np.sin(2 * np.pi * 392.0 * t)  # G4
            + 0.10 * np.sin(2 * np.pi * 493.9 * t)  # B4
        )
        # Fast strum decay
        decay = np.exp(-t * 12.0)
        wave *= decay
        end_idx = min(start_idx + chord_len, len(signal))
        actual_len = end_idx - start_idx
        signal[start_idx:end_idx] += wave[:actual_len]
    return signal


def _generate_keyboard(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Soft keyboard/pad tones with slow attack and sustained notes.

    Should NOT produce dense false onsets like plain sine hum.
    """
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    note_times = [
        (0.2, 261.6, 1.2),  # C4, short
        (1.6, 329.6, 1.0),  # E4
        (2.8, 392.0, 0.8),  # G4
    ]
    for start_t, freq, dur in note_times:
        start_idx = int(start_t * sr)
        note_len = int(dur * sr)
        t = np.arange(note_len) / sr
        # Soft pad: fundamental + harmonics, low amplitude to avoid
        # sustained-tone false onsets during the steady-state portion.
        wave = 0.15 * (
            0.5 * np.sin(2 * np.pi * freq * t)
            + 0.3 * np.sin(2 * np.pi * freq * 2 * t)
            + 0.15 * np.sin(2 * np.pi * freq * 3 * t)
            + 0.05 * np.sin(2 * np.pi * freq * 4 * t)
        )
        # Slow attack (80ms) + hold + gentle release
        attack_samples = int(0.08 * sr)
        env = np.ones(note_len)
        if attack_samples < note_len:
            env[:attack_samples] = np.linspace(0, 1, attack_samples)
        release_start = int(note_len * 0.75)
        if release_start < note_len:
            release_len = note_len - release_start
            env[release_start:] = np.linspace(1, 0, release_len)
        wave *= env
        end_idx = min(start_idx + note_len, len(signal))
        actual_len = end_idx - start_idx
        signal[start_idx:end_idx] += wave[:actual_len]
    return signal


def _generate_drums(sr: int = SAMPLE_RATE) -> np.ndarray:
    """Synthetic drum kit: kick, snare, and hi-hat in a simple pattern.

    Two-bar 4/4 pattern at 120 BPM, each bar:
    beat 1: kick + hi-hat, beat 2: hi-hat, beat 3: snare + hi-hat, beat 4: hi-hat.
    """
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    bpm = 120.0
    beat_interval = 60.0 / bpm  # 0.5s

    def _kick(t_start: float, amp: float = 0.8) -> None:
        idx = int(t_start * sr)
        length = int(0.15 * sr)
        t = np.arange(length) / sr
        # Frequency sweep: high → low
        freq_sweep = 150.0 * np.exp(-t * 20.0) + 50.0
        wave = amp * np.sin(2 * np.pi * freq_sweep * t)
        decay = np.exp(-t * 10.0)
        wave *= decay
        end = min(idx + length, len(signal))
        signal[idx:end] += wave[: end - idx]

    def _snare(t_start: float, amp: float = 0.7) -> None:
        idx = int(t_start * sr)
        length = int(0.12 * sr)
        rng = np.random.default_rng(int(t_start * 1000))
        noise = rng.normal(0, 0.5, length)
        t = np.arange(length) / sr
        tone = amp * 0.4 * np.sin(2 * np.pi * 200.0 * t)
        wave = tone + amp * 0.6 * noise
        decay = np.exp(-t * 15.0)
        wave *= decay
        end = min(idx + length, len(signal))
        signal[idx:end] += wave[: end - idx]

    def _hat(t_start: float, amp: float = 0.3) -> None:
        idx = int(t_start * sr)
        length = int(0.05 * sr)
        rng = np.random.default_rng(int(t_start * 9999))
        noise = rng.normal(0, 0.5, length)
        t = np.arange(length) / sr
        tone = amp * 0.3 * np.sin(2 * np.pi * 8000.0 * t)
        wave = tone + amp * 0.7 * noise
        decay = np.exp(-t * 40.0)
        wave *= decay
        end = min(idx + length, len(signal))
        signal[idx:end] += wave[: end - idx]

    # Play the pattern
    for bar in range(2):
        bar_start = bar * 4 * beat_interval  # 0.0, 2.0
        for beat in range(4):
            t_beat = bar_start + beat * beat_interval
            if beat == 0:  # kick + hat
                _kick(t_beat)
                _hat(t_beat)
            elif beat == 2:  # snare + hat
                _snare(t_beat)
                _hat(t_beat)
            else:  # hat only
                _hat(t_beat)

    return signal


def _generate_pulse_bpm(sr: int = SAMPLE_RATE, bpm: float = 120.0) -> np.ndarray:
    """Generate a pulse at a specific BPM."""
    interval = 60.0 / bpm
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    t = 0.0
    while t < DEFAULT_DURATION:
        imp = _make_impulse(sr, t, amp=0.7)
        signal += imp
        t += interval
    return signal


def _generate_loose_pulse(sr: int = SAMPLE_RATE, bpm: float = 120.0, jitter_ms: float = 15.0) -> np.ndarray:
    """Generate a pulse with timing jitter."""
    interval = 60.0 / bpm
    rng = np.random.default_rng(42)
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    t = 0.0
    while t < DEFAULT_DURATION:
        jitter = rng.normal(0, jitter_ms / 1000.0)
        actual_t = max(0, t + jitter)
        imp = _make_impulse(sr, actual_t, amp=0.7)
        if actual_t < DEFAULT_DURATION - 0.01:
            signal += imp
        t += interval
    return signal


def _generate_push_pull(sr: int = SAMPLE_RATE, bpm: float = 120.0, push_ms: float = 30.0) -> np.ndarray:
    """Alternating early/late feel around a base BPM."""
    interval = 60.0 / bpm
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    t = 0.0
    beat = 0
    while t < DEFAULT_DURATION:
        offset = push_ms / 1000.0 if beat % 2 == 0 else -push_ms / 1000.0
        actual_t = max(0, t + offset)
        imp = _make_impulse(sr, actual_t, amp=0.7)
        if actual_t < DEFAULT_DURATION - 0.01:
            signal += imp
        t += interval
        beat += 1
    return signal


def _generate_tempo_drift(sr: int = SAMPLE_RATE, start_bpm: float = 110.0, end_bpm: float = 125.0) -> np.ndarray:
    """Gradual tempo drift from start_bpm to end_bpm."""
    signal = np.zeros(int(DEFAULT_DURATION * sr))
    num_beats_at_avg = int(DEFAULT_DURATION / (60.0 / ((start_bpm + end_bpm) / 2)))
    predicted_dur = num_beats_at_avg * 60.0 / ((start_bpm + end_bpm) / 2)
    t = 0.0
    beat = 0
    while t < DEFAULT_DURATION:
        frac = beat / max(num_beats_at_avg - 1, 1)
        current_bpm = start_bpm + (end_bpm - start_bpm) * frac
        imp = _make_impulse(sr, t, amp=0.7)
        signal += imp
        interval = 60.0 / current_bpm
        t += interval
        beat += 1
    return signal


_SYNTHETIC_GENERATORS = {
    "pulse": _generate_pulse,
    "noisy": _generate_noisy,
    "hum": _generate_hum,
    "sparse": _generate_sparse,
    "density-rise": _generate_density_rise,
    "pulse-with-noise": _generate_pulse_with_noise,
    "bass": _generate_bass,
    "guitar": _generate_guitar,
    "keyboard": _generate_keyboard,
    "drums": _generate_drums,
    "pulse-70": lambda sr=SAMPLE_RATE: _generate_pulse_bpm(sr, 70),
    "pulse-90": lambda sr=SAMPLE_RATE: _generate_pulse_bpm(sr, 90),
    "pulse-120": lambda sr=SAMPLE_RATE: _generate_pulse_bpm(sr, 120),
    "pulse-150": lambda sr=SAMPLE_RATE: _generate_pulse_bpm(sr, 150),
    "pulse-180": lambda sr=SAMPLE_RATE: _generate_pulse_bpm(sr, 180),
    "pulse-loose-small-120": lambda sr=SAMPLE_RATE: _generate_loose_pulse(sr, 120, 12),
    "pulse-loose-medium-120": lambda sr=SAMPLE_RATE: _generate_loose_pulse(sr, 120, 30),
    "pulse-loose-heavy-120": lambda sr=SAMPLE_RATE: _generate_loose_pulse(sr, 120, 75),
    "push-pull-120": lambda sr=SAMPLE_RATE: _generate_push_pull(sr, 120, 30),
    "tempo-drift-110-125": lambda sr=SAMPLE_RATE: _generate_tempo_drift(sr, 110, 125),
}

# ── WAV File Loading ───────────────────────────────────────────────


def load_wav(filepath: str | Path) -> tuple[np.ndarray, int]:
    """Load a WAV file and return (float64_samples, sample_rate).

    Supports 8/16/24/32-bit mono and stereo.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"WAV file not found: {filepath}")

    with wave.open(str(filepath), "rb") as wf:
        n_channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        framerate = wf.getframerate()
        n_frames = wf.getnframes()
        raw_data = wf.readframes(n_frames)

    if sample_width == 1:
        fmt = f"<{n_frames * n_channels}B"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float64)
        samples = (samples - 128.0) / 128.0
    elif sample_width == 2:
        fmt = f"<{n_frames * n_channels}h"
        samples = np.array(struct.unpack(fmt, raw_data), dtype=np.float64)
        samples = samples / 32768.0
    elif sample_width == 3:
        raw = np.frombuffer(raw_data, dtype=np.uint8).astype(np.float64)
        total = n_frames * n_channels
        reshaped = raw.reshape(total, 3)
        samples = reshaped[:, 0] + reshaped[:, 1] * 256.0 + reshaped[:, 2] * 65536.0
        mask = samples >= 8388608.0
        samples[mask] -= 16777216.0
        samples = samples / 8388608.0
    elif sample_width == 4:
        if n_frames > 0:
            fmt = f"<{n_frames * n_channels}i"
            samples_i = np.array(struct.unpack(fmt, raw_data), dtype=np.float64)
            if np.max(np.abs(samples_i)) < 1.0:
                fmt_f = f"<{n_frames * n_channels}f"
                samples = np.array(struct.unpack(fmt_f, raw_data), dtype=np.float64)
            else:
                samples = samples_i / 2147483648.0
        else:
            samples = np.array([], dtype=np.float64)
    else:
        raise ValueError(f"Unsupported sample width: {sample_width} bytes")

    if n_channels > 1:
        samples = samples.reshape(n_frames, n_channels)
        samples = samples.mean(axis=1)
    else:
        samples = samples.reshape(n_frames)

    return samples.astype(np.float64), framerate


# ── Batch Helpers (Phase 5) ────────────────────────────────────────


@dataclass
class _BatchRecord:
    """Lightweight record for batch diagnostics processing."""

    source: str
    result: "DetectorResult | None" = None
    error: str = ""
    quality: str = ""
    state: str = ""
    event_count: int = 0
    raw_count: int = 0
    rejected: int = 0
    bpm: str = "-"
    confidence: float = 0.0
    flags: str = ""

    @classmethod
    def from_success(
        cls,
        source_name: str,
        result: DetectorResult,
    ) -> "_BatchRecord":
        d = result.diagnostics
        bpm_str = f"{d.pulse_bpm:.1f}" if d.pulse_bpm is not None else "-"
        flags_str = ", ".join(d.summary_flags) if d.summary_flags else "-"
        return cls(
            source=source_name,
            result=result,
            quality=d.input_quality,
            state=d.musical_state,
            event_count=d.event_count,
            raw_count=d.event_count + d.rejected_count,
            rejected=d.rejected_count,
            bpm=bpm_str,
            confidence=d.pulse_confidence,
            flags=flags_str,
        )

    @classmethod
    def from_error(cls, source_name: str, error_msg: str) -> "_BatchRecord":
        return cls(source=source_name, result=None, error=error_msg)


def run_detector_for_source(
    source_name: str,
    signal: np.ndarray,
    sample_rate: int,
    output_dir: Path,
    save: bool = True,
) -> _BatchRecord:
    """Run detector on one signal and optionally save reports.

    Returns a _BatchRecord with the result and pre-computed summary fields.
    On error, returns a record with error field populated.
    """
    try:
        detector = Detector()
        result = detector.detect(signal, sample_rate)
    except Exception as exc:
        return _BatchRecord.from_error(source_name, str(exc))

    record = _BatchRecord.from_success(source_name, result)

    if save:
        output_dir.mkdir(parents=True, exist_ok=True)

        # Text report
        report_path = output_dir / f"{source_name}_diagnostics.txt"
        report_path.write_text(Detector.format_report(result), encoding="utf-8")

        # JSON events
        json_path = output_dir / f"{source_name}_events.json"
        events_data = []
        for e in result.events:
            events_data.append({
                "time_seconds": round(e.time_seconds, 6),
                "strength": round(e.strength, 4),
                "frequency_region": e.frequency_region,
                "energy": round(e.energy, 4),
                "density": round(e.density, 4),
            })
        json_path.write_text(json.dumps(events_data, indent=2), encoding="utf-8")

    return record


def write_batch_summary(
    records: list[_BatchRecord],
    output_path: Path,
) -> str:
    """Generate a Markdown batch summary and write it to *output_path*.

    Returns the summary text.
    """
    # Separate error records
    good = [r for r in records if not r.error]
    errors = [r for r in records if r.error]
    total = len(records)

    quality_counts = Counter(r.quality for r in good)
    state_counts = Counter(r.state for r in good)

    # Collate flags across all records
    all_flags: Counter[str] = Counter()
    for r in good:
        d = r.result.diagnostics
        if d:
            for f in d.summary_flags:
                all_flags[f] += 1

    # Best / worst
    best = [r for r in good if r.quality == "good"]
    worst = [r for r in good if r.quality in ("noisy", "unusable")]

    lines: list[str] = []
    lines.append("# Detector Batch Summary")
    lines.append("")
    lines.append(f"**Total inputs:** {total}")
    lines.append(f"**Successful:** {len(good)}")
    if errors:
        lines.append(f"**Errors:** {len(errors)}")
    lines.append("")

    # ── Quality breakdown ──────────────────────────────────────────
    lines.append("## Input Quality")
    lines.append("")
    for label in ("good", "usable", "weak", "noisy", "unusable"):
        count = quality_counts.get(label, 0)
        lines.append(f"- **{label}**: {count}")
    lines.append("")

    # ── Musical state breakdown ────────────────────────────────────
    lines.append("## Musical State")
    lines.append("")
    for label in ("stable_pulse", "sparse_hits", "dense_activity", "silence", "noise", "unstable"):
        count = state_counts.get(label, 0)
        lines.append(f"- **{label}**: {count}")
    lines.append("")

    # ── Summary flags ──────────────────────────────────────────────
    if all_flags:
        lines.append("## Summary Flags")
        lines.append("")
        for flag, count in all_flags.most_common():
            lines.append(f"- `{flag}`: {count}")
        lines.append("")

    # ── Best candidates ────────────────────────────────────────────
    if best:
        lines.append("## Best Candidates for Downstream Testing")
        lines.append("")
        for r in best:
            d = r.result.diagnostics
            bpm_str = f"{d.pulse_bpm:.1f} BPM" if d.pulse_bpm else "unknown"
            lines.append(f"- **{r.source}** — {bpm_str}, confidence {d.pulse_confidence:.3f}")
        lines.append("")

    # ── Problem inputs ─────────────────────────────────────────────
    if worst:
        lines.append("## Problem Inputs")
        lines.append("")
        for r in worst:
            flags_str = ", ".join(r.result.diagnostics.summary_flags)
            lines.append(f"- **{r.source}** — quality={r.quality}, state={r.state}")
            if flags_str:
                lines.append(f"  Flags: {flags_str}")
        lines.append("")

    # ── Errors ─────────────────────────────────────────────────────
    if errors:
        lines.append("## Errors")
        lines.append("")
        for r in errors:
            lines.append(f"- **{r.source}**: {r.error}")
        lines.append("")

    # ── Compact table ──────────────────────────────────────────────
    lines.append("## Per-Input Table")
    lines.append("")
    lines.append(
        "| Source | Quality | State | Events | Raw | Rejected | BPM | Conf | Flags |"
    )
    lines.append(
        "|--------|---------|-------|--------|-----|----------|-----|------|-------|"
    )
    for r in good:
        lines.append(
            f"| {r.source} | {r.quality} | {r.state} | {r.event_count} | "
            f"{r.raw_count} | {r.rejected} | {r.bpm} | "
            f"{r.confidence:.3f} | {r.flags} |"
        )
    for r in errors:
        lines.append(f"| {r.source} | ERROR | error | - | - | - | - | - | {r.error} |")
    lines.append("")

    text = "\n".join(lines)
    output_path.write_text(text, encoding="utf-8")
    return text


# ── Main ────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detector Diagnostics Demo — synthetic, WAV file, or batch input",
    )

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--synthetic",
        choices=[
            "pulse", "noisy", "hum", "sparse", "density-rise", "pulse-with-noise",
            "bass", "guitar", "keyboard", "drums",
            "pulse-70", "pulse-90", "pulse-120", "pulse-150", "pulse-180",
            "pulse-loose-small-120", "pulse-loose-medium-120", "pulse-loose-heavy-120",
            "push-pull-120", "tempo-drift-110-125",
        ],
        help="Synthetic input type",
    )
    mode_group.add_argument(
        "--file",
        type=str,
        metavar="PATH",
        help="Path to a WAV file for analysis",
    )
    mode_group.add_argument(
        "--batch-synthetic",
        action="store_true",
        help="Run all built-in synthetic scenarios",
    )
    mode_group.add_argument(
        "--batch-dir",
        type=str,
        metavar="DIR",
        help="Run detector on all WAV files in a directory",
    )

    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "artifacts" / "detector"),
        help="Directory for output artifacts (default: artifacts/detector)",
    )
    parser.add_argument(
        "--summary-file",
        default=None,
        help="Batch summary filename (default: detector_batch_summary.md in output-dir)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving artifacts to disk",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories (--batch-dir only)",
    )
    parser.add_argument(
        "--pattern",
        default="*.wav",
        help="Glob pattern for WAV files (default: *.wav)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s | %(name)s | %(message)s",
    )

    out_dir = Path(args.output_dir)

    # ─── Single-synthetic mode ─────────────────────────────────────
    if args.synthetic:
        gen_fn = _SYNTHETIC_GENERATORS[args.synthetic]
        signal = gen_fn(SAMPLE_RATE)
        sample_rate = SAMPLE_RATE
        source_label = args.synthetic

        print(f"\n[DEMO] Source: {source_label}")
        print(f"       Signal shape: {signal.shape}, sample rate: {sample_rate}")

        record = run_detector_for_source(
            source_label, signal, sample_rate, out_dir, save=not args.no_save,
        )
        if record.error:
            print(f"[ERROR] {record.error}", file=sys.stderr)
            sys.exit(1)

        print(Detector.format_report(record.result))
        if not args.no_save:
            print(f"[SAVED] Report: {out_dir / f'{source_label}_diagnostics.txt'}")
            print(f"[SAVED] Events JSON: {out_dir / f'{source_label}_events.json'}")

    # ─── Single-file mode ──────────────────────────────────────────
    elif args.file:
        filepath = Path(args.file)
        source_label = f"file_{filepath.stem}"
        try:
            signal, sample_rate = load_wav(filepath)
        except (FileNotFoundError, ValueError, wave.Error) as exc:
            print(f"[ERROR] Failed to load WAV: {exc}", file=sys.stderr)
            sys.exit(1)

        print(f"\n[DEMO] Source: {source_label}")
        print(f"       Signal shape: {signal.shape}, sample rate: {sample_rate}")

        record = run_detector_for_source(
            source_label, signal, sample_rate, out_dir, save=not args.no_save,
        )
        if record.error:
            print(f"[ERROR] {record.error}", file=sys.stderr)
            sys.exit(1)

        print(Detector.format_report(record.result))
        if not args.no_save:
            print(f"[SAVED] Report: {out_dir / f'{source_label}_diagnostics.txt'}")
            print(f"[SAVED] Events JSON: {out_dir / f'{source_label}_events.json'}")

    # ─── Batch synthetic ───────────────────────────────────────────
    elif args.batch_synthetic:
        records: list[_BatchRecord] = []
        for name, gen_fn in _SYNTHETIC_GENERATORS.items():
            signal = gen_fn(SAMPLE_RATE)
            record = run_detector_for_source(
                name, signal, SAMPLE_RATE, out_dir, save=not args.no_save,
            )
            records.append(record)
            label = "DONE" if not record.error else f"ERROR: {record.error}"
            print(f"  [{label}] {name}")

        summary_path = Path(args.summary_file) if args.summary_file else out_dir / "detector_batch_summary.md"
        write_batch_summary(records, summary_path)
        print(f"\n[SAVED] Batch summary: {summary_path}")
        print(f"  {len(records)} inputs processed")

    # ─── Batch directory ───────────────────────────────────────────
    elif args.batch_dir:
        batch_dir = Path(args.batch_dir)
        if not batch_dir.is_dir():
            print(f"[ERROR] Not a directory: {batch_dir}", file=sys.stderr)
            sys.exit(1)

        # Collect WAV files
        if args.recursive:
            wav_files = sorted(batch_dir.rglob(args.pattern))
        else:
            wav_files = sorted(batch_dir.glob(args.pattern))

        if not wav_files:
            print(f"[WARN] No files matching '{args.pattern}' found in {batch_dir}")
            sys.exit(0)

        records: list[_BatchRecord] = []
        for wav_path in wav_files:
            source_label = f"file_{wav_path.stem}"
            try:
                signal, sr = load_wav(wav_path)
                record = run_detector_for_source(
                    source_label, signal, sr, out_dir, save=not args.no_save,
                )
            except (FileNotFoundError, ValueError, wave.Error, OSError) as exc:
                record = _BatchRecord.from_error(source_label, str(exc))
            records.append(record)
            label = "DONE" if not record.error else f"ERROR: {record.error}"
            print(f"  [{label}] {source_label}")

        summary_path = Path(args.summary_file) if args.summary_file else out_dir / "detector_batch_summary.md"
        write_batch_summary(records, summary_path)
        print(f"\n[SAVED] Batch summary: {summary_path}")
        print(f"  {len(records)} inputs processed")

    print()


if __name__ == "__main__":
    main()