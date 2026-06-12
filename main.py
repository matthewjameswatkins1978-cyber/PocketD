"""Pocket Drummer — entry point.

Usage
-----
    python main.py --milestone 1 --midi-port "loopMIDI"       # MIDI groove mode
    python main.py --milestone 2                              # perception engine demo
    python main.py --milestone 2 --mode streaming             # streaming mode
    python main.py --milestone 3                              # pulse tracker demo
    python main.py --milestone 4                              # bar tracker demo
    python main.py --milestone 5                              # groove intent demo
    python main.py --list-ports                               # list MIDI ports
"""

from __future__ import annotations

import argparse
import logging
import sys

from midi_out import find_output_port, list_output_ports
from scheduler import run_hardcoded_groove


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pocket Drummer — listen, lock, groove.",
    )
    parser.add_argument(
        "--milestone",
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5],
        help="Milestone: 1=groove→MIDI, 2=event listener, 3=pulse tracker, 4=bar tracker, 5=groove intent",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="offline",
        choices=["offline", "streaming"],
        help="Mode for milestones 2 and 3. Default: offline",
    )
    parser.add_argument(
        "--midi-port",
        type=str,
        default="",
        help="MIDI output port name (substring match) — milestone 1 only",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List available MIDI output ports and exit",
    )
    parser.add_argument(
        "--groove",
        type=str,
        default="simple_rock",
        help="Groove id from data/grooves.yaml (default: simple_rock) — milestone 1",
    )
    parser.add_argument(
        "--bpm",
        type=float,
        default=120.0,
        help="Tempo in BPM (default: 120)",
    )
    parser.add_argument(
        "--bars",
        type=int,
        default=None,
        help="Number of bars to play (default: loop until Ctrl+C) — milestone 1",
    )
    parser.add_argument(
        "--complexity",
        type=int,
        default=5,
        choices=[1, 2, 3, 4, 5],
        help="Complexity level 1-5 (default: 5 = full groove) — milestone 1",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=8.0,
        help="Duration in seconds for perception demos (default: 8) — milestones 2/3",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser


def run_perception_engine(args: argparse.Namespace) -> int:
    """Run the perception engine demo (Module 1 — Event Listener)."""
    log = logging.getLogger("pocket_drummer")

    try:
        from perception.event_listener import (
            AudioFrame,
            EventListener,
            detect_events_from_audio,
        )
        from perception.models import MusicalEvent
    except ImportError as exc:
        log.error("Perception engine not available: %s", exc)
        log.error("Ensure all dependencies are installed (pip install -r requirements.txt)")
        return 1

    import numpy as np

    log.info("Milestone 2 — Perception Engine: Event Listener")
    log.info("Mode: %s | Duration: %.1fs | BPM: %.1f", args.mode, args.duration, args.bpm)

    # ── Generate synthetic drum-like audio ──────────────────────
    def generate_signal(
        sample_rate: int = 44100,
        duration: float = 8.0,
        bpm: float = 120.0,
    ) -> np.ndarray:
        total = int(sample_rate * duration)
        signal = np.zeros(total, dtype=np.float64)
        beat = 60.0 / bpm

        def hit(time_sec: float, freq: float, decay: float, amp: float) -> None:
            start = int(time_sec * sample_rate)
            if start >= total:
                return
            length = int(decay * sample_rate)
            end = min(start + length, total)
            t = np.arange(end - start) / sample_rate
            if len(t) == 0:
                return
            h = amp * np.sin(2 * np.pi * freq * t) * np.exp(-t * 20.0 / decay)
            signal[start:end] += h[: len(t)]

        for b in [0, 2, 4, 6]:
            offset = b * beat
            hit(offset, 60.0, 0.2, 0.8)
            hit(offset, 150.0, 0.05, 0.6)

        for b in [1, 3, 5, 7]:
            offset = b * beat
            hit(offset, 200.0, 0.15, 0.7)
            hit(offset, 400.0, 0.08, 0.5)

        for b in range(int(8 * 2)):
            offset = b * beat / 2
            amp = 0.3 if b % 2 == 0 else 0.2
            hit(offset, 8000.0, 0.04, amp)

        fill_start = 4.0
        for i in range(32):
            offset = fill_start + (i / 32) * 2.0
            freq = 100.0 + (i % 3) * 300.0
            amp = min(0.9, 0.4 + (i % 4) * 0.1)
            hit(offset, freq, 0.06, amp)

        max_val = np.max(np.abs(signal))
        if max_val > 0:
            signal = signal / max_val * 0.95
        return signal

    sample_rate = 44100

    if args.mode == "offline":
        print("=" * 50)
        print("PERCEPTION ENGINE — MODULE 1: EVENT LISTENER (Offline)")
        print("=" * 50)
        audio = generate_signal(sample_rate=sample_rate, duration=args.duration, bpm=args.bpm)
        events = detect_events_from_audio(audio, sample_rate)

        print(f"\nDetected {len(events)} musical events in {args.duration:.1f}s")
        for ev in events:
            strength = round(ev.strength * 100)
            energy = round(ev.energy * 100)
            density = round(ev.density * 100)
            region = ev.frequency_region.capitalize()
            print(f"  [{ev.time_seconds:6.2f}s] Strength={strength:3d}  Region={region:8s}  "
                  f"Energy={energy:3d}%  Density={density:3d}%")

        if events:
            regions: dict[str, int] = {}
            for e in events:
                regions[e.frequency_region] = regions.get(e.frequency_region, 0) + 1
            print(f"\n  Frequency breakdown:")
            for r, c in sorted(regions.items(), key=lambda x: -x[1]):
                print(f"    {r.capitalize():8s}: {c:3d} events")
        return 0

    else:  # streaming
        print("=" * 50)
        print("PERCEPTION ENGINE — MODULE 1: EVENT LISTENER (Streaming)")
        print("=" * 50)
        audio = generate_signal(sample_rate=sample_rate, duration=min(args.duration, 4.0), bpm=args.bpm)

        received: list[MusicalEvent] = []

        def callback(ev: MusicalEvent) -> None:
            received.append(ev)
            strength = round(ev.strength * 100)
            energy = round(ev.energy * 100)
            region = ev.frequency_region.capitalize()
            print(f"  [{ev.time_seconds:6.2f}s] Attack  Strength={strength:3d}  "
                  f"Region={region:8s}  Energy={energy:3d}%")

        listener = EventListener(sample_rate=sample_rate, callback=callback)
        frame_size = 1024

        for i in range(len(audio) // frame_size):
            start = i * frame_size
            end = start + frame_size
            frame = AudioFrame(
                samples=audio[start:end],
                sample_rate=sample_rate,
                time_seconds=end / sample_rate,
            )
            listener.process_frame(frame)

        flushed = listener.flush()
        print(f"\n  Streamed {len(received)} events (+ {len(flushed)} flushed)")
        return 0


def run_pulse_tracker(args: argparse.Namespace) -> int:
    """Run the pulse tracker demo (Module 2 — Pulse Perception)."""
    log = logging.getLogger("pocket_drummer")

    try:
        from perception.models import MusicalEvent
        from perception.pulse import PulseTracker
    except ImportError as exc:
        log.error("Pulse tracker not available: %s", exc)
        return 1

    log.info("Milestone 3 — Pulse Tracker: competing tempo hypotheses")
    log.info("Mode: %s | Duration: %.1fs | BPM: %.1f", args.mode, args.duration, args.bpm)

    def _event(t: float, strength: float = 0.8, energy: float = 0.5) -> MusicalEvent:
        return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=0.5)

    def _print_state(state, label: str = "") -> None:
        if state.best_bpm is not None:
            print(f"  [{state.best_bpm:.1f} BPM | Confidence: {state.confidence:.0%} | "
                  f"Stability: {state.stability}]")
        else:
            print(f"  [-- BPM | Confidence: {state.confidence:.0%} | "
                  f"Stability: {state.stability}]")
        for hyp in state.hypotheses[:3]:
            print(f"    {hyp.bpm:.1f} BPM | Confidence: {hyp.confidence:.0%}")

    # ── Pattern 1: Steady 120 BPM ─────────────────────────────
    print("\n" + "=" * 50)
    print("PATTERN 1: Steady 120 BPM")
    print("=" * 50)

    tracker = PulseTracker()
    interval = 60.0 / 120.0
    for i in range(10):
        t = i * interval
        tracker.process_event(_event(t))
        state = tracker.get_state()
        if i >= 2:
            print(f"\n[{t:.2f}s] PULSE STATE")
            _print_state(state)

    # ── Pattern 2: Half-time ambiguity ────────────────────────
    print("\n" + "=" * 50)
    print("PATTERN 2: Half-time ambiguity (events at 1.0s)")
    print("=" * 50)

    tracker2 = PulseTracker()
    for i in range(8):
        t = i * 1.0
        strength = 1.0 if i % 2 == 0 else 0.6
        tracker2.process_event(_event(t, strength=strength))
        state = tracker2.get_state()
        print(f"\n[{t:.2f}s] PULSE STATE")
        _print_state(state)

    # ── Pattern 3: Pulse decay over silence ───────────────────
    print("\n" + "=" * 50)
    print("PATTERN 3: Pulse decay over silence")
    print("=" * 50)

    tracker3 = PulseTracker()
    for i in range(6):
        t = i * 0.5
        tracker3.process_event(_event(t))
        state = tracker3.get_state()

    print(f"\nAfter 6 events (3.0s):")
    _print_state(tracker3.get_state())

    state = tracker3.advance_time(current_time=15.0)
    print(f"\nAfter 12s silence (15.0s):")
    _print_state(state)

    print("\n" + "=" * 50)
    print("Module 3 complete — pulse perception established.")
    print("=" * 50)
    return 0


def run_groove_intent(args: argparse.Namespace) -> int:
    """Run the groove intent engine demo (Module 4 — Groove Intent)."""
    log = logging.getLogger("pocket_drummer")

    try:
        from drummer.intent import GrooveIntentEngine
        from perception.bar import BarHypothesis, BarState
        from perception.models import MusicalEvent
        from perception.pulse import PulseHypothesis, PulseState
    except ImportError as exc:
        log.error("Groove intent engine not available: %s", exc)
        return 1

    log.info("Milestone 5 — Groove Intent Engine: perception-to-behaviour")

    def _event(t: float, strength: float = 0.8, energy: float = 0.5, density: float = 0.5) -> MusicalEvent:
        return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=density)

    def _pulse(conf: float) -> PulseState:
        return PulseState(
            hypotheses=[PulseHypothesis(bpm=120.0, confidence=conf, matches=10)],
            best_bpm=120.0, confidence=conf, stability="stable" if conf > 0.5 else "rising",
        )

    def _bar(conf: float, beat: int) -> BarState:
        return BarState(
            hypotheses=[BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=4, confidence=conf)],
            best_hypothesis=BarHypothesis(bpm=120.0, beat_interval=0.5, beats_per_bar=4, confidence=conf),
            is_confident=conf > 0.5, estimated_bar_position=float(beat),
            estimated_beat_in_bar=beat, confidence=conf, timestamp=0.0,
        )

    def _print(i) -> None:
        print(f"  {i.action.name:15s}  Play={str(i.should_play):5s}  Fill={str(i.should_fill):5s}  "
              f"Complex={i.suggested_complexity:.2f}  Vel={i.suggested_velocity:.2f}")
        print(f"    Energy={i.energy_level:.2f}  Density={i.density_level:.2f}  "
              f"Pulse={i.pulse_confidence:.0%}  Bar={i.bar_confidence:.0%}")
        print(f"    Reason: {i.reason}")

    engine = GrooveIntentEngine()
    t = 0.0

    print("\n" + "=" * 55)
    print("PHASE 1: WAIT — low confidence")
    print("=" * 55)
    for _ in range(3):
        t += 0.5
        intent = engine.update(_event(t, energy=0.4), _pulse(0.2), _bar(0.15, 0))
        print(f"\n[{t:.1f}s]")
        _print(intent)

    print("\n" + "=" * 55)
    print("PHASE 2: ENTER — confidence rises")
    print("=" * 55)
    for i in range(3):
        t += 0.5
        intent = engine.update(_event(t, energy=0.5), _pulse(0.50), _bar(0.40, i % 4))
        print(f"\n[{t:.1f}s]")
        _print(intent)

    print("\n" + "=" * 55)
    print("PHASE 3: HOLD — steady musical input")
    print("=" * 55)
    for i in range(4):
        t += 0.5
        intent = engine.update(_event(t, energy=0.55, density=0.5), _pulse(0.75), _bar(0.70, i % 4))
        print(f"\n[{t:.1f}s]")
        _print(intent)

    print("\n" + "=" * 55)
    print("PHASE 4: BUILD — energy and density rising")
    print("=" * 55)
    for i in range(4):
        t += 0.5
        intent = engine.update(_event(t, energy=0.4 + i * 0.15, density=0.3 + i * 0.2), _pulse(0.80), _bar(0.80, i % 4))
        print(f"\n[{t:.1f}s]")
        _print(intent)

    print("\n" + "=" * 55)
    print("PHASE 5: REDUCE — energy dropping")
    print("=" * 55)
    for i in range(4):
        t += 0.5
        intent = engine.update(_event(t, energy=0.8 - i * 0.18, density=0.5 - i * 0.12), _pulse(0.80), _bar(0.80, i % 4))
        print(f"\n[{t:.1f}s]")
        _print(intent)

    print("\n" + "=" * 55)
    print("Module 5 complete — groove intent established.")
    print("The drummer knows what kind of behaviour is appropriate.")
    print("=" * 55)
    return 0


def run_bar_tracker(args: argparse.Namespace) -> int:
    """Run the bar tracker demo (Module 3 — Bar / Downbeat Tracker)."""
    log = logging.getLogger("pocket_drummer")

    try:
        from perception.bar import BarTracker
        from perception.models import MusicalEvent
        from perception.pulse import PulseTracker
    except ImportError as exc:
        log.error("Bar tracker not available: %s", exc)
        return 1

    log.info("Milestone 4 — Bar Tracker: downbeat and bar position estimation")

    def _event(t: float, strength: float = 0.8, energy: float = 0.5) -> MusicalEvent:
        return MusicalEvent(time_seconds=t, strength=strength, energy=energy, density=0.5)

    def _print_bar(state, label: str = "") -> None:
        if state.best_hypothesis is None:
            print(f"  [-- BPM | no bar hypothesis yet]")
            return
        best = state.best_hypothesis
        beat_pos = state.estimated_beat_in_bar
        if beat_pos is None:
            beat_pos = "?"
        print(f"  BPM: {best.bpm:.1f}  |  Downbeat@{best.downbeat_time:.2f}s  |  "
              f"Beat-in-bar: {beat_pos}  |  Conf: {state.confidence:.0%}  |  "
              f"Hypotheses: {len(state.hypotheses)}")

    # ── Pattern 1: Clear 4/4 at 120 BPM ──────────────────────
    print("\n" + "=" * 55)
    print("PATTERN 1: Clear 4/4 at 120 BPM (4 bars)")
    print("=" * 55)

    pulse_tracker = PulseTracker()
    bar_tracker = BarTracker()
    beat = 0.5

    for bar_num in range(4):
        base = bar_num * 2.0
        for offset, strength in [(0.0, 1.0), (0.5, 0.3), (1.0, 0.9), (1.5, 0.3)]:
            ev = _event(base + offset, strength=strength, energy=strength * 0.9 + 0.05)
            pulse_state = pulse_tracker.process_event(ev)
            bar_state = bar_tracker.update(ev, pulse_state)

        state = bar_tracker.get_state(current_time=base + 2.0)
        print(f"\n[Bar {bar_num + 1}] t={base + 2.0:.1f}s  "
              f"Pulse: {pulse_state.best_bpm:.0f} BPM ({pulse_state.confidence:.0%})")
        _print_bar(state)

    # ── Pattern 2: Ambiguous half-bar ─────────────────────────
    print("\n" + "=" * 55)
    print("PATTERN 2: Ambiguous (strong beats every 1.0s)")
    print("=" * 55)

    pulse_tracker2 = PulseTracker()
    bar_tracker2 = BarTracker()

    for i in range(8):
        t = i * 1.0
        strength = 1.0 if i % 2 == 0 else 0.4
        ev = _event(t, strength=strength, energy=strength * 0.9 + 0.05)
        pulse_state = pulse_tracker2.process_event(ev)
        bar_state = bar_tracker2.update(ev, pulse_state)

        if pulse_state.best_bpm is not None:
            print(f"\n[{t:.1f}s] Pulse={pulse_state.best_bpm:.0f} BPM")
        else:
            print(f"\n[{t:.1f}s] Pulse=-- BPM")
        _print_bar(bar_state)

    print("\n" + "=" * 55)
    print("Module 4 complete — bar perception established.")
    print("The animal knows where the bars breathe.")
    print("=" * 55)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("pocket_drummer")

    # ── List MIDI ports and exit ─────────────────────────────
    if args.list_ports:
        ports = list_output_ports()
        if not ports:
            print("No MIDI output ports found.")
            print("Create one in loopMIDI, then retry.")
            return 1
        print("MIDI output ports:")
        for i, name in enumerate(ports):
            print(f"  [{i}] {name}")
        return 0

    # ── Milestone 5: Groove Intent ──────────────────────────
    if args.milestone == 5:
        return run_groove_intent(args)

    # ── Milestone 4: Bar Tracker ────────────────────────────
    if args.milestone == 4:
        return run_bar_tracker(args)

    # ── Milestone 3: Pulse Tracker ───────────────────────────
    if args.milestone == 3:
        return run_pulse_tracker(args)

    # ── Milestone 2: Perception Engine ───────────────────────
    if args.milestone == 2:
        return run_perception_engine(args)

    # ── Milestone 1: Drummer groove → MIDI ───────────────────
    if not args.midi_port:
        log.error("Milestone 1 requires --midi-port. Use --list-ports to see options.")
        return 1

    try:
        port = find_output_port(args.midi_port)
    except ValueError as exc:
        log.error("%s", exc)
        return 1

    log.info("Milestone %d — hard-coded groove → MIDI", args.milestone)
    log.info("Port: %s | Groove: %s | BPM: %.1f", port, args.groove, args.bpm)
    log.info("Press Ctrl+C to stop.")

    if args.milestone == 1:
        run_hardcoded_groove(
            midi_port=port,
            groove_id=args.groove,
            bpm=args.bpm,
            bars=args.bars,
            complexity_level=args.complexity,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())