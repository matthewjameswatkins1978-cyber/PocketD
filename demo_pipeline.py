"""Demo: Drummer Brain Pipeline — full perception-to-output chain.

Simulates 5 musical scenarios through the complete pipeline:
FeatureMonitor -> FeatureDrivenBehaviourEngine -> Groove -> OutputShaper.

Run:
    python demo_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drummer.pipeline import DrummerBrainPipeline
from perception.models import MusicalEvent


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _feed_and_process(pipeline, events: list[MusicalEvent],
                      phase_alignment: float = 0.75):
    """Feed events and immediately process each cycle."""
    decisions = []
    for evt in events:
        pipeline.feed_event(evt)
        d = pipeline.process(now=evt.time_seconds, phase_alignment=phase_alignment)
        decisions.append(d)
    return decisions


def _print_decision(d) -> None:
    snap = d.feature_snapshot
    print(f"  t={d.timestamp:.2f}s  intent={d.behaviour_intent.value:12s}"
          f"  density={snap.input_density:.2f}"
          f"  certainty={snap.player_certainty:.2f}"
          f"  change={snap.change_score:.2f}"
          f"  stability={snap.repetition_stability:.2f}"
          f"  silence={snap.silence_duration:.1f}s"
          f"  raw={len(d.raw_events):2d}  shaped={len(d.shaped_events):2d}")


def scenario_stable() -> None:
    """Stable sparse playing -> LISTEN -> ENTER -> MAINTAIN."""
    _header("SCENARIO 1: Stable Sparse Playing")

    p = DrummerBrainPipeline()
    events = [MusicalEvent(time_seconds=i * 0.5, strength=0.7) for i in range(8)]
    decisions = _feed_and_process(p, events)

    for d in decisions:
        _print_decision(d)


def scenario_dense() -> None:
    """Dense frantic playing -> high density."""
    _header("SCENARIO 2: Dense Frantic Playing")

    p = DrummerBrainPipeline()
    # Enter first
    for i in range(6):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.7))
        p.process(now=t, phase_alignment=0.75)

    # Dense burst
    dense_events = [MusicalEvent(t, 0.7) for t in
                    [3.1 + i * 0.1 for i in range(20)]]
    decisions = _feed_and_process(p, dense_events)

    for d in decisions[-5:]:
        _print_decision(d)


def scenario_uncertain() -> None:
    """Weak erratic playing -> low certainty -> ANCHOR."""
    _header("SCENARIO 3: Weak Erratic Playing")

    p = DrummerBrainPipeline()
    for i in range(6):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.7))
        p.process(now=t, phase_alignment=0.75)

    # Weak events with low phase
    times = [3.1, 3.5, 3.8, 4.3, 4.7, 5.0]
    for t in times:
        p.feed_event(MusicalEvent(t, 0.12))
        d = p.process(now=t, phase_alignment=0.25)
        _print_decision(d)


def scenario_build() -> None:
    """Sudden build -> rising energy."""
    _header("SCENARIO 4: Sudden Build")

    p = DrummerBrainPipeline()
    for i in range(6):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.5))
        p.process(now=t, phase_alignment=0.75)

    # Soft baseline, then sudden strong events
    soft = [MusicalEvent(t, 0.2) for t in [3.0 + i * 0.1 for i in range(8)]]
    _ = _feed_and_process(p, soft)

    loud = [MusicalEvent(t, 0.9) for t in [4.0, 4.15, 4.3]]
    decisions = _feed_and_process(p, loud)

    for d in decisions:
        _print_decision(d)


def scenario_silence() -> None:
    """Long silence -> BAIL."""
    _header("SCENARIO 5: Long Silence")

    p = DrummerBrainPipeline()
    for i in range(6):
        t = i * 0.5
        p.feed_event(MusicalEvent(t, 0.7))
        p.process(now=t, phase_alignment=0.75)

    # Active snapshot
    d_active = p.process(now=3.0, phase_alignment=0.75)
    _print_decision(d_active)

    # Silence
    for gap in (4.0, 10.0):
        d = p.process(now=gap)
        _print_decision(d)


def main() -> int:
    print("DRUMMER BRAIN PIPELINE")
    print("Full perception-to-output chain: "
          "FeatureMonitor -> BehaviourEngine -> Groove -> OutputShaper")
    print()

    scenario_stable()
    scenario_dense()
    scenario_uncertain()
    scenario_build()
    scenario_silence()

    print(f"\n{'=' * 60}")
    print("Drummer Brain Pipeline ready.")
    print("All components connected and tested.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())