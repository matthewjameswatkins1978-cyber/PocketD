"""Demo: Feature Monitor — Module 5: Musical Feature Tracking.

Demonstrates the Feature Monitor's ability to summarise player behaviour
from a stream of MusicalEvent objects.  Simulates four musical scenarios
and prints FeatureSnapshot values.

Run:
    python demo_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from perception.features import FeatureMonitor, FeatureMonitorConfig
from perception.models import MusicalEvent


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _print_snap(label: str, snap) -> None:
    """Pretty-print a FeatureSnapshot."""
    print(f"\n  [{label}]  t={snap.timestamp:.2f}s")
    print(f"    input_density       : {snap.input_density:.3f}")
    print(f"    strength_ema        : {snap.strength_ema:.3f}")
    print(f"    fast_strength_ema   : {snap.fast_strength_ema:.3f}")
    print(f"    slow_strength_ema   : {snap.slow_strength_ema:.3f}")
    print(f"    change_score        : {snap.change_score:.3f}")
    print(f"    silence_duration    : {snap.silence_duration:.3f}s")
    print(f"    repetition_stability: {snap.repetition_stability:.3f}")
    pa = f"{snap.phase_alignment:.3f}" if snap.phase_alignment is not None else "None"
    print(f"    phase_alignment     : {pa}")
    print(f"    player_certainty    : {snap.player_certainty:.3f}")


def scenario_sparse_stable() -> None:
    """Simulate sparse but stable playing — confident, regular hits."""
    _header("SCENARIO 1: Sparse Stable Playing")

    fm = FeatureMonitor()
    beats = 16
    spacing = 0.5  # 120 BPM quarter notes
    for i in range(beats):
        t = i * spacing
        event = MusicalEvent(
            time_seconds=t,
            strength=0.7,
            frequency_region="low" if i % 2 == 0 else "low_mid",
        )
        fm.feed(event)

    # Final snapshot with good phase alignment
    snap = fm.snapshot(now=beats * spacing, phase_alignment=0.9)
    _print_snap("Final (good phase)", snap)

    # Also show what phase=None looks like
    snap_no_phase = fm.snapshot(now=beats * spacing)
    _print_snap("Final (no phase)", snap_no_phase)

    print("\n  -> Expected: high repetition_stability, moderate density,")
    print("    high certainty (especially with phase alignment).")


def scenario_dense_frantic() -> None:
    """Simulate dense, frantic playing — lots of events, high energy."""
    _header("SCENARIO 2: Dense Frantic Playing")

    config = FeatureMonitorConfig(max_expected_density=8.0)
    fm = FeatureMonitor(config=config)
    # 16th notes at 150 BPM ≈ 0.1s spacing
    n_hits = 40
    spacing = 0.1
    for i in range(n_hits):
        t = i * spacing
        # Varying strength slightly but staying high
        strength = 0.7 + (i % 3) * 0.1
        fm.feed(MusicalEvent(time_seconds=t, strength=min(strength, 1.0)))

    snap = fm.snapshot(now=n_hits * spacing, phase_alignment=0.5)
    _print_snap("Final (moderate phase)", snap)

    print("\n  -> Expected: high density (clamped at 1.0),")
    print("    high strength_ema, moderate repetition_stability,")
    print("    certainty reflects strength and stability.")


def scenario_sudden_build() -> None:
    """Simulate a sudden build — player goes from soft to loud."""
    _header("SCENARIO 3: Sudden Build")

    config = FeatureMonitorConfig(
        fast_strength_alpha=0.4,
        slow_strength_alpha=0.05,
    )
    fm = FeatureMonitor(config=config)

    # Phase 1: Soft playing for 2 seconds
    for i in range(8):
        t = i * 0.25
        fm.feed(MusicalEvent(time_seconds=t, strength=0.15))

    snap_soft = fm.snapshot(now=2.0)
    _print_snap("After soft playing", snap_soft)

    # Phase 2: Sudden loud event
    fm.feed(MusicalEvent(time_seconds=2.1, strength=0.95))
    snap_loud = fm.snapshot(now=2.15)
    _print_snap("After sudden loud hit", snap_loud)

    # Phase 3: Sustained loud playing
    for i in range(10):
        t = 2.2 + i * 0.25
        fm.feed(MusicalEvent(time_seconds=t, strength=0.9))
    snap_sustained = fm.snapshot(now=5.0)
    _print_snap("After sustained loud", snap_sustained)

    print("\n  -> Expected: change_score peaks after the sudden hit,")
    print("    then drops as slow EMA catches up.")
    print(f"    change_score trend: {snap_soft.change_score:.3f} -> "
          f"{snap_loud.change_score:.3f} -> {snap_sustained.change_score:.3f}")


def scenario_silence_drop() -> None:
    """Simulate a sudden silence / drop."""
    _header("SCENARIO 4: Silence / Drop")

    config = FeatureMonitorConfig(decay_per_second=0.5)
    fm = FeatureMonitor(config=config)

    # Play actively for a while
    for i in range(12):
        t = i * 0.25
        fm.feed(MusicalEvent(time_seconds=t, strength=0.7))

    snap_active = fm.snapshot(now=3.0, phase_alignment=0.8)
    _print_snap("During active playing", snap_active)

    # Now silence for 3 seconds
    import math
    for gap in [0.5, 1.0, 2.0, 3.0]:
        t = 3.0 + gap
        snap = fm.snapshot(now=t)
        expected_decay = math.exp(-0.5 * gap)
        print(f"\n  [Silence after {gap:.1f}s]  t={t:.2f}s")
        print(f"    strength_ema        : {snap.strength_ema:.3f}  "
              f"(expected decay factor: {expected_decay:.3f})")
        print(f"    silence_duration    : {snap.silence_duration:.3f}s")
        print(f"    player_certainty    : {snap.player_certainty:.3f}")

    print("\n  -> Expected: strength_ema decays over silence.")
    print("    silence_duration increases linearly.")
    print("    certainty drops as strength fades.")


def main() -> int:
    print("FEATURE MONITOR — MODULE 5: MUSICAL FEATURE TRACKING")
    print("Demonstrates continuous musical feature summarisation.")

    scenario_sparse_stable()
    scenario_dense_frantic()
    scenario_sudden_build()
    scenario_silence_drop()

    print(f"\n{'=' * 60}")
    print("Feature Monitor Module 5 ready.")
    print("The feature layer can now feed the Behaviour Engine.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())