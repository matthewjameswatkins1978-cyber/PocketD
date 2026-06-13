"""Demo: Feature-Driven Behaviour Engine.

Shows how FeatureSnapshot values translate into behaviour decisions:
LISTEN -> ENTER -> MAINTAIN, BUILD, REDUCE, ANCHOR, BAIL/DROP.

Run:
    python demo_feature_behaviour.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drummer.behaviour import (
    BehaviourIntent,
    FeatureDrivenBehaviourEngine,
    DrummerProfile,
)
from perception.features import FeatureSnapshot


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _snap(**kwargs) -> FeatureSnapshot:
    """Shorthand for building a FeatureSnapshot."""
    defaults = {
        "timestamp": 0.0,
        "input_density": 0.0,
        "strength_ema": 0.0,
        "fast_strength_ema": 0.0,
        "slow_strength_ema": 0.0,
        "change_score": 0.0,
        "silence_duration": 0.0,
        "repetition_stability": 0.0,
        "phase_alignment": None,
        "player_certainty": 0.0,
    }
    defaults.update(kwargs)
    return FeatureSnapshot(**defaults)


def print_snap(label: str, snap: FeatureSnapshot) -> None:
    print(f"\n  [{label}]  certainty={snap.player_certainty:.3f}  "
          f"stability={snap.repetition_stability:.3f}  "
          f"density={snap.input_density:.3f}  "
          f"change={snap.change_score:.3f}  "
          f"silence={snap.silence_duration:.1f}s")


def scenario_enter_and_maintain() -> None:
    """Stable sparse playing -> ENTER -> MAINTAIN."""
    _header("SCENARIO 1: Stable Sparse Playing -> ENTER -> MAINTAIN")

    eng = FeatureDrivenBehaviourEngine()
    t = 0.0
    for i in range(6):
        snap = _snap(
            timestamp=t,
            repetition_stability=0.80,
            player_certainty=0.70,
            phase_alignment=0.70,
            input_density=0.25,
            silence_duration=0.2,
        )
        d = eng.evaluate(snap)
        print_snap(f"t={t:.1f}s", snap)
        print(f"    -> {d.intent.value:12s} ({d.reason})")
        t += 0.5


def scenario_sudden_build() -> None:
    """Sudden strength increase -> BUILD."""
    _header("SCENARIO 2: Sudden Build")

    eng = FeatureDrivenBehaviourEngine()
    # ENTER
    for i in range(3):
        eng.evaluate(_snap(
            timestamp=float(i) * 0.5,
            repetition_stability=0.80,
            player_certainty=0.70,
            phase_alignment=0.70,
        ))

    # MAINTAIN baseline
    snap = _snap(timestamp=2.0, repetition_stability=0.80,
                 player_certainty=0.70, input_density=0.3, change_score=0.05)
    d = eng.evaluate(snap)
    print_snap("baseline", snap)
    print(f"    -> {d.intent.value:12s} ({d.reason})")

    # BUILD trigger
    snap2 = _snap(timestamp=2.5, repetition_stability=0.75,
                  player_certainty=0.65, input_density=0.3,
                  change_score=0.30, phase_alignment=0.65)
    d2 = eng.evaluate(snap2)
    print_snap("build!", snap2)
    print(f"    -> {d2.intent.value:12s} ({d2.reason})")


def scenario_density_inversion() -> None:
    """Frantic dense playing -> REDUCE."""
    _header("SCENARIO 3: Frantic Dense Playing -> REDUCE")

    eng = FeatureDrivenBehaviourEngine()
    for i in range(3):
        eng.evaluate(_snap(
            timestamp=float(i) * 0.5,
            repetition_stability=0.80,
            player_certainty=0.70,
            phase_alignment=0.70,
        ))

    # MAINTAIN
    d0 = eng.evaluate(_snap(timestamp=2.0, repetition_stability=0.75,
                             player_certainty=0.70, input_density=0.3))
    print(f"\n  [baseline]  -> {d0.intent.value}")

    # Density spike
    snap = _snap(timestamp=2.5, input_density=0.90, player_certainty=0.70,
                 repetition_stability=0.75, phase_alignment=0.65)
    d = eng.evaluate(snap)
    print_snap("dense", snap)
    print(f"    -> {d.intent.value:12s} ({d.reason})")


def scenario_uncertainty_anchor() -> None:
    """Weak erratic playing -> ANCHOR."""
    _header("SCENARIO 4: Weak Erratic Playing -> ANCHOR")

    eng = FeatureDrivenBehaviourEngine()
    for i in range(3):
        eng.evaluate(_snap(
            timestamp=float(i) * 0.5,
            repetition_stability=0.80,
            player_certainty=0.70,
            phase_alignment=0.70,
        ))

    # Uncertainty hits
    snap = _snap(timestamp=2.0, player_certainty=0.25,
                 repetition_stability=0.30, phase_alignment=0.30)
    d = eng.evaluate(snap)
    print_snap("uncertain", snap)
    print(f"    -> {d.intent.value:12s} ({d.reason})")


def scenario_silence_bail() -> None:
    """Long silence -> BAIL."""
    _header("SCENARIO 5: Long Silence -> BAIL")

    eng = FeatureDrivenBehaviourEngine()
    for i in range(3):
        eng.evaluate(_snap(
            timestamp=float(i) * 0.5,
            repetition_stability=0.80,
            player_certainty=0.70,
            phase_alignment=0.70,
        ))

    # Silent snapshot
    snap = _snap(timestamp=10.0, silence_duration=3.0,
                 player_certainty=0.0, repetition_stability=0.0)
    d = eng.evaluate(snap)
    print_snap("silence", snap)
    print(f"    -> {d.intent.value:12s} ({d.reason})")


def main() -> int:
    print("FEATURE-DRIVEN BEHAVIOUR ENGINE")
    print("Connecting FeatureSnapshot -> BehaviourIntent")
    print("Priority: BAIL > ENTER > ANCHOR > BUILD > REDUCE > MAINTAIN")

    scenario_enter_and_maintain()
    scenario_sudden_build()
    scenario_density_inversion()
    scenario_uncertainty_anchor()
    scenario_silence_bail()

    print(f"\n{'=' * 60}")
    print("Feature-Driven Behaviour Engine ready.")
    print("Decisions now respond to smoothed FeatureSnapshot evidence.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())