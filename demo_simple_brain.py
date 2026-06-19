"""Demo: Simple Brain v0 — diagnostic walk through 5 musical scenarios.

Feeds hand-crafted FeatureSnapshot values to SimpleBrain and prints
each decision with scores.  No MIDI, no audio — pure logic inspection.

Beat names are real groove IDs from ``data/grooves.yaml``.
"""

from __future__ import annotations

from drummer.simple_brain import BrainAction, SimpleBrain
from drummer.simple_brain_grooves import resolve_simple_brain_groove
from groove_library import load_grooves
from perception.features import FeatureSnapshot


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _snap(
    *,
    input_density: float = 0.0,
    repetition_stability: float = 0.0,
    player_certainty: float = 0.0,
    change_score: float = 0.0,
    silence_duration: float = 0.0,
) -> FeatureSnapshot:
    """Build a FeatureSnapshot for the demo."""
    return FeatureSnapshot(
        timestamp=0.0,
        input_density=input_density,
        repetition_stability=repetition_stability,
        player_certainty=player_certainty,
        change_score=change_score,
        silence_duration=silence_duration,
    )


def _print_decision(i: int, decision) -> None:
    """Pretty-print one BrainDecision with groove info."""
    grooves = load_grooves()
    print(f"  [{i:2d}]  {decision.action.value:<8s}  "
          f"beat={str(decision.beat_name):<16s}  "
          f"conf={decision.confidence:.3f}")
    if decision.beat_name and decision.beat_name != "silence":
        groove = resolve_simple_brain_groove(decision.beat_name)
        if groove:
            print(f"       groove: {groove.name} — {groove.description}")
        else:
            print(f"       groove: (unknown)")
    print(f"       reason: {decision.reason}")
    if decision.scores:
        items = sorted(decision.scores.items(), key=lambda kv: -kv[1])
        score_str = ", ".join(f"{name}={score:.3f}" for name, score in items)
        print(f"       scores: {score_str}")
    print()


# ---------------------------------------------------------------------------
# Main demo
# ---------------------------------------------------------------------------


def main() -> None:
    brain = SimpleBrain()

    print("Beat bank loaded from data/grooves.yaml:")
    bank_names = sorted(b.name for b in brain._beat_bank if not b.is_silence)
    print(f"  enabled grooves: {', '.join(bank_names)}")
    print(f"  + silence (special sentinel)")
    print()

    scenario_count = 0

    # ===================================================================
    # Scenario 1: Silent / uncertain start
    # ===================================================================
    scenario_count += 1
    print(f"=" * 64)
    print(f" SCENARIO {scenario_count}: Silent / uncertain start")
    print(f"=" * 64)
    print("  Low-confidence snapshots — brain should stay in LISTEN.\n")

    for i in range(6):
        snap = _snap(
            input_density=0.05,
            repetition_stability=0.1,
            player_certainty=0.15,
            change_score=0.02,
            silence_duration=float(i) * 0.5,
        )
        decision = brain.decide(snap)
        _print_decision(i, decision)

    # ===================================================================
    # Scenario 2: Lock into steady dense groove
    # ===================================================================
    scenario_count += 1
    print(f"=" * 64)
    print(f" SCENARIO {scenario_count}: Lock into steady dense groove")
    print(f"=" * 64)
    print("  Confident, dense, stable snapshots — brain locks, then chooses.\n")

    for i in range(7):
        snap = _snap(
            input_density=0.80,
            repetition_stability=0.65,
            player_certainty=0.72,
            change_score=0.05,
            silence_duration=0.0,
        )
        decision = brain.decide(snap)
        _print_decision(i, decision)

    # ===================================================================
    # Scenario 3: Hold through small variation
    # ===================================================================
    scenario_count += 1
    print(f"=" * 64)
    print(f" SCENARIO {scenario_count}: Hold through small variation")
    print(f"=" * 64)
    print("  Slight density/stability wobble — brain should HOLD.\n")

    wobbles = [
        (0.78, 0.62, 0.05),
        (0.75, 0.58, 0.08),
        (0.82, 0.66, 0.03),
        (0.79, 0.60, 0.06),
    ]
    for i, (density, stability, change) in enumerate(wobbles):
        snap = _snap(
            input_density=density,
            repetition_stability=stability,
            player_certainty=0.68,
            change_score=change,
            silence_duration=0.0,
        )
        decision = brain.decide(snap)
        _print_decision(i, decision)

    # ===================================================================
    # Scenario 4: Major change to sparse section
    # ===================================================================
    scenario_count += 1
    print(f"=" * 64)
    print(f" SCENARIO {scenario_count}: Major change to sparse section")
    print(f"=" * 64)
    print("  High change_score + density drop — brain should switch.\n")

    for i in range(3):
        snap = _snap(
            input_density=0.18,
            repetition_stability=0.50,
            player_certainty=0.68,
            change_score=0.45,
            silence_duration=0.0,
        )
        decision = brain.decide(snap)
        _print_decision(i, decision)

    # ===================================================================
    # Scenario 5: Confidence collapse causing relisten
    # ===================================================================
    scenario_count += 1
    print(f"=" * 64)
    print(f" SCENARIO {scenario_count}: Confidence collapse -> relisten")
    print(f"=" * 64)
    print("  Confidence drops below relisten threshold — brain dumps state.\n")

    for i in range(4):
        snap = _snap(
            input_density=0.05,
            repetition_stability=0.0,
            player_certainty=0.08,
            change_score=0.01,
            silence_duration=3.0 + float(i),
        )
        decision = brain.decide(snap)
        _print_decision(i, decision)

    # ===================================================================
    # Recap: final brain state
    # ===================================================================
    print(f"=" * 64)
    print(" FINAL STATE")
    print(f"=" * 64)
    print(f"  has_locked:                     {brain.has_locked}")
    print(f"  current_beat:                   {brain.current_beat}")
    print(f"  consecutive_confident_snapshots: {brain.consecutive_confident_snapshots}")
    print(f"  consecutive_uncertain_snapshots: {brain.consecutive_uncertain_snapshots}")
    print()
    print("Done.  Use --verbose with pytest to inspect internals:")
    print("  python -m pytest tests/test_simple_brain.py -v")


if __name__ == "__main__":
    main()