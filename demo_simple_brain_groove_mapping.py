"""Demo: Simple Brain beat-name → groove-id mapping.

Since Simple Brain v1, beat names are real groove IDs from
``data/grooves.yaml``, so the mapping is identity.  This demo
shows the direct relationship between chosen beat names and
loaded Groove objects.

Runs SimpleBrain over synthetic snapshots and prints:
- chosen beat_name (which is the groove ID)
- the loaded groove display name and description
- silence shows as "no groove"
- no MIDI playback
"""

from __future__ import annotations

from drummer.simple_brain import LOCK_SNAPSHOTS, SimpleBrain
from drummer.simple_brain_grooves import (
    resolve_simple_brain_groove,
    simple_brain_beat_to_groove_id,
)
from perception.features import FeatureSnapshot


def _snap(**kwargs: float) -> FeatureSnapshot:
    return FeatureSnapshot(timestamp=0.0, **kwargs)


def _print_mapping(decision) -> None:
    beat = decision.beat_name
    groove_id = simple_brain_beat_to_groove_id(beat)
    groove_obj = resolve_simple_brain_groove(beat)
    print(f"  action:     {decision.action.value}")
    print(f"  beat_name:  {beat}")
    print(f"  groove_id:  {groove_id} (beat_name IS the groove ID)")
    if groove_obj:
        print(f"  groove_obj: {groove_obj.name}  —  {groove_obj.description}")
    else:
        print(f"  groove_obj: None (silence/unknown)")
    print(f"  reason:     {decision.reason}")
    print()


def main() -> None:
    brain = SimpleBrain()

    # -- Dense confident -> one of the dense grooves --
    print("=" * 60)
    print("  DENSE CONFIDENT INPUT")
    print("=" * 60)
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(input_density=0.80, repetition_stability=0.65, player_certainty=0.72)
        )
    decision = brain.decide(
        _snap(input_density=0.80, repetition_stability=0.65, player_certainty=0.72)
    )
    _print_mapping(decision)

    # -- Sparse confident -> half_time --
    brain2 = SimpleBrain()
    print("=" * 60)
    print("  SPARSE CONFIDENT INPUT")
    print("=" * 60)
    for _ in range(LOCK_SNAPSHOTS):
        brain2.decide(
            _snap(
                input_density=0.15, repetition_stability=0.48, player_certainty=0.68
            )
        )
    decision2 = brain2.decide(
        _snap(input_density=0.15, repetition_stability=0.48, player_certainty=0.68)
    )
    _print_mapping(decision2)

    # -- Medium density -> shuffle --
    brain_med = SimpleBrain()
    print("=" * 60)
    print("  MEDIUM DENSITY INPUT")
    print("=" * 60)
    for _ in range(LOCK_SNAPSHOTS):
        brain_med.decide(
            _snap(
                input_density=0.50, repetition_stability=0.50, player_certainty=0.65
            )
        )
    decision_med = brain_med.decide(
        _snap(input_density=0.50, repetition_stability=0.50, player_certainty=0.65)
    )
    _print_mapping(decision_med)

    # -- Collapsed confidence -> silence (after hold) --
    brain3 = SimpleBrain()
    print("=" * 60)
    print("  COLLAPSED CONFIDENCE -> SILENCE")
    print("=" * 60)
    for _ in range(LOCK_SNAPSHOTS):
        brain3.decide(
            _snap(
                input_density=0.80, repetition_stability=0.65, player_certainty=0.72
            )
        )
    # Hold briefly.
    brain3.decide(
        _snap(input_density=0.80, repetition_stability=0.65, player_certainty=0.70)
    )
    # Collapse confidence.
    decisions = []
    for i in range(4):
        decisions.append(
            brain3.decide(
                _snap(
                    input_density=0.03,
                    repetition_stability=0.0,
                    player_certainty=0.05,
                    silence_duration=3.0 + i * 0.5,
                )
            )
        )
    _print_mapping(decisions[-1])

    # -- LISTEN (no beat chosen yet) --
    brain4 = SimpleBrain()
    print("=" * 60)
    print("  LISTEN (no beat chosen)")
    print("=" * 60)
    decision4 = brain4.decide(_snap(player_certainty=0.10))
    _print_mapping(decision4)

    print("\nAll Simple Brain beat names and their mappings:")
    from drummer.simple_brain import load_simple_brain_beat_bank

    bank = load_simple_brain_beat_bank()
    for beat in bank:
        groove_id = simple_brain_beat_to_groove_id(beat.name)
        groove_obj = resolve_simple_brain_groove(beat.name)
        if groove_obj:
            info = f"{groove_obj.name} — {groove_obj.description}"
        else:
            info = "no groove (silence)"
        print(f"  {beat.name:<18s} -> {str(groove_id):<14s}  ({info})")


if __name__ == "__main__":
    main()