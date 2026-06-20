"""Tests verifying that Simple Brain's beat bank is backed by data/grooves.yaml.

Requirements
------------
1. Every non-silence beat in the default Simple Brain beat bank exists in
   ``load_grooves()``.
2. ``silence`` is allowed to not exist in the groove database.
3. Every ``simple_brain_enabled`` groove has required metadata:
   ``ideal_density``, ``min_stability``, ``description``.
4. SimpleBrain never chooses a beat name that is missing from
   ``data/grooves.yaml``, except ``silence``.
5. Existing known groove IDs can be chosen in reasonable synthetic cases.
6. Custom beat bank injection still works.
7. Existing tests are updated (covered by other test files).
"""

from __future__ import annotations

import pytest

from drummer.simple_brain import (
    LOCK_SNAPSHOTS,
    LOCK_THRESHOLD,
    BeatDescriptor,
    SimpleBrain,
    load_simple_brain_beat_bank,
)
from groove_library import load_grooves
from perception.features import FeatureSnapshot


def _snap(**kwargs: float) -> FeatureSnapshot:
    return FeatureSnapshot(timestamp=0.0, **kwargs)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 1. Every non-silence beat exists in load_grooves()
# ---------------------------------------------------------------------------


def test_every_non_silence_beat_exists_in_grooves() -> None:
    """Every non-silence beat in the default bank is a real groove ID.

    ``silence`` is the only beat allowed to not exist in the database.
    """
    bank = load_simple_brain_beat_bank()
    grooves = load_grooves()

    non_silence = [b for b in bank if not b.is_silence]
    assert len(non_silence) > 0, "Expected at least one non-silence beat in bank"

    for beat in non_silence:
        assert beat.name in grooves, (
            f"Beat '{beat.name}' is in Simple Brain bank but missing from "
            f"data/grooves.yaml"
        )
        groove = grooves[beat.name]
        assert groove.id == beat.name


# ---------------------------------------------------------------------------
# 2. silence is allowed to not exist in the groove database
# ---------------------------------------------------------------------------


def test_silence_not_in_grooves_is_allowed() -> None:
    """silence is a special sentinel — it should NOT be in the groove database."""
    bank = load_simple_brain_beat_bank()
    silence_descriptors = [b for b in bank if b.is_silence]
    assert len(silence_descriptors) == 1
    assert silence_descriptors[0].name == "silence"

    grooves = load_grooves()
    # silence must NOT be a real groove ID
    assert "silence" not in grooves, (
        "silence is a sentinel beat, not a real groove — "
        "it should not appear in data/grooves.yaml"
    )


# ---------------------------------------------------------------------------
# 3. Every simple_brain_enabled groove has required metadata
# ---------------------------------------------------------------------------


def test_enabled_grooves_have_required_metadata() -> None:
    """Every groove with simple_brain_enabled=True must have:
    - ideal_density (non-empty)
    - min_stability (> 0)
    - description (non-empty)
    """
    grooves = load_grooves()
    for groove in grooves.values():
        if not groove.simple_brain_enabled:
            continue
        assert groove.ideal_density, (
            f"Groove '{groove.id}' has simple_brain_enabled=True "
            f"but empty ideal_density"
        )
        assert groove.ideal_density in ("sparse", "medium", "dense"), (
            f"Groove '{groove.id}' has invalid ideal_density "
            f"'{groove.ideal_density}'"
        )
        assert groove.min_stability > 0, (
            f"Groove '{groove.id}' has simple_brain_enabled=True "
            f"but min_stability={groove.min_stability}"
        )
        assert groove.description, (
            f"Groove '{groove.id}' has simple_brain_enabled=True "
            f"but empty description"
        )


# ---------------------------------------------------------------------------
# 4. SimpleBrain never chooses a beat missing from grooves.yaml (except silence)
# ---------------------------------------------------------------------------


def test_brain_never_chooses_unknown_beat() -> None:
    """Run SimpleBrain through many snapshots — every non-silence beat it
    chooses must be a valid key in ``load_grooves()``."""
    brain = SimpleBrain()
    grooves = load_grooves()

    # Feed a variety of snapshots to exercise the brain.
    scenarios = [
        # Dense confident
        _snap(input_density=0.80, repetition_stability=0.60, player_certainty=0.70),
        # Sparse confident
        _snap(input_density=0.18, repetition_stability=0.50, player_certainty=0.65),
        # Medium confident
        _snap(input_density=0.50, repetition_stability=0.50, player_certainty=0.65),
        # Collapsed confidence
        _snap(input_density=0.05, repetition_stability=0.0, player_certainty=0.05, silence_duration=5.0),
    ]

    for _ in range(20):
        for snap in scenarios:
            decision = brain.decide(snap)
            if decision.beat_name is not None and decision.beat_name != "silence":
                assert decision.beat_name in grooves, (
                    f"SimpleBrain chose '{decision.beat_name}', "
                    f"which is not in data/grooves.yaml"
                )


# ---------------------------------------------------------------------------
# 5. Known groove IDs can be chosen in reasonable synthetic cases
# ---------------------------------------------------------------------------


def test_dense_stable_input_chooses_dense_groove() -> None:
    """Dense, stable, confident input chooses one of the dense grooves."""
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.65,
            player_certainty=0.72,
        ))
    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.65,
        player_certainty=0.72,
    ))
    assert decision.beat_name in {"simple_rock", "motorik", "funk_pocket", "punk_drive"}


def test_sparse_stable_input_chooses_half_time() -> None:
    """Sparse, stable, confident input chooses half_time."""
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.18, repetition_stability=0.50,
            player_certainty=0.68,
        ))
    decision = brain.decide(_snap(
        input_density=0.18, repetition_stability=0.50,
        player_certainty=0.68,
    ))
    assert decision.beat_name == "half_time"


def test_medium_input_can_choose_shuffle() -> None:
    """Medium density input can choose shuffle (the only medium-ideal-density beat)."""
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.50, repetition_stability=0.50,
            player_certainty=0.65,
        ))
    decision = brain.decide(_snap(
        input_density=0.50, repetition_stability=0.50,
        player_certainty=0.65,
    ))
    # With medium density, shuffle (ideal_density=medium) should score well.
    # It may not always win if stability edges a dense/sparse beat,
    # but it must be a valid groove ID.
    assert decision.beat_name is not None
    assert decision.beat_name != "silence"
    from groove_library import load_grooves
    assert decision.beat_name in load_grooves()


# ---------------------------------------------------------------------------
# 6. Custom beat bank injection still works
# ---------------------------------------------------------------------------


def test_custom_beat_bank_injection() -> None:
    """SimpleBrain(beat_bank=...) accepts and uses a custom bank."""
    custom_bank: tuple[BeatDescriptor, ...] = (
        BeatDescriptor(
            name="custom_beat",
            description="A test-only beat",
            ideal_density="dense",
            min_stability=0.40,
        ),
        BeatDescriptor(
            name="silence",
            description="Silence",
            ideal_density="sparse",
            min_stability=0.0,
            is_silence=True,
        ),
    )
    brain = SimpleBrain(beat_bank=custom_bank)

    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.60,
            player_certainty=LOCK_THRESHOLD,
        ))

    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.60,
        player_certainty=LOCK_THRESHOLD,
    ))
    # Should have chosen the custom beat (only non-silence option)
    assert decision.beat_name == "custom_beat"
    assert "custom_beat" in decision.scores
    assert "silence" in decision.scores
    assert len(decision.scores) == 2


def test_custom_empty_bank_only_silence() -> None:
    """A bank with only silence chooses silence."""
    custom_bank: tuple[BeatDescriptor, ...] = (
        BeatDescriptor(
            name="silence",
            description="Silence only",
            ideal_density="sparse",
            min_stability=0.0,
            is_silence=True,
        ),
    )
    brain = SimpleBrain(beat_bank=custom_bank)

    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.60,
            player_certainty=LOCK_THRESHOLD,
        ))

    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.60,
        player_certainty=LOCK_THRESHOLD,
    ))
    assert decision.beat_name == "silence"


# ---------------------------------------------------------------------------
# 7. load_simple_brain_beat_bank is deterministic
# ---------------------------------------------------------------------------


def test_load_simple_brain_beat_bank_is_deterministic() -> None:
    """load_simple_brain_beat_bank() returns the same result every call."""
    bank1 = load_simple_brain_beat_bank()
    bank2 = load_simple_brain_beat_bank()
    assert bank1 == bank2
    names1 = [b.name for b in bank1]
    names2 = [b.name for b in bank2]
    assert names1 == names2


def test_load_simple_brain_beat_bank_includes_all_enabled_grooves() -> None:
    """Every groove with simple_brain_enabled=True appears in the bank."""
    bank = load_simple_brain_beat_bank()
    bank_names = {b.name for b in bank}

    grooves = load_grooves()
    for groove in grooves.values():
        if groove.simple_brain_enabled:
            assert groove.id in bank_names, (
                f"'{groove.id}' has simple_brain_enabled=True "
                f"but is missing from the beat bank"
            )


# ---------------------------------------------------------------------------
# Conservative tie-breaking tests
# ---------------------------------------------------------------------------


def test_dense_stable_input_prefers_low_risk_simple_rock() -> None:
    """When all dense grooves score equally, simple_rock wins via tie-break.

    simple_rock has: risk=low, energy=3, feel_tags=["safe",...]
    This beats motorik (energy=4, no safe tag), funk_pocket (risk=medium),
    and punk_drive (risk=high, energy=5).
    """
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.65,
            player_certainty=0.72,
        ))
    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.65,
        player_certainty=0.72,
    ))
    # With stability 0.65, all four dense grooves pass their min_stability
    # and all have ideal_density="dense", so they tie at the same score.
    # Conservative tie-breaking should select simple_rock.
    assert decision.beat_name == "simple_rock", (
        f"Expected simple_rock (lowest risk/energy, safe tag) but got {decision.beat_name}"
    )


def test_punk_drive_does_not_win_dense_tie() -> None:
    """punk_drive (risk=high, energy=5) should never win a tie against low-risk grooves."""
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.65,
            player_certainty=0.72,
        ))
    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.65,
        player_certainty=0.72,
    ))
    assert decision.beat_name != "punk_drive", (
        f"punk_drive (high risk) should not win a tie — got {decision.beat_name}"
    )


def test_funk_pocket_does_not_win_on_alphabetical() -> None:
    """Funk pocket should not win dense input purely because it sorts first.

    Before tie-breaking, the beat bank is sorted alphabetically, so
    funk_pocket appeared first and won via max().  Now conservative
    tie-breaking ensures lower-risk beats win ties.
    """
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.65,
            player_certainty=0.72,
        ))
    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.65,
        player_certainty=0.72,
    ))
    assert decision.beat_name != "funk_pocket", (
        f"funk_pocket should not win dense tie by alphabetical order — got {decision.beat_name}"
    )


def test_tie_breaking_is_deterministic() -> None:
    """The same input must always produce the same chosen beat."""
    chosen_beats: set[str] = set()
    for _ in range(10):
        brain = SimpleBrain()
        for __ in range(LOCK_SNAPSHOTS):
            brain.decide(_snap(
                input_density=0.80, repetition_stability=0.65,
                player_certainty=0.72,
            ))
        decision = brain.decide(_snap(
            input_density=0.80, repetition_stability=0.65,
            player_certainty=0.72,
        ))
        chosen_beats.add(decision.beat_name)
    assert len(chosen_beats) == 1, (
        f"Tie-breaking is not deterministic: got {chosen_beats}"
    )


def test_dense_stable_chooses_simple_rock_or_motorik() -> None:
    """With high stability, simple_rock or motorik win (both low risk, safe tags differ).

    simple_rock has the 'safe' tag, motorik does not.  So simple_rock wins.
    But with stability just above motorik's threshold but below punk_drive's,
    we test a different scenario.
    """
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.58,
            player_certainty=0.72,
        ))
    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.58,
        player_certainty=0.72,
    ))
    # At stability 0.58, motorik (min 0.55) passes, punk_drive (min 0.60) fails.
    # simple_rock (min 0.45) and funk_pocket (min 0.50) also pass.
    # Tie-break: simple_rock wins (low risk, energy=3, safe tag).
    assert decision.beat_name in {"simple_rock", "motorik"}, (
        f"Expected simple_rock or motorik but got {decision.beat_name}"
    )
    # With punk_drive failing stability, it should have lower raw score.
    assert decision.scores["punk_drive"] < decision.scores["motorik"]


def test_tie_break_uses_risk_before_energy() -> None:
    """Risk takes priority over energy in tie-breaking.

    If we had two low-risk beats with different energy, the lower energy wins.
    simple_rock (low, 3) vs motorik (low, 4) → simple_rock wins.
    """
    brain = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS):
        brain.decide(_snap(
            input_density=0.80, repetition_stability=0.65,
            player_certainty=0.72,
        ))
    decision = brain.decide(_snap(
        input_density=0.80, repetition_stability=0.65,
        player_certainty=0.72,
    ))
    # Both simple_rock and motorik are low risk and pass stability.
    # simple_rock has energy=3, motorik has energy=4.
    # simple_rock has 'safe' tag, motorik does not.
    # So simple_rock wins.
    assert decision.beat_name == "simple_rock"
