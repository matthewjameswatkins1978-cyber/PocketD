"""Batch musical-situation tests for Simple Brain v0.

Exercises ``SimpleBrain.decide()`` over longer synthetic sequences
to catch bad musical behaviour before live playtesting.

Scenarios
---------
1.  Silent intro -> steady dense groove
2.  Steady dense groove with small wobble
3.  Dense verse -> sparse breakdown
4.  Sparse breakdown -> dense recovery
5.  Confidence collapse during hold
6.  Noisy uncertain player never locks
7.  High confidence but unstable playing
8.  Boundary thresholds

Helpers at module level keep individual tests short and declarative.
"""

from __future__ import annotations

import pytest

from drummer.simple_brain import (
    LOCK_SNAPSHOTS,
    LOCK_THRESHOLD,
    RELISTEN_SNAPSHOTS,
    RELISTEN_THRESHOLD,
    SWITCH_CONFIDENCE,
    SWITCH_THRESHOLD,
    BrainAction,
    BrainDecision,
    SimpleBrain,
)
from perception.features import FeatureSnapshot


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _snap(
    *,
    input_density: float = 0.0,
    repetition_stability: float = 0.0,
    player_certainty: float = 0.0,
    change_score: float = 0.0,
    silence_duration: float = 0.0,
    timestamp: float = 0.0,
) -> FeatureSnapshot:
    """Build a FeatureSnapshot with only the fields SimpleBrain uses."""
    return FeatureSnapshot(
        timestamp=timestamp,
        input_density=input_density,
        repetition_stability=repetition_stability,
        player_certainty=player_certainty,
        change_score=change_score,
        silence_duration=silence_duration,
    )


def run_sequence(
    brain: SimpleBrain | None = None,
    snapshots: list[FeatureSnapshot] | None = None,
) -> list[BrainDecision]:
    """Feed a list of snapshots through a brain and return all decisions.

    If no brain is given a fresh ``SimpleBrain()`` is created.
    """
    if brain is None:
        brain = SimpleBrain()
    if snapshots is None:
        snapshots = []
    return [brain.decide(snap) for snap in snapshots]


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_eventually_chooses(
    decisions: list[BrainDecision],
    allowed_beats: set[str],
    *,
    msg: str = "",
) -> int:
    """Assert that at least one decision is CHOOSE with a beat in *allowed_beats*.

    Returns the index of the first CHOOSE.
    """
    prefix = f"{msg}: " if msg else ""
    for i, d in enumerate(decisions):
        if d.action == BrainAction.CHOOSE and d.beat_name in allowed_beats:
            return i
    names = {d.beat_name for d in decisions if d.action == BrainAction.CHOOSE}
    raise AssertionError(
        f"{prefix}no CHOOSE with beat in {allowed_beats}; "
        f"chose {names or 'nothing'}"
    )


def assert_holds_mostly(
    decisions: list[BrainDecision],
    expected_beat: str,
    min_hold_ratio: float = 0.75,
) -> None:
    """After the first CHOOSE, verify HOLDs dominate and hold the right beat."""
    post_choose = [
        d for d in decisions if d.action in (BrainAction.HOLD, BrainAction.CHOOSE)
    ]
    if not post_choose:
        return
    holds = [d for d in post_choose if d.action == BrainAction.HOLD]
    ratio = len(holds) / len(post_choose)
    assert ratio >= min_hold_ratio, (
        f"Expected >= {min_hold_ratio:.0%} HOLD after choose, "
        f"got {ratio:.0%} ({len(holds)}/{len(post_choose)})"
    )
    wrong = [d for d in holds if d.beat_name != expected_beat]
    assert not wrong, f"{len(wrong)} HOLDs had beat {wrong[0].beat_name!r}, expected {expected_beat!r}"


def count_switches(decisions: list[BrainDecision]) -> int:
    """Count CHOOSE decisions after the initial choice.

    The first CHOOSE is the initial selection; each subsequent CHOOSE
    is a switch.
    """
    chooses = [d for d in decisions if d.action == BrainAction.CHOOSE]
    return max(0, len(chooses) - 1)


# ---------------------------------------------------------------------------
# Batch scenario builders
# ---------------------------------------------------------------------------

# All groove IDs from data/grooves.yaml with simple_brain_enabled=true.
# dense: simple_rock, motorik, funk_pocket, punk_drive
# spare: half_time
# medium: shuffle

DENSE_BEATS = {"simple_rock", "motorik", "funk_pocket", "punk_drive"}
SPARSE_BEATS = {"half_time"}


def _lock_dense(brain: SimpleBrain) -> None:
    """Feed LOCK_SNAPSHOTS confident dense snapshots so the brain locks."""
    for i in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )


def _lock_sparse(brain: SimpleBrain) -> None:
    """Feed LOCK_SNAPSHOTS confident sparse snapshots so the brain locks."""
    for i in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(
                input_density=0.18,
                repetition_stability=0.50,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )


def _dense_snap(
    wobble: float = 0.0,
) -> FeatureSnapshot:
    """A confident dense snapshot with optional small wobble."""
    return _snap(
        input_density=0.78 + wobble * 0.04,
        repetition_stability=0.60 + wobble * 0.04,
        player_certainty=0.68 + wobble * 0.04,
        change_score=abs(wobble) * 0.02,
    )


def _sparse_snap(
    change_score: float = 0.05,
) -> FeatureSnapshot:
    """A confident sparse snapshot."""
    return _snap(
        input_density=0.18,
        repetition_stability=0.50,
        player_certainty=0.65,
        change_score=change_score,
    )


# ---------------------------------------------------------------------------
# 1. Silent intro -> steady dense groove
# ---------------------------------------------------------------------------


def test_silent_intro_to_steady_dense_groove() -> None:
    """8 uncertain snapshots then 12 confident dense ones.

    Expected: LISTEN throughout intro, eventually CHOOSE a dense beat,
    then mostly HOLD with <= 1 switch total.
    """
    brain = SimpleBrain()

    # -- Silent intro --
    intro = [
        _snap(
            input_density=0.03,
            repetition_stability=0.05,
            player_certainty=0.10,
            silence_duration=float(i) * 0.5,
            timestamp=float(i),
        )
        for i in range(8)
    ]
    decisions_intro = run_sequence(brain, intro)

    assert all(d.action == BrainAction.LISTEN for d in decisions_intro)
    assert all(d.beat_name is None for d in decisions_intro)
    assert all(d.reason for d in decisions_intro)

    # -- Confident dense section --
    dense = [_dense_snap() for _ in range(12)]
    decisions_dense = run_sequence(brain, dense)
    decisions = decisions_intro + decisions_dense

    assert_eventually_chooses(decisions, DENSE_BEATS)

    chose_idx = next(
        i for i, d in enumerate(decisions) if d.action == BrainAction.CHOOSE
    )
    chosen = decisions[chose_idx].beat_name
    assert chosen is not None

    # After lock and choose, should mostly HOLD.
    post_choose = decisions[chose_idx + 1 :]
    assert_holds_mostly(post_choose, chosen, min_hold_ratio=0.7)

    # At most 1 switch (the initial choice + maybe one adjustment).
    assert count_switches(decisions) <= 1

    # Every decision has a reason.
    assert all(isinstance(d.reason, str) and len(d.reason) > 0 for d in decisions)

    # Confidence in [0, 1].
    assert all(0.0 <= d.confidence <= 1.0 for d in decisions)


# ---------------------------------------------------------------------------
# 2. Steady dense groove with small wobble
# ---------------------------------------------------------------------------


def test_steady_dense_groove_small_wobble() -> None:
    """Lock, then 32 snapshots with tiny wobble, change_score always < threshold.

    Expected: zero switches after initial choose.
    """
    brain = SimpleBrain()
    _lock_dense(brain)  # locks and chooses

    assert brain.current_beat in DENSE_BEATS
    chosen = brain.current_beat

    # 32 snapshots with small parameter wobble.
    wobbles: list[FeatureSnapshot] = []
    for i in range(32):
        offset = (i % 7) - 3  # -3..+3
        wobbles.append(
            _snap(
                input_density=0.78 + offset * 0.01,
                repetition_stability=0.60 + offset * 0.01,
                player_certainty=0.68,
                change_score=abs(offset) * 0.02,
                timestamp=float(LOCK_SNAPSHOTS + i),
            )
        )
    decisions = run_sequence(brain, wobbles)

    assert all(d.action == BrainAction.HOLD for d in decisions)
    assert all(d.beat_name == chosen for d in decisions)
    assert count_switches(decisions) == 0

    # Hold reasons should mention the small change_score.
    assert all("change_score" in d.reason for d in decisions)


# ---------------------------------------------------------------------------
# 3. Dense verse -> sparse breakdown
# ---------------------------------------------------------------------------


def test_dense_verse_to_sparse_breakdown() -> None:
    """Lock dense, hold briefly, then sudden sparse with high change_score.

    Expected: switches to half_time, holds after switch,
    does NOT choose silence during confident sparse playing.
    """
    brain = SimpleBrain()

    # Dense lock + hold.
    _lock_dense(brain)
    hold1 = [_dense_snap(wobble=0.0) for _ in range(3)]
    run_sequence(brain, hold1)
    assert brain.current_beat in DENSE_BEATS

    # Sparse transition with high change_score.
    transition = [
        _snap(
            input_density=0.20,
            repetition_stability=0.45,
            player_certainty=0.68,
            change_score=SWITCH_THRESHOLD + 0.10,
            timestamp=100.0 + float(i),
        )
        for i in range(2)
    ]
    decisions_trans = run_sequence(brain, transition)

    # Verify a switch happened.
    assert_eventually_chooses(decisions_trans, SPARSE_BEATS, msg="sparse breakdown")

    # The brain should now be holding a sparse beat.
    assert brain.current_beat in SPARSE_BEATS

    # Subsequent sparse snapshots should HOLD, not choose silence.
    post = [
        _snap(
            input_density=0.18,
            repetition_stability=0.48,
            player_certainty=0.65,
            change_score=0.05,
            timestamp=200.0 + float(i),
        )
        for i in range(6)
    ]
    decisions_post = run_sequence(brain, post)
    assert all(d.action == BrainAction.HOLD for d in decisions_post)
    assert all(d.beat_name in SPARSE_BEATS for d in decisions_post)
    assert all("silence" not in d.beat_name for d in decisions_post if d.beat_name)


# ---------------------------------------------------------------------------
# 4. Sparse breakdown -> dense recovery
# ---------------------------------------------------------------------------


def test_sparse_breakdown_to_dense_recovery() -> None:
    """Lock sparse, then sudden dense section with high change_score.

    Expected: starts with half_time, switches to a dense beat,
    then holds without flip-flopping.
    """
    brain = SimpleBrain()

    # Sparse lock + brief hold.
    _lock_sparse(brain)
    run_sequence(brain, [_sparse_snap(change_score=0.05) for _ in range(3)])
    assert brain.current_beat in SPARSE_BEATS

    # Dense recovery.
    recovery = [
        _snap(
            input_density=0.80,
            repetition_stability=0.60,
            player_certainty=0.72,
            change_score=SWITCH_THRESHOLD + 0.12,
            timestamp=100.0 + float(i),
        )
        for i in range(2)
    ]
    decisions_recovery = run_sequence(brain, recovery)
    assert_eventually_chooses(
        decisions_recovery, DENSE_BEATS, msg="dense recovery"
    )

    # Hold the dense beat — no flip-flopping.
    assert brain.current_beat in DENSE_BEATS
    post = [_dense_snap(wobble=0.0) for _ in range(6)]
    decisions_post = run_sequence(brain, post)
    assert all(d.action == BrainAction.HOLD for d in decisions_post)
    assert all(
        d.beat_name in DENSE_BEATS for d in decisions_post
    )


# ---------------------------------------------------------------------------
# 5. Confidence collapse during hold
# ---------------------------------------------------------------------------


def test_confidence_collapse_during_hold() -> None:
    """Lock, hold, then confidence drops below RELISTEN_THRESHOLD.

    Expected: returns to LISTEN, beat=None, has_locked=False,
    can re-lock later.
    """
    brain = SimpleBrain()
    _lock_dense(brain)
    assert brain.has_locked
    assert brain.current_beat is not None

    # Collapse confidence.
    collapsed = [
        _snap(
            input_density=0.05,
            repetition_stability=0.0,
            player_certainty=RELISTEN_THRESHOLD - 0.05,
            silence_duration=3.0 + float(i),
            timestamp=50.0 + float(i),
        )
        for i in range(RELISTEN_SNAPSHOTS)
    ]
    decisions = run_sequence(brain, collapsed)

    # Last decision should be LISTEN.
    assert decisions[-1].action == BrainAction.LISTEN
    assert decisions[-1].beat_name is None
    assert "relistening" in decisions[-1].reason

    # Internal state cleared.
    assert brain.current_beat is None
    assert not brain.has_locked

    # Later confident snapshots can re-lock and choose.
    recovery = [_dense_snap() for _ in range(LOCK_SNAPSHOTS)]
    decisions_recovery = run_sequence(brain, recovery)
    assert decisions_recovery[-1].action == BrainAction.CHOOSE
    assert decisions_recovery[-1].beat_name is not None


# ---------------------------------------------------------------------------
# 6. Noisy uncertain player never locks
# ---------------------------------------------------------------------------


def test_noisy_uncertain_player_never_locks() -> None:
    """40 snapshots with wildly changing density, certainty always < threshold.

    Expected: all decisions LISTEN, no beat chosen, scores empty.
    """
    brain = SimpleBrain()
    snaps: list[FeatureSnapshot] = []
    for i in range(40):
        density = (i % 3) * 0.3 + 0.1  # cycles 0.1, 0.4, 0.7
        snaps.append(
            _snap(
                input_density=density,
                repetition_stability=0.1 + (i % 4) * 0.05,
                player_certainty=LOCK_THRESHOLD - 0.05,
                change_score=(i % 5) * 0.1,
                timestamp=float(i),
            )
        )
    decisions = run_sequence(brain, snaps)

    assert all(d.action == BrainAction.LISTEN for d in decisions)
    assert all(d.beat_name is None for d in decisions)
    assert all(not d.scores for d in decisions)
    assert all(isinstance(d.reason, str) and len(d.reason) > 0 for d in decisions)

    # Lock counters should be cycling (never reaching LOCK_SNAPSHOTS).
    assert brain.consecutive_confident_snapshots < LOCK_SNAPSHOTS
    assert not brain.has_locked


# ---------------------------------------------------------------------------
# 7. High confidence but unstable playing
# ---------------------------------------------------------------------------


def test_high_confidence_unstable_playing() -> None:
    """player_certainty >= 0.50 but repetition_stability is low (0.20).

    Expected: brain locks, chooses a beat, but should prefer ones with
    lower min_stability requirements where the stability_ok term
    differentiates.
    """
    brain = SimpleBrain()

    snaps = [
        _snap(
            input_density=0.55,  # medium
            repetition_stability=0.20,
            player_certainty=0.65,
            timestamp=float(i),
        )
        for i in range(LOCK_SNAPSHOTS)
    ]
    decisions = run_sequence(brain, snaps)

    decision = decisions[-1]
    assert decision.action == BrainAction.CHOOSE
    assert decision.beat_name is not None
    # punk_drive requires min_stability=0.60 — unlikely to win with stability 0.20
    assert decision.beat_name != "punk_drive", (
        "punk_drive requires stability 0.60, unlikely to win with stability 0.20"
    )
    # motorik requires min_stability=0.55 — also unlikely
    assert decision.beat_name != "motorik", (
        "motorik requires stability 0.55, unlikely to win with stability 0.20"
    )

    # Check scoring: beats with higher min_stability should score lower.
    assert decision.scores.get("half_time", 0) >= decision.scores.get("punk_drive", 0)
    assert decision.scores.get("shuffle", 0) >= decision.scores.get("motorik", 0)

    # Reason and scores are populated.
    assert "choosing" in decision.reason
    assert len(decision.scores) > 0


# ---------------------------------------------------------------------------
# 8. Boundary thresholds
# ---------------------------------------------------------------------------
# Semantics verified:
#   LOCK_THRESHOLD:    >= counts, < does not
#   RELISTEN_THRESHOLD: <  counts as collapsed, >= does not
#   SWITCH_THRESHOLD:   >= is considered, < is not


def test_boundary_thresholds_lock_and_relisten() -> None:
    """Confidence exactly LOCK_THRESHOLD counts; just below resets.

    Confidence exactly RELISTEN_THRESHOLD does NOT trigger collapse.
    """
    # -- Lock boundary --
    brain = SimpleBrain()

    # At exactly LOCK_THRESHOLD: should count.
    for _ in range(LOCK_SNAPSHOTS):
        d = brain.decide(_snap(player_certainty=LOCK_THRESHOLD))
    assert brain.has_locked

    # Reset: below LOCK_THRESHOLD resets.
    brain2 = SimpleBrain()
    for _ in range(LOCK_SNAPSHOTS - 1):
        brain2.decide(_snap(player_certainty=LOCK_THRESHOLD))
    brain2.decide(_snap(player_certainty=LOCK_THRESHOLD - 0.01))
    assert brain2.consecutive_confident_snapshots == 0

    # -- Relisten boundary --
    brain3 = SimpleBrain()
    _lock_dense(brain3)
    # Confidence exactly RELISTEN_THRESHOLD should NOT be uncertain.
    for _ in range(RELISTEN_SNAPSHOTS):
        d = brain3.decide(
            _snap(
                input_density=0.10,
                player_certainty=RELISTEN_THRESHOLD,
                timestamp=100.0,
            )
        )
    # Should still be holding (or at least not relistening).
    assert brain3.has_locked
    assert brain3.current_beat is not None

    # Just below RELISTEN_THRESHOLD triggers relisten.
    brain4 = SimpleBrain()
    _lock_dense(brain4)
    for _ in range(RELISTEN_SNAPSHOTS):
        d = brain4.decide(
            _snap(
                input_density=0.10,
                player_certainty=RELISTEN_THRESHOLD - 0.01,
                timestamp=100.0,
            )
        )
    assert not brain4.has_locked
    assert brain4.current_beat is None


def test_boundary_switch_threshold() -> None:
    """change_score exactly SWITCH_THRESHOLD allows switching.

    The density drop from dense (0.80) to sparse (0.18) gives sparse
    beats a large score advantage, so the switch fires at exactly the
    boundary — proving the inclusive ``>=`` semantics work.
    """
    brain = SimpleBrain()
    _lock_dense(brain)  # picks a dense beat

    old = brain.current_beat

    decision = brain.decide(
        _snap(
            input_density=0.18,
            repetition_stability=0.55,
            player_certainty=0.70,
            change_score=SWITCH_THRESHOLD,
            timestamp=100.0,
        )
    )

    assert decision.action == BrainAction.CHOOSE
    assert decision.beat_name != old
    assert "switching" in decision.reason
    assert decision.beat_name in SPARSE_BEATS