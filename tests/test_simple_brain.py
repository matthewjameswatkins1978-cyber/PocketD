"""Tests for Simple Brain v0 — Lock → Choose → Hold → Relisten."""

from __future__ import annotations

import pytest

from drummer.simple_brain import (
    LOCK_SNAPSHOTS,
    LOCK_THRESHOLD,
    MIN_HOLD_CONFIDENCE,
    RELISTEN_SNAPSHOTS,
    RELISTEN_THRESHOLD,
    SWITCH_CONFIDENCE,
    SWITCH_THRESHOLD,
    BrainAction,
    SimpleBrain,
    _score_all,
    analyse_snapshot,
)
from perception.features import FeatureSnapshot


# ---------------------------------------------------------------------------
# Helper — build a FeatureSnapshot quickly
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
    """Create a FeatureSnapshot with only the fields the simple brain uses."""
    return FeatureSnapshot(
        timestamp=timestamp,
        input_density=input_density,
        repetition_stability=repetition_stability,
        player_certainty=player_certainty,
        change_score=change_score,
        silence_duration=silence_duration,
    )


# ---------------------------------------------------------------------------
# 1. Low confidence stays in LISTEN
# ---------------------------------------------------------------------------


def test_low_confidence_stays_in_listen() -> None:
    """When confidence never reaches the lock threshold, stay in LISTEN."""
    brain = SimpleBrain()

    for _ in range(10):
        decision = brain.decide(_snap(player_certainty=0.30))
        assert decision.action == BrainAction.LISTEN
        assert decision.beat_name is None
        assert "listening" in decision.reason
        assert decision.reason  # never empty


# ---------------------------------------------------------------------------
# 2. Confidence reset on dip
# ---------------------------------------------------------------------------


def test_confidence_resets_lock_on_drop() -> None:
    """A dip below LOCK_THRESHOLD resets the confident-snapshot counter."""
    brain = SimpleBrain()

    # Build 3 confident snapshots (just below threshold to lock)
    for i in range(LOCK_SNAPSHOTS - 1):
        decision = brain.decide(
            _snap(player_certainty=LOCK_THRESHOLD, timestamp=float(i))
        )
        assert decision.action == BrainAction.LISTEN
        assert "listening" in decision.reason

    # Dip
    decision = brain.decide(_snap(player_certainty=LOCK_THRESHOLD - 0.01))
    assert decision.action == BrainAction.LISTEN
    assert "reset" in decision.reason or "dipped" in decision.reason

    # Now one confident snapshot again — should be at 1, not at 4
    decision = brain.decide(_snap(player_certainty=LOCK_THRESHOLD))
    assert decision.action == BrainAction.LISTEN
    assert "1/" in decision.reason


# ---------------------------------------------------------------------------
# 3. Lock then choose obvious dense beat
# ---------------------------------------------------------------------------


def test_lock_then_choose_obvious_dense_beat() -> None:
    """After LOCK_SNAPSHOTS confident dense snapshots, choose a dense beat."""
    brain = SimpleBrain()

    # Dense, stable, confident snapshots
    for i in range(LOCK_SNAPSHOTS):
        decision = brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    # The last call (index LOCK_SNAPSHOTS-1) triggers lock and choose
    assert decision.action == BrainAction.CHOOSE
    assert decision.beat_name is not None
    assert decision.beat_name in {"simple_rock", "motorik", "funk_pocket", "punk_drive"}
    assert "choosing" in decision.reason
    assert decision.scores  # non-empty scores dict
    # Ensure dense beats scored higher than sparse ones
    # Check that at least one dense beat outscores half_time
    dense_beats = {"simple_rock", "motorik", "funk_pocket", "punk_drive"}
    max_sparse = max(
        decision.scores.get(b, 0) for b in ["half_time", "shuffle"]
    )
    min_dense = min(
        decision.scores.get(b, 0) for b in dense_beats
    )
    assert min_dense >= max_sparse, (
        f"Expected dense beats to outscore sparse, got dense_min={min_dense:.3f}, sparse_max={max_sparse:.3f}"
    )


# ---------------------------------------------------------------------------
# 4. Sparse input chooses sparse groove, not silence
# ---------------------------------------------------------------------------


def test_sparse_input_chooses_sparse_groove_not_silence() -> None:
    """Sparse but confident input should pick half_time, not silence."""
    brain = SimpleBrain()

    for i in range(LOCK_SNAPSHOTS):
        decision = brain.decide(
            _snap(
                input_density=0.20,
                repetition_stability=0.50,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    assert decision.action == BrainAction.CHOOSE
    assert decision.beat_name is not None
    # Must pick a real sparse groove, not silence
    assert decision.beat_name == "half_time"
    assert decision.beat_name != "silence"
    assert decision.scores["half_time"] > decision.scores["silence"]


# ---------------------------------------------------------------------------
# 5. Silence only wins when confidence collapses
# ---------------------------------------------------------------------------


def test_silence_only_wins_when_confidence_collapses() -> None:
    """Silence outscores sparse grooves only when player_certainty is very low.

    Uses direct scoring to avoid coupling to the lock/choose state machine.
    """
    # -- Normal sparse passage (confident) --
    # sparse density, reasonable stability, good confidence, no silence
    normal = _snap(
        input_density=0.20,
        repetition_stability=0.50,
        player_certainty=0.70,
        silence_duration=0.0,
    )
    scores_normal = _score_all(analyse_snapshot(normal))
    # In a normal sparse passage, silence should lose to real sparse grooves.
    assert scores_normal["half_time"] > scores_normal["silence"]

    # -- Collapsed confidence + long silence --
    collapsed = _snap(
        input_density=0.05,
        repetition_stability=0.0,
        player_certainty=RELISTEN_THRESHOLD - 0.05,
        silence_duration=5.0,
    )
    scores_collapsed = _score_all(analyse_snapshot(collapsed))
    # With confidence below relisten threshold AND meaningful silence,
    # silence should outscore sparse grooves.
    assert scores_collapsed["silence"] > scores_collapsed["half_time"]

    # -- Sparse but still confident (relisten-like edge) --
    edge = _snap(
        input_density=0.15,
        repetition_stability=0.30,
        player_certainty=RELISTEN_THRESHOLD + 0.05,
        silence_duration=0.5,
    )
    scores_edge = _score_all(analyse_snapshot(edge))
    # Confidence just above relisten threshold — silence should NOT win.
    assert scores_edge["silence"] < scores_edge["half_time"]


# ---------------------------------------------------------------------------
# 6. Hold after choose
# ---------------------------------------------------------------------------


def test_hold_after_choose() -> None:
    """After choosing a beat, similar snapshots should produce HOLD."""
    brain = SimpleBrain()

    # Lock and choose
    for i in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    chosen = brain.current_beat
    assert chosen is not None

    # Feed similar snapshots — should hold
    for i in range(5):
        decision = brain.decide(
            _snap(
                input_density=0.78,
                repetition_stability=0.58,
                player_certainty=0.68,
                change_score=0.05,
                timestamp=float(LOCK_SNAPSHOTS + i),
            )
        )
        assert decision.action == BrainAction.HOLD
        assert decision.beat_name == chosen
        assert "holding" in decision.reason


# ---------------------------------------------------------------------------
# 7. No switch on tiny change
# ---------------------------------------------------------------------------


def test_no_switch_on_tiny_change() -> None:
    """A change_score just below threshold should not trigger a switch."""
    brain = SimpleBrain()

    # Lock and choose
    for i in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    chosen = brain.current_beat
    assert chosen is not None

    # Feed a snapshot with change_score just below threshold
    # but density shifted to sparse (dramatic musical change that the
    # simple brain should notice IF change_score were high)
    decision = brain.decide(
        _snap(
            input_density=0.20,
            repetition_stability=0.50,
            player_certainty=0.65,
            change_score=SWITCH_THRESHOLD - 0.01,
            timestamp=10.0,
        )
    )

    # Should still hold because change_score is below threshold
    assert decision.action == BrainAction.HOLD
    assert decision.beat_name == chosen
    assert "change_score" in decision.reason


# ---------------------------------------------------------------------------
# 8. Major change triggers switch
# ---------------------------------------------------------------------------


def test_major_change_triggers_switch() -> None:
    """A large change_score + density drop produces a beat switch.

    The dense beat's score collapses at sparse density while sparse
    beats score high, giving a delta well above SWITCH_CONFIDENCE.
    """
    brain = SimpleBrain()

    # Lock with dense input, choose a dense beat.
    for i in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    old_beat = brain.current_beat
    assert old_beat is not None

    # Major change: sparse, high change_score, still confident.
    decision = brain.decide(
        _snap(
            input_density=0.15,
            repetition_stability=0.55,
            player_certainty=0.70,
            change_score=SWITCH_THRESHOLD + 0.10,
            timestamp=10.0,
        )
    )

    assert decision.beat_name != old_beat, (
        f"Expected a switch away from {old_beat} but got HOLD"
    )
    assert decision.action == BrainAction.CHOOSE
    assert "switching" in decision.reason
    assert decision.beat_name == "half_time"

    # Verify the score delta meets the switch threshold.
    old_score = decision.scores[old_beat]
    new_score = decision.scores[decision.beat_name]
    assert new_score - old_score >= SWITCH_CONFIDENCE, (
        f"Score delta {new_score - old_score:.3f} < {SWITCH_CONFIDENCE}"
    )


# ---------------------------------------------------------------------------
# 9. Low confidence while holding triggers relisten
# ---------------------------------------------------------------------------


def test_low_confidence_while_holding_returns_to_listen() -> None:
    """Confidence below RELISTEN_THRESHOLD for RELISTEN_SNAPSHOTS causes relisten."""
    brain = SimpleBrain()

    # Lock and choose
    for i in range(LOCK_SNAPSHOTS):
        brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    assert brain.current_beat is not None
    assert brain.has_locked

    # Collapse confidence for RELISTEN_SNAPSHOTS
    for i in range(RELISTEN_SNAPSHOTS - 1):
        decision = brain.decide(
            _snap(
                input_density=0.10,
                repetition_stability=0.0,
                player_certainty=RELISTEN_THRESHOLD - 0.05,
                timestamp=10.0 + float(i),
            )
        )
        # May still be holding or may have already relistened

    # This one should trigger relisten
    decision = brain.decide(
        _snap(
            input_density=0.10,
            repetition_stability=0.0,
            player_certainty=RELISTEN_THRESHOLD - 0.05,
            timestamp=12.0,
        )
    )

    assert decision.action == BrainAction.LISTEN
    assert decision.beat_name is None
    assert "relistening" in decision.reason
    assert brain.current_beat is None
    assert not brain.has_locked


# ---------------------------------------------------------------------------
# 10. Reason is never empty
# ---------------------------------------------------------------------------


def test_reason_is_never_empty() -> None:
    """Every BrainDecision must have a non-empty reason string."""
    brain = SimpleBrain()

    scenarios = [
        _snap(player_certainty=0.0),
        _snap(player_certainty=0.55, input_density=0.5, repetition_stability=0.5),
        _snap(player_certainty=0.80, input_density=0.80, repetition_stability=0.70),
    ]

    for snap in scenarios:
        decision = brain.decide(snap)
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0

    # Feed more to get through lock and check hold reasons too
    for _ in range(10):
        decision = brain.decide(
            _snap(player_certainty=0.70, input_density=0.80, repetition_stability=0.60)
        )
        assert isinstance(decision.reason, str)
        assert len(decision.reason) > 0


# ---------------------------------------------------------------------------
# 11. Scores are transparent after choose and hold
# ---------------------------------------------------------------------------


def test_scores_are_transparent_after_choose_and_hold() -> None:
    """After choosing and holding, the scores dict should contain all beat names."""
    brain = SimpleBrain()

    # Lock
    for i in range(LOCK_SNAPSHOTS):
        decision = brain.decide(
            _snap(
                input_density=0.80,
                repetition_stability=0.60,
                player_certainty=0.70,
                timestamp=float(i),
            )
        )

    # Last call chooses — scores should be populated
    assert decision.scores
    assert "simple_rock" in decision.scores
    assert "motorik" in decision.scores
    assert "half_time" in decision.scores
    assert "shuffle" in decision.scores
    assert "funk_pocket" in decision.scores
    assert "punk_drive" in decision.scores
    assert "silence" in decision.scores

    # All scores should be floats in [0, 1]
    for name, score in decision.scores.items():
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"{name} score {score} out of [0, 1]"

    # Hold
    decision = brain.decide(
        _snap(
            input_density=0.80,
            repetition_stability=0.60,
            player_certainty=0.70,
            change_score=0.05,
            timestamp=10.0,
        )
    )
    assert decision.action == BrainAction.HOLD
    assert len(decision.scores) == 7  # 6 grooves + silence


# ---------------------------------------------------------------------------
# 12. Extreme inputs do not crash
# ---------------------------------------------------------------------------


def test_extreme_inputs_do_not_crash() -> None:
    """Edge-case snapshots must produce valid decisions without exceptions."""
    brain = SimpleBrain()

    # All zeros
    decision = brain.decide(_snap())
    assert decision.action == BrainAction.LISTEN
    assert decision.beat_name is None

    # Perfect 1.0 everything
    decision = brain.decide(
        _snap(
            input_density=1.0,
            repetition_stability=1.0,
            player_certainty=1.0,
            change_score=1.0,
            silence_duration=100.0,
        )
    )
    assert decision.action in {BrainAction.LISTEN, BrainAction.CHOOSE, BrainAction.HOLD}

    # Negative values (shouldn't happen, but be robust)
    decision = brain.decide(
        _snap(
            input_density=-0.5,
            repetition_stability=-0.5,
            player_certainty=-0.5,
            change_score=-0.5,
            silence_duration=-10.0,
        )
    )
    assert decision.action in {BrainAction.LISTEN, BrainAction.CHOOSE, BrainAction.HOLD}

    # None-like phase_alignment (handled by FeatureSnapshot defaults)
    snap = FeatureSnapshot(
        timestamp=0.0,
        phase_alignment=None,
        input_density=0.5,
        repetition_stability=0.5,
        player_certainty=0.5,
    )
    decision = brain.decide(snap)
    assert decision.action in {BrainAction.LISTEN, BrainAction.CHOOSE, BrainAction.HOLD}