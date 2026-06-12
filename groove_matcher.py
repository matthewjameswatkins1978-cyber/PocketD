"""Groove selection heuristic for the live drummer prototype."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from models import Groove
from groove_library import load_grooves

from drummer.rules import (
    DrummerModel,
    GrooveCandidateScore,
    ModelGrooveDecision,
)

_DENSITY_SCORE = {"low": 0.2, "medium": 0.5, "high": 0.8}


def select_groove(fingerprint: dict, confidence: float = 0.5, personality: str = "Anchor"):
    """Choose the most appropriate groove from the local library.

    The heuristic uses confidence, density, and syncopation to prefer stable
    grooves when the musician is uncertain and more energetic grooves when the
    player is already locked in.
    """
    grooves = load_grooves()
    density = float(fingerprint.get("density", 0.5))
    syncopation = float(fingerprint.get("syncopation", 0.0))
    strong_beats = set(fingerprint.get("strong_beats", []))

    personality = personality.lower()
    if "instigator" in personality:
        preferred = {"punk_drive", "motorik"}
        bias = 0.15
    elif "pocket" in personality:
        preferred = {"funk_pocket", "simple_rock"}
        bias = 0.05
    else:
        preferred = {"simple_rock", "motorik"}
        bias = 0.0

    candidates: list[tuple[float, object]] = []
    for groove in grooves.values():
        density_match = 1.0 - abs(density - _DENSITY_SCORE.get(groove.density, 0.5))
        energy_match = 1.0 - abs(confidence - groove.energy / 5.0)
        sync_match = 0.0
        if groove.id in {"shuffle", "funk_pocket"} and syncopation > 0.25:
            sync_match = 0.1
        if groove.id == "half_time" and syncopation < 0.15:
            sync_match = 0.1

        strong_beat_bonus = 0.05 if strong_beats and groove.id in {"simple_rock", "motorik"} else 0.0
        personality_bonus = 0.1 if groove.id in preferred else 0.0
        total = density_match * 0.35 + energy_match * 0.35 + sync_match * 0.15 + strong_beat_bonus + personality_bonus + bias

        candidates.append((total, groove))

    return max(candidates, key=lambda item: item[0])[1]  # type: ignore[return-value]


# ---- Tempo-aware groove selection for synthetic sections -----------------


_CONFIDENCE_LOW_THRESHOLD = 0.4
_CONFIDENCE_HIGH_THRESHOLD = 0.7


def select_groove_by_tempo(
    bpm: float,
    confidence: float,
    previous_groove_id: str | None = None,
    personality: str = "Anchor",
) -> Groove:
    """Choose a groove based on estimated tempo and timing confidence.

    Simple rule table (extension of the existing heuristic):

    * confidence < ``_CONFIDENCE_LOW_THRESHOLD`` → keep previous if set,
      else fall back to ``simple_rock`` (safe basic groove).
    * tempo 105–130 BPM with high confidence → ``motorik`` (driving steady).
    * tempo  80–104 BPM with high confidence → ``half_time`` (sparser feel).
    * everything else → ``simple_rock``.

    The function is deterministic for a given (bpm, confidence, previous)
    tuple so that section processing is repeatable.
    """
    grooves = load_grooves()

    # Low confidence → stay safe, don't make wild changes
    if confidence < _CONFIDENCE_LOW_THRESHOLD:
        if previous_groove_id is not None and previous_groove_id in grooves:
            return grooves[previous_groove_id]
        return grooves["simple_rock"]

    # High confidence: map tempo to a suitable groove
    if confidence >= _CONFIDENCE_HIGH_THRESHOLD and 105.0 <= bpm <= 130.0:
        return grooves["motorik"]

    if confidence >= _CONFIDENCE_HIGH_THRESHOLD and 80.0 <= bpm <= 104.0:
        return grooves["half_time"]

    # Moderate confidence or unusual tempo → safe middle ground
    return grooves["simple_rock"]


# ---- Reason-coded groove decision ---------------------------------------


@dataclass(frozen=True)
class GrooveDecision:
    """A groove decision result with reason code and context.

    Attributes
    ----------
    selected_groove_id : str
        The id of the groove that was selected.
    reason : str
        One of the reason codes explaining *why* this groove was chosen.
    tempo : float
        The estimated BPM that was used for the decision.
    confidence : float
        The timing confidence that was used for the decision.
    previous_groove_id : str | None
        The previous groove id, if any was provided.
    changed : bool | None
        ``True`` if the selected groove differs from the previous one,
        ``False`` if it is the same, ``None`` if there was no previous groove.
    """

    selected_groove_id: str
    reason: str
    tempo: float
    confidence: float
    previous_groove_id: str | None
    changed: bool | None


def select_groove_decision(
    bpm: float,
    confidence: float,
    previous_groove_id: str | None = None,
    personality: str = "Anchor",
) -> GrooveDecision:
    """Return a ``GrooveDecision`` that explains why a groove was selected.

    Internally mirrors the logic of ``select_groove_by_tempo`` but packages
    the result with a human-readable reason code.

    Reason codes
    ------------
    * ``high_confidence_motorik``
    * ``high_confidence_half_time``
    * ``high_confidence_simple_rock``
    * ``low_confidence_keep_previous``
    * ``low_confidence_default``
    * ``unknown_fallback``
    """
    groove = select_groove_by_tempo(
        bpm=bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        personality=personality,
    )

    selected_id = groove.id

    # Determine reason code
    reason: str
    if confidence < _CONFIDENCE_LOW_THRESHOLD:
        if previous_groove_id is not None and previous_groove_id in load_grooves():
            reason = "low_confidence_keep_previous"
        else:
            reason = "low_confidence_default"
    elif confidence >= _CONFIDENCE_HIGH_THRESHOLD and selected_id == "motorik":
        reason = "high_confidence_motorik"
    elif confidence >= _CONFIDENCE_HIGH_THRESHOLD and selected_id == "half_time":
        reason = "high_confidence_half_time"
    elif selected_id == "simple_rock":
        reason = "high_confidence_simple_rock"
    else:
        reason = "unknown_fallback"

    # Determine changed / held / no-previous
    changed: bool | None
    if previous_groove_id is None:
        changed = None
    elif selected_id == previous_groove_id:
        changed = False
    else:
        changed = True

    return GrooveDecision(
        selected_groove_id=selected_id,
        reason=reason,
        tempo=bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        changed=changed,
    )


# ---- Model-based groove decision -----------------------------------------
#
# Additive — does not replace the existing selectors above.


def select_groove_decision_with_model(
    bpm: float,
    confidence: float,
    model: DrummerModel,
    previous_groove_id: str | None = None,
) -> ModelGrooveDecision:
    """Score candidate grooves using a ``DrummerModel``'s scoring rules.

    Parameters
    ----------
    bpm : float
        Estimated tempo in BPM.
    confidence : float
        Timing confidence (0–1).
    model : DrummerModel
        The model whose ``groove_scoring`` and ``confidence`` rules will
        be used.
    previous_groove_id : str | None
        The groove that was playing previously, if any.

    Returns
    -------
    ModelGrooveDecision
        The selected groove with full candidate score breakdown.

    Reason codes
    ------------
    * ``model_low_confidence_keep_previous`` — confidence below
      ``model.confidence.min_to_change``, kept previous groove.
    * ``model_low_confidence_default`` — confidence below threshold
      and no previous groove, fell back to ``model.default_groove_id``.
    * ``model_highest_score`` — normal scoring path.
    """
    rules = model.groove_scoring
    change_penalty_value = model.transition.change_penalty
    min_to_change = model.confidence.min_to_change

    # ---- Low-confidence fallback -----------------------------------------
    if confidence < min_to_change:
        if previous_groove_id is not None:
            return ModelGrooveDecision(
                selected_groove_id=previous_groove_id,
                reason="model_low_confidence_keep_previous",
                tempo=bpm,
                confidence=confidence,
                previous_groove_id=previous_groove_id,
                changed=False,
                candidate_scores=(),
            )
        else:
            return ModelGrooveDecision(
                selected_groove_id=model.default_groove_id,
                reason="model_low_confidence_default",
                tempo=bpm,
                confidence=confidence,
                previous_groove_id=None,
                changed=None,
                candidate_scores=(),
            )

    # ---- Score each candidate groove -------------------------------------
    scores: list[GrooveCandidateScore] = []

    for gp in rules.grooves:
        # 1. Tempo score: linear falloff from ideal
        tempo_distance = abs(bpm - gp.ideal_tempo)
        tempo_tolerance = max(gp.tempo_tolerance, 1.0)  # avoid divide-by-zero
        tempo_score = max(0.0, 1.0 - tempo_distance / tempo_tolerance)

        # 2. Confidence score
        confidence_score = confidence

        # 3. Preference score
        preference_score = gp.preference

        # 4. Change penalty
        if previous_groove_id is not None and gp.groove_id != previous_groove_id:
            penalty = change_penalty_value
        else:
            penalty = 0.0

        # 5. Total
        total_score = (
            tempo_score * rules.tempo_weight
            + confidence_score * rules.confidence_weight
            + preference_score * rules.preference_weight
            - penalty
        )

        scores.append(
            GrooveCandidateScore(
                groove_id=gp.groove_id,
                total_score=total_score,
                tempo_score=tempo_score,
                confidence_score=confidence_score,
                preference_score=preference_score,
                change_penalty=penalty,
            )
        )

    # ---- Pick the winner -------------------------------------------------
    # Break ties by preferring the previous groove to reduce flicker.
    scores_sorted = sorted(
        scores,
        key=lambda s: (
            s.total_score,
            1.0 if previous_groove_id is not None and s.groove_id == previous_groove_id else 0.0,
        ),
        reverse=True,
    )
    winner = scores_sorted[0]

    # ---- Build result ----------------------------------------------------
    selected_id = winner.groove_id

    changed: bool | None
    if previous_groove_id is None:
        changed = None
    elif selected_id == previous_groove_id:
        changed = False
    else:
        changed = True

    return ModelGrooveDecision(
        selected_groove_id=selected_id,
        reason="model_highest_score",
        tempo=bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        changed=changed,
        candidate_scores=tuple(scores_sorted),
    )