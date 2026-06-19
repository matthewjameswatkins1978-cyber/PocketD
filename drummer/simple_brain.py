"""Simple Brain v0 — conservative, explainable beat-selection engine.

Lock → Choose → Hold → Relisten

This module sits alongside the existing complex behaviour engine
(``drummer/behaviour.py``) and does **not** modify or depend on it.

Design principles
-----------------
* Conservative: beat bank derived from ``data/grooves.yaml``, 3 scoring
  factors, simple thresholds.
* Readable: one function per concept, English reason strings.
* Explainable: ``BrainDecision.scores`` exposes the full score table.
* Deterministic: no randomness, no EMA, no hidden state beyond
  counter fields.
* Database-driven: every non-silence beat name is a real groove ID from
  ``data/grooves.yaml``.  ``silence`` is the only special sentinel.

Input
-----
* ``perception.features.FeatureSnapshot`` — the player's musical state.

Output
------
* ``BrainDecision`` — chosen beat name, confidence, action, reason, scores.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from perception.features import FeatureSnapshot

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhraseAnalysis:
    """Distilled view of a FeatureSnapshot for beat matching."""

    density_label: str  # "sparse" | "medium" | "dense"
    stability: float
    confidence: float
    change_score: float
    silence_duration: float


@dataclass(frozen=True)
class BeatDescriptor:
    """Describes a known beat and the conditions it prefers."""

    name: str
    description: str
    ideal_density: str  # "sparse" | "medium" | "dense"
    min_stability: float
    is_silence: bool = False
    risk: str = "medium"  # "low" | "medium" | "high"
    energy: int = 3  # 1–5
    feel_tags: tuple[str, ...] = ()  # e.g. ("safe", "driving")


class BrainAction(Enum):
    """What phase the simple brain is in."""

    LISTEN = "LISTEN"
    CHOOSE = "CHOOSE"
    HOLD = "HOLD"


@dataclass(frozen=True)
class BrainDecision:
    """Output of ``SimpleBrain.decide()`` — one decision per call."""

    beat_name: str | None
    confidence: float
    reason: str
    action: BrainAction
    scores: dict[str, float]


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LOCK_THRESHOLD = 0.50
"""Minimum ``player_certainty`` to count one snapshot toward lock."""

LOCK_SNAPSHOTS = 4
"""Number of consecutive confident snapshots required to lock."""

SWITCH_THRESHOLD = 0.30
"""Minimum ``change_score`` to consider switching beats."""

SWITCH_CONFIDENCE = 0.35
"""Minimum score advantage a candidate needs to replace the current beat."""

MIN_HOLD_CONFIDENCE = 0.30
"""Minimum confidence required to allow switching (safety floor)."""

RELISTEN_THRESHOLD = 0.20
"""Confidence below this for too long triggers a return to LISTEN."""

RELISTEN_SNAPSHOTS = 2
"""Number of consecutive uncertain snapshots before re-listening."""

# ---------------------------------------------------------------------------
# Beat bank — derived from data/grooves.yaml
# ---------------------------------------------------------------------------

# Backwards-compatible alias so code that referenced the old constant
# still works during migration.
BEAT_BANK: tuple[BeatDescriptor, ...] = ()


def load_simple_brain_beat_bank() -> tuple[BeatDescriptor, ...]:
    """Build the default Simple Brain beat bank from ``data/grooves.yaml``.

    Returns
    -------
    tuple[BeatDescriptor, ...]
        One ``BeatDescriptor`` per groove that has
        ``simple_brain_enabled: true``, plus the special ``silence``
        sentinel.  The descriptor name is the real groove ID for every
        non-silence beat.

    Notes
    -----
    This is called once at ``SimpleBrain`` construction when no custom
    beat bank is supplied.  To add a new beat that Simple Brain can
    choose, add ``simple_brain_enabled: true`` and the required metadata
    fields to the groove in ``data/grooves.yaml``.
    """
    from groove_library import load_grooves

    all_grooves = load_grooves()
    descriptors: list[BeatDescriptor] = []

    for groove in all_grooves.values():
        if not groove.simple_brain_enabled:
            continue
        feel_tags = tuple(groove.feel_tags) if groove.feel_tags else ()
        descriptors.append(
            BeatDescriptor(
                name=groove.id,
                description=groove.description or groove.name,
                ideal_density=groove.ideal_density or "medium",
                min_stability=groove.min_stability,
                is_silence=False,
                risk=groove.risk,
                energy=groove.energy,
                feel_tags=feel_tags,
            )
        )

    # Silence is a special non-groove sentinel — it has no corresponding
    # entry in data/grooves.yaml.
    descriptors.append(
        BeatDescriptor(
            name="silence",
            description="No drumming — let the music breathe",
            ideal_density="sparse",
            min_stability=0.0,
            is_silence=True,
        )
    )

    return tuple(sorted(descriptors, key=lambda d: d.name))


# ---------------------------------------------------------------------------
# Helper: snapshot → PhraseAnalysis
# ---------------------------------------------------------------------------


def analyse_snapshot(snapshot: FeatureSnapshot) -> PhraseAnalysis:
    """Convert a raw FeatureSnapshot into a PhraseAnalysis.

    Density binning:
    * < 0.33  → sparse
    * < 0.66  → medium
    * ≥ 0.66  → dense
    """
    density = snapshot.input_density
    if density < 0.33:
        density_label = "sparse"
    elif density < 0.66:
        density_label = "medium"
    else:
        density_label = "dense"

    return PhraseAnalysis(
        density_label=density_label,
        stability=snapshot.repetition_stability,
        confidence=snapshot.player_certainty,
        change_score=snapshot.change_score,
        silence_duration=snapshot.silence_duration,
    )


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------

# Ordered from smallest to largest so adjacency checks work naturally.
_DENSITY_ORDER = ("sparse", "medium", "dense")


def _density_index(label: str) -> int:
    """Return the ordinal position of a density label."""
    try:
        return _DENSITY_ORDER.index(label)
    except ValueError:
        return -1


def _density_similarity(input_label: str, ideal_label: str) -> float:
    """How well the input density matches the beat's ideal density.

    Returns
    -------
    float
        1.0  — exact match
        0.5  — adjacent densities (sparse ↔ medium, medium ↔ dense)
        0.0  — opposite densities (sparse ↔ dense)
    """
    i_idx = _density_index(input_label)
    b_idx = _density_index(ideal_label)
    if i_idx < 0 or b_idx < 0:
        return 0.0
    diff = abs(i_idx - b_idx)
    if diff == 0:
        return 1.0
    if diff == 1:
        return 0.5
    return 0.0


def _score_normal_beat(beat: BeatDescriptor, analysis: PhraseAnalysis) -> float:
    """Score a non-silence beat against the current phrase analysis."""
    density_sim = _density_similarity(analysis.density_label, beat.ideal_density)
    stability_ok = 1.0 if analysis.stability >= beat.min_stability else 0.0

    return (
        0.45 * density_sim
        + 0.30 * stability_ok
        + 0.25 * analysis.confidence
    )


def _score_silence(analysis: PhraseAnalysis) -> float:
    """Score the special 'silence' beat.

    Silence only wins when confidence has collapsed.  In normal sparse
    passages, real sparse grooves should outscore it.
    """
    # Base score from low confidence — the lower the confidence,
    # the more silence makes sense.
    if analysis.confidence < RELISTEN_THRESHOLD:
        confidence_score = 1.0 - analysis.confidence  # e.g. 0.05 conf → 0.95
    else:
        confidence_score = 0.0

    # Small boost if there's been meaningful silence.
    silence_boost = min(analysis.silence_duration / 4.0, 0.3)

    return (0.70 * confidence_score) + (0.30 * silence_boost)


def _score_beat(beat: BeatDescriptor, analysis: PhraseAnalysis) -> float:
    """Score a single beat, dispatching to normal or silence logic."""
    if beat.is_silence:
        return _score_silence(analysis)
    return _score_normal_beat(beat, analysis)


def _score_all(
    analysis: PhraseAnalysis,
    beat_bank: tuple[BeatDescriptor, ...] | None = None,
) -> dict[str, float]:
    """Score every beat in the given bank and return a name→score dict.

    If no bank is provided, ``load_simple_brain_beat_bank()`` is used.
    """
    bank = beat_bank if beat_bank is not None else load_simple_brain_beat_bank()
    return {beat.name: _score_beat(beat, analysis) for beat in bank}


# ---------------------------------------------------------------------------
# Conservative tie-breaking
# ---------------------------------------------------------------------------

# Risk order: lower risk wins ties.
_RISK_ORDER = {"low": 0, "medium": 1, "high": 2}

# When scores are equal (or within epsilon), prefer conservative beats.
_TIE_EPSILON = 0.0001


def _select_best_beat(
    scores: dict[str, float],
    beat_bank: tuple[BeatDescriptor, ...],
) -> str:
    """Choose the best beat from *scores* with conservative tie-breaking.

    Tie-break priority (when scores differ by < _TIE_EPSILON):
    1. Higher score
    2. Lower risk (low → medium → high)
    3. Lower energy
    4. ``"safe"`` feel tag present
    5. Alphabetical by name (deterministic fallback)

    This ensures that when dense grooves all score equally,
    ``simple_rock`` (low risk, energy=3, safe tag) wins over
    ``funk_pocket`` (medium risk, energy=4, no safe tag) and
    ``punk_drive`` (high risk, energy=5).
    """
    # Build a lookup: beat_name → descriptor
    desc_by_name: dict[str, BeatDescriptor] = {d.name: d for d in beat_bank}

    def _tie_key(name: str) -> tuple[float, int, int, int, str]:
        score = scores.get(name, -1.0)
        # Invert score so higher scores sort first.
        neg_score = -score if score > -1.0 else 1.0
        desc = desc_by_name.get(name)
        if desc is None or desc.is_silence:
            return (neg_score, 0, 0, 0, name)
        risk_rank = _RISK_ORDER.get(desc.risk, 1)
        energy = desc.energy
        has_safe = 0 if "safe" in desc.feel_tags else 1
        return (neg_score, risk_rank, energy, has_safe, name)

    return min(scores, key=_tie_key)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# SimpleBrain
# ---------------------------------------------------------------------------


class SimpleBrain:
    """Lock → Choose → Hold → Relisten beat-selection engine.

    Parameters
    ----------
    beat_bank : tuple[BeatDescriptor, ...] | None
        Optional custom beat bank.  If ``None``, the default bank is
        loaded from ``data/grooves.yaml`` via ``load_simple_brain_beat_bank()``.
    """

    def __init__(
        self, beat_bank: tuple[BeatDescriptor, ...] | None = None
    ) -> None:
        self._beat_bank = (
            beat_bank if beat_bank is not None else load_simple_brain_beat_bank()
        )

        # Internal state
        self.consecutive_confident_snapshots: int = 0
        self.consecutive_uncertain_snapshots: int = 0
        self.current_beat: str | None = None
        self.has_locked: bool = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decide(self, snapshot: FeatureSnapshot) -> BrainDecision:
        """Produce one decision for the given snapshot.

        Parameters
        ----------
        snapshot : FeatureSnapshot
            A point-in-time summary of the player's musical behaviour.

        Returns
        -------
        BrainDecision
            The chosen action, beat, confidence, reason, and scores.
        """
        analysis = analyse_snapshot(snapshot)

        # --- Relisten: confidence collapse ---
        if self.has_locked or self.current_beat is not None:
            if analysis.confidence < RELISTEN_THRESHOLD:
                self.consecutive_uncertain_snapshots += 1
            else:
                self.consecutive_uncertain_snapshots = 0

            if self.consecutive_uncertain_snapshots >= RELISTEN_SNAPSHOTS:
                return self._do_relisten(analysis)

        # --- Lock: build confidence ---
        if not self.has_locked:
            return self._do_lock(analysis)

        # --- Choose or Hold ---
        return self._do_choose_or_hold(analysis)

    # ------------------------------------------------------------------
    # Phase implementations
    # ------------------------------------------------------------------

    def _do_relisten(self, analysis: PhraseAnalysis) -> BrainDecision:
        """Confidence collapsed — dump state and return to listening."""
        snap_count = self.consecutive_uncertain_snapshots

        self.consecutive_confident_snapshots = 0
        self.consecutive_uncertain_snapshots = 0
        self.current_beat = None
        self.has_locked = False

        return BrainDecision(
            beat_name=None,
            confidence=analysis.confidence,
            action=BrainAction.LISTEN,
            reason=(
                f"relistening: confidence {analysis.confidence:.2f} below "
                f"threshold {RELISTEN_THRESHOLD:.2f} for "
                f"{snap_count} snapshots"
            ),
            scores={},
        )

    def _do_lock(self, analysis: PhraseAnalysis) -> BrainDecision:
        """In lock phase — accumulate confident snapshots."""
        if analysis.confidence >= LOCK_THRESHOLD:
            self.consecutive_confident_snapshots += 1
        else:
            self.consecutive_confident_snapshots = 0

        remaining = LOCK_SNAPSHOTS - self.consecutive_confident_snapshots

        if self.consecutive_confident_snapshots >= LOCK_SNAPSHOTS:
            self.has_locked = True
            # Fall through: the *next* call will actually choose.
            # But we need to return a decision now.
            # We'll do the choose immediately in this same call.
            return self._do_choose_or_hold(analysis)

        # Not yet locked.
        if self.consecutive_confident_snapshots == 0 and remaining == LOCK_SNAPSHOTS:
            detail = "confidence dipped, lock reset"
        else:
            detail = (
                f"{self.consecutive_confident_snapshots}/{LOCK_SNAPSHOTS} "
                f"confident snapshots"
            )

        return BrainDecision(
            beat_name=None,
            confidence=analysis.confidence,
            action=BrainAction.LISTEN,
            reason=f"listening: {detail}",
            scores={},
        )

    def _do_choose_or_hold(self, analysis: PhraseAnalysis) -> BrainDecision:
        """Score all beats and decide whether to choose or hold."""
        scores = _score_all(analysis, self._beat_bank)
        # Use conservative tie-breaking so that when multiple beats
        # have equal or near-equal scores, safer/lower-risk grooves win.
        best_name = _select_best_beat(scores, self._beat_bank)
        best_score = scores[best_name]

        # If we have no current beat, we're choosing for the first time.
        if self.current_beat is None:
            self.current_beat = best_name
            return BrainDecision(
                beat_name=best_name,
                confidence=best_score,
                action=BrainAction.CHOOSE,
                reason=(
                    f"choosing: locked after {LOCK_SNAPSHOTS} confident "
                    f"snapshots; best match is {best_name}"
                ),
                scores=scores,
            )

        # We're holding — check whether we should switch.
        current_score = scores.get(self.current_beat, 0.0)

        change_ok = analysis.change_score >= SWITCH_THRESHOLD
        conf_ok = analysis.confidence >= MIN_HOLD_CONFIDENCE
        different = best_name != self.current_beat
        advantage = best_score - current_score
        advantage_ok = advantage >= SWITCH_CONFIDENCE

        if change_ok and conf_ok and different and advantage_ok:
            old_beat = self.current_beat
            self.current_beat = best_name
            return BrainDecision(
                beat_name=best_name,
                confidence=best_score,
                action=BrainAction.CHOOSE,
                reason=(
                    f"switching: {old_beat} -> {best_name} "
                    f"(change={analysis.change_score:.2f}, "
                    f"delta={advantage:.2f})"
                ),
                scores=scores,
            )

        # Build a descriptive hold reason.
        reasons: list[str] = []
        if not change_ok:
            reasons.append(
                f"change_score {analysis.change_score:.2f} < "
                f"threshold {SWITCH_THRESHOLD:.2f}"
            )
        if not conf_ok:
            reasons.append(
                f"confidence {analysis.confidence:.2f} < "
                f"switch minimum {MIN_HOLD_CONFIDENCE:.2f}"
            )
        if not different:
            reasons.append(
                f"best candidate is still {self.current_beat}"
            )
        elif not advantage_ok:
            # Omit delta when best candidate is unchanged (already covered above).
            reasons.append(
                f"score delta {advantage:.2f} < "
                f"switch threshold {SWITCH_CONFIDENCE:.2f}"
            )

        detail = "; ".join(reasons) if reasons else "no major change detected"

        return BrainDecision(
            beat_name=self.current_beat,
            confidence=current_score,
            action=BrainAction.HOLD,
            reason=f"holding: {detail}",
            scores=scores,
        )