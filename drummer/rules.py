"""Reusable frozen dataclass definitions for Bunny Deluxe models.

Rules define *possible* behaviour.
Models (in ``drummer.models``) choose *actual* behaviour.
The engine (elsewhere) runs that behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# ConfidenceRules  — thresholds for the confidence engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfidenceRules:
    """Thresholds that control how the confidence engine classifies timing.

    Parameters
    ----------
    timing_low : float
        Below this confidence (0–1) the player is considered unstable;
        the system should favour safe, simple grooves.
    timing_high : float
        Above this confidence (0–1) the player is locked in; more
        interesting / driving grooves become available.
    recovery_threshold : float
        Minimum confidence needed before the system will attempt to
        transition away from a recovery state.
    min_to_change : float
        Minimum confidence required to consider switching grooves at
        all.  Below this threshold the model will stick with the
        previous groove (or the default if none).
    """

    timing_low: float = 0.4
    timing_high: float = 0.7
    recovery_threshold: float = 0.3
    min_to_change: float = 0.3


# ---------------------------------------------------------------------------
# TransitionRules  — controls for groove transitions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TransitionRules:
    """Rules that govern when and how the drummer switches grooves.

    Parameters
    ----------
    min_hold_bars : int
        Minimum number of bars the drummer should stay on a groove
        before considering a transition.
    confidence_drop_allowed : float
        How much confidence can drop before a transition is forced
        (relative to the confidence when the current groove was chosen).
    same_groove_cooldown_bars : int
        If a groove was just deselected, how many bars before it can
        be selected again.
    change_penalty : float
        Penalty subtracted from the total score of any groove that
        differs from the previous one.  Higher values discourage
        unnecessary switching.
    """

    min_hold_bars: int = 4
    confidence_drop_allowed: float = 0.15
    same_groove_cooldown_bars: int = 8
    change_penalty: float = 0.10


# ---------------------------------------------------------------------------
# HumanizeRules  — per-instrument timing and velocity variation
# ---------------------------------------------------------------------------


def _default_timing_bias() -> dict[str, float]:
    return {"kick": 0.0, "snare": 0.0, "hat": 0.0}


def _default_timing_jitter() -> dict[str, float]:
    return {"kick": 2.0, "snare": 4.0, "hat": 6.0}


def _default_velocity_jitter() -> dict[str, int]:
    return {"kick": 3, "snare": 4, "hat": 6}


@dataclass(frozen=True)
class HumanizeRules:
    """Micro-timing and velocity variation applied during event rendering.

    Higher spreads make the drummer sound looser / more organic.
    Lower spreads keep the drummer tight and machine-like.

    Per-instrument values are looked up by instrument name
    (``"kick"``, ``"snare"``, ``"hat"``) via dict fields.  Instruments
    not listed fall back to the global ceiling values.

    Parameters
    ----------
    timing_amount_ms : float
        Global ceiling for timing randomisation (ms).  Used as a fallback
        when an instrument is not in ``timing_jitter_ms``.
    velocity_amount : int
        Global ceiling for velocity randomisation (MIDI units).  Used as
        a fallback when an instrument is not in ``velocity_jitter``.
    timing_bias_ms : dict[str, float]
        Systematic timing offset per instrument (ms).  Negative values
        push the hit slightly early, positive values slightly late.
        Defaults to 0.0 for kick, snare, and hat.
    timing_jitter_ms : dict[str, float]
        Half-range of uniform timing jitter per instrument (ms).
        Defaults to ``{"kick": 2.0, "snare": 4.0, "hat": 6.0}``.
    velocity_jitter : dict[str, int]
        Half-range of uniform velocity jitter per instrument (MIDI
        units).  Defaults to ``{"kick": 3, "snare": 4, "hat": 6}``.
    seeded : bool
        If ``True``, humanisation should use a deterministic seed for
        reproducible output.
    """

    timing_amount_ms: float = 8.0
    velocity_amount: int = 6
    timing_bias_ms: dict[str, float] = field(default_factory=_default_timing_bias)
    timing_jitter_ms: dict[str, float] = field(default_factory=_default_timing_jitter)
    velocity_jitter: dict[str, int] = field(default_factory=_default_velocity_jitter)
    seeded: bool = False


# ---------------------------------------------------------------------------
# VariationRules  — controlled pattern-level variation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VariationRules:
    """Controls how a DrummerModel introduces human-like beat variation.

    Variation should be rule-based and weighted, not purely random.
    The drummer should sound like the same drummer making small choices,
    not like a slot machine falling down stairs.

    Future variation implementation rules:

    * Preserve the core groove identity (kick/snare skeleton).
    * Use deterministic seed support for reproducible tests.
    * Never remove an essential backbeat snare unless the model explicitly
      allows it via ``allow_snare_variation``.
    * Never create impossible MIDI values or negative timestamps.
    * Avoid too many variations within a single bar.
    * Prefer small hat/velocity/ghost-note changes before changing kick
      or snare structural hits.
    * Expose reason codes / variation labels where possible.

    Useful future variation labels:

        ``"hat_velocity_variation"``
        ``"hat_skip"``
        ``"kick_extra_low_probability"``
        ``"bar_end_pickup"``
        ``"ghost_snare_added"``
        ``"no_variation"``

    Parameters
    ----------
    variation_probability : float
        Overall probability (0–1) that *any* variation occurs in a bar.
    max_variations_per_bar : int
        Maximum number of variations allowed in a single bar to avoid
        sounding chaotic.
    allow_kick_variation : bool
        Whether the kick drum pattern may be altered (extra hits, skipped
        hits, or small placement shifts).
    allow_snare_variation : bool
        Whether the snare / backbeat pattern may be altered. Should be
        ``False`` for most models to preserve the core backbeat.
    allow_hat_variation : bool
        Whether hi-hat pattern, velocity, or open/closed state may vary.
    ghost_note_probability : float
        Probability (0–1) of adding a ghost note (very low velocity) on
        snare or hat at an otherwise empty step.
    bar_end_variation_probability : float
        Probability (0–1) of adding a small pickup / fill-like variation
        at the end of a bar.
    seeded : bool
        If ``True``, the model should use a deterministic seed so that
        variation output is reproducible for testing.
    """

    variation_probability: float = 0.05
    max_variations_per_bar: int = 1
    allow_kick_variation: bool = False
    allow_snare_variation: bool = False
    allow_hat_variation: bool = True
    ghost_note_probability: float = 0.02
    bar_end_variation_probability: float = 0.05
    seeded: bool = False


# ---------------------------------------------------------------------------
# GroovePreference  — one groove in a model's candidate pool
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GroovePreference:
    """A single groove the model knows about and how to score it.

    Parameters
    ----------
    groove_id : str
        The id of the groove in the YAML library.
    ideal_tempo : float
        The BPM at which this groove sounds best for the model.
    tempo_tolerance : float
        Half-width of the tempo range (in BPM) over which the groove
        is considered playable.  The tempo score drops linearly from
        1.0 at ``ideal_tempo`` to 0.0 at ``ideal_tempo +/- tolerance``.
    preference : float
        Base preference weight for this groove.  Higher values make
        the model more likely to choose this groove all else equal.
    """

    groove_id: str
    ideal_tempo: float
    tempo_tolerance: float = 30.0
    preference: float = 0.5


# ---------------------------------------------------------------------------
# GrooveScoringRules  — scoring weights for model-based groove selection
# ---------------------------------------------------------------------------


def _default_grooves() -> list[GroovePreference]:
    return []


@dataclass(frozen=True)
class GrooveScoringRules:
    """Weights and candidate grooves used by the model-based groove selector.

    The total score for a candidate groove is::

        total = (tempo_score * tempo_weight
               + confidence_score * confidence_weight
               + preference * preference_weight
               - change_penalty_if_applicable)

    Legacy fields (density_weight, energy_weight, syncopation_weight,
    strong_beat_bonus, personality_bonus) are kept for the original
    heuristic selector and are not used by the model-based path.

    Parameters
    ----------
    tempo_weight : float
        Weight for the tempo-fit score.
    confidence_weight : float
        Weight for the confidence score.
    preference_weight : float
        Weight for the model's base preference for each groove.
    grooves : list[GroovePreference]
        Candidate grooves the model will consider, each with its own
        ideal tempo, tolerance, and preference weight.
    density_weight : float
        Legacy — used by the original ``select_groove`` heuristic.
    energy_weight : float
        Legacy — used by the original heuristic.
    syncopation_weight : float
        Legacy — used by the original heuristic.
    strong_beat_bonus : float
        Legacy — used by the original heuristic.
    personality_bonus : float
        Legacy — used by the original heuristic.
    """

    tempo_weight: float = 0.40
    confidence_weight: float = 0.30
    preference_weight: float = 0.30
    grooves: list[GroovePreference] = field(default_factory=_default_grooves)

    # Legacy fields (kept for backward compatibility with original selector)
    density_weight: float = 0.35
    energy_weight: float = 0.35
    syncopation_weight: float = 0.15
    strong_beat_bonus: float = 0.05
    personality_bonus: float = 0.10


# ---------------------------------------------------------------------------
# GrooveCandidateScore  — per-candidate breakdown
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GrooveCandidateScore:
    """Breakdown of how a single groove scored in model-based selection.

    Parameters
    ----------
    groove_id : str
        The groove's id.
    total_score : float
        The final total score after all weights and penalties.
    tempo_score : float
        How well the tempo matches (0–1).
    confidence_score : float
        The current confidence value.
    preference_score : float
        The model's base preference for this groove.
    change_penalty : float
        Penalty applied (0 if holding, >0 if switching).
    """

    groove_id: str
    total_score: float
    tempo_score: float
    confidence_score: float
    preference_score: float
    change_penalty: float


# ---------------------------------------------------------------------------
# ModelGrooveDecision  — decision result with candidate scores
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelGrooveDecision:
    """A groove decision produced by the model-based scorer.

    Parameters
    ----------
    selected_groove_id : str
        The id of the groove that was selected.
    reason : str
        Reason code explaining why this groove was chosen.
        One of:
        ``"model_low_confidence_keep_previous"``
        ``"model_low_confidence_default"``
        ``"model_highest_score"``
    tempo : float
        The estimated BPM used for the decision.
    confidence : float
        The timing confidence used for the decision.
    previous_groove_id : str | None
        The previous groove id, if any was provided.
    changed : bool | None
        ``True`` if the selected groove differs from the previous one,
        ``False`` if it is the same, ``None`` if there was no previous.
    candidate_scores : tuple[GrooveCandidateScore, ...]
        Per-candidate score breakdowns for high-confidence decisions.
        Empty for low-confidence fallback decisions.
    """

    selected_groove_id: str
    reason: str
    tempo: float
    confidence: float
    previous_groove_id: str | None
    changed: bool | None
    candidate_scores: tuple[GrooveCandidateScore, ...] = ()


# ---------------------------------------------------------------------------
# DrummerModel  — swappable personality that bundles all rules
# ---------------------------------------------------------------------------
#
# Each DrummerModel bundles:
#   * a model identity (id, name, description)
#   * confidence thresholds
#   * transition policies
#   * humanisation (micro-timing) rules
#   * variation (pattern-level) rules
#   * groove scoring weights + candidate grooves
#   * a preferred groove list and a safe fallback
#
# Rules define possible behaviour.
# Models choose actual behaviour.
# The engine runs behaviour.
#
# The model is *not* wired into the scheduler yet — this is the foundation
# stage. Once wired, the scheduler will:
#
#   1. Load the active DrummerModel.
#   2. Select a groove (via the model's preferences + context).
#   3. Optionally apply small variations per the model's VariationRules
#      at render/playback time.
#
# Later models can be busier / more restless by adjusting these parameters.


@dataclass(frozen=True)
class DrummerModel:
    """A swappable drummer personality.

    Parameters
    ----------
    id : str
        Unique identifier (e.g. ``"motorik_tight"``).
    name : str
        Human-readable name (e.g. ``"Motorik Tight"``).
    description : str
        Short description of the model's character and intended use.
    confidence : ConfidenceRules
        Thresholds for timing confidence classification.
    transition : TransitionRules
        Rules governing groove transitions.
    humanize : HumanizeRules
        Per-instrument micro-timing / velocity variation.
    variation : VariationRules
        Rules controlling pattern-level beat variation.
    groove_scoring : GrooveScoringRules
        Weights + candidate grooves for model-based selection.
    complexity_level : int
        Base complexity level (1–10) used by the scheduler's step filter.
    preferred_groove_ids : list[str] | None
        Ordered list of groove IDs the model prefers.
    default_groove_id : str
        Fallback groove ID when no preference can be satisfied.
    """

    id: str
    name: str
    description: str
    confidence: ConfidenceRules = field(default_factory=ConfidenceRules)
    transition: TransitionRules = field(default_factory=TransitionRules)
    humanize: HumanizeRules = field(default_factory=HumanizeRules)
    variation: VariationRules = field(default_factory=VariationRules)
    groove_scoring: GrooveScoringRules = field(default_factory=GrooveScoringRules)
    complexity_level: int = 5
    preferred_groove_ids: list[str] | None = None
    default_groove_id: str = "simple_rock"
