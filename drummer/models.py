"""Preset DrummerModel instances for Bunny Deluxe.

Models choose *actual* behaviour (they import rule dataclasses from
``drummer.rules`` and compose them into named personalities).

Add new preset models here rather than scattering them across the
codebase.
"""

from __future__ import annotations

from drummer.rules import (
    ConfidenceRules,
    DrummerModel,
    GroovePreference,
    GrooveScoringRules,
    HumanizeRules,
    TransitionRules,
    VariationRules,
)

# ---------------------------------------------------------------------------
# HOT-ROD  (default / proof-of-concept model)
# ---------------------------------------------------------------------------
# Straight-ahead rock with a steady backbeat.  Occasional small bar-end
# fills and hat variations.  Kick may take small liberties; snare is
# always solid.  A safe default for most pop / rock contexts.

SIMPLE_ROCK_SAFE_VARIATION = VariationRules(
    variation_probability=0.08,
    max_variations_per_bar=1,
    allow_kick_variation=True,
    allow_snare_variation=False,
    allow_hat_variation=True,
    ghost_note_probability=0.04,
    bar_end_variation_probability=0.10,
    seeded=True,
)

SIMPLE_ROCK_SAFE_HUMANIZE = HumanizeRules(
    timing_amount_ms=8.0,
    velocity_amount=6,
    timing_bias_ms={"kick": 0.0, "snare": 0.0, "hat": 0.0},
    timing_jitter_ms={"kick": 2.0, "snare": 4.0, "hat": 6.0},
    velocity_jitter={"kick": 3, "snare": 4, "hat": 6},
    seeded=True,
)

SIMPLE_ROCK_SAFE_CONFIDENCE = ConfidenceRules(
    timing_low=0.4,
    timing_high=0.7,
    recovery_threshold=0.3,
    min_to_change=0.3,
)

SIMPLE_ROCK_SAFE_TRANSITION = TransitionRules(
    min_hold_bars=4,
    confidence_drop_allowed=0.15,
    same_groove_cooldown_bars=8,
    change_penalty=0.10,
)

SIMPLE_ROCK_SAFE_GROOVES = [
    GroovePreference(groove_id="simple_rock", ideal_tempo=110.0, tempo_tolerance=35.0, preference=0.8),
    GroovePreference(groove_id="motorik", ideal_tempo=120.0, tempo_tolerance=20.0, preference=0.4),
    GroovePreference(groove_id="half_time", ideal_tempo=90.0, tempo_tolerance=20.0, preference=0.3),
    GroovePreference(groove_id="punk_drive", ideal_tempo=140.0, tempo_tolerance=25.0, preference=0.2),
    GroovePreference(groove_id="funk_pocket", ideal_tempo=100.0, tempo_tolerance=15.0, preference=0.3),
]

SIMPLE_ROCK_SAFE_SCORING = GrooveScoringRules(
    tempo_weight=0.40,
    confidence_weight=0.30,
    preference_weight=0.30,
    grooves=SIMPLE_ROCK_SAFE_GROOVES,
    density_weight=0.35,
    energy_weight=0.35,
    syncopation_weight=0.15,
    strong_beat_bonus=0.05,
    personality_bonus=0.10,
)

SIMPLE_ROCK_SAFE_MODEL = DrummerModel(
    id="simple_rock_safe",
    name="Simple Rock (Safe)",
    description=(
        "Straight-ahead rock with a steady backbeat. "
        "Occasional small bar-end fills and hat variations. "
        "Kick may take small liberties; snare is always solid. "
        "A safe default for most pop / rock contexts."
    ),
    confidence=SIMPLE_ROCK_SAFE_CONFIDENCE,
    transition=SIMPLE_ROCK_SAFE_TRANSITION,
    humanize=SIMPLE_ROCK_SAFE_HUMANIZE,
    variation=SIMPLE_ROCK_SAFE_VARIATION,
    groove_scoring=SIMPLE_ROCK_SAFE_SCORING,
    complexity_level=5,
    preferred_groove_ids=["simple_rock", "motorik"],
    default_groove_id="simple_rock",
)

# ---------------------------------------------------------------------------
# MOTORIK TIGHT  — steady, driving motorik pulse
# ---------------------------------------------------------------------------
# Hat velocities vary rarely; kick and snare are locked.
# Suitable for krautrock, minimal techno, or any locked-in groove.

MOTORIK_TIGHT_VARIATION = VariationRules(
    variation_probability=0.05,
    max_variations_per_bar=1,
    allow_kick_variation=False,
    allow_snare_variation=False,
    allow_hat_variation=True,
    ghost_note_probability=0.02,
    bar_end_variation_probability=0.05,
    seeded=True,
)

MOTORIK_TIGHT_HUMANIZE = HumanizeRules(
    timing_amount_ms=4.0,
    velocity_amount=3,
    timing_bias_ms={"kick": 0.0, "snare": 0.0, "hat": 0.0},
    timing_jitter_ms={"kick": 1.0, "snare": 2.0, "hat": 4.0},
    velocity_jitter={"kick": 2, "snare": 2, "hat": 3},
    seeded=True,
)

MOTORIK_TIGHT_CONFIDENCE = ConfidenceRules(
    timing_low=0.5,
    timing_high=0.8,
    recovery_threshold=0.3,
    min_to_change=0.4,
)

MOTORIK_TIGHT_TRANSITION = TransitionRules(
    min_hold_bars=6,
    confidence_drop_allowed=0.20,
    same_groove_cooldown_bars=12,
    change_penalty=0.15,
)

MOTORIK_TIGHT_GROOVES = [
    GroovePreference(groove_id="motorik", ideal_tempo=120.0, tempo_tolerance=25.0, preference=1.0),
    GroovePreference(groove_id="simple_rock", ideal_tempo=110.0, tempo_tolerance=30.0, preference=0.3),
    GroovePreference(groove_id="punk_drive", ideal_tempo=140.0, tempo_tolerance=20.0, preference=0.5),
    GroovePreference(groove_id="half_time", ideal_tempo=90.0, tempo_tolerance=15.0, preference=0.1),
]

MOTORIK_TIGHT_SCORING = GrooveScoringRules(
    tempo_weight=0.45,
    confidence_weight=0.25,
    preference_weight=0.30,
    grooves=MOTORIK_TIGHT_GROOVES,
    density_weight=0.40,
    energy_weight=0.30,
    syncopation_weight=0.10,
    strong_beat_bonus=0.10,
    personality_bonus=0.10,
)

MOTORIK_TIGHT_MODEL = DrummerModel(
    id="motorik_tight",
    name="Motorik Tight",
    description=(
        "Steady, driving motorik pulse with extremely tight control. "
        "Hat velocities vary rarely; kick and snare are locked. "
        "Suitable for krautrock, minimal techno, or any locked-in groove."
    ),
    confidence=MOTORIK_TIGHT_CONFIDENCE,
    transition=MOTORIK_TIGHT_TRANSITION,
    humanize=MOTORIK_TIGHT_HUMANIZE,
    variation=MOTORIK_TIGHT_VARIATION,
    groove_scoring=MOTORIK_TIGHT_SCORING,
    complexity_level=5,
    preferred_groove_ids=["motorik"],
    default_groove_id="motorik",
)

# ---------------------------------------------------------------------------
# SPARSE POST-PUNK  — lean, roomy pattern
# ---------------------------------------------------------------------------
# Kick can shift slightly; snare stays solid on the backbeat.
# Hat pattern opens up and ghosts are more common.
# Suitable for post-punk, dub, or minimal indie.

SPARSE_POSTPUNK_VARIATION = VariationRules(
    variation_probability=0.15,
    max_variations_per_bar=2,
    allow_kick_variation=True,
    allow_snare_variation=False,
    allow_hat_variation=True,
    ghost_note_probability=0.08,
    bar_end_variation_probability=0.15,
    seeded=True,
)

SPARSE_POSTPUNK_HUMANIZE = HumanizeRules(
    timing_amount_ms=10.0,
    velocity_amount=8,
    timing_bias_ms={"kick": 0.0, "snare": 0.0, "hat": 1.0},
    timing_jitter_ms={"kick": 3.0, "snare": 5.0, "hat": 8.0},
    velocity_jitter={"kick": 4, "snare": 5, "hat": 8},
    seeded=True,
)

SPARSE_POSTPUNK_CONFIDENCE = ConfidenceRules(
    timing_low=0.35,
    timing_high=0.65,
    recovery_threshold=0.25,
    min_to_change=0.25,
)

SPARSE_POSTPUNK_TRANSITION = TransitionRules(
    min_hold_bars=3,
    confidence_drop_allowed=0.20,
    same_groove_cooldown_bars=6,
    change_penalty=0.08,
)

SPARSE_POSTPUNK_GROOVES = [
    GroovePreference(groove_id="half_time", ideal_tempo=90.0, tempo_tolerance=25.0, preference=0.7),
    GroovePreference(groove_id="simple_rock", ideal_tempo=110.0, tempo_tolerance=30.0, preference=0.5),
    GroovePreference(groove_id="motorik", ideal_tempo=120.0, tempo_tolerance=20.0, preference=0.3),
    GroovePreference(groove_id="punk_drive", ideal_tempo=140.0, tempo_tolerance=20.0, preference=0.2),
    GroovePreference(groove_id="funk_pocket", ideal_tempo=100.0, tempo_tolerance=15.0, preference=0.4),
]

SPARSE_POSTPUNK_SCORING = GrooveScoringRules(
    tempo_weight=0.35,
    confidence_weight=0.25,
    preference_weight=0.40,
    grooves=SPARSE_POSTPUNK_GROOVES,
    density_weight=0.30,
    energy_weight=0.30,
    syncopation_weight=0.20,
    strong_beat_bonus=0.05,
    personality_bonus=0.15,
)

SPARSE_POSTPUNK_MODEL = DrummerModel(
    id="sparse_postpunk",
    name="Sparse Post-Punk",
    description=(
        "Lean, roomy pattern with space for the bass and vocal. "
        "Kick can shift slightly; snare stays solid on the backbeat. "
        "Hat pattern opens up and ghosts are more common. "
        "Suitable for post-punk, dub, or minimal indie."
    ),
    confidence=SPARSE_POSTPUNK_CONFIDENCE,
    transition=SPARSE_POSTPUNK_TRANSITION,
    humanize=SPARSE_POSTPUNK_HUMANIZE,
    variation=SPARSE_POSTPUNK_VARIATION,
    groove_scoring=SPARSE_POSTPUNK_SCORING,
    complexity_level=4,
    preferred_groove_ids=["half_time", "simple_rock"],
    default_groove_id="simple_rock",
)

# ---------------------------------------------------------------------------
# Lookup registry  (optional convenience for the future engine)
# ---------------------------------------------------------------------------

BUILTIN_MODELS: dict[str, DrummerModel] = {
    "simple_rock_safe": SIMPLE_ROCK_SAFE_MODEL,
    "motorik_tight": MOTORIK_TIGHT_MODEL,
    "sparse_postpunk": SPARSE_POSTPUNK_MODEL,
}
