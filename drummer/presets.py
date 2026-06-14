"""Drummer Presets — quick-switch temperaments for playtesting feel.

Each preset bundles a ``DrummerProfile``, ``PhraseMarkerConfig``, and
``OutputShapingConfig`` into a single ``DrummerPresetConfig`` that tunes
the drummer's personality *without* changing the core brain logic.

Design principle
----------------
Presets are *not* difficulty modes or gimmicks.
They are drummer temperaments — subtle shifts in confidence thresholds,
phrase marker frequency, velocity assertiveness, and entry timing.

All presets preserve:
* DROP output contract (> 0 events, sparse kicks, no crash)
* BAIL output contract (exactly 0 events)
* FINAL_BAIL output contract (exactly kick + crash on beat 1)
* No phrase markers during ANCHOR, DROP, BAIL, or FINAL_BAIL
* No complex fills, no crash spam, no busy hats all the time
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from drummer.behaviour import DrummerProfile, ConservativePocketDrummer
from drummer.phrase_markers import PhraseMarkerConfig
from drummer.output_shaping import OutputShapingConfig


# ---------------------------------------------------------------------------
# DrummerPreset enum
# ---------------------------------------------------------------------------


class DrummerPreset(str, Enum):
    """Named drummer temperaments for quick playtesting.

    Each preset represents a coherent personality that adjusts how
    confidently, assertively, and decoratively the drummer plays —
    without altering the core behaviour decision logic.
    """

    CAUTIOUS = "cautious"
    NORMAL = "normal"
    BRAVER = "braver"

    @classmethod
    def list_names(cls) -> list[str]:
        """Return all valid preset names."""
        return [m.value for m in cls]

    @classmethod
    def from_name(cls, name: str) -> DrummerPreset:
        """Parse a string into a DrummerPreset, case-insensitively.

        Raises ``ValueError`` with a helpful message listing valid
        options if no match is found.
        """
        v = name.strip().lower()
        for member in cls:
            if v == member.value or v == member.name.lower():
                return member
        valid = ", ".join(cls.list_names())
        raise ValueError(
            f"Unknown preset '{name}'. Valid presets: {valid}"
        )


# ---------------------------------------------------------------------------
# DrummerPresetConfig — bundles all configurable pieces for one preset
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrummerPresetConfig:
    """Complete configuration bundle for a single drummer temperament.

    Parameters
    ----------
    name : str
        Human-readable preset name (e.g. ``"Cautious Bunny"``).
    profile : DrummerProfile
        Behaviour decision thresholds (entry, bail, build, reduce, etc.).
    phrase_config : PhraseMarkerConfig
        Phrase marker thresholds (ear-perks and frills).
    output_config : OutputShapingConfig
        Output velocity/articulation shaping knobs.
    """

    name: str
    profile: DrummerProfile = field(default_factory=lambda: ConservativePocketDrummer)
    phrase_config: PhraseMarkerConfig = field(default_factory=PhraseMarkerConfig)
    output_config: OutputShapingConfig = field(default_factory=OutputShapingConfig)


# ============================================================================
# Normal preset — current default behaviour (reference)
# ============================================================================

_NORMAL_PROFILE = ConservativePocketDrummer
_NORMAL_PHRASE = PhraseMarkerConfig()
_NORMAL_OUTPUT = OutputShapingConfig()

NORMAL_PRESET = DrummerPresetConfig(
    name="Normal Bunny",
    profile=_NORMAL_PROFILE,
    phrase_config=_NORMAL_PHRASE,
    output_config=_NORMAL_OUTPUT,
)

# ============================================================================
# Cautious preset — safer, slower, less decorative
# ============================================================================

# Cautious profile: higher entry thresholds, slower to jump in
_CAUTIOUS_PROFILE = DrummerProfile(
    name="Cautious Bunny",
    # Base thresholds
    hysteresis_margin=0.10,
    bail_silence_seconds=0.50,
    density_inversion_threshold=0.75,
    fill_probability_base=0.05,
    energy_ema_alpha=0.10,
    density_ema_alpha=0.10,
    # Stage 2 — more conservative entry
    min_pulse_confidence=0.80,
    min_bar_confidence=0.75,
    full_entry_confidence=0.90,
    soft_entry_confidence=0.80,
    min_observation_seconds=2.00,  # longer observation window
    severe_uncertainty_threshold=0.35,
    maintain_hysteresis_margin=0.10,
    # Stage 3 — slower builds, quicker reduce
    fast_energy_ema_alpha=0.15,
    slow_energy_ema_alpha=0.02,
    build_trend_threshold=0.18,  # needs stronger trend to build
    reduce_trend_threshold=-0.10,
    drop_trend_threshold=-0.30,
    max_density_for_build=0.75,  # less density tolerance for build
    low_energy_threshold_for_drop=0.25,
    density_collapse_ratio_for_drop=0.35,
    min_build_duration_seconds=2.0,
    min_reduce_duration_seconds=2.0,
    min_drop_duration_seconds=1.0,
    # Stage 4 — feature-driven thresholds
    enter_certainty_threshold=0.70,   # need more certainty to enter
    enter_repetition_threshold=0.75,
    enter_confirmation_snapshots=4,   # more confirmations before entry
    build_change_threshold=0.25,      # needs stronger change to build
    build_certainty_threshold=0.60,
    build_repetition_threshold=0.70,
    build_phase_threshold=0.60,
    build_max_density_without_phrase=0.75,
    reduce_density_threshold=0.70,    # reduce sooner (lower threshold)
    anchor_certainty_threshold=0.45,  # anchor at slightly higher certainty
    anchor_repetition_threshold=0.40,
    anchor_phase_threshold=0.50,
    anchor_min_bars=1,
    anchor_recovery_certainty_threshold=0.55,
    anchor_recovery_phase_threshold=0.65,
    anchor_recovery_stability_threshold=0.65,
    anchor_recovery_max_density=0.70,
    feature_bail_silence_seconds=1.50,
    feature_hysteresis_margin=0.10,
    # DROP thresholds
    drop_density_threshold=0.70,
    drop_silence_min_seconds=0.05,
    drop_silence_max_seconds=1.50,
    drop_change_threshold=0.04,
    drop_min_certainty_threshold=0.20,
    drop_phase_threshold=0.50,
    # FINAL_BAIL thresholds
    final_bail_change_threshold=0.20,
    final_bail_silence_min_seconds=0.25,
    final_bail_silence_max_seconds=1.00,
    final_bail_min_certainty=0.45,
    final_bail_phase_threshold=0.55,
    final_bail_recent_strength_threshold=0.60,
)

_CAUTIOUS_PHRASE = PhraseMarkerConfig(
    enabled=True,
    # Higher confidence thresholds → fewer phrase markers
    eight_bar_min_confidence=0.60,     # 0.45 → 0.60
    sixteen_bar_min_confidence=0.75,   # 0.60 → 0.75
    # Tighter feature health requirements
    min_phase=0.70,        # 0.60 → 0.70
    min_certainty=0.70,    # 0.60 → 0.70
    min_stability=0.70,    # 0.60 → 0.70
    max_density=0.70,      # 0.75 → 0.70
    # Post-ANCHOR: longer grace period
    bars_after_anchor_grace=6,     # 4 → 6
    min_confidence_post_anchor=0.70,  # 0.60 → 0.70
    # Softer velocity for ear-perks
    ear_perk_kick_boost=5,              # 8 → 5
    ear_perk_ghost_snare_velocity=28,   # 35 → 28
    ear_perk_hat_lift_velocity=70,      # 85 → 70
    ear_perk_pickup_kick_velocity=65,   # 75 → 65
    # Softer frills
    frill_snare_pickup_velocity=80,     # 95 → 80
    frill_kick_velocity=85,             # 100 → 85
    frill_hat_flourish_velocity=70,     # 85 → 70
)

_CAUTIOUS_OUTPUT = OutputShapingConfig(
    # Less humanization → more machine-tight, less expression
    humanize_amount=0.6,
    # Reduce settings — more aggressive thinning
    reduce_min_snare_velocity=65,   # 60 → 65 (remove more snares)
    reduce_thin_hats=True,
    reduce_strip_ghosts=True,
    reduce_preserve_strong_beats=True,
    # Anchor settings
    anchor_strip_ghosts=True,
    anchor_strip_syncopated=True,
    anchor_simplify_hats=True,
    anchor_reduce_velocity_variation=True,
    anchor_target_velocity=95,      # 100 → 95
    # Build settings — less assertive
    build_velocity_boost=6,         # 12 → 6
    build_max_velocity=127,
    build_open_hats=True,
    # Enter settings — softer, more gentle
    enter_velocity_cap=85,          # 100 → 85
    enter_soft_scale=0.75,          # 0.85 → 0.75
)

CAUTIOUS_PRESET = DrummerPresetConfig(
    name="Cautious Bunny",
    profile=_CAUTIOUS_PROFILE,
    phrase_config=_CAUTIOUS_PHRASE,
    output_config=_CAUTIOUS_OUTPUT,
)

# ============================================================================
# Braver preset — more confident, firmer, still musical
# ============================================================================

_BRAVER_PROFILE = DrummerProfile(
    name="Braver Bunny",
    # Base thresholds
    hysteresis_margin=0.10,
    bail_silence_seconds=0.50,
    density_inversion_threshold=0.75,
    fill_probability_base=0.05,
    energy_ema_alpha=0.10,
    density_ema_alpha=0.10,
    # Stage 2 — more confident entry
    min_pulse_confidence=0.70,
    min_bar_confidence=0.65,
    full_entry_confidence=0.80,
    soft_entry_confidence=0.70,
    min_observation_seconds=1.00,   # shorter observation window
    severe_uncertainty_threshold=0.35,
    maintain_hysteresis_margin=0.10,
    # Stage 3 — faster builds, more assertive
    fast_energy_ema_alpha=0.15,
    slow_energy_ema_alpha=0.02,
    build_trend_threshold=0.12,     # needs less trend to build
    reduce_trend_threshold=-0.10,
    drop_trend_threshold=-0.30,
    max_density_for_build=0.85,     # tolerate more density before blocking build
    low_energy_threshold_for_drop=0.25,
    density_collapse_ratio_for_drop=0.35,
    min_build_duration_seconds=2.0,
    min_reduce_duration_seconds=2.0,
    min_drop_duration_seconds=1.0,
    # Stage 4 — feature-driven thresholds
    enter_certainty_threshold=0.55,    # less certainty needed to enter
    enter_repetition_threshold=0.60,
    enter_confirmation_snapshots=2,    # fewer confirmations before entry
    build_change_threshold=0.15,       # needs less change to build
    build_certainty_threshold=0.50,
    build_repetition_threshold=0.55,
    build_phase_threshold=0.50,
    build_max_density_without_phrase=0.85,
    reduce_density_threshold=0.80,     # reduce later (higher threshold)
    anchor_certainty_threshold=0.35,   # anchor at lower certainty (more tolerant)
    anchor_repetition_threshold=0.30,
    anchor_phase_threshold=0.40,
    anchor_min_bars=1,
    anchor_recovery_certainty_threshold=0.45,
    anchor_recovery_phase_threshold=0.55,
    anchor_recovery_stability_threshold=0.55,
    anchor_recovery_max_density=0.80,
    feature_bail_silence_seconds=1.50,
    feature_hysteresis_margin=0.10,
    # DROP thresholds
    drop_density_threshold=0.70,
    drop_silence_min_seconds=0.05,
    drop_silence_max_seconds=1.50,
    drop_change_threshold=0.04,
    drop_min_certainty_threshold=0.20,
    drop_phase_threshold=0.50,
    # FINAL_BAIL thresholds
    final_bail_change_threshold=0.20,
    final_bail_silence_min_seconds=0.25,
    final_bail_silence_max_seconds=1.00,
    final_bail_min_certainty=0.45,
    final_bail_phase_threshold=0.55,
    final_bail_recent_strength_threshold=0.60,
)

_BRAVER_PHRASE = PhraseMarkerConfig(
    enabled=True,
    # Lower confidence thresholds → more frequent phrase markers
    eight_bar_min_confidence=0.30,     # 0.45 → 0.30
    sixteen_bar_min_confidence=0.45,   # 0.60 → 0.45
    # Looser feature health requirements
    min_phase=0.50,        # 0.60 → 0.50
    min_certainty=0.50,    # 0.60 → 0.50
    min_stability=0.50,    # 0.60 → 0.50
    max_density=0.80,      # 0.75 → 0.80
    # Post-ANCHOR: shorter grace period
    bars_after_anchor_grace=2,     # 4 → 2
    min_confidence_post_anchor=0.50,  # 0.60 → 0.50
    # Bolder velocity for ear-perks
    ear_perk_kick_boost=10,             # 8 → 10
    ear_perk_ghost_snare_velocity=42,   # 35 → 42
    ear_perk_hat_lift_velocity=95,      # 85 → 95
    ear_perk_pickup_kick_velocity=85,   # 75 → 85
    # More assertive frills
    frill_snare_pickup_velocity=105,    # 95 → 105
    frill_kick_velocity=110,            # 100 → 110
    frill_hat_flourish_velocity=95,     # 85 → 95
)

_BRAVER_OUTPUT = OutputShapingConfig(
    # Full humanization
    humanize_amount=1.0,
    # Reduce settings — less aggressive thinning
    reduce_min_snare_velocity=55,   # 60 → 55 (preserve more snares)
    reduce_thin_hats=True,
    reduce_strip_ghosts=True,
    reduce_preserve_strong_beats=True,
    # Anchor settings
    anchor_strip_ghosts=True,
    anchor_strip_syncopated=True,
    anchor_simplify_hats=True,
    anchor_reduce_velocity_variation=True,
    anchor_target_velocity=105,     # 100 → 105
    # Build settings — more assertive
    build_velocity_boost=18,        # 12 → 18
    build_max_velocity=127,
    build_open_hats=True,
    # Enter settings — more confident
    enter_velocity_cap=115,         # 100 → 115
    enter_soft_scale=0.95,          # 0.85 → 0.95
)

BRAVER_PRESET = DrummerPresetConfig(
    name="Braver Bunny",
    profile=_BRAVER_PROFILE,
    phrase_config=_BRAVER_PHRASE,
    output_config=_BRAVER_OUTPUT,
)

# ============================================================================
# Preset lookup table
# ============================================================================

_PRESET_REGISTRY: dict[DrummerPreset, DrummerPresetConfig] = {
    DrummerPreset.CAUTIOUS: CAUTIOUS_PRESET,
    DrummerPreset.NORMAL: NORMAL_PRESET,
    DrummerPreset.BRAVER: BRAVER_PRESET,
}


def get_drummer_preset(name: str) -> DrummerPresetConfig:
    """Look up a drummer preset by name.

    Parameters
    ----------
    name : str
        One of ``"cautious"``, ``"normal"``, ``"braver"``.

    Returns
    -------
    DrummerPresetConfig
        The complete configuration bundle for this temperament.

    Raises
    ------
    ValueError
        If the name is not a recognised preset.
    """
    preset = DrummerPreset.from_name(name)
    return _PRESET_REGISTRY[preset]


def list_drummer_presets() -> list[str]:
    """Return all valid preset names (as strings)."""
    return DrummerPreset.list_names()