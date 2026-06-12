"""Tests for the rule/model architecture.

Verifies:
1. Rule dataclasses import from ``drummer.rules``.
2. Preset models import from ``drummer.models``.
3. All dataclasses are frozen enough that top-level attribute assignment fails.
4. Each preset DrummerModel has all required fields.
5. Humanize and variation rules include kick, snare, and hat parameters.
6. All probability/threshold values are within sensible ranges.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------

from drummer.rules import (  # noqa: E402
    ConfidenceRules,
    DrummerModel,
    GrooveScoringRules,
    HumanizeRules,
    TransitionRules,
    VariationRules,
)

from drummer.models import (  # noqa: E402
    BUILTIN_MODELS,
    MOTORIK_TIGHT_MODEL,
    SIMPLE_ROCK_SAFE_MODEL,
    SPARSE_POSTPUNK_MODEL,
)


# ===================================================================
# 1.  Rule dataclasses import correctly
# ===================================================================


class TestRuleDataclassesImport:
    """Verify all rule dataclasses can be imported and instantiated."""

    def test_confidence_rules_imports(self) -> None:
        rules = ConfidenceRules()
        assert rules.timing_low == 0.4
        assert rules.timing_high == 0.7

    def test_transition_rules_imports(self) -> None:
        rules = TransitionRules()
        assert rules.min_hold_bars == 4

    def test_humanize_rules_imports(self) -> None:
        rules = HumanizeRules()
        assert rules.timing_amount_ms == 8.0
        assert rules.velocity_amount == 6

    def test_variation_rules_imports(self) -> None:
        rules = VariationRules()
        assert rules.variation_probability == 0.05
        assert not rules.allow_kick_variation

    def test_groove_scoring_rules_imports(self) -> None:
        rules = GrooveScoringRules()
        assert abs(rules.density_weight - 0.35) < 1e-6

    def test_drummer_model_imports(self) -> None:
        model = DrummerModel(id="test", name="Test", description="A test model")
        assert model.id == "test"
        assert model.name == "Test"


# ===================================================================
# 2.  Preset models import correctly
# ===================================================================


class TestPresetModelsImport:
    """Verify the three preset model constants are available."""

    def test_simple_rock_safe_imports(self) -> None:
        assert SIMPLE_ROCK_SAFE_MODEL.id == "simple_rock_safe"

    def test_motorik_tight_imports(self) -> None:
        assert MOTORIK_TIGHT_MODEL.id == "motorik_tight"

    def test_sparse_postpunk_imports(self) -> None:
        assert SPARSE_POSTPUNK_MODEL.id == "sparse_postpunk"

    def test_builtin_registry_exists(self) -> None:
        assert "simple_rock_safe" in BUILTIN_MODELS
        assert "motorik_tight" in BUILTIN_MODELS
        assert "sparse_postpunk" in BUILTIN_MODELS


# ===================================================================
# 3.  Frozen dataclass enforcement
# ===================================================================


class TestDataclassesAreFrozen:
    """Top-level attribute assignment on frozen dataclasses should fail."""

    def test_confidence_rules_is_frozen(self) -> None:
        r = ConfidenceRules()
        with pytest.raises(AttributeError):
            r.timing_low = 0.99  # type: ignore[misc]

    def test_transition_rules_is_frozen(self) -> None:
        r = TransitionRules()
        with pytest.raises(AttributeError):
            r.min_hold_bars = 99  # type: ignore[misc]

    def test_humanize_rules_is_frozen(self) -> None:
        r = HumanizeRules()
        with pytest.raises(AttributeError):
            r.timing_amount_ms = 99.0  # type: ignore[misc]

    def test_variation_rules_is_frozen(self) -> None:
        r = VariationRules()
        with pytest.raises(AttributeError):
            r.variation_probability = 0.99  # type: ignore[misc]

    def test_groove_scoring_rules_is_frozen(self) -> None:
        r = GrooveScoringRules()
        with pytest.raises(AttributeError):
            r.density_weight = 0.99  # type: ignore[misc]

    def test_drummer_model_is_frozen(self) -> None:
        m = DrummerModel(id="x", name="x", description="x")
        with pytest.raises(AttributeError):
            m.id = "y"  # type: ignore[misc]


# ===================================================================
# 4.  Each preset model has all required fields
# ===================================================================

PRESETS = [
    ("simple_rock_safe", SIMPLE_ROCK_SAFE_MODEL),
    ("motorik_tight", MOTORIK_TIGHT_MODEL),
    ("sparse_postpunk", SPARSE_POSTPUNK_MODEL),
]


class TestPresetModelFields:
    """Each preset DrummerModel must provide every required field."""

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_id(self, label: str, model: DrummerModel) -> None:
        assert model.id and isinstance(model.id, str)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_name(self, label: str, model: DrummerModel) -> None:
        assert model.name and isinstance(model.name, str)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_description(self, label: str, model: DrummerModel) -> None:
        assert model.description and isinstance(model.description, str)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_confidence_rules(self, label: str, model: DrummerModel) -> None:
        assert isinstance(model.confidence, ConfidenceRules)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_transition_rules(self, label: str, model: DrummerModel) -> None:
        assert isinstance(model.transition, TransitionRules)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_humanize_rules(self, label: str, model: DrummerModel) -> None:
        assert isinstance(model.humanize, HumanizeRules)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_variation_rules(self, label: str, model: DrummerModel) -> None:
        assert isinstance(model.variation, VariationRules)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_groove_scoring_rules(self, label: str, model: DrummerModel) -> None:
        assert isinstance(model.groove_scoring, GrooveScoringRules)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_preferred_groove_ids(self, label: str, model: DrummerModel) -> None:
        assert model.preferred_groove_ids is not None
        assert len(model.preferred_groove_ids) > 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_default_groove_id(self, label: str, model: DrummerModel) -> None:
        assert model.default_groove_id and isinstance(model.default_groove_id, str)

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_has_complexity_level(self, label: str, model: DrummerModel) -> None:
        assert 1 <= model.complexity_level <= 10


# ===================================================================
# 5.  Humanize rules include kick, snare, and hat parameters
# ===================================================================


class TestHumanizeRulesPerInstrument:
    """Every HumanizeRules must have per-instrument timing + velocity settings."""

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_kick_timing(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "kick" in h.timing_jitter_ms
        assert h.timing_jitter_ms["kick"] >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_kick_velocity(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "kick" in h.velocity_jitter
        assert h.velocity_jitter["kick"] >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_snare_timing(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "snare" in h.timing_jitter_ms
        assert h.timing_jitter_ms["snare"] >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_snare_velocity(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "snare" in h.velocity_jitter
        assert h.velocity_jitter["snare"] >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_hat_timing(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "hat" in h.timing_jitter_ms
        assert h.timing_jitter_ms["hat"] >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_hat_velocity(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "hat" in h.velocity_jitter
        assert h.velocity_jitter["hat"] >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_has_timing_bias(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert "kick" in h.timing_bias_ms
        assert "snare" in h.timing_bias_ms
        assert "hat" in h.timing_bias_ms


# ===================================================================
# 6.  Variation rules include kick, snare, and hat parameters
# ===================================================================


class TestVariationRulesPerInstrument:
    """Every VariationRules must reference kick, snare, and hat."""

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_variation_has_kick(self, label: str, model: DrummerModel) -> None:
        v = model.variation
        assert hasattr(v, "allow_kick_variation")

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_variation_has_snare(self, label: str, model: DrummerModel) -> None:
        v = model.variation
        assert hasattr(v, "allow_snare_variation")

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_variation_has_hat(self, label: str, model: DrummerModel) -> None:
        v = model.variation
        assert hasattr(v, "allow_hat_variation")


# ===================================================================
# 7.  All probability / threshold values are within sensible ranges
# ===================================================================


class TestProbabilityValuesAreInRange:
    """Verify floats are in [0, 1] and ints are non-negative."""

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_variation_probability_in_range(self, label: str, model: DrummerModel) -> None:
        assert 0.0 <= model.variation.variation_probability <= 1.0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_ghost_note_probability_in_range(self, label: str, model: DrummerModel) -> None:
        assert 0.0 <= model.variation.ghost_note_probability <= 1.0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_bar_end_variation_probability_in_range(self, label: str, model: DrummerModel) -> None:
        assert 0.0 <= model.variation.bar_end_variation_probability <= 1.0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_max_variations_non_negative(self, label: str, model: DrummerModel) -> None:
        assert model.variation.max_variations_per_bar >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_confidence_thresholds_in_range(self, label: str, model: DrummerModel) -> None:
        c = model.confidence
        assert 0.0 <= c.timing_low <= 1.0
        assert 0.0 <= c.timing_high <= 1.0
        assert 0.0 <= c.recovery_threshold <= 1.0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_humanize_amounts_non_negative(self, label: str, model: DrummerModel) -> None:
        h = model.humanize
        assert h.timing_amount_ms >= 0
        assert h.velocity_amount >= 0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_scoring_weights_non_negative_and_bounded(self, label: str, model: DrummerModel) -> None:
        s = model.groove_scoring
        assert s.density_weight >= 0
        assert s.energy_weight >= 0
        assert s.syncopation_weight >= 0
        assert s.strong_beat_bonus >= 0
        assert s.personality_bonus >= 0
        total_core = s.density_weight + s.energy_weight + s.syncopation_weight
        assert total_core < 1.0  # should leave room for bonuses
        assert s.strong_beat_bonus + s.personality_bonus < 1.0
        total_all = total_core + s.strong_beat_bonus + s.personality_bonus
        assert total_all <= 1.0

    @pytest.mark.parametrize("label,model", PRESETS, ids=[p[0] for p in PRESETS])
    def test_transition_bars_positive(self, label: str, model: DrummerModel) -> None:
        t = model.transition
        assert t.min_hold_bars > 0
        assert t.same_groove_cooldown_bars > 0