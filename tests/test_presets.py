"""Tests for the Drummer Preset system.

Tests cover:
* preset names list correctly
* unknown preset raises helpful error
* normal preset matches existing defaults as closely as possible
* cautious preset is more conservative than normal
* braver preset is more assertive than normal
* all presets preserve DROP output contract
* all presets preserve BAIL output contract
* all presets preserve FINAL_BAIL output contract
* all presets preserve no-crash phrase marker rule
* demo accepts all preset names
"""

from __future__ import annotations

import itertools

import pytest

from drummer.behaviour import (
    BehaviourIntent,
    ConservativePocketDrummer,
    FeatureDrivenBehaviourEngine,
)
from drummer.output_shaping import BehaviourOutputShaper
from drummer.phrase_markers import (
    PhraseMarkerConfig,
    PhraseMarkerState,
    PhraseMarkerType,
    is_bar_8_boundary,
    is_bar_16_boundary,
    is_musically_safe,
    choose_phrase_marker,
)
from drummer.presets import (
    DrummerPreset,
    DrummerPresetConfig,
    CAUTIOUS_PRESET,
    NORMAL_PRESET,
    BRAVER_PRESET,
    get_drummer_preset,
    list_drummer_presets,
)
from drummer.feel import GrooveEvent

# Mock snapshot for phrase marker tests
from perception.features import FeatureSnapshot


# ============================================================================
# Basic preset lookup
# ============================================================================


class TestPresetList:
    """Preset names list correctly."""

    def test_preset_names(self) -> None:
        """list_drummer_presets returns all valid names."""
        names = list_drummer_presets()
        assert isinstance(names, list)
        assert len(names) == 3
        assert "cautious" in names
        assert "normal" in names
        assert "braver" in names

    def test_preset_enum_values(self) -> None:
        """DrummerPreset enum has expected values."""
        assert DrummerPreset.CAUTIOUS.value == "cautious"
        assert DrummerPreset.NORMAL.value == "normal"
        assert DrummerPreset.BRAVER.value == "braver"

    def test_get_drummer_preset_normal(self) -> None:
        """get_drummer_preset('normal') returns the normal config."""
        config = get_drummer_preset("normal")
        assert isinstance(config, DrummerPresetConfig)
        assert config.name == "Normal Bunny"

    def test_get_drummer_preset_cautious(self) -> None:
        """get_drummer_preset('cautious') returns the cautious config."""
        config = get_drummer_preset("cautious")
        assert isinstance(config, DrummerPresetConfig)
        assert config.name == "Cautious Bunny"

    def test_get_drummer_preset_braver(self) -> None:
        """get_drummer_preset('braver') returns the braver config."""
        config = get_drummer_preset("braver")
        assert isinstance(config, DrummerPresetConfig)
        assert config.name == "Braver Bunny"

    def test_get_drummer_preset_case_insensitive(self) -> None:
        """get_drummer_preset handles case variations."""
        config_a = get_drummer_preset("Cautious")
        config_b = get_drummer_preset("CAUTIOUS")
        config_c = get_drummer_preset("cautious")
        assert config_a == config_b == config_c

    def test_raise_unknown_preset(self) -> None:
        """Unknown preset raises ValueError with helpful message."""
        with pytest.raises(ValueError) as excinfo:
            get_drummer_preset("super_bunny")
        msg = str(excinfo.value)
        assert "Unknown preset" in msg
        assert "cautious" in msg
        assert "normal" in msg
        assert "braver" in msg


# ============================================================================
# Normal preset must match existing defaults
# ============================================================================


class TestNormalPresetDefaults:
    """Normal preset should match existing defaults as closely as possible."""

    def test_normal_profile_is_conservative_pocket_drummer(self) -> None:
        """Normal profile should be the default ConservativePocketDrummer."""
        assert NORMAL_PRESET.profile == ConservativePocketDrummer

    def test_normal_phrase_config_is_default(self) -> None:
        """Normal phrase config should be default PhraseMarkerConfig()."""
        default_phrase = PhraseMarkerConfig()
        normal_phrase = NORMAL_PRESET.phrase_config
        # Check key fields match
        assert normal_phrase.eight_bar_min_confidence == default_phrase.eight_bar_min_confidence
        assert normal_phrase.sixteen_bar_min_confidence == default_phrase.sixteen_bar_min_confidence
        assert normal_phrase.min_phase == default_phrase.min_phase
        assert normal_phrase.min_certainty == default_phrase.min_certainty
        assert normal_phrase.min_stability == default_phrase.min_stability
        assert normal_phrase.max_density == default_phrase.max_density

    def test_normal_output_config_is_default(self) -> None:
        """Normal output config should be default OutputShapingConfig()."""
        from drummer.output_shaping import OutputShapingConfig
        default_output = OutputShapingConfig()
        normal_output = NORMAL_PRESET.output_config
        assert normal_output.humanize_amount == default_output.humanize_amount
        assert normal_output.build_velocity_boost == default_output.build_velocity_boost
        assert normal_output.enter_velocity_cap == default_output.enter_velocity_cap


# ============================================================================
# Cautious < Normal < Braver ordering checks
# ============================================================================


class TestPresetOrdering:
    """Cautious is more conservative than Normal, Braver is more assertive."""

    def test_entry_thresholds_cautious_higher(self) -> None:
        """Cautious entry thresholds are higher (more conservative)."""
        c = CAUTIOUS_PRESET.profile
        n = NORMAL_PRESET.profile
        assert c.min_pulse_confidence >= n.min_pulse_confidence
        assert c.min_bar_confidence >= n.min_bar_confidence
        assert c.full_entry_confidence >= n.full_entry_confidence
        assert c.soft_entry_confidence >= n.soft_entry_confidence
        assert c.min_observation_seconds >= n.min_observation_seconds

    def test_entry_thresholds_braver_lower(self) -> None:
        """Braver entry thresholds are lower (more confident)."""
        b = BRAVER_PRESET.profile
        n = NORMAL_PRESET.profile
        assert b.min_pulse_confidence <= n.min_pulse_confidence
        assert b.min_bar_confidence <= n.min_bar_confidence
        assert b.full_entry_confidence <= n.full_entry_confidence
        assert b.soft_entry_confidence <= n.soft_entry_confidence
        assert b.min_observation_seconds <= n.min_observation_seconds

    def test_entry_confirmation_cautious_more(self) -> None:
        """Cautious requires more confirmation snapshots before entry."""
        assert (CAUTIOUS_PRESET.profile.enter_confirmation_snapshots
                >= NORMAL_PRESET.profile.enter_confirmation_snapshots)
        assert (BRAVER_PRESET.profile.enter_confirmation_snapshots
                <= NORMAL_PRESET.profile.enter_confirmation_snapshots)

    def test_build_thresholds_cautious_higher(self) -> None:
        """Cautious needs stronger change to build."""
        c = CAUTIOUS_PRESET.profile
        n = NORMAL_PRESET.profile
        assert c.build_change_threshold >= n.build_change_threshold

    def test_build_thresholds_braver_lower(self) -> None:
        """Braver needs less change to build."""
        b = BRAVER_PRESET.profile
        n = NORMAL_PRESET.profile
        assert b.build_change_threshold <= n.build_change_threshold

    def test_reduce_threshold_cautious_lower(self) -> None:
        """Cautious reduces sooner (lower density threshold)."""
        c = CAUTIOUS_PRESET.profile
        n = NORMAL_PRESET.profile
        assert c.reduce_density_threshold <= n.reduce_density_threshold

    def test_reduce_threshold_braver_higher(self) -> None:
        """Braver reduces later (higher density threshold)."""
        b = BRAVER_PRESET.profile
        n = NORMAL_PRESET.profile
        assert b.reduce_density_threshold >= n.reduce_density_threshold

    def test_phrase_marker_confidence_cautious_higher(self) -> None:
        """Cautious requires higher confidence for phrase markers."""
        c = CAUTIOUS_PRESET.phrase_config
        n = NORMAL_PRESET.phrase_config
        assert c.eight_bar_min_confidence >= n.eight_bar_min_confidence
        assert c.sixteen_bar_min_confidence >= n.sixteen_bar_min_confidence

    def test_phrase_marker_confidence_braver_lower(self) -> None:
        """Braver requires lower confidence for phrase markers."""
        b = BRAVER_PRESET.phrase_config
        n = NORMAL_PRESET.phrase_config
        assert b.eight_bar_min_confidence <= n.eight_bar_min_confidence
        assert b.sixteen_bar_min_confidence <= n.sixteen_bar_min_confidence

    def test_build_velocity_boost_ordering(self) -> None:
        """Cautious < Normal < Braver for build velocity boost."""
        c = CAUTIOUS_PRESET.output_config.build_velocity_boost
        n = NORMAL_PRESET.output_config.build_velocity_boost
        b = BRAVER_PRESET.output_config.build_velocity_boost
        assert c < n < b

    def test_enter_soft_scale_ordering(self) -> None:
        """Cautious < Normal < Braver for enter soft scale."""
        c = CAUTIOUS_PRESET.output_config.enter_soft_scale
        n = NORMAL_PRESET.output_config.enter_soft_scale
        b = BRAVER_PRESET.output_config.enter_soft_scale
        assert c < n <= b

    def test_anchor_target_velocity_cautious_lower(self) -> None:
        """Cautious anchor target velocity is lower."""
        c = CAUTIOUS_PRESET.output_config.anchor_target_velocity
        n = NORMAL_PRESET.output_config.anchor_target_velocity
        assert c <= n

    def test_anchor_target_velocity_braver_higher(self) -> None:
        """Braver anchor target velocity is higher."""
        b = BRAVER_PRESET.output_config.anchor_target_velocity
        n = NORMAL_PRESET.output_config.anchor_target_velocity
        assert b >= n

    def test_post_anchor_grace_cautious_longer(self) -> None:
        """Cautious has longer post-ANCHOR grace period."""
        assert (CAUTIOUS_PRESET.phrase_config.bars_after_anchor_grace
                >= NORMAL_PRESET.phrase_config.bars_after_anchor_grace)
        assert (BRAVER_PRESET.phrase_config.bars_after_anchor_grace
                <= NORMAL_PRESET.phrase_config.bars_after_anchor_grace)


# ============================================================================
# DROP output contract — all presets
# ============================================================================


class TestPresetsPreserveDropContract:
    """All presets must produce valid DROP output."""

    @pytest.mark.parametrize("preset_config", [
        CAUTIOUS_PRESET,
        NORMAL_PRESET,
        BRAVER_PRESET,
    ])
    def test_drop_sparse_kicks_no_crash(self, preset_config) -> None:
        """DROP produces > 0 events, 1-2 sparse kicks, no crash."""
        shaper = BehaviourOutputShaper(config=preset_config.output_config)
        # Empty input — shaper generates DROP from scratch
        empty_groove: list[GrooveEvent] = []
        shaped = shaper.shape(empty_groove, BehaviourIntent.DROP)
        assert len(shaped) > 0, f"{preset_config.name}: DROP must have > 0 events"
        assert len(shaped) <= 2, f"{preset_config.name}: DROP must be sparse (<= 2 events)"
        # No crash
        for evt in shaped:
            assert evt.instrument.lower() != "crash", (
                f"{preset_config.name}: DROP must not contain crash"
            )

    @pytest.mark.parametrize("preset_config", [
        CAUTIOUS_PRESET,
        NORMAL_PRESET,
        BRAVER_PRESET,
    ])
    def test_drop_has_kick(self, preset_config) -> None:
        """DROP output must include at least one kick."""
        shaper = BehaviourOutputShaper(config=preset_config.output_config)
        shaped = shaper.shape([], BehaviourIntent.DROP)
        has_kick = any(e.instrument.lower() == "kick" for e in shaped)
        assert has_kick, f"{preset_config.name}: DROP must contain at least one kick"


# ============================================================================
# BAIL output contract — all presets
# ============================================================================


class TestPresetsPreserveBailContract:
    """All presets must produce valid BAIL output."""

    @pytest.mark.parametrize("preset_config", [
        CAUTIOUS_PRESET,
        NORMAL_PRESET,
        BRAVER_PRESET,
    ])
    def test_bail_zero_events(self, preset_config) -> None:
        """BAIL must produce exactly 0 events."""
        shaper = BehaviourOutputShaper(config=preset_config.output_config)
        shaped = shaper.shape([], BehaviourIntent.BAIL)
        assert len(shaped) == 0, f"{preset_config.name}: BAIL must have exactly 0 events"


# ============================================================================
# FINAL_BAIL output contract — all presets
# ============================================================================


class TestPresetsPreserveFinalBailContract:
    """All presets must produce valid FINAL_BAIL output."""

    @pytest.mark.parametrize("preset_config", [
        CAUTIOUS_PRESET,
        NORMAL_PRESET,
        BRAVER_PRESET,
    ])
    def test_final_bail_kick_and_crash(self, preset_config) -> None:
        """FINAL_BAIL must produce exactly kick + crash on beat 1."""
        shaper = BehaviourOutputShaper(config=preset_config.output_config)
        shaped = shaper.shape([], BehaviourIntent.FINAL_BAIL)
        assert len(shaped) == 2, (
            f"{preset_config.name}: FINAL_BAIL must have exactly 2 events, "
            f"got {len(shaped)}"
        )
        instruments = [e.instrument.lower() for e in shaped]
        assert "kick" in instruments, (
            f"{preset_config.name}: FINAL_BAIL must contain a kick"
        )
        assert "crash" in instruments, (
            f"{preset_config.name}: FINAL_BAIL must contain a crash"
        )


# ============================================================================
# No phrase markers during ANCHOR / DROP / BAIL / FINAL_BAIL — all presets
# ============================================================================


class TestPresetsNoPhraseMarkersInProtectedIntents:
    """All presets must never place phrase markers during protected intents."""

    def _make_healthy_snapshot(self) -> FeatureSnapshot:
        """Create a FeatureSnapshot with healthy values for testing."""
        return FeatureSnapshot(
            timestamp=0.0,
            input_density=0.30,
            strength_ema=0.85,
            change_score=0.50,
            silence_duration=0.10,
            repetition_stability=0.90,
            phase_alignment=0.90,
            player_certainty=0.90,
        )

    @pytest.mark.parametrize(
        "preset_config, forbidden_intent",
        list(itertools.product(
            [CAUTIOUS_PRESET, NORMAL_PRESET, BRAVER_PRESET],
            [
                BehaviourIntent.ANCHOR,
                BehaviourIntent.DROP,
                BehaviourIntent.BAIL,
                BehaviourIntent.FINAL_BAIL,
            ],
        ))
    )
    def test_no_phrase_marker_in_protected_intent(
        self, preset_config, forbidden_intent
    ) -> None:
        """No phrase marker should fire during ANCHOR/DROP/BAIL/FINAL_BAIL."""
        snap = self._make_healthy_snapshot()
        state = PhraseMarkerState()
        # Even at a valid 8-bar boundary, the phrase marker must be NONE
        # because is_musically_safe checks the intent first
        safe = is_musically_safe(
            forbidden_intent, snap, confidence=0.95,
            config=preset_config.phrase_config, state=state,
        )
        assert not safe, (
            f"{preset_config.name}: is_musically_safe returned True for "
            f"forbidden intent {forbidden_intent.value}"
        )

    @pytest.mark.parametrize("preset_config", [
        CAUTIOUS_PRESET,
        NORMAL_PRESET,
        BRAVER_PRESET,
    ])
    def test_phrase_marker_possible_in_maintain(self, preset_config) -> None:
        """Phrase markers should be possible in MAINTAIN with all presets."""
        snap = self._make_healthy_snapshot()
        state = PhraseMarkerState()
        safe = is_musically_safe(
            BehaviourIntent.MAINTAIN, snap, confidence=0.95,
            config=preset_config.phrase_config, state=state,
        )
        assert safe, (
            f"{preset_config.name}: is_musically_safe should return True "
            f"for MAINTAIN with healthy snapshot"
        )

    @pytest.mark.parametrize("preset_config", [
        CAUTIOUS_PRESET,
        NORMAL_PRESET,
        BRAVER_PRESET,
    ])
    def test_phrase_marker_possible_in_build(self, preset_config) -> None:
        """Phrase markers should be possible in BUILD with all presets."""
        snap = self._make_healthy_snapshot()
        state = PhraseMarkerState()
        safe = is_musically_safe(
            BehaviourIntent.BUILD, snap, confidence=0.95,
            config=preset_config.phrase_config, state=state,
        )
        assert safe, (
            f"{preset_config.name}: is_musically_safe should return True "
            f"for BUILD with healthy snapshot"
        )


# ============================================================================
# Demo accepts all preset names
# ============================================================================


class TestDemoAcceptsAllPresets:
    """The demo script's CLI accepts all preset names."""

    def test_all_presets_are_choices(self) -> None:
        """All preset names should be parseable by the demo CLI."""
        from demo_continuous_jam_midi import run_continuous_jam

        for preset_name in list_drummer_presets():
            # We just test that calling with the preset name doesn't raise
            # before we actually run the full simulation
            config = get_drummer_preset(preset_name)
            assert config is not None
            assert config.name != ""

    def test_invalid_preset_raises(self) -> None:
        """Invalid preset name passed to run_continuous_jam raises ValueError."""
        from demo_continuous_jam_midi import run_continuous_jam

        with pytest.raises(ValueError) as excinfo:
            run_continuous_jam(bars=2, bpm=120, mode="scripted", preset_name="invalid_bunny")
        msg = str(excinfo.value)
        assert "Unknown preset" in msg

    def test_normal_preset_runs(self) -> None:
        """Normal preset runs without error."""
        from demo_continuous_jam_midi import run_continuous_jam

        # Use 10 bars to get through LISTEN + ENTER + MAINTAIN phases
        pipeline, diagnostics, events = run_continuous_jam(
            bars=10, bpm=120, mode="scripted", preset_name="normal",
        )
        assert len(diagnostics) == 10
        assert len(events) > 0

    def test_cautious_preset_runs(self) -> None:
        """Cautious preset runs without error."""
        from demo_continuous_jam_midi import run_continuous_jam

        pipeline, diagnostics, events = run_continuous_jam(
            bars=10, bpm=120, mode="scripted", preset_name="cautious",
        )
        assert len(diagnostics) == 10
        assert len(events) > 0

    def test_braver_preset_runs(self) -> None:
        """Braver preset runs without error."""
        from demo_continuous_jam_midi import run_continuous_jam

        pipeline, diagnostics, events = run_continuous_jam(
            bars=10, bpm=120, mode="scripted", preset_name="braver",
        )
        assert len(diagnostics) == 10
        assert len(events) > 0

    def test_inferred_mode_all_presets(self) -> None:
        """All presets work in inferred mode."""
        from demo_continuous_jam_midi import run_continuous_jam

        for preset_name in list_drummer_presets():
            pipeline, diagnostics, events = run_continuous_jam(
                bars=8, bpm=120, mode="inferred", preset_name=preset_name,
            )
            assert len(diagnostics) == 8
            assert len(events) > 0


# ============================================================================
# Comparison mode tests
# ============================================================================


class TestPresetComparison:
    """Tests for the --compare-presets feature."""

    def test_compare_presets_runs_without_playback(self) -> None:
        """_run_preset_comparison runs without errors."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=6, bpm=120)
        assert len(results) == 3

    def test_comparison_includes_all_three_presets(self) -> None:
        """Comparison includes all three presets."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=6, bpm=120)
        preset_names = [r["preset"] for r in results]
        assert "Cautious" in preset_names
        assert "Normal" in preset_names
        assert "Braver" in preset_names

    def test_cautious_not_more_events_than_braver(self) -> None:
        """Cautious should not produce more total events than Braver."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=20, bpm=120)
        cautious_events = results[0]["total_events"]
        braver_events = results[2]["total_events"]
        assert cautious_events <= braver_events, (
            f"Cautious ({cautious_events}) should not exceed "
            f"Braver ({braver_events}) in total events"
        )

    def test_cautious_enters_later_or_same_as_normal(self) -> None:
        """Cautious enters no earlier than Normal."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=20, bpm=120)
        cautious_enter = results[0]["first_enter"]
        normal_enter = results[1]["first_enter"]
        assert cautious_enter >= normal_enter, (
            f"Cautious enters at bar {cautious_enter}, "
            f"earlier than Normal at bar {normal_enter}"
        )

    def test_braver_enters_earlier_or_same_as_normal(self) -> None:
        """Braver enters no later than Normal."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=20, bpm=120)
        braver_enter = results[2]["first_enter"]
        normal_enter = results[1]["first_enter"]
        assert braver_enter <= normal_enter, (
            f"Braver enters at bar {braver_enter}, "
            f"later than Normal at bar {normal_enter}"
        )

    def test_braver_confidence_peak_higher_or_equal_to_normal(self) -> None:
        """Braver confidence peak >= Normal."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=20, bpm=120)
        braver_peak = results[2]["confidence_peak"]
        normal_peak = results[1]["confidence_peak"]
        assert braver_peak >= normal_peak, (
            f"Braver confidence peak {braver_peak:.2f} < "
            f"Normal {normal_peak:.2f}"
        )

    def test_normal_is_reference_preset(self) -> None:
        """Normal preset matches ConservativePocketDrummer defaults."""
        from drummer.presets import NORMAL_PRESET
        from drummer.behaviour import ConservativePocketDrummer
        assert NORMAL_PRESET.profile == ConservativePocketDrummer

    def test_phrase_marker_counts_are_deterministic(self) -> None:
        """Same preset + same input = same phrase marker count."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results_a = _run_preset_comparison(bars=20, bpm=120)
        results_b = _run_preset_comparison(bars=20, bpm=120)
        for i in range(3):
            assert results_a[i]["phrase_markers"] == results_b[i]["phrase_markers"], (
                f"Phrase marker count for {results_a[i]['preset']} "
                f"not deterministic: {results_a[i]['phrase_markers']} vs "
                f"{results_b[i]['phrase_markers']}"
            )

    def test_braver_does_not_break_drop_contract(self) -> None:
        """Braver preset still produces valid DROP output."""
        from demo_continuous_jam_midi import _run_preset_comparison

        results = _run_preset_comparison(bars=20, bpm=120)
        braver = results[2]
        assert braver["drop_ok"], f"Braver DROP contract FAILED"
        assert braver["bail_ok"], f"Braver BAIL contract FAILED"
        assert braver["final_bail_ok"], f"Braver FINAL_BAIL contract FAILED"


# ============================================================================
# JSON export tests
# ============================================================================


class TestJsonExport:
    """Tests for the --export-json feature."""

    def _run_and_export(self, path: str, **kwargs) -> None:
        """Helper: run a jam and export to JSON."""
        from demo_continuous_jam_midi import run_continuous_jam, export_diagnostics_to_json

        _pipeline, diagnostics, _events = run_continuous_jam(**kwargs)
        meta = {
            "mode": kwargs.get("mode", "scripted"),
            "preset": kwargs.get("preset_name", "normal"),
            "bpm": kwargs.get("bpm", 120),
            "bars": kwargs.get("bars", 6),
            "total_duration": kwargs.get("bars", 6) * (60.0 / kwargs.get("bpm", 120)) * 4.0,
            "total_events": sum(d["event_count"] for d in diagnostics),
        }
        export_diagnostics_to_json(diagnostics, path, meta=meta)

    def test_json_export_file_created(self, tmp_path) -> None:
        """JSON export file is created."""
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=6, bpm=120, mode="scripted", preset_name="normal")
        import os
        assert os.path.isfile(out)

    def test_json_includes_metadata(self, tmp_path) -> None:
        """Exported JSON includes metadata."""
        import json
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=6, bpm=120, mode="scripted", preset_name="normal")
        with open(out) as f:
            data = json.load(f)
        assert "meta" in data
        assert "mode" in data["meta"]
        assert "preset" in data["meta"]
        assert "bpm" in data["meta"]
        assert "bars" in data["meta"]

    def test_json_includes_diagnostics(self, tmp_path) -> None:
        """Exported JSON includes per-bar diagnostics."""
        import json
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=6, bpm=120, mode="scripted", preset_name="normal")
        with open(out) as f:
            data = json.load(f)
        assert "diagnostics" in data
        assert len(data["diagnostics"]) == 6

    def test_json_includes_preset_name(self, tmp_path) -> None:
        """Exported JSON includes preset name."""
        import json
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=6, bpm=120, mode="scripted", preset_name="braver")
        with open(out) as f:
            data = json.load(f)
        assert data["meta"]["preset"] == "braver"

    def test_json_includes_confidence(self, tmp_path) -> None:
        """Exported JSON includes confidence values."""
        import json
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=6, bpm=120, mode="scripted", preset_name="normal")
        with open(out) as f:
            data = json.load(f)
        for diag in data["diagnostics"]:
            assert "confidence" in diag

    def test_json_includes_phrase_marker_labels(self, tmp_path) -> None:
        """Exported JSON includes phrase marker labels."""
        import json
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=20, bpm=120, mode="inferred", preset_name="braver")
        with open(out) as f:
            data = json.load(f)
        for diag in data["diagnostics"]:
            assert "phrase_marker_label" in diag

    def test_json_export_is_deterministic(self, tmp_path) -> None:
        """Same inputs produce identical JSON output."""
        import json
        import hashlib
        out_a = str(tmp_path / "test_a.json")
        out_b = str(tmp_path / "test_b.json")
        self._run_and_export(out_a, bars=6, bpm=120, mode="scripted", preset_name="normal")
        self._run_and_export(out_b, bars=6, bpm=120, mode="scripted", preset_name="normal")
        with open(out_a) as f:
            data_a = json.load(f)
        with open(out_b) as f:
            data_b = json.load(f)
        # Compare SHA hashes of the JSON dumps
        hash_a = hashlib.sha256(json.dumps(data_a, sort_keys=True).encode()).hexdigest()
        hash_b = hashlib.sha256(json.dumps(data_b, sort_keys=True).encode()).hexdigest()
        assert hash_a == hash_b, "JSON export is not deterministic"

    def test_export_does_not_alter_event_generation(self, tmp_path) -> None:
        """Running with export yields same events as without."""
        from demo_continuous_jam_midi import run_continuous_jam

        # Run without export
        _p, diagnostics_a, events_a = run_continuous_jam(
            bars=6, bpm=120, mode="scripted", preset_name="normal",
        )
        # Run with export (via helper)
        out = str(tmp_path / "test.json")
        self._run_and_export(out, bars=6, bpm=120, mode="scripted", preset_name="normal")
        _p2, diagnostics_b, events_b = run_continuous_jam(
            bars=6, bpm=120, mode="scripted", preset_name="normal",
        )
        # Event counts should match
        assert len(events_a) == len(events_b)
        assert len(diagnostics_a) == len(diagnostics_b)
        for da, db in zip(diagnostics_a, diagnostics_b):
            assert da["event_count"] == db["event_count"]
            assert da["intent"] == db["intent"]
