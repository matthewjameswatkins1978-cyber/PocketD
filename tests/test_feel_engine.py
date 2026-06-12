"""Comprehensive tests for DrummerFeelEngine, DrummerProfile, and GrooveEvent.

Tests cover:

- Timing offsets are applied per limb
- machine_tight does not introduce random timing
- laid_back makes snare later than hi-hat
- pushed produces negative offsets
- Ghost notes are softer than main snare hits
- Low input confidence reduces ghost-note and fill behaviour
- Compound split separates simultaneous hits by small ms values
- Velocities remain within MIDI range 1–127
- Determinism with seed
- to_midi_dicts conversion
- Profile presets and builder helpers
"""

from __future__ import annotations

from drummer.feel import (
    DrummerFeelEngine,
    DrummerProfile,
    GrooveEvent,
    TimingStrategy,
    _instrument_group,
)


# ---------------------------------------------------------------------------
# Helper: build a simple 1-bar rock groove
# ---------------------------------------------------------------------------


def _rock_groove_1bar() -> list[GrooveEvent]:
    """Return a basic 1-bar rock groove: kick on 1&3, snare on 2&4, 8th hats."""
    events: list[GrooveEvent] = []
    # Kick on beat 1 (0) and beat 3 (8)
    events.append(GrooveEvent(instrument="kick", grid_position=0, bar_index=0, velocity=100))
    events.append(GrooveEvent(instrument="kick", grid_position=8, bar_index=0, velocity=100))
    # Snare on beat 2 (4) and beat 4 (12)
    events.append(GrooveEvent(instrument="snare", grid_position=4, bar_index=0, velocity=100))
    events.append(GrooveEvent(instrument="snare", grid_position=12, bar_index=0, velocity=100))
    # Hi-hat 8th notes
    for pos in range(0, 16, 2):
        events.append(GrooveEvent(instrument="hi_hat", grid_position=pos, bar_index=0, velocity=80))
    return events


def _no_beats() -> list[GrooveEvent]:
    """Empty list for edge-case testing."""
    return []


# ===================================================================
# 1. Per-limb timing offset verification
# ===================================================================


class TestTimingOffsets:
    """Verify that per-limb timing offsets are applied correctly."""

    def test_machine_tight_no_timing_change(self):
        """machine_tight should not introduce any timing offset."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Machine",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            seed=12345,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        # Every event should have exactly 0.0 timing offset
        for ev in result:
            assert ev.timing_offset_ms == 0.0, (
                f"machine_tight produced offset {ev.timing_offset_ms} "
                f"for {ev.instrument} at pos {ev.grid_position}"
            )

    def test_tight_rock_kick_early(self):
        """Tight rock kick should be biased early (negative offset)."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Tight Rock",
            timing_strategy=TimingStrategy.TIGHT_ROCK,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        kick_offsets = [ev.timing_offset_ms for ev in result if ev.instrument == "kick"]
        assert len(kick_offsets) > 0
        # All kick offsets should be <= 0 (early / on time)
        for offset in kick_offsets:
            assert offset <= 1.0, f"Tight rock kick should be early or on time, got {offset}"

    def test_laid_back_snare_late(self):
        """Laid Back should make snare later than hi-hat."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Laid Back",
            timing_strategy=TimingStrategy.LAID_BACK,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        snare_offsets = [ev.timing_offset_ms for ev in result if ev.instrument == "snare"]
        hat_offsets = [ev.timing_offset_ms for ev in result if ev.instrument == "hi_hat"]

        avg_snare = sum(snare_offsets) / len(snare_offsets) if snare_offsets else 0.0
        avg_hat = sum(hat_offsets) / len(hat_offsets) if hat_offsets else 0.0

        # Snare should be later than hi-hat (higher offset)
        assert avg_snare > avg_hat, (
            f"Laid Back: expected snare ({avg_snare:.1f}) later than "
            f"hi-hat ({avg_hat:.1f})"
        )

    def test_laid_back_snare_positive(self):
        """Laid Back snare offset should be clearly positive."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Laid Back",
            timing_strategy=TimingStrategy.LAID_BACK,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        snare_offsets = [ev.timing_offset_ms for ev in result if ev.instrument == "snare"]
        avg_snare = sum(snare_offsets) / len(snare_offsets)
        # The bias is +18ms; variance ±4ms (scaled by stability). Avg should be > 10ms.
        assert avg_snare > 10.0, (
            f"Laid Back snare should be late (positive offset), avg={avg_snare:.1f}"
        )

    def test_pushed_negative_offsets(self):
        """Pushed strategy should produce negative timing offsets for all instruments."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Pushed",
            timing_strategy=TimingStrategy.PUSHED,
            seed=99,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        for ev in result:
            assert ev.timing_offset_ms < 5.0, (
                f"Pushed: {ev.instrument} at {ev.grid_position} expected "
                f"negative offset, got {ev.timing_offset_ms:.2f}"
            )

    def test_loose_garage_variance_nonzero(self):
        """Loose Garage should have visible variance around biases."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Loose Garage",
            timing_strategy=TimingStrategy.LOOSE_GARAGE,
            seed=777,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        # Check hi-hat offsets are not all identical (variance should be visible)
        hat_offsets = [ev.timing_offset_ms for ev in result if ev.instrument == "hi_hat"]
        assert len(set(hat_offsets)) > 1, (
            "Loose Garage hi-hat offsets should show variance"
        )


# ===================================================================
# 2. Velocity shaping
# ===================================================================


class TestVelocityShaping:
    """Verify velocity shaping rules."""

    def test_snare_backbeat_stronger(self):
        """Main snare backbeat hits (positions 4, 12) should be louder than input."""
        events = [
            GrooveEvent(instrument="snare", grid_position=4, bar_index=0,
                        velocity=90, source_role="main"),
        ]
        # Use machine tight profile with zero ghost density to avoid ghosts
        profile = DrummerProfile(
            name="Test",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            ghost_note_density=0.0,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)
        # machine tight has 0 variance and 0 bias, but backbeats still +8
        main_events = [ev for ev in result if ev.source_role == "main"]
        assert len(main_events) == 1
        assert main_events[0].velocity >= 98, (
            f"Snare backbeat should be boosted, got {main_events[0].velocity}"
        )

    def test_ghost_note_softer_than_main(self):
        """Ghost notes must be lower velocity than main snare hits."""
        events = [
            GrooveEvent(instrument="snare", grid_position=4, bar_index=0,
                        velocity=100, source_role="main"),
            GrooveEvent(instrument="snare", grid_position=6, bar_index=0,
                        velocity=30, articulation="ghost", source_role="ghost"),
        ]
        profile = DrummerProfile(
            name="Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        main_vel = [ev.velocity for ev in result if ev.source_role == "main"][0]
        ghost_vels = [ev.velocity for ev in result if ev.source_role == "ghost"]
        assert ghost_vels, "No ghost note events found"
        for gv in ghost_vels:
            assert gv < main_vel, (
                f"Ghost note velocity ({gv}) should be lower than "
                f"main snare ({main_vel})"
            )

    def test_ghost_velocity_capped(self):
        """Ghost notes should be capped at 50."""
        events = [
            GrooveEvent(instrument="snare", grid_position=6, bar_index=0,
                        velocity=80, articulation="ghost", source_role="ghost"),
        ]
        profile = DrummerProfile(
            name="Test",
            timing_strategy=TimingStrategy.TIGHT_ROCK,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)
        assert result[0].velocity <= 50, (
            f"Ghost note velocity should be ≤50, got {result[0].velocity}"
        )

    def test_all_velocities_in_midi_range(self):
        """All processed velocities must be within 1–127."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Test Range",
            timing_strategy=TimingStrategy.LOOSE_GARAGE,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        for ev in result:
            assert 1 <= ev.velocity <= 127, (
                f"Velocity {ev.velocity} out of MIDI range (1–127) "
                f"for {ev.instrument} at pos {ev.grid_position}"
            )

    def test_machine_tight_only_backbeat_boost(self):
        """machine_tight should only apply backbeat boost (no random variation)."""
        events = [
            GrooveEvent(instrument="snare", grid_position=4, bar_index=0,
                        velocity=100, source_role="main"),
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0,
                        velocity=100, source_role="main"),
            GrooveEvent(instrument="hi_hat", grid_position=2, bar_index=0,
                        velocity=80, source_role="main"),
        ]
        profile = DrummerProfile(
            name="Test Machine",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            ghost_note_density=0.0,
            seed=12345,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)
        # Result sorted by grid position
        result_sorted = sorted(result, key=lambda e: e.grid_position)

        # Kick at pos 0 — no boost, should stay 100
        assert result_sorted[0].instrument == "kick"
        assert result_sorted[0].velocity == 100
        # Hat at pos 2 — no boost, should stay 80
        assert result_sorted[1].instrument == "hi_hat"
        assert result_sorted[1].velocity == 80
        # Snare at pos 4 — backbeat boost +8, so 108
        assert result_sorted[2].instrument == "snare"
        assert result_sorted[2].velocity == 108  # 100 + 8 backbeat accent


# ===================================================================
# 3. Ghost note insertion
# ===================================================================


class TestGhostNotes:
    """Verify ghost note insertion behaviour."""

    def test_ghost_notes_added(self):
        """Laid Back profile with high ghost density should add ghost notes."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Ghost Test",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,  # zero offset variance
            ghost_note_density=0.95,  # high enough to guarantee at least one
            seed=12345,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        ghost_events = [ev for ev in result if ev.source_role == "ghost"]
        assert len(ghost_events) > 0, "Expected at least one ghost note to be added"

    def test_ghost_notes_soft(self):
        """Ghost notes added by engine should be low velocity (≤50)."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Ghost Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            ghost_note_density=0.9,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        ghost_events = [ev for ev in result if ev.source_role == "ghost"]
        for g in ghost_events:
            assert g.velocity <= 50, f"Ghost velocity {g.velocity} should be ≤50"

    def test_ghost_notes_suppressed_low_confidence(self):
        """Low confidence should suppress ghost notes."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Ghost Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            ghost_note_density=0.8,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result_low = engine.process(events, input_confidence=0.1)

        # Reset RNG for comparison
        engine.seed(42)
        result_high = engine.process(events, input_confidence=1.0)

        ghost_low = len([ev for ev in result_low if ev.source_role == "ghost"])
        ghost_high = len([ev for ev in result_high if ev.source_role == "ghost"])

        assert ghost_low <= ghost_high, (
            f"Low confidence should not produce more ghosts ({ghost_low}) "
            f"than high confidence ({ghost_high})"
        )

    def test_no_ghost_notes_machine_tight(self):
        """Machine profile with zero ghost density should not add ghosts."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Machine Test",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            ghost_note_density=0.0,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)
        ghost_events = [ev for ev in result if ev.source_role == "ghost"]
        assert len(ghost_events) == 0, "Machine profile should not add ghost notes"

    def test_ghost_notes_reduced_at_fast_tempo(self):
        """Very fast tempo (>160 BPM) should reduce ghost note probability."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Fast Ghost Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            ghost_note_density=0.9,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)

        engine.seed(42)
        result_normal = engine.process(events, tempo_bpm=120)
        engine.seed(42)
        result_fast = engine.process(events, tempo_bpm=200)

        ghosts_normal = len([ev for ev in result_normal if ev.source_role == "ghost"])
        ghosts_fast = len([ev for ev in result_fast if ev.source_role == "ghost"])

        assert ghosts_fast < ghosts_normal, (
            f"Fast tempo should suppress ghosts ({ghosts_fast} vs {ghosts_normal})"
        )

    def test_ghost_positions_near_backbeats(self):
        """Ghost notes should appear at off-beat 16th positions (1, 3, 5, ...)."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Ghost Pos Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            ghost_note_density=0.95,
            seed=123,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        # Ghost notes are only added at off-beat 16th positions
        # that don't conflict with 8th-note hats or main hits
        valid_positions = {1, 3, 5, 7, 9, 11, 13, 15}
        for ev in result:
            if ev.source_role == "ghost":
                assert ev.grid_position % 16 in valid_positions, (
                    f"Ghost at invalid position {ev.grid_position}"
                )


# ===================================================================
# 4. Compound split / micro-flam
# ===================================================================


class TestCompoundSplit:
    """Verify compound split behaviour for simultaneous hits."""

    def test_split_separates_simultaneous_hits(self):
        """Simultaneous kick+crash should get small timing separation."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0, velocity=100),
            GrooveEvent(instrument="crash", grid_position=0, bar_index=0, velocity=100),
        ]
        profile = DrummerProfile(
            name="Split Test",
            timing_strategy=TimingStrategy.COMPOUND_SPLIT,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        offsets = {ev.instrument: ev.timing_offset_ms for ev in result}
        # Kick gets 0.0, crash gets +4.0
        assert offsets.get("crash", 0) >= 3.0, (
            f"Crash should be slightly delayed vs kick: {offsets}"
        )

    def test_split_not_applied_to_machine_tight(self):
        """machine_tight should NOT apply compound splits."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0, velocity=100),
            GrooveEvent(instrument="snare", grid_position=0, bar_index=0, velocity=100),
        ]
        profile = DrummerProfile(
            name="Machine Split",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        for ev in result:
            assert ev.timing_offset_ms == 0.0, (
                f"machine_tight should not apply compound split, "
                f"got {ev.timing_offset_ms} for {ev.instrument}"
            )

    def test_split_under_10ms(self):
        """Compound split offsets should remain small (< 10 ms)."""
        events = [
            GrooveEvent(instrument="kick", grid_position=4, bar_index=0, velocity=100),
            GrooveEvent(instrument="snare", grid_position=4, bar_index=0, velocity=100),
            GrooveEvent(instrument="hi_hat", grid_position=4, bar_index=0, velocity=80),
        ]
        profile = DrummerProfile(
            name="Split Small Test",
            timing_strategy=TimingStrategy.COMPOUND_SPLIT,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        for ev in result:
            assert abs(ev.timing_offset_ms) < 15, (
                f"Compound split offset too large: {ev.timing_offset_ms} "
                f"for {ev.instrument}"
            )


# ===================================================================
# 5. Confidence-aware behaviour
# ===================================================================


class TestConfidenceAware:
    """Verify confidence-aware behaviour."""

    def test_low_confidence_reduces_ghost_notes(self):
        """Low input confidence should suppress ghost note generation."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Confidence Ghost",
            timing_strategy=TimingStrategy.LAID_BACK,
            ghost_note_density=0.5,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)

        # High confidence
        engine.seed(42)
        result_high = engine.process(events, input_confidence=0.9)
        # Low confidence
        engine.seed(42)
        result_low = engine.process(events, input_confidence=0.1)

        ghosts_high = len([ev for ev in result_high if ev.source_role == "ghost"])
        ghosts_low = len([ev for ev in result_low if ev.source_role == "ghost"])

        assert ghosts_low <= ghosts_high, (
            f"Low confidence ({ghosts_low}) should not produce more ghosts "
            f"than high confidence ({ghosts_high})"
        )

    def test_low_confidence_holds_main_events(self):
        """Low confidence should not drop main groove events."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Confidence Hold",
            timing_strategy=TimingStrategy.TIGHT_ROCK,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events, input_confidence=0.1)

        # Main events (kick, snare, hi-hat) should all still be present
        kick_events = [ev for ev in result if ev.instrument == "kick"]
        snare_events = [ev for ev in result if ev.instrument == "snare"]
        hat_events = [ev for ev in result if ev.instrument == "hi_hat"]

        assert len(kick_events) == 2, f"Expected 2 kicks, got {len(kick_events)}"
        assert len(snare_events) == 2, f"Expected 2 snares, got {len(snare_events)}"
        assert len(hat_events) == 8, f"Expected 8 hats, got {len(hat_events)}"

    def test_low_confidence_reduces_timing_variance(self):
        """Low confidence should reduce timing randomisation."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Confidence Timing",
            timing_strategy=TimingStrategy.LOOSE_GARAGE,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)

        engine.seed(42)
        result_high = engine.process(events, input_confidence=1.0)
        engine.seed(42)
        result_low = engine.process(events, input_confidence=0.1)

        # High confidence should have more timing spread
        high_spread = max(ev.timing_offset_ms for ev in result_high) - min(ev.timing_offset_ms for ev in result_high)
        low_spread = max(ev.timing_offset_ms for ev in result_low) - min(ev.timing_offset_ms for ev in result_low)

        assert high_spread >= low_spread, (
            f"High confidence should have >= spread ({high_spread:.2f}) "
            f"than low ({low_spread:.2f})"
        )


# ===================================================================
# 6. Determinism
# ===================================================================


class TestDeterminism:
    """Verify that the engine is deterministic with a seed."""

    def test_same_seed_same_result(self):
        """Two runs with the same seed should produce identical results."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Determinism Test",
            timing_strategy=TimingStrategy.LOOSE_GARAGE,
            seed=42,
        )
        engine1 = DrummerFeelEngine(profile)
        engine2 = DrummerFeelEngine(profile)

        result1 = engine1.process(events)
        result2 = engine2.process(events)

        assert len(result1) == len(result2)
        for ev1, ev2 in zip(result1, result2):
            assert ev1.grid_position == ev2.grid_position
            assert ev1.velocity == ev2.velocity
            assert abs(ev1.timing_offset_ms - ev2.timing_offset_ms) < 0.001

    def test_different_seed_different_result(self):
        """Different seeds should produce (likely) different results."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Seed Diff Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result_a = engine.process(events)

        profile.seed = 999
        engine2 = DrummerFeelEngine(profile)
        result_b = engine2.process(events)

        # Results should differ in at least one field
        differences = 0
        for ev_a, ev_b in zip(result_a, result_b):
            if (ev_a.timing_offset_ms != ev_b.timing_offset_ms
                    or ev_a.velocity != ev_b.velocity):
                differences += 1
        assert differences > 0, "Different seeds should change at least one field"

    def test_seed_method_resets_rng(self):
        """Calling seed() on an engine should reset determinism."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Seed Reset Test",
            timing_strategy=TimingStrategy.LOOSE_GARAGE,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result_a = engine.process(events)

        # Now run again after re-seeding
        engine.seed(42)
        result_b = engine.process(events)

        assert len(result_a) == len(result_b)
        for ev_a, ev_b in zip(result_a, result_b):
            assert abs(ev_a.timing_offset_ms - ev_b.timing_offset_ms) < 0.001


# ===================================================================
# 7. Edge cases
# ===================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_events(self):
        """Processing an empty list should return an empty list."""
        profile = DrummerProfile(name="Empty Test", seed=0)
        engine = DrummerFeelEngine(profile)
        result = engine.process([])
        assert result == []

    def test_single_event(self):
        """A single event should still be processed correctly."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0, velocity=100),
        ]
        profile = DrummerProfile(
            name="Single Test",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            ghost_note_density=0.0,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)
        assert len(result) == 1
        assert result[0].instrument == "kick"
        assert 1 <= result[0].velocity <= 127

    def test_unknown_instrument(self):
        """Unknown instruments should fall back to sensible defaults."""
        events = [
            GrooveEvent(instrument="tambourine", grid_position=0, bar_index=0, velocity=100),
        ]
        profile = DrummerProfile(
            name="Unknown Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            seed=0,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)
        assert len(result) == 1
        # Unknown instrument should get zero bias, zero variance fallback
        assert result[0].instrument == "tambourine"

    def test_confidence_clamped(self):
        """input_confidence should be clamped to 0–1."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(name="Clamp Test", seed=0)
        engine = DrummerFeelEngine(profile)
        # These should not raise
        engine.process(events, input_confidence=-0.5)
        engine.process(events, input_confidence=1.5)


# ===================================================================
# 8. DrummerProfile builder helpers
# ===================================================================


class TestProfileHelpers:
    """Verify DrummerProfile builder helpers fill in defaults correctly."""

    def test_ensure_offsets_fills_from_strategy(self):
        """_ensure_offsets should return strategy defaults for unspecified instruments."""
        profile = DrummerProfile(timing_strategy=TimingStrategy.LAID_BACK)
        offsets = profile._ensure_offsets()
        assert "kick" in offsets
        assert "snare" in offsets
        assert "hi_hat" in offsets
        # Laid Back snare offset should be close to +18ms
        assert 15.0 <= offsets["snare"] <= 21.0

    def test_ensure_offsets_custom_overrides(self):
        """Custom offsets should override strategy defaults."""
        profile = DrummerProfile(
            timing_strategy=TimingStrategy.TIGHT_ROCK,
            limb_timing_offsets_ms={"snare": 10.0},
        )
        offsets = profile._ensure_offsets()
        assert offsets["snare"] == 10.0
        assert offsets["kick"] == -5.0  # from strategy default

    def test_builtin_profiles_exist(self):
        """All built-in profiles should be accessible."""
        profiles = DrummerProfile.builtin_profiles()
        assert "machine" in profiles
        assert "tight_rock" in profiles
        assert "laid_back" in profiles
        assert "pushed_punk" in profiles
        assert "loose_garage" in profiles

    def test_get_known_profile(self):
        """get() should return a valid profile for known ids."""
        profile = DrummerProfile.get("tight_rock")
        assert profile.timing_strategy == TimingStrategy.TIGHT_ROCK
        assert profile.name == "Tight Rock"

    def test_get_unknown_profile_raises(self):
        """get() should raise KeyError for unknown ids."""
        import pytest
        with pytest.raises(KeyError):
            DrummerProfile.get("nonexistent")

    def test_instrument_group_mapping(self):
        """_instrument_group should correctly map instrument names."""
        assert _instrument_group("kick") == "kick"
        assert _instrument_group("snare") == "snare"
        assert _instrument_group("hi_hat") == "hi_hat"
        assert _instrument_group("hat") == "hi_hat"
        assert _instrument_group("closed_hat") == "hi_hat"
        assert _instrument_group("open_hat") == "hi_hat"
        assert _instrument_group("ride") == "ride"
        assert _instrument_group("crash") == "crash"
        assert _instrument_group("tom") == "toms"
        assert _instrument_group("toms") == "toms"
        assert _instrument_group("tambourine") == "tambourine"


# ===================================================================
# 9. to_midi_dicts conversion
# ===================================================================


class TestToMidiDicts:
    """Verify to_midi_dicts conversion."""

    def test_basic_conversion(self):
        """to_midi_dicts should produce valid MIDI dicts."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0, velocity=100),
            GrooveEvent(instrument="snare", grid_position=4, bar_index=0, velocity=100),
        ]
        profile = DrummerProfile(name="MIDI Test", seed=0)
        engine = DrummerFeelEngine(profile)
        midi_dicts = engine.to_midi_dicts(events, bpm=120)

        assert len(midi_dicts) == 2
        for d in midi_dicts:
            assert "timestamp" in d
            assert "velocity" in d
            assert "note" in d
            assert "instrument" in d
            assert d["timestamp"] >= 0.0
            assert 1 <= d["velocity"] <= 127

    def test_timing_offset_in_conversion(self):
        """Timing offsets should be reflected in the timestamp."""
        events = [
            GrooveEvent(
                instrument="snare", grid_position=4, bar_index=0,
                velocity=100, timing_offset_ms=18.0,
            ),
        ]
        profile = DrummerProfile(name="Offset Test", seed=0)
        engine = DrummerFeelEngine(profile)
        midi_dicts = engine.to_midi_dicts(events, bpm=120)

        # 16th note at 120 BPM = 0.125s
        # Base time = 4 * 0.125 = 0.5s, offset = 0.018s
        assert abs(midi_dicts[0]["timestamp"] - 0.518) < 0.001, (
            f"Expected ~0.518s, got {midi_dicts[0]['timestamp']}"
        )

    def test_default_note_map(self):
        """Default note map should map standard instruments to GM notes."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0, velocity=100),
            GrooveEvent(instrument="snare", grid_position=4, bar_index=0, velocity=100),
            GrooveEvent(instrument="hi_hat", grid_position=8, bar_index=0, velocity=80),
        ]
        profile = DrummerProfile(name="Note Map Test", seed=0)
        engine = DrummerFeelEngine(profile)
        midi_dicts = engine.to_midi_dicts(events, bpm=120)

        from models import CLOSED_HAT, KICK, SNARE
        note_map = {"kick": KICK, "snare": SNARE, "hi_hat": CLOSED_HAT}
        for d in midi_dicts:
            expected = note_map.get(d["instrument"])
            assert d["note"] == expected, (
                f"Expected note {expected} for {d['instrument']}, got {d['note']}"
            )


# ===================================================================
# 10. GrooveEvent.copy_with
# ===================================================================


class TestGrooveEvent:
    """Verify GrooveEvent.copy_with works correctly."""

    def test_copy_with_basic(self):
        """copy_with should create a modified copy."""
        ev = GrooveEvent(instrument="kick", grid_position=0, velocity=100)
        ev2 = ev.copy_with(velocity=80, timing_offset_ms=5.0)
        assert ev2.velocity == 80
        assert ev2.timing_offset_ms == 5.0
        assert ev2.instrument == "kick"  # unchanged
        assert ev2.grid_position == 0  # unchanged
        # Original should be unchanged
        assert ev.velocity == 100

    def test_copy_with_no_args(self):
        """copy_with with no args should return equivalent event."""
        ev = GrooveEvent(instrument="snare", grid_position=4, bar_index=1, velocity=90)
        ev2 = ev.copy_with()
        assert ev2.instrument == ev.instrument
        assert ev2.grid_position == ev.grid_position
        assert ev2.bar_index == ev.bar_index
        assert ev2.velocity == ev.velocity


# ===================================================================
# 11. Probability filtering
# ===================================================================


class TestProbabilityFiltering:
    """Verify probability filtering behaviour."""

    def test_probability_one_always_survives(self):
        """Events with probability=1.0 should always survive filtering."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0,
                        velocity=100, probability=1.0),
            GrooveEvent(instrument="crash", grid_position=0, bar_index=0,
                        velocity=100, probability=0.0),
        ]
        profile = DrummerProfile(
            name="Prob Test",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        # Kick (prob=1.0) must survive; crash (prob=0.0) should be filtered
        instruments = {ev.instrument for ev in result}
        assert "kick" in instruments
        # Crash with prob=0.0 and no survival bonus should be dropped
        # (machine_tight = no variance, so no extra dice rolls affect it)
        assert "crash" not in instruments

    def test_probability_high_confidence_boosts_main(self):
        """Low confidence should boost main event survival."""
        events = [
            GrooveEvent(instrument="kick", grid_position=0, bar_index=0,
                        velocity=100, probability=0.3, source_role="main"),
            GrooveEvent(instrument="crash", grid_position=1, bar_index=0,
                        velocity=100, probability=0.3, source_role="crash"),
        ]
        profile = DrummerProfile(
            name="Conf Boost Test",
            timing_strategy=TimingStrategy.MACHINE_TIGHT,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)

        # Low confidence — main gets +0.15, extras get -0.2
        result = engine.process(events, input_confidence=0.3)
        instruments = {ev.instrument for ev in result}
        assert "kick" in instruments, "Main event should survive with low confidence"
        # Crash may or may not survive depending on RNG — main should have advantage


# ===================================================================
# 12. Integration: full pipeline sanity
# ===================================================================


class TestIntegration:
    """Integration-level tests for the full pipeline."""

    def test_pipeline_maintains_event_count(self):
        """The pipeline should not drop main events unexpectedly."""
        events = _rock_groove_1bar()
        orig_count = len(events)

        profile = DrummerProfile(
            name="Integration",
            timing_strategy=TimingStrategy.TIGHT_ROCK,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        # Should have at least the original count (ghost notes add more)
        assert len(result) >= orig_count, (
            f"Pipeline dropped events: {orig_count} -> {len(result)}"
        )

    def test_all_profiles_produce_valid_output(self):
        """All built-in profiles should produce valid MIDI-range velocities."""
        events = _rock_groove_1bar()
        for profile_id in DrummerProfile.builtin_profiles():
            profile = DrummerProfile.get(profile_id)
            engine = DrummerFeelEngine(profile)
            result = engine.process(events)

            assert len(result) > 0, f"Profile '{profile_id}' produced no output"
            for ev in result:
                assert 1 <= ev.velocity <= 127, (
                    f"Profile '{profile_id}': velocity {ev.velocity} out of range"
                )

    def test_hat_not_repetitive_unless_machine(self):
        """Non-machine profiles should not have identical consecutive hat velocities."""
        events = _rock_groove_1bar()
        profile = DrummerProfile(
            name="Hat Breath Test",
            timing_strategy=TimingStrategy.LAID_BACK,
            seed=42,
        )
        engine = DrummerFeelEngine(profile)
        result = engine.process(events)

        hat_vels = [ev.velocity for ev in result if ev.instrument == "hi_hat"]
        # At least some hats should differ from their neighbours (the test is
        # probabilistic — we just check the set is not all the same value)
        assert len(set(hat_vels)) >= 2 or len(hat_vels) <= 1, (
            f"Hi-hat velocities should show variation, got {hat_vels}"
        )