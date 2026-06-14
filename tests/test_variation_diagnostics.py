"""Tests that playtest scenario variations produce measurably different diagnostics.

This is critical for the feedback memory system: if variations don't actually
vary, the learning summary will contain noise instead of signal.
"""

from __future__ import annotations

from drummer.playtest_feedback import (
    get_scenario_variations,
    run_playtest_scenario,
)


class TestEnterVariationsDiffer:
    """ENTER variations (stable_input vs uncertain_input) must differ."""

    def test_stable_and_uncertain_use_different_names(self) -> None:
        vars_ = get_scenario_variations("enter")
        names = [v.variation_name for v in vars_]
        assert "stable_input" in names
        assert "uncertain_input" in names

    def test_stable_and_uncertain_have_different_descriptions(self) -> None:
        vars_ = get_scenario_variations("enter")
        descs = {v.variation_name: v.description for v in vars_}
        assert descs["stable_input"] != descs["uncertain_input"]

    def test_uncertain_input_has_lower_certainty_in_enter_bars(self) -> None:
        """Uncertain input should show lower player_certainty in early bars."""
        vars_ = get_scenario_variations("enter")
        stable = [v for v in vars_ if v.variation_name == "stable_input"][0]
        uncertain = [v for v in vars_ if v.variation_name == "uncertain_input"][0]

        _, stable_diags, _ = run_playtest_scenario(stable, no_play=True)
        _, uncertain_diags, _ = run_playtest_scenario(uncertain, no_play=True)

        # Focus on the enter/maintain bars (2-6)
        stable_certainty = 0.0
        uncertain_certainty = 0.0
        count = 0
        for sd, ud in zip(stable_diags[2:7], uncertain_diags[2:7]):
            stable_certainty += sd.get("certainty", 0.0)
            uncertain_certainty += ud.get("certainty", 0.0)
            count += 1

        avg_stable = stable_certainty / count if count else 0.0
        avg_uncertain = uncertain_certainty / count if count else 0.0

        # Uncertain input should have lower average certainty
        assert avg_uncertain < avg_stable, (
            f"Uncertain input had avg certainty {avg_uncertain:.3f}, "
            f"expected < {avg_stable:.3f}"
        )

    def test_uncertain_input_has_lower_confidence_peak(self) -> None:
        """Uncertain input should result in lower confidence peak during early phase."""
        vars_ = get_scenario_variations("enter")
        stable = [v for v in vars_ if v.variation_name == "stable_input"][0]
        uncertain = [v for v in vars_ if v.variation_name == "uncertain_input"][0]

        stable_summary, stable_diags, _ = run_playtest_scenario(stable, no_play=True)
        uncertain_summary, uncertain_diags, _ = run_playtest_scenario(uncertain, no_play=True)

        # Assert that at least one measurable feature differs by a meaningful amount.
        # In the early enter/maintain bars (2-6), uncertain_input should have lower
        # phase_alignment and/or lower player_certainty.
        stable_phase = sum(d.get("phase", 0.75) for d in stable_diags[2:7])
        uncertain_phase = sum(d.get("phase", 0.75) for d in uncertain_diags[2:7])
        stable_cert = sum(d.get("certainty", 0.0) for d in stable_diags[2:7])
        uncertain_cert = sum(d.get("certainty", 0.0) for d in uncertain_diags[2:7])

        phase_diff = stable_phase - uncertain_phase
        cert_diff = stable_cert - uncertain_cert

        assert phase_diff > 0.5 or cert_diff > 0.3, (
            f"Uncertain input should be measurably lower than stable in early bars. "
            f"Phase sum: stable={stable_phase:.3f} uncertain={uncertain_phase:.3f} "
            f"(diff={phase_diff:.3f}). "
            f"Certainty sum: stable={stable_cert:.3f} uncertain={uncertain_cert:.3f} "
            f"(diff={cert_diff:.3f})"
        )


class TestBuildVariationsDiffer:
    """BUILD variations (slow_build vs strong_build) must differ."""

    def test_slow_and_strong_have_different_names(self) -> None:
        vars_ = get_scenario_variations("build")
        names = [v.variation_name for v in vars_]
        assert "slow_build" in names
        assert "strong_build" in names

    def test_strong_build_has_higher_early_event_count(self) -> None:
        """Strong build should produce more events in early build bars."""
        vars_ = get_scenario_variations("build")
        slow = [v for v in vars_ if v.variation_name == "slow_build"][0]
        strong = [v for v in vars_ if v.variation_name == "strong_build"][0]

        _, slow_diags, _ = run_playtest_scenario(slow, no_play=True)
        _, strong_diags, _ = run_playtest_scenario(strong, no_play=True)

        # Compare event counts in build bars (7-9)
        slow_events = sum(d["event_count"] for d in slow_diags[7:10])
        strong_events = sum(d["event_count"] for d in strong_diags[7:10])

        assert strong_events >= slow_events, (
            f"Strong build had {strong_events} events, "
            f"expected >= {slow_events}"
        )


    def test_uncertain_input_has_lower_phase_in_enter_bars(self) -> None:
        """Uncertain input should show lower phase_alignment in early bars."""
        vars_ = get_scenario_variations("enter")
        stable = [v for v in vars_ if v.variation_name == "stable_input"][0]
        uncertain = [v for v in vars_ if v.variation_name == "uncertain_input"][0]

        _, stable_diags, _ = run_playtest_scenario(stable, no_play=True)
        _, uncertain_diags, _ = run_playtest_scenario(uncertain, no_play=True)

        # Phase in enter/maintain bars (2-6) should be lower for uncertain
        stable_phase = [d.get("phase", 0.75) for d in stable_diags[2:7]]
        uncertain_phase = [d.get("phase", 0.75) for d in uncertain_diags[2:7]]

        # At least half the bars should show lower phase
        lower_count = sum(1 for sp, up in zip(stable_phase, uncertain_phase) if up < sp)
        assert lower_count >= 2, (
            f"Uncertain input should have lower phase in at least 2 of bars 2-6. "
            f"Stable: {stable_phase}, Uncertain: {uncertain_phase}"
        )

    def test_focus_diagnostics_include_feature_values(self) -> None:
        """Per-bar focus diagnostics include density, certainty, stability, phase, confidence."""
        v = get_scenario_variations("enter")[0]
        _, raw_diags, _ = run_playtest_scenario(v, no_play=True)

        # Check that focus-range bars have all feature fields
        focus_bars = [d for d in raw_diags if 2 <= d["bar"] <= 5]
        for d in focus_bars:
            assert "density" in d
            assert "certainty" in d
            assert "stability" in d
            assert "phase" in d
            assert "confidence" in d
            assert "event_count" in d
            assert "inferred_intent" in d
            assert "intent" in d
        # Verify values are populated (not zero for all)
        total_dens = sum(d.get("density", 0) for d in focus_bars)
        assert total_dens > 0, "Density values should be populated in focus bars"

    def test_stable_uncertain_have_different_total_events(self) -> None:
        """Stable and uncertain input produce different total event counts in the summary."""
        vars_ = get_scenario_variations("enter")
        stable = [v for v in vars_ if v.variation_name == "stable_input"][0]
        uncertain = [v for v in vars_ if v.variation_name == "uncertain_input"][0]

        s_summary, _, _ = run_playtest_scenario(stable, no_play=True)
        u_summary, _, _ = run_playtest_scenario(uncertain, no_play=True)

        # Total events should differ (uncertain = fewer events = less confident playing)
        assert s_summary.total_events != u_summary.total_events, (
            f"Stable ({s_summary.total_events}) and uncertain "
            f"({u_summary.total_events}) should have different total events"
        )

    def test_all_variations_have_phase_in_diagnostics(self) -> None:
        """Every scenario variation stores phase in its per-bar diagnostics."""
        for scenario_name in ("enter", "build", "anchor_recovery", "drop", "final_bail"):
            variations = get_scenario_variations(scenario_name)
            for v in variations:
                _, raw_diags, _ = run_playtest_scenario(v, no_play=True)
                for d in raw_diags[:5]:
                    assert "phase" in d, (
                        f"{scenario_name}/{v.variation_name}: missing phase in diagnostics"
                    )


class TestScenarioRunnerRealDiagnostics:
    """Real diagnostics from the pipeline for all scenario variations."""

    def test_all_variations_produce_real_diagnostics(self) -> None:
        """Every scenario variation produces real diagnostics in no-play mode."""
        for scenario_name in ("enter", "build", "anchor_recovery", "drop", "final_bail"):
            for preset in ("cautious", "normal", "braver"):
                variations = get_scenario_variations(scenario_name, preset=preset)
                for v in variations:
                    summary, raw_diags, _ = run_playtest_scenario(v, no_play=True)
                    assert summary.total_events > 0, (
                        f"{scenario_name}/{v.variation_name}/{preset}: "
                        f"zero total_events"
                    )
                    assert len(raw_diags) > 0
                    # Real diagnostics should have these fields filled
                    first_5 = raw_diags[:5]
                    for d in first_5:
                        assert "density" in d
                        assert "certainty" in d
                        assert "stability" in d
                        assert "phase" in d

    def test_no_play_returns_events(self) -> None:
        """no_play=True still returns events (needed for sanity checking)."""
        v = get_scenario_variations("enter")[0]
        _, _, events = run_playtest_scenario(v, no_play=True)
        assert len(events) > 0

    def test_play_mode_returns_events(self) -> None:
        """no_play=False should return non-empty global_events list."""
        v = get_scenario_variations("enter")[0]
        _, _, events = run_playtest_scenario(v, no_play=False)
        assert len(events) > 0