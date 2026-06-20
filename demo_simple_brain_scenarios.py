"""Demo: Simple Brain v0 — named scenarios with trace output.

Runs SimpleBrain over five named synthetic snapshot scenarios inspired
by existing Bunny Deluxe playtest forms:

    enter/stable_input       — enter with confident dense playing
    enter/uncertain_input    — enter with borderline confidence
    drop/deliberate_sparse   — deliberate sparse breakdown
    build/strong_build       — strong build-up
    anchor_recovery/weak     — weak input recovery after uncertainty

Each scenario produces a compact trace table using
``render_simple_brain_trace_table``.

No MIDI, no audio — pure logic inspection.
"""

from __future__ import annotations

from drummer.simple_brain import (
    LOCK_SNAPSHOTS,
    LOCK_THRESHOLD,
    RELISTEN_THRESHOLD,
    SWITCH_THRESHOLD,
    SimpleBrain,
)
from drummer.simple_brain_trace import render_simple_brain_trace_table
from perception.features import FeatureSnapshot


# ---------------------------------------------------------------------------
# Snapshot helpers
# ---------------------------------------------------------------------------


def _snap(
    *,
    input_density: float = 0.0,
    repetition_stability: float = 0.0,
    player_certainty: float = 0.0,
    change_score: float = 0.0,
    silence_duration: float = 0.0,
) -> FeatureSnapshot:
    return FeatureSnapshot(
        timestamp=0.0,
        input_density=input_density,
        repetition_stability=repetition_stability,
        player_certainty=player_certainty,
        change_score=change_score,
        silence_duration=silence_duration,
    )


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------


def run_scenario(
    name: str,
    brain: SimpleBrain,
    snapshots: list[tuple[str, FeatureSnapshot]],
) -> list[dict]:
    """Run a brain through named snapshots and collect trace rows.

    Parameters
    ----------
    name : str
        Scenario label for the section column.
    brain : SimpleBrain
        Brain instance (may carry state from a previous scenario).
    snapshots : list[tuple[str, FeatureSnapshot]]
        Pairs of (section_label, snapshot).

    Returns
    -------
    list[dict]
        Trace rows ready for ``render_simple_brain_trace_table``.
    """
    rows: list[dict] = []
    for i, (section, snap) in enumerate(snapshots):
        decision = brain.decide(snap)
        rows.append(
            {
                "bar": i,
                "section": section,
                "action": decision.action.value,
                "beat": decision.beat_name,
                "confidence": decision.confidence,
                "input_density": snap.input_density,
                "player_certainty": snap.player_certainty,
                "stability": snap.repetition_stability,
                "change_score": snap.change_score,
                "silence": snap.silence_duration,
                "reason": decision.reason,
            }
        )
    return rows


# ===================================================================
# Scenarios
# ===================================================================


def scenario_enter_stable() -> list[dict]:
    """enter/stable_input — dense confident playing from the start."""
    brain = SimpleBrain()
    snaps: list[tuple[str, FeatureSnapshot]] = []
    for i in range(LOCK_SNAPSHOTS + 3):
        snaps.append(
            (
                "intro",
                _snap(
                    input_density=0.80,
                    repetition_stability=0.65,
                    player_certainty=0.72,
                    change_score=0.04,
                ),
            )
        )
    return run_scenario("enter/stable_input", brain, snaps)


def scenario_enter_uncertain() -> list[dict]:
    """enter/uncertain_input — confidence hovers near lock threshold."""
    brain = SimpleBrain()
    snaps: list[tuple[str, FeatureSnapshot]] = []
    values = [
        0.48, 0.52, 0.49, 0.53, 0.51, 0.55, 0.52, 0.54,
        0.50, 0.53, 0.56, 0.55, 0.52, 0.57, 0.54, 0.58,
    ]
    for i, cert in enumerate(values):
        snaps.append(
            (
                "uncertain",
                _snap(
                    input_density=0.60,
                    repetition_stability=0.40,
                    player_certainty=cert,
                    change_score=0.06,
                ),
            )
        )
    return run_scenario("enter/uncertain_input", brain, snaps)


def scenario_drop_sparse() -> list[dict]:
    """drop/deliberate_sparse — dense lock then sparse breakdown."""
    brain = SimpleBrain()
    snaps: list[tuple[str, FeatureSnapshot]] = []

    # Lock dense.
    for i in range(LOCK_SNAPSHOTS):
        snaps.append(
            (
                "verse",
                _snap(
                    input_density=0.80,
                    repetition_stability=0.65,
                    player_certainty=0.72,
                    change_score=0.04,
                ),
            )
        )
    # Hold dense briefly.
    for i in range(3):
        snaps.append(
            (
                "verse",
                _snap(
                    input_density=0.78,
                    repetition_stability=0.62,
                    player_certainty=0.68,
                    change_score=0.06,
                ),
            )
        )
    # Sparse breakdown.
    for i in range(5):
        snaps.append(
            (
                "breakdown",
                _snap(
                    input_density=0.18,
                    repetition_stability=0.48,
                    player_certainty=0.66,
                    change_score=SWITCH_THRESHOLD + 0.10,
                ),
            )
        )
    return run_scenario("drop/deliberate_sparse", brain, snaps)


def scenario_build_strong() -> list[dict]:
    """build/strong_build — sparse lock then powerful build to dense."""
    brain = SimpleBrain()
    snaps: list[tuple[str, FeatureSnapshot]] = []

    # Lock sparse.
    for i in range(LOCK_SNAPSHOTS):
        snaps.append(
            (
                "intro",
                _snap(
                    input_density=0.15,
                    repetition_stability=0.45,
                    player_certainty=0.65,
                    change_score=0.04,
                ),
            )
        )
    # Hold sparse briefly.
    for i in range(2):
        snaps.append(
            (
                "intro",
                _snap(
                    input_density=0.17,
                    repetition_stability=0.46,
                    player_certainty=0.64,
                    change_score=0.05,
                ),
            )
        )
    # Build to dense.
    for i in range(5):
        density = 0.20 + i * 0.14
        snaps.append(
            (
                "build",
                _snap(
                    input_density=density,
                    repetition_stability=0.50 + i * 0.04,
                    player_certainty=0.68,
                    change_score=SWITCH_THRESHOLD + 0.05 + i * 0.02,
                ),
            )
        )
    return run_scenario("build/strong_build", brain, snaps)


def scenario_anchor_recovery() -> list[dict]:
    """anchor_recovery/weak — dense lock, confidence wobble, recovery."""
    brain = SimpleBrain()
    snaps: list[tuple[str, FeatureSnapshot]] = []

    # Lock dense.
    for i in range(LOCK_SNAPSHOTS):
        snaps.append(
            (
                "verse",
                _snap(
                    input_density=0.80,
                    repetition_stability=0.65,
                    player_certainty=0.72,
                    change_score=0.04,
                ),
            )
        )
    # Confidence wobbles near relisten threshold but doesn't completely
    # collapse — hovers around 0.22-0.28.
    for i in range(4):
        snaps.append(
            (
                "weak",
                _snap(
                    input_density=0.40,
                    repetition_stability=0.20,
                    player_certainty=0.25 - i * 0.02,
                    change_score=0.08,
                    silence_duration=0.5 + float(i) * 0.5,
                ),
            )
        )
    # Recovery — confidence comes back.
    for i in range(5):
        snaps.append(
            (
                "recovery",
                _snap(
                    input_density=0.70 + i * 0.05,
                    repetition_stability=0.40 + i * 0.06,
                    player_certainty=0.45 + i * 0.08,
                    change_score=0.10,
                ),
            )
        )
    return run_scenario("anchor_recovery/weak", brain, snaps)


# ===================================================================
# Main
# ===================================================================


def main() -> None:
    scenarios = [
        ("enter/stable_input", scenario_enter_stable),
        ("enter/uncertain_input", scenario_enter_uncertain),
        ("drop/deliberate_sparse", scenario_drop_sparse),
        ("build/strong_build", scenario_build_strong),
        ("anchor_recovery/weak", scenario_anchor_recovery),
    ]

    for name, scenario_fn in scenarios:
        print(f"{'=' * 70}")
        print(f"  SCENARIO: {name}")
        print(f"{'=' * 70}")
        trace = scenario_fn()
        table = render_simple_brain_trace_table(trace)
        print(table)
        print()

    print("Done.  Run with:")
    print("  python demo_simple_brain_scenarios.py")


if __name__ == "__main__":
    main()