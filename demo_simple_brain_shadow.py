"""Demo: Simple Brain v0 — shadow mode over real FeatureMonitor output.

Runs SimpleBrain in parallel with the existing perception path over
synthetic MusicalEvent timelines, rendering compact trace tables
bar by bar.  No MIDI output — pure diagnostic.

TODO: ``build_scenario_timeline`` lives in ``demo_continuous_jam_midi.py``
for now.  Scenario timeline builders should eventually move into a
reusable module (e.g. ``synthetic/timelines.py``).
"""

from __future__ import annotations

import sys
from pathlib import Path

# -- Reuse existing synthetic timeline builder --
#    (will be extracted to a proper module later)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from demo_continuous_jam_midi import (  # noqa: E402
    _SECTION_PHASE,
    build_scenario_timeline,
)

from drummer.simple_brain import SimpleBrain  # noqa: E402
from drummer.simple_brain_trace import render_simple_brain_trace_table  # noqa: E402
from perception.features import FeatureMonitor  # noqa: E402
from perception.models import MusicalEvent  # noqa: E402

# ---------------------------------------------------------------------------
# Scenario list
# ---------------------------------------------------------------------------

SCENARIOS: list[tuple[str, str]] = [
    ("enter", "stable_input"),
    ("enter", "uncertain_input"),
    ("drop", "deliberate_sparse"),
    ("drop", "pullback_after_build"),
    ("build", "strong_build"),
    ("anchor_recovery", "weak_input_recovery"),
]

# ---------------------------------------------------------------------------
# Shadow runner
# ---------------------------------------------------------------------------


def run_shadow(
    scenario: str,
    variation: str,
    bpm: float = 120.0,
) -> tuple[list[dict], str]:
    """Run SimpleBrain in shadow mode over one scenario.

    Parameters
    ----------
    scenario : str
        Scenario name (e.g. "enter", "drop").
    variation : str
        Variation name (e.g. "stable_input", "deliberate_sparse").
    bpm : float
        Tempo in beats per minute.

    Returns
    -------
    trace : list[dict]
        Per-bar trace rows.
    title : str
        Descriptive title for rendering.
    """
    # Build the synthetic timeline.
    # Returns list[list[MusicalEvent]] — one list of events per bar.
    timeline_bars: list = build_scenario_timeline(  # type: ignore[assignment]
        scenario=scenario,
        playtest_variation=variation,
        bpm=bpm,
        bars=16,
    )

    # Set up perception and SimpleBrain.
    monitor = FeatureMonitor()
    brain = SimpleBrain()

    # Timing: tempo → bar duration.
    seconds_per_beat = 60.0 / bpm
    seconds_per_bar = 4.0 * seconds_per_beat

    rows: list[dict] = []
    current_time: float = 0.0

    for bar_idx, events in enumerate(timeline_bars):
        bar_start = current_time
        bar_end = bar_start + seconds_per_bar

        # Feed all events in this bar to the FeatureMonitor.
        for evt in events:
            monitor.feed(evt)

        # Derive a simple section label from event density.
        bar = timeline_bars[bar_idx]
        n_events = len(bar)
        if n_events == 0:
            section = "silent"
        elif n_events <= 2:
            section = "sparse"
        elif n_events <= 6:
            section = "medium"
        else:
            section = "dense"

        # Look up phase alignment from section name.
        phase = _SECTION_PHASE.get(section.upper(), None)

        # Take a snapshot at bar end and feed to SimpleBrain.
        snapshot = monitor.snapshot(bar_end, phase_alignment=phase)
        decision = brain.decide(snapshot)

        rows.append(
            {
                "bar": bar_idx,
                "section": section,
                "action": decision.action.value,
                "beat": decision.beat_name,
                "confidence": decision.confidence,
                "input_density": snapshot.input_density,
                "player_certainty": snapshot.player_certainty,
                "stability": snapshot.repetition_stability,
                "change_score": snapshot.change_score,
                "silence": snapshot.silence_duration,
                "reason": decision.reason,
                "scores": decision.scores,
            }
        )

        current_time = bar_end

    title = (
        f"SHADOW: {scenario}/{variation}  "
        f"({len(timeline_bars)} bars @ {bpm:.0f} bpm)"
    )
    return rows, title


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    for scenario, variation in SCENARIOS:
        trace, title = run_shadow(scenario, variation)
        print(f"{'=' * 70}")
        print(f"  {title}")
        print(f"{'=' * 70}")
        table = render_simple_brain_trace_table(trace)
        print(table)
        print()

    print("Done.  Run with:")
    print("  python demo_simple_brain_shadow.py")


if __name__ == "__main__":
    main()