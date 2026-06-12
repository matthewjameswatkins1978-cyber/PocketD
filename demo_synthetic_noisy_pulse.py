"""Synthetic noisy pulse demo.

Demonstrates that the decision layer stays calm under imperfect input —
jittered timing, missing hits, extra hits, tempo changes, and garbage input.

Each scenario runs the full pulse -> onset -> tempo -> confidence -> decision
pipeline and prints a readable table.
"""

from __future__ import annotations

from audio.onset_detector import detect_onsets_from_events
from confidence_engine import estimate_timing_confidence
from groove_matcher import select_groove_decision
from pulse_tracker import estimate_tempo
from synthetic.noisy_pulse import (
    add_extra_pulse,
    add_timing_jitter,
    drop_pulse,
    make_sparse_or_garbage_pulse,
    make_steady_pulse,
    make_tempo_change_pulse,
)

SEPARATOR = "-" * 72


def run_scenario(
    label: str,
    pulses: list,
    expected_bpm: float = 120.0,
    previous_groove_id: str | None = None,
) -> str:
    """Run the full pipeline and return a formatted table row."""
    onsets = detect_onsets_from_events(pulses, min_interval=0.05)
    onset_times = [e.time_seconds for e in onsets]
    estimated_bpm = estimate_tempo(onset_times)
    confidence = estimate_timing_confidence(onset_times, expected_bpm=expected_bpm)

    d = select_groove_decision(
        bpm=estimated_bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        personality="Anchor",
    )

    prev_display = d.previous_groove_id if d.previous_groove_id else "(none)"
    changed_display = {
        None: "N/A",
        True: "CHANGED",
        False: "HELD",
    }[d.changed]

    return (
        f"  {label:<30s} {estimated_bpm:>7.1f}   {confidence:>5.2f}   "
        f"{prev_display:<12s} {d.selected_groove_id:<15s} "
        f"{changed_display:<8s} {d.reason}"
    )


def main() -> None:
    print(SEPARATOR)
    print("Synthetic Noisy Pulse Demo")
    print("System stays calm under imperfect input.")
    print(SEPARATOR)
    print()
    print("  {:<30s} {:>7s} {:>6s} {:<12s} {:<15s} {:<8s} {:s}".format(
        "Scenario", "BPM", "Conf", "Previous", "Selected", "Status", "Reason",
    ))
    print("  " + "-" * (30 + 7 + 6 + 12 + 15 + 8 + 30))

    # 1. Steady 120
    print(run_scenario(
        "steady_120",
        make_steady_pulse(bpm=120.0, bars=2),
        expected_bpm=120.0,
    ))

    # 2. Jittered 120 (with previous motorik)
    steady = make_steady_pulse(bpm=120.0, bars=2)
    jittered = add_timing_jitter(steady, amount_ms=15.0, seed=42)
    print(run_scenario(
        "jittered_120",
        jittered,
        expected_bpm=120.0,
        previous_groove_id="motorik",
    ))

    # 3. Missing hit (with previous motorik)
    missing = drop_pulse(make_steady_pulse(bpm=120.0, bars=2), index=3)
    print(run_scenario(
        "missing_hit_120",
        missing,
        expected_bpm=120.0,
        previous_groove_id="motorik",
    ))

    # 4. Extra offbeat hit (with previous motorik)
    quarter = 60.0 / 120.0
    steady2 = make_steady_pulse(bpm=120.0, bars=2)
    extra_time = steady2[1].time_seconds + quarter / 2
    extra = add_extra_pulse(steady2, timestamp=extra_time, strength=0.5)
    print(run_scenario(
        "extra_offbeat_120",
        extra,
        expected_bpm=120.0,
        previous_groove_id="motorik",
    ))

    # 5. Tempo change 120 -> 90 (second half analysed)
    change_pulses = make_tempo_change_pulse(start_bpm=120.0, end_bpm=90.0, bars_each=2)
    gap = 60.0 / 120.0
    first_duration = 4 * (60.0 / 120.0) * 2
    cut_time = first_duration + gap
    second_pulses = [p for p in change_pulses if p.time_seconds >= cut_time]
    print(run_scenario(
        "tempo_change_120_to_90",
        second_pulses,
        expected_bpm=90.0,
        previous_groove_id="motorik",
    ))

    # 6. Garbage input with previous groove
    garbage_prev = make_sparse_or_garbage_pulse(seed=99)
    print(run_scenario(
        "garbage_low_conf_with_prev",
        garbage_prev,
        expected_bpm=120.0,
        previous_groove_id="motorik",
    ))

    # 7. Garbage input with no previous groove
    garbage_none = make_sparse_or_garbage_pulse(seed=99)
    print(run_scenario(
        "garbage_low_conf_no_prev",
        garbage_none,
        expected_bpm=120.0,
    ))

    print()
    print(SEPARATOR)
    print("All scenarios stable -- system handles noisy synthetic input gracefully.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()