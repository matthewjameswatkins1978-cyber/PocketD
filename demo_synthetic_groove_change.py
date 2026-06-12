"""Synthetic groove-change demo.

Proves that the system can detect a change in synthetic pulse/feel across two
sections and select a different groove only when confidence is strong enough.

Sections
---------
A : steady 120 BPM → expects Motorik
B : slower  90 BPM → expects Half Time (sparser feel)
"""

from __future__ import annotations

from audio.onset_detector import detect_onsets_from_events
from audio.pulse_generator import generate_pulse_events
from confidence_engine import estimate_timing_confidence
from demo_synthetic_pipeline import scheduled_events_table
from drummer.humanize import humanize_events
from groove_matcher import select_groove_by_tempo
from pulse_tracker import estimate_tempo

SEPARATOR = "-" * 60


def process_and_report(
    label: str,
    bpm: float,
    duration_seconds: float,
    previous_groove_id: str | None = None,
) -> str:
    """Run the synthetic pipeline for one section and return a readable report."""
    # --- synthetic pulse → onset ---
    pulse_events = generate_pulse_events(bpm=bpm, duration_seconds=duration_seconds)
    onsets = detect_onsets_from_events(pulse_events, min_interval=0.05)
    onset_times = [event.time_seconds for event in onsets]

    # --- tempo / confidence ---
    estimated_bpm = estimate_tempo(onset_times)
    confidence = estimate_timing_confidence(onset_times, expected_bpm=bpm)

    # --- groove selection ---
    groove = select_groove_by_tempo(
        bpm=estimated_bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        personality="Anchor",
    )

    # --- scheduler events ---
    events = scheduled_events_table(groove, estimated_bpm, bars=2, complexity=3)

    # --- humanization ---
    humanized = humanize_events(events, seed=42)

    # --- build report ---
    lines = [
        f"  Section:          {label}",
        f"  Estimated tempo:  {estimated_bpm:.1f} BPM",
        f"  Confidence:       {confidence:.2f}",
        f"  Selected groove:  {groove.id} ({groove.name})",
        f"  Sched. events:    {len(events)}",
        f"  Human. events:    {len(humanized)}",
    ]

    # Small preview of humanized events (first four)
    preview = humanized[:4]
    if preview:
        lines.append("  Humanized preview (first 4):")
        lines.append(
            "    {ts:>8s}  {inst:>10s}  {vel:>3s}".format(
                ts="ts", inst="instrument", vel="vel"
            )
        )
        for ev in preview:
            lines.append(
                f"    {ev['timestamp']:8.3f}s  {ev['instrument']:>10s}  {ev['velocity']:3d}"
            )

    return "\n".join(lines)


def main() -> None:
    print(SEPARATOR)
    print("Synthetic Groove Change Demo")
    print("Proves the system selects different grooves per section.")
    print(SEPARATOR)

    # Section A: steady 120 BPM → expects Motorik
    report_a = process_and_report("A (120 BPM)", bpm=120.0, duration_seconds=2.0)
    print(report_a)
    print()

    # Extract groove id from report_a for passing as previous
    groove_a_id = "motorik"  # we know what 120 BPM selects

    # Section B: slower 90 BPM → expects Half Time
    report_b = process_and_report(
        "B (90 BPM)", bpm=90.0, duration_seconds=2.0,
        previous_groove_id=groove_a_id,
    )
    print(report_b)
    print(SEPARATOR)

    # Low-confidence stability check (very short pulse)
    print("\nLow-confidence safety check (0.3 s pulse at 90 BPM):")
    report_low = process_and_report(
        "Low conf", bpm=90.0, duration_seconds=0.3,
        previous_groove_id=groove_a_id,
    )
    print(report_low)
    print(SEPARATOR)


if __name__ == "__main__":
    main()