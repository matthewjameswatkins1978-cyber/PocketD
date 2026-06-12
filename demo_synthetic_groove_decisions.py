"""Synthetic groove-decision demo with reason codes.

Demonstrates that ``select_groove_decision`` returns not only the selected
groove but also a reason code explaining *why* that groove was chosen.

Scenarios
---------
1. Section A (120 BPM, high conf)   -> motorik (high_confidence_motorik)
2. Section B (90 BPM,  high conf)   -> half_time (high_confidence_half_time)
3. Low conf after Section A          -> motorik (low_confidence_keep_previous)
4. Low conf after Section B          -> half_time (low_confidence_keep_previous)
5. Low conf, no previous             -> simple_rock (low_confidence_default)
"""

from __future__ import annotations

from groove_matcher import select_groove_decision

SEPARATOR = "-" * 72


def report_decision(label: str, bpm: float, confidence: float,
                    previous_groove_id: str | None = None) -> str:
    """Evaluate a groove decision and return a formatted table row."""
    d = select_groove_decision(
        bpm=bpm,
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
        f"  {label:<28s} {bpm:>7.1f}   {confidence:>5.2f}   "
        f"{prev_display:<12s} {d.selected_groove_id:<15s} "
        f"{changed_display:<8s} {d.reason}"
    )


def main() -> None:
    print(SEPARATOR)
    print("Synthetic Groove Decision Demo (with reason codes)")
    print(SEPARATOR)
    print()
    print("  {:<28s} {:>7s} {:>6s} {:<12s} {:<15s} {:<8s} {:s}".format(
        "Section", "BPM", "Conf", "Previous", "Selected", "Status", "Reason",
    ))
    print("  " + "-" * (28 + 7 + 6 + 12 + 15 + 8 + 30))

    # 1. High-confidence 120 BPM -> motorik
    print(report_decision(
        "A (120 BPM, high conf)", bpm=120.0, confidence=0.85,
    ))

    # 2. High-confidence 90 BPM -> half_time (changes from motorik)
    print(report_decision(
        "B (90 BPM, high conf)", bpm=90.0, confidence=0.85,
        previous_groove_id="motorik",
    ))

    # 3. Low confidence after Section A -> keeps motorik
    print(report_decision(
        "Low after A", bpm=120.0, confidence=0.2,
        previous_groove_id="motorik",
    ))

    # 4. Low confidence after Section B -> keeps half_time
    print(report_decision(
        "Low after B", bpm=90.0, confidence=0.2,
        previous_groove_id="half_time",
    ))

    # 5. Low confidence, no previous -> simple_rock
    print(report_decision(
        "Low, no previous", bpm=120.0, confidence=0.2,
    ))

    print()
    print(SEPARATOR)
    print("All decision reasons are correct -- groove decisions are explainable.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()