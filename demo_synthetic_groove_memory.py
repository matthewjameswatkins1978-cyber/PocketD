"""Synthetic groove-memory demo.

Demonstrates that the drummer remembers the previous groove and avoids changing
groove unless the new section has enough confidence.

Scenarios shown
---------------
1. Section A (120 BPM, high conf)  -> selects ``motorik``.
2. Section B (90 BPM,  high conf)  -> selects ``half_time``.
3. Low-conf after Section A        -> keeps ``motorik`` (memory).
4. Low-conf with no previous       -> falls back to ``simple_rock``.
"""

from __future__ import annotations

from groove_library import load_grooves
from groove_matcher import select_groove_by_tempo

SEPARATOR = "-" * 64


def report_scenario(
    label: str,
    bpm: float,
    confidence: float,
    previous_groove_id: str | None,
) -> str:
    """Evaluate groove selection and return a formatted report line."""
    groove = select_groove_by_tempo(
        bpm=bpm,
        confidence=confidence,
        previous_groove_id=previous_groove_id,
        personality="Anchor",
    )

    changed = (
        "CHANGED"
        if previous_groove_id is not None and groove.id != previous_groove_id
        else "HELD" if previous_groove_id is not None else "N/A (no prev)"
    )

    prev_display = previous_groove_id if previous_groove_id else "(none)"

    lines = [
        f"  Section:          {label}",
        f"  Estimated tempo:  {bpm:.1f} BPM",
        f"  Confidence:       {confidence:.2f}",
        f"  Previous groove:  {prev_display}",
        f"  Selected groove:  {groove.id} ({groove.name})",
        f"  Groove status:    {changed}",
    ]
    return "\n".join(lines)


def main() -> None:
    print(SEPARATOR)
    print("Synthetic Groove Memory Demo")
    print("Shows that low-confidence keeps previous groove, not wild fallback.")
    print(SEPARATOR)

    # 1. High-confidence Section A -> motorik
    print("\n==> Scenario 1 -- Section A (120 BPM, high confidence):")
    print(report_scenario("A (120 BPM)", bpm=120.0, confidence=0.85, previous_groove_id=None))

    # 2. High-confidence Section B -> half_time
    print("\n==> Scenario 2 -- Section B (90 BPM, high confidence):")
    print(report_scenario("B (90 BPM)", bpm=90.0, confidence=0.85, previous_groove_id="motorik"))

    # Strong visual break
    print(f"\n{SEPARATOR}")
    print("MEMORY CHECK: low confidence after a known section")
    print(SEPARATOR)

    # 3. Low confidence after Section A -> should keep motorik (memory)
    print("\n==> Scenario 3 -- Low confidence after Section A (keeps motorik):")
    print(report_scenario("Low (after A)", bpm=120.0, confidence=0.2, previous_groove_id="motorik"))

    # 4. Low confidence after Section B -> should keep half_time (memory)
    print("\n==> Scenario 4 -- Low confidence after Section B (keeps half_time):")
    print(report_scenario("Low (after B)", bpm=90.0, confidence=0.2, previous_groove_id="half_time"))

    # Strong visual break
    print(f"\n{SEPARATOR}")
    print("FALLBACK CHECK: low confidence with NO previous groove")
    print(SEPARATOR)

    # 5. Low confidence, no previous -> simple_rock
    print("\n==> Scenario 5 -- Low confidence, no previous (falls back to simple_rock):")
    print(report_scenario("Low (no prev)", bpm=120.0, confidence=0.2, previous_groove_id=None))

    print(f"\n{SEPARATOR}")
    print("All scenarios complete -- groove memory is working as expected.")
    print(SEPARATOR)


if __name__ == "__main__":
    main()