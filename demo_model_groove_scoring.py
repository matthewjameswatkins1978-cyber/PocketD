"""Demonstrate model-based groove scoring with all three preset models.

Shows candidate scores for several scenarios, including how change
penalty and confidence thresholds affect decisions.
"""

from __future__ import annotations

from groove_matcher import select_groove_decision_with_model
from drummer.models import (
    MOTORIK_TIGHT_MODEL,
    SIMPLE_ROCK_SAFE_MODEL,
    SPARSE_POSTPUNK_MODEL,
)


def _fmt_score(val: float) -> str:
    return f"{val:5.3f}"


def print_decision(
    label: str,
    bpm: float,
    confidence: float,
    model,
    previous_groove_id: str | None = None,
) -> None:
    decision = select_groove_decision_with_model(
        bpm=bpm,
        confidence=confidence,
        model=model,
        previous_groove_id=previous_groove_id,
    )

    changed_str = {
        True: "CHANGED",
        False: "HELD",
        None: "N/A (first)",
    }.get(decision.changed, "?")

    print(f"  Model:           {model.name}")
    print(f"  Tempo:           {bpm:.0f} BPM")
    print(f"  Confidence:      {confidence:.2f}")
    print(f"  Previous groove: {previous_groove_id or 'none'}")
    print(f"  Selected:        {decision.selected_groove_id}")
    print(f"  Status:          {changed_str}")
    print(f"  Reason:          {decision.reason}")
    print()

    if decision.candidate_scores:
        print(f"  {'Candidate':14s} | {'Total':>8s} | {'Tempo':>8s} | {'Conf':>8s} | {'Pref':>8s} | {'Penalty':>8s}")
        print(f"  {'-'*14} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8} | {'-'*8}")
        for cs in decision.candidate_scores:
            print(
                f"  {cs.groove_id:14s} | {_fmt_score(cs.total_score):>8s} | "
                f"{_fmt_score(cs.tempo_score):>8s} | {_fmt_score(cs.confidence_score):>8s} | "
                f"{_fmt_score(cs.preference_score):>8s} | {_fmt_score(cs.change_penalty):>8s}"
            )
    else:
        print("  (No candidate scores — low-confidence fallback)")
    print()


def main() -> None:
    print("=" * 92)
    print("Model-Based Groove Scoring — Scenario Demonstrations")
    print("=" * 92)
    print()

    # Scenario 1: Motorik at 120 BPM, high confidence
    print("--- Scenario 1: Motorik @ 120 BPM, high confidence (0.85) -----------")
    print_decision("Motorik@120", bpm=120.0, confidence=0.85, model=MOTORIK_TIGHT_MODEL)

    # Scenario 2: Motorik at 90 BPM, high confidence (should switch away)
    print("--- Scenario 2: Motorik @ 90 BPM, high confidence (0.85) ------------")
    print_decision("Motorik@90", bpm=90.0, confidence=0.85, model=MOTORIK_TIGHT_MODEL)

    # Scenario 3: Sparse post-punk at 90 BPM, high confidence
    print("--- Scenario 3: Sparse Post-Punk @ 90 BPM, high confidence (0.85) ---")
    print_decision("PostPunk@90", bpm=90.0, confidence=0.85, model=SPARSE_POSTPUNK_MODEL)

    # Scenario 4: Simple rock at 120 BPM, high confidence
    print("--- Scenario 4: Simple Rock @ 120 BPM, high confidence (0.85) -------")
    print_decision("Rock@120", bpm=120.0, confidence=0.85, model=SIMPLE_ROCK_SAFE_MODEL)

    # Scenario 5: Low confidence with previous groove
    print("--- Scenario 5: Low confidence (0.1) with previous=half_time --------")
    print_decision(
        "LowConf@Prev", bpm=120.0, confidence=0.1,
        model=SIMPLE_ROCK_SAFE_MODEL, previous_groove_id="half_time",
    )

    # Scenario 6: Low confidence with no previous groove
    print("--- Scenario 6: Low confidence (0.1) with no previous groove --------")
    print_decision("LowConf@None", bpm=120.0, confidence=0.1, model=SIMPLE_ROCK_SAFE_MODEL)

    # Scenario 7: Change penalty — Motorik at 120 with previous=motorik
    # (same groove, should hold — penalty only applies to others)
    print("--- Scenario 7: Motorik @ 120, previous=motorik (hold test) --------")
    print_decision(
        "MotorikHold", bpm=120.0, confidence=0.8,
        model=MOTORIK_TIGHT_MODEL, previous_groove_id="motorik",
    )

    # Scenario 8: Change penalty — Motorik at 120 with previous=half_time
    print("--- Scenario 8: Motorik @ 120, previous=half_time (switch test) -----")
    print_decision(
        "MotorikSwitch", bpm=120.0, confidence=0.8,
        model=MOTORIK_TIGHT_MODEL, previous_groove_id="half_time",
    )


if __name__ == "__main__":
    main()