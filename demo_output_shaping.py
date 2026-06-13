"""Demo: Behaviour-Driven Output Shaping.

Shows how a single groove pattern transforms under each BehaviourIntent.
Prints event grids for easy visual comparison.

Run:
    python demo_output_shaping.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from drummer.behaviour import BehaviourIntent
from drummer.output_shaping import BehaviourOutputShaper
from drummer.feel import GrooveEvent


# ─── Demo groove — a foundational rock pattern with some decoration ───

_ORIGINAL: list[GrooveEvent] = [
    # Beat 1 — kick + hat
    GrooveEvent("kick", 0, velocity=110),
    GrooveEvent("hi_hat", 0, velocity=75),
    # off-8th hat
    GrooveEvent("hi_hat", 2, velocity=65),
    # ghost snare
    GrooveEvent("snare", 3, velocity=22, articulation="ghost"),
    # Beat 2 — snare + hat
    GrooveEvent("snare", 4, velocity=105),
    GrooveEvent("hi_hat", 4, velocity=75),
    # 16th hat
    GrooveEvent("hi_hat", 5, velocity=50),
    # off-8th hat
    GrooveEvent("hi_hat", 6, velocity=65),
    # syncopated kick decoration
    GrooveEvent("kick", 7, velocity=80),
    # Beat 3 — kick + hat
    GrooveEvent("kick", 8, velocity=108),
    GrooveEvent("hi_hat", 8, velocity=75),
    # off-8th hat
    GrooveEvent("hi_hat", 10, velocity=65),
    # Beat 4 — snare + hat
    GrooveEvent("snare", 12, velocity=107),
    GrooveEvent("hi_hat", 12, velocity=75),
    # 16th ghost snare
    GrooveEvent("snare", 13, velocity=20, articulation="ghost"),
    # off-8th hat
    GrooveEvent("hi_hat", 14, velocity=65),
    # extra kick lead
    GrooveEvent("kick", 15, velocity=85),
]


def _header(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _grid(events: list[GrooveEvent]) -> None:
    """Print a visual 16th-note grid."""
    bar: list[list[str]] = [[] for _ in range(16)]
    for e in events:
        pos = e.grid_position % 16
        label = f"{e.instrument[:3]:>3s}:{e.velocity}"
        if e.articulation == "ghost":
            label = f"{e.instrument[:3]:>3s}:Ghost"
        elif e.source_role == "ghost":
            label = f"{e.instrument[:3]:>3s}:Gsrc"
        bar[pos].append(label)

    print(f"  {'-' * 55}")
    print(f"  pos | events")
    print(f"  {'-' * 55}")
    for pos in range(16):
        beat_marker = ">" if pos % 4 == 0 else " "
        entries = ", ".join(bar[pos]) if bar[pos] else "·"
        print(f"  {beat_marker}{pos:2d}  | {entries}")
    print(f"  {'-' * 55}")
    print(f"  {len(events)} events total")


def main() -> int:
    print("BEHAVIOUR-DRIVEN OUTPUT SHAPING")
    print("How BehaviourIntent transforms a drum pattern")

    shaper = BehaviourOutputShaper()

    # Original
    _header("ORIGINAL GROOVE")
    _grid(_ORIGINAL)

    # MAINTAIN
    _header("MAINTAIN (preserve pocket)")
    _grid(shaper.shape(_ORIGINAL, BehaviourIntent.MAINTAIN))

    # REDUCE
    _header("REDUCE (player too busy -> simplify)")
    _grid(shaper.shape(_ORIGINAL, BehaviourIntent.REDUCE))

    # ANCHOR
    _header("ANCHOR (player uncertain -> clear pulse)")
    _grid(shaper.shape(_ORIGINAL, BehaviourIntent.ANCHOR))

    # BUILD
    _header("BUILD (rising energy -> more intensity)")
    _grid(shaper.shape(_ORIGINAL, BehaviourIntent.BUILD))

    # ENTER_SOFT
    _header("ENTER_SOFT (controlled entry)")
    _grid(shaper.shape(_ORIGINAL, BehaviourIntent.ENTER_SOFT))

    # BAIL
    _header("BAIL (silence -> suppress)")
    events = shaper.shape(_ORIGINAL, BehaviourIntent.BAIL)
    print(f"  Result: {len(events)} events (empty = suppressed)")

    # ANCHOR from silence
    _header("ANCHOR from empty input")
    _grid(shaper.shape([], BehaviourIntent.ANCHOR))

    print(f"\n{'=' * 60}")
    print("Output Shaping ready.")
    print("Behaviour intent now directly shapes drum output.")
    print(f"{'=' * 60}")
    return 0


if __name__ == "__main__":
    sys.exit(main())