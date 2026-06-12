"""Demonstrate model-aware humanization with all three preset models.

Takes the same scheduled events and humanizes them with each model's
HumanizeRules, then prints a comparison table.
"""

from __future__ import annotations

from drummer.humanize import humanize_events
from drummer.models import (
    MOTORIK_TIGHT_MODEL,
    SIMPLE_ROCK_SAFE_MODEL,
    SPARSE_POSTPUNK_MODEL,
)
from models import KICK, SNARE, CLOSED_HAT

SEED = 42
MODELS = [
    ("Motorik Tight", MOTORIK_TIGHT_MODEL),
    ("Simple Rock (Safe)", SIMPLE_ROCK_SAFE_MODEL),
    ("Sparse Post-Punk", SPARSE_POSTPUNK_MODEL),
]


def _make_events() -> list[dict]:
    """Build a bar of 16th-note groove events (simple rock pattern)."""
    return [
        {"timestamp": t / 8.0, "step": i, "instrument": inst, "note": note, "velocity": vel}
        for i, (t, inst, note, vel) in enumerate(
            [
                (0, "kick", KICK, 100),
                (1, "hat", CLOSED_HAT, 80),
                (2, "snare", SNARE, 100),
                (3, "hat", CLOSED_HAT, 80),
                (4, "kick", KICK, 100),
                (5, "hat", CLOSED_HAT, 80),
                (6, "snare", SNARE, 100),
                (7, "hat", CLOSED_HAT, 80),
                (8, "kick", KICK, 100),
                (9, "hat", CLOSED_HAT, 80),
                (10, "snare", SNARE, 100),
                (11, "hat", CLOSED_HAT, 80),
                (12, "kick", KICK, 100),
                (13, "hat", CLOSED_HAT, 80),
                (14, "snare", SNARE, 100),
                (15, "hat", CLOSED_HAT, 80),
            ]
        )
    ]


def _fmt_ts(seconds: float) -> str:
    return f"{seconds * 1000:8.2f}ms"


def _format_table_row(
    label: str,
    orig_ts: list[float],
    hum_ts: list[float],
    orig_vel: list[int],
    hum_vel: list[int],
    width: int = 8,
) -> str:
    ts_parts = [f"{_fmt_ts(o)} -> {_fmt_ts(h)}" for o, h in zip(orig_ts[:width], hum_ts[:width])]
    vel_parts = [f"{o:3d} -> {h:3d}" for o, h in zip(orig_vel[:width], hum_vel[:width])]
    return (
        f"{label:22s} | "
        f"Timestamps: {' | '.join(ts_parts)}\n"
        f"{'':22s} | "
        f"Velocities: {' | '.join(vel_parts)}"
    )


def main() -> None:
    events = _make_events()
    orig_times = [e["timestamp"] for e in events]
    orig_vels = [e["velocity"] for e in events]

    print("=" * 100)
    print("Model-Aware Humanization Comparison")
    print(f"(seed={SEED}, {len(events)} events, showing first 6 per model)")
    print("=" * 100)
    print()

    for name, model in MODELS:
        humanized = humanize_events(events, humanize_rules=model.humanize, seed=SEED)
        hum_times = [e["timestamp"] for e in humanized]
        hum_vels = [e["velocity"] for e in humanized]
        print(_format_table_row(name, orig_times, hum_times, orig_vels, hum_vels, width=6))
        print()

    # Summary statistics
    print("-" * 100)
    print("Summary statistics per model:")
    print(f"{'Model':22s} | {'Avg |dt|':>9s} | {'Max |dt|':>9s} | {'Avg |dv|':>9s} | {'Max |dv|':>9s}")
    print("-" * 100)

    for name, model in MODELS:
        humanized = humanize_events(events, humanize_rules=model.humanize, seed=SEED)
        timing_deltas = [
            abs(h["timestamp"] - o["timestamp"])
            for o, h in zip(events, humanized)
        ]
        velocity_deltas = [
            abs(h["velocity"] - o["velocity"])
            for o, h in zip(events, humanized)
        ]
        avg_t = sum(timing_deltas) / len(timing_deltas) * 1000  # to ms
        max_t = max(timing_deltas) * 1000
        avg_v = sum(velocity_deltas) / len(velocity_deltas)
        max_v = max(velocity_deltas)
        print(
            f"{name:22s} | {avg_t:9.2f}ms | {max_t:9.2f}ms | "
            f"{avg_v:9.2f}  | {max_v:9.2f}"
        )

    print()
    print("Musical interpretation:")
    print("  Motorik Tight      -- smallest timing/velocity changes, very locked")
    print("  Simple Rock (Safe) -- moderate changes, stable and reliable")
    print("  Sparse Post-Punk   -- loosest feel, more timing/velocity variation")


if __name__ == "__main__":
    main()