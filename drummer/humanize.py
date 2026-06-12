"""Controlled humanization for synthetic drum events.

Adds small, instrument-appropriate timing and velocity variations
to make programmed drums feel more organic without being sloppy.

Supports two calling conventions:

1. **Legacy** (backward-compatible):
   ``humanize_events(events, timing_amount_ms=8.0, velocity_amount=6, seed=42)``

2. **Model-aware**:
   ``humanize_events(events, humanize_rules=MOTORIK_TIGHT_MODEL.humanize, seed=42)``

When *humanize_rules* is provided its per-instrument dicts are used;
otherwise the legacy scalars and module-level defaults apply.
"""

from __future__ import annotations

import random
from typing import Any

from drummer.rules import HumanizeRules

# Legacy per-instrument timing spread (std-dev in ms) — used only when
# no HumanizeRules instance is passed.
_INSTRUMENT_TIMING_MS: dict[str, float] = {
    "kick": 2.0,  # most stable
    "snare": 4.0,  # slightly more movement
    "hat": 6.0,  # most freedom
}

# Legacy per-instrument velocity variation (std-dev in raw MIDI velocity units)
_INSTRUMENT_VELOCITY_AMOUNT: dict[str, int] = {
    "kick": 3,
    "snare": 4,
    "hat": 6,
}


def humanize_events(
    events: list[dict[str, Any]],
    timing_amount_ms: float = 8.0,
    velocity_amount: int = 6,
    seed: int | None = None,
    humanize_rules: HumanizeRules | None = None,
) -> list[dict[str, Any]]:
    """Return a new list of events with subtle humanization applied.

    Parameters
    ----------
    events : list[dict]
        Original scheduled events. Each dict must have at least
        ``timestamp``, ``velocity``, and ``instrument`` keys.
    timing_amount_ms : float
        Global ceiling for timing randomisation (ms). Used as a fallback
        when *humanize_rules* is not provided or when an instrument is
        not found in the rules' ``timing_jitter_ms``.
    velocity_amount : int
        Global ceiling for velocity randomisation. Used as a fallback
        when *humanize_rules* is not provided or when an instrument is
        not found in the rules' ``velocity_jitter``.
    seed : int or None
        If set, the RNG is seeded so results are deterministic.
    humanize_rules : HumanizeRules or None
        If provided, per-instrument ``timing_bias_ms``, ``timing_jitter_ms``
        and ``velocity_jitter`` are read from this instance.  Instruments
        not found in the dicts fall back to the global *timing_amount_ms*
        / *velocity_amount* values.

    Returns
    -------
    list[dict]
        New list of events with humanized timestamps and velocities.
        Every other field is copied verbatim.
    """
    rng = random.Random(seed)

    # Decide which per-instrument tables to use.
    if humanize_rules is not None:
        timing_bias = humanize_rules.timing_bias_ms
        timing_jitter = humanize_rules.timing_jitter_ms
        velocity_jitter = humanize_rules.velocity_jitter
        fallback_timing = humanize_rules.timing_amount_ms
        fallback_velocity = humanize_rules.velocity_amount
    else:
        timing_bias = {}
        timing_jitter = _INSTRUMENT_TIMING_MS
        velocity_jitter = _INSTRUMENT_VELOCITY_AMOUNT
        fallback_timing = timing_amount_ms
        fallback_velocity = velocity_amount

    out: list[dict[str, Any]] = []
    for ev in events:
        inst = ev.get("instrument", "")

        # Look up per-instrument values, falling back to globals.
        bias_ms = timing_bias.get(inst, 0.0)
        jitter_ms = timing_jitter.get(inst, fallback_timing)
        vel_jitter = velocity_jitter.get(inst, fallback_velocity)

        # Timing offset in seconds: bias + uniform jitter
        offset_s = (bias_ms + rng.uniform(-jitter_ms, jitter_ms)) / 1000.0
        new_ts = max(0.0, ev["timestamp"] + offset_s)

        # Velocity offset (uniform jitter within ±vel_jitter)
        offset_vel = rng.randint(-vel_jitter, vel_jitter)
        new_vel = max(0, min(127, ev["velocity"] + offset_vel))

        new_ev = dict(ev)
        new_ev["timestamp"] = new_ts
        new_ev["velocity"] = new_vel
        out.append(new_ev)

    # Sort by timestamp to keep event order sensible
    out.sort(key=lambda e: e["timestamp"])
    return out