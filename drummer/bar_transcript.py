"""Bar Transcript — compact, information-dense per-bar diagnostic report.

Produces Lucy-readable bar-by-bar summaries showing exactly what the
drummer played: instrument counts, velocities, note positions, and
diagnostic flags.

Pure module — no MIDI, no playback, no side effects.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from typing import Any


# ---------------------------------------------------------------------------
# Instrument abbreviation mapping
# ---------------------------------------------------------------------------

_INST_ABBREV: dict[str, str] = {
    "kick": "K",
    "snare": "S",
    "hi_hat": "H",
    "closed_hat": "H",
    "open_hat": "H",
    "ride": "R",
    "crash": "C",
    "toms": "T",
    "tom": "T",
    "hi_tom": "T",
    "mid_tom": "T",
    "low_tom": "T",
}


def _abbrev(instrument: str) -> str:
    """Return compact instrument abbreviation."""
    lower = instrument.lower().replace(" ", "_")
    return _INST_ABBREV.get(lower, instrument[:1].upper())


def _count_instrument(events: list, inst: str) -> int:
    """Count events matching *inst* (fuzzy match)."""
    lower = inst.lower()
    count = 0
    for e in events:
        ei = e.instrument.lower().replace(" ", "_")
        if lower == "hat":
            if ei in ("hi_hat", "closed_hat", "open_hat", "hat", "hh", "ch", "oh"):
                count += 1
        elif lower == "crash":
            if ei in ("crash", "cr", "cc"):
                count += 1
        elif lower == "ride":
            if ei in ("ride", "rd", "rc"):
                count += 1
        elif lower == "snare":
            if ei in ("snare", "sn", "sd", "rimshot"):
                count += 1
        elif lower == "kick":
            if ei in ("kick", "kik"):
                count += 1
        elif lower == "toms":
            if "tom" in ei:
                count += 1
    return count


# ---------------------------------------------------------------------------
# Compact note position formatting
# ---------------------------------------------------------------------------


def _build_note_positions(events: list) -> str:
    """Build compact note position string like 'K@1:118,S@3:78,H@1&2&3&4:52'.

    Events are sorted by grid_position then instrument abbreviation.
    Consecutive same-instrument-same-velocity events are merged.
    """
    if not events:
        return "—"

    # Group by instrument abbreviation
    groups: dict[str, list] = defaultdict(list)
    for e in events:
        ab = _abbrev(e.instrument)
        groups[ab].append(e)

    parts: list[str] = []
    for ab in sorted(groups.keys()):
        evs = groups[ab]
        # Group by velocity, then sort by position within each velocity group
        by_vel: dict[int, list[int]] = defaultdict(list)
        for e in evs:
            by_vel[e.velocity].append(e.grid_position % 16)

        vel_parts: list[str] = []
        for vel in sorted(by_vel.keys()):
            positions = sorted(set(by_vel[vel]))
            pos_str = "&".join(str(p) for p in positions)
            vel_parts.append(f"{pos_str}:{vel}")

        parts.append(f"{ab}@{';'.join(vel_parts)}")

    return ",".join(parts)


# ---------------------------------------------------------------------------
# Flag computation
# ---------------------------------------------------------------------------


def _compute_flags(bar_events: list, bar_index: int, all_bars: list[dict]) -> list[str]:
    """Compute per-bar diagnostic flags.

    *all_bars* is a list of per-bar dicts with keys: event_count, kick, snare,
    hat, ride, crash, max_velocity.  Index aligns with bar_index.
    """
    flags: list[str] = []
    b = all_bars[bar_index]
    ec = b["event_count"]
    kick = b["kick"]
    snare = b["snare"]
    hat = b["hat"]
    ride = b["ride"]
    crash = b["crash"]
    maxv = b["max_velocity"]

    # LOUD_ISOLATED_KICK: single event, kick only, velocity > 100
    if ec == 1 and kick == 1 and maxv > 100:
        flags.append("LOUD_ISOLATED_KICK")
        # Check previous bar for REPEATED
        if bar_index > 0:
            prev = all_bars[bar_index - 1]
            if (prev["event_count"] == 1 and prev["kick"] == 1
                    and prev["max_velocity"] > 100):
                flags.append("REPEATED_LOUD_ISOLATED_KICK")

    # VERY_SPARSE
    if 1 <= ec <= 2:
        flags.append("VERY_SPARSE")

    # NO_SNARE
    if snare == 0 and ec > 0:
        flags.append("NO_SNARE")

    # NO_HATS
    if hat == 0 and ec > 0:
        flags.append("NO_HATS")

    # HATS_8THS
    if hat >= 8:
        flags.append("HATS_8THS")

    # HATS_QUARTERS
    if 3 <= hat <= 5:
        flags.append("HATS_QUARTERS")

    # HAT_DENSITY_DROPPED
    if bar_index > 0:
        prev_hat = all_bars[bar_index - 1].get("hat", 0)
        if prev_hat >= 4 and hat <= prev_hat - 3:
            flags.append("HAT_DENSITY_DROPPED")

    # RECOVERY_GETS_THINNER
    if bar_index > 0:
        prev_ec = all_bars[bar_index - 1].get("event_count", 0)
        if ec < prev_ec and ec > 0:
            flags.append("RECOVERY_GETS_THINNER")

    # CRASH_PRESENT
    if crash > 0:
        flags.append("CRASH_PRESENT")

    # BUSY_BAR
    if ec >= 12:
        flags.append("BUSY_BAR")

    # POSSIBLE_FILL
    if ec >= 10 and snare >= 1 and (kick + snare) >= 4:
        flags.append("POSSIBLE_FILL")

    # TIMING_SPREAD_WIDE (only if timing_offset_ms data is present on events)
    if bar_events:
        offsets = [abs(e.timing_offset_ms) for e in bar_events if hasattr(e, "timing_offset_ms")]
        if offsets and max(offsets) > 8.0:
            flags.append("TIMING_SPREAD_WIDE")

    return flags


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BarLine:
    """One bar transcript line."""

    bar: int
    section: str
    rendered_intent: str
    event_count: int
    kick_count: int
    snare_count: int
    hat_count: int
    ride_count: int
    crash_count: int
    max_velocity: int
    avg_velocity: int
    note_positions: str
    flags: list[str] = field(default_factory=list)


@dataclass
class BarTranscript:
    """Complete bar transcript for a playtest run."""

    scenario: str = ""
    preset: str = ""
    variation: str = ""
    bars: int = 0
    total_events: int = 0
    suspicious: list[str] = field(default_factory=list)
    bar_lines: list[BarLine] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_bar_transcript(
    global_events: list,
    raw_diags: list[dict],
    scenario: str = "",
    preset: str = "",
    variation: str = "",
) -> BarTranscript:
    """Build a BarTranscript from global events and per-bar diagnostics.

    Parameters
    ----------
    global_events : list[GrooveEvent]
        All GrooveEvents from the run.
    raw_diags : list[dict]
        Per-bar diagnostic records (must include bar, section, rendered_intent).
    scenario : str
        Scenario name (e.g. "drop").
    preset : str
        Preset name (e.g. "cautious").
    variation : str
        Variation name (e.g. "deliberate_sparse").

    Returns
    -------
    BarTranscript
    """
    # Group events by bar
    bar_events: dict[int, list] = defaultdict(list)
    for e in global_events:
        bar_events[e.bar_index].append(e)

    # Build per-bar data dicts first (needed for cross-bar flags)
    all_bar_data: list[dict] = []
    n_bars = len(raw_diags)
    for bar_idx in range(n_bars):
        evs = bar_events.get(bar_idx, [])
        kick = _count_instrument(evs, "kick")
        snare = _count_instrument(evs, "snare")
        hat = _count_instrument(evs, "hat")
        ride = _count_instrument(evs, "ride")
        crash = _count_instrument(evs, "crash")
        maxv = max((e.velocity for e in evs), default=0)
        avgv = round(sum(e.velocity for e in evs) / len(evs)) if evs else 0
        all_bar_data.append({
            "bar": bar_idx,
            "event_count": len(evs),
            "kick": kick,
            "snare": snare,
            "hat": hat,
            "ride": ride,
            "crash": crash,
            "max_velocity": maxv,
            "avg_velocity": avgv,
        })

    # Build BarLines with flags
    bar_lines: list[BarLine] = []
    suspicious: list[str] = []
    total_events = 0

    for bar_idx in range(n_bars):
        diag = raw_diags[bar_idx]
        bd = all_bar_data[bar_idx]
        evs = bar_events.get(bar_idx, [])
        total_events += bd["event_count"]

        section = diag.get("section", "?")
        rendered_intent = diag.get("rendered_intent", diag.get("intent", "?"))

        flags = _compute_flags(evs, bar_idx, all_bar_data)

        if flags:
            flag_desc = "; ".join(flags).lower().replace("_", " ")
            suspicious.append(f"bar {bar_idx} {flag_desc}")

        note_pos = _build_note_positions(evs)

        bar_lines.append(BarLine(
            bar=bar_idx,
            section=section,
            rendered_intent=rendered_intent,
            event_count=bd["event_count"],
            kick_count=bd["kick"],
            snare_count=bd["snare"],
            hat_count=bd["hat"],
            ride_count=bd["ride"],
            crash_count=bd["crash"],
            max_velocity=bd["max_velocity"],
            avg_velocity=bd["avg_velocity"],
            note_positions=note_pos,
            flags=flags,
        ))

    return BarTranscript(
        scenario=scenario,
        preset=preset,
        variation=variation,
        bars=n_bars,
        total_events=total_events,
        suspicious=suspicious,
        bar_lines=bar_lines,
    )


# ---------------------------------------------------------------------------
# Text renderer
# ---------------------------------------------------------------------------


def render_bar_transcript_text(transcript: BarTranscript) -> str:
    """Render a BarTranscript as a compact text report."""
    lines: list[str] = []

    # Header
    lines.append(f"BAR TRANSCRIPT: {transcript.scenario}/{transcript.preset}/{transcript.variation}")
    lines.append("")

    # Summary
    sus_count = len(transcript.suspicious)
    lines.append(f"SUMMARY: bars={transcript.bars} total_events={transcript.total_events} suspicious={sus_count}")
    if transcript.suspicious:
        lines.append(f"SUSPICIOUS: {'; '.join(transcript.suspicious)}")
    lines.append("")

    # Column header
    lines.append(
        f"{'Bar':>3s} {'Section':>12s} {'Intent':>12s} {'ev':>3s} "
        f"{'K':>2s} {'S':>2s} {'H':>2s} {'R':>2s} {'C':>2s} "
        f"{'maxv':>4s} {'avgv':>4s}  {'notes':<40s}  {'flags'}"
    )
    lines.append("-" * 140)

    # Bar lines
    for bl in transcript.bar_lines:
        flag_str = " ".join(bl.flags) if bl.flags else ""
        lines.append(
            f"{bl.bar:3d} {bl.section:>12s} {bl.rendered_intent:>12s} "
            f"{bl.event_count:3d} {bl.kick_count:2d} {bl.snare_count:2d} "
            f"{bl.hat_count:2d} {bl.ride_count:2d} {bl.crash_count:2d} "
            f"{bl.max_velocity:4d} {bl.avg_velocity:4d}  "
            f"{bl.note_positions:<40s}  {flag_str}"
        )

    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# JSON renderer
# ---------------------------------------------------------------------------


def render_bar_transcript_json(transcript: BarTranscript) -> str:
    """Render a BarTranscript as a JSON string."""
    # Build dict manually for clean output
    data: dict[str, Any] = {
        "scenario": transcript.scenario,
        "preset": transcript.preset,
        "variation": transcript.variation,
        "bars": transcript.bars,
        "total_events": transcript.total_events,
        "suspicious_count": len(transcript.suspicious),
        "suspicious": transcript.suspicious,
        "bar_lines": [
            {
                "bar": bl.bar,
                "section": bl.section,
                "rendered_intent": bl.rendered_intent,
                "event_count": bl.event_count,
                "kick_count": bl.kick_count,
                "snare_count": bl.snare_count,
                "hat_count": bl.hat_count,
                "ride_count": bl.ride_count,
                "crash_count": bl.crash_count,
                "max_velocity": bl.max_velocity,
                "avg_velocity": bl.avg_velocity,
                "note_positions": bl.note_positions,
                "flags": bl.flags,
            }
            for bl in transcript.bar_lines
        ],
    }
    return json.dumps(data, indent=2)


# ---------------------------------------------------------------------------
# File output helpers
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def save_bar_transcript(
    transcript: BarTranscript,
    txt_path: str,
    json_path: str,
) -> None:
    """Save transcript as both text and JSON files."""
    _ensure_dir(txt_path)
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(render_bar_transcript_text(transcript))
        f.write("\n")

    _ensure_dir(json_path)
    with open(json_path, "w", encoding="utf-8") as f:
        f.write(render_bar_transcript_json(transcript))
        f.write("\n")