"""
MIDI inspection/debug tool.

Given a MIDI file path, prints a clear timing report to reveal:
- Whether the file is slowing down
- Wrong delta times
- Incorrect note placement

Usage:
    python tools/inspect_midi.py path/to/file.mid
"""

import sys
from collections import defaultdict
from typing import Optional

try:
    import mido
except ImportError:
    print("ERROR: mido is required. Install with: pip install mido")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Drum note name lookup
# ---------------------------------------------------------------------------

DRUM_NAMES: dict[int, str] = {
    36: "kick",
    38: "snare",
    42: "closed_hat",
    46: "open_hat",
    49: "crash",
    51: "ride",
    76: "metronome_click",
}


def drum_name(note: int) -> str:
    """Return drum name for *note*, or 'note_{note}' if unknown."""
    return DRUM_NAMES.get(note, f"note_{note}")


# ---------------------------------------------------------------------------
# Core inspection data
# ---------------------------------------------------------------------------


def extract_timing_data(filepath: str) -> dict:
    """
    Read a MIDI file and return a dictionary with all timing information.

    Returns
    -------
    dict with keys:
        - ticks_per_beat
        - tempo (int, microseconds per beat) or None
        - bpm (float) or None
        - tracks: list of track dicts (see below)
        - note_on_count (int)
        - note_on_events (list of dicts, first 50)

    Each track dict:
        - index
        - name
        - note_on_events (list of dicts with absolute_tick, delta_from_previous,
          note, drum_name, velocity, channel)
    """
    mid = mido.MidiFile(filepath)
    ticks_per_beat: int = mid.ticks_per_beat

    # --- discover tempo ---------------------------------------------------
    tempo: Optional[int] = None  # microseconds per quarter note
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                tempo = msg.tempo
                break
        if tempo is not None:
            break

    bpm: Optional[float] = None
    if tempo is not None:
        bpm = mido.tempo2bpm(tempo)

    # --- walk every track -------------------------------------------------
    all_note_on: list[dict] = []
    tracks_info: list[dict] = []

    for track_idx, track in enumerate(mid.tracks):
        absolute_tick: float = 0.0
        last_note_on_tick: Optional[float] = None
        track_events: list[dict] = []

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "note_on":
                delta = 0.0
                if last_note_on_tick is not None:
                    delta = absolute_tick - last_note_on_tick
                last_note_on_tick = absolute_tick

                event = {
                    "absolute_tick": absolute_tick,
                    "delta_from_previous_note_on": delta,
                    "note": msg.note,
                    "drum_name": drum_name(msg.note),
                    "velocity": msg.velocity,
                    "channel": msg.channel,
                    "track": track_idx,
                }
                track_events.append(event)
                all_note_on.append(event)

        # track name
        track_name = f"Track {track_idx}"
        for msg in track:
            if msg.type == "track_name":
                track_name = msg.name
                break

        tracks_info.append(
            {
                "index": track_idx,
                "name": track_name,
                "note_on_events": track_events,
            }
        )

    # --- spacing analysis -------------------------------------------------
    note_42_spacings: list[float] = []
    note_76_spacings: list[float] = []
    note_36_positions: list[float] = []
    note_38_positions: list[float] = []

    prev_42: Optional[float] = None
    prev_76: Optional[float] = None

    for ev in all_note_on:
        if ev["note"] == 42:
            if prev_42 is not None:
                note_42_spacings.append(ev["absolute_tick"] - prev_42)
            prev_42 = ev["absolute_tick"]
        if ev["note"] == 76:
            if prev_76 is not None:
                note_76_spacings.append(ev["absolute_tick"] - prev_76)
            prev_76 = ev["absolute_tick"]
        if ev["note"] == 36:
            note_36_positions.append(ev["absolute_tick"])
        if ev["note"] == 38:
            note_38_positions.append(ev["absolute_tick"])

    return {
        "ticks_per_beat": ticks_per_beat,
        "tempo": tempo,
        "bpm": bpm,
        "track_count": len(mid.tracks),
        "note_on_count": len(all_note_on),
        "note_on_events": all_note_on[:50],
        "tracks": tracks_info,
        "tick_positions": {  # keep full sets for tests
            "hi_hat_42": [ev["absolute_tick"] for ev in all_note_on if ev["note"] == 42],
            "metronome_76": [ev["absolute_tick"] for ev in all_note_on if ev["note"] == 76],
            "kick_36": note_36_positions,
            "snare_38": note_38_positions,
        },
        "hi_hat_spacings": note_42_spacings,
        "metronome_spacings": note_76_spacings,
    }


# ---------------------------------------------------------------------------
# Problem detection functions
# ---------------------------------------------------------------------------


def detect_increasing_spacing(spacings: list[float], label: str = "hi-hat") -> list[str]:
    """
    Return warnings if spacings appear to increase over time (first half vs
    second half average).
    """
    warnings: list[str] = []
    if len(spacings) < 4:
        return warnings
    mid = len(spacings) // 2
    first_half = spacings[:mid]
    second_half = spacings[mid:]
    avg_first = sum(first_half) / len(first_half)
    avg_second = sum(second_half) / len(second_half)
    if avg_second > avg_first * 1.15:  # 15% or more increase
        warnings.append(
            f"WARNING: {label} spacing appears to increase "
            f"(first-half avg {avg_first:.1f}, second-half avg {avg_second:.1f}); "
            f"exporter may be using absolute ticks as delta times."
        )
    return warnings


def detect_large_gaps(
    events: list[dict], threshold_ticks: float = 100000
) -> list[str]:
    """Return warnings for note_on gaps larger than *threshold_ticks*."""
    warnings: list[str] = []
    for ev in events:
        if ev["delta_from_previous_note_on"] > threshold_ticks:
            warnings.append(
                f"WARNING: large timing gap at tick {ev['absolute_tick']:.0f} "
                f"(delta {ev['delta_from_previous_note_on']:.0f} ticks, "
                f"note {ev['note']} {ev['drum_name']})"
            )
    return warnings


def check_tempo_present(tempo: Optional[int]) -> list[str]:
    """Return a warning if *tempo* is None."""
    if tempo is None:
        return ["WARNING: no tempo meta message found."]
    return []


def run_all_checks(data: dict) -> list[str]:
    """Run all problem-detection checks and return a list of warning strings."""
    warnings: list[str] = []

    # tempo
    warnings.extend(check_tempo_present(data["tempo"]))

    # increasing hi-hat spacing
    warnings.extend(detect_increasing_spacing(data["hi_hat_spacings"], "hi-hat"))

    # large gaps
    warnings.extend(detect_large_gaps(data["note_on_events"]))

    return warnings


# ---------------------------------------------------------------------------
# Printer
# ---------------------------------------------------------------------------


def pretty_print(data: dict, warnings: list[str]) -> None:
    """Print the inspection report to stdout."""

    print("=" * 60)
    print("MIDI INSPECTION REPORT")
    print("=" * 60)

    print(f"\n  ticks_per_beat:            {data['ticks_per_beat']}")
    if data["bpm"] is not None and data["tempo"] is not None:
        print(f"  tempo (us/qn):             {data['tempo']}")
        print(f"  BPM:                       {data['bpm']:.2f}")
    else:
        print(f"  tempo:                     Not set (default 500000 us/qn -> 120 BPM)")

    print(f"  track count:               {data['track_count']}")
    print(f"  total note_on count:       {data['note_on_count']}")

    print(f"\n{'-' * 60}")
    print("First 50 note_on events:")
    print(f"{'-' * 60}")
    header = (
        f"{'abs_tick':>10}  {'delta_prev':>10}  {'note':>4}  "
        f"{'drum':<20}  {'vel':>3}  {'ch':>2}"
    )
    print(header)
    print("-" * len(header))
    for ev in data["note_on_events"]:
        print(
            f"{ev['absolute_tick']:>10.0f}  "
            f"{ev['delta_from_previous_note_on']:>10.0f}  "
            f"{ev['note']:>4}  "
            f"{ev['drum_name']:<20}  "
            f"{ev['velocity']:>3}  "
            f"{ev['channel']:>2}"
        )

    # --- spacing analysis -------------------------------------------------
    print(f"\n{'-' * 60}")
    print("Spacing analysis:")
    print(f"{'-' * 60}")

    hh = data["hi_hat_spacings"]
    mm = data["metronome_spacings"]
    kk = data["tick_positions"]["kick_36"]
    ss = data["tick_positions"]["snare_38"]

    if hh:
        print(f"  hi-hat (42) spacings:  {len(hh)} intervals")
        print(f"    min: {min(hh):.0f}  max: {max(hh):.0f}  "
              f"avg: {sum(hh)/len(hh):.1f}")
        if len(hh) >= 2:
            print(f"    first 5: {[f'{s:.0f}' for s in hh[:5]]}")
            print(f"    last  5: {[f'{s:.0f}' for s in hh[-5:]]}")
    else:
        print(f"  hi-hat (42):  no events")

    if mm:
        print(f"  metronome (76) spacings:  {len(mm)} intervals")
        print(f"    min: {min(mm):.0f}  max: {max(mm):.0f}  "
              f"avg: {sum(mm)/len(mm):.1f}")
    else:
        print(f"  metronome (76):  no events")

    if kk:
        print(f"  kick (36) positions:      {len(kk)} events")
        print(f"    first 5: {[f'{p:.0f}' for p in kk[:5]]}")
    else:
        print(f"  kick (36):  no events")

    if ss:
        print(f"  snare (38) positions:     {len(ss)} events")
        print(f"    first 5: {[f'{p:.0f}' for p in ss[:5]]}")
    else:
        print(f"  snare (38):  no events")

    # --- warnings ---------------------------------------------------------
    if warnings:
        print(f"\n{'-' * 60}")
        print("Detected problems:")
        print(f"{'-' * 60}")
        for w in warnings:
            print(f"  {w}")
    else:
        print(f"\n  No problems detected.")

    print(f"\n{'=' * 60}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    if not argv:
        print("Usage: python tools/inspect_midi.py path/to/file.mid")
        sys.exit(1)

    filepath = argv[0]
    data = extract_timing_data(filepath)
    warnings = run_all_checks(data)
    pretty_print(data, warnings)


if __name__ == "__main__":
    main()