"""Demo: compare DrummerFeelEngine presets on a single 1-bar rock groove.

Outputs a readable table of every processed event (instrument, grid position,
velocity, timing offset, source role) for each built-in feel profile.

MIDI file export is NOT yet wired - the project has no MIDI file writing
dependency.  A TODO is left for when that is added.

Usage:
    python demo_feel_engine.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the package root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from drummer.feel import DrummerFeelEngine, DrummerProfile, GrooveEvent

# ---------------------------------------------------------------------------
# Fixed seed used for all profiles - deterministic comparison
# ---------------------------------------------------------------------------
FIXED_SEED = 42
TEMPO_BPM = 120.0

# Profiles to test (key matches DrummerProfile.builtin_profiles())
PROFILES_TO_DEMO: list[tuple[str, str]] = [
    ("machine", "Machine"),
    ("tight_rock", "Tight Rock"),
    ("laid_back", "Laid Back Pocket"),
    ("pushed_punk", "Pushed Punk"),
    ("loose_garage", "Loose Garage"),
]

# ---------------------------------------------------------------------------
# Base groove: 1 bar, 16th-note grid
#   kick  on  1  (pos 0) and 3  (pos 8)
#   snare on  2  (pos 4) and 4  (pos 12)
#   hi-hat on every 8th note  (pos 0, 2, 4, 6, 8, 10, 12, 14)
# ---------------------------------------------------------------------------
GROOVE_POSITIONS: list[tuple[str, int, int]] = [
    # (instrument, grid_position, velocity)
    ("kick", 0, 100),
    ("kick", 8, 100),
    ("snare", 4, 100),
    ("snare", 12, 100),
    ("hi_hat", 0, 90),
    ("hi_hat", 2, 80),
    ("hi_hat", 4, 85),
    ("hi_hat", 6, 80),
    ("hi_hat", 8, 90),
    ("hi_hat", 10, 80),
    ("hi_hat", 12, 85),
    ("hi_hat", 14, 80),
]


def build_base_groove() -> list[GrooveEvent]:
    """Return a list of GrooveEvents for the base 1-bar rock groove."""
    return [
        GrooveEvent(
            instrument=inst,
            grid_position=pos,
            bar_index=0,
            velocity=vel,
            probability=1.0,
            timing_offset_ms=0.0,
            articulation="default",
            source_role="main",
        )
        for inst, pos, vel in GROOVE_POSITIONS
    ]


def fmt_row(
    profile_name: str,
    instrument: str,
    grid_pos: int,
    velocity: int,
    timing_ms: float,
    source_role: str,
) -> str:
    """Format a single table row (fixed-width columns)."""
    return (
        f"{profile_name:<22s}  "
        f"{instrument:<10s}  "
        f"{grid_pos:3d}          "
        f"{velocity:3d}      "
        f"{timing_ms:+8.2f}   "
        f"{source_role:<10s}"
    )


def print_header() -> None:
    """Print the table header."""
    print()
    print("=" * 105)
    print(
        f"{'Profile':<22s}  {'Instrument':<10s}  "
        f"{'GridPos':<6s}  {'Vel':<5s}  "
        f"{'Off ms':<8s}  {'Source Role'}"
    )
    print("=" * 105)


def run_smoke_check() -> None:
    """Minimal smoke test: ensure each profile runs without raising."""
    base_events = build_base_groove()
    assert len(base_events) == 12, f"Expected 12 base events, got {len(base_events)}"

    for profile_key, profile_name in PROFILES_TO_DEMO:
        profile = DrummerProfile.get(profile_key)
        profile.seed = FIXED_SEED  # ensure deterministic
        engine = DrummerFeelEngine(profile)

        # Process a single bar
        result = engine.process(base_events, tempo_bpm=TEMPO_BPM)
        assert isinstance(result, list), f"Expected list, got {type(result)}"
        # At minimum we expect the 12 base events (ghosts may add more)
        assert len(result) >= 12, (
            f"{profile_name}: expected >= 12 events after processing, "
            f"got {len(result)}"
        )

        # Verify all returned events have required fields
        for ev in result:
            assert ev.instrument, "Missing instrument"
            assert isinstance(ev.grid_position, int)
            assert 1 <= ev.velocity <= 127, f"Velocity out of range: {ev.velocity}"
            assert isinstance(ev.timing_offset_ms, float)
            assert ev.source_role in ("main", "ghost", "fill", "crash", "transition"), (
                f"Unknown source_role: {ev.source_role}"
            )

    print("[OK] Smoke check passed: all profiles processed without errors.\n")


# Map profile keys to output filenames
PROFILE_FILENAMES: dict[str, str] = {
    "machine": "feel_machine.mid",
    "tight_rock": "feel_tight_rock.mid",
    "laid_back": "feel_laid_back_pocket.mid",
    "pushed_punk": "feel_pushed_punk.mid",
    "loose_garage": "feel_loose_garage.mid",
}


def main() -> None:
    """Run the feel-engine demo for all built-in profiles."""

    # --- Smoke check first ---
    run_smoke_check()

    base_events = build_base_groove()

    # --- Process each profile, print results, and export MIDI ---
    for profile_key, profile_name in PROFILES_TO_DEMO:
        profile = DrummerProfile.get(profile_key)
        profile.seed = FIXED_SEED
        engine = DrummerFeelEngine(profile)

        result = engine.process(base_events, tempo_bpm=TEMPO_BPM)

        # Sort events by grid position for clean reading
        result.sort(key=lambda e: (e.bar_index, e.grid_position, e.instrument))

        print_header()
        for ev in result:
            print(fmt_row(
                profile_name,
                ev.instrument,
                ev.grid_position,
                ev.velocity,
                ev.timing_offset_ms,
                ev.source_role,
            ))
        print()

        # --- MIDI export ---
        from drummer.midi_export import export_groove_events_to_midi

        midi_filename = PROFILE_FILENAMES[profile_key]
        export_groove_events_to_midi(
            result,
            output_path=midi_filename,
            tempo_bpm=TEMPO_BPM,
            ticks_per_beat=480,
        )
        print(f"  -> Exported {midi_filename}")
        print()

    # --- Summary ---
    print("-" * 105)
    print("MIDI files exported:")
    for profile_key, profile_name in PROFILES_TO_DEMO:
        print(f"  {PROFILE_FILENAMES[profile_key]:35s}  ({profile_name})")
    print("-" * 105)


if __name__ == "__main__":
    main()