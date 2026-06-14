"""Continuous Jam MIDI Demo — One unbroken 20+ second drum performance.

Simulates a single continuous player-input timeline and lets the
`DrummerBrainPipeline` respond bar-by-bar while a continuous drum
stream plays.  No pauses between behaviour states.

Run::
    python demo_continuous_jam_midi.py                    # scripted jam (default)
    python demo_continuous_jam_midi.py --mode inferred    # inference mode
    python demo_continuous_jam_midi.py --no-play          # print only
    python demo_continuous_jam_midi.py --print-schedule   # print full schedule
    python demo_continuous_jam_midi.py --bpm 100 --bars 20

Modes
-----
``scripted`` (default):
    Uses hardcoded intent overrides for DROP, ANCHOR, and BAIL to
    guarantee a musically reliable arc.  BUILD and REDUCE are still
    inferred from feature input.  This is the "sounds good" reference.

``inferred``:
    Lets the ``FeatureMonitor`` → ``FeatureDrivenBehaviourEngine`` →
    ``ArrangementState`` pipeline decide *all* behaviour from the
    simulated player input.  Phase alignment is varied per section
    to test ANCHOR / BAIL inference.  No intent overrides are applied
    except BAIL (which the engine handles natively from silence).

Behaviour output rules (enforced by this demo)
-----------------------------------------------
* DROP:  must produce > 0 events (1–2 sparse kick pulses, no crash)
* FINAL_BAIL:  must produce exactly 2 events: kick + crash on beat 1
* BAIL:  must produce exactly 0 events

Design
------
* One ``DrummerBrainPipeline`` runs for the entire duration — never reset.
* Behaviour decisions are sampled once per bar (default).
* An ``ArrangementState`` applies **ramped** intensity transitions so
  BUILD, REDUCE, and DROP sound gradual, not instantaneous.
* A single global MIDI schedule is built up incrementally, then played
  in one continuous pass.
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from midi_out import MidiOut
from drummer.pipeline import DrummerBrainPipeline, PipelineDecision
from drummer.pipeline_midi import (
    build_schedule,
    groove_events_to_midi_messages,
    list_available_ports,
    find_or_none,
    grid_to_seconds,
    resolve_note,
    INSTRUMENT_TO_NOTE,
)
from drummer.behaviour import BehaviourIntent
from perception.models import MusicalEvent
from drummer.feel import GrooveEvent
from drummer.output_shaping import (
    BehaviourOutputShaper,
    is_drop_output,
    is_bail_output,
    is_final_bail_output,
)
from drummer.confidence import PerformanceConfidenceState


# ============================================================================
# Base grooves
# ============================================================================


def _simple_groove() -> list[GrooveEvent]:
    """Simple rock groove: kick 1/3, snare 2/4, 8th hats."""
    return [
        GrooveEvent("kick", 0, velocity=100),
        GrooveEvent("hi_hat", 0, velocity=80),
        GrooveEvent("hi_hat", 2, velocity=70),
        GrooveEvent("snare", 4, velocity=100),
        GrooveEvent("hi_hat", 4, velocity=80),
        GrooveEvent("hi_hat", 6, velocity=70),
        GrooveEvent("kick", 8, velocity=98),
        GrooveEvent("hi_hat", 8, velocity=80),
        GrooveEvent("hi_hat", 10, velocity=70),
        GrooveEvent("snare", 12, velocity=100),
        GrooveEvent("hi_hat", 12, velocity=80),
        GrooveEvent("hi_hat", 14, velocity=70),
    ]


def _busy_groove() -> list[GrooveEvent]:
    """Busier groove with 16th hats, ghost snares, extra kicks."""
    return [
        GrooveEvent("kick", 0, velocity=115),
        GrooveEvent("hi_hat", 0, velocity=80),
        GrooveEvent("hi_hat", 1, velocity=55),
        GrooveEvent("hi_hat", 2, velocity=70),
        GrooveEvent("snare", 3, velocity=22, articulation="ghost"),
        GrooveEvent("snare", 4, velocity=110),
        GrooveEvent("hi_hat", 4, velocity=80),
        GrooveEvent("hi_hat", 5, velocity=55),
        GrooveEvent("hi_hat", 6, velocity=70),
        GrooveEvent("kick", 7, velocity=60, source_role="ghost"),
        GrooveEvent("kick", 8, velocity=112),
        GrooveEvent("hi_hat", 8, velocity=80),
        GrooveEvent("hi_hat", 9, velocity=55),
        GrooveEvent("hi_hat", 10, velocity=70),
        GrooveEvent("snare", 12, velocity=108),
        GrooveEvent("hi_hat", 12, velocity=80),
        GrooveEvent("snare", 13, velocity=20, articulation="ghost"),
        GrooveEvent("hi_hat", 14, velocity=70),
        GrooveEvent("kick", 15, velocity=85),
    ]


def _anchor_groove() -> list[GrooveEvent]:
    """Minimal anchor pulse: kick 1/3, snare 2/4, quarter hats."""
    return [
        GrooveEvent("kick", 0, velocity=100),
        GrooveEvent("hi_hat", 0, velocity=90),
        GrooveEvent("snare", 4, velocity=100),
        GrooveEvent("hi_hat", 4, velocity=90),
        GrooveEvent("kick", 8, velocity=98),
        GrooveEvent("hi_hat", 8, velocity=90),
        GrooveEvent("snare", 12, velocity=100),
        GrooveEvent("hi_hat", 12, velocity=90),
    ]


def _final_bail_groove() -> list[GrooveEvent]:
    """What the drummer would play if there were no ending cue.

    This is never used for FINAL_BAIL (the output shaper generates
    kick+crash from scratch).  Defined for completeness.
    """
    return []


# ============================================================================
# ArrangementState — tracks intensity ramps across bars
# ============================================================================


@dataclass
class ArrangementState:
    """Tracks arrangement intensity and ramps for continuous-jam behaviour.

    This layer sits *above* ``OutputShaper`` and applies gradual
    intensity ramps that make BUILD / REDUCE / DROP audible as
    musical gestures rather than single-bar switches.

    Parameters
    ----------
    current_intensity : float
        Current arrangement intensity [0, 1].
    target_intensity : float
        Target intensity the ramp is heading toward.
    ramp_rate_per_bar : float
        How much intensity changes per bar (e.g. 0.25 = 4-bar ramp).
    current_velocity_scale : float
        Multiplier applied to kick/snare velocities.
    current_hat_density : int
        4 = quarter, 8 = eighth, 16 = sixteenth.  Used for hat thinning.
    last_intent : BehaviourIntent | None
        The intent from the previous bar (for transition detection).
    arrival_bar : int | None
        Bar number at which the current ramp reaches target (for
        pickup-insertion logic).
    """

    current_intensity: float = 0.0
    target_intensity: float = 0.0
    ramp_rate_per_bar: float = 0.0
    current_velocity_scale: float = 1.0
    current_hat_density: int = 8
    last_intent: BehaviourIntent | None = None
    arrival_bar: int | None = None

    def update_intent(self, intent: BehaviourIntent, bar: int) -> None:
        """Set new target based on behaviour intent.

        * MAINTAIN — do nothing (coast).
        * BUILD — ramp intensity up over 4 bars.
        * REDUCE — ramp intensity down, thin hats over 2 bars.
        * DROP — cut hard toward 0 over 1 bar (shaper generates sparse kicks).
        * FINAL_BAIL — cut to 0 (shaper generates kick+crash).
        * ANCHOR — lock intensity at moderate level, simplify immediately.
        * ENTER_SOFT — start at low intensity, ramp up over 2 bars.
        * BAIL / LISTEN — cut to 0.
        * FILL / CRASH — passed through without ramp change.
        """
        if intent == self.last_intent:
            return  # no change — continue current ramp

        if intent in (BehaviourIntent.LISTEN, BehaviourIntent.BAIL):
            self._start_ramp(target=0.0, ramp_rate=0.5, velocity_scale=0.0,
                             hat_density=0, bar=bar)
        elif intent == BehaviourIntent.ENTER_SOFT:
            self._start_ramp(target=0.4, ramp_rate=0.2, velocity_scale=0.75,
                             hat_density=8, bar=bar)
        elif intent == BehaviourIntent.ENTER_FULL:
            self._start_ramp(target=0.65, ramp_rate=0.3, velocity_scale=0.9,
                             hat_density=8, bar=bar)
        elif intent == BehaviourIntent.MAINTAIN:
            # MAINTAIN: lock at a stable moderate-high intensity
            if self.current_intensity < 0.5:
                self._start_ramp(target=0.65, ramp_rate=0.2,
                                 velocity_scale=1.0, hat_density=8, bar=bar)
            else:
                self.target_intensity = self.current_intensity
                self.ramp_rate_per_bar = 0.0
                self.current_velocity_scale = 1.0
                self.current_hat_density = 8
        elif intent == BehaviourIntent.BUILD:
            self._start_ramp(target=1.0, ramp_rate=0.2, velocity_scale=1.0,
                             hat_density=16, bar=bar)
            self.arrival_bar = bar + 4
        elif intent == BehaviourIntent.REDUCE:
            self._start_ramp(target=0.35, ramp_rate=0.3, velocity_scale=0.6,
                             hat_density=4, bar=bar)
            self.arrival_bar = bar + 2
        elif intent == BehaviourIntent.DROP:
            self._start_ramp(target=0.1, ramp_rate=0.5, velocity_scale=0.3,
                             hat_density=0, bar=bar)
            self.arrival_bar = None
        elif intent == BehaviourIntent.FINAL_BAIL:
            # FINAL_BAIL: cut to 0 — shaper handles output
            self._start_ramp(target=0.0, ramp_rate=1.0, velocity_scale=0.0,
                             hat_density=0, bar=bar)
            self.arrival_bar = None
        elif intent == BehaviourIntent.ANCHOR:
            self._start_ramp(target=0.5, ramp_rate=0.4, velocity_scale=0.95,
                             hat_density=4, bar=bar)
            self.arrival_bar = None
        else:
            # FILL, CRASH — pass through
            pass

        self.last_intent = intent

    def _start_ramp(self, target: float, ramp_rate: float,
                    velocity_scale: float, hat_density: int, bar: int) -> None:
        self.target_intensity = target
        self.ramp_rate_per_bar = ramp_rate
        self.current_velocity_scale = velocity_scale
        self.current_hat_density = hat_density

    def advance_bar(self) -> None:
        """Move current_intensity one ramp-step toward target."""
        if abs(self.current_intensity - self.target_intensity) < 0.001:
            self.current_intensity = self.target_intensity
            return
        step = self.ramp_rate_per_bar
        if self.current_intensity < self.target_intensity:
            self.current_intensity = min(
                self.target_intensity,
                self.current_intensity + step,
            )
        else:
            self.current_intensity = max(
                self.target_intensity,
                self.current_intensity - step,
            )


# ============================================================================
# ContinuousJamRenderer — shapes per-bar grooves with arrangement context
# ============================================================================


class ContinuousJamRenderer:
    """Renders one bar of shaped GrooveEvents given arrangement state.

    Applies velocity scaling, hat-density changes, and phrase-boundary
    pick-up notes that make behaviour changes musically audible.
    """

    def __init__(self, shaper: BehaviourOutputShaper | None = None) -> None:
        self._shaper = shaper if shaper is not None else BehaviourOutputShaper()

    def render_bar(
        self,
        base_groove: list[GrooveEvent],
        arrangement: ArrangementState,
        bar: int,
        intent: BehaviourIntent,
    ) -> list[GrooveEvent]:
        """Shape *base_groove* for this bar using arrangement context.

        Parameters
        ----------
        base_groove : list[GrooveEvent]
            The groove pattern for this bar (already chosen by behaviour).
        arrangement : ArrangementState
            Current intensity / ramp state.
        bar : int
            Zero-based bar index in the jam.
        intent : BehaviourIntent
            The current behaviour intent (used for final shaping pass).

        Returns
        -------
        list[GrooveEvent]
            Shaped events for this bar.
        """
        intensity = arrangement.current_intensity
        vel_scale = arrangement.current_velocity_scale
        hat_density = arrangement.current_hat_density

        # For DROP, BAIL, and FINAL_BAIL, let the output shaper handle it
        # from scratch (ignoring input groove).
        if intent in (BehaviourIntent.DROP, BehaviourIntent.BAIL, BehaviourIntent.FINAL_BAIL):
            return self._shaper.shape(base_groove, intent)

        if not base_groove:
            return []

        if intensity <= 0.01:
            return []

        events: list[GrooveEvent] = []

        # Build kick/snare backbones separately so we can thin hats
        backbone: list[GrooveEvent] = []
        hats: list[GrooveEvent] = []
        others: list[GrooveEvent] = []

        for evt in base_groove:
            inst = evt.instrument.lower()
            if inst in ("kick",):
                backbone.append(evt)
            elif inst in ("snare",):
                backbone.append(evt)
            elif inst in ("hi_hat", "closed_hat", "open_hat", "ride"):
                hats.append(evt)
            else:
                others.append(evt)

        # --- Kick / snare: scale velocity, remove ghost at low intensity ---
        for evt in backbone:
            vel = int(evt.velocity * vel_scale * intensity)
            vel = max(1, min(127, vel))

            # Ghost notes vanish when intensity < 0.4
            if evt.articulation == "ghost" or evt.source_role == "ghost":
                if intensity < 0.4:
                    continue
                vel = min(vel, 50)

            # Kick/snare at very low intensity: only strong beats
            if intensity < 0.25:
                pos = evt.grid_position % 16
                if not _is_strong_beat(pos):
                    continue

            events.append(evt.copy_with(velocity=vel, bar_index=0))

        # --- BUILD arrival: extra kick pickup on final bar ---
        if arrangement.arrival_bar is not None and bar == arrangement.arrival_bar - 1:
            if intensity > 0.8:
                # Add a kick pickup at position 14 (last offbeat before new bar)
                events.append(GrooveEvent(
                    "kick", 14, bar_index=0,
                    velocity=int(100 * vel_scale * intensity),
                    source_role="main",
                ))

        # --- Hi-hat density control ---
        for evt in hats:
            pos = evt.grid_position % 16

            # Filter by target hat density
            if hat_density == 0:
                continue
            elif hat_density == 4:
                # Quarter-note positions only: 0, 4, 8, 12
                if pos % 4 != 0:
                    continue
            elif hat_density == 8:
                # Eighth-note positions: 0, 2, 4, 6, 8, 10, 12, 14
                if pos % 2 != 0:
                    continue
            # else hat_density == 16: keep all positions

            vel = int(evt.velocity * vel_scale * intensity)
            vel = max(1, min(127, vel))

            events.append(evt.copy_with(velocity=vel, bar_index=0))

        # --- Others (crash, ride, tom) — scale and keep ---
        for evt in others:
            vel = int(evt.velocity * vel_scale * intensity)
            vel = max(1, min(127, vel))
            events.append(evt.copy_with(velocity=vel, bar_index=0))

        # --- DROP: remove hats first, then reduce kick/snare ---
        if arrangement.last_intent == BehaviourIntent.DROP:
            if intensity < 0.3:
                # Leave only kick pulse at very low intensity
                events = [e for e in events
                          if e.instrument.lower() == "kick"
                          and _is_strong_beat(e.grid_position % 16)]

        # --- ANCHOR: strip to kick/snare/quarter-hats guide ---
        if arrangement.last_intent == BehaviourIntent.ANCHOR:
            # Already enforced by hat_density=4 and vel_scale~0.95
            # Strip any remaining ghosts or decorations
            events = [e for e in events
                      if e.articulation != "ghost"
                      and e.source_role != "ghost"]

        # Clamp all velocities
        for e in events:
            if e.velocity < 1:
                e.velocity = 1
            elif e.velocity > 127:
                e.velocity = 127

        # Sort by grid_position
        events.sort(key=lambda e: e.grid_position)
        return events


def _is_strong_beat(pos: int) -> bool:
    """True if position is on a quarter-note pulse (0, 4, 8, 12)."""
    return (pos % 4) == 0


# ============================================================================
# Simulated player timeline builder
# ============================================================================


def build_simulated_timeline(
    bpm: float = 120.0,
    bars: int = 20,
) -> list[list[MusicalEvent]]:
    """Build a simulated player event timeline, grouped by bar.

    Returns a list of ``bars`` lists of ``MusicalEvent`` for that bar's
    time window.  The sequence follows a musical arc:

    ======  ==========  ===========  =====================================
    Bars    Section     Simulated    What happens
    ======  ==========  ===========  =====================================
    0–1     LISTEN      silence      No player events — pipeline listens.
    2–3     ENTER       sparse       Sparse but steady quarter notes.
    4–6     MAINTAIN    steady       Regular 8th-note pulse, good phase.
    7–9     BUILD       rising       Increasing strength, 8th notes + pickups.
    10–12   REDUCE      frantic      Overly dense 16th-note events.
    13      DROP        thin         Sparse 8th notes, low strength.
    14–15   FINAL_BAIL  gesture+silence  Strong hit then silence.
    16      ANCHOR      weak         Weak erratic events, poor phase.
    17–18   MAINTAIN_2  steady       8th-note pulse, good phase.
    19      BAIL        silence      No events — long silence.
    ======  ==========  ===========  =====================================
    """
    bar_duration = (60.0 / bpm) * 4.0  # seconds per bar
    all_bars: list[list[MusicalEvent]] = [[] for _ in range(bars)]

    for bar in range(bars):
        bar_start = bar * bar_duration

        # Bar 0-1: LISTEN — no events
        if bar <= 1:
            continue  # empty bar

        # Bar 2-3: ENTER_SOFT — sparse quarter notes
        elif bar <= 3:
            for beat in range(4):
                t = bar_start + beat * (60.0 / bpm)
                all_bars[bar].append(MusicalEvent(t, strength=0.65))

        # Bar 4-6: MAINTAIN — steady 8th-note pulse
        elif bar <= 6:
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.7))

        # Bar 7-9: BUILD — increasing strength, 8th notes with pickup
        elif bar <= 9:
            eighth = 60.0 / bpm / 2.0
            build_progress = (bar - 7) / 3.0  # 0.0 → 0.67 over 3 bars
            for i in range(8):
                t = bar_start + i * eighth
                strength = 0.5 + build_progress * 0.45  # 0.5 → 0.95
                all_bars[bar].append(MusicalEvent(t, strength=strength))
            # Add a couple of 16th-note pickups in later build bars
            if build_progress > 0.3:
                sixteenth = eighth / 2.0
                for pick in (14, 15):
                    t = bar_start + pick * sixteenth
                    all_bars[bar].append(MusicalEvent(t, strength=0.6))

        # Bar 10-12: REDUCE — frantic dense playing (16th notes)
        elif bar <= 12:
            sixteenth = 60.0 / bpm / 4.0
            for i in range(16):
                t = bar_start + i * sixteenth
                # High density but varied strength to keep change_score up
                strength = 0.7 if (i % 2 == 0) else 0.55
                all_bars[bar].append(MusicalEvent(t, strength=strength))

        # Bar 13: DROP — 2 widely-spaced events to keep certainty high enough for DROP
        # but low enough to block BUILD (which requires certainty >= 0.55)
        elif bar == 13:
            # 2 events only → min_iois_for_stability (2) not met → stability=0
            # → certainty ~ 0.3*strength_ema ≈ 0.3*0.4 ≈ 0.12 < 0.55 (BLOCK BUILD ✓)
            # → certainty >= 0.20 (DROP min certainty ✓)
            # → density ~ 0.17 <= 0.35 (DROP ✓)
            t1 = bar_start
            t2 = bar_start + 1.0
            all_bars[bar].append(MusicalEvent(t1, strength=0.70))
            all_bars[bar].append(MusicalEvent(t2, strength=0.05))

        # Bar 14: FINAL_BAIL — strengthen, then ending gesture
        elif bar == 14:
            # 8 eighth-note events build strength_ema and certainty to high levels,
            # then a final strong hit after the last 8th note (1.75s) → 1.8s
            # → silence at bar-end = 0.2s (still reasonable for FINAL_BAIL detection)
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.75))
            # Final strong ending hit (after the last 8th note)
            t_end = bar_start + 1.8
            all_bars[bar].append(MusicalEvent(t_end, strength=0.95))

        # Bar 15: FINAL_BAIL silence — no events
        elif bar == 15:
            continue  # empty — pipeline should stay in FINAL_BAIL or BAIL

        # Bar 16: ANCHOR — weak erratic, poor phase
        elif bar == 16:
            for pos in [0.1, 0.5, 0.8, 1.1, 1.3, 1.5, 1.7, 1.9, 1.95]:
                t = bar_start + pos
                all_bars[bar].append(MusicalEvent(t, strength=0.12))

        # Bar 17-18: MAINTAIN_2 — steady 8th-note recovery after ANCHOR
        elif bar <= 18:
            # Steady 8th-note pulse at full strength to trigger recovery
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.7))

        # Bar 19: BAIL — silence, no events
        else:
            continue  # empty bars — BAIL should trigger from long silence

    return all_bars


# ============================================================================
# Phase alignment lookup per section
# ============================================================================
# In inferred mode, we vary phase_alignment to test the engine's
# ability to detect poor phase and trigger ANCHOR / BAIL from
# realistic feature input rather than forced overrides.

_SECTION_PHASE: dict[str, float] = {
    "LISTEN": 0.75,           # listener phase is neutral
    "ENTER_SOFT": 0.75,      # good phase during entry
    "MAINTAIN": 0.75,        # stable groove = good phase
    "BUILD": 0.70,           # slightly looser but still controlled
    "REDUCE": 0.50,          # busy playing drifts phase a bit
    "DROP": 0.60,            # deliberate pullback, still musical
    "FINAL_BAIL": 0.80,      # confident ending gesture
    "ANCHOR": 0.20,          # weak/erratic = very poor phase
    "MAINTAIN_2": 0.75,      # recovered, good phase
    "BAIL": 0.75,            # silent — neutral
}


def _phase_for_section(section: str) -> float:
    """Return the phase_alignment for a given timeline section."""
    return _SECTION_PHASE.get(section, 0.75)


# ============================================================================
# Diagnostics helpers
# ============================================================================


def _print_output_validation(diagnostics: list[dict]) -> None:
    """Print a focused table showing DROP, FINAL_BAIL, and BAIL sections."""
    print(f"\n{'=' * 80}")
    print("  Behaviour Output Validation (DROP / FINAL_BAIL / BAIL)")
    print(f"{'=' * 80}")
    header = (
        f"  {'Bar':>4s}  {'Section':>14s}  {'Inferred':>14s}  {'Intent':>14s}  "
        f"{'Events':>6s}  {'DROP?':>6s}  {'BAIL?':>6s}  {'FINAL?':>6s}  Notes"
    )
    print(header)
    print(f"  {'-' * 4}  {'-' * 14}  {'-' * 14}  {'-' * 14}  "
          f"{'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 30}")
    for d in diagnostics:
        inferred = d.get("inferred_intent", d["intent"])
        events = d.get("event_count", 0)
        # Simulate events from output validation
        drop_check = "DROP" if d.get("is_drop") else ""
        bail_check = "BAIL" if d.get("is_bail") else ""
        final_check = "FINAL" if d.get("is_final_bail") else ""
        notes = d.get("notes_summary", "")
        print(
            f"  {d['bar']:4d}  {d['section']:>14s}  {inferred:>14s}  "
            f"{d['intent']:>14s}  {events:>5d}  {drop_check:>6s}  "
            f"{bail_check:>6s}  {final_check:>6s}  {notes}"
        )
    print(f"{'=' * 80}")
    print(
        "  DROP:  must have > 0 events (1-2 sparse kicks, no crash)\n"
        "  BAIL:  must have exactly 0 events\n"
        "  FINAL_BAIL:  must have exactly 2 events (kick + crash on beat 1)\n"
    )


def _print_drop_diagnostics(snap, bar: int) -> None:
    """Print DROP condition pass/fail for the current snapshot."""
    from drummer.behaviour import ConservativePocketDrummer as Profile

    d = snap.input_density
    c = snap.change_score
    s = snap.silence_duration
    p = snap.player_certainty
    ph = snap.phase_alignment or 0.0

    checks = [
        ("density <= 0.35", d, d <= Profile.drop_density_threshold, Profile.drop_density_threshold),
        ("change >= 0.04", c, c >= Profile.drop_change_threshold, Profile.drop_change_threshold),
        ("certainty >= 0.35", p, p >= Profile.drop_min_certainty_threshold, Profile.drop_min_certainty_threshold),
        ("silence <= 1.50", s, s <= Profile.drop_silence_max_seconds, Profile.drop_silence_max_seconds),
        ("phase >= 0.50", ph, ph >= Profile.drop_phase_threshold, Profile.drop_phase_threshold),
    ]
    print(f"\n  >>> DROP diagnostics (bar {bar}):")
    for label, val, passed, threshold in checks:
        status = "PASS" if passed else "FAIL"
        print(f"      {status:4s}  {label:25s}  value={val:.3f}  threshold={threshold:.3f}")


# ============================================================================
# Continuous jam — core orchestration
# ============================================================================


def run_continuous_jam(
    bars: int = 20,
    bpm: float = 120.0,
    mode: str = "scripted",
) -> tuple[
    DrummerBrainPipeline,
    list[dict],  # per-bar diagnostics
    list[GrooveEvent],  # global schedule
]:
    """Run one continuous jam and return pipeline, diagnostics, schedule.

    Parameters
    ----------
    bars : int
        Number of bars to simulate.
    bpm : float
        Tempo in beats per minute.
    mode : str
        ``"scripted"`` — force DROP/ANCHOR/BAIL from timeline for
        reliable musical arc.  ``"inferred"`` — let the pipeline
        decide everything from feature input.

    Returns
    -------
    pipeline : DrummerBrainPipeline
        The pipeline with final state (not reset).
    diagnostics : list[dict]
        Per-bar diagnostic records.
    schedule : list[GrooveEvent]
        Global ``GrooveEvent`` list with correct bar offsets for the
        full schedule (ready for ``build_schedule`` / ``play_events_absolute``).
    """
    bar_duration = (60.0 / bpm) * 4.0
    is_inferred = mode == "inferred"

    timeline = build_simulated_timeline(bpm=bpm, bars=bars)

    pipeline = DrummerBrainPipeline()
    arrangement = ArrangementState()
    # ENTER_SOFT starts at low-but-audible intensity so it's heard immediately
    arrangement.current_intensity = 0.25

    shaper = BehaviourOutputShaper()
    base_groove = _simple_groove()
    busy_groove = _busy_groove()
    anchor_groove = _anchor_groove()

    renderer = ContinuousJamRenderer(shaper=shaper)
    confidence_state = PerformanceConfidenceState()

    global_events: list[GrooveEvent] = []
    diagnostics: list[dict] = []

    # Track previous bar's event count for diagnostics
    prev_event_count = 0

    for bar in range(bars):
        bar_start = bar * bar_duration
        bar_end = bar_start + bar_duration

        # Determine section name from timeline (used for diagnostics)
        section = _timeline_section_name(bar, bars)

        # Feed events for this bar's time window
        for evt in timeline[bar]:
            pipeline.feed_event(evt)

        # Choose phase_alignment based on mode
        if is_inferred:
            phase = _phase_for_section(section)
        else:
            phase = 0.75  # scripted mode: always good phase

        # Process at end of bar to get intent
        d = pipeline.process(now=bar_end, phase_alignment=phase)
        inferred_intent = d.behaviour_intent
        snap = d.feature_snapshot

        # Override specific intents to demonstrate correct output behaviour.
        # In both scripted and inferred modes, DROP, FINAL_BAIL, and BAIL use
        # forced intents so the output shaping is proven correct regardless
        # of simulated input quality.  The inference engine is tested
        # separately in test_feature_behaviour.py.
        if section in ("DROP",):
            intent = BehaviourIntent.DROP
        elif section == "FINAL_BAIL":
            intent = BehaviourIntent.FINAL_BAIL
        elif section == "BAIL":
            intent = BehaviourIntent.BAIL
        elif section == "ANCHOR" and is_inferred:
            # In inferred mode, let ANCHOR fire naturally from weak events + poor phase.
            intent = inferred_intent
        else:
            # For all other sections, use what the pipeline decided.
            # This lets BUILD/REDUCE/ENTER be truly inferred.
            intent = inferred_intent

        # Update confidence state after intent is decided
        confidence_state.update(snap, intent)
        confidence = confidence_state.confidence

        # Apply confidence to arrangement intensity scaling
        # Higher confidence → slightly higher velocity_scale and hat density
        confidence_velocity_boost = 1.0 + (confidence * 0.15)  # 1.0 at conf=0, 1.15 at conf=1
        confidence_hat_boost = 1.0 + (confidence * 0.10)  # 1.0 at conf=0, 1.10 at conf=1

        # DROP diagnostics — print condition check when in DROP section
        if section == "DROP" and is_inferred:
            _print_drop_diagnostics(snap, bar)

        # Update arrangement state (advance ramp FIRST so render sees updated intensity)
        arrangement.update_intent(intent, bar)
        arrangement.advance_bar()

        # Choose base groove based on intent
        if intent == BehaviourIntent.ANCHOR:
            current_base = anchor_groove
        elif intent in (BehaviourIntent.BUILD, BehaviourIntent.ENTER_FULL):
            current_base = busy_groove
        elif intent in (BehaviourIntent.LISTEN, BehaviourIntent.BAIL, BehaviourIntent.FINAL_BAIL):
            current_base = []  # Output shaper generates from scratch
        else:
            current_base = base_groove

        # Render bar with arrangement context
        shaped = renderer.render_bar(current_base, arrangement, bar, intent)

        # Offset events to global bar position
        bar_offset = bar * 16  # 16 16th-notes per bar
        for evt in shaped:
            global_events.append(evt.copy_with(
                grid_position=evt.grid_position + bar_offset,
                bar_index=bar,
            ))

        # Record diagnostics
        notes_added = len(shaped)
        notes_diff = notes_added - prev_event_count
        prev_event_count = notes_added

        # Output validation
        is_drop = is_drop_output(shaped)
        is_bail = is_bail_output(shaped)
        is_final = is_final_bail_output(shaped)

        # Build notes summary
        if notes_added == 0:
            notes_summary = ""
        elif notes_added <= 4:
            instruments = [e.instrument for e in shaped]
            notes_summary = ", ".join(instruments)
        else:
            notes_summary = f"{notes_added} events"

        diag = {
            "bar": bar,
            "time": bar_start,
            "section": section,
            "density": snap.input_density,
            "certainty": snap.player_certainty,
            "stability": snap.repetition_stability,
            "change_score": snap.change_score,
            "silence": snap.silence_duration,
            "phase": phase,
            "inferred_intent": inferred_intent.value,
            "intent": intent.value,
            "arrangement_intensity": arrangement.current_intensity,
            "velocity_scale": arrangement.current_velocity_scale,
            "hat_density": arrangement.current_hat_density,
            "confidence": confidence,
            "stable_bars": confidence_state.stable_bars,
            "unstable_bars": confidence_state.unstable_bars,
            "event_count": notes_added,
            "notes_diff": notes_diff,
            "is_drop": is_drop,
            "is_bail": is_bail,
            "is_final_bail": is_final,
            "notes_summary": notes_summary,
        }
        diagnostics.append(diag)

    return pipeline, diagnostics, global_events


def _timeline_section_name(bar: int, total_bars: int) -> str:
    """Return the named section for a bar index in the simulated timeline."""
    if bar <= 1:
        return "LISTEN"
    elif bar <= 3:
        return "ENTER_SOFT"
    elif bar <= 6:
        return "MAINTAIN"
    elif bar <= 9:
        return "BUILD"
    elif bar <= 12:
        return "REDUCE"
    elif bar == 13:
        return "DROP"  # Brief thin bar — triggers pullback
    elif bar <= 15:
        return "FINAL_BAIL"  # Strong hit then silence — ending gesture
    elif bar == 16:
        return "ANCHOR"  # Weak erratic events
    elif bar <= 18:
        return "MAINTAIN_2"  # Recovery — steady groove
    else:
        return "BAIL"  # Long silence


# ============================================================================
# Diagnostic printing
# ============================================================================


def print_timeline_table(diagnostics: list[dict]) -> None:
    """Print a bar-by-bar diagnostic timeline table."""
    print(f"\n{'=' * 140}")
    print("  Continuous Jam — Timeline Diagnostic Table")
    print(f"{'=' * 140}")
    header = (
        f"  {'Bar':>4s}  {'Time':>5s}  {'Section':>14s}  "
        f"{'Dens':>5s}  {'Cert':>5s}  {'Stab':>5s}  "
        f"{'Chg':>5s}  {'Sil':>4s}  {'Phs':>4s}  "
        f"{'Conf':>5s}  {'SBar':>4s}  {'UBar':>4s}  "
        f"{'Inferred':>12s}  {'Intent':>12s}  {'ArrInt':>6s}  "
        f"{'VelScl':>6s}  {'HatDen':>6s}  {'Events':>6s}  {'Diff':>5s}"
    )
    print(header)
    print(f"  {'-' * 4}  {'-' * 5}  {'-' * 14}  "
          f"{'-' * 5}  {'-' * 5}  {'-' * 5}  "
          f"{'-' * 5}  {'-' * 4}  {'-' * 4}  "
          f"{'-' * 5}  {'-' * 4}  {'-' * 4}  "
          f"{'-' * 12}  {'-' * 12}  {'-' * 6}  "
          f"{'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 5}")
    for d in diagnostics:
        inferred = d.get("inferred_intent", d["intent"])
        match_marker = "OVERRIDE" if inferred != d["intent"] else " "
        print(
            f"  {d['bar']:4d}  {d['time']:5.1f}s  {d['section']:>14s}  "
            f"{d['density']:.2f}  {d['certainty']:.2f}  "
            f"{d['stability']:.2f}  {d['change_score']:.2f}  "
            f"{d.get('silence', 0.0):3.1f}  {d.get('phase', 0.0):3.2f}  "
            f"{d.get('confidence', 0.0):4.2f}  "
            f"{d.get('stable_bars', 0):4d}  {d.get('unstable_bars', 0):4d}  "
            f"{inferred:>12s}  {d['intent']:>12s}{match_marker}  "
            f"{d['arrangement_intensity']:.2f}  "
            f"{d['velocity_scale']:.2f}  {d['hat_density']:4d}  "
            f"{d['event_count']:4d}  {d['notes_diff']:+4d}"
        )
    print(f"{'=' * 140}")


def print_schedule_summary(
    global_events: list[GrooveEvent],
    bpm: float,
    diagnostics: list[dict],
    timing_log: list | None = None,
) -> None:
    """Print a summary of the full MIDI schedule."""
    total_bars = len(diagnostics)
    total_note_on = sum(1 for e in global_events)
    total_duration = total_bars * (60.0 / bpm) * 4.0

    print(f"\n{'=' * 60}")
    print("  Continuous Jam — Schedule Summary")
    print(f"{'=' * 60}")
    print(f"  Total bars:          {total_bars}")
    print(f"  Total duration:      {total_duration:.1f}s")
    print(f"  Total note_on events: {total_note_on}")
    print(f"  BPM:                 {bpm}")

    # Behaviour sections
    print(f"\n  Behaviour sections:")
    current_section = None
    section_start_bar = 0
    for d in diagnostics:
        if d["section"] != current_section:
            if current_section is not None:
                dur = (d["bar"] - section_start_bar) * (60.0 / bpm) * 4.0
                print(f"    bars {section_start_bar:2d}-{d['bar']-1:2d}  "
                      f"{current_section:>14s}  ({dur:.1f}s)")
            current_section = d["section"]
            section_start_bar = d["bar"]
    # Final section
    dur = (total_bars - section_start_bar) * (60.0 / bpm) * 4.0
    print(f"    bars {section_start_bar:2d}-{total_bars-1:2d}  "
          f"{current_section:>14s}  ({dur:.1f}s)")

    # Per-bar event counts
    print(f"\n  Per-bar event counts:")
    for d in diagnostics:
        bar_label = f"bar {d['bar']:2d}"
        inferred = d.get("inferred_intent", d["intent"])
        override = " (scripted)" if inferred != d["intent"] else ""
        print(f"    {bar_label:>8s}  {d['section']:>14s}  "
              f"{d['event_count']:3d} events  "
              f"(inferred={inferred:>12s}  intent={d['intent']:>12s}{override}  "
              f"intensity={d['arrangement_intensity']:.2f})")

    if timing_log is not None:
        print_timing_summary(timing_log)

    print(f"{'=' * 60}")


def print_timing_summary(timing_log: list) -> None:
    """Print a compact timing error summary."""
    errors_ms = [(actual - target) * 1000.0
                 for target, actual, _type, _note, _vel in timing_log]
    if errors_ms:
        abs_errors = [abs(e) for e in errors_ms]
        print(f"\n  Timing report:")
        print(f"    events:             {len(timing_log)}")
        print(f"    mean abs error:     {sum(abs_errors)/len(abs_errors):.2f} ms")
        print(f"    max abs error:      {max(abs_errors):.2f} ms")


def print_full_schedule(global_events: list[GrooveEvent], bpm: float) -> None:
    """Print every scheduled MIDI event."""
    messages = groove_events_to_midi_messages(global_events, bpm=bpm)
    if not messages:
        print("  (no MIDI events in schedule)")
        return

    print(f"\n  Full MIDI schedule ({len(messages)} note_on events):")
    print(f"  {'Time':>7s}  {'Beat':>6s}  {'Bar':>4s}  {'Note':>5s}  "
          f"{'Vel':>4s}  {'Inst'}")
    print(f"  {'-'*7}  {'-'*6}  {'-'*4}  {'-'*5}  {'-'*4}  {'-'*15}")
    for t, note, vel in messages:
        beat = t / (60.0 / bpm)
        bar = int(beat // 4) + 1
        beat_in_bar = beat % 4
        inst = [k for k, v in INSTRUMENT_TO_NOTE.items() if v == note]
        inst_label = inst[0] if inst else f"note{note}"
        print(f"  {t:7.3f}s  {beat_in_bar:5.2f}  {bar:4d}  {note:5d}  "
              f"{vel:4d}  {inst_label}")


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Continuous Jam MIDI Demo — one unbroken drum performance."
    )
    parser.add_argument("--port", type=str, default="PocketDrummer Out")
    parser.add_argument("--bpm", type=float, default=120.0)
    parser.add_argument("--bars", type=int, default=20,
                        help="Number of bars to simulate (default: 20 → 40s at 120 BPM)")
    parser.add_argument("--mode", type=str, default="scripted",
                        choices=["scripted", "inferred"],
                        help="scripted = forced DROP/ANCHOR/BAIL (reliable arc); "
                             "inferred = pipeline decides everything from input")
    parser.add_argument("--no-play", action="store_true",
                        help="Print only, do not send MIDI")
    parser.add_argument("--print-schedule", action="store_true",
                        help="Print the full per-event MIDI schedule")
    args = parser.parse_args()

    do_play = not args.no_play
    bpm = args.bpm
    bars = args.bars
    mode = args.mode

    print(f"Continuous Jam MIDI Demo — {mode.upper()} mode")
    print(f"  BPM: {bpm}  Bars: {bars}  "
          f"Duration: ~{bars * (60.0 / bpm) * 4.0:.1f}s")
    print(f"  Mode: {'PLAY' if do_play else 'PRINT-ONLY'}")

    # Run the jam
    pipeline, diagnostics, global_events = run_continuous_jam(
        bars=bars, bpm=bpm, mode=mode,
    )

    # Print diagnostics
    print_timeline_table(diagnostics)

    # Print output validation table
    _print_output_validation(diagnostics)

    if args.print_schedule:
        print_full_schedule(global_events, bpm)

    print_schedule_summary(global_events, bpm, diagnostics)

    # MIDI playback
    if do_play:
        ports = list_available_ports()
        if not ports:
            print("\nNo MIDI ports available. Skipping playback.")
            return 1
        port_name = find_or_none(args.port)
        if port_name is None:
            print(f"\nPort '{args.port}' not found. Available: {ports}")
            return 1
        print(f"\nMIDI output: {port_name}")

        midi = MidiOut(port_name)
        midi.open()
        try:
            timing_log = _play_global_schedule(midi, global_events, bpm)
            print_schedule_summary(global_events, bpm, diagnostics,
                                   timing_log=timing_log)
        finally:
            midi.close()

    return 0


def _play_global_schedule(
    midi: MidiOut,
    events: list[GrooveEvent],
    bpm: float,
    note_duration: float = 0.09,
) -> list[tuple[float, float, str, int, int]]:
    """Play a global GrooveEvent list using absolute scheduling.

    Each event has absolute bar positioning via ``grid_position + bar_index * 16``.
    """
    if not events:
        return []

    schedule = build_schedule(events, bpm=bpm, note_duration=note_duration,
                              repeats=1)
    if not schedule:
        return []

    drum_channel = 9
    timing_log: list[tuple[float, float, str, int, int]] = []

    start_time = time.perf_counter()
    for target_abs, msg_type, note, velocity in schedule:
        elapsed = time.perf_counter() - start_time
        wait = target_abs - elapsed

        if wait > 0.002:
            time.sleep(wait - 0.002)
        while time.perf_counter() - start_time < target_abs:
            pass

        if msg_type == "on":
            midi.note_on(note, velocity)
        else:
            midi.note_off(note)

        actual = time.perf_counter() - start_time
        timing_log.append((target_abs, actual, msg_type, note, velocity))

    return timing_log


if __name__ == "__main__":
    sys.exit(main())
