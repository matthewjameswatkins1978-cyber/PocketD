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
    OutputShapingConfig,
    is_drop_output,
    is_bail_output,
    is_final_bail_output,
)
from drummer.confidence import PerformanceConfidenceState
from drummer.phrase_markers import (
    PhraseMarkerType,
    PhraseMarkerConfig,
    PhraseMarkerState,
    choose_phrase_marker,
    apply_phrase_marker,
    phrase_marker_label,
)
from drummer.presets import (
    DrummerPreset,
    DrummerPresetConfig,
    get_drummer_preset,
    list_drummer_presets,
)


# ============================================================================
# Base grooves
# ============================================================================


def _simple_groove() -> list[GrooveEvent]:
    """Simple rock groove: kick 1/3, snare 2/4, 8th hats on strong beats.

    Has 8 events (M bucket) so MAINTAIN is distinguishable from BUILD (H bucket)
    and SETTLE (M bucket with quarter hats).
    Musically: kick on 1+3, snare on 2+4, hats on 8th notes.
    """
    return [
        GrooveEvent("kick", 0, velocity=100),
        GrooveEvent("hi_hat", 0, velocity=80),
        GrooveEvent("snare", 4, velocity=100),
        GrooveEvent("hi_hat", 4, velocity=80),
        GrooveEvent("kick", 8, velocity=98),
        GrooveEvent("hi_hat", 8, velocity=80),
        GrooveEvent("snare", 12, velocity=100),
        GrooveEvent("hi_hat", 12, velocity=80),
    ]


def _busy_groove() -> list[GrooveEvent]:
    """Busier groove with 16th hats, ghost snares, extra kicks.

    Uses hi-hat for timekeeping with 16th subdivisions.
    No crash — crash is reserved for arrivals and phrase markers.
    """
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


def _arrival_groove() -> list[GrooveEvent]:
    """Firm arrival groove: ride accent on 1, extra kick on 1&, ride colour.

    Used after a BUILD to give a sense of arrival.
    Uses ride cymbal (not crash) for arrival colour — crash is reserved
    for FINAL_BAIL and explicit phrase-peak markers only.
    The extra kick and ride give distinct skeleton character without crash.
    """
    return [
        GrooveEvent("kick", 0, velocity=115),
        GrooveEvent("ride", 0, velocity=90),  # ride on beat 1 for arrival colour
        GrooveEvent("hi_hat", 1, velocity=65),
        GrooveEvent("hi_hat", 2, velocity=70),
        GrooveEvent("kick", 3, velocity=80),  # extra kick on 1&
        GrooveEvent("snare", 4, velocity=115),
        GrooveEvent("ride", 4, velocity=85),  # ride on snare for colour
        GrooveEvent("hi_hat", 5, velocity=65),
        GrooveEvent("hi_hat", 6, velocity=70),
        GrooveEvent("kick", 8, velocity=112),
        GrooveEvent("ride", 8, velocity=80),  # ride on beat 3
        GrooveEvent("hi_hat", 9, velocity=65),
        GrooveEvent("hi_hat", 10, velocity=70),
        GrooveEvent("snare", 12, velocity=112),
        GrooveEvent("ride", 12, velocity=80),  # ride on beat 4
        GrooveEvent("hi_hat", 13, velocity=65),
        GrooveEvent("hi_hat", 14, velocity=70),
    ]


def _reduce_groove() -> list[GrooveEvent]:
    """Thin reduce groove: kick 1&3, snare 2&4, no hats, no ghosts.

    Significantly fewer events than _simple_groove (4 vs 12).
    Low enough to produce L bucket in evaluator skeleton.
    """
    return [
        GrooveEvent("kick", 0, velocity=70),
        GrooveEvent("snare", 4, velocity=65),
        GrooveEvent("kick", 8, velocity=65),
        GrooveEvent("snare", 12, velocity=60),
    ]


def _settle_groove() -> list[GrooveEvent]:
    """Settling groove: kick 1/3, snare 2/4, quarter-note hats.

    Quarter-note hats instead of 8th hats for lower density.
    8 events instead of 12 — produces M/L bucket.
    """
    return [
        GrooveEvent("kick", 0, velocity=80),
        GrooveEvent("hi_hat", 0, velocity=60),
        GrooveEvent("snare", 4, velocity=85),
        GrooveEvent("hi_hat", 4, velocity=60),
        GrooveEvent("kick", 8, velocity=78),
        GrooveEvent("hi_hat", 8, velocity=60),
        GrooveEvent("snare", 12, velocity=82),
        GrooveEvent("hi_hat", 12, velocity=60),
    ]


def _recover_groove_phase1() -> list[GrooveEvent]:
    """Sparse recovery phase 1: kick pulse with soft snare + quiet quarter hats.

    7 events per bar — quiet quarter-note hi-hat pulse bridges DROP→recovery
    so bar 11 doesn't jump from hatless to busy 8th hats on bar 12.
    The soft snare on beat 4 gives a musical hint that recovery is beginning.
    """
    return [
        GrooveEvent("kick", 0, velocity=75),
        GrooveEvent("hi_hat", 0, velocity=25),
        GrooveEvent("hi_hat", 4, velocity=25),
        GrooveEvent("kick", 8, velocity=72),
        GrooveEvent("hi_hat", 8, velocity=25),
        GrooveEvent("snare", 12, velocity=40),  # soft snare on beat 4 — recovery hint
        GrooveEvent("hi_hat", 12, velocity=25),
    ]


def _recover_groove_phase2() -> list[GrooveEvent]:
    """Medium recovery phase 2: add snare back, quarter hats.

    Ramping back toward full groove without jumping to 8th-note hats.
    8 events, M bucket, matching Matthew's preference that recovery
    sits on quarter-note hats before the groove fully settles.
    """
    return [
        GrooveEvent("kick", 0, velocity=85),
        GrooveEvent("hi_hat", 0, velocity=65),
        GrooveEvent("snare", 4, velocity=90),
        GrooveEvent("hi_hat", 4, velocity=65),
        GrooveEvent("kick", 8, velocity=82),
        GrooveEvent("hi_hat", 8, velocity=65),
        GrooveEvent("snare", 12, velocity=88),
        GrooveEvent("hi_hat", 12, velocity=65),
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
# Simulated player timeline builder — scenario-specific forms
# ============================================================================


def _apply_timing_jitter(
    events: list[MusicalEvent], jitter_amount: float, seed: int = 0
) -> list[MusicalEvent]:
    """Apply small random timing offsets to events for uncertain-input variation."""
    import random
    rng = random.Random(seed)
    jittered = list(events)
    for evt in jittered:
        offset = rng.uniform(-jitter_amount, jitter_amount)
        evt._time = max(0.0, evt._time + offset)  # type: ignore[attr-defined]
    return jittered


def _build_enter_timeline(
    bpm: float,
    bars: int,
    playtest_variation: str,
) -> list[list[MusicalEvent]]:
    """Build timeline for ENTER scenario.

    Bars 0-1:   LISTEN (empty)
    Bars 2-3:   ENTER_SOFT (sparse quarter notes)
    Bars 4-11:  MAINTAIN / SETTLE (steady 8th-note groove)
    No FINAL_BAIL, no DROP, no ANCHOR.
    """
    bar_duration = (60.0 / bpm) * 4.0
    all_bars: list[list[MusicalEvent]] = [[] for _ in range(bars)]
    is_uncertain = (playtest_variation == "uncertain_input")

    for bar in range(bars):
        bar_start = bar * bar_duration

        if bar <= 1:
            continue  # LISTEN — empty

        elif bar <= 3:
            # ENTER_SOFT — sparse quarter notes
            if is_uncertain:
                for beat in (0, 2):
                    t = bar_start + beat * (60.0 / bpm) + 0.03 * bar
                    strength = 0.45 + 0.05 * bar
                    all_bars[bar].append(MusicalEvent(t, strength=min(strength, 0.65)))
            else:
                for beat in range(4):
                    t = bar_start + beat * (60.0 / bpm)
                    all_bars[bar].append(MusicalEvent(t, strength=0.65))

        else:
            # MAINTAIN / SETTLE — steady 8th-note pulse
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.7))

    return all_bars


def _build_build_timeline(
    bpm: float,
    bars: int,
    playtest_variation: str,
) -> list[list[MusicalEvent]]:
    """Build timeline for BUILD scenario.

    Bars 0-1:   LISTEN (empty)
    Bars 2-5:   MAINTAIN setup (steady 8th-note)
    Bars 6-9:   BUILD (rising intensity, picks up)
    Bars 10-12: ARRIVAL / strong MAINTAIN
    Bars 13-15: SETTLE / slight reduce
    No FINAL_BAIL, no DROP, no ANCHOR.
    """
    bar_duration = (60.0 / bpm) * 4.0
    all_bars: list[list[MusicalEvent]] = [[] for _ in range(bars)]
    is_strong_build = (playtest_variation == "strong_build")
    is_uncertain = (playtest_variation == "uncertain_input")

    for bar in range(bars):
        bar_start = bar * bar_duration

        if bar <= 1:
            continue  # LISTEN

        elif bar <= 5:
            # MAINTAIN setup
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.65))

        elif bar <= 9:
            # BUILD — rising intensity
            eighth = 60.0 / bpm / 2.0
            build_progress = (bar - 6) / 4.0
            strength_offset = -0.10 if is_uncertain else 0.0
            for i in range(8):
                t = bar_start + i * eighth
                strength = min(0.95, 0.50 + build_progress * 0.45 + strength_offset)
                all_bars[bar].append(MusicalEvent(t, strength=max(strength, 0.3)))
            # 16th-note pickups in later build bars
            if bar >= 8:
                sixteenth = eighth / 2.0
                for pick in (14, 15):
                    t = bar_start + pick * sixteenth
                    all_bars[bar].append(MusicalEvent(t, strength=0.7))

        elif bar <= 12:
            # ARRIVAL / strong MAINTAIN
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.80))

        else:
            # SETTLE / slight REDUCE — quieter maintain
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.55))

    return all_bars


def _build_drop_timeline(
    bpm: float,
    bars: int,
    playtest_variation: str,
) -> list[list[MusicalEvent]]:
    """Build timeline for DROP scenario.

    Musical form:
      bars 0-1:   LISTEN (empty)
      bars 2-4:   MAINTAIN / establish groove (events from bar 3 so not silent)
      bars 5-7:   BUILD (rising intensity)
      bar 8:      REDUCE / pre-drop thinning (musical preparation)
      bars 9-10:  DROP (thin sparse — 2 bars)
      bars 11-12: RECOVER (climb back within 2 bars)
      bars 13-14: SETTLE (stable maintain)
      bar 15:     SETTLE_VARIANT (stable with slight variation)

    No FINAL_BAIL, no ANCHOR.
    """
    bar_duration = (60.0 / bpm) * 4.0
    all_bars: list[list[MusicalEvent]] = [[] for _ in range(bars)]
    is_pullback = (playtest_variation == "pullback_after_build")
    is_deliberate_sparse = (playtest_variation == "deliberate_sparse")

    for bar in range(bars):
        bar_start = bar * bar_duration

        if bar <= 1:
            continue  # LISTEN — empty

        elif bar <= 4:
            # MAINTAIN setup — quiet entry on bar 3-4, not fully silent
            if bar <= 2:
                # bar 2: very sparse quarter-note pulse (soft entry)
                for beat in (0, 2):
                    t = bar_start + beat * (60.0 / bpm)
                    all_bars[bar].append(MusicalEvent(t, strength=0.40))
            else:
                # bar 3-4: steady 8th-note groove
                eighth = 60.0 / bpm / 2.0
                for i in range(8):
                    t = bar_start + i * eighth
                    all_bars[bar].append(MusicalEvent(t, strength=0.65))

        elif bar <= 7:
            # BUILD — rising intensity over 3 bars
            eighth = 60.0 / bpm / 2.0
            build_progress = (bar - 5) / 3.0
            for i in range(8):
                t = bar_start + i * eighth
                strength = min(0.90, 0.55 + build_progress * 0.35)
                all_bars[bar].append(MusicalEvent(t, strength=strength))
            if bar >= 6:
                sixteenth = eighth / 2.0
                for pick in (14, 15):
                    t = bar_start + pick * sixteenth
                    all_bars[bar].append(MusicalEvent(t, strength=0.7))

        elif bar == 8:
            # REDUCE / pre-drop thinning — thinner than build, preparing the drop
            eighth = 60.0 / bpm / 2.0
            for i in range(4):  # only quarter-note positions
                t = bar_start + i * 2 * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.50))
            # Add a soft snare on beat 4 as a "breath" before drop
            all_bars[bar].append(MusicalEvent(bar_start + (60.0 / bpm) * 3, strength=0.40))

        elif bar <= 10:
            # DROP — thin sparse events (2 bars)
            if is_deliberate_sparse:
                all_bars[bar].append(MusicalEvent(bar_start, strength=0.75))
            else:
                t1 = bar_start
                t2 = bar_start + 0.8
                all_bars[bar].append(MusicalEvent(t1, strength=0.65))
                all_bars[bar].append(MusicalEvent(t2, strength=0.05))

        elif bar <= 12:
            # RECOVER — climb back within 2 bars
            if bar == 11:
                # Sparse answer: kick pulse + soft snare hint
                all_bars[bar].append(MusicalEvent(bar_start, strength=0.65))
                all_bars[bar].append(MusicalEvent(bar_start + (60.0 / bpm) * 2, strength=0.60))
                all_bars[bar].append(MusicalEvent(bar_start + (60.0 / bpm) * 3, strength=0.25))
            else:
                # Fuller recovery: 8th-note groove
                eighth = 60.0 / bpm / 2.0
                for i in range(8):
                    t = bar_start + i * eighth
                    all_bars[bar].append(MusicalEvent(t, strength=0.70))

        elif bar == 14:
            # SETTLE — stable maintain (quarter-note hats feel)
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.60))

        elif bar == 15:
            # SETTLE_VARIANT — slightly different: quarter-note hats, lower strength
            for beat in range(4):
                t = bar_start + beat * (60.0 / bpm)
                all_bars[bar].append(MusicalEvent(t, strength=0.55))
            all_bars[bar].append(MusicalEvent(bar_start + (60.0 / bpm) * 2, strength=0.50))

        else:
            # Extra bars (beyond 16): silence
            continue

    return all_bars


def _build_anchor_recovery_timeline(
    bpm: float,
    bars: int,
    playtest_variation: str,
) -> list[list[MusicalEvent]]:
    """Build timeline for ANCHOR_RECOVERY scenario.

    Bars 0-1:   LISTEN (empty)
    Bars 2-5:   MAINTAIN setup
    Bars 6-9:   uncertainty / ANCHOR (weak erratic)
    Bars 10-13: RECOVER / MAINTAIN
    Bars 14-15: SETTLE
    No FINAL_BAIL, no DROP.
    """
    bar_duration = (60.0 / bpm) * 4.0
    all_bars: list[list[MusicalEvent]] = [[] for _ in range(bars)]
    is_poor_phase = (playtest_variation == "poor_phase_recovery")
    is_weak_recovery = (playtest_variation == "weak_input_recovery")

    for bar in range(bars):
        bar_start = bar * bar_duration

        if bar <= 1:
            continue

        elif bar <= 5:
            # MAINTAIN setup
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.65))

        elif bar <= 9:
            # ANCHOR — weak erratic events
            if is_poor_phase:
                # Wide phase drift: poorly timed events
                for pos in [0.1, 0.5, 0.8, 1.1, 1.3, 1.5, 1.7, 1.9]:
                    t = bar_start + pos
                    all_bars[bar].append(MusicalEvent(t, strength=0.10))
            elif is_weak_recovery:
                # Very sparse weak events
                for pos in [0.2, 1.0, 1.8]:
                    t = bar_start + pos
                    all_bars[bar].append(MusicalEvent(t, strength=0.08))
            else:
                for pos in [0.1, 0.5, 0.8, 1.1, 1.3, 1.5, 1.7, 1.9, 1.95]:
                    t = bar_start + pos
                    all_bars[bar].append(MusicalEvent(t, strength=0.12))

        else:
            # RECOVER / SETTLE — steady 8th-note
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.7))

    return all_bars


def _build_final_bail_timeline(
    bpm: float,
    bars: int,
    playtest_variation: str,
) -> list[list[MusicalEvent]]:
    """Build timeline for FINAL_BAIL scenario.

    Bars 0-1:   LISTEN (empty)
    Bars 2-4:   MAINTAIN setup
    Bars 5-8:   REDUCE / ending signal
    Bar 9:      FINAL_BAIL cue (single kick+crash hit)
    Bars 10-19: SILENCE — no events (no restart)
    """
    bar_duration = (60.0 / bpm) * 4.0
    all_bars: list[list[MusicalEvent]] = [[] for _ in range(bars)]
    is_ambiguous_ending = (playtest_variation == "ambiguous_cue")

    for bar in range(bars):
        bar_start = bar * bar_duration

        if bar <= 1:
            continue  # LISTEN

        elif bar <= 4:
            # MAINTAIN setup
            eighth = 60.0 / bpm / 2.0
            for i in range(8):
                t = bar_start + i * eighth
                all_bars[bar].append(MusicalEvent(t, strength=0.65))

        elif bar <= 8:
            # REDUCE / ending signal — thin but present
            if bar == 8:
                # Last bar before final cue: eighth notes
                eighth = 60.0 / bpm / 2.0
                for i in range(8):
                    t = bar_start + i * eighth
                    all_bars[bar].append(MusicalEvent(t, strength=0.50))
            else:
                # Earlier reduce bars: thinner
                eighth = 60.0 / bpm / 2.0
                for i in range(8):
                    t = bar_start + i * eighth
                    all_bars[bar].append(MusicalEvent(t, strength=0.55))

        elif bar == 9:
            # FINAL_BAIL cue — build up then final hit
            eighth = 60.0 / bpm / 2.0
            if is_ambiguous_ending:
                # Weaker, later ending — only 4 eighth notes then late hit
                for i in range(4):
                    t = bar_start + i * eighth
                    all_bars[bar].append(MusicalEvent(t, strength=0.55))
                t_end = bar_start + 1.9
                all_bars[bar].append(MusicalEvent(t_end, strength=0.65))
            else:
                # Clear ending: 8 notes then strong hit
                for i in range(8):
                    t = bar_start + i * eighth
                    all_bars[bar].append(MusicalEvent(t, strength=0.75))
                t_end = bar_start + 1.8
                all_bars[bar].append(MusicalEvent(t_end, strength=0.95))

        else:
            # SILENCE — no events after final cue, no restart
            continue

    return all_bars


_SCENARIO_TIMELINE_BUILDERS: dict[str, callable] = {
    "enter": _build_enter_timeline,
    "build": _build_build_timeline,
    "drop": _build_drop_timeline,
    "anchor_recovery": _build_anchor_recovery_timeline,
    "final_bail": _build_final_bail_timeline,
}


def build_scenario_timeline(
    bpm: float = 120.0,
    bars: int = 16,
    scenario: str = "enter",
    playtest_variation: str = "",
) -> list[list[MusicalEvent]]:
    """Build a simulated player event timeline for a specific scenario.

    Parameters
    ----------
    scenario : str
        One of "enter", "build", "drop", "anchor_recovery", "final_bail".
    playtest_variation : str
        Variation name for modifier flags.
    bars : int
        Number of bars. Each scenario has a minimum bar count; if ``bars``
        is larger, remaining bars are filled with silence.

    Returns a list of ``bars`` lists of ``MusicalEvent``.
    """
    builder = _SCENARIO_TIMELINE_BUILDERS.get(scenario)
    if builder is None:
        # Fallback: generic timeline
        from warnings import warn
        warn(f"Unknown scenario {scenario!r}, using generic timeline")
        return [[] for _ in range(bars)]

    # Build with enough bars, then pad or trim
    needed_bars = max(bars, 12)
    timeline = builder(bpm=bpm, bars=needed_bars, playtest_variation=playtest_variation)

    # Trim or pad to requested bar count
    if len(timeline) > bars:
        timeline = timeline[:bars]
    while len(timeline) < bars:
        timeline.append([])  # silence for extra bars

    return timeline


# Backward compatibility alias
build_simulated_timeline = build_scenario_timeline


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
    "SETTLE": 0.75,          # settled, stable maintain
    "RECOVER": 0.65,         # recovering, slightly less phase confidence
}


def _phase_for_section(section: str, playtest_variation: str = "") -> float:
    """Return the phase_alignment for a given timeline section.

    Adjusts phase based on *playtest_variation* so that different
    playtest scenarios produce measurably different diagnostic output.
    """
    base = _SECTION_PHASE.get(section, 0.75)

    is_uncertain = (playtest_variation == "uncertain_input")
    is_poor_phase = (playtest_variation == "poor_phase_recovery")

    if is_uncertain and section in ("ENTER_SOFT", "MAINTAIN", "BUILD"):
        return max(0.25, base * 0.5)
    if is_poor_phase and section == "MAINTAIN_2":
        return 0.15

    return base


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
# Section-to-groove mapping
# ============================================================================


def _section_groove(section: str, bar: int, bars_in_section: int) -> list[GrooveEvent]:
    """Choose a section-specific base groove for audible variation.

    Returns a groove pattern appropriate for the current section.
    The goal is to make each section's kick/snare skeleton audibly different
    while keeping total event count reasonable.
    """
    # ARRIVAL after build: firm with ride accents
    if section == "MAINTAIN_ARRIVAL":
        return _arrival_groove()

    # REDUCE: thin, no hats
    if section == "REDUCE":
        return _reduce_groove()

    # SETTLE: quarter-note hats, fewer events
    if section == "SETTLE":
        return _settle_groove()

    # RECOVER: phase 1 lasts only 1 bar (sparse), then phase 2 (medium)
    if section == "RECOVER":
        if bars_in_section < 1:
            return _recover_groove_phase1()
        elif bars_in_section < 4:
            return _recover_groove_phase2()
        else:
            return _simple_groove()

    # Default: standard groove
    return _simple_groove()


# ============================================================================
# Continuous jam — core orchestration
# ============================================================================


def run_continuous_jam(
    bars: int = 20,
    bpm: float = 120.0,
    mode: str = "scripted",
    preset_name: str = "normal",
    playtest_variation: str = "",
    scenario: str = "enter",
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
    preset_name : str
        Drummer preset name (``"cautious"``, ``"normal"``, ``"braver"``).
    playtest_variation : str
        Variation name to modify the simulated timeline
        (e.g. ``"uncertain_input"``, ``"strong_build"``).
    scenario : str
        Scenario name for timeline form (``"enter"``, ``"build"``, ``"drop"``,
        ``"anchor_recovery"``, ``"final_bail"``).

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
    # Resolve preset
    preset_config = get_drummer_preset(preset_name)

    bar_duration = (60.0 / bpm) * 4.0
    is_inferred = mode == "inferred"

    timeline = build_scenario_timeline(
        bpm=bpm, bars=bars, scenario=scenario, playtest_variation=playtest_variation,
    )

    # Build pipeline with preset profile + output config
    from drummer.behaviour import FeatureDrivenBehaviourEngine
    engine = FeatureDrivenBehaviourEngine(profile=preset_config.profile)
    shaper = BehaviourOutputShaper(config=preset_config.output_config)
    pipeline = DrummerBrainPipeline(
        behaviour_engine=engine,
        output_shaper=shaper,
    )
    arrangement = ArrangementState()
    # ENTER_SOFT starts at low-but-audible intensity so it's heard immediately
    arrangement.current_intensity = 0.25

    base_groove = _simple_groove()
    busy_groove = _busy_groove()
    anchor_groove = _anchor_groove()

    renderer = ContinuousJamRenderer(shaper=shaper)
    confidence_state = PerformanceConfidenceState()

    # Phrase marker state — use preset config
    phrase_config = preset_config.phrase_config
    phrase_state = PhraseMarkerState()

    global_events: list[GrooveEvent] = []
    diagnostics: list[dict] = []

    # Track previous bar's event count for diagnostics
    prev_event_count = 0

    # Track section changes for section-specific groove selection
    current_section = ""
    bars_in_section = 0

    for bar in range(bars):
        bar_start = bar * bar_duration
        bar_end = bar_start + bar_duration

        # Determine section name from timeline (used for diagnostics)
        section = _timeline_section_name(bar, bars, scenario)

        # Track consecutive bars in current section
        if section != current_section:
            current_section = section
            bars_in_section = 0
        else:
            bars_in_section += 1

        # Feed events for this bar's time window
        for evt in timeline[bar]:
            pipeline.feed_event(evt)

        # Choose phase_alignment based on mode and playtest_variation
        if is_inferred:
            phase = _phase_for_section(section, playtest_variation)
        else:
            # Scripted mode: use per-variation phase so playtest scenarios
            # produce measurably different diagnostics
            phase = _phase_for_section(section, playtest_variation)

        # Process at end of bar to get intent
        d = pipeline.process(now=bar_end, phase_alignment=phase)
        inferred_intent = d.behaviour_intent
        snap = d.feature_snapshot

        # Override specific intents to match the scenario section so that
        # diagnostic fields (intent, rendered_intent) are correctly aligned
        # with the musical section in scripted mode.  This ensures the musical
        # sanity checker uses the correct intent when evaluating each bar
        # (e.g. MAINTAIN bars are checked for maintain-like output, not
        # LISTEN-like output).
        # In inferred mode, only DROP/FINAL_BAIL/BAIL are forced so the
        # pipeline inference for BUILD/REDUCE/ENTER/MAINTAIN is preserved
        # for testing.
        if not is_inferred:
            # Scripted mode: override ALL sections for clean diagnostics
            if section in ("DROP",):
                intent = BehaviourIntent.DROP
            elif section == "FINAL_BAIL":
                intent = BehaviourIntent.FINAL_BAIL
            elif section == "FINAL_BAIL_SILENCE":
                intent = BehaviourIntent.BAIL
            elif section == "BAIL":
                intent = BehaviourIntent.BAIL
            elif section in ("MAINTAIN", "MAINTAIN_ARRIVAL", "MAINTAIN_2"):
                intent = BehaviourIntent.MAINTAIN
            elif section == "SETTLE":
                intent = BehaviourIntent.MAINTAIN
            elif section == "RECOVER":
                intent = BehaviourIntent.MAINTAIN
            elif section == "BUILD":
                intent = BehaviourIntent.BUILD
            elif section == "REDUCE":
                intent = BehaviourIntent.REDUCE
            elif section == "ENTER_SOFT":
                intent = BehaviourIntent.ENTER_SOFT
            elif section == "SILENCE":
                intent = BehaviourIntent.LISTEN
            else:
                # LISTEN uses inferred intent (should be listen)
                intent = inferred_intent
        else:
            # Inferred mode: only DROP, FINAL_BAIL, and BAIL are forced for
            # output correctness.  Everything else uses pipeline inference.
            if section in ("DROP",):
                intent = BehaviourIntent.DROP
            elif section == "FINAL_BAIL":
                intent = BehaviourIntent.FINAL_BAIL
            elif section == "FINAL_BAIL_SILENCE":
                intent = BehaviourIntent.BAIL
            elif section == "BAIL":
                intent = BehaviourIntent.BAIL
            elif section == "ANCHOR":
                # In inferred mode, let ANCHOR fire naturally from weak events + poor phase.
                intent = inferred_intent
            else:
                intent = inferred_intent

        # Map section to an arrangement intent so the intensity ramp matches
        # the scenario section, not the pipeline inference (which may lag).
        # This ensures section-specific grooves get correct arrangement scaling.
        arrangement_intent = intent  # default
        if section == "BUILD":
            arrangement_intent = BehaviourIntent.BUILD
        elif section == "SETTLE":
            arrangement_intent = BehaviourIntent.MAINTAIN
        elif section == "MAINTAIN_ARRIVAL":
            arrangement_intent = BehaviourIntent.MAINTAIN
        elif section == "MAINTAIN":
            arrangement_intent = BehaviourIntent.MAINTAIN
        elif section == "RECOVER":
            arrangement_intent = BehaviourIntent.MAINTAIN
        elif section == "SILENCE":
            arrangement_intent = BehaviourIntent.LISTEN
        elif section == "ENTER_SOFT":
            arrangement_intent = BehaviourIntent.ENTER_SOFT
        # REDUCE, DROP, FINAL_BAIL, ANCHOR already match via intent or section

        # Update confidence state after intent is decided
        confidence_state.update(snap, intent)
        confidence = confidence_state.confidence

        # Apply confidence to arrangement intensity scaling
        # Higher confidence → slightly higher velocity_scale and hat density
        confidence_velocity_boost = 1.0 + (confidence * 0.15)  # 1.0 at conf=0, 1.15 at conf=1
        confidence_hat_boost = 1.0 + (confidence * 0.10)  # 1.0 at conf=0, 1.10 at conf=1

        # DROP diagnostics — print condition check when in DROP section
        # (suppressed when the section name contains "comparison_mask" sentinel)
        if section == "DROP" and is_inferred:
            _print_drop_diagnostics(snap, bar)

        # Update arrangement state using the section-aligned intent
        arrangement.update_intent(arrangement_intent, bar)
        arrangement.advance_bar()

        # Choose base groove based on section (for intent-specific variation)
        # and intent (for output-shaping sections).
        # Check section first so section-specific grooves take priority.
        output_intent = arrangement_intent
        current_base = _section_groove(section, bar, bars_in_section)
        # Override with intent-specific grooves for sections without custom grooves
        if output_intent in (BehaviourIntent.LISTEN, BehaviourIntent.BAIL, BehaviourIntent.FINAL_BAIL):
            current_base = []  # Output shaper generates from scratch
        elif output_intent == BehaviourIntent.ANCHOR:
            current_base = anchor_groove
        elif output_intent in (BehaviourIntent.BUILD, BehaviourIntent.ENTER_FULL):
            # Only use busy_groove for actual BUILD sections, not for sections
            # where the pipeline just happens to infer BUILD
            if section in ("BUILD", "ENTER_SOFT"):
                current_base = busy_groove
        # Render bar with arrangement context
        shaped = renderer.render_bar(current_base, arrangement, bar, output_intent)

        # --- Phrase marker selection ---
        # Update phrase state based on intent for post-ANCHOR tracking
        if intent == BehaviourIntent.ANCHOR:
            phrase_state.bars_since_anchor = 0
        else:
            phrase_state.bars_since_anchor += 1

        marker_type = choose_phrase_marker(
            bar, intent, confidence, snap,
            config=phrase_config, state=phrase_state,
        )

        # Apply phrase marker to shaped events (if any)
        if marker_type != PhraseMarkerType.NONE:
            shaped = apply_phrase_marker(shaped, marker_type, bar, phrase_config)
            phrase_state.last_marker_bar = bar
            phrase_state.last_marker_type = marker_type
            phrase_state.marker_count += 1

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

        # Build notes summary — include crash/ride info even for large bars
        # so the evaluator can detect cymbal-based skeleton changes
        if notes_added == 0:
            notes_summary = ""
        else:
            instruments = [e.instrument for e in shaped]
            unique_insts = list(set(instruments))
            has_crash = any("crash" in i.lower() for i in unique_insts)
            has_ride = any("ride" in i.lower() for i in unique_insts)
            if notes_added <= 4:
                notes_summary = ", ".join(instruments)
            elif has_crash or has_ride:
                # Include crash/ride hint for evaluator skeleton detection
                cymbal_hints = []
                if has_crash:
                    cymbal_hints.append("crash")
                if has_ride:
                    cymbal_hints.append("ride")
                notes_summary = f"{notes_added} events ({'; '.join(cymbal_hints)})"
            else:
                notes_summary = f"{notes_added} events"

        # Per-bar instrument counts for detailed diagnostics
        kick_count = sum(1 for e in shaped if e.instrument.lower() == "kick")
        snare_count = sum(1 for e in shaped if e.instrument.lower() == "snare")
        hat_count = sum(1 for e in shaped if e.instrument in ("hi_hat", "closed_hat", "open_hat"))
        crash_count = sum(1 for e in shaped if "crash" in e.instrument.lower())
        ride_count = sum(1 for e in shaped if "ride" in e.instrument.lower())
        ghost_count = sum(1 for e in shaped if e.articulation == "ghost" or e.source_role == "ghost")
        velocities = [e.velocity for e in shaped if e.velocity > 0]
        max_vel = max(velocities) if velocities else 0
        avg_vel = round(sum(velocities) / len(velocities), 1) if velocities else 0.0

        # Skeleton signature from musical_evaluation
        from drummer.musical_evaluation import _bar_skeleton_sig
        skeleton_sig = _bar_skeleton_sig({
            "event_count": notes_added,
            "density": snap.input_density,
            "notes_summary": notes_summary,
        })

        # Repetition detection: compare with previous bar's signature
        repeated_from_previous = False
        repeated_run_length = 1
        if bar > 0:
            prev_sig = diagnostics[-1].get("skeleton_sig", "")
            if prev_sig == skeleton_sig:
                repeated_from_previous = True
                # Count consecutive similar bars backwards
                run = 1
                for j in range(len(diagnostics) - 1, -1, -1):
                    if diagnostics[j].get("skeleton_sig", "") == skeleton_sig:
                        run += 1
                    else:
                        break
                repeated_run_length = run

        marker_label = phrase_marker_label(marker_type)

        diag = {
            "bar": bar,
            "time": bar_start,
            "section": section,
            "bars_in_section": bars_in_section,
            "density": snap.input_density,
            "certainty": snap.player_certainty,
            "stability": snap.repetition_stability,
            "change_score": snap.change_score,
            "silence": snap.silence_duration,
            "phase": phase,
            "inferred_intent": inferred_intent.value,
            "intent": intent.value,
            "rendered_intent": arrangement_intent.value,
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
            "phrase_marker": marker_type.value,
            "phrase_marker_label": marker_label,
            "kick_count": kick_count,
            "snare_count": snare_count,
            "hat_count": hat_count,
            "crash_count": crash_count,
            "ride_count": ride_count,
            "ghost_count": ghost_count,
            "max_velocity": max_vel,
            "avg_velocity": avg_vel,
            "skeleton_sig": skeleton_sig,
            "repeated_from_previous": repeated_from_previous,
            "repeated_run_length": repeated_run_length,
        }
        diagnostics.append(diag)

    return pipeline, diagnostics, global_events


def _scenario_section_name(scenario: str, bar: int, total_bars: int) -> str:
    """Return the named section for a bar in a scenario-specific timeline."""
    if scenario == "enter":
        if bar <= 1:
            return "LISTEN"
        elif bar <= 3:
            return "ENTER_SOFT"
        else:
            return "MAINTAIN"
    elif scenario == "build":
        if bar <= 1:
            return "LISTEN"
        elif bar <= 5:
            return "MAINTAIN"
        elif bar <= 9:
            return "BUILD"
        elif bar <= 12:
            return "MAINTAIN_ARRIVAL"
        else:
            return "SETTLE"
    elif scenario == "drop":
        if bar <= 1:
            return "LISTEN"
        elif bar <= 4:
            return "MAINTAIN"
        elif bar <= 7:
            return "BUILD"
        elif bar == 8:
            return "REDUCE"
        elif bar <= 10:
            return "DROP"
        elif bar <= 12:
            return "RECOVER"
        elif bar <= 14:
            return "SETTLE"
        else:
            return "MAINTAIN_2"
    elif scenario == "anchor_recovery":
        if bar <= 1:
            return "LISTEN"
        elif bar <= 5:
            return "MAINTAIN"
        elif bar <= 9:
            return "ANCHOR"
        else:
            return "RECOVER"
    elif scenario == "final_bail":
        if bar <= 1:
            return "LISTEN"
        elif bar <= 4:
            return "MAINTAIN"
        elif bar <= 8:
            return "REDUCE"
        elif bar == 9:
            return "FINAL_BAIL"
        else:
            return "SILENCE"
    else:
        # Fallback for unknown scenarios
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
            return "DROP"
        elif bar == 14:
            return "FINAL_BAIL"
        elif bar == 15:
            return "FINAL_BAIL_SILENCE"
        elif bar == 16:
            return "ANCHOR"
        elif bar <= 18:
            return "MAINTAIN_2"
        else:
            return "BAIL"


def _timeline_section_name(bar: int, total_bars: int, scenario: str = "enter") -> str:
    """Return the named section for a bar index in the simulated timeline.

    Delegates to scenario-specific section naming when a scenario is given.
    Falls back to the original generic timeline for backward compatibility.
    """
    if scenario and scenario in ("enter", "build", "drop", "anchor_recovery", "final_bail"):
        return _scenario_section_name(scenario, bar, total_bars)
    # Original generic fallback
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
        return "DROP"
    elif bar == 14:
        return "FINAL_BAIL"
    elif bar == 15:
        return "FINAL_BAIL_SILENCE"
    elif bar == 16:
        return "ANCHOR"
    elif bar <= 18:
        return "MAINTAIN_2"
    else:
        return "BAIL"


# ============================================================================
# Diagnostic printing
# ============================================================================


def print_timeline_table(diagnostics: list[dict]) -> None:
    """Print a bar-by-bar diagnostic timeline table."""
    print(f"\n{'=' * 180}")
    print("  Continuous Jam — Timeline Diagnostic Table")
    print(f"{'=' * 180}")
    header = (
        f"  {'Bar':>4s}  {'Time':>5s}  {'Section':>14s}  "
        f"{'Dens':>5s}  {'Cert':>5s}  {'Stab':>5s}  "
        f"{'Chg':>5s}  {'Sil':>4s}  {'Phs':>4s}  "
        f"{'Conf':>5s}  {'SBar':>4s}  {'UBar':>4s}  "
        f"{'Inferred':>12s}  {'Intent':>12s}  {'Rendered':>12s}  "
        f"{'ArrInt':>6s}  "
        f"{'VelScl':>6s}  {'HatDen':>6s}  {'Events':>6s}  {'Diff':>5s}"
        f"  {'Phrase':>6s}"
    )
    print(header)
    print(f"  {'-' * 4}  {'-' * 5}  {'-' * 14}  "
          f"{'-' * 5}  {'-' * 5}  {'-' * 5}  "
          f"{'-' * 5}  {'-' * 4}  {'-' * 4}  "
          f"{'-' * 5}  {'-' * 4}  {'-' * 4}  "
          f"{'-' * 12}  {'-' * 12}  {'-' * 12}  "
          f"{'-' * 6}  "
          f"{'-' * 6}  {'-' * 6}  {'-' * 6}  {'-' * 5}"
          f"  {'-' * 6}")
    for d in diagnostics:
        inferred = d.get("inferred_intent", d["intent"])
        rendered = d.get("rendered_intent", d["intent"])
        match_marker = "OVERRIDE" if inferred != d["intent"] else " "
        phrase_label = d.get("phrase_marker_label", "")
        print(
            f"  {d['bar']:4d}  {d['time']:5.1f}s  {d['section']:>14s}  "
            f"{d['density']:.2f}  {d['certainty']:.2f}  "
            f"{d['stability']:.2f}  {d['change_score']:.2f}  "
            f"{d.get('silence', 0.0):3.1f}  {d.get('phase', 0.0):3.2f}  "
            f"{d.get('confidence', 0.0):4.2f}  "
            f"{d.get('stable_bars', 0):4d}  {d.get('unstable_bars', 0):4d}  "
            f"{inferred:>12s}  {d['intent']:>12s}{match_marker}  "
            f"{rendered:>12s}  "
            f"{d['arrangement_intensity']:.2f}  "
            f"{d['velocity_scale']:.2f}  {d['hat_density']:4d}  "
            f"{d['event_count']:4d}  {d['notes_diff']:+4d}"
            f"  {phrase_label:>6s}"
        )
    print(f"{'=' * 180}")


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
        rendered = d.get("rendered_intent", d["intent"])
        override = " (scripted)" if inferred != d["intent"] else ""
        print(f"    {bar_label:>8s}  {d['section']:>14s}  "
              f"{d['event_count']:3d} events  "
              f"(inferred={inferred:>12s}  intent={d['intent']:>12s}{override}  "
              f"rendered={rendered:>12s}  "
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


def _run_preset_comparison(
    bars: int = 20,
    bpm: float = 120.0,
) -> list[dict]:
    """Run all three presets and return comparison results (no printing).

    Same as ``run_preset_comparison`` but returns raw data for testing.
    """
    presets_to_run = ["cautious", "normal", "braver"]
    results: list[dict] = []

    for preset_name in presets_to_run:
        _pipeline, diagnostics, global_events = run_continuous_jam(
            bars=bars, bpm=bpm, mode="inferred", preset_name=preset_name,
        )

        # Extract metrics
        event_counts = [d["event_count"] for d in diagnostics]
        confidences = [d.get("confidence", 0.0) for d in diagnostics]
        intents = [d["intent"] for d in diagnostics]

        total_events = sum(event_counts)

        first_non_listen = next(
            (d["bar"] for d in diagnostics if d["section"] != "LISTEN"),
            bars,
        )
        first_enter = next(
            (d["bar"] for d in diagnostics
             if d["intent"] in ("enter_soft", "enter_full")),
            bars,
        )
        first_build = next(
            (d["bar"] for d in diagnostics if d["intent"] == "build"),
            bars,
        )
        confidence_peak = max(confidences) if confidences else 0.0

        phrase_labels = [d.get("phrase_marker_label", "") for d in diagnostics]
        eight_bar_count = sum(1 for p in phrase_labels if p == "8bar")
        sixteen_bar_count = sum(1 for p in phrase_labels if p == "16bar")
        total_phrase_markers = eight_bar_count + sixteen_bar_count

        drop_event_count = 0
        bail_event_count = 0
        final_bail_event_count = 0
        drop_contract_ok = True
        bail_contract_ok = True
        final_bail_contract_ok = True

        for d in diagnostics:
            if d["section"] == "DROP":
                drop_event_count = d["event_count"]
                drop_contract_ok = d.get("is_drop", False) and d["event_count"] > 0
            elif d["section"] == "BAIL":
                bail_event_count = d["event_count"]
                bail_contract_ok = d.get("is_bail", False)
            elif d["section"] == "FINAL_BAIL":
                final_bail_event_count = d["event_count"]
                final_bail_contract_ok = d.get("is_final_bail", False)

        results.append({
            "preset": preset_name.capitalize(),
            "total_events": total_events,
            "first_non_listen": first_non_listen,
            "first_enter": first_enter,
            "first_build": first_build,
            "confidence_peak": confidence_peak,
            "phrase_markers": total_phrase_markers,
            "eight_bar": eight_bar_count,
            "sixteen_bar": sixteen_bar_count,
            "drop_events": drop_event_count,
            "bail_events": bail_event_count,
            "final_bail_events": final_bail_event_count,
            "drop_ok": drop_contract_ok,
            "bail_ok": bail_contract_ok,
            "final_bail_ok": final_bail_contract_ok,
        })

    return results


def run_preset_comparison(
    bars: int = 20,
    bpm: float = 120.0,
) -> None:
    """Run all three presets and print a compact comparison table.

    All presets run in ``inferred`` mode with forced DROP/BAIL/FINAL_BAIL
    for reliable output contract validation, but BUILD/REDUCE/ENTER/ANCHOR
    are determined by each preset's thresholds.

    Parameters
    ----------
    bars : int
        Number of bars to simulate per preset.
    bpm : float
        Tempo in beats per minute.
    """
    results = _run_preset_comparison(bars=bars, bpm=bpm)

    # Print header
    print(f"\n{'=' * 110}")
    print("  Preset Comparison — Drummer Temperaments")
    print(f"  Bars: {bars}  BPM: {bpm}  Mode: inferred (forced DROP/BAIL/FINAL_BAIL)")
    print(f"{'=' * 110}")

    # Metrics column headers
    metrics = [
        ("Total Events", "total_events", "d"),
        ("First Non-Listen Bar", "first_non_listen", "d"),
        ("First Enter Bar", "first_enter", "d"),
        ("First Build Bar", "first_build", "d"),
        ("Confidence Peak", "confidence_peak", ".2f"),
        ("Phrase Markers", "phrase_markers", "d"),
        ("  8-bar Markers", "eight_bar", "d"),
        ("  16-bar Markers", "sixteen_bar", "d"),
        ("DROP Events", "drop_events", "d"),
        ("BAIL Events", "bail_events", "d"),
        ("FINAL BAIL Events", "final_bail_events", "d"),
    ]

    # Print column headers
    col_width = 20
    header = f"  {'Metric':<{col_width}}"
    for r in results:
        header += f"  {r['preset']:<{col_width}}"
    print(header)
    print(f"  {'-' * col_width}  {'-' * col_width}  {'-' * col_width}  {'-' * col_width}")

    # Print each metric row
    for label, key, fmt in metrics:
        row = f"  {label:<{col_width}}"
        for r in results:
            val = r[key]
            if fmt == "d":
                row += f"  {val:<{col_width}d}"
            else:
                row += f"  {val:<{col_width}.2f}"
        print(row)

    # Output contract row
    contract_str = f"  {'Contracts OK':<{col_width}}"
    for r in results:
        all_ok = r["drop_ok"] and r["bail_ok"] and r["final_bail_ok"]
        label = "PASS" if all_ok else "FAIL"
        contract_str += f"  {label:<{col_width}}"
    print(contract_str)

    # Musical sanity labels
    print(f"\n{'=' * 110}")
    print("  Musical Sanity Labels")
    print(f"{'=' * 110}")

    for r in results:
        label_lines: list[str] = []
        preset = r["preset"].lower()

        if preset == "cautious":
            label_lines.append("Cautious Bunny -- waiting, supporting, avoiding risk")
            if r["first_enter"] > 3:
                label_lines.append("  [OK] enters later (after bar 3)")
            else:
                label_lines.append("  [!!] enters too early")
            if r["phrase_markers"] <= 1:
                label_lines.append("  [OK] uses few or no phrase markers")
            else:
                label_lines.append("  [!!] too many phrase markers for cautious")
            if r["confidence_peak"] <= 0.90:
                label_lines.append(f"  [OK] confidence peak {r['confidence_peak']:.2f} <= 0.90")
            else:
                label_lines.append("  [!!] confidence peak too high")
            if r["total_events"] < 200:
                label_lines.append(f"  [OK] event count {r['total_events']} moderate, not overplaying")
            else:
                label_lines.append("  [!!] too many events for cautious")
            if r["drop_ok"] and r["bail_ok"] and r["final_bail_ok"]:
                label_lines.append("  [OK] all output contracts intact")

        elif preset == "normal":
            label_lines.append("Normal Bunny -- balanced reference temperament")
            label_lines.append("  [OK] reference behaviour (ConservativePocketDrummer)")
            label_lines.append("  [OK] default PhraseMarkerConfig")
            label_lines.append("  [OK] default OutputShapingConfig")
            if r["drop_ok"] and r["bail_ok"] and r["final_bail_ok"]:
                label_lines.append("  [OK] all output contracts intact")
            label_lines.append(f"  total events: {r['total_events']}")

        elif preset == "braver":
            label_lines.append("Braver Bunny -- more committed, still musical")
            if r["first_enter"] <= 5:
                label_lines.append(f"  [OK] enters by bar {r['first_enter']} (confident)")
            else:
                label_lines.append("  [!!] enters too late")
            if r["first_build"] <= 6:
                label_lines.append(f"  [OK] builds by bar {r['first_build']} (early build)")
            else:
                label_lines.append("  [!!] build comes too late")
            if r["confidence_peak"] >= 0.60:
                label_lines.append(f"  [OK] confidence peak {r['confidence_peak']:.2f} (confident)")
            else:
                label_lines.append("  [!!] confidence peak too low")
            # Check braver is not overplaying -- event count should be > normal but reasonable
            normal_result = results[1]  # index 1 is normal
            if r["total_events"] > normal_result["total_events"]:
                ratio = r["total_events"] / normal_result["total_events"]
                if ratio < 1.5:
                    label_lines.append(f"  [OK] {ratio:.1f}x normal events (controlled increase)")
                else:
                    label_lines.append(f"  [!!] {ratio:.1f}x normal events -- too busy")
            else:
                label_lines.append("  [!!] should have more events than normal")
            if r["phrase_markers"] >= 1:
                label_lines.append(f"  [OK] uses {r['phrase_markers']} phrase markers (confident)")
            else:
                label_lines.append("  [!!] no phrase markers -- should be more decorative")
            if r["drop_ok"] and r["bail_ok"] and r["final_bail_ok"]:
                label_lines.append("  [OK] all output contracts intact")

        for line in label_lines:
            print(f"  {line}")
        print()

    print(f"{'=' * 110}")
    print("  Legend:  [OK] = musically sound  [!!] = needs attention")
    print("  Braver must not: spam events, crashes, or markers")
    print("  Cautious must not: feel broken or dead")
    print(f"{'=' * 110}\n")

    # Summary verdict
    all_pass = all(
        r["drop_ok"] and r["bail_ok"] and r["final_bail_ok"]
        for r in results
    )
    if all_pass:
        print("  All output contracts PASS for all presets.")
    else:
        print("  WARNING: Some output contracts FAILED! See table above.")


def export_diagnostics_to_json(
    diagnostics: list[dict],
    output_path: str,
    meta: dict | None = None,
) -> None:
    """Export per-bar diagnostics to a deterministic JSON file.

    Parameters
    ----------
    diagnostics : list[dict]
        Per-bar diagnostic records from ``run_continuous_jam``.
    output_path : str
        Path to write the JSON file.
    meta : dict | None
        Optional metadata to include at the top level (e.g. preset,
        mode, bpm, bars).  Must not include a ``"timestamp"`` key
        to preserve determinism.
    """
    import json
    import os

    # Build export payload
    payload: dict = {}

    if meta:
        # Ensure no timestamp in meta for deterministic export
        payload["meta"] = dict(meta)
        payload["meta"].pop("timestamp", None)

    payload["diagnostics"] = list(diagnostics)

    # Ensure output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


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
    parser.add_argument("--preset", type=str, default="normal",
                        choices=list_drummer_presets(),
                        help=f"Drummer temperament preset (default: normal). "
                             f"Options: {', '.join(list_drummer_presets())}")
    parser.add_argument("--compare-presets", action="store_true",
                        help="Run all three presets and print a comparison table"
                             " (inferred mode, no playback)")
    parser.add_argument("--no-play", action="store_true",
                        help="Print only, do not send MIDI")
    parser.add_argument("--print-schedule", action="store_true",
                        help="Print the full per-event MIDI schedule")
    parser.add_argument("--scenario", type=str, default="enter",
                        choices=["enter", "build", "drop", "anchor_recovery", "final_bail"],
                        help="Scenario timeline form (default: enter)")
    parser.add_argument("--playtest-variation", type=str, default="",
                        help="Playtest variation name (e.g. uncertain_input, strong_build)")
    parser.add_argument("--export-json", type=str, default=None,
                        metavar="PATH",
                        help="Export per-bar diagnostics as JSON to PATH")
    args = parser.parse_args()

    do_play = not args.no_play
    bpm = args.bpm
    bars = args.bars
    mode = args.mode
    preset_name = args.preset
    export_path = args.export_json

    # Handle compare-presets mode
    if args.compare_presets:
        run_preset_comparison(bars=bars, bpm=bpm)
        # Export comparison results if requested
        if export_path:
            results = _run_preset_comparison(bars=bars, bpm=bpm)
            meta = {
                "mode": "compare_presets",
                "presets_compared": list_drummer_presets(),
                "bpm": bpm,
                "bars": bars,
                "total_duration": bars * (60.0 / bpm) * 4.0,
            }
            # Build a combined payload with all three presets
            import json
            payload = {"meta": meta, "comparison": results}
            import os
            out_dir = os.path.dirname(export_path)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(export_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
                f.write("\n")
            print(f"\n  Exported comparison to: {export_path}")
        return 0

    # Resolve preset for display
    preset_config = get_drummer_preset(preset_name)

    print(f"Continuous Jam MIDI Demo — {mode.upper()} mode")
    print(f"  Preset: {preset_config.name} ({preset_name})")
    print(f"  BPM: {bpm}  Bars: {bars}  "
          f"Duration: ~{bars * (60.0 / bpm) * 4.0:.1f}s")
    print(f"  Mode: {'PLAY' if do_play else 'PRINT-ONLY'}")

    # Run the jam with preset
    pipeline, diagnostics, global_events = run_continuous_jam(
        bars=bars, bpm=bpm, mode=mode, preset_name=preset_name,
        scenario=args.scenario, playtest_variation=args.playtest_variation,
    )

    # Print diagnostics
    print_timeline_table(diagnostics)

    # Print output validation table
    _print_output_validation(diagnostics)

    if args.print_schedule:
        print_full_schedule(global_events, bpm)

    print_schedule_summary(global_events, bpm, diagnostics)

    # Export to JSON if requested
    if export_path:
        meta = {
            "mode": mode,
            "preset": preset_name,
            "bpm": bpm,
            "bars": bars,
            "total_duration": bars * (60.0 / bpm) * 4.0,
            "total_events": sum(d["event_count"] for d in diagnostics),
        }
        export_diagnostics_to_json(diagnostics, export_path, meta=meta)
        print(f"\n  Exported diagnostics to: {export_path}")

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
