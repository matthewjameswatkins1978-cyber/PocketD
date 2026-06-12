"""DrummerFeelEngine — repeatable human drummer timing and velocity behaviour.

Based on drummer microtiming principles:
    - Limb-specific timing offsets
    - Timing variance per instrument group
    - Velocity shaping (not random-only)
    - Ghost note support
    - Compound split / micro-flam for simultaneous hits
    - Confidence-aware behaviour
    - Stability-driven variation

Architecture separation:
    - Groove layer: decides *what* notes happen.
    - Feel layer: decides *when* and *how hard* those notes happen.

The engine is deterministic when provided with a ``seed``, enabling
reproducible test results.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum

# ---------------------------------------------------------------------------
# TimingStrategy  — supported timing-feel modes
# ---------------------------------------------------------------------------


class TimingStrategy(str, Enum):
    """Named timing strategies representing different drummer feels.

    Each strategy defines per-limb timing offsets that bias hits
    early (negative) or late (positive) relative to the grid.
    """

    MACHINE_TIGHT = "machine_tight"
    TIGHT_ROCK = "tight_rock"
    LAID_BACK = "laid_back"
    PUSHED = "pushed"
    HAT_ANCHOR_SNARE_LATE = "hat_anchor_snare_late"
    COMPOUND_SPLIT = "compound_split"
    LOOSE_GARAGE = "loose_garage"


# ---------------------------------------------------------------------------
# Default limb timing offsets (ms) per strategy
# ---------------------------------------------------------------------------
# Bias is applied BEFORE random variance.
# Convention: negative = early, positive = late.

_DEFAULT_STRATEGY_OFFSETS: dict[TimingStrategy, dict[str, float]] = {
    TimingStrategy.MACHINE_TIGHT: {
        "kick": 0.0,
        "snare": 0.0,
        "hi_hat": 0.0,
        "ride": 0.0,
        "crash": 0.0,
        "toms": 0.0,
    },
    TimingStrategy.TIGHT_ROCK: {
        "kick": -5.0,
        "snare": 3.0,
        "hi_hat": -2.0,
        "ride": -1.0,
        "crash": 0.0,
        "toms": -3.0,
    },
    TimingStrategy.LAID_BACK: {
        "kick": 3.0,
        "snare": 18.0,
        "hi_hat": 0.0,
        "ride": 2.0,
        "crash": 5.0,
        "toms": 10.0,
    },
    TimingStrategy.PUSHED: {
        "kick": -15.0,
        "snare": -12.0,
        "hi_hat": -8.0,
        "ride": -6.0,
        "crash": -10.0,
        "toms": -8.0,
    },
    TimingStrategy.HAT_ANCHOR_SNARE_LATE: {
        "kick": 0.0,
        "snare": 18.0,
        "hi_hat": -4.0,
        "ride": -2.0,
        "crash": 0.0,
        "toms": 5.0,
    },
    TimingStrategy.COMPOUND_SPLIT: {
        "kick": 0.0,
        "snare": 3.0,
        "hi_hat": -2.0,
        "ride": 0.0,
        "crash": 2.0,
        "toms": 0.0,
    },
    TimingStrategy.LOOSE_GARAGE: {
        "kick": 0.0,
        "snare": 8.0,
        "hi_hat": -3.0,
        "ride": -2.0,
        "crash": 5.0,
        "toms": 3.0,
    },
}

# Default timing variance (ms) per instrument per strategy.
# Higher values = looser feel.  machine_tight gets zero variance.
_DEFAULT_STRATEGY_VARIANCE: dict[TimingStrategy, dict[str, float]] = {
    TimingStrategy.MACHINE_TIGHT: {
        "kick": 0.0,
        "snare": 0.0,
        "hi_hat": 0.0,
        "ride": 0.0,
        "crash": 0.0,
        "toms": 0.0,
    },
    TimingStrategy.TIGHT_ROCK: {
        "kick": 1.0,
        "snare": 1.5,
        "hi_hat": 2.0,
        "ride": 2.0,
        "crash": 3.0,
        "toms": 2.0,
    },
    TimingStrategy.LAID_BACK: {
        "kick": 3.0,
        "snare": 4.0,
        "hi_hat": 5.0,
        "ride": 4.0,
        "crash": 6.0,
        "toms": 5.0,
    },
    TimingStrategy.PUSHED: {
        "kick": 2.0,
        "snare": 3.0,
        "hi_hat": 4.0,
        "ride": 3.0,
        "crash": 5.0,
        "toms": 4.0,
    },
    TimingStrategy.HAT_ANCHOR_SNARE_LATE: {
        "kick": 2.0,
        "snare": 3.0,
        "hi_hat": 2.0,
        "ride": 3.0,
        "crash": 5.0,
        "toms": 4.0,
    },
    TimingStrategy.COMPOUND_SPLIT: {
        "kick": 1.0,
        "snare": 2.0,
        "hi_hat": 2.0,
        "ride": 2.0,
        "crash": 3.0,
        "toms": 2.0,
    },
    TimingStrategy.LOOSE_GARAGE: {
        "kick": 5.0,
        "snare": 7.0,
        "hi_hat": 8.0,
        "ride": 7.0,
        "crash": 10.0,
        "toms": 8.0,
    },
}

# Default velocity bias per instrument per strategy (MIDI units).
# Positive = louder, negative = softer.
_DEFAULT_VELOCITY_BIAS: dict[TimingStrategy, dict[str, int]] = {
    TimingStrategy.MACHINE_TIGHT: {
        "kick": 0,
        "snare": 0,
        "hi_hat": 0,
        "ride": 0,
        "crash": 0,
        "toms": 0,
    },
    TimingStrategy.TIGHT_ROCK: {
        "kick": 0,
        "snare": 5,
        "hi_hat": -2,
        "ride": 0,
        "crash": 3,
        "toms": 0,
    },
    TimingStrategy.LAID_BACK: {
        "kick": -2,
        "snare": 3,
        "hi_hat": -1,
        "ride": -1,
        "crash": 2,
        "toms": -2,
    },
    TimingStrategy.PUSHED: {
        "kick": 3,
        "snare": 5,
        "hi_hat": 0,
        "ride": 2,
        "crash": 5,
        "toms": 3,
    },
    TimingStrategy.HAT_ANCHOR_SNARE_LATE: {
        "kick": 0,
        "snare": 6,
        "hi_hat": -3,
        "ride": -1,
        "crash": 2,
        "toms": 0,
    },
    TimingStrategy.COMPOUND_SPLIT: {
        "kick": 0,
        "snare": 2,
        "hi_hat": -1,
        "ride": 0,
        "crash": 2,
        "toms": 0,
    },
    TimingStrategy.LOOSE_GARAGE: {
        "kick": 0,
        "snare": 4,
        "hi_hat": -2,
        "ride": 0,
        "crash": 5,
        "toms": 2,
    },
}

# Default velocity variance per instrument per strategy (MIDI units).
_DEFAULT_VELOCITY_VARIANCE: dict[TimingStrategy, dict[str, int]] = {
    TimingStrategy.MACHINE_TIGHT: {
        "kick": 0,
        "snare": 0,
        "hi_hat": 0,
        "ride": 0,
        "crash": 0,
        "toms": 0,
    },
    TimingStrategy.TIGHT_ROCK: {
        "kick": 2,
        "snare": 2,
        "hi_hat": 4,
        "ride": 3,
        "crash": 4,
        "toms": 3,
    },
    TimingStrategy.LAID_BACK: {
        "kick": 4,
        "snare": 5,
        "hi_hat": 6,
        "ride": 5,
        "crash": 7,
        "toms": 6,
    },
    TimingStrategy.PUSHED: {
        "kick": 3,
        "snare": 4,
        "hi_hat": 5,
        "ride": 4,
        "crash": 6,
        "toms": 5,
    },
    TimingStrategy.HAT_ANCHOR_SNARE_LATE: {
        "kick": 3,
        "snare": 3,
        "hi_hat": 4,
        "ride": 4,
        "crash": 6,
        "toms": 5,
    },
    TimingStrategy.COMPOUND_SPLIT: {
        "kick": 2,
        "snare": 3,
        "hi_hat": 4,
        "ride": 3,
        "crash": 5,
        "toms": 4,
    },
    TimingStrategy.LOOSE_GARAGE: {
        "kick": 5,
        "snare": 6,
        "hi_hat": 8,
        "ride": 7,
        "crash": 9,
        "toms": 8,
    },
}

# ---------------------------------------------------------------------------
# Additional compound split offsets (ms) — used when COMPOUND_SPLIT is active
# ---------------------------------------------------------------------------
# When two instruments fire on the same grid position, the second instrument
# gets a small additional separation offset.

_COMPOUND_SPLIT_PAIRS: dict[tuple[str, str], tuple[float, float]] = {
    ("kick", "crash"): (0.0, 4.0),  # kick stays, crash +4ms
    ("snare", "hi_hat"): (8.0, -3.0),  # snare +8ms, hi_hat -3ms
    ("kick", "hi_hat"): (1.0, -4.0),  # kick +1ms, hi_hat -4ms
    ("snare", "crash"): (0.0, 5.0),
    ("kick", "snare"): (0.0, 3.0),
    ("hi_hat", "crash"): (0.0, 6.0),
    ("toms", "crash"): (0.0, 4.0),
}


# ---------------------------------------------------------------------------
# Instrument group resolution
# ---------------------------------------------------------------------------


def _instrument_group(inst: str) -> str:
    """Map a raw instrument name to a group key used in timing/velocity tables.

    Accepts common variations like ``"hat"``, ``"closed_hat"``, ``"open_hat"``.
    """
    lower = inst.lower().replace(" ", "_")
    if lower in ("kick", "kik"):
        return "kick"
    if lower in ("snare", "sn", "sd"):
        return "snare"
    if lower in ("hat", "hi_hat", "hi-hat", "closed_hat", "open_hat", "hh", "ch", "oh"):
        return "hi_hat"
    if lower in ("ride", "rd", "rc"):
        return "ride"
    if lower in ("crash", "cr", "cc"):
        return "crash"
    if lower in ("tom", "toms", "hi_tom", "mid_tom", "low_tom", "tom1", "tom2", "tom3"):
        return "toms"
    return lower


# ---------------------------------------------------------------------------
# GrooveEvent  — a single note event in the groove pipeline
# ---------------------------------------------------------------------------


@dataclass
class GrooveEvent:
    """A single percussive event flowing through the drum engine pipeline.

    The ``source_role`` indicates where the event came from for downstream
    processing (e.g. ghost notes are not varied, fills may be suppressed
    when confidence is low).

    Parameters
    ----------
    instrument : str
        Instrument name (e.g. ``"kick"``, ``"snare"``, ``"hi_hat"``).
    grid_position : int
        Position within the bar in 16th-note units (0–15 for a single bar,
        or 0–63 for 4 bars with 16th notes).
    bar_index : int
        Which bar this event belongs to (0-indexed).
    velocity : int
        MIDI velocity in range 1–127.
    probability : float
        Likelihood (0–1) this event actually sounds. The engine filters
        events whose probability check fails.
    timing_offset_ms : float
        Net timing deviation from the grid (ms). Positive = late.
        This is filled in by the Feel Engine.
    duration : int
        MIDI note-off delta or gate time in arbitrary units. 0 = default.
    articulation : str
        Playing style hint: ``"default"``, ``"accent"``, ``"ghost"``,
        ``"flam"``, ``"open"``, ``"closed"``.
    source_role : str
        Origin context: ``"main"``, ``"ghost"``, ``"fill"``, ``"crash"``,
        ``"transition"``.
    """

    instrument: str
    grid_position: int
    bar_index: int = 0
    velocity: int = 100
    probability: float = 1.0
    timing_offset_ms: float = 0.0
    duration: int = 0
    articulation: str = "default"
    source_role: str = "main"

    def copy_with(self, **kwargs) -> GrooveEvent:
        """Return a shallow copy with overridden fields."""
        d = {**self.__dict__, **kwargs}
        return GrooveEvent(**d)


# ---------------------------------------------------------------------------
# DrummerProfile  — a named personality that bundles feel parameters
# ---------------------------------------------------------------------------


@dataclass
class DrummerProfile:
    """A named drummer personality defining timing, velocity, and behaviour.

    All parameters can be overridden per-use.  Preset profiles are provided
    in ``DrummerProfile.builtin_profiles()``.

    Parameters
    ----------
    name : str
        Human-readable label (e.g. ``"Tight Rock"``).
    timing_strategy : TimingStrategy
        Which timing-offset strategy to use.
    limb_timing_offsets_ms : dict[str, float]
        Per-instrument timing offset (ms). Keys are instrument groups.
    limb_timing_variance_ms : dict[str, float]
        Per-instrument random variance half-range (ms).
    limb_velocity_bias : dict[str, int]
        Per-instrument velocity offset (MIDI units).
    velocity_variance : dict[str, int]
        Per-instrument velocity random variance half-range (MIDI units).
    ghost_note_density : float
        Probability (0–1) of adding a ghost snare note at suitable positions.
    fill_density : float
        Probability (0–1) of fill-like variation behaviour.
    crash_tendency : float
        Probability (0–1) of crash cymbal usage at accent points.
    tempo_drift_tendency : float
        Tendency (0–1) for the drummer's internal tempo to drift.
        Higher values = more drift.  Reserved for future use.
    section_push_pull : float
        How much the drummer pushes/pulls time across section boundaries.
        Positive = rush, negative = drag.  Reserved for future use.
    complexity_tolerance : int
        Maximum complexity level (1–10) the profile can handle without
        simplification.
    stability_level : float
        How consistent the drummer is (0 = wild, 1 = rock solid).
        Higher stability = fewer fills, less variation, stronger repetition.
    seed : int or None
        Deterministic seed for reproducible output.  ``None`` = unpredictable.
    """

    name: str = "Default"
    timing_strategy: TimingStrategy = TimingStrategy.MACHINE_TIGHT
    limb_timing_offsets_ms: dict[str, float] = field(default_factory=dict)
    limb_timing_variance_ms: dict[str, float] = field(default_factory=dict)
    limb_velocity_bias: dict[str, int] = field(default_factory=dict)
    velocity_variance: dict[str, int] = field(default_factory=dict)
    ghost_note_density: float = 0.05
    fill_density: float = 0.05
    crash_tendency: float = 0.1
    tempo_drift_tendency: float = 0.0
    section_push_pull: float = 0.0
    complexity_tolerance: int = 10
    stability_level: float = 1.0
    seed: int | None = None

    # ------------------------------------------------------------------
    # Builder helpers  — fill missing entries from strategy defaults
    # ------------------------------------------------------------------

    def _ensure_offsets(self) -> dict[str, float]:
        """Return limb_timing_offsets_ms, filling missing groups from strategy defaults."""
        base = dict(_DEFAULT_STRATEGY_OFFSETS.get(self.timing_strategy, {}))
        base.update(self.limb_timing_offsets_ms)
        return base

    def _ensure_variance(self) -> dict[str, float]:
        """Return limb_timing_variance_ms, filling missing groups from strategy defaults."""
        base = dict(_DEFAULT_STRATEGY_VARIANCE.get(self.timing_strategy, {}))
        base.update(self.limb_timing_variance_ms)
        return base

    def _ensure_velocity_bias(self) -> dict[str, int]:
        """Return limb_velocity_bias, filling missing groups from strategy defaults."""
        base = dict(_DEFAULT_VELOCITY_BIAS.get(self.timing_strategy, {}))
        base.update(self.limb_velocity_bias)
        return base

    def _ensure_velocity_variance(self) -> dict[str, int]:
        """Return velocity_variance, filling missing groups from strategy defaults."""
        base = dict(_DEFAULT_VELOCITY_VARIANCE.get(self.timing_strategy, {}))
        base.update(self.velocity_variance)
        return base

    # ------------------------------------------------------------------
    # Built-in preset profiles
    # ------------------------------------------------------------------

    @staticmethod
    def builtin_profiles() -> dict[str, DrummerProfile]:
        """Return a dict of all built-in preset profiles keyed by short name.

        Presets:
            - ``"machine"``: Zero variance, tight as a drum machine.
            - ``"tight_rock"``: Controlled timing, strong backbeat.
            - ``"laid_back"``: Snare late, hats stable, moderate ghosts.
            - ``"pushed_punk"``: Whole kit early, high energy, few ghosts.
            - ``"loose_garage"``: Higher variance, simple parts, low polish.
        """
        return {
            "machine": DrummerProfile(
                name="Machine",
                timing_strategy=TimingStrategy.MACHINE_TIGHT,
                ghost_note_density=0.0,
                fill_density=0.0,
                crash_tendency=0.0,
                tempo_drift_tendency=0.0,
                stability_level=1.0,
                seed=0,
            ),
            "tight_rock": DrummerProfile(
                name="Tight Rock",
                timing_strategy=TimingStrategy.TIGHT_ROCK,
                ghost_note_density=0.04,
                fill_density=0.08,
                crash_tendency=0.05,
                tempo_drift_tendency=0.05,
                stability_level=0.85,
                seed=0,
            ),
            "laid_back": DrummerProfile(
                name="Laid Back Pocket",
                timing_strategy=TimingStrategy.LAID_BACK,
                ghost_note_density=0.12,
                fill_density=0.10,
                crash_tendency=0.08,
                tempo_drift_tendency=0.10,
                stability_level=0.65,
                seed=0,
            ),
            "pushed_punk": DrummerProfile(
                name="Pushed Punk",
                timing_strategy=TimingStrategy.PUSHED,
                ghost_note_density=0.02,
                fill_density=0.15,
                crash_tendency=0.15,
                tempo_drift_tendency=0.15,
                stability_level=0.50,
                seed=0,
            ),
            "loose_garage": DrummerProfile(
                name="Loose Garage",
                timing_strategy=TimingStrategy.LOOSE_GARAGE,
                ghost_note_density=0.06,
                fill_density=0.12,
                crash_tendency=0.20,
                tempo_drift_tendency=0.25,
                stability_level=0.35,
                seed=0,
            ),
        }

    @staticmethod
    def get(profile_id: str) -> DrummerProfile:
        """Retrieve a built-in profile by id, or raise KeyError."""
        profiles = DrummerProfile.builtin_profiles()
        if profile_id in profiles:
            return profiles[profile_id]
        raise KeyError(
            f"Unknown profile '{profile_id}'. "
            f"Available: {list(profiles.keys())}"
        )


# ---------------------------------------------------------------------------
# DrummerFeelEngine  — core timing/velocity/feel processor
# ---------------------------------------------------------------------------


class DrummerFeelEngine:
    """Apply human drummer feel to a list of ``GrooveEvent`` objects.

    Pipeline (called via ``process()``)::

        1. Per-limb timing offset (bias first, randomise second)
        2. Velocity shaping
        3. Compound split / micro-flam (optional)
        4. Ghost note insertion
        5. Probability filtering
        6. Confidence-aware suppression

    The engine is deterministic when the profile has a ``seed`` value.
    """

    def __init__(self, profile: DrummerProfile) -> None:
        self.profile = profile
        self._rng = random.Random(profile.seed)

    def seed(self, value: int) -> None:
        """Re-seed the internal RNG for reproducibility."""
        self._rng = random.Random(value)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        events: list[GrooveEvent],
        input_confidence: float | None = None,
        tempo_bpm: float | None = None,
    ) -> list[GrooveEvent]:
        """Run events through the full feel-processing pipeline.

        Parameters
        ----------
        events : list[GrooveEvent]
            Input groove events (e.g. from a groove generator).
        input_confidence : float or None
            Optional confidence value (0–1).  Low confidence suppresses
            variation and ghost notes.
        tempo_bpm : float or None
            Optional tempo in BPM.  Very fast tempos suppress ghost notes.

        Returns
        -------
        list[GrooveEvent]
            Processed events with timing offsets, velocity shaping, ghost
            notes, and probability filtering applied.
        """
        if not events:
            return []

        # Determine confidence level (default = high)
        confidence = 1.0 if input_confidence is None else max(0.0, min(1.0, input_confidence))

        # Step 1: Per-limb timing offsets (bias first, randomise second)
        events = self._apply_timing_offsets(events, confidence)

        # Step 2: Velocity shaping
        events = self._apply_velocity_shaping(events, confidence)

        # Step 3: Compound split / micro-flam for simultaneous hits
        if self.profile.timing_strategy in (
            TimingStrategy.COMPOUND_SPLIT,
            TimingStrategy.LOOSE_GARAGE,
            TimingStrategy.LAID_BACK,
        ):
            events = self._apply_compound_split(events, confidence)

        # Step 4: Ghost note insertion
        events = self._add_ghost_notes(events, confidence, tempo_bpm)

        # Step 5: Probability filtering
        events = self._apply_probability_filtering(events, confidence)

        # Step 6: Sort by bar index, then grid position
        events.sort(key=lambda e: (e.bar_index, e.grid_position))

        return events

    # ------------------------------------------------------------------
    # Step 1: Per-limb timing offsets
    # ------------------------------------------------------------------

    def _apply_timing_offsets(
        self,
        events: list[GrooveEvent],
        confidence: float,
    ) -> list[GrooveEvent]:
        """Apply per-instrument timing bias and variance.

        Rule: bias first, randomise second.
        """
        offsets = self.profile._ensure_offsets()
        variances = self.profile._ensure_variance()

        # Scale variance by confidence: lower confidence = less variation
        confidence_scale = confidence
        # Scale variance by stability: higher stability = less variation
        stability_scale = self.profile.stability_level

        out: list[GrooveEvent] = []
        for ev in events:
            group = _instrument_group(ev.instrument)
            bias = offsets.get(group, 0.0)
            var = variances.get(group, 0.0)

            # Apply stability scaling: high stability = low variance
            effective_var = var * (1.0 - stability_scale * 0.5)
            # Apply confidence scaling: low confidence = less randomisation
            effective_var *= confidence_scale

            # Random uniformly distributed offset around the bias
            if effective_var > 0:
                jitter = self._rng.uniform(-effective_var, effective_var)
            else:
                jitter = 0.0

            total_offset = bias + jitter

            out.append(ev.copy_with(timing_offset_ms=ev.timing_offset_ms + total_offset))

        return out

    # ------------------------------------------------------------------
    # Step 2: Velocity shaping
    # ------------------------------------------------------------------

    def _apply_velocity_shaping(
        self,
        events: list[GrooveEvent],
        confidence: float,
    ) -> list[GrooveEvent]:
        """Apply velocity shaping rules.

        Rules:
            - Main snare backbeats (positions 4, 12 in 16th-note grid) are stronger.
            - Ghost notes are much softer (already set, but enforce upper cap).
            - Hi-hat breathes across the bar: downbeat hats slightly stronger.
            - Avoid identical repeated hat velocities unless machine_tight.
            - Late snares (positive offset) may be slightly fatter/louder.
            - Pushed parts may be sharper but not necessarily louder.
        """
        bias_table = self.profile._ensure_velocity_bias()
        var_table = self.profile._ensure_velocity_variance()
        is_machine = self.profile.timing_strategy == TimingStrategy.MACHINE_TIGHT

        out: list[GrooveEvent] = []
        # Track previous hat velocity for anti-repetition logic
        prev_hat_velocity: int | None = None

        for ev in events:
            group = _instrument_group(ev.instrument)
            vel = ev.velocity

            # Apply per-instrument velocity bias
            bias = bias_table.get(group, 0)
            vel += bias

            # Apply per-instrument velocity variance
            var_val = var_table.get(group, 0)
            if var_val > 0 and confidence > 0:
                # Scale variance by confidence and stability
                effective_var = var_val * (0.3 + 0.7 * confidence)
                effective_var *= (0.5 + 0.5 * self.profile.stability_level)
                vel += self._rng.randint(-int(effective_var), int(effective_var))

            # --- Rule-based shaping ---

            # Snare backbeats (positions 4, 12 in 16th-note grid within a bar)
            if group == "snare" and ev.source_role == "main":
                pos_in_bar = ev.grid_position % 16
                if pos_in_bar in (4, 12):
                    vel += 8  # backbeat accent

            # Enforce ghost-note velocity cap
            if ev.articulation == "ghost" or ev.source_role == "ghost":
                vel = min(vel, 50)  # ghost notes are soft

            # Hi-hat breathing
            if group == "hi_hat" and not is_machine:
                pos_in_bar = ev.grid_position % 16
                # Downbeat hats (positions 0, 8) slightly stronger
                if pos_in_bar in (0, 8):
                    vel += 3
                # Avoid identical repeated velocities
                if prev_hat_velocity is not None and abs(vel - prev_hat_velocity) < 2:
                    vel += self._rng.choice([-2, 2])

            # Late snares are slightly fatter/louder
            if group == "snare" and ev.timing_offset_ms > 5.0:
                vel += 3

            # Clamp to valid MIDI range
            vel = max(1, min(127, vel))

            if group == "hi_hat":
                prev_hat_velocity = vel

            out.append(ev.copy_with(velocity=vel))

        return out

    # ------------------------------------------------------------------
    # Step 3: Compound split / micro-flam
    # ------------------------------------------------------------------

    def _apply_compound_split(
        self,
        events: list[GrooveEvent],
        confidence: float,
    ) -> list[GrooveEvent]:
        """Separate simultaneous hits on the same grid position by small ms offsets.

        Only applies when the profile's timing strategy supports compound split
        behaviour.  Most splits are under 10 ms.
        """
        if not events:
            return []

        # Group events by (bar_index, grid_position)
        position_groups: dict[tuple[int, int], list[int]] = {}
        for i, ev in enumerate(events):
            key = (ev.bar_index, ev.grid_position)
            position_groups.setdefault(key, []).append(i)

        out = list(events)
        for indices in position_groups.values():
            if len(indices) < 2:
                continue

            # Build list of (instrument, index) pairs
            pairs = [(out[i].instrument, i) for i in indices]
            # Sort by instrument for deterministic pair resolution
            pairs.sort(key=lambda p: p[0])

            for a_idx in range(len(pairs)):
                for b_idx in range(a_idx + 1, len(pairs)):
                    inst_a, i_a = pairs[a_idx]
                    inst_b, i_b = pairs[b_idx]

                    # Look up compound split pair
                    key = (_instrument_group(inst_a), _instrument_group(inst_b))
                    key_rev = (_instrument_group(inst_b), _instrument_group(inst_a))

                    if key in _COMPOUND_SPLIT_PAIRS:
                        offset_a, offset_b = _COMPOUND_SPLIT_PAIRS[key]
                    elif key_rev in _COMPOUND_SPLIT_PAIRS:
                        offset_b, offset_a = _COMPOUND_SPLIT_PAIRS[key_rev]
                    else:
                        continue

                    # Scale by confidence: low confidence = less pronounced split
                    scale = 0.5 + 0.5 * confidence
                    out[i_a] = out[i_a].copy_with(
                        timing_offset_ms=out[i_a].timing_offset_ms + offset_a * scale
                    )
                    out[i_b] = out[i_b].copy_with(
                        timing_offset_ms=out[i_b].timing_offset_ms + offset_b * scale
                    )

        return out

    # ------------------------------------------------------------------
    # Step 4: Ghost note insertion
    # ------------------------------------------------------------------

    def _add_ghost_notes(
        self,
        events: list[GrooveEvent],
        confidence: float,
        tempo_bpm: float | None,
    ) -> list[GrooveEvent]:
        """Add snare ghost notes based on profile density.

        Rules:
            - Ghost notes are low velocity (10–40).
            - Ghost notes sit on 16th-note positions near backbeats.
            - Ghost note probability drops if confidence is low or tempo is fast.
        """
        density = self.profile.ghost_note_density

        # Reduce ghost notes when confidence is low
        density *= confidence

        # Reduce ghost notes when tempo is very fast (>160 BPM)
        if tempo_bpm is not None and tempo_bpm > 160:
            density *= max(0.1, 1.0 - (tempo_bpm - 160) / 80.0)

        # No ghost notes if density is effectively zero
        if density <= 0:
            return events

        # Collect occupied positions per bar to avoid overcrowding
        occupied: dict[int, set[int]] = {}
        for ev in events:
            occupied.setdefault(ev.bar_index, set()).add(ev.grid_position)

        # Determine max bars
        max_bar = max((ev.bar_index for ev in events), default=0)

        # Ghost note positions: 16th-note positions near backbeats
        # Backbeats are at 4, 12; ghost candidates are at 1, 3, 5, 7, 9, 11, 13, 15
        # (off-beat 16th positions that don't overlap with main 8th-note hats)
        ghost_candidates = [1, 3, 5, 7, 9, 11, 13, 15]

        new_events = list(events)

        for bar_idx in range(max_bar + 1):
            bar_occupied = occupied.get(bar_idx, set())
            for candidate_pos in ghost_candidates:
                # Skip if position is already occupied in this bar
                if candidate_pos in bar_occupied:
                    continue

                # Roll for ghost note
                if self._rng.random() < density:
                    # Ghost velocity: low, with small variation
                    ghost_vel = max(1, min(50, 20 + self._rng.randint(-8, 12)))
                    ghost = GrooveEvent(
                        instrument="snare",
                        grid_position=candidate_pos,
                        bar_index=bar_idx,
                        velocity=ghost_vel,
                        probability=1.0,
                        timing_offset_ms=0.0,
                        articulation="ghost",
                        source_role="ghost",
                    )
                    new_events.append(ghost)

        return new_events

    # ------------------------------------------------------------------
    # Step 5: Probability filtering
    # ------------------------------------------------------------------

    def _apply_probability_filtering(
        self,
        events: list[GrooveEvent],
        confidence: float,
    ) -> list[GrooveEvent]:
        """Filter events based on their probability value.

        For each event, generate a random number.  If the number exceeds
        the event's probability, the event is removed.

        When confidence is low, main events have a slightly higher chance
        of survival (the drummer holds the groove rather than changing it).
        Ghost notes and fills are more likely to be dropped when confidence
        is low.
        """
        out: list[GrooveEvent] = []
        for ev in events:
            # Determine survival bonus based on role and confidence
            survival_bonus = 0.0
            if confidence < 0.5:
                if ev.source_role == "main":
                    survival_bonus = 0.15  # hold the groove
                elif ev.source_role in ("ghost", "fill", "crash"):
                    survival_bonus = -0.2  # suppress extras

            effective_prob = max(0.0, min(1.0, ev.probability + survival_bonus))

            if effective_prob >= 1.0:
                out.append(ev)
            else:
                if self._rng.random() < effective_prob:
                    out.append(ev)

        return out

    # ------------------------------------------------------------------
    # Convenience: convert processed events to MIDI dict format
    # ------------------------------------------------------------------

    def to_midi_dicts(
        self,
        events: list[GrooveEvent],
        bpm: float,
        midi_note_map: dict[str, int] | None = None,
        step_duration_16th: float | None = None,
    ) -> list[dict]:
        """Convert processed GrooveEvents to MIDI dicts for the scheduler/humanizer.

        Each output dict has keys: ``timestamp``, ``velocity``, ``instrument``,
        ``note``, ``duration``, ``source_role``.

        Parameters
        ----------
        events : list[GrooveEvent]
            Processed groove events.
        bpm : float
            Tempo in BPM.  Used to convert grid positions to timestamps
            when *step_duration_16th* is not provided.
        midi_note_map : dict[str, int] or None
            Optional mapping from instrument name to MIDI note number.
            Defaults to a standard GM drum map.
        step_duration_16th : float or None
            Pre-computed duration of one 16th note in seconds.  If not
            provided, it is computed from *bpm*.

        Returns
        -------
        list[dict]
            MIDI-ready events sorted by timestamp.
        """
        if midi_note_map is None:
            from models import CLOSED_HAT, CRASH, KICK, OPEN_HAT, RIDE, SNARE

            midi_note_map = {
                "kick": KICK,
                "snare": SNARE,
                "hi_hat": CLOSED_HAT,
                "closed_hat": CLOSED_HAT,
                "open_hat": OPEN_HAT,
                "crash": CRASH,
                "ride": RIDE,
            }

        if step_duration_16th is None:
            step_duration_16th = 60.0 / bpm / 4.0  # seconds per 16th note

        out: list[dict] = []
        for ev in events:
            # Compute base timestamp from grid position
            base_time = (ev.bar_index * 16 + ev.grid_position) * step_duration_16th
            # Apply timing offset (convert ms to seconds)
            timestamp = base_time + ev.timing_offset_ms / 1000.0
            timestamp = max(0.0, timestamp)

            # Resolve MIDI note
            note = midi_note_map.get(
                _instrument_group(ev.instrument),
                midi_note_map.get(ev.instrument, 36),  # fallback to kick
            )

            out.append({
                "timestamp": timestamp,
                "velocity": ev.velocity,
                "instrument": ev.instrument,
                "note": note,
                "duration": ev.duration,
                "source_role": ev.source_role,
            })

        out.sort(key=lambda d: d["timestamp"])
        return out