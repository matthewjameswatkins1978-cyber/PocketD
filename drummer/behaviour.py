"""Behaviour Engine — the brain between Perception and Performance.

Stage 1: Foundation with BAIL logic, EMA smoothing, intent tracking.
Stage 2: LISTEN / ENTER_SOFT / ENTER_FULL / MAINTAIN with pulse and bar awareness.
Stage 3: BUILD / REDUCE / DROP dynamic energy-response behaviour.
Stage 4: Feature-driven behaviour using FeatureSnapshot for BUILD, REDUCE,
    ANCHOR, ENTER, and BAIL decisions with hysteresis and confirmation.

Does NOT generate MIDI, sequence notes, or schedule beats.
Decides *intent only* — what kind of drumming behaviour is appropriate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


# ---------------------------------------------------------------------------
# BehaviourIntent — the set of possible drummer behaviour intents
# ---------------------------------------------------------------------------


class BehaviourIntent(str, Enum):
    """High-level drummer behaviour intentions.

    These describe *what kind of thing* the drummer should be doing,
    not *which notes* to play.  No MIDI or pattern data lives here.
    """

    LISTEN = "listen"
    BAIL = "bail"
    ENTER_SOFT = "enter_soft"
    ENTER_FULL = "enter_full"
    MAINTAIN = "maintain"
    BUILD = "build"
    REDUCE = "reduce"
    ANCHOR = "anchor"
    FILL = "fill"
    CRASH = "crash"
    DROP = "drop"


def parse_behaviour_intent(value: str) -> BehaviourIntent:
    """Parse a string into a BehaviourIntent, case-insensitively.

    Accepts both enum names (``"REDUCE"``) and enum values
    (``"reduce"``).  Raises ``ValueError`` with a helpful message
    listing valid options if no match is found.
    """
    v = value.strip().lower()
    for member in BehaviourIntent:
        if v == member.value.lower() or v == member.name.lower():
            return member
    valid = [f"{m.name} ({m.value})" for m in BehaviourIntent]
    raise ValueError(
        f"'{value}' is not a valid BehaviourIntent. "
        f"Valid options: {', '.join(valid)}"
    )


# ---------------------------------------------------------------------------
# DrummerProfile — behaviour threshold tuning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DrummerProfile:
    """Thresholds and tuning parameters that shape drummer behaviour.

    Note: ``drummer.feel`` has a separate ``DrummerProfile`` for
    timing/velocity feel.  This one is for behaviour decision thresholds.
    """

    name: str
    hysteresis_margin: float
    bail_silence_seconds: float
    density_inversion_threshold: float
    fill_probability_base: float
    energy_ema_alpha: float
    density_ema_alpha: float

    # Stage 2 — conservative entry thresholds
    min_pulse_confidence: float = 0.75
    min_bar_confidence: float = 0.70
    full_entry_confidence: float = 0.85
    soft_entry_confidence: float = 0.75
    min_observation_seconds: float = 1.50
    severe_uncertainty_threshold: float = 0.35
    maintain_hysteresis_margin: float = 0.10

    # Stage 3 — energy-response thresholds
    fast_energy_ema_alpha: float = 0.15
    slow_energy_ema_alpha: float = 0.02
    build_trend_threshold: float = 0.15
    reduce_trend_threshold: float = -0.10
    drop_trend_threshold: float = -0.30
    max_density_for_build: float = 0.80
    low_energy_threshold_for_drop: float = 0.25
    density_collapse_ratio_for_drop: float = 0.35
    min_build_duration_seconds: float = 2.0
    min_reduce_duration_seconds: float = 2.0
    min_drop_duration_seconds: float = 1.0

    # Stage 4 — feature-driven behaviour thresholds
    enter_certainty_threshold: float = 0.65
    enter_repetition_threshold: float = 0.70
    enter_confirmation_snapshots: int = 3
    build_change_threshold: float = 0.20
    build_certainty_threshold: float = 0.55
    build_repetition_threshold: float = 0.65
    build_phase_threshold: float = 0.55
    build_max_density_without_phrase: float = 0.80
    reduce_density_threshold: float = 0.75
    anchor_certainty_threshold: float = 0.40
    anchor_repetition_threshold: float = 0.35
    anchor_phase_threshold: float = 0.45
    feature_bail_silence_seconds: float = 1.50
    feature_hysteresis_margin: float = 0.10


# Conservative default — stable, doesn't jump at minor changes
ConservativePocketDrummer = DrummerProfile(
    name="Conservative Pocket Drummer",
    hysteresis_margin=0.10,
    bail_silence_seconds=0.50,
    density_inversion_threshold=0.75,
    fill_probability_base=0.05,
    energy_ema_alpha=0.10,
    density_ema_alpha=0.10,
    min_pulse_confidence=0.75,
    min_bar_confidence=0.70,
    full_entry_confidence=0.85,
    soft_entry_confidence=0.75,
    min_observation_seconds=1.50,
    severe_uncertainty_threshold=0.35,
    maintain_hysteresis_margin=0.10,
    # Stage 3 defaults
    fast_energy_ema_alpha=0.15,
    slow_energy_ema_alpha=0.02,
    build_trend_threshold=0.15,
    reduce_trend_threshold=-0.10,
    drop_trend_threshold=-0.30,
    max_density_for_build=0.80,
    low_energy_threshold_for_drop=0.25,
    density_collapse_ratio_for_drop=0.35,
    min_build_duration_seconds=2.0,
    min_reduce_duration_seconds=2.0,
    min_drop_duration_seconds=1.0,
    # Stage 4 defaults
    enter_certainty_threshold=0.65,
    enter_repetition_threshold=0.70,
    enter_confirmation_snapshots=3,
    build_change_threshold=0.20,
    build_certainty_threshold=0.55,
    build_repetition_threshold=0.65,
    build_phase_threshold=0.55,
    build_max_density_without_phrase=0.80,
    reduce_density_threshold=0.75,
    anchor_certainty_threshold=0.40,
    anchor_repetition_threshold=0.35,
    anchor_phase_threshold=0.45,
    feature_bail_silence_seconds=1.50,
    feature_hysteresis_margin=0.10,
)


# ---------------------------------------------------------------------------
# BehaviourDecision — the output of the Behaviour Engine
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BehaviourDecision:
    """A single behavioural decision produced by the Behaviour Engine.

    Carries only intent and meta-data.  No musical target fields
    (beat, bar, subdivision, velocity, MIDI timing) belong here yet.
    """

    intent: BehaviourIntent
    confidence: float
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    evaluated_at: float = 0.0


# ---------------------------------------------------------------------------
# BehaviourEngine — the core decision loop (Stages 1-3, unchanged)
# ---------------------------------------------------------------------------


class BehaviourEngine:
    """Decides *what kind* of drumming behaviour is appropriate right now.

    The engine receives musical events (and optionally pulse/bar state)
    and returns a ``BehaviourDecision`` indicating the drummer's intent.

    Stage 1 implements:
        - Event tracking with EMA energy/density smoothing
        - BAIL (emergency silence override)
        - Conservative fallback to LISTEN / MAINTAIN

    Stage 2 adds:
        - Pulse and bar confidence gating
        - Observation window before entry
        - ENTER_SOFT / ENTER_FULL decisions
        - MAINTAIN with hysteresis after entry
        - Severe uncertainty collapse detection

    Stage 3 adds:
        - Dual fast/slow energy EMA for trend detection
        - BUILD on sustained rising energy
        - REDUCE on falling energy or moderate confidence loss
        - DROP on severe energy collapse
        - Minimum-duration cooldowns for dynamic states
    """

    def __init__(self, profile: DrummerProfile | None = None) -> None:
        self.profile = profile if profile is not None else ConservativePocketDrummer
        self.previous_intent: BehaviourIntent = BehaviourIntent.LISTEN
        self.smoothed_energy: float | None = None
        self.smoothed_density: float | None = None
        self.last_event_time: float | None = None
        self.has_seen_event: bool = False

        # Stage 2 — internal entry tracking state
        self.first_event_time: float | None = None
        self.has_entered: bool = False
        self.entered_at: float | None = None

        # Stage 3 — dual energy EMA and dynamic state tracking
        self.fast_energy_ema: float | None = None
        self.slow_energy_ema: float | None = None
        self.last_intent_change_time: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        current_time: float,
        recent_events: list,
        pulse_state: Any = None,
        bar_state: Any = None,
    ) -> BehaviourDecision:
        """Evaluate the current musical situation and return a behavioural intent.

        Parameters
        ----------
        current_time : float
            The current timestamp in seconds.
        recent_events : list
            A list of ``MusicalEvent`` objects (or compatible dicts/objects)
            representing recent musical activity.  May be empty.
        pulse_state : Any, optional
            PulseState from the Pulse Tracker (used in Stage 2+).
        bar_state : Any, optional
            BarState from the Bar Tracker (used in Stage 2+).

        Returns
        -------
        BehaviourDecision
        """
        # Stage 1 — Update internal state from events
        if recent_events:
            self._update_from_events(recent_events)

        # Stage 1 — Emergency silence override (only after seeing events)
        decision = self._check_bail(current_time)
        if decision is not None:
            self._record_intent_change(decision.intent, current_time)
            return decision

        # Stage 2 — Musical decision using pulse and bar state
        # Preserve Stage 1 fallback when no pulse/bar state is provided
        if pulse_state is None and bar_state is None:
            return self._fallback_decision(current_time)
        return self._musical_decision(current_time, pulse_state, bar_state)

    # ------------------------------------------------------------------
    # Event tracking (Stage 1) with Stage 3 dual EMA
    # ------------------------------------------------------------------

    def _update_from_events(self, recent_events: list) -> None:
        """Ingest recent events and update internal smoothed state."""
        self.has_seen_event = True

        # Record first-event time for observation window tracking
        if self.first_event_time is None:
            earliest = recent_events[0]
            earliest_time = self._get_event_time(earliest)
            if earliest_time is not None:
                self.first_event_time = earliest_time

        # Track the most recent event time
        latest = recent_events[-1]
        latest_time = self._get_event_time(latest)
        if latest_time is not None:
            self.last_event_time = latest_time

        # Compute average energy and density across the batch
        energies = []
        densities = []
        for evt in recent_events:
            e = self._get_event_energy(evt)
            if e is not None:
                energies.append(e)
            d = self._get_event_density(evt)
            if d is not None:
                densities.append(d)

        # Smooth energy (Stage 1 single EMA — preserved for backward compat)
        if energies:
            avg_energy = sum(energies) / len(energies)
            self.smoothed_energy = self._ema(
                current=avg_energy,
                previous=self.smoothed_energy,
                alpha=self.profile.energy_ema_alpha,
            )

            # Stage 3 — Dual EMA tracking
            self.fast_energy_ema = self._ema(
                current=avg_energy,
                previous=self.fast_energy_ema,
                alpha=self.profile.fast_energy_ema_alpha,
            )
            self.slow_energy_ema = self._ema(
                current=avg_energy,
                previous=self.slow_energy_ema,
                alpha=self.profile.slow_energy_ema_alpha,
            )

        # Smooth density
        if densities:
            avg_density = sum(densities) / len(densities)
            self.smoothed_density = self._ema(
                current=avg_density,
                previous=self.smoothed_density,
                alpha=self.profile.density_ema_alpha,
            )

    # ------------------------------------------------------------------
    # BAIL logic (Stage 1)
    # ------------------------------------------------------------------

    def _check_bail(self, current_time: float) -> BehaviourDecision | None:
        """Return a BAIL decision if the silence threshold has been exceeded.

        BAIL is only possible after the engine has seen at least one event.
        It does not trigger on a fresh engine that has never received audio.
        """
        if not self.has_seen_event or self.last_event_time is None:
            return None

        silence_duration = current_time - self.last_event_time
        if silence_duration > self.profile.bail_silence_seconds:
            return BehaviourDecision(
                intent=BehaviourIntent.BAIL,
                confidence=1.0,
                reason="Silence exceeded bail threshold",
                scores={
                    "silence_duration": silence_duration,
                    "bail_silence_seconds": self.profile.bail_silence_seconds,
                },
                evaluated_at=current_time,
            )

        return None

    # ------------------------------------------------------------------
    # Stage 2+3 — Musical decision logic
    # ------------------------------------------------------------------

    def _musical_decision(
        self,
        current_time: float,
        pulse_state: Any,
        bar_state: Any,
    ) -> BehaviourDecision:
        """Evaluate pulse and bar state to produce a musical intent.

        Priority order:
        1. If already entered: Stage 3 dynamic logic (BUILD/REDUCE/DROP/MAINTAIN)
           with Stage 2 severe-collapse fallback
        2. If not entered: LISTEN / ENTER_SOFT / ENTER_FULL based on entry score
        """
        # Extract confidences
        pulse_conf = self._get_pulse_confidence(pulse_state)
        bar_conf = self._get_bar_confidence(bar_state)

        if not self.has_entered:
            # Stage 2 entry logic
            stability = self._get_stability(pulse_state, bar_state)
            observation_readiness = self._observation_readiness(current_time)
            entry_score = self._compute_entry_score(
                pulse_conf, bar_conf, stability, observation_readiness
            )
            return self._entry_decision(
                current_time, pulse_conf, bar_conf, stability,
                observation_readiness, entry_score,
            )

        # Stage 3 — Dynamic energy-response after entry
        return self._dynamic_decision(current_time, pulse_conf, bar_conf)

    # ------------------------------------------------------------------
    # Stage 3 — Dynamic energy-response decision (BUILD / REDUCE / DROP / MAINTAIN)
    # ------------------------------------------------------------------

    def _dynamic_decision(
        self,
        current_time: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> BehaviourDecision:
        """Decide BUILD, REDUCE, DROP, or MAINTAIN after entry.

        Decision order:
            1. Cooldown enforcement for dynamic states
            2. DROP (severe energy collapse)
            3. REDUCE (falling energy or moderate confidence loss)
            4. BUILD (sustained rising energy, stable, low density)
            5. MAINTAIN (default when no dynamic applies)
        """
        profile = self.profile
        energy_trend = self._energy_trend()

        # Cooldown: enforce minimum duration for current dynamic state
        cooldown = self._check_dynamic_cooldown(current_time)
        if cooldown is not None:
            self._record_intent_change(cooldown.intent, current_time)
            return cooldown

        # Stage 2 severe collapse check — confidence collapsed below absolute floor
        if pulse_conf < profile.severe_uncertainty_threshold or bar_conf < profile.severe_uncertainty_threshold:
            decision = BehaviourDecision(
                intent=BehaviourIntent.LISTEN,
                confidence=min(pulse_conf, bar_conf),
                reason="Listening: confidence collapsed after entry",
                scores={
                    "pulse_confidence": pulse_conf,
                    "bar_confidence": bar_conf,
                    "entry_score": min(pulse_conf, bar_conf),
                    "severe_uncertainty_threshold": profile.severe_uncertainty_threshold,
                },
                evaluated_at=current_time,
            )
            self._record_intent_change(decision.intent, current_time)
            return decision

        # Check DROP conditions
        drop_decision = self._check_drop(current_time, energy_trend)
        if drop_decision is not None:
            self._record_intent_change(drop_decision.intent, current_time)
            return drop_decision

        # Check REDUCE conditions
        reduce_decision = self._check_reduce(current_time, energy_trend, pulse_conf, bar_conf)
        if reduce_decision is not None:
            self._record_intent_change(reduce_decision.intent, current_time)
            return reduce_decision

        # Check BUILD conditions
        build_decision = self._check_build(current_time, energy_trend, pulse_conf, bar_conf)
        if build_decision is not None:
            self._record_intent_change(build_decision.intent, current_time)
            return build_decision

        # Default: MAINTAIN
        maintain_confidence = min(pulse_conf, bar_conf)
        decision = BehaviourDecision(
            intent=BehaviourIntent.MAINTAIN,
            confidence=maintain_confidence,
            reason="Maintain: no sustained dynamic change detected",
            scores={
                "pulse_confidence": pulse_conf,
                "bar_confidence": bar_conf,
                "fast_energy_ema": self.fast_energy_ema or 0.0,
                "slow_energy_ema": self.slow_energy_ema or 0.0,
                "energy_trend": energy_trend,
            },
            evaluated_at=current_time,
        )
        self._record_intent_change(decision.intent, current_time)
        return decision

    # ------------------------------------------------------------------
    # DROP logic
    # ------------------------------------------------------------------

    def _check_drop(
        self,
        current_time: float,
        energy_trend: float,
    ) -> BehaviourDecision | None:
        """Return DROP if energy has collapsed severely after active playing."""
        profile = self.profile

        # Must be entered
        if not self.has_entered:
            return None

        # Energy trend must be severely negative
        if energy_trend > profile.drop_trend_threshold:
            return None

        # Fast energy must be low
        fast = self.fast_energy_ema
        if fast is None or fast > profile.low_energy_threshold_for_drop:
            return None

        scores: dict[str, float] = {
            "fast_energy_ema": fast,
            "slow_energy_ema": self.slow_energy_ema or 0.0,
            "energy_trend": energy_trend,
            "drop_trend_threshold": profile.drop_trend_threshold,
            "low_energy_threshold_for_drop": profile.low_energy_threshold_for_drop,
        }
        if self.smoothed_density is not None:
            scores["smoothed_density"] = self.smoothed_density

        return BehaviourDecision(
            intent=BehaviourIntent.DROP,
            confidence=1.0 - fast,  # higher confidence when energy is lower
            reason="Drop: severe energy collapse after active playing",
            scores=scores,
            evaluated_at=current_time,
        )

    # ------------------------------------------------------------------
    # REDUCE logic
    # ------------------------------------------------------------------

    def _check_reduce(
        self,
        current_time: float,
        energy_trend: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> BehaviourDecision | None:
        """Return REDUCE if energy is falling or confidence is dipping."""
        profile = self.profile

        if not self.has_entered:
            return None

        maintain_pulse_threshold = profile.min_pulse_confidence - profile.maintain_hysteresis_margin
        maintain_bar_threshold = profile.min_bar_confidence - profile.maintain_hysteresis_margin

        # Check energy trend
        energy_falling = energy_trend <= profile.reduce_trend_threshold

        # Check moderate confidence loss (below maintain but above severe)
        pulse_weak = (
            pulse_conf < maintain_pulse_threshold
            and pulse_conf >= profile.severe_uncertainty_threshold
        )
        bar_weak = (
            bar_conf < maintain_bar_threshold
            and bar_conf >= profile.severe_uncertainty_threshold
        )

        if not energy_falling and not pulse_weak and not bar_weak:
            return None

        # Build reason
        if energy_falling:
            reason = "Reduce: energy trend falling"
        elif pulse_weak or bar_weak:
            reason = "Reduce: confidence below maintain threshold"
        else:
            reason = "Reduce: dynamic back-off"

        reduce_confidence = min(
            pulse_conf, bar_conf,
            1.0 - abs(energy_trend) if energy_falling else 1.0,
        )

        scores: dict[str, float] = {
            "fast_energy_ema": self.fast_energy_ema or 0.0,
            "slow_energy_ema": self.slow_energy_ema or 0.0,
            "energy_trend": energy_trend,
            "reduce_trend_threshold": profile.reduce_trend_threshold,
            "pulse_confidence": pulse_conf,
            "bar_confidence": bar_conf,
        }

        return BehaviourDecision(
            intent=BehaviourIntent.REDUCE,
            confidence=reduce_confidence,
            reason=reason,
            scores=scores,
            evaluated_at=current_time,
        )

    # ------------------------------------------------------------------
    # BUILD logic
    # ------------------------------------------------------------------

    def _check_build(
        self,
        current_time: float,
        energy_trend: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> BehaviourDecision | None:
        """Return BUILD if energy is clearly rising and situation is stable."""
        profile = self.profile

        if not self.has_entered:
            return None

        # Energy trend must be positive and above build threshold
        if energy_trend < profile.build_trend_threshold:
            return None

        # Pulse and bar confidence must be healthy
        if pulse_conf < profile.min_pulse_confidence:
            return None
        if bar_conf < profile.min_bar_confidence:
            return None

        # Density must be below max (dense performer needs space, not more drums)
        if self.smoothed_density is not None and self.smoothed_density > profile.max_density_for_build:
            return None

        build_confidence = min(
            pulse_conf, bar_conf,
            energy_trend / profile.build_trend_threshold,  # scale: 1.0 at threshold, higher above
        )

        scores: dict[str, float] = {
            "fast_energy_ema": self.fast_energy_ema or 0.0,
            "slow_energy_ema": self.slow_energy_ema or 0.0,
            "energy_trend": energy_trend,
            "build_trend_threshold": profile.build_trend_threshold,
            "pulse_confidence": pulse_conf,
            "bar_confidence": bar_conf,
            "max_density_for_build": profile.max_density_for_build,
        }
        if self.smoothed_density is not None:
            scores["smoothed_density"] = self.smoothed_density

        return BehaviourDecision(
            intent=BehaviourIntent.BUILD,
            confidence=min(build_confidence, 1.0),
            reason="Build: sustained rising energy with stable confidence",
            scores=scores,
            evaluated_at=current_time,
        )

    # ------------------------------------------------------------------
    # Cooldown / minimum duration enforcement
    # ------------------------------------------------------------------

    def _check_dynamic_cooldown(
        self,
        current_time: float,
    ) -> BehaviourDecision | None:
        """Return the current intent if minimum duration has not elapsed.

        Only enforces cooldown for BUILD, REDUCE, and DROP states.
        BAIL always bypasses cooldown (handled before this is called).

        Returns None if cooldown does not apply.
        """
        profile = self.profile

        # Only enforce cooldown for dynamic states
        if self.previous_intent == BehaviourIntent.BUILD:
            min_duration = profile.min_build_duration_seconds
        elif self.previous_intent == BehaviourIntent.REDUCE:
            min_duration = profile.min_reduce_duration_seconds
        elif self.previous_intent == BehaviourIntent.DROP:
            min_duration = profile.min_drop_duration_seconds
        else:
            return None  # MAINTAIN or entry intents have no cooldown

        elapsed = current_time - self.last_intent_change_time
        if elapsed < min_duration:
            # Stay in current state
            return BehaviourDecision(
                intent=self.previous_intent,
                confidence=0.5,  # placeholder for cooldown hold
                reason=f"Cooldown: minimum {self.previous_intent.value} duration not elapsed",
                scores={
                    "elapsed": elapsed,
                    "min_duration": min_duration,
                },
                evaluated_at=current_time,
            )

        return None

    # ------------------------------------------------------------------
    # Entry decision (Stage 2 — unchanged)
    # ------------------------------------------------------------------

    def _entry_decision(
        self,
        current_time: float,
        pulse_conf: float,
        bar_conf: float,
        stability: float,
        observation_readiness: float,
        entry_score: float,
    ) -> BehaviourDecision:
        """Decide whether to LISTEN, ENTER_SOFT, or ENTER_FULL."""
        profile = self.profile

        listen_reason = self._listen_block_reason(
            pulse_conf, bar_conf, observation_readiness, entry_score
        )
        if listen_reason is not None:
            decision = BehaviourDecision(
                intent=BehaviourIntent.LISTEN,
                confidence=entry_score,
                reason=listen_reason,
                scores={
                    "pulse_confidence": pulse_conf,
                    "bar_confidence": bar_conf,
                    "stability": stability,
                    "observation_readiness": observation_readiness,
                    "entry_score": entry_score,
                },
                evaluated_at=current_time,
            )
            self._record_intent_change(decision.intent, current_time)
            return decision

        if entry_score >= profile.full_entry_confidence:
            decision = self._enter_full(current_time, entry_score, pulse_conf, bar_conf)
        else:
            decision = self._enter_soft(current_time, entry_score, pulse_conf, bar_conf)
        self._record_intent_change(decision.intent, current_time)
        return decision

    def _listen_block_reason(
        self,
        pulse_conf: float,
        bar_conf: float,
        observation_readiness: float,
        entry_score: float,
    ) -> str | None:
        """Return a LISTEN reason if any entry gate is blocked, else None."""
        profile = self.profile

        if pulse_conf == 0.0 and bar_conf == 0.0:
            return "Listening: no pulse or bar state"
        if pulse_conf == 0.0:
            return "Listening: no pulse state"
        if bar_conf == 0.0:
            return "Listening: no bar state"
        if pulse_conf < profile.min_pulse_confidence:
            return "Listening: pulse confidence below threshold"
        if bar_conf < profile.min_bar_confidence:
            return "Listening: bar confidence below threshold"
        if observation_readiness < 1.0:
            return "Listening: observation window incomplete"
        if entry_score < profile.soft_entry_confidence:
            return "Listening: entry score below threshold"
        return None

    def _enter_soft(
        self,
        current_time: float,
        entry_score: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> BehaviourDecision:
        """Record ENTER_SOFT and mark the drummer as entered."""
        self.has_entered = True
        self.entered_at = current_time
        return BehaviourDecision(
            intent=BehaviourIntent.ENTER_SOFT,
            confidence=entry_score,
            reason="Enter soft: pulse and bar confidence stable after observation window",
            scores={
                "pulse_confidence": pulse_conf,
                "bar_confidence": bar_conf,
                "entry_score": entry_score,
            },
            evaluated_at=current_time,
        )

    def _enter_full(
        self,
        current_time: float,
        entry_score: float,
        pulse_conf: float,
        bar_conf: float,
    ) -> BehaviourDecision:
        """Record ENTER_FULL and mark the drummer as entered."""
        self.has_entered = True
        self.entered_at = current_time
        return BehaviourDecision(
            intent=BehaviourIntent.ENTER_FULL,
            confidence=entry_score,
            reason="Enter full: strong pulse and bar confidence after observation window",
            scores={
                "pulse_confidence": pulse_conf,
                "bar_confidence": bar_conf,
                "entry_score": entry_score,
            },
            evaluated_at=current_time,
        )

    # ------------------------------------------------------------------
    # Entry score computation (Stage 2 — unchanged)
    # ------------------------------------------------------------------

    def _compute_entry_score(
        self,
        pulse_conf: float,
        bar_conf: float,
        stability: float,
        observation_readiness: float,
    ) -> float:
        """Compute the composite entry score."""
        return pulse_conf * bar_conf * stability * observation_readiness

    def _observation_readiness(self, current_time: float) -> float:
        """Compute how ready the engine is based on observation time."""
        if self.first_event_time is None:
            return 0.0
        elapsed = current_time - self.first_event_time
        ratio = elapsed / self.profile.min_observation_seconds
        return max(0.0, min(1.0, ratio))

    # ------------------------------------------------------------------
    # Energy trend (Stage 3)
    # ------------------------------------------------------------------

    def _energy_trend(self) -> float:
        """Compute energy_trend = fast_energy_ema - slow_energy_ema.

        Returns 0.0 if either EMA is None.
        """
        if self.fast_energy_ema is None or self.slow_energy_ema is None:
            return 0.0
        return self.fast_energy_ema - self.slow_energy_ema

    # ------------------------------------------------------------------
    # Intent change tracking (Stage 3)
    # ------------------------------------------------------------------

    def _record_intent_change(self, new_intent: BehaviourIntent, current_time: float) -> None:
        """Record intent change time if intent differs from previous."""
        if new_intent != self.previous_intent:
            self.last_intent_change_time = current_time
        self.previous_intent = new_intent

    # ------------------------------------------------------------------
    # Stage 1 fallback
    # ------------------------------------------------------------------

    def _fallback_decision(self, current_time: float) -> BehaviourDecision:
        """Return the Stage 1 fallback decision.

        Confidence is 0.0 because this is a placeholder, not a
        confident musical decision.
        """
        if self.previous_intent == BehaviourIntent.MAINTAIN:
            intent = BehaviourIntent.MAINTAIN
        else:
            intent = BehaviourIntent.LISTEN

        self._record_intent_change(intent, current_time)
        return BehaviourDecision(
            intent=intent,
            confidence=0.0,
            reason="Stage 1 fallback state",
            scores={},
            evaluated_at=current_time,
        )

    # ------------------------------------------------------------------
    # Helper: safe PulseState / BarState access (Stage 2 — unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_pulse_confidence(pulse_state: Any) -> float:
        """Safely extract pulse confidence from a PulseState object."""
        if pulse_state is None:
            return 0.0
        if hasattr(pulse_state, "confidence"):
            val = pulse_state.confidence
            if isinstance(val, (int, float)):
                return float(val)
        return 0.0

    @staticmethod
    def _get_bar_confidence(bar_state: Any) -> float:
        """Safely extract bar confidence from a BarState object."""
        if bar_state is None:
            return 0.0
        if hasattr(bar_state, "confidence"):
            val = bar_state.confidence
            if isinstance(val, (int, float)):
                return float(val)
        return 0.0

    @staticmethod
    def _get_stability(pulse_state: Any, bar_state: Any) -> float:
        """Compute a combined stability score from pulse and bar state."""
        pulse_stability = 0.0
        if pulse_state is not None and hasattr(pulse_state, "stability"):
            label = getattr(pulse_state, "stability", "unknown")
            pulse_stability = BehaviourEngine._map_stability_label(label)

        bar_stability = 0.0
        if bar_state is not None:
            if hasattr(bar_state, "is_confident") and bar_state.is_confident:
                bar_stability = 1.0

        pulse_conf = BehaviourEngine._get_pulse_confidence(pulse_state)
        bar_conf = BehaviourEngine._get_bar_confidence(bar_state)
        if pulse_stability == 0.0 and bar_stability == 0.0:
            if pulse_conf >= 0.85 and bar_conf >= 0.85:
                return 1.0
            return 0.0

        if pulse_stability > 0.0 and bar_stability > 0.0:
            return (pulse_stability + bar_stability) / 2.0
        if pulse_stability > 0.0:
            return pulse_stability
        return bar_stability

    @staticmethod
    def _map_stability_label(label: str) -> float:
        """Map a pulse stability label string to a float."""
        mapping: dict[str, float] = {
            "stable": 1.0,
            "rising": 0.7,
            "falling": 0.4,
            "unknown": 0.0,
        }
        return mapping.get(label.lower() if isinstance(label, str) else "unknown", 0.0)

    # ------------------------------------------------------------------
    # Helpers (Stage 1)
    # ------------------------------------------------------------------

    @staticmethod
    def _get_event_time(event: Any) -> float | None:
        """Safely extract time from an event object."""
        if hasattr(event, "time_seconds"):
            return event.time_seconds
        if hasattr(event, "time"):
            return event.time
        if isinstance(event, dict) and "time_seconds" in event:
            return event["time_seconds"]
        if isinstance(event, dict) and "time" in event:
            return event["time"]
        return None

    @staticmethod
    def _get_event_energy(event: Any) -> float | None:
        """Safely extract energy from an event object."""
        if hasattr(event, "energy"):
            return event.energy
        if hasattr(event, "strength"):
            return event.strength
        if isinstance(event, dict) and "energy" in event:
            return event["energy"]
        if isinstance(event, dict) and "strength" in event:
            return event["strength"]
        return None

    @staticmethod
    def _get_event_density(event: Any) -> float | None:
        """Safely extract density from an event object."""
        if hasattr(event, "density"):
            return event.density
        if isinstance(event, dict) and "density" in event:
            return event["density"]
        return 0.0

    @staticmethod
    def _ema(current: float, previous: float | None, alpha: float) -> float:
        """Standard exponential moving average.

        new = alpha * current + (1 - alpha) * previous

        If ``previous`` is None, the current value is used as the
        initial seed.
        """
        if previous is None:
            return current
        return alpha * current + (1 - alpha) * previous


# ============================================================================
# Stage 4 — Feature-Driven Behaviour Engine
# ============================================================================


class FeatureDrivenBehaviourEngine:
    """Feature-driven behaviour engine using ``FeatureSnapshot``.

    This engine replaces raw event processing with structured feature
    evaluation.  It produces ``BehaviourDecision`` values from
    ``FeatureSnapshot`` objects rather than raw ``MusicalEvent`` lists.

    Priority order within a single evaluation:
        1. BAIL — silence exceeded feature bail threshold
        2. ENTER — pre-entry gating with confirmation counter
        3. ANCHOR — low certainty, poor repetition, poor phase
        4. BUILD — rising strength + change_score, decent certainty
        5. REDUCE — density inversion (player too busy)
        6. MAINTAIN — default bias to hold the pocket
        7. LISTEN — fallback (pre-entry without sufficient evidence)
    """

    def __init__(self, profile: DrummerProfile | None = None) -> None:
        self.profile = profile if profile is not None else ConservativePocketDrummer
        self.previous_intent: BehaviourIntent = BehaviourIntent.LISTEN
        self.has_entered: bool = False
        self.last_snapshot: Optional[Any] = None  # FeatureSnapshot or None

        # Confirmation tracking
        self._enter_confirmation_count: int = 0
        self._same_intent_count: int = 0

        # Timestamp tracking
        self._last_eval_time: float = 0.0
        self.entered_at: float | None = None

    def evaluate(
        self,
        snapshot,
        pulse_state: Any = None,
        bar_state: Any = None,
    ) -> BehaviourDecision:
        """Evaluate a ``FeatureSnapshot`` and return a behavioural decision.

        Parameters
        ----------
        snapshot : FeatureSnapshot
            Current feature summary from the Feature Monitor.
        pulse_state : Any, optional
            PulseState for pulse confidence extraction (optional).
        bar_state : Any, optional
            BarState for bar confidence extraction (optional).

        Returns
        -------
        BehaviourDecision
        """
        self.last_snapshot = snapshot
        now = snapshot.timestamp
        self._last_eval_time = now

        # 1. BAIL — silence override (highest priority)
        bail = self._check_feature_bail(snapshot)
        if bail is not None:
            self._record_intent_change(bail.intent)
            return bail

        # 2. Pre-entry: check ENTER conditions
        if not self.has_entered:
            return self._evaluate_entry(snapshot, pulse_state, bar_state)

        # 3. Priority-ordered dynamic decisions after entry
        # ANCHOR — protect the pocket when player is uncertain (highest)
        anchor = self._check_anchor(snapshot)
        if anchor is not None:
            self._record_intent_change(anchor.intent)
            return anchor

        # REDUCE — density inversion (player too busy)
        # Comes BEFORE BUILD because the drummer should not reward
        # frantic density with more complexity. BUILD can still
        # override REDUCE, but only when all controlled-build gates pass.
        reduce_ = self._check_feature_reduce(snapshot)
        if reduce_ is not None:
            # Allow BUILD to override REDUCE when controlled-build is clear
            build = self._check_feature_build(snapshot)
            if build is not None and build.confidence > reduce_.confidence:
                self._record_intent_change(build.intent)
                return build
            self._record_intent_change(reduce_.intent)
            return reduce_

        # BUILD — rising energy and change (only when REDUCE didn't fire)
        build = self._check_feature_build(snapshot)
        if build is not None:
            self._record_intent_change(build.intent)
            return build

        # Default: MAINTAIN — hold the pocket
        maintain = self._make_maintain(snapshot)
        self._record_intent_change(maintain.intent)
        return maintain

    def reset(self) -> None:
        """Clear all internal state back to factory-fresh."""
        self.previous_intent = BehaviourIntent.LISTEN
        self.has_entered = False
        self.last_snapshot = None
        self._enter_confirmation_count = 0
        self._same_intent_count = 0
        self._last_eval_time = 0.0
        self.entered_at = None

    # ------------------------------------------------------------------
    # Feature BAIL
    # ------------------------------------------------------------------

    def _check_feature_bail(self, snap) -> BehaviourDecision | None:
        """Return BAIL if silence exceeds the feature bail threshold.

        BAIL only fires after the engine has entered (has_entered is True).
        An empty pre-entry engine with long silence stays in LISTEN.
        """
        # Bail only makes sense after we've entered — otherwise we're
        # just waiting for input and should stay in LISTEN.
        if not self.has_entered:
            return None

        profile = self.profile
        silence = snap.silence_duration

        if silence > profile.feature_bail_silence_seconds:
            return BehaviourDecision(
                intent=BehaviourIntent.BAIL,
                confidence=min(1.0, silence / (profile.feature_bail_silence_seconds * 2)),
                reason="Feature BAIL: silence exceeded feature bail threshold",
                scores={
                    "silence_duration": silence,
                    "feature_bail_silence_seconds": profile.feature_bail_silence_seconds,
                },
                evaluated_at=snap.timestamp,
            )

        return None

    # ------------------------------------------------------------------
    # ENTER logic
    # ------------------------------------------------------------------

    def _evaluate_entry(
        self,
        snap,
        pulse_state: Any,
        bar_state: Any,
    ) -> BehaviourDecision:
        """Check whether to ENTER or continue LISTENing.

        Entry requires:
        * repetition_stability >= enter_repetition_threshold
        * player_certainty >= enter_certainty_threshold
        * If phase_alignment is available, it should not be terrible
        * Sustained evidence (confirmation counter)
        """
        profile = self.profile

        stability = snap.repetition_stability
        certainty = snap.player_certainty
        phase = snap.phase_alignment

        # Check basic thresholds
        stable_enough = stability >= profile.enter_repetition_threshold
        certain_enough = certainty >= profile.enter_certainty_threshold
        phase_ok = (
            phase is None
            or phase >= 0.5  # phase not actively bad
        )

        if stable_enough and certain_enough and phase_ok:
            self._enter_confirmation_count += 1
        else:
            # Reset confirmation on any failed snapshot
            self._enter_confirmation_count = 0

        # Require sustained confirmation
        if self._enter_confirmation_count >= profile.enter_confirmation_snapshots:
            self.has_entered = True
            self.entered_at = snap.timestamp
            return BehaviourDecision(
                intent=BehaviourIntent.ENTER_SOFT,
                confidence=certainty,
                reason="Feature ENTER: sustained repetition stability and player certainty",
                scores={
                    "repetition_stability": stability,
                    "player_certainty": certainty,
                    "phase_alignment": phase or 0.0,
                    "enter_certainty_threshold": profile.enter_certainty_threshold,
                    "enter_repetition_threshold": profile.enter_repetition_threshold,
                    "confirmation_count": self._enter_confirmation_count,
                },
                evaluated_at=snap.timestamp,
            )

        # Still LISTENing
        blocks: list[str] = []
        if not stable_enough:
            blocks.append(f"repetition_stability {stability:.3f} < {profile.enter_repetition_threshold}")
        if not certain_enough:
            blocks.append(f"player_certainty {certainty:.3f} < {profile.enter_certainty_threshold}")
        if not phase_ok:
            blocks.append(f"phase_alignment {phase:.3f} is poor")

        return BehaviourDecision(
            intent=BehaviourIntent.LISTEN,
            confidence=0.3,
            reason=f"Feature LISTEN: {', '.join(blocks)}" if blocks else "Feature LISTEN: waiting for entry conditions",
            scores={
                "repetition_stability": stability,
                "player_certainty": certainty,
                "phase_alignment": phase or 0.0,
                "confirmation_count": self._enter_confirmation_count,
            },
            evaluated_at=snap.timestamp,
        )

    # ------------------------------------------------------------------
    # ANCHOR logic
    # ------------------------------------------------------------------

    def _check_anchor(self, snap) -> BehaviourDecision | None:
        """Return ANCHOR if the player appears uncertain.

        ANCHOR means: simplify, play clearly, support the pulse metronomically.
        It is a protective state — the drummer should NOT be clever.

        Triggers when:
        * player_certainty < anchor_certainty_threshold
        * or repetition_stability < anchor_repetition_threshold
        * or phase_alignment < anchor_phase_threshold (if provided)

        Hysteresis: once in ANCHOR, recovery requires values above
        (threshold + hysteresis_margin).
        """
        profile = self.profile

        certainty = snap.player_certainty
        stability = snap.repetition_stability
        phase = snap.phase_alignment

        if self.previous_intent == BehaviourIntent.ANCHOR:
            # Already in ANCHOR — check if we've recovered enough to leave
            recovery_certainty = profile.anchor_certainty_threshold + profile.feature_hysteresis_margin
            recovery_stability = profile.anchor_repetition_threshold + profile.feature_hysteresis_margin
            recovery_phase = profile.anchor_phase_threshold + profile.feature_hysteresis_margin
            if (
                certainty >= recovery_certainty
                and stability >= recovery_stability
                and (phase is None or phase >= recovery_phase)
            ):
                return None  # Recovered enough to leave ANCHOR
            # Still unsteady — stay in ANCHOR
            # Determine anchor confidence
            low_certainty = certainty < profile.anchor_certainty_threshold
            low_stability = stability < profile.anchor_repetition_threshold
            poor_phase = (
                phase is not None
                and phase < profile.anchor_phase_threshold
            )
            anchor_conf = 1.0 - max(
                (profile.anchor_certainty_threshold - certainty) if low_certainty else 0,
                (profile.anchor_repetition_threshold - stability) if low_stability else 0,
                (profile.anchor_phase_threshold - (phase or 0)) if poor_phase else 0,
            )
            anchor_conf = max(0.0, min(1.0, anchor_conf))

            parts: list[str] = []
            if low_certainty:
                parts.append(f"player_certainty {certainty:.3f} < {profile.anchor_certainty_threshold}")
            if low_stability:
                parts.append(f"repetition_stability {stability:.3f} < {profile.anchor_repetition_threshold}")
            if poor_phase:
                parts.append(f"phase_alignment {phase:.3f} < {profile.anchor_phase_threshold}")

            return BehaviourDecision(
                intent=BehaviourIntent.ANCHOR,
                confidence=anchor_conf,
                reason=f"ANCHOR: {'; '.join(parts)}" if parts else "ANCHOR: recovering, still below hysteresis threshold",
                scores={
                    "player_certainty": certainty,
                    "repetition_stability": stability,
                    "phase_alignment": phase or 0.0,
                    "anchor_certainty_threshold": profile.anchor_certainty_threshold,
                    "anchor_repetition_threshold": profile.anchor_repetition_threshold,
                    "anchor_phase_threshold": profile.anchor_phase_threshold,
                },
                evaluated_at=snap.timestamp,
            )

        # Not already in ANCHOR — check entry conditions
        low_certainty = certainty < profile.anchor_certainty_threshold
        low_stability = stability < profile.anchor_repetition_threshold
        poor_phase = (
            phase is not None
            and phase < profile.anchor_phase_threshold
        )

        if not (low_certainty or low_stability or poor_phase):
            return None  # Everything is fine, no ANCHOR needed

        # Enter ANCHOR
        anchor_conf = 1.0 - max(
            (profile.anchor_certainty_threshold - certainty) if low_certainty else 0,
            (profile.anchor_repetition_threshold - stability) if low_stability else 0,
            (profile.anchor_phase_threshold - (phase or 0)) if poor_phase else 0,
        )
        anchor_conf = max(0.0, min(1.0, anchor_conf))

        parts: list[str] = []
        if low_certainty:
            parts.append(f"player_certainty {certainty:.3f} < {profile.anchor_certainty_threshold}")
        if low_stability:
            parts.append(f"repetition_stability {stability:.3f} < {profile.anchor_repetition_threshold}")
        if poor_phase:
            parts.append(f"phase_alignment {phase:.3f} < {profile.anchor_phase_threshold}")

        return BehaviourDecision(
            intent=BehaviourIntent.ANCHOR,
            confidence=anchor_conf,
            reason=f"ANCHOR: {'; '.join(parts)}",
            scores={
                "player_certainty": certainty,
                "repetition_stability": stability,
                "phase_alignment": phase or 0.0,
                "anchor_certainty_threshold": profile.anchor_certainty_threshold,
                "anchor_repetition_threshold": profile.anchor_repetition_threshold,
                "anchor_phase_threshold": profile.anchor_phase_threshold,
            },
            evaluated_at=snap.timestamp,
        )

    # ------------------------------------------------------------------
    # Feature BUILD logic
    # ------------------------------------------------------------------

    def _check_feature_build(self, snap) -> BehaviourDecision | None:
        """Return BUILD if strength is rising in a controlled, intentional way.

        BUILD is NOT about raw energy — it's about controlled, musical lift.
        The drummer should not reward frantic density with more complexity.

        Conditions:
        * change_score >= build_change_threshold
        * player_certainty >= build_certainty_threshold
        * repetition_stability >= build_repetition_threshold (controlled playing)
        * phase_alignment >= build_phase_threshold (if provided)
        * input_density is not chaotic: either below build_max_density_without_phrase,
          OR change_score is clearly dominant (controlled build through density)

        Hysteresis: if already BUILD, require change_score to drop below
        (threshold - hysteresis_margin) to exit.
        """
        profile = self.profile

        change = snap.change_score
        certainty = snap.player_certainty
        density = snap.input_density
        stability = snap.repetition_stability
        phase = snap.phase_alignment

        # Hysteresis: if already in BUILD, check exit threshold
        if self.previous_intent == BehaviourIntent.BUILD:
            exit_threshold = profile.build_change_threshold - profile.feature_hysteresis_margin
            if change >= exit_threshold:
                # Still building — stay in BUILD
                return BehaviourDecision(
                    intent=BehaviourIntent.BUILD,
                    confidence=min(certainty, min(change / profile.build_change_threshold, 1.0)),
                    reason=f"Feature BUILD (hold): change_score {change:.3f} within hysteresis band",
                    scores={
                        "change_score": change,
                        "player_certainty": certainty,
                        "input_density": density,
                        "build_change_threshold": profile.build_change_threshold,
                        "build_certainty_threshold": profile.build_certainty_threshold,
                    },
                    evaluated_at=snap.timestamp,
                )
            # Dropped below exit threshold — allow state transition
            return None

        # Not in BUILD — check entry thresholds

        # 1. Change score must be elevated
        if change < profile.build_change_threshold:
            return None

        # 2. Player certainty must be decent
        if certainty < profile.build_certainty_threshold:
            return None

        # 3. Repetition must be controlled — erratic playing blocks BUILD
        if stability < profile.build_repetition_threshold:
            return None

        # 4. Phase alignment must be reasonable (if provided)
        if phase is not None and phase < profile.build_phase_threshold:
            return None

        # 5. Density gating:
        #    - If density is below the "frantic" ceiling, BUILD is allowed
        #    - If density is high but change_score is clearly dominant
        #      (e.g. 2x the build threshold), BUILD can override density
        #    - Otherwise, block BUILD (let REDUCE handle it)
        if density >= profile.build_max_density_without_phrase:
            # High density — require strong controlled-build evidence
            if change < profile.build_change_threshold * 1.5:
                return None  # Not enough build evidence to override density

        build_confidence = min(
            certainty,
            min(change / profile.build_change_threshold, 1.0),
            stability,
        )

        return BehaviourDecision(
            intent=BehaviourIntent.BUILD,
            confidence=build_confidence,
            reason=f"Feature BUILD: change_score {change:.3f} >= {profile.build_change_threshold}, "
                   f"certainty {certainty:.3f} >= {profile.build_certainty_threshold}, "
                   f"stability {stability:.3f} >= {profile.build_repetition_threshold}",
            scores={
                "change_score": change,
                "player_certainty": certainty,
                "input_density": density,
                "repetition_stability": stability,
                "phase_alignment": phase or 0.0,
                "build_change_threshold": profile.build_change_threshold,
                "build_certainty_threshold": profile.build_certainty_threshold,
                "build_repetition_threshold": profile.build_repetition_threshold,
                "build_phase_threshold": profile.build_phase_threshold,
            },
            evaluated_at=snap.timestamp,
        )

    # ------------------------------------------------------------------
    # Feature REDUCE (Density Inversion) logic
    # ------------------------------------------------------------------

    def _check_feature_reduce(self, snap) -> BehaviourDecision | None:
        """Return REDUCE if input density is high (density inversion).

        When the player is busy, the drummer should simplify rather
        than match the complexity.  This is *musical restraint* — not panic.

        Hysteresis: if already REDUCE, require density to drop below
        (threshold - hysteresis_margin) to exit.
        """
        profile = self.profile

        density = snap.input_density

        # Hysteresis: if already in REDUCE, check exit threshold
        if self.previous_intent == BehaviourIntent.REDUCE:
            exit_threshold = profile.reduce_density_threshold - profile.feature_hysteresis_margin
            if density >= exit_threshold:
                # Still busy — stay in REDUCE
                return BehaviourDecision(
                    intent=BehaviourIntent.REDUCE,
                    confidence=min(1.0, density / profile.reduce_density_threshold),
                    reason=f"Feature REDUCE (hold): input_density {density:.3f} within hysteresis band",
                    scores={
                        "input_density": density,
                        "reduce_density_threshold": profile.reduce_density_threshold,
                        "player_certainty": snap.player_certainty,
                    },
                    evaluated_at=snap.timestamp,
                )
            # Density dropped enough — allow state transition
            return None

        # Not in REDUCE — check entry threshold
        if density < profile.reduce_density_threshold:
            return None

        reduce_confidence = min(1.0, density / profile.reduce_density_threshold)

        return BehaviourDecision(
            intent=BehaviourIntent.REDUCE,
            confidence=reduce_confidence,
            reason=f"Feature REDUCE: input_density {density:.3f} >= "
                   f"{profile.reduce_density_threshold} (density inversion — player too busy)",
            scores={
                "input_density": density,
                "reduce_density_threshold": profile.reduce_density_threshold,
                "player_certainty": snap.player_certainty,
            },
            evaluated_at=snap.timestamp,
        )

    # ------------------------------------------------------------------
    # MAINTAIN — default bias
    # ------------------------------------------------------------------

    def _make_maintain(self, snap) -> BehaviourDecision:
        """Return a MAINTAIN decision — the drummer's default bias."""
        return BehaviourDecision(
            intent=BehaviourIntent.MAINTAIN,
            confidence=snap.player_certainty,
            reason="Feature MAINTAIN: holding the pocket (default bias)",
            scores={
                "player_certainty": snap.player_certainty,
                "repetition_stability": snap.repetition_stability,
                "input_density": snap.input_density,
                "change_score": snap.change_score,
            },
            evaluated_at=snap.timestamp,
        )

    # ------------------------------------------------------------------
    # Intent change tracking
    # ------------------------------------------------------------------

    def _record_intent_change(self, new_intent: BehaviourIntent) -> None:
        """Record the new intent and track same-intent streak."""
        if new_intent != self.previous_intent:
            self._same_intent_count = 0
        else:
            self._same_intent_count += 1
        self.previous_intent = new_intent