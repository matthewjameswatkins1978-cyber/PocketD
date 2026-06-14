"""Phrase Marker / Ear-Perk Layer.

Adds tiny musical signposts during stable playing:
    - 8-bar ear-perk: a single subtle event to perk the listener's ears
    - 16-bar frill: a small 1-3 event pickup to mark location in the form

Design principle: "What's musical?" — tasteful, tiny, deterministic, easy to disable.

Bar counting convention
-----------------------
* Bar index 0 = first bar (musical bar 1)
* 8-bar marker fires on musical bars 8, 16, 24, 32...
  (zero-indexed bars 7, 15, 23, 31...)
* 16-bar marker fires on musical bars 16, 32, 48...
  (zero-indexed bars 15, 31, 47...)
* 16-bar marker takes priority over 8-bar marker when both fall on the same bar.

Allowed conditions
------------------
* intent is MAINTAIN or BUILD
* confidence is medium or high
* player_certainty, phase_alignment, repetition_stability are healthy
* density is not frantic
* not in ANCHOR, DROP, BAIL, or FINAL_BAIL
* not immediately after ANCHOR unless confidence has rebuilt

Output contracts preserved
--------------------------
* DROP remains sparse and unaffected
* BAIL remains 0 events
* FINAL_BAIL remains exactly kick + crash
* ANCHOR does not get decorative phrase markers
* REDUCE does not get phrase markers (unless explicitly stable later)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from drummer.behaviour import BehaviourIntent
from drummer.confidence import PerformanceConfidenceState
from drummer.feel import GrooveEvent
from perception.features import FeatureSnapshot


# ---------------------------------------------------------------------------
# PhraseMarkerType
# ---------------------------------------------------------------------------


class PhraseMarkerType(str, Enum):
    """The type of phrase marker selected for a given bar.

    Values are ordered by priority: higher value = higher priority.
    """

    NONE = "none"
    EIGHT_BAR_EAR_PERK = "eight_bar_ear_perk"
    SIXTEEN_BAR_FRILL = "sixteen_bar_frill"

    # Priority ordering: SIXTEEN_BAR_FRILL > EIGHT_BAR_EAR_PERK > NONE
    @property
    def priority(self) -> int:
        return {"none": 0, "eight_bar_ear_perk": 1, "sixteen_bar_frill": 2}[self.value]


# ---------------------------------------------------------------------------
# PhraseMarkerConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhraseMarkerConfig:
    """Tuning parameters for phrase marker selection.

    All thresholds have sensible defaults tuned for musical taste.
    """

    enabled: bool = True
    """Master enable/disable switch.  Set to False to suppress all markers."""

    # --- Thresholds ---
    eight_bar_min_confidence: float = 0.45
    """Minimum confidence for an 8-bar ear-perk to fire."""

    sixteen_bar_min_confidence: float = 0.60
    """Minimum confidence for a 16-bar frill to fire."""

    min_phase: float = 0.60
    """Minimum phase_alignment for any phrase marker."""

    min_certainty: float = 0.60
    """Minimum player_certainty for any phrase marker."""

    min_stability: float = 0.60
    """Minimum repetition_stability for any phrase marker."""

    max_density: float = 0.75
    """Maximum input_density for any phrase marker (not frantic)."""

    bars_after_anchor_grace: int = 4
    """Number of bars to wait after ANCHOR before allowing phrase markers,
    unless confidence has rebuilt above the threshold."""

    min_confidence_post_anchor: float = 0.60
    """If confidence is >= this after ANCHOR, grace period is bypassed."""

    # --- Velocity boosts for ear-perks ---
    ear_perk_kick_boost: int = 8
    """Velocity increase for stronger kick on beat 1 during 8-bar ear-perk."""

    ear_perk_ghost_snare_velocity: int = 35
    """Velocity for extra ghost snare during 8-bar ear-perk."""

    ear_perk_hat_lift_velocity: int = 85
    """Velocity for short closed-hat lift during 8-bar ear-perk."""

    ear_perk_pickup_kick_velocity: int = 75
    """Velocity for pickup kick during 8-bar ear-perk."""

    # --- 16-bar frill settings ---
    frill_snare_pickup_velocity: int = 95
    """Velocity for snare pickup into next bar during 16-bar frill."""

    frill_kick_velocity: int = 100
    """Velocity for kick notes in 16-bar frill gestures."""

    frill_hat_flourish_velocity: int = 85
    """Velocity for hat flourish notes in 16-bar frill."""


# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_PHRASE_CONFIG = PhraseMarkerConfig()


# ---------------------------------------------------------------------------
# PhraseMarkerState
# ---------------------------------------------------------------------------


@dataclass
class PhraseMarkerState:
    """Tracks state for phrase marker decisions.

    Parameters
    ----------
    bars_since_anchor : int
        Number of bars elapsed since the last ANCHOR intent.
    last_marker_bar : int
        The bar index (zero-indexed) where the last phrase marker was placed.
    last_marker_type : PhraseMarkerType
        The type of the last phrase marker placed.
    marker_count : int
        Total number of phrase markers placed so far.
    """

    bars_since_anchor: int = 0
    last_marker_bar: int = -1
    last_marker_type: PhraseMarkerType = PhraseMarkerType.NONE
    marker_count: int = 0

    def reset(self) -> None:
        """Reset all state to factory-fresh."""
        self.bars_since_anchor = 0
        self.last_marker_bar = -1
        self.last_marker_type = PhraseMarkerType.NONE
        self.marker_count = 0


# ---------------------------------------------------------------------------
# Phrase marker selection
# ---------------------------------------------------------------------------


def is_bar_8_boundary(bar_index: int) -> bool:
    """Check if *bar_index* (zero-indexed) is an 8-bar boundary.

    Musical bar 8 → zero-indexed bar 7 (bar_index % 8 == 7)
    Musical bar 16 → zero-indexed bar 15, etc.
    """
    if bar_index < 7:
        return False
    return (bar_index + 1) % 8 == 0


def is_bar_16_boundary(bar_index: int) -> bool:
    """Check if *bar_index* (zero-indexed) is a 16-bar boundary.

    Musical bar 16 → zero-indexed bar 15 (bar_index % 16 == 15)
    Musical bar 32 → zero-indexed bar 31, etc.
    """
    if bar_index < 15:
        return False
    return (bar_index + 1) % 16 == 0


def is_musically_safe(
    intent: BehaviourIntent,
    snapshot: FeatureSnapshot,
    confidence: float,
    config: PhraseMarkerConfig = _DEFAULT_PHRASE_CONFIG,
    state: PhraseMarkerState | None = None,
) -> bool:
    """Check if it is musically safe to place a phrase marker.

    Returns True only when all conditions are favourable.
    """
    # Only MAINTAIN or BUILD are eligible for phrase markers
    if intent not in (BehaviourIntent.MAINTAIN, BehaviourIntent.BUILD):
        return False

    # Confidence check
    if confidence < config.eight_bar_min_confidence:
        return False

    # Feature health checks
    certainty = snapshot.player_certainty
    phase = snapshot.phase_alignment or 0.0
    stability = snapshot.repetition_stability
    density = snapshot.input_density

    if phase < config.min_phase:
        return False
    if certainty < config.min_certainty:
        return False
    if stability < config.min_stability:
        return False
    if density > config.max_density:
        return False

    # Post-ANCHOR grace period check
    if state is not None and state.bars_since_anchor < config.bars_after_anchor_grace:
        # Bypass grace period if confidence has rebuilt sufficiently
        if confidence < config.min_confidence_post_anchor:
            return False

    return True


def choose_phrase_marker(
    bar_index: int,
    intent: BehaviourIntent,
    confidence: float,
    snapshot: FeatureSnapshot,
    config: PhraseMarkerConfig = _DEFAULT_PHRASE_CONFIG,
    state: PhraseMarkerState | None = None,
) -> PhraseMarkerType:
    """Choose the appropriate phrase marker for this bar.

    Parameters
    ----------
    bar_index : int
        Zero-indexed bar number.
    intent : BehaviourIntent
        Current behaviour intent.
    confidence : float
        Current confidence value [0.0, 1.0].
    snapshot : FeatureSnapshot
        Current feature snapshot.
    config : PhraseMarkerConfig
        Configuration thresholds.
    state : PhraseMarkerState | None
        Optional state tracker (used for post-ANCHOR grace period).

    Returns
    -------
    PhraseMarkerType
        The type of phrase marker to apply.
    """
    if not config.enabled:
        return PhraseMarkerType.NONE

    if not is_musically_safe(intent, snapshot, confidence, config, state):
        return PhraseMarkerType.NONE

    # Check 16-bar boundary first (higher priority)
    if is_bar_16_boundary(bar_index):
        if confidence >= config.sixteen_bar_min_confidence:
            return PhraseMarkerType.SIXTEEN_BAR_FRILL

    # Check 8-bar boundary
    if is_bar_8_boundary(bar_index):
        if confidence >= config.eight_bar_min_confidence:
            return PhraseMarkerType.EIGHT_BAR_EAR_PERK

    return PhraseMarkerType.NONE


# ---------------------------------------------------------------------------
# Phrase marker rendering
# ---------------------------------------------------------------------------
# These functions add/subtract events from a shaped bar to implement the
# chosen phrase marker.  They are deterministic (no random choices).
# ---------------------------------------------------------------------------


def apply_eight_bar_ear_perk(
    events: list[GrooveEvent],
    bar_index: int,
    config: PhraseMarkerConfig = _DEFAULT_PHRASE_CONFIG,
) -> list[GrooveEvent]:
    """Apply a tiny 8-bar ear-perk to the given bar events.

    Selects one of several deterministic patterns based on ``bar_index``
    to ensure variety across different 8-bar boundaries without randomness.

    Pattern selection (deterministic by bar_index // 8):
    - Pattern 0: Slightly stronger kick on beat 1 (boost velocity of existing kick)
    - Pattern 1: Tiny extra ghost snare at position 14 (pickup into next bar)
    - Pattern 2: Short closed-hat lift at position 14
    - Pattern 3: One small pickup kick at position 14

    Returns a new list with the marker applied.  At most +1 event added
    or one existing event modified.
    """
    cfg = config

    if not events:
        # No events to mark — add a subtle kick on beat 1
        return [
            GrooveEvent("kick", 0, bar_index=bar_index,
                        velocity=cfg.ear_perk_pickup_kick_velocity,
                        source_role="main")
        ]

    pattern = (bar_index // 8) % 4
    result = list(events)

    if pattern == 0:
        # Slightly stronger kick on beat 1
        applied = False
        for i, evt in enumerate(result):
            if _is_kick(evt) and (evt.grid_position % 16) == 0:
                new_vel = min(127, evt.velocity + cfg.ear_perk_kick_boost)
                result[i] = evt.copy_with(velocity=new_vel)
                applied = True
                break
        if not applied:
            # No kick on beat 1 — add a gentle one
            result.append(GrooveEvent(
                "kick", 0, bar_index=bar_index,
                velocity=cfg.ear_perk_kick_boost + 80,
                source_role="main",
            ))

    elif pattern == 1:
        # Tiny extra ghost snare at position 14 (pickup into next bar)
        pos = 14
        # Check no event already at this position
        if not any(e.grid_position % 16 == pos for e in result):
            result.append(GrooveEvent(
                "snare", pos, bar_index=bar_index,
                velocity=cfg.ear_perk_ghost_snare_velocity,
                articulation="ghost", source_role="ghost",
            ))

    elif pattern == 2:
        # Short closed-hat lift at position 14
        pos = 14
        if not any(e.grid_position % 16 == pos for e in result):
            result.append(GrooveEvent(
                "hi_hat", pos, bar_index=bar_index,
                velocity=cfg.ear_perk_hat_lift_velocity,
                articulation="closed", source_role="main",
            ))

    elif pattern == 3:
        # One small pickup kick at position 14
        pos = 14
        if not any(e.grid_position % 16 == pos for e in result):
            result.append(GrooveEvent(
                "kick", pos, bar_index=bar_index,
                velocity=cfg.ear_perk_pickup_kick_velocity,
                source_role="main",
            ))

    return result


def apply_sixteen_bar_frill(
    events: list[GrooveEvent],
    bar_index: int,
    config: PhraseMarkerConfig = _DEFAULT_PHRASE_CONFIG,
) -> list[GrooveEvent]:
    """Apply a small 16-bar frill/pickup to the given bar events.

    Selects one of several deterministic patterns based on ``bar_index``:
    - Pattern 0: Tiny snare pickup (positions 13, 15) into next bar 1
    - Pattern 1: Kick-snare-kick gesture (positions 10, 12, 14)
    - Pattern 2: Small hat flourish (positions 10, 12, 14)

    No crash.  At most +3 events.
    """
    cfg = config
    result = list(events)

    pattern = (bar_index // 16) % 3

    if pattern == 0:
        # Tiny snare pickup: two ghost snares at 13, 15 leading into next bar
        for pos in (13, 15):
            if not any(e.grid_position % 16 == pos for e in result):
                result.append(GrooveEvent(
                    "snare", pos, bar_index=bar_index,
                    velocity=cfg.frill_snare_pickup_velocity,
                    articulation="ghost", source_role="ghost",
                ))

    elif pattern == 1:
        # Kick-snare-kick gesture: positions 10, 12, 14
        gestures = [
            ("kick", 10, cfg.frill_kick_velocity),
            ("snare", 12, cfg.frill_snare_pickup_velocity),
            ("kick", 14, cfg.frill_kick_velocity),
        ]
        for inst, pos, vel in gestures:
            if not any(e.grid_position % 16 == pos for e in result):
                result.append(GrooveEvent(
                    inst, pos, bar_index=bar_index,
                    velocity=vel, source_role="main",
                ))

    elif pattern == 2:
        # Small hat flourish: positions 10, 12, 14
        for pos in (10, 12, 14):
            if not any(e.grid_position % 16 == pos for e in result):
                result.append(GrooveEvent(
                    "hi_hat", pos, bar_index=bar_index,
                    velocity=cfg.frill_hat_flourish_velocity,
                    articulation="closed", source_role="main",
                ))

    return result


def apply_phrase_marker(
    events: list[GrooveEvent],
    marker_type: PhraseMarkerType,
    bar_index: int,
    config: PhraseMarkerConfig = _DEFAULT_PHRASE_CONFIG,
) -> list[GrooveEvent]:
    """Apply the chosen phrase marker to the bar's events.

    Parameters
    ----------
    events : list[GrooveEvent]
        The shaped events for this bar (already processed by OutputShaper).
    marker_type : PhraseMarkerType
        The type of phrase marker to apply.
    bar_index : int
        Zero-indexed bar number (used for deterministic pattern selection).
    config : PhraseMarkerConfig
        Configuration for marker rendering.

    Returns
    -------
    list[GrooveEvent]
        Events with phrase marker applied, sorted by grid_position.
    """
    if marker_type == PhraseMarkerType.NONE or not config.enabled:
        return list(events)

    if marker_type == PhraseMarkerType.EIGHT_BAR_EAR_PERK:
        result = apply_eight_bar_ear_perk(events, bar_index, config)
    elif marker_type == PhraseMarkerType.SIXTEEN_BAR_FRILL:
        result = apply_sixteen_bar_frill(events, bar_index, config)
    else:
        result = list(events)

    # Sort by grid_position
    result.sort(key=lambda e: e.grid_position)
    return result


# ---------------------------------------------------------------------------
# Instrument group helpers
# ---------------------------------------------------------------------------


def _is_kick(evt: GrooveEvent) -> bool:
    """True if this event is a kick drum."""
    inst = evt.instrument.lower()
    return inst in ("kick", "kik")


def _count_events_by_type(events: list[GrooveEvent]) -> dict[str, int]:
    """Count events by instrument type."""
    counts: dict[str, int] = {}
    for evt in events:
        counts[evt.instrument] = counts.get(evt.instrument, 0) + 1
    return counts


# ---------------------------------------------------------------------------
# Convenience: get a human-readable label
# ---------------------------------------------------------------------------


def phrase_marker_label(marker_type: PhraseMarkerType) -> str:
    """Return a short human-readable label for the marker type."""
    labels = {
        PhraseMarkerType.NONE: "",
        PhraseMarkerType.EIGHT_BAR_EAR_PERK: "8bar",
        PhraseMarkerType.SIXTEEN_BAR_FRILL: "16bar",
    }
    return labels.get(marker_type, "")