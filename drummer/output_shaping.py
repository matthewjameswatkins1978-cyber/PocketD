"""Behaviour-Driven Output Shaping.

Translates behaviour intent into MIDI-note-level modifications.
Accepts ``GrooveEvent`` lists and ``BehaviourIntent`` values and returns
shaped event lists that match the drummer's current intention.

Design contract
---------------
* Pure: input notes + intent → shaped notes (no side effects, no clock).
* Deterministic: same inputs always produce same outputs.
* Instrument-aware: identifies kick, snare, hi_hat, ride, crash, toms.
* Preserves time order (sorted by grid_position then bar_index).
* Never crashes on unknown instruments.
* Timing is preserved — shaping only alters velocity and articulation.
* ``humanize_amount`` scales all velocity/articulation modifications
  from 0.0 (pure machine grid) to 1.0 (full expression).

Behaviour output rules
----------------------
* DROP:  must produce > 0 events, usually 1–2 sparse kicks.  No crash.
* BAIL:  must produce exactly 0 events (song is over).
* FINAL_BAIL:  must produce exactly 2 events: kick (note 36) + crash (note 49)
  on beat 1 (grid_position 0), then no further notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drummer.behaviour import BehaviourIntent
from drummer.feel import GrooveEvent, _instrument_group


# ---------------------------------------------------------------------------
# OutputShapingConfig
# ---------------------------------------------------------------------------


@dataclass
class OutputShapingConfig:
    """Tuning knobs for the Output Shaper.

    All parameters have sensible defaults.  Tweak them to adjust how
    aggressively each intent shapes the output.

    ``humanize_amount`` scales all velocity/articulation modifications.
    0.0 = machine-tight (no changes).  1.0 = full expression.
    """

    # Global scaling
    humanize_amount: float = 1.0
    """0.0 = machine-tight (no velocity/articulation changes).
    1.0 = full expression.  0.25 = subtle."""

    # Reduce settings
    reduce_min_snare_velocity: int = 60
    """Snare notes with velocity below this are candidates for removal."""

    reduce_thin_hats: bool = True
    """When True, thin 16th-note hi-hats to 8th notes during REDUCE."""

    reduce_strip_ghosts: bool = True
    """When True, remove ghost notes during REDUCE."""

    reduce_preserve_strong_beats: bool = True
    """When True, preserve kick on beat 1 and snare on 2 & 4 during REDUCE."""

    # Anchor settings
    anchor_strip_ghosts: bool = True
    """When True, remove ghost notes during ANCHOR."""

    anchor_strip_syncopated: bool = True
    """When True, remove syncopated kick decorations (off-beat kicks)."""

    anchor_simplify_hats: bool = True
    """When True, thin hi-hats to quarter or 8th notes during ANCHOR."""

    anchor_reduce_velocity_variation: bool = True
    """When True, squash velocity variation toward a target during ANCHOR."""

    anchor_target_velocity: int = 100
    """Target velocity for velocity simplification during ANCHOR."""

    # Build settings
    build_velocity_boost: int = 12
    """Add this much velocity during BUILD (clamped at 127)."""

    build_max_velocity: int = 127
    """Maximum velocity allowed after BUILD boost."""

    build_open_hats: bool = True
    """When True, convert closed hi-hat articulation to 'open' during BUILD."""

    # Enter settings
    enter_velocity_cap: int = 100
    """Maximum velocity during ENTER_SOFT (first entry should not be obnoxious)."""

    enter_soft_scale: float = 0.85
    """Scale factor applied to velocities during ENTER_SOFT."""

    # Ghost note identification
    ghost_max_velocity: int = 35
    """Notes with velocity at or below this are considered ghost-velocity."""

    # DROP output tuning
    drop_kick_velocity: int = 100
    """Velocity for kick note(s) during DROP."""
    drop_num_kicks: int = 1
    """Number of kick pulses during DROP (1 or 2)."""
    drop_kick_grid_positions: tuple = (0, 8)
    """Grid positions (16th-note) for DROP kicks.
    Default (0, 8) → beat 1 and beat 3.  Only first ``drop_num_kicks`` used."""

    # FINAL_BAIL output tuning
    final_bail_kick_velocity: int = 110
    """Velocity for the kick on the final hit."""
    final_bail_crash_velocity: int = 110
    """Velocity for the crash on the final hit."""
    final_bail_kick_grid: int = 0
    """Grid position for the kick (beat 1)."""
    final_bail_crash_grid: int = 0
    """Grid position for the crash (beat 1, same hit)."""


# ---------------------------------------------------------------------------
# Output-validation helpers
# ---------------------------------------------------------------------------


def is_drop_output(events: list[GrooveEvent]) -> bool:
    """Confirm events look like a DROP output.

    DROP must have:
    * event count > 0
    * event count <= 2 (or very sparse — no full groove)
    * no crash
    * no full hats
    * no normal groove (only kick allowed, optionally snare at very low vel)
    """
    if not events:
        return False
    if len(events) > 2:
        # More than 2 events is not "sparse" — fail
        return False
    for evt in events:
        group = _instrument_group(evt.instrument)
        if group == "crash":
            return False
        if group in ("hi_hat", "ride") and evt.velocity > 40:
            # Hats/ride with moderate velocity suggest full groove
            return False
        if group == "snare" and evt.velocity > 50:
            return False
    # At least one kick is expected
    has_kick = any(_instrument_group(e.instrument) == "kick" for e in events)
    return has_kick


def is_bail_output(events: list[GrooveEvent]) -> bool:
    """Confirm events look like a BAIL output (exactly 0 events)."""
    return len(events) == 0


def is_final_bail_output(events: list[GrooveEvent]) -> bool:
    """Confirm events look like a FINAL_BAIL output.

    FINAL_BAIL must have:
    * exactly 2 events
    * one kick (group "kick")
    * one crash (group "crash")
    * both on grid_position 0 (beat 1)
    """
    if len(events) != 2:
        return False
    groups = [_instrument_group(e.instrument) for e in events]
    if "kick" not in groups or "crash" not in groups:
        return False
    for evt in events:
        if evt.grid_position != 0:
            return False
    return True


# ---------------------------------------------------------------------------
# Instrument classification helpers
# ---------------------------------------------------------------------------


def _is_kick(evt: GrooveEvent) -> bool:
    """True if this event is a kick drum."""
    return _instrument_group(evt.instrument) == "kick"


def _is_snare(evt: GrooveEvent) -> bool:
    """True if this event is a snare drum."""
    return _instrument_group(evt.instrument) == "snare"


def _is_hat(evt: GrooveEvent) -> bool:
    """True if this event is a hi-hat."""
    return _instrument_group(evt.instrument) == "hi_hat"


def _is_ride(evt: GrooveEvent) -> bool:
    """True if this event is a ride cymbal."""
    return _instrument_group(evt.instrument) == "ride"


def _is_crash(evt: GrooveEvent) -> bool:
    """True if this event is a crash cymbal."""
    return _instrument_group(evt.instrument) == "crash"


def _is_tom(evt: GrooveEvent) -> bool:
    """True if this event is a tom drum."""
    return _instrument_group(evt.instrument) == "toms"


def _is_ghost(evt: GrooveEvent, config: OutputShapingConfig) -> bool:
    """True if this event is a ghost note by articulation or velocity."""
    if evt.articulation == "ghost":
        return True
    if evt.source_role == "ghost":
        return True
    if evt.velocity <= config.ghost_max_velocity and (
        _is_snare(evt) or _is_hat(evt)
    ):
        return True
    return False


# ---------------------------------------------------------------------------
# Beat position helpers
# ---------------------------------------------------------------------------


def _beat_in_bar(grid_position: int) -> int:
    """Return the beat number (1-4) for a 16th-note grid position."""
    pos_in_bar = grid_position % 16
    beat_map = {0: 1, 4: 2, 8: 3, 12: 4}
    return beat_map.get(pos_in_bar, -1)


def _is_strong_beat(grid_position: int) -> bool:
    """True if this grid position falls on a quarter-note pulse (beat)."""
    return (grid_position % 4) == 0


def _is_eighth_note(grid_position: int) -> bool:
    """True if this grid position is an 8th note (even 16th position)."""
    return (grid_position % 2) == 0


# ---------------------------------------------------------------------------
# BehaviourOutputShaper
# ---------------------------------------------------------------------------


class BehaviourOutputShaper:
    """Shapes a list of GrooveEvents according to a BehaviourIntent.

    Usage
    -----
    >>> shaper = BehaviourOutputShaper()
    >>> shaped = shaper.shape(groove_events, BehaviourIntent.REDUCE)
    """

    def __init__(self, config: OutputShapingConfig | None = None) -> None:
        self.config = config if config is not None else OutputShapingConfig()

    def shape(
        self,
        events: list[GrooveEvent],
        intent: BehaviourIntent,
        bar_position: int | None = None,
    ) -> list[GrooveEvent]:
        """Shape a sequence of GrooveEvents according to the given intent.

        Parameters
        ----------
        events : list[GrooveEvent]
            Input drum events to shape.
        intent : BehaviourIntent
            The current behaviour intention.
        bar_position : int | None
            Current 16th-note position within the bar (optional context).

        Returns
        -------
        list[GrooveEvent]
            Shaped events, sorted by grid_position then bar_index.
        """
        # Dispatch to intent-specific shaper
        # DROP, BAIL, FINAL_BAIL all produce their own output from scratch
        # regardless of input events (the input groove is discarded).
        if intent == BehaviourIntent.DROP:
            result = self._shape_drop()
        elif intent == BehaviourIntent.BAIL:
            result = self._shape_bail(events)
        elif intent == BehaviourIntent.FINAL_BAIL:
            result = self._shape_final_bail()
        elif intent == BehaviourIntent.MAINTAIN:
            result = self._shape_maintain(events)
        elif intent == BehaviourIntent.REDUCE:
            result = self._shape_reduce(events)
        elif intent == BehaviourIntent.ANCHOR:
            result = self._shape_anchor(events)
        elif intent == BehaviourIntent.BUILD:
            result = self._shape_build(events)
        elif intent in (BehaviourIntent.ENTER_SOFT, BehaviourIntent.ENTER_FULL):
            result = self._shape_enter(events)
        else:
            # LISTEN, FILL, CRASH, etc. — pass through unchanged
            result = list(events)

        # Always sort output by bar_index then grid_position
        result.sort(key=lambda e: (e.bar_index, e.grid_position))
        return result

    # ------------------------------------------------------------------
    # DROP — sparse kick pulses (1–2 events, no crash)
    # ------------------------------------------------------------------

    def _shape_drop(self) -> list[GrooveEvent]:
        """DROP returns 1–2 sparse kick pulses, no crash, no hats, no snare.

        The drummer is still active but playing very sparsely.
        """
        cfg = self.config
        num_kicks = max(1, min(2, cfg.drop_num_kicks))
        positions = cfg.drop_kick_grid_positions[:num_kicks]
        return [
            GrooveEvent(
                instrument="kick",
                grid_position=pos,
                velocity=cfg.drop_kick_velocity,
                articulation="default",
                source_role="main",
            )
            for pos in positions
        ]

    # ------------------------------------------------------------------
    # FINAL_BAIL — kick + crash on beat 1, then stop
    # ------------------------------------------------------------------

    def _shape_final_bail(self) -> list[GrooveEvent]:
        """FINAL_BAIL returns exactly kick + crash on beat 1 (grid 0).

        The drummer thinks this is a deliberate ending.
        """
        cfg = self.config
        return [
            GrooveEvent(
                instrument="kick",
                grid_position=cfg.final_bail_kick_grid,
                velocity=cfg.final_bail_kick_velocity,
                articulation="default",
                source_role="main",
            ),
            GrooveEvent(
                instrument="crash",
                grid_position=cfg.final_bail_crash_grid,
                velocity=cfg.final_bail_crash_velocity,
                articulation="default",
                source_role="main",
            ),
        ]

    # ------------------------------------------------------------------
    # MAINTAIN — preserve the pocket (machine-tight unless humanized)
    # ------------------------------------------------------------------

    def _shape_maintain(self, events: list[GrooveEvent]) -> list[GrooveEvent]:
        """MAINTAIN leaves the groove mostly unchanged.

        At ``humanize_amount == 0.0``, output is identical to input.
        """
        return list(events)

    # ------------------------------------------------------------------
    # REDUCE — simplify when the player is busy
    # ------------------------------------------------------------------

    def _shape_reduce(self, events: list[GrooveEvent]) -> list[GrooveEvent]:
        """REDUCE strips ghost notes, decorations, and thins hats."""
        cfg = self.config
        result: list[GrooveEvent] = []

        for evt in events:
            # Remove ghost notes — but NEVER ghost-strip essential backbeat
            if cfg.reduce_strip_ghosts and _is_ghost(evt, cfg):
                if _is_snare(evt):
                    beat = _beat_in_bar(evt.grid_position)
                    if cfg.reduce_preserve_strong_beats and beat in (2, 4) and _is_strong_beat(evt.grid_position):
                        pass
                    else:
                        continue
                elif _is_kick(evt):
                    if cfg.reduce_preserve_strong_beats and _is_strong_beat(evt.grid_position) and _beat_in_bar(evt.grid_position) == 1:
                        pass
                    else:
                        continue
                else:
                    continue

            # Remove low-velocity snare decorations
            if _is_snare(evt) and evt.velocity < cfg.reduce_min_snare_velocity:
                beat = _beat_in_bar(evt.grid_position)
                if not (cfg.reduce_preserve_strong_beats and beat in (2, 4) and _is_strong_beat(evt.grid_position)):
                    continue

            # Thin 16th-note hi-hats to 8th notes
            if cfg.reduce_thin_hats and _is_hat(evt):
                if not _is_eighth_note(evt.grid_position):
                    continue

            result.append(evt)

        return result

    # ------------------------------------------------------------------
    # ANCHOR — become clearer and more metronomic
    # ------------------------------------------------------------------

    def _shape_anchor(self, events: list[GrooveEvent]) -> list[GrooveEvent]:
        """ANCHOR strips decorations and simplifies to a clear pulse.

        At ``humanize_amount == 0.0``, only structural simplification
        applies (no velocity changes).  At 1.0, full velocity pull toward
        anchor_target_velocity.
        """
        cfg = self.config
        amount = cfg.humanize_amount
        result: list[GrooveEvent] = []

        for evt in events:
            # Remove ghost notes
            if cfg.anchor_strip_ghosts and _is_ghost(evt, cfg):
                continue

            # Strip syncopated kick decorations
            if cfg.anchor_strip_syncopated and _is_kick(evt):
                beat = _beat_in_bar(evt.grid_position)
                if not _is_strong_beat(evt.grid_position):
                    continue
                if beat not in (1, 3):
                    continue

            # Strip syncopated snare decorations — only 2 and 4
            if cfg.anchor_strip_syncopated and _is_snare(evt):
                beat = _beat_in_bar(evt.grid_position)
                if not _is_strong_beat(evt.grid_position):
                    continue
                if beat not in (2, 4):
                    continue

            # Simplify hi-hats to quarter notes
            if cfg.anchor_simplify_hats and _is_hat(evt):
                if not _is_eighth_note(evt.grid_position):
                    continue
                if not _is_strong_beat(evt.grid_position):
                    continue

            # Reduce velocity variation (scaled by humanize_amount)
            if cfg.anchor_reduce_velocity_variation and amount > 0.0:
                target = cfg.anchor_target_velocity
                current = evt.velocity
                # Pull toward target, scaled by amount
                blend = 0.5 * amount
                new_vel = int(current + (target - current) * blend)
                new_vel = max(1, min(127, new_vel))
                if new_vel != current:
                    evt = evt.copy_with(velocity=new_vel)

            result.append(evt)

        # Ensure minimum anchor pulse
        has_beat1_kick = any(
            _is_kick(e) and _is_strong_beat(e.grid_position)
            and _beat_in_bar(e.grid_position) == 1
            for e in result
        )
        has_beat2_snare = any(
            _is_snare(e) and _is_strong_beat(e.grid_position)
            and _beat_in_bar(e.grid_position) == 2
            for e in result
        )
        has_beat3_kick = any(
            _is_kick(e) and _is_strong_beat(e.grid_position)
            and _beat_in_bar(e.grid_position) == 3
            for e in result
        )
        has_beat4_snare = any(
            _is_snare(e) and _is_strong_beat(e.grid_position)
            and _beat_in_bar(e.grid_position) == 4
            for e in result
        )

        bar_index = events[0].bar_index if events else 0

        if not has_beat1_kick:
            result.append(GrooveEvent(
                instrument="kick", grid_position=0, bar_index=bar_index,
                velocity=cfg.anchor_target_velocity, articulation="default",
                source_role="main",
            ))

        if not has_beat2_snare:
            result.append(GrooveEvent(
                instrument="snare", grid_position=4, bar_index=bar_index,
                velocity=cfg.anchor_target_velocity, articulation="default",
                source_role="main",
            ))

        if not has_beat3_kick:
            result.append(GrooveEvent(
                instrument="kick", grid_position=8, bar_index=bar_index,
                velocity=cfg.anchor_target_velocity, articulation="default",
                source_role="main",
            ))

        if not has_beat4_snare:
            result.append(GrooveEvent(
                instrument="snare", grid_position=12, bar_index=bar_index,
                velocity=cfg.anchor_target_velocity, articulation="default",
                source_role="main",
            ))

        has_hats = any(_is_hat(e) for e in result)
        if not has_hats:
            for pos in (0, 4, 8, 12):
                result.append(GrooveEvent(
                    instrument="hi_hat", grid_position=pos, bar_index=bar_index,
                    velocity=cfg.anchor_target_velocity, articulation="default",
                    source_role="main",
                ))

        return result

    # ------------------------------------------------------------------
    # BUILD — increase energy in a controlled way
    # ------------------------------------------------------------------

    def _shape_build(self, events: list[GrooveEvent]) -> list[GrooveEvent]:
        """BUILD boosts velocity and optionally opens hats.

        Velocity boost is scaled by ``humanize_amount``.  At 0.0,
        BUILD adds no velocity (pure machine grid).
        """
        cfg = self.config
        amount = cfg.humanize_amount
        result: list[GrooveEvent] = []

        boost = int(cfg.build_velocity_boost * amount)

        for evt in events:
            if boost > 0:
                new_vel = min(cfg.build_max_velocity, evt.velocity + boost)
                evt = evt.copy_with(velocity=new_vel)

            # Open hats if configured and amount > 0
            if cfg.build_open_hats and amount > 0.0 and _is_hat(evt):
                if evt.articulation in ("default", "closed"):
                    evt = evt.copy_with(articulation="open")

            result.append(evt)

        return result

    # ------------------------------------------------------------------
    # ENTER_SOFT / ENTER_FULL — controlled entry
    # ------------------------------------------------------------------

    def _shape_enter(self, events: list[GrooveEvent]) -> list[GrooveEvent]:
        """ENTER softens velocities for a controlled entry.

        Velocity scaling is interpolated between 1.0 (raw) at
        ``humanize_amount == 0.0`` and ``enter_soft_scale`` at 1.0.
        """
        cfg = self.config
        amount = cfg.humanize_amount
        result: list[GrooveEvent] = []

        # Interpolate between 1.0 (raw) and enter_soft_scale
        effective_scale = 1.0 - (1.0 - cfg.enter_soft_scale) * amount

        for evt in events:
            new_vel = int(evt.velocity * effective_scale)
            new_vel = max(1, min(cfg.enter_velocity_cap, new_vel))
            evt = evt.copy_with(velocity=new_vel)
            result.append(evt)

        return result

    # ------------------------------------------------------------------
    # BAIL — song is over, output 0 events
    # ------------------------------------------------------------------

    def _shape_bail(self, events: list[GrooveEvent]) -> list[GrooveEvent]:
        """BAIL returns an empty output list — the song is over."""
        return []
