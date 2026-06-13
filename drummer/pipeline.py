"""Drummer Brain Pipeline — deterministic orchestrator.

Connects FeatureMonitor → FeatureDrivenBehaviourEngine → groove source
→ BehaviourOutputShaper in one clean, testable pipeline.

This is NOT live audio, NOT live MIDI, NOT real-time scheduling.
It is a simulated brain pipeline for tests and demos.

Design contract
---------------
* Pure and deterministic: same inputs → same outputs.
* No wall-clock dependency except through passed-in ``now``.
* No audio I/O, no MIDI output side effects.
* All internal components are injectable for testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from perception.models import MusicalEvent
from perception.features import FeatureMonitor, FeatureSnapshot
from drummer.behaviour import BehaviourIntent, FeatureDrivenBehaviourEngine
from drummer.output_shaping import BehaviourOutputShaper
from drummer.feel import GrooveEvent


# ---------------------------------------------------------------------------
# Default groove — a simple one-bar rock pattern for testing
# ---------------------------------------------------------------------------


def _default_groove() -> list[GrooveEvent]:
    """Return a simple one-bar rock groove.

    * kick on beats 1 and 3 (grid 0, 8)
    * snare on beats 2 and 4 (grid 4, 12)
    * hi-hat on 8th notes (grid 0, 2, 4, 6, 8, 10, 12, 14)
    """
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


# ---------------------------------------------------------------------------
# PipelineDecision — one frame of pipeline output
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineDecision:
    """A single frame of output from the Drummer Brain Pipeline.

    Captures everything needed to understand what the drummer thought
    and what it decided to play.
    """

    timestamp: float
    """The time (seconds) this decision was evaluated for."""

    feature_snapshot: FeatureSnapshot
    """Current feature summary from the Feature Monitor."""

    behaviour_intent: BehaviourIntent
    """The behaviour decision for this moment."""

    raw_events: list[GrooveEvent] = field(default_factory=list)
    """The groove events before shaping (the pattern the drummer would play)."""

    shaped_events: list[GrooveEvent] = field(default_factory=list)
    """The groove events after shaping (what actually gets played)."""

    phase_alignment: Optional[float] = None
    """Phase alignment value, if provided."""


# ---------------------------------------------------------------------------
# DrummerBrainPipeline
# ---------------------------------------------------------------------------


class DrummerBrainPipeline:
    """Orchestrates the full perception-to-output decision chain.

    Parameters
    ----------
    feature_monitor : FeatureMonitor | None
        The feature monitor to use.  If None, a default is created.
    behaviour_engine : FeatureDrivenBehaviourEngine | None
        The behaviour engine to use.  If None, a default is created.
    output_shaper : BehaviourOutputShaper | None
        The output shaper to use.  If None, a default is created.
    groove_provider : callable | None
        A callable ``() -> list[GrooveEvent]`` that returns the raw
        groove pattern the drummer would play.  If None, a simple
        default rock groove is used.
    """

    def __init__(
        self,
        feature_monitor: FeatureMonitor | None = None,
        behaviour_engine: FeatureDrivenBehaviourEngine | None = None,
        output_shaper: BehaviourOutputShaper | None = None,
        groove_provider: callable | None = None,
    ) -> None:
        self._monitor = feature_monitor if feature_monitor is not None else FeatureMonitor()
        self._engine = behaviour_engine if behaviour_engine is not None else FeatureDrivenBehaviourEngine()
        self._shaper = output_shaper if output_shaper is not None else BehaviourOutputShaper()
        self._groove_provider = groove_provider if groove_provider is not None else _default_groove

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def feed_event(self, event: MusicalEvent) -> FeatureSnapshot:
        """Feed a single MusicalEvent into the Feature Monitor.

        This updates the monitor's internal state but does NOT return
        a pipeline decision — use ``process()`` for that.

        Parameters
        ----------
        event : MusicalEvent
            The detected musical event.

        Returns
        -------
        FeatureSnapshot
            The updated feature snapshot after ingesting this event.
        """
        return self._monitor.feed(event)

    def process(
        self,
        now: float,
        phase_alignment: float | None = None,
    ) -> PipelineDecision:
        """Process the current pipeline state and return a decision.

        Steps:
            1. Take a snapshot from the Feature Monitor.
            2. If phase_alignment is provided, inject it into the snapshot.
            3. Evaluate the behaviour engine.
            4. Get the raw groove from the groove provider.
            5. Shape the groove via the Output Shaper.
            6. Return a PipelineDecision.

        Parameters
        ----------
        now : float
            Current time in seconds.
        phase_alignment : float | None
            Optional phase alignment from pulse/bar tracker.

        Returns
        -------
        PipelineDecision
            The complete pipeline frame.
        """
        # Step 1 — Feature snapshot
        snapshot = self._monitor.snapshot(now, phase_alignment=phase_alignment)

        # Step 2 — Behaviour decision
        behaviour_decision = self._engine.evaluate(snapshot)
        intent = behaviour_decision.intent

        # Step 3 — Get raw groove
        # For LISTEN and BAIL → no groove
        if intent in (BehaviourIntent.LISTEN, BehaviourIntent.BAIL):
            raw_events: list[GrooveEvent] = []
        else:
            raw_events = list(self._groove_provider())

        # Step 4 — Shape the groove
        shaped_events = self._shaper.shape(raw_events, intent)

        return PipelineDecision(
            timestamp=now,
            feature_snapshot=snapshot,
            behaviour_intent=intent,
            raw_events=raw_events,
            shaped_events=shaped_events,
            phase_alignment=phase_alignment,
        )

    def reset(self) -> None:
        """Reset all internal state back to factory-fresh."""
        self._monitor.reset()
        self._engine.reset()

    # ------------------------------------------------------------------
    # Accessors for testing
    # ------------------------------------------------------------------

    @property
    def monitor(self) -> FeatureMonitor:
        """The internal Feature Monitor."""
        return self._monitor

    @property
    def engine(self) -> FeatureDrivenBehaviourEngine:
        """The internal Behaviour Engine."""
        return self._engine

    @property
    def shaper(self) -> BehaviourOutputShaper:
        """The internal Output Shaper."""
        return self._shaper