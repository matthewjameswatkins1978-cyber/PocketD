"""Drummer package — perception-to-behaviour, humanization, and feel."""

from __future__ import annotations

from drummer.behaviour import (
    BehaviourDecision,
    BehaviourEngine,
    BehaviourIntent,
    ConservativePocketDrummer,
    FeatureDrivenBehaviourEngine,
)
from drummer.feel import (
    DrummerFeelEngine,
    DrummerProfile,
    GrooveEvent,
    TimingStrategy,
)
from drummer.golden_reference import (
    GoldenReference,
    load_golden_references,
    save_golden_diagnostics,
    save_golden_reference,
)
from drummer.humanize import humanize_events
from drummer.musical_doctor import (
    DoctorProblem,
    DoctorReport,
    diagnose_bar_transcript,
)
from drummer.intent import (
    GrooveAction,
    GrooveIntent,
    GrooveIntentEngine,
)

__all__ = [
    "BehaviourDecision",
    "BehaviourEngine",
    "BehaviourIntent",
    "ConservativePocketDrummer",
    "DoctorProblem",
    "DoctorReport",
    "DrummerFeelEngine",
    "DrummerProfile",
    "FeatureDrivenBehaviourEngine",
    "GoldenReference",
    "GrooveAction",
    "GrooveEvent",
    "GrooveIntent",
    "GrooveIntentEngine",
    "TimingStrategy",
    "diagnose_bar_transcript",
    "humanize_events",
    "load_golden_references",
    "save_golden_diagnostics",
    "save_golden_reference",
]
