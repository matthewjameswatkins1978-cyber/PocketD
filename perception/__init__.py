"""Perception Engine — audio-to-understanding pipeline.

Modules
-------
Module 1 — Event Listener
    Transforms raw audio into structured MusicalEvent objects.

Module 2 — Pulse Tracker
    Maintains competing tempo/pulse hypotheses from a stream of events.

Module 3 — Bar / Downbeat Tracker
    Estimates likely bar position and downbeat location from events + pulse.

Module 5 — Feature Monitor
    Continuously summarises what the player appears to be doing musically
    using simple explainable features.  Produces FeatureSnapshot values
    for the behaviour engine.  Does NOT make drum decisions itself.

Long-term pipeline
------------------
Audio → Event Listener → Pulse Tracker → Bar Tracker → Feature Monitor
    → Groove Tracker → Energy Tracker → Section Memory → Behaviour Engine → Drummer

Public API
----------
- MusicalEvent              — dataclass for a single detected musical event
- FrequencyRegion           — Literal type for rough frequency bands
- PulseHypothesis           — a single competing BPM hypothesis
- PulseState                — current belief state of the pulse tracker
- PulseTracker              — maintains competing pulse hypotheses
- BarHypothesis             — a single competing bar-position hypothesis
- BarState                  — current belief state of the bar tracker
- BarTracker                — maintains competing bar-phase hypotheses
- FeatureSnapshot           — point-in-time summary of player behaviour
- FeatureMonitorConfig      — tuning knobs for the Feature Monitor
- FeatureMonitor            — observes events, produces FeatureSnapshots
"""

from __future__ import annotations

from perception.bar import BarHypothesis, BarState, BarTracker
from perception.density import AttackDensityTracker
from perception.detector import Detector, DetectorConfig, DetectorDiagnostics, DetectorResult
from perception.energy import compute_energy
from perception.event_listener import (
    AudioFrame,
    EventListener,
    detect_events_from_audio,
)
from perception.frequency import classify_frequency
from perception.features import FeatureMonitor, FeatureMonitorConfig, FeatureSnapshot
from perception.models import FrequencyRegion, MusicalEvent
from perception.pulse import PulseHypothesis, PulseState, PulseTracker

__all__ = [
    "AttackDensityTracker",
    "AudioFrame",
    "BarHypothesis",
    "BarState",
    "BarTracker",
    "Detector",
    "DetectorConfig",
    "DetectorDiagnostics",
    "DetectorResult",
    "EventListener",
    "FeatureMonitor",
    "FeatureMonitorConfig",
    "FeatureSnapshot",
    "FrequencyRegion",
    "MusicalEvent",
    "PulseHypothesis",
    "PulseState",
    "PulseTracker",
    "classify_frequency",
    "compute_energy",
    "detect_events_from_audio",
]
