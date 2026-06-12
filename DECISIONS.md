# DECISIONS — Pocket Drummer

## Architectural Decision Record

This document records important architectural decisions and why they were made, including rejected alternatives. It exists so future development understands the reasoning behind the current design.

---

## Decision 1: Competing hypotheses over locked answers

**Date:** Module 2 (Pulse Tracker)
**Status:** Accepted

**Decision:** Pulse and bar trackers maintain multiple competing hypotheses with confidence levels rather than outputting a single locked answer.

**Why:** A human drummer does not instantly know the exact BPM. They form multiple possible interpretations, and confidence grows as evidence accumulates. Premature locking leads to incorrect tempo judgments that are hard to recover from. The system preserves uncertainty and allows confidence to build naturally.

**Rejected alternative:** A traditional BPM detector that returns a single answer after averaging intervals. This would produce wrong answers on sparse or ambiguous input, and would have no mechanism for expressing uncertainty or recovering from errors.

---

## Decision 2: Perception-performer separation

**Date:** Project inception
**Status:** Accepted

**Decision:** The drummer (performance layer) never receives raw audio. It only receives interpreted musical state — `MusicalEvent`, `PulseState`, `BarState`, `GrooveIntent`.

**Why:** This enforces the design philosophy that Pocket Drummer is a listener first. Raw audio contains noise, transients, and artefacts that should not directly influence drumming decisions. The perception layer filters and interprets, and the performance layer acts on understanding.

**Architecture:** `Audio → Understanding → Drumming`, not `Audio → Drumming`.

**Rejected alternative:** Feeding onset times and amplitudes directly into a pattern generator. This couples drum generation to audio artefacts and makes it impossible to express "I'm not confident enough to play yet."

---

## Decision 3: Human-readable reasons on every decision

**Date:** Module 4 (Groove Intent Engine)
**Status:** Accepted

**Decision:** Every `GrooveIntent` carries a `reason` string that explains in human-readable terms why the decision was made. Examples: `"waiting: pulse confidence too low"`, `"build: energy rising, density rising"`.

**Why:** Debugging a behaviour system is hard. Explainable decisions make it possible to understand why the drummer entered, waited, built, reduced, or prepared a fill. This is essential for development, testing, and demo output.

**Rejected alternative:** Silent decisions with no explanation. Would make the system a black box impossible to debug or tune.

---

## Decision 4: Events, not notes

**Date:** Module 1 (Event Listener)
**Status:** Accepted

**Decision:** The system detects musical events (attacks with strength, energy, density, frequency region) rather than identifying specific notes, chords, or pitches.

**Why:** A human drummer reacts to timing, accents, energy, repetition, and density — not to specific pitches. The system should extract musical meaning from audio, not musical notation. This also avoids the complexity and fragility of pitch detection.

**Rejected alternative:** Note-level transcription with pitch and harmony analysis. This would be fragile, computationally expensive, and would encourage pattern-matching on note sequences rather than behavioural response to musical energy.

---

## Decision 5: MIDI only, no audio synthesis

**Date:** Project inception
**Status:** Accepted

**Decision:** Pocket Drummer outputs MIDI, not audio. It connects to existing drum instruments (EZDrummer, Superior Drummer, Logic Drummer, VSTs).

**Why:** Drum synthesis is a solved problem. Excellent drum VSTs exist. Pocket Drummer's value is in the behavioural intelligence — the listening, pulse tracking, bar detection, and intent decisions. Generating audio would be redundant and would distract from the core mission.

**Rejected alternative:** Built-in drum sample playback or synthesis. Would require maintaining sample libraries, dealing with audio latency, and competing with dedicated drum instruments that already do this better.

---

## Decision 6: No machine learning in the core loop

**Date:** Project inception
**Status:** Accepted

**Decision:** Avoid deep learning, neural networks, and large models in the core decision-making pipeline. Use DSP, statistics, explicit rules, and state machines.

**Why:** 
1. **Explainability** — Every decision should be traceable to explicit rules and thresholds.
2. **Predictability** — Behaviour should be consistent and tunable, not a black box.
3. **Latency** — DSP and statistics are faster than model inference.
4. **Dependency** — No GPU requirement, no large model downloads, no framework version issues.
5. **Philosophy** — Pocket Drummer models drumming behaviour, not statistical patterns in training data.

**Rejected alternative:** Using RNNs, transformers, or reinforcement learning to generate drum patterns. This would make the system opaque, unpredictable, and dependent on training data quality.

---

## Decision 7: Python as implementation language

**Date:** Project inception
**Status:** Accepted

**Decision:** Implement the core system in Python with NumPy for DSP.

**Why:** Python enables rapid prototyping of the behavioural rules and state machines that form the core of Pocket Drummer. NumPy provides efficient FFT and array operations. The MIDI libraries (mido, python-rtmidi) have good Python support. Python's readability aligns with the goal of explainable decisions.

**Rejected alternative:** C++ or Rust for lower latency. The trade-off in development speed is not worth it at this stage. If latency becomes a bottleneck in live performance, critical paths can be extracted later.

---

## Decision 8: YAML for groove and configuration data

**Date:** Early development
**Status:** Accepted

**Decision:** Store groove patterns, fill patterns, and feel profiles in YAML files under `data/`.

**Why:** YAML is human-readable and editable without special tools. Groove patterns can be added, modified, and shared easily. The data format is self-documenting.

**Rejected alternative:** Hard-coded patterns in Python or binary formats. Makes the system inflexible and prevents users from creating custom grooves.

---

## Decision 9: Synthetic event generation for development

**Date:** Module 1 (Event Listener)
**Status:** Accepted

**Decision:** Develop and test the perception pipeline using synthetic audio and synthetic events before connecting real audio input.

**Why:** Synthetic input provides controlled, repeatable test conditions. Pulse tracking, bar tracking, and intent decisions can be verified against known inputs before dealing with the variability of real audio. This accelerates development and improves test reliability.

**Rejected alternative:** Developing directly against microphone input. Would make tests flaky, slow down iteration, and make it hard to distinguish bugs in perception logic from issues with audio capture.

---

## Decision 10: Threshold-based intent decisions

**Date:** Module 4 (Groove Intent Engine)
**Status:** Accepted

**Decision:** Groove intent decisions use explicit thresholds (`MIN_PULSE_CONFIDENCE_TO_PLAY`, `HIGH_CONFIDENCE`, etc.) and trend detection rather than learned or fuzzy logic.

**Why:** Thresholds are explicit, tunable, and explainable. They can be adjusted based on testing and user feedback. The system can explain exactly why it chose WAIT instead of ENTER ("pulse confidence 0.35 < threshold 0.40").

**Rejected alternative:** Fuzzy logic or weighted scoring. While smoother, this would make it harder to explain specific decisions and harder to tune behaviour predictably.

---

## Decision 11: Python 3.14 support with winmm fallback

**Date:** Development setup
**Status:** Accepted

**Decision:** Support Python 3.14 using a Windows Multimedia (winmm) MIDI fallback when python-rtmidi is unavailable.

**Why:** Python 3.14 is the current version and python-rtmidi may not have prebuilt wheels. The winmm fallback (`_winmm.py`) provides MIDI output without requiring a C++ compiler. This keeps the project accessible and installable on Windows without extra tooling.

**Rejected alternative:** Requiring python-rtmidi and a C++ compiler. Would create unnecessary friction for setup and development.