# PROJECT CONTEXT — Bunny Deluxe

## What is Bunny Deluxe?

Bunny Deluxe is a **behavioural drummer**, not a transcription system.

It listens to musical input (audio events), builds an understanding of what is happening musically, and produces MIDI drum performances that feel like another musician in the room.

## What problem is it solving?

Most drum software asks "what notes are being played?" and sequences patterns.

Bunny Deluxe asks **"what is happening?"** and reacts like a human drummer:

- It senses pulse, not just BPM
- It tracks energy, not just velocity
- It feels bar boundaries, not just time grids
- It builds intensity, makes space, and prepares fills contextually

A human drummer cannot see the performer. They listen, feel the music, and respond. Bunny Deluxe does the same.

## Core Philosophy

1. **Behaviour over AI** — Bunny Deluxe simulates drumming behaviour through explicit rules, state machines, and statistical reasoning. It does not rely on machine learning or deep learning.

2. **Listening first, playing second** — The system does not immediately produce output. It listens, builds confidence, and only enters when it understands the musical situation.

3. **Stability over reactivity** — The system should not jump at every change. Confidence builds gradually. Uncertainty is preserved rather than prematurely resolved.

4. **Explainable decisions** — Every musical decision should have a human-readable reason. The system tracks why it chose to enter, build, reduce, or fill.

5. **MIDI only** — Bunny Deluxe outputs MIDI, not audio. It connects to existing drum instruments (EZDrummer, Superior Drummer, Logic Drummer, VSTs). No audio synthesis.

6. **DSP and statistics first** — Signal processing, state machines, and statistical reasoning are preferred. Machine learning is not used in the core loop.

## What is intentionally out of scope?

- Audio synthesis
- Note recognition / pitch detection
- Chord recognition
- Genre classification
- Transcription
- Large language model integration
- Deep learning in the core loop
- GUI as primary interface (CLI/demos are the development interface)

## The Question

The success criterion for every feature is:

> **"Does this make Bunny Deluxe feel more like another drummer in the room?"**

If the answer is no, the feature should be questioned before being added.
