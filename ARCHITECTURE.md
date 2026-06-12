# ARCHITECTURE — Pocket Drummer

## Overview

Pocket Drummer follows a **layered perception-to-performance pipeline**:

```
Audio Input
    │
    ▼
┌─────────────────────────────────────────┐
│            PERCEPTION LAYER             │
│                                         │
│  Event Listener  ──►  MusicalEvent      │
│  Pulse Tracker   ──►  PulseState        │
│  Bar Tracker     ──►  BarState          │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│           BEHAVIOUR LAYER               │
│                                         │
│  Groove Intent   ──►  GrooveIntent      │
│  Engine                                  │
└─────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────┐
│          PERFORMANCE LAYER              │
│                                         │
│  Groove Selection  (upcoming)           │
│  Fill Generation   (upcoming)           │
│  Feel Engine       ──►  GrooveEvent[]   │
│  Humanisation      ──►  GrooveEvent[]   │
│  MIDI Export       ──►  .mid / live     │
└─────────────────────────────────────────┘
```

## Data Flow

```
Audio Samples
  → MusicalEvent (time, strength, energy, density, frequency_region)
    → PulseState (competing BPM hypotheses with confidence)
      → BarState (competing bar-phase hypotheses with confidence)
        → GrooveIntent (action, complexity, velocity, should_play, should_fill)
          → GrooveSelection (upcoming — specific pattern choice)
            → GrooveEvent[] (timed MIDI-triggerable events with feel)
              → MIDI bytes → VST/Drum Instrument
```

Each stage only sees the output of the previous stage. No stage reaches around to raw audio. The drummer (performance layer) never sees audio directly — only interpreted musical state.

## Module Responsibilities

### `perception/` — The Nervous System

| Module | File | Responsibility |
|--------|------|---------------|
| Event Listener | `perception/event_listener.py` | Detect musical attacks, classify frequency region, compute energy/density |
| Pulse Tracker | `perception/pulse.py` | Maintain competing BPM hypotheses with confidence and stability |
| Bar Tracker | `perception/bar.py` | Estimate downbeat position and beat-in-bar from events + pulse |
| Models | `perception/models.py` | `MusicalEvent` dataclass, `FrequencyRegion` type, band definitions |
| Frequency | `perception/frequency.py` | FFT analysis, band energy, frequency region classification |
| Energy | `perception/energy.py` | RMS energy computation, windowed energy |
| Density | `perception/density.py` | Sliding-window attack density tracking |

### `drummer/` — The Body

| Module | File | Responsibility |
|--------|------|---------------|
| Groove Intent | `drummer/intent.py` | Convert perception state to drummer behaviour decisions |
| Feel Engine | `drummer/feel.py` | Apply timing offsets, velocity shaping, ghost notes per feel profile |
| Humanisation | `drummer/humanize.py` | Add micro-timing variation and velocity randomness |
| MIDI Export | `drummer/midi_export.py` | Write GrooveEvent[] to .mid file |
| Models | `drummer/models.py` | Drummer model dataclasses (confidence rules, transition rules, etc.) |
| Rules | `drummer/rules.py` | Rule-based drummer behaviour definitions |

### Supporting Modules

| Module | File | Responsibility |
|--------|------|---------------|
| Main | `main.py` | CLI entry point with milestone routing |
| Groove Library | `groove_library.py` | Load groove patterns from `data/grooves.yaml` |
| Scheduler | `scheduler.py` | 16th-note clock for hard-coded groove playback (legacy) |
| MIDI Out | `midi_out.py` | Windows MIDI output port management |
| Models | `models.py` | Legacy core dataclasses (AccentEvent, PulseState, Groove, etc.) |

## Key Design Decisions

1. **Events, not notes** — The system detects musical events (attacks with energy) rather than identifying specific notes or pitches.

2. **Competing hypotheses** — Pulse and bar trackers maintain multiple interpretations with confidence levels rather than locking to a single answer. This models human uncertainty.

3. **Confidence-first** — Every decision gates on confidence. The drummer waits until pulse and bar confidence are sufficient before entering.

4. **Separation of perception and performance** — The performer never sees raw audio. It only sees interpreted musical state (Intent objects). This prevents the drummer from overfitting to audio quirks.

5. **Human-readable reasons** — Every decision carries a `reason` string for debugging and demos.

6. **MIDI only** — Output is MIDI, not audio. The system connects to existing drum VSTs.

7. **No ML in the core loop** — Decisions use explicit rules, statistics, DSP, and state machines. No neural networks or large models.

## Test Strategy

- Each module has its own test file in `tests/`
- Tests use synthetic events with known properties
- 396 tests total across the project (all passing)
- Demo scripts (`demo_*.py`) provide human-verifiable output

## Package Structure

```
pocket_drummer/
├── perception/           # Perception engine (listening)
│   ├── __init__.py
│   ├── models.py         # MusicalEvent, FrequencyRegion
│   ├── event_listener.py # Module 1 - Event detection
│   ├── pulse.py          # Module 2 - Pulse tracking
│   ├── bar.py            # Module 3 - Bar/downbeat tracking
│   ├── frequency.py      # FFT and frequency analysis
│   ├── energy.py         # RMS energy computation
│   └── density.py        # Attack density tracking
├── drummer/              # Performance engine (playing)
│   ├── __init__.py
│   ├── intent.py         # Module 4 - Groove Intent Engine
│   ├── feel.py           # Feel engine (timing, velocity)
│   ├── humanize.py       # Micro-timing humanisation
│   ├── models.py         # Drummer model dataclasses
│   ├── rules.py          # Rule definitions
│   └── midi_export.py    # MIDI file export
├── tests/                # Test suite
├── data/                 # YAML data files (grooves, fills, feel profiles)
├── tools/                # Utility tools (MIDI inspection)
├── synthetic/            # Synthetic pulse generation for testing
├── gui/                  # Optional GUI components
├── demo_*.py             # Demonstration scripts
├── main.py               # CLI entry point
├── PROJECT_CONTEXT.md    # Project constitution
├── ROADMAP.md            # Development roadmap
├── ARCHITECTURE.md       # This file
└── DECISIONS.md          # Architectural decision record