# ROADMAP — Pocket Drummer

## Completed Milestones

### Module 1 — Event Listener ✅
**File:** `perception/event_listener.py`
**Tests:** `tests/test_perception.py` (48 tests)

Transforms raw audio into structured `MusicalEvent` objects. Detects attacks, classifies frequency regions (sub/low/low_mid/high_mid/high), computes local RMS energy, and tracks attack density. Supports both offline (full buffer) and streaming (frame-by-frame) modes.

**Success condition met:** A bass note, guitar hit, mute, accent, or strum consistently creates meaningful event data.

### Module 2 — Pulse Tracker ✅
**File:** `perception/pulse.py`
**Tests:** `tests/test_pulse.py` (34 tests)

Maintains competing tempo/pulse hypotheses from a stream of events. Does NOT output a single locked BPM. Maintains multiple interpretations simultaneously (e.g. 120 BPM, 60 BPM, 240 BPM) with confidence levels. Handles human timing variation, half-time/double-time ambiguity, accent influence, and silence decay.

**Success condition met:** The software can gradually converge on a pulse without requiring manual tempo input.

### Module 3 — Bar / Downbeat Tracker ✅
**File:** `perception/bar.py`
**Tests:** `tests/test_bar.py` (22 tests)

Estimates likely bar position and downbeat location from events + pulse state. Maintains competing bar-phase hypotheses with confidence. Uses event strength/energy to weight downbeat candidates. Preserves uncertainty when the downbeat is ambiguous. Tolerates human timing jitter.

**Success condition met:** The software can locate likely bar boundaries.

### Module 4 — Groove Intent Engine ✅
**File:** `drummer/intent.py`
**Tests:** `tests/test_groove_intent.py` (25 tests)

Converts perception state (pulse, bar, events) into high-level drummer behaviour decisions. Implements WAIT, ENTER, HOLD, BUILD, REDUCE, SIMPLIFY, PREPARE_FILL, MARK_DOWNBEAT, and RESET actions. Produces `GrooveIntent` objects with suggested complexity, velocity, and human-readable reasons. Does NOT generate MIDI or drum patterns.

**Success condition met:** The drummer knows what kind of behaviour is appropriate.

---

## Current Work

### Behaviour Engine Refinement
The Groove Intent Engine (`drummer/intent.py`) is functional but conservative. Thresholds and decision logic will be refined as the perception pipeline stabilises with real audio input.

---

## Upcoming Milestones

### Module 5 — Groove Selection
Convert `GrooveIntent` into a specific `Groove` selection from the groove library (`data/grooves.yaml`). The system should choose appropriate patterns based on:
- Energy level and density
- Pulse tempo
- Bar position
- Suggested complexity and velocity from intent engine
- Current drummer model (e.g. motorik_tight, sparse_postpunk)

No new MIDI generation — just pattern selection decisions.

### Module 6 — Fill Generation
Translate `PREPARE_FILL` intent into actual fill patterns. Fills should:
- Respect bar boundaries
- Vary with energy and density
- Not repeat identically
- Lead back into the next bar cleanly
- Scale with complexity level

### Module 7 — Dynamics Engine
Convert `GrooveIntent.suggested_velocity` and energy trends into velocity shaping:
- Build intensity gradually during BUILD phases
- Drop velocity during REDUCE/SIMPLIFY
- Accent downbeats during MARK_DOWNBEAT
- Shape backbeat (beat 3) velocity based on feel profile

### Module 8 — Humanisation
Already partially implemented in `drummer/humanize.py` and `drummer/feel.py`. Future work:
- Connect humanisation to GrooveIntent (more humanisation when confident, less when uncertain)
- Per-instrument timing offsets based on feel profile
- Dynamic ghost note probability tied to energy/density

### Module 9 — Entry Logic
Refine how the drummer enters the music:
- Wait for sufficient pulse and bar confidence
- Enter on a downbeat (not mid-phrase)
- Start with simplified pattern, grow into full groove
- Match entry velocity to current energy level

### Module 10 — MIDI Performance Generation
Combine all modules into a real-time MIDI output pipeline:
- Event detection → Pulse tracking → Bar tracking → Groove intent → Groove selection → Feel engine → Humanisation → MIDI export

### Module 11 — Live Audio Input
Replace synthetic event generation with real audio input via `sounddevice`:
- Microphone capture
- Onset detection on live stream
- Feed real events through the perception pipeline

---

## Long-Term Vision

The completed system will:
1. Listen to live audio (any instrument, any style)
2. Detect musical events
3. Estimate pulse with competing hypotheses
4. Track bar position and downbeats
5. Decide appropriate drummer behaviour
6. Select grooves that fit the music
7. Generate fills at phrase boundaries
8. Shape dynamics to match musical energy
9. Humanise timing and velocity
10. Output MIDI to any drum VST

The goal is not to be the flashiest drummer.

The goal is to be the drummer you want in the room.