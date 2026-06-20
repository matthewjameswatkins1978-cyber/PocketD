# Bunny Deluxe

A Python prototype for a behavioural MIDI drummer. Bunny Deluxe listens to rhythmic input, builds confidence about pulse and bar position, chooses drummer behaviour, shapes feel and dynamics, and outputs MIDI to an external drum instrument such as EZDrummer.

**Philosophy:** listen -> understand -> groove.

## Current Capabilities

- Detects musical events from audio or synthetic input.
- Tracks pulse using competing tempo hypotheses with confidence.
- Estimates bar position and downbeat confidence.
- Converts perception state into drummer behaviour intents such as wait, enter, hold, build, reduce, simplify, prepare fill, and reset.
- Runs synthetic jam scenarios for repeatable development and testing.
- Selects and renders MIDI grooves with presets, feel shaping, humanisation, and output shaping.
- Exports MIDI files or sends live MIDI to a virtual port / drum instrument.
- Prints or exports engine decision traces so behaviour can be debugged bar by bar.

## Simple Brain v0

A lightweight, explainable beat-selection engine (`drummer/simple_brain.py`) that follows a **Lock → Choose → Hold → Relisten** cycle:

1. **Lock** — listen until the player is confident enough (4+ snapshots above threshold)
2. **Choose** — score all beats from `data/grooves.yaml` against current density, stability, and confidence, then pick the best match
3. **Hold** — keep the chosen beat unless a major, high-confidence musical change occurs
4. **Relisten** — if confidence collapses, dump state and return to listening

### Beat knowledge layer

Simple Brain chooses from real beat definitions in `data/grooves.yaml`.
Every non-silence beat it can choose is a real groove ID.  `silence` is
the only special sentinel — it has no corresponding groove data.

To add a new beat that Simple Brain can select:
1. Add the groove pattern to `data/grooves.yaml`
2. Set `simple_brain_enabled: true` and include the required metadata fields (`ideal_density`, `min_stability`, `description`, `feel_tags`)

The existing complex behaviour engine (`drummer/behaviour.py`) is preserved untouched as legacy/research.

**Enabled grooves:** `simple_rock`, `motorik`, `half_time`, `shuffle`, `funk_pocket`, `punk_drive`

Run the diagnostic demo:

```powershell
python demo_simple_brain.py
```

Run the named trace scenario demo (inspired by existing Bunny Deluxe playtest forms):
```powershell
python demo_simple_brain_scenarios.py
```

Run the shadow-mode demo (Simple Brain over real FeatureMonitor output):
```powershell
python demo_simple_brain_shadow.py
```

Run the full Simple Brain test suite (unit + batch + trace + database):
```powershell
python -m pytest tests/test_simple_brain.py tests/test_simple_brain_batch.py tests/test_simple_brain_trace.py tests/test_simple_brain_shadow.py tests/test_simple_brain_groove_database.py -v
```

## Project Docs

This README is the quick-start and current-state overview. The deeper project story lives in:

| File | Purpose |
|------|---------|
| `PROJECT_CONTEXT.md` | What Bunny Deluxe is, what problem it solves, and what is intentionally out of scope |
| `ARCHITECTURE.md` | Pipeline design, module responsibilities, and data flow |
| `ROADMAP.md` | Completed milestones, current work, and upcoming modules |
| `DECISIONS.md` | Architectural decision records and rejected alternatives |

## Setup

```powershell
cd C:\Users\Matmus\pocket_drummer
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

On **Python 3.14**, `python-rtmidi` has no prebuilt wheel yet. The app automatically uses a **Windows winmm fallback** — no compiler needed. loopMIDI ports show up as normal MIDI output devices.

### loopMIDI + EZDrummer

1. Create a virtual port in **loopMIDI** (e.g. `PocketDrummer Out`).
2. In **EZDrummer**, set MIDI input to that port.
3. List ports:

```powershell
python main.py --list-ports
```

## Bunny V1 live pocket

Bunny V1 is the conservative live vertical slice: listen for an unambiguous
pulse and bar phase, arm for a future downbeat, play a locked straight pocket,
degrade safely when evidence weakens, and optionally mirror one repeated kick
slot. It is separate from Simple Brain and the larger behaviour engine.

The hardware runner is explicitly opt-in. With no arguments it opens nothing:

```powershell
.\.venv\Scripts\python.exe demo_live_clap_to_loopmidi.py
```

Check routing first:

```powershell
.\.venv\Scripts\python.exe demo_live_clap_to_loopmidi.py --list-audio
.\.venv\Scripts\python.exe demo_live_clap_to_loopmidi.py --list-midi
```

Then run a bounded 30-second playtest:

```powershell
.\.venv\Scripts\python.exe demo_live_clap_to_loopmidi.py --run-live --device-name "AG06/AG03" --channel 1 --port "PocketDrummer Out" --duration 30
```

Use `--no-midi` to exercise audio, perception, and control without emitting
notes. Every live run writes a JSON diagnostic trace under
`artifacts/bunny_live/`, including state transitions, detected events, locked
BPM/grid, scheduler jitter/drops, and audio callback overflow counts. Press
**Ctrl+C** for an explicit stop and MIDI close.

The values in `LiveConfig` remain initial tuning hypotheses. Adjust them only
from trace-backed playtest evidence; do not bury tuned thresholds in runtime
logic.

Run the complete hardware-free Bunny suite:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/test_live_*.py
```

## Basic MIDI Groove

Plays a 16-step groove at a fixed BPM. This is the simplest MIDI smoke test.

```powershell
python main.py --midi-port "PocketDrummer" --groove simple_rock --bpm 120
```

Options:

| Flag | Default | Description |
|------|---------|-------------|
| `--midi-port` | (required) | Substring match on output port name |
| `--groove` | `simple_rock` | `simple_rock`, `motorik`, or `half_time` |
| `--bpm` | `120` | Tempo |
| `--bars` | loop | Bars to play before stopping |
| `--complexity` | `5` | 1–5 complexity ladder |
| `--list-ports` | | Show MIDI outputs and exit |

Press **Ctrl+C** to stop.

## Project structure

```
pocket_drummer/
  main.py              # CLI entry point
  models.py            # Core dataclasses
  groove_library.py    # Load grooves.yaml
  midi_out.py          # MIDI output
  scheduler.py         # 16th-note clock
  data/grooves.yaml    # Groove definitions
  ...
```

## Continuous Jam Demo

Runs the behavioural drummer through a synthetic scenario without requiring live audio.

```powershell
python demo_continuous_jam_midi.py --scenario build --preset normal --bars 16
```

Use `--no-play` when you want to generate diagnostics without sending MIDI, or `--export-json` when you want to save the rendered schedule data.

## Engine Decision Trace

Diagnostic tool that shows why the drummer chose each behaviour intent, bar by bar.

```powershell
# Export per-bar trace as JSON
python demo_continuous_jam_midi.py --scenario drop --preset cautious --no-play --engine-trace artifacts/engine_trace_drop.json

# Print a compact trace table to the terminal
python demo_continuous_jam_midi.py --scenario drop --preset cautious --no-play --print-engine-trace

# Both export and print
python demo_continuous_jam_midi.py --scenario drop --preset cautious --no-play --engine-trace artifacts/engine_trace_drop.json --print-engine-trace
```

The trace table shows per-bar columns for selected/rendered intent, confidence, feature values (density, certainty, stability, change, silence), and the engine's reason. When the scenario overrides the engine's decision, the reason column shows `OVERRIDE`.

The trace data is diagnostic only. It does not change musical behaviour or MIDI output.

## MIDI notes (General MIDI drums)

| Instrument | Note |
|------------|------|
| Kick | 36 |
| Snare | 38 |
| Closed hat | 42 |
