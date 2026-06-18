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
