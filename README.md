# Pocket Drummer

A Python prototype that listens to rhythmic input, detects pulse, chooses a simple drum groove, and outputs MIDI to an external drum instrument (e.g. EZDrummer).

**Philosophy:** listen → lock → groove.

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

## Milestone 1 — Hard-coded MIDI groove

Plays a 16-step groove at a fixed BPM. No mic, no pulse tracking yet.

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

## MIDI notes (General MIDI drums)

| Instrument | Note |
|------------|------|
| Kick | 36 |
| Snare | 38 |
| Closed hat | 42 |
