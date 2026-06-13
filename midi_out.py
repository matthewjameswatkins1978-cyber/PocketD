"""MIDI output — mido/python-rtmidi when available, winmm fallback on Windows."""

from __future__ import annotations

import logging
import sys

from models import CLOSED_HAT, KICK, SNARE

logger = logging.getLogger(__name__)

DEFAULT_VELOCITY = 100
DRUM_CHANNEL = 9  # GM drums (MIDI channel 10, zero-indexed 9)

_BACKEND: str | None = None
_mido = None


def _init_backend() -> str:
    global _BACKEND, _mido
    if _BACKEND is not None:
        return _BACKEND

    try:
        import mido  # noqa: WPS433

        mido.set_backend("mido.backends.rtmidi")
        mido.get_output_names()
        _mido = mido
        _BACKEND = "mido"
        logger.debug("MIDI backend: mido + python-rtmidi")
        return _BACKEND
    except Exception as exc:
        logger.debug("mido/rtmidi unavailable (%s), trying winmm", exc)

    if sys.platform == "win32":
        _BACKEND = "winmm"
        logger.debug("MIDI backend: Windows winmm")
        return _BACKEND

    raise RuntimeError(
        "No MIDI backend available. Install python-rtmidi (Python 3.11–3.13) "
        "or run on Windows for the built-in winmm fallback."
    )


def list_output_ports() -> list[str]:
    backend = _init_backend()
    if backend == "mido":
        assert _mido is not None
        return _mido.get_output_names()
    return _winmm_list_ports()


def find_output_port(name_substring: str) -> str:
    needle = name_substring.lower()
    for port in list_output_ports():
        if needle in port.lower():
            return port
    available = list_output_ports()
    raise ValueError(
        f"No MIDI output port matching '{name_substring}'. "
        f"Available: {available or '(none)'}"
    )


# --- winmm fallback (Windows, no python-rtmidi required) ---

def _winmm_list_ports() -> list[str]:
    from _winmm import list_midi_output_devices

    return list_midi_output_devices()


class _WinmmPort:
    def __init__(self, device_id: int, name: str) -> None:
        from _winmm import WinmmMidiOut

        self.name = name
        self._out = WinmmMidiOut(device_id)

    def send_note(self, note: int, velocity: int) -> None:
        self._out.note_on(note, velocity, channel=DRUM_CHANNEL)
        self._out.note_off(note, channel=DRUM_CHANNEL)

    def note_on(self, note: int, velocity: int) -> None:
        self._out.note_on(note, velocity, channel=DRUM_CHANNEL)

    def note_off(self, note: int) -> None:
        self._out.note_off(note, channel=DRUM_CHANNEL)

    def close(self) -> None:
        self._out.close()


class _MidoPort:
    def __init__(self, port_name: str) -> None:
        assert _mido is not None
        self.name = port_name
        self._port = _mido.open_output(port_name)

    def send_note(self, note: int, velocity: int) -> None:
        self._port.send(
            _mido.Message("note_on", note=note, velocity=velocity, channel=DRUM_CHANNEL)
        )
        self._port.send(
            _mido.Message("note_off", note=note, velocity=0, channel=DRUM_CHANNEL)
        )

    def note_on(self, note: int, velocity: int) -> None:
        self._port.send(
            _mido.Message("note_on", note=note, velocity=velocity, channel=DRUM_CHANNEL)
        )

    def note_off(self, note: int) -> None:
        self._port.send(
            _mido.Message("note_off", note=note, velocity=0, channel=DRUM_CHANNEL)
        )

    def close(self) -> None:
        self._port.close()


class MidiOut:
    """Send General MIDI drum notes to an external port."""

    def __init__(self, port_name: str) -> None:
        self.port_name = port_name
        self._port: _MidoPort | _WinmmPort | None = None

    def open(self) -> None:
        backend = _init_backend()
        if backend == "mido":
            self._port = _MidoPort(self.port_name)
        else:
            from _winmm import resolve_device_id

            device_id = resolve_device_id(self.port_name)
            self._port = _WinmmPort(device_id, self.port_name)
        logger.info("MIDI output opened: %s (backend=%s)", self.port_name, backend)

    def close(self) -> None:
        if self._port is not None:
            self._port.close()
            self._port = None
            logger.info("MIDI output closed")

    def __enter__(self) -> MidiOut:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def send_note(self, note: int, velocity: int = DEFAULT_VELOCITY) -> None:
        if self._port is None:
            raise RuntimeError("MIDI port not open")
        self._port.send_note(note, velocity)

    def note_on(self, note: int, velocity: int = DEFAULT_VELOCITY) -> None:
        """Send note_on only (no automatic note_off)."""
        if self._port is None:
            raise RuntimeError("MIDI port not open")
        self._port.note_on(note, velocity)

    def note_off(self, note: int) -> None:
        """Send note_off."""
        if self._port is None:
            raise RuntimeError("MIDI port not open")
        self._port.note_off(note)

    def send_kick(self, velocity: int = DEFAULT_VELOCITY) -> None:
        self.send_note(KICK, velocity)

    def send_snare(self, velocity: int = DEFAULT_VELOCITY) -> None:
        self.send_note(SNARE, velocity)

    def send_hat(self, velocity: int = 80) -> None:
        self.send_note(CLOSED_HAT, velocity)
