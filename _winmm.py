"""Windows Multimedia API MIDI output via ctypes (no python-rtmidi required)."""

from __future__ import annotations

import ctypes
from ctypes import wintypes

winmm = ctypes.WinDLL("winmm")

MIDI_MAPPER = -1
MIDI_OUT_CACHE_NOTUSED = 0
MIDIERR_BASE = 64
MMSYSERR_NOERROR = 0

MAXPNAMELEN = 32


class MIDIOUTCAPSA(ctypes.Structure):
    _fields_ = [
        ("wMid", wintypes.WORD),
        ("wPid", wintypes.WORD),
        ("vDriverVersion", wintypes.UINT),
        ("szPname", ctypes.c_char * MAXPNAMELEN),
        ("wTechnology", wintypes.WORD),
        ("wVoices", wintypes.WORD),
        ("wNotes", wintypes.WORD),
        ("wChannelMask", wintypes.WORD),
        ("dwSupport", wintypes.DWORD),
    ]


def _check(result: int, action: str) -> None:
    if result != MMSYSERR_NOERROR:
        raise OSError(f"winmm {action} failed with error code {result}")


def list_midi_output_devices() -> list[str]:
    count = winmm.midiOutGetNumDevs()
    names: list[str] = []
    caps = MIDIOUTCAPSA()
    for device_id in range(count):
        result = winmm.midiOutGetDevCapsA(device_id, ctypes.byref(caps), ctypes.sizeof(caps))
        if result == MMSYSERR_NOERROR:
            names.append(caps.szPname.decode("ascii", errors="replace"))
    return names


def resolve_device_id(port_name: str) -> int:
    """Match port name substring to device id, or use MIDI mapper."""
    needle = port_name.lower()
    count = winmm.midiOutGetNumDevs()
    caps = MIDIOUTCAPSA()
    for device_id in range(count):
        result = winmm.midiOutGetDevCapsA(device_id, ctypes.byref(caps), ctypes.sizeof(caps))
        if result == MMSYSERR_NOERROR:
            name = caps.szPname.decode("ascii", errors="replace")
            if needle in name.lower():
                return device_id
    # loopMIDI ports appear as named devices; if only one output, use it
    if count == 1:
        return 0
    available = list_midi_output_devices()
    raise ValueError(
        f"No MIDI output device matching '{port_name}'. Available: {available or '(none)'}"
    )


class WinmmMidiOut:
    def __init__(self, device_id: int) -> None:
        self._handle = wintypes.HANDLE()
        _check(
            winmm.midiOutOpen(
                ctypes.byref(self._handle),
                device_id,
                0,
                0,
                0,
            ),
            "midiOutOpen",
        )

    def _short_msg(self, status: int, data1: int, data2: int) -> None:
        msg = status | (data1 << 8) | (data2 << 16)
        _check(winmm.midiOutShortMsg(self._handle, msg), "midiOutShortMsg")

    def note_on(self, note: int, velocity: int, channel: int = 9) -> None:
        self._short_msg(0x90 | channel, note & 0x7F, velocity & 0x7F)

    def note_off(self, note: int, velocity: int = 0, channel: int = 9) -> None:
        self._short_msg(0x80 | channel, note & 0x7F, velocity & 0x7F)

    def close(self) -> None:
        if self._handle:
            winmm.midiOutClose(self._handle)
            self._handle = wintypes.HANDLE()
