"""Real-time audio ingress helpers for Bunny's single clock domain."""

from __future__ import annotations

import queue
from dataclasses import dataclass
from typing import Any

import numpy as np

from drummer.live_models import MonotonicClock


def _time_field(time_info: object, name: str) -> float:
    if isinstance(time_info, dict):
        return float(time_info[name])
    return float(getattr(time_info, name))


class PortAudioClockBridge:
    """Translate PortAudio stream timestamps into the monotonic clock domain.

    The offset is captured once per stream so callback scheduling jitter cannot
    move the musical grid after it has locked.
    """

    def __init__(self, clock: MonotonicClock) -> None:
        self._clock = clock
        self._offset: float | None = None

    def frame_end(
        self,
        time_info: object,
        frames: int,
        sample_rate: float,
    ) -> float:
        now = self._clock()
        try:
            current_time = _time_field(time_info, "currentTime")
            input_start = _time_field(time_info, "inputBufferAdcTime")
        except (AttributeError, KeyError, TypeError, ValueError):
            return now
        if self._offset is None:
            self._offset = now - current_time
        return self._offset + input_start + frames / sample_rate

    def reset(self) -> None:
        self._offset = None


@dataclass(frozen=True)
class QueuedAudioBlock:
    samples: np.ndarray
    frame_end: float
    status: str


@dataclass
class AudioIngressDiagnostics:
    callbacks: int = 0
    queued_blocks: int = 0
    dropped_blocks: int = 0
    status_events: int = 0


class LiveAudioIngress:
    """A non-blocking sounddevice callback that only copies audio to a queue."""

    def __init__(
        self,
        sample_rate: int,
        channel_index: int,
        clock: MonotonicClock,
        *,
        max_blocks: int = 256,
    ) -> None:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if channel_index < 0:
            raise ValueError("channel_index must be non-negative")
        if max_blocks < 1:
            raise ValueError("max_blocks must be at least 1")
        self.sample_rate = sample_rate
        self.channel_index = channel_index
        self._bridge = PortAudioClockBridge(clock)
        self._queue: queue.Queue[QueuedAudioBlock] = queue.Queue(max_blocks)
        self.diag = AudioIngressDiagnostics()

    def callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: object,
        status: Any,
    ) -> None:
        """Copy one selected channel; never run trackers or MIDI here."""
        self.diag.callbacks += 1
        status_text = str(status) if status else ""
        if status_text:
            self.diag.status_events += 1
        data = np.asarray(indata)
        if data.ndim == 1:
            if self.channel_index != 0:
                self.diag.dropped_blocks += 1
                return
            selected = data
        else:
            if self.channel_index >= data.shape[1]:
                self.diag.dropped_blocks += 1
                return
            selected = data[:, self.channel_index]
        block = QueuedAudioBlock(
            samples=np.asarray(selected, dtype=np.float32).copy(),
            frame_end=self._bridge.frame_end(time_info, frames, self.sample_rate),
            status=status_text,
        )
        try:
            self._queue.put_nowait(block)
            self.diag.queued_blocks += 1
        except queue.Full:
            self.diag.dropped_blocks += 1

    def drain(self, limit: int | None = None) -> list[QueuedAudioBlock]:
        blocks: list[QueuedAudioBlock] = []
        while limit is None or len(blocks) < limit:
            try:
                blocks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return blocks

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()
