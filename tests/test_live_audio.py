"""Tests for callback-safe live audio ingress and clock translation."""

from __future__ import annotations

import numpy as np
import pytest

from drummer.live_audio import LiveAudioIngress, PortAudioClockBridge
from tests.fake_clock import FakeClock


def test_portaudio_bridge_uses_one_stable_monotonic_offset() -> None:
    clock = FakeClock(1000.0)
    bridge = PortAudioClockBridge(clock.now)
    first = bridge.frame_end(
        {"currentTime": 20.0, "inputBufferAdcTime": 19.9}, 100, 1000
    )
    clock.advance(0.3)  # callback arrival jitter must not change the mapping
    second = bridge.frame_end(
        {"currentTime": 20.1, "inputBufferAdcTime": 20.0}, 100, 1000
    )

    assert first == pytest.approx(1000.0)
    assert second == pytest.approx(1000.1)


def test_bridge_falls_back_to_callback_clock_without_time_metadata() -> None:
    clock = FakeClock(50.0)
    bridge = PortAudioClockBridge(clock.now)
    assert bridge.frame_end({}, 128, 44100) == 50.0


def test_bridge_advances_by_samples_when_windows_driver_times_are_zero() -> None:
    clock = FakeClock(50.0)
    bridge = PortAudioClockBridge(clock.now)
    broken_mme_time = {
        "currentTime": 0.0,
        "inputBufferAdcTime": 0.0,
    }

    first = bridge.frame_end(broken_mme_time, 100, 1000)
    second = bridge.frame_end(broken_mme_time, 100, 1000)
    third = bridge.frame_end(
        {"currentTime": 0.0, "inputBufferAdcTime": 0.1}, 100, 1000
    )

    assert first == pytest.approx(50.0)
    assert second == pytest.approx(50.1)
    assert third == pytest.approx(50.2)


def test_audio_callback_copies_selected_channel_and_timestamp() -> None:
    clock = FakeClock(100.0)
    ingress = LiveAudioIngress(1000, 1, clock.now)
    audio = np.column_stack((np.zeros(4), np.arange(4, dtype=np.float32)))

    ingress.callback(
        audio,
        4,
        {"currentTime": 5.0, "inputBufferAdcTime": 4.996},
        None,
    )
    audio[:, 1] = 99.0  # queued data must be independent of PortAudio's buffer
    blocks = ingress.drain()

    assert len(blocks) == 1
    assert blocks[0].samples.tolist() == [0.0, 1.0, 2.0, 3.0]
    assert blocks[0].frame_end == pytest.approx(100.0)
    assert ingress.queue_depth == 0


def test_audio_callback_never_blocks_when_queue_is_full() -> None:
    clock = FakeClock(100.0)
    ingress = LiveAudioIngress(1000, 0, clock.now, max_blocks=1)
    info = {"currentTime": 1.0, "inputBufferAdcTime": 0.99}
    samples = np.zeros((10, 1), dtype=np.float32)

    ingress.callback(samples, 10, info, None)
    ingress.callback(samples, 10, info, "overflow")

    assert ingress.queue_depth == 1
    assert ingress.diag.callbacks == 2
    assert ingress.diag.dropped_blocks == 1
    assert ingress.diag.status_events == 1


def test_paused_ingress_discards_count_in_without_reporting_drops() -> None:
    clock = FakeClock(100.0)
    ingress = LiveAudioIngress(1000, 0, clock.now, max_blocks=1)
    samples = np.zeros((10, 1), dtype=np.float32)
    info = {"currentTime": 0.0, "inputBufferAdcTime": 0.0}

    ingress.pause()
    for _ in range(5):
        ingress.callback(samples, 10, info, None)
    ingress.resume()
    ingress.callback(samples, 10, info, None)

    assert ingress.diag.callbacks == 6
    assert ingress.diag.dropped_blocks == 0
    assert ingress.diag.queued_blocks == 1
    assert ingress.queue_depth == 1
