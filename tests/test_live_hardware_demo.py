"""Pure tests for the opt-in Bunny hardware runner."""

from __future__ import annotations

import json

from demo_live_clap_to_loopmidi import build_trace_record, main, play_count_in, save_trace
from drummer.live_models import ControllerSnapshot
from drummer.live_runtime import RuntimeSnapshot
from tests.fake_clock import FakeClock
from tests.fake_midi import FakeMidiSink


def _snapshot() -> RuntimeSnapshot:
    return RuntimeSnapshot(
        controller=ControllerSnapshot(
            state="PLAYING",
            generation=2,
            locked_bpm=120.0,
            bar_epoch=100.0,
            beat_period=0.5,
            current_bar_index=3,
            current_slot=7,
            mirror_active=True,
            mirror_slot=3,
            anchor_count=0,
            mirror_count=0,
            queue_depth=4,
            computed_at=106.0,
        ),
        queue_depth=4,
        total_emitted=20,
        total_dropped=1,
        total_late=2,
    )


def test_no_arguments_never_open_hardware(capsys) -> None:
    assert main([]) == 0
    assert "Hardware remains closed" in capsys.readouterr().out


def test_trace_record_contains_operational_diagnostics() -> None:
    record = build_trace_record(
        106.0,
        _snapshot(),
        event_count=9,
        audio_callbacks=100,
        audio_queue_depth=2,
        audio_dropped_blocks=1,
    )
    assert record["state"] == "PLAYING"
    assert record["locked_bpm"] == 120.0
    assert record["mirror_slot"] == 3
    assert record["emitted"] == 20
    assert record["audio_dropped_blocks"] == 1


def test_trace_payload_is_saved_as_json(tmp_path) -> None:
    path = save_trace({"timeline": [{"state": "LISTENING"}]}, tmp_path / "trace.json")
    assert json.loads(path.read_text(encoding="utf-8"))["timeline"][0]["state"] == "LISTENING"


def test_count_in_is_four_audible_hats_with_an_accented_last_click() -> None:
    clock = FakeClock(0.0)
    sink = FakeMidiSink(clock.now)
    sleeps: list[float] = []

    play_count_in(sink, 4, 0.5, sleeps.append)

    assert [event.note for event in sink.events] == [42, 42, 42, 42]
    assert [event.velocity for event in sink.events] == [85, 85, 85, 110]
    assert sleeps == [0.5, 0.5, 0.5, 0.5]
