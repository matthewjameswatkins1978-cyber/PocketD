"""Pure tests for the opt-in Bunny hardware runner."""

from __future__ import annotations

import json

from demo_live_clap_to_loopmidi import build_trace_record, main, save_trace
from drummer.live_models import ControllerSnapshot
from drummer.live_runtime import RuntimeSnapshot


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
