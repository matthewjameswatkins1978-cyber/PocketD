"""Tests for golden reference storage."""

from __future__ import annotations

import json
import os
import tempfile

import pytest

from drummer.golden_reference import (
    GoldenReference,
    save_golden_reference,
    load_golden_references,
    save_golden_diagnostics,
    serialize_golden_reference,
)


def _tmp_path(suffix: str = ".tmp") -> str:
    """Create a safe temporary path using NamedTemporaryFile."""
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    return tmp.name


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_golden() -> GoldenReference:
    return GoldenReference(
        scenario="build",
        preset="cautious",
        variation="slow_build",
        command="python demo_playtest_interview.py --scenario build --preset cautious",
        timestamp="2026-06-16T21:00:00",
        user_rating=5,
        user_note="Sounded great — protect this feel.",
        approval_status="approved",
        tag="protect_this_feel",
        diagnostics_path="data/golden_diagnostics/build_cautious_slow_build.json",
        midi_path=None,
    )


# ---------------------------------------------------------------------------
# Serialization tests
# ---------------------------------------------------------------------------


class TestGoldenReferenceSerialization:
    def test_to_dict_has_all_required_fields(self, sample_golden: GoldenReference) -> None:
        d = sample_golden.to_dict()
        assert d["scenario"] == "build"
        assert d["preset"] == "cautious"
        assert d["variation"] == "slow_build"
        assert d["command"] == "python demo_playtest_interview.py --scenario build --preset cautious"
        assert d["timestamp"] == "2026-06-16T21:00:00"
        assert d["user_rating"] == 5
        assert d["user_note"] == "Sounded great — protect this feel."
        assert d["approval_status"] == "approved"
        assert d["tag"] == "protect_this_feel"
        assert d["diagnostics_path"] == "data/golden_diagnostics/build_cautious_slow_build.json"
        assert d["midi_path"] is None

    def test_deterministic_serialization(self, sample_golden: GoldenReference) -> None:
        line1 = serialize_golden_reference(sample_golden)
        line2 = serialize_golden_reference(sample_golden)
        assert line1 == line2

    def test_serialize_produces_valid_json(self, sample_golden: GoldenReference) -> None:
        line = serialize_golden_reference(sample_golden)
        parsed = json.loads(line)
        assert parsed["scenario"] == "build"
        assert parsed["preset"] == "cautious"

    def test_missing_diagnostics_path_is_allowed(self) -> None:
        ref = GoldenReference(
            scenario="drop",
            preset="cautious",
            variation="deliberate_sparse",
            diagnostics_path=None,
        )
        d = ref.to_dict()
        assert d["diagnostics_path"] is None
        # Serialization should still work
        line = serialize_golden_reference(ref)
        parsed = json.loads(line)
        assert parsed["diagnostics_path"] is None

    def test_scenario_preset_note_roundtrip(self) -> None:
        ref = GoldenReference(
            scenario="build",
            preset="cautious",
            user_note="Matthew liked this.",
        )
        d = ref.to_dict()
        assert d["scenario"] == "build"
        assert d["preset"] == "cautious"
        assert d["user_note"] == "Matthew liked this."


# ---------------------------------------------------------------------------
# JSONL persistence tests
# ---------------------------------------------------------------------------


class TestGoldenReferencePersistence:
    def test_save_and_load_single(self, sample_golden: GoldenReference) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            save_golden_reference(sample_golden, path=tmp)
            entries = load_golden_references(path=tmp)
            assert len(entries) == 1
            assert entries[0]["scenario"] == "build"
            assert entries[0]["preset"] == "cautious"
            assert entries[0]["user_rating"] == 5
            assert entries[0]["user_note"] == "Sounded great — protect this feel."
            assert entries[0]["approval_status"] == "approved"
            assert entries[0]["tag"] == "protect_this_feel"
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_append_multiple_references(self) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            ref1 = GoldenReference(
                scenario="build", preset="cautious", variation="slow_build",
                user_rating=5, user_note="Great build.",
            )
            ref2 = GoldenReference(
                scenario="drop", preset="cautious", variation="deliberate_sparse",
                user_rating=4, user_note="Nice drop.",
            )
            save_golden_reference(ref1, path=tmp)
            save_golden_reference(ref2, path=tmp)

            entries = load_golden_references(path=tmp)
            assert len(entries) == 2
            assert entries[0]["scenario"] == "build"
            assert entries[1]["scenario"] == "drop"
            assert entries[0]["user_note"] == "Great build."
            assert entries[1]["user_note"] == "Nice drop."
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_load_from_non_existent_file_returns_empty(self) -> None:
        entries = load_golden_references(path="/no/such/file_xyz_123.jsonl")
        assert entries == []

    def test_each_entry_is_one_json_line(self) -> None:
        tmp = _tmp_path(suffix=".jsonl")
        try:
            for i in range(3):
                ref = GoldenReference(
                    scenario="build", preset="cautious",
                    variation=f"run_{i}", user_rating=4 + i,
                )
                save_golden_reference(ref, path=tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                lines = [l.strip() for l in f if l.strip()]
            assert len(lines) == 3
            for line in lines:
                obj = json.loads(line)
                assert "scenario" in obj
                assert "preset" in obj
                assert "variation" in obj
                assert "approval_status" in obj
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_save_creates_parent_directory(self) -> None:
        tmp_dir = _tmp_path(suffix="")
        os.remove(tmp_dir)  # Now a non-existent path
        tmp = os.path.join(tmp_dir, "subdir", "test.jsonl")
        try:
            ref = GoldenReference(scenario="build", preset="cautious")
            save_golden_reference(ref, path=tmp)
            assert os.path.exists(tmp)
            entries = load_golden_references(path=tmp)
            assert len(entries) == 1
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Diagnostics snapshot tests
# ---------------------------------------------------------------------------


class TestGoldenDiagnostics:
    def test_save_and_load_diagnostics_snapshot(self) -> None:
        tmp = _tmp_path(suffix=".json")
        diags = [
            {"bar": 0, "section": "LISTEN", "event_count": 0, "intent": "listen"},
            {"bar": 1, "section": "ENTER_SOFT", "event_count": 5, "intent": "enter_soft"},
            {"bar": 2, "section": "MAINTAIN_1", "event_count": 8, "intent": "maintain"},
        ]
        try:
            save_golden_diagnostics(diags, path=tmp)
            assert os.path.exists(tmp)
            with open(tmp, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            assert len(loaded) == 3
            assert loaded[0]["bar"] == 0
            assert loaded[1]["section"] == "ENTER_SOFT"
            assert loaded[2]["event_count"] == 8
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def test_diagnostics_snapshot_creates_parent_dir(self) -> None:
        tmp_dir = _tmp_path(suffix="")
        os.remove(tmp_dir)
        tmp = os.path.join(tmp_dir, "nested", "diags.json")
        diags = [{"bar": 0, "event_count": 0}]
        try:
            save_golden_diagnostics(diags, path=tmp)
            assert os.path.exists(tmp)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
            if os.path.exists(tmp_dir):
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)