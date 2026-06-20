"""Tests for Simple Brain beat-name → groove-id resolver.

Since Simple Brain v1, beat names are real groove IDs, so the mapping
is identity for non-silence beats.  ``silence`` and unknown beat names
map to None.
"""

from __future__ import annotations

from drummer.simple_brain_grooves import (
    resolve_simple_brain_groove,
    simple_brain_beat_to_groove_id,
)
from groove_library import get_groove


# ---------------------------------------------------------------------------
# Beat-name → groove-id mapping
# ---------------------------------------------------------------------------


def test_known_beats_resolve_to_themselves() -> None:
    """Known groove IDs map to themselves."""
    for groove_id in ("simple_rock", "motorik", "half_time", "shuffle",
                       "funk_pocket", "punk_drive"):
        assert simple_brain_beat_to_groove_id(groove_id) == groove_id


def test_silence_resolves_to_none() -> None:
    """silence beat resolves to None (no groove data)."""
    assert simple_brain_beat_to_groove_id("silence") is None
    assert resolve_simple_brain_groove("silence") is None


def test_none_beat_name_resolves_to_none() -> None:
    """None beat name (LISTEN state) resolves to None."""
    assert simple_brain_beat_to_groove_id(None) is None
    assert resolve_simple_brain_groove(None) is None


def test_unknown_beat_name_resolves_to_none() -> None:
    """A beat name not in grooves.yaml returns None (safe default)."""
    assert simple_brain_beat_to_groove_id("bossa_nova") is None
    assert resolve_simple_brain_groove("bossa_nova") is None
    # Old invented names that no longer exist
    assert simple_brain_beat_to_groove_id("four_on_floor") is None
    assert simple_brain_beat_to_groove_id("laid_back") is None


def test_groove_ids_exist_in_library() -> None:
    """Every known groove ID returned by the resolver loads from groove_library."""
    groove_ids = [
        "simple_rock", "motorik", "half_time", "shuffle",
        "funk_pocket", "punk_drive",
    ]
    for groove_id in groove_ids:
        resolved = simple_brain_beat_to_groove_id(groove_id)
        assert resolved == groove_id
        groove = get_groove(groove_id)
        assert groove is not None
        assert groove.id == groove_id


# ---------------------------------------------------------------------------
# Groove-object resolution
# ---------------------------------------------------------------------------


def test_resolve_returns_groove_objects() -> None:
    """resolve_simple_brain_groove returns actual Groove objects."""
    for beat_name in ("simple_rock", "motorik", "half_time", "shuffle",
                       "funk_pocket", "punk_drive"):
        groove = resolve_simple_brain_groove(beat_name)
        assert groove is not None
        assert groove.id == beat_name
        assert hasattr(groove, "steps")
        assert hasattr(groove, "kick_steps")


def test_resolve_is_pure() -> None:
    """Same input produces same output (deterministic)."""
    assert resolve_simple_brain_groove("simple_rock") == resolve_simple_brain_groove("simple_rock")
    assert simple_brain_beat_to_groove_id("motorik") == simple_brain_beat_to_groove_id("motorik")
    # Unknown is consistently None
    assert simple_brain_beat_to_groove_id("nonexistent") is None
    assert resolve_simple_brain_groove("nonexistent") is None