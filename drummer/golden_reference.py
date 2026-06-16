"""Golden Reference — ear-approved run frozen as a benchmark.

Stores known-good playtest runs that Matthew's ear has approved.  These
are preserved as regression targets so future changes can be compared
against ear-tested output.

Design contract
---------------
* Pure data model + JSONL persistence.
* No MIDI, no playback, no evaluator changes.
* Simple append-only storage.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------


@dataclass
class GoldenReference:
    """An ear-approved run frozen as a benchmark."""

    scenario: str
    preset: str
    variation: str = ""
    command: str = ""
    timestamp: str = ""          # ISO-8601
    user_rating: int = 5
    user_note: str = ""
    approval_status: str = "approved"
    tag: str = "protect_this_feel"
    diagnostics_path: str | None = None
    midi_path: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# JSONL persistence
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    """Create parent directory of *path* if it doesn't exist."""
    parent = os.path.dirname(path)
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def serialize_golden_reference(ref: GoldenReference) -> str:
    """Return a JSON line string for *ref* (no trailing newline)."""
    return json.dumps(ref.to_dict(), sort_keys=True)


def save_golden_reference(
    ref: GoldenReference,
    path: str = "data/golden_references.jsonl",
) -> None:
    """Append *ref* as one JSON line to the file at *path*."""
    _ensure_dir(path)
    line = serialize_golden_reference(ref)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_golden_references(
    path: str = "data/golden_references.jsonl",
) -> list[dict]:
    """Read all JSON lines from *path* and return a list of dicts.

    Returns an empty list if the file does not exist.
    """
    if not os.path.exists(path):
        return []
    entries: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                entries.append(json.loads(stripped))
    return entries


# ---------------------------------------------------------------------------
# Diagnostics snapshot helper
# ---------------------------------------------------------------------------


def save_golden_diagnostics(
    diagnostics: list[dict],
    path: str,
) -> None:
    """Save a list of per-bar diagnostic dicts as a JSON snapshot."""
    _ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(diagnostics, f, indent=2, sort_keys=True)
        f.write("\n")