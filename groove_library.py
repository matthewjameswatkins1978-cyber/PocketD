"""Load groove definitions from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml

from models import Groove

DATA_DIR = Path(__file__).resolve().parent / "data"
GROOVES_PATH = DATA_DIR / "grooves.yaml"


def load_grooves(path: Path | None = None) -> dict[str, Groove]:
    """Load all grooves keyed by id."""
    path = path or GROOVES_PATH
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    grooves: dict[str, Groove] = {}
    for item in raw.get("grooves", []):
        groove = Groove(
            id=item["id"],
            name=item["name"],
            bars=item["bars"],
            steps=item["steps"],
            kick_steps=list(item["kick_steps"]),
            snare_steps=list(item["snare_steps"]),
            hat_steps=list(item["hat_steps"]),
            energy=item["energy"],
            density=item["density"],
            risk=item["risk"],
            # Simple Brain metadata (optional — safe defaults from Groove dataclass)
            simple_brain_enabled=item.get("simple_brain_enabled", False),
            ideal_density=item.get("ideal_density", ""),
            min_stability=float(item.get("min_stability", 0.0)),
            feel_tags=list(item.get("feel_tags", [])),
            description=item.get("description", ""),
        )
        grooves[groove.id] = groove
    return grooves


def get_groove(groove_id: str, path: Path | None = None) -> Groove:
    grooves = load_grooves(path)
    if groove_id not in grooves:
        available = ", ".join(sorted(grooves))
        raise KeyError(f"Unknown groove '{groove_id}'. Available: {available}")
    return grooves[groove_id]
