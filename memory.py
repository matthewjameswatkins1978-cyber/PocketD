"""Lightweight session memory for previous groove settings."""

from __future__ import annotations


class SessionMemory:
    """Recall previously used groove settings for similar fingerprints."""

    def __init__(self) -> None:
        self._entries: list[dict] = []

    def store(self, label: str, fingerprint: dict, groove: str, feel: str, energy: float) -> None:
        self._entries.append(
            {
                "label": label,
                "fingerprint": fingerprint,
                "groove": groove,
                "feel": feel,
                "energy": energy,
            }
        )

    def recall(self, fingerprint: dict) -> dict | None:
        best_entry: dict | None = None
        best_score = 0.0

        for entry in self._entries:
            entry_fp = entry["fingerprint"]
            density_similarity = 1.0 - abs(float(entry_fp.get("density", 0.5)) - float(fingerprint.get("density", 0.5)))
            strong_beats = set(entry_fp.get("strong_beats", []))
            query_beats = set(fingerprint.get("strong_beats", []))
            beat_similarity = 1.0 if not strong_beats and not query_beats else len(strong_beats & query_beats) / max(1, len(strong_beats | query_beats))
            score = 0.6 * density_similarity + 0.4 * beat_similarity

            if score > best_score:
                best_score = score
                best_entry = entry

        if best_entry is not None and best_score >= 0.35:
            return best_entry
        return None
