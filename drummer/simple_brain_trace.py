"""Simple Brain Decision Trace — compact table renderer.

Pure function: takes a list of per-snapshot trace dicts and returns a
formatted text table.  No MIDI, no playback, no side effects.

Mirrors the pattern used in ``drummer/engine_trace.py`` but produces
rows specific to Simple Brain: action, beat, confidence, feature
columns, and the brain's reason string.

Design contract
---------------
* Pure and deterministic: same trace → same output.
* Does not import or depend on ``drummer/behaviour.py``.
* Diagnostic only — does not change behaviour, output, or thresholds.
"""

from __future__ import annotations


def render_simple_brain_trace_table(trace: list[dict]) -> str:
    """Render a Simple Brain Decision Trace as a compact text table.

    Parameters
    ----------
    trace : list[dict]
        Per-snapshot trace entries.  Each dict may contain any of the
        keys listed below; missing keys render as empty or 0.0.

        Expected keys
        -------------
        bar : int
            Logical bar/snapshot index.
        section : str
            Named section label (e.g. "intro", "verse", "breakdown").
        action : str
            ``BrainAction.value`` — LISTEN, CHOOSE, or HOLD.
        beat : str or None
            Selected beat name, or ``"none"`` if no beat.
        confidence : float
            Decision confidence.
        input_density : float
            Input density from the FeatureSnapshot.
        player_certainty : float
            Player certainty from the FeatureSnapshot.
        stability : float
            Repetition stability from the FeatureSnapshot.
        change_score : float
            Change score from the FeatureSnapshot.
        silence : float
            Silence duration from the FeatureSnapshot.
        reason : str
            The brain's decision reason.

    Returns
    -------
    str
        A compact multi-line table.
    """
    if not trace:
        return "SIMPLE BRAIN TRACE\n  (empty)"

    lines: list[str] = []
    lines.append("SIMPLE BRAIN TRACE")

    # Column header
    header = (
        f"{'bar':>3s}  {'section':<12s}  {'action':<8s}  {'beat':<14s}  "
        f"{'conf':>5s}  {'dens':>5s}  {'cert':>5s}  {'stab':>5s}  "
        f"{'chg':>5s}  {'sil':>5s}  reason"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for entry in trace:
        bar = entry.get("bar", 0)
        section = _truncate(str(entry.get("section", "")), 12)
        action = _truncate(str(entry.get("action", "")), 8)
        beat_raw = entry.get("beat")
        if beat_raw is None:
            beat_str = "none"
        else:
            beat_str = str(beat_raw)
        beat = _truncate(beat_str, 14)

        conf = entry.get("confidence", 0.0) or 0.0
        dens = entry.get("input_density", 0.0) or 0.0
        cert = entry.get("player_certainty", 0.0) or 0.0
        stab = entry.get("stability", 0.0) or 0.0
        chg = entry.get("change_score", 0.0) or 0.0
        sil = entry.get("silence", 0.0) or 0.0

        reason = _truncate(str(entry.get("reason", "")), 60)

        lines.append(
            f"{bar:3d}  {section:<12s}  {action:<8s}  {beat:<14s}  "
            f"{conf:5.2f}  {dens:5.2f}  {cert:5.2f}  {stab:5.2f}  "
            f"{chg:5.2f}  {sil:5.1f}  {reason}"
        )

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to *max_len* characters, appending '...' if needed."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."