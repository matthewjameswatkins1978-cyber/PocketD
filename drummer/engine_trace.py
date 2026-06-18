"""Engine Decision Trace — compact table renderer for engine trace data.

Pure function: takes a list of per-bar trace dicts and returns a
formatted text table.  No MIDI, no playback, no side effects.

This is diagnostic only — it does not change behaviour, output,
or thresholds.
"""

from __future__ import annotations


def render_engine_trace_table(trace: list[dict]) -> str:
    """Render an Engine Decision Trace as a compact text table.

    Parameters
    ----------
    trace : list[dict]
        Per-bar trace entries as produced by ``run_continuous_jam()``
        with ``engine_trace`` enabled.

    Returns
    -------
    str
        A compact multi-line table.
    """
    if not trace:
        return "ENGINE TRACE\n  (empty)"

    lines: list[str] = []
    lines.append("ENGINE TRACE")

    # Column header
    header = (
        f"{'bar':>3s}  {'section':<12s}  {'selected':<12s}  {'rendered':<12s}  "
        f"{'conf':>5s}  {'dens':>5s}  {'cert':>5s}  {'stab':>5s}  "
        f"{'chg':>5s}  {'sil':>5s}  reason"
    )
    lines.append(header)
    lines.append("-" * len(header))

    for entry in trace:
        bar = entry.get("bar", 0)
        section = _truncate(str(entry.get("section", "")), 12)

        # Compare raw values for override detection before truncation
        raw_selected = str(entry.get("selected_intent", ""))
        raw_rendered = str(entry.get("rendered_intent", ""))
        selected = _truncate(raw_selected, 12)
        rendered = _truncate(raw_rendered, 12)

        conf = entry.get("decision_confidence", 0.0) or 0.0
        dens = entry.get("input_density", 0.0) or 0.0
        cert = entry.get("player_certainty", 0.0) or 0.0
        stab = entry.get("repetition_stability", 0.0) or 0.0
        chg = entry.get("change_score", 0.0) or 0.0
        sil = entry.get("silence_duration", 0.0) or 0.0

        reason = _short_reason(entry, raw_selected, raw_rendered)

        lines.append(
            f"{bar:3d}  {section:<12s}  {selected:<12s}  {rendered:<12s}  "
            f"{conf:5.2f}  {dens:5.2f}  {cert:5.2f}  {stab:5.2f}  "
            f"{chg:5.2f}  {sil:5.1f}  {reason}"
        )

    return "\n".join(lines)


def _truncate(text: str, max_len: int) -> str:
    """Truncate text to max_len characters."""
    if len(text) <= max_len:
        return text
    return text[:max_len - 3] + "..."


def _short_reason(entry: dict, selected: str, rendered: str) -> str:
    """Build a compact reason string for one trace row.

    If selected != rendered, show an OVERRIDE note.
    Otherwise, shorten the decision reason to fit.
    """
    decision_reason = str(entry.get("decision_reason", ""))

    # Override detection: selected != rendered
    if selected != rendered:
        return f"OVERRIDE: engine selected {selected}"

    # Strip verbose prefix like "Feature LISTEN: "
    shortened = _strip_feature_prefix(decision_reason)

    return _truncate(shortened, 60)


def _strip_feature_prefix(reason: str) -> str:
    """Remove the 'Feature XXX: ' prefix if present for compactness."""
    if reason.startswith("Feature "):
        idx = reason.find(": ")
        if idx != -1:
            return reason[idx + 2:]
    return reason