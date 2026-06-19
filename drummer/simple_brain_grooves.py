"""Simple Brain beat-name → groove-id resolver.

Since Simple Brain v1, every non-silence ``BrainDecision.beat_name`` is a
real groove ID from ``data/grooves.yaml``.  The mapping is therefore an
identity function for all non-silence beats.  ``silence`` maps to
``None`` (no groove data), and unknown/``None`` beat names also return
``None``.

Design contract
---------------
* Pure function: same input → same output.
* Does not import or depend on ``drummer/behaviour.py``.
* ``silence`` maps to ``None`` — no groove data.
* Unknown beat names return ``None`` (safe default).
* All non-silence beat names produced by the default
  ``SimpleBrain`` are valid groove IDs.
"""

from __future__ import annotations

from groove_library import get_groove


def simple_brain_beat_to_groove_id(beat_name: str | None) -> str | None:
    """Resolve a Simple Brain beat name to a groove-library id.

    Parameters
    ----------
    beat_name : str or None
        A beat name as returned by ``BrainDecision.beat_name``.

    Returns
    -------
    str or None
        A groove id recognised by ``groove_library.get_groove()``,
        or ``None`` if the beat is silence, unknown, or ``beat_name``
        was ``None`` itself.
    """
    if beat_name is None:
        return None
    if beat_name == "silence":
        return None
    # Since Simple Brain v1, non-silence beat names are real groove IDs.
    # Verify the groove exists in the library as a safety check.
    try:
        get_groove(beat_name)
        return beat_name
    except KeyError:
        return None


def resolve_simple_brain_groove(beat_name: str | None):
    """Resolve a Simple Brain beat name to a loaded Groove object.

    Parameters
    ----------
    beat_name : str or None
        A beat name as returned by ``BrainDecision.beat_name``.

    Returns
    -------
    Groove or None
        The loaded ``Groove`` object from ``groove_library``, or
        ``None`` for silence / unknown / None beat_name.
    """
    groove_id = simple_brain_beat_to_groove_id(beat_name)
    if groove_id is None:
        return None
    try:
        return get_groove(groove_id)
    except KeyError:
        return None