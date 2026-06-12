"""Attack density tracking for the perception engine.

Density measures how many attacks occur within a recent time window.
This helps distinguish sparse playing (low density) from busy passages
(high density), which is useful for the drummer to decide on fill intensity.
"""

from __future__ import annotations

import bisect


class AttackDensityTracker:
    """Track the density of detected attacks over a sliding time window.

    The tracker maintains a buffer of recent attack timestamps and can report
    how many attacks have occurred within a configurable lookback window.

    Parameters
    ----------
    window_seconds : float
        The time window (in seconds) over which to count attacks.
        Default 2.0 seconds — roughly two bars at 120 BPM.
    """

    def __init__(self, window_seconds: float = 2.0) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        self._window = window_seconds
        self._timestamps: list[float] = []

    def record_attack(self, time_seconds: float) -> None:
        """Record a new attack at the given time.

        Automatically prunes timestamps outside the current window.
        """
        self._timestamps.append(time_seconds)
        self._prune(time_seconds)

    def _prune(self, now: float) -> None:
        """Remove timestamps older than the window."""
        cutoff = now - self._window
        # Use bisect to find the first index >= cutoff
        idx = bisect.bisect_left(self._timestamps, cutoff)
        if idx > 0:
            self._timestamps = self._timestamps[idx:]

    def density(self, now: float | None = None) -> float:
        """Return the number of attacks in the current window.

        Parameters
        ----------
        now : float | None
            Current time in seconds. If None, uses the latest timestamp.
            If no timestamps exist, returns 0.

        Returns
        -------
        float
            Attack count over the window (not normalised).
            This is a raw count, useful for relative comparisons.
        """
        if not self._timestamps:
            return 0.0
        if now is None:
            now = self._timestamps[-1]
        self._prune(now)
        return float(len(self._timestamps))

    def normalised_density(self, now: float | None = None) -> float:
        """Return density normalised to [0, 1].

        Normalisation maps the raw count to [0, 1] based on a heuristic
        maximum density. 16 attacks per 2-second window (i.e. 32nd notes
        at 120 BPM) is considered maximum density.

        Returns
        -------
        float
            Normalised density in [0, 1].
        """
        raw = self.density(now)
        max_density = 16.0  # heuristic maximum
        return min(1.0, raw / max_density)

    def reset(self) -> None:
        """Clear all tracked timestamps."""
        self._timestamps.clear()

    @property
    def count(self) -> int:
        """The current number of tracked timestamps (in the window)."""
        if self._timestamps:
            self._prune(self._timestamps[-1])
        return len(self._timestamps)