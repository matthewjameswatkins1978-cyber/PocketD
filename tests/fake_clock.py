"""Injectable fake clock for deterministic controller tests."""

from __future__ import annotations

from collections.abc import Callable

from drummer.live_models import MonotonicClock


class FakeClock:
    """A fake monotonic clock that advances only when explicitly told.

    Usage in tests::

        clock = FakeClock(100.0)
        controller = LiveController(config, clock=clock.now)
        clock.advance(0.5)  # move time forward by 0.5 seconds
    """

    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        """Return the current fake time."""
        return self._now

    def advance(self, delta: float) -> None:
        """Move the clock forward by *delta* seconds."""
        if delta < 0:
            raise ValueError("FakeClock cannot go backwards")
        self._now += delta

    def set(self, absolute: float) -> None:
        """Set the clock to an absolute time."""
        if absolute < self._now:
            raise ValueError("FakeClock cannot go backwards")
        self._now = absolute

    @property
    def as_callable(self) -> MonotonicClock:
        """Return a bound ``Callable[[], float]`` for injection."""
        return self.now


def fake_clock_factory(start: float = 0.0) -> tuple[FakeClock, MonotonicClock]:
    """Create a FakeClock and a callable pointing to its ``now`` method."""
    fc = FakeClock(start)
    return fc, fc.now