"""Performance Confidence / Boldness Layer.

Tracks how confident the drummer should feel based on recent player
stability.  Confidence influences rendering intensity but does NOT
change behaviour intent decisions.

Design contract
---------------
* Pure and deterministic: same inputs → same outputs.
* Confidence is a single float [0.0, 1.0].
* Rises slowly with stability, drops quickly with uncertainty.
* Does NOT replace BehaviourIntent, FeatureMonitor, or OutputShaper.
* Does NOT randomise anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from drummer.behaviour import BehaviourIntent
from perception.features import FeatureSnapshot


@dataclass
class PerformanceConfidenceState:
    """Tracks drummer confidence based on player stability.

    Parameters
    ----------
    confidence : float
        Current confidence level [0.0, 1.0].  Starts at 0.0 (cautious).
    stable_bars : int
        Number of consecutive stable bars (MAINTAIN/BUILD with good features).
    unstable_bars : int
        Number of consecutive unstable bars (ANCHOR/REDUCE/poor features).
    bars_since_anchor : int
        Bars elapsed since last ANCHOR intent.
    bars_since_bail : int
        Bars elapsed since last BAIL or FINAL_BAIL intent.
    last_intent : BehaviourIntent | None
        The intent from the previous update.
    """

    confidence: float = 0.0
    stable_bars: int = 0
    unstable_bars: int = 0
    bars_since_anchor: int = 0
    bars_since_bail: int = 0
    last_intent: BehaviourIntent | None = None

    # --- tuning constants (class-level, not per-instance) ---
    _STABLE_INCREASE: float = 0.10
    """Confidence increase per stable bar."""
    _STRONG_STABLE_INCREASE: float = 0.15
    """Confidence increase per strongly stable bar (BUILD with good features)."""
    _UNCERTAINTY_DROP: float = -0.25
    """Confidence drop per unstable bar."""
    _ANCHOR_DROP: float = -0.35
    """Confidence drop when ANCHOR fires."""
    _BAIL_RESET: float = 0.0
    """Confidence after BAIL or FINAL_BAIL."""

    # --- recovery thresholds ---
    _CERTAINTY_HIGH: float = 0.60
    _PHASE_HIGH: float = 0.60
    _STABILITY_HIGH: float = 0.60
    _DENSITY_LOW: float = 0.75  # must be <= this

    _CERTAINTY_LOW: float = 0.45
    _PHASE_LOW: float = 0.50
    _STABILITY_LOW: float = 0.50

    def update(
        self,
        snapshot: FeatureSnapshot,
        intent: BehaviourIntent,
    ) -> float:
        """Update confidence based on the latest snapshot and intent.

        Parameters
        ----------
        snapshot : FeatureSnapshot
            Current feature summary from the Feature Monitor.
        intent : BehaviourIntent
            The behaviour intent decided for this moment.

        Returns
        -------
        float
            The updated confidence value [0.0, 1.0].
        """
        certainty = snapshot.player_certainty
        phase = snapshot.phase_alignment or 0.0
        stability = snapshot.repetition_stability
        density = snapshot.input_density

        # Track intent transitions for counters
        is_anchor = intent == BehaviourIntent.ANCHOR
        is_bail = intent in (BehaviourIntent.BAIL, BehaviourIntent.FINAL_BAIL)
        is_stable_intent = intent in (BehaviourIntent.MAINTAIN, BehaviourIntent.BUILD)
        is_unstable_intent = intent in (
            BehaviourIntent.ANCHOR, BehaviourIntent.REDUCE, BehaviourIntent.LISTEN,
        )

        # Update counters
        if is_anchor:
            self.bars_since_anchor = 0
        else:
            self.bars_since_anchor += 1

        if is_bail:
            self.bars_since_bail = 0
        else:
            self.bars_since_bail += 1

        # Check feature quality
        features_good = (
            certainty >= self._CERTAINTY_HIGH
            and phase >= self._PHASE_HIGH
            and stability >= self._STABILITY_HIGH
            and density <= self._DENSITY_LOW
        )
        features_poor = (
            certainty < self._CERTAINTY_LOW
            or phase < self._PHASE_LOW
            or stability < self._STABILITY_LOW
        )

        # Determine if this bar is stable or unstable
        is_stable = is_stable_intent and features_good
        is_strongly_stable = (
            intent == BehaviourIntent.BUILD
            and features_good
        )
        is_unstable = is_unstable_intent or features_poor

        if is_stable:
            self.stable_bars += 1
            self.unstable_bars = 0
        elif is_unstable:
            self.unstable_bars += 1
            self.stable_bars = 0
        else:
            # Neutral bar — neither clearly stable nor unstable
            # Keep counters where they are
            pass

        # Apply confidence change
        if is_bail:
            # BAIL or FINAL_BAIL — reset confidence
            self.confidence = self._BAIL_RESET
        elif is_anchor:
            # ANCHOR — significant drop
            self.confidence += self._ANCHOR_DROP
        elif is_strongly_stable:
            # BUILD + good features — strong increase
            self.confidence += self._STRONG_STABLE_INCREASE
        elif is_stable:
            # MAINTAIN + good features — steady increase
            self.confidence += self._STABLE_INCREASE
        elif is_unstable:
            # Unstable bar — drop confidence
            self.confidence += self._UNCERTAINTY_DROP

        # Clamp
        self.confidence = max(0.0, min(1.0, self.confidence))

        self.last_intent = intent
        return self.confidence

    def reset(self) -> None:
        """Reset all state to factory-fresh (confidence = 0.0)."""
        self.confidence = 0.0
        self.stable_bars = 0
        self.unstable_bars = 0
        self.bars_since_anchor = 0
        self.bars_since_bail = 0
        self.last_intent = None

    @property
    def is_confident(self) -> bool:
        """True when confidence is medium-high or above (>= 0.50)."""
        return self.confidence >= 0.50

    @property
    def is_cautious(self) -> bool:
        """True when confidence is low (< 0.35)."""
        return self.confidence < 0.35

    @property
    def confidence_bracket(self) -> str:
        """Return a human-readable bracket label."""
        if self.confidence >= 0.70:
            return "high"
        elif self.confidence >= 0.35:
            return "medium"
        else:
            return "low"
