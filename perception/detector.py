"""Detector Diagnostics — input-side diagnostic layer for Pocket Drummer.

Answers:
- What did the system hear?
- When were attacks detected?
- How strong were they?
- Was the input noisy, sparse, spammy, stable, or pulse-like?

This is purely diagnostic. Warnings are for reporting only and are NOT
fed into behaviour decisions, confidence engines, or output shaping.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from perception.event_listener import detect_events_from_audio
from perception.models import MusicalEvent
from perception.pulse import PulseTracker

log = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────


@dataclass
class DetectorConfig:
    """Tunable knobs for the Detector diagnostics layer.

    All defaults are conservative.  Raise thresholds if you see too many
    false positives; lower them if real attacks are being missed.
    """

    # Passed-through to onset detection
    threshold_mult: float = 0.75
    std_mult: float = 0.35
    min_interval: float = 0.05  # seconds

    # Warning thresholds (diagnostic-only)
    min_events_for_pulse: int = 4
    spam_threshold_hz: float = 20.0
    low_signal_rms: float = 0.02
    clipping_threshold: float = 0.95
    pulse_confidence_min: float = 0.3
    noise_floor_percentile: float = 10.0

    # Pulse tracker construction
    pulse_min_bpm: float = 40.0
    pulse_max_bpm: float = 250.0

    # ── Noise Gate (Phase 2) ────────────────────────────────────────
    noise_gate_enabled: bool = True
    min_event_energy: float = 0.01
    min_event_strength: float = 0.0
    noise_floor_ratio: float = 3.0

    # ── Hum / Sustained-Tone Rejection (Phase 6) ───────────────────
    # Rejects steady hum/sine tones that produce dense, monotonic
    # false-positive events in low-frequency bands.
    hum_rejection_enabled: bool = True
    # Minimum events/sec to suspect hum/sustained tone
    hum_rate_threshold: float = 8.0
    # If this fraction of accepted events is in sub+low bands → suspect hum
    hum_low_freq_ratio: float = 0.75
    # If only this many distinct frequency regions exist → monotonic signal
    hum_max_distinct_regions: int = 2

    # ── Sustained-Tone Rejection (Phase 8) ─────────────────────────
    # Rejects keyboard/pad-like sustained tones that produce repeated
    # false onsets at steady intervals with near-constant energy.
    sustained_rejection_enabled: bool = True
    # Minimum events/sec to suspect sustained false-onset spam
    sustained_min_event_rate: float = 8.0
    # Max inter-quartile range of event energy within dominant region
    # (low variation = flat sustained tone)
    sustained_energy_variation_max: float = 0.015
    # Fraction of events that must be in dominant region to trigger
    sustained_region_dominance_ratio: float = 0.8
    # Max gap (seconds) between consecutive events in same region
    # for them to be considered part of a sustained run
    sustained_max_gap_seconds: float = 0.15

    # ── Summary thresholds (Phase 4) ─────────────────────────────────
    sparse_rate_threshold: float = 1.0
    dense_rate_threshold: float = 10.0
    stable_rate_max: float = 10.0


# ── Diagnostics Snapshot ─────────────────────────────────────────────


@dataclass
class DetectorDiagnostics:
    """Read-only snapshot of everything the detector observed.

    This is the diagnostic output; it does NOT influence behaviour.
    """

    # Basic stats
    duration_seconds: float = 0.0
    event_count: int = 0
    events_per_second: float = 0.0
    rejected_count: int = 0  # events filtered by noise gate (Phase 2)

    # Energy / strength
    avg_strength: float = 0.0
    peak_strength: float = 0.0
    avg_energy: float = 0.0
    peak_energy: float = 0.0
    estimated_noise_floor: float = 0.0
    signal_rms: float = 0.0
    clipping_fraction: float = 0.0

    # Density
    density_final: float = 0.0
    density_mean_first_half: float = 0.0
    density_mean_second_half: float = 0.0
    density_trend: str = "flat"

    # Frequency regions
    frequency_region_counts: dict[str, int] = field(default_factory=dict)

    # Pulse estimate (optional — only computed when there are enough events)
    pulse_bpm: float | None = None
    pulse_confidence: float = 0.0
    pulse_stability: str = "unknown"

    # Warnings (list of human-readable strings)
    warnings: list[str] = field(default_factory=list)

    # Raw events for downstream inspection
    events: list[MusicalEvent] = field(default_factory=list)

    # ── Phase 9: Per-filter rejection counts ────────────────────────
    rejected_by_noise_gate: int = 0
    rejected_by_hum: int = 0
    rejected_by_sustained: int = 0

    # ── Phase 10: Timing diagnostics ────────────────────────────────
    median_ioi_ms: float = 0.0
    ioi_spread_ms: float = 0.0
    ioi_cv: float = 0.0
    timing_feel: str = "unknown"
    timing_flags: list[str] = field(default_factory=list)
    expected_bpm: float | None = None

    # ── Phase 4: Automated Interpretation ───────────────────────────
    input_quality: str = "unknown"
    musical_state: str = "unknown"
    recommendation: str = ""
    summary_flags: list[str] = field(default_factory=list)


# ── Result Container ─────────────────────────────────────────────────


@dataclass
class DetectorResult:
    """Thin container returned by Detector.detect()."""

    diagnostics: DetectorDiagnostics
    events: list[MusicalEvent]
    config: DetectorConfig


# ── Helper ───────────────────────────────────────────────────────────


def _recommendation_for_noise(d: "DetectorDiagnostics") -> str:
    """Produce a noise-state recommendation based on per-filter counts."""
    if d.rejected_by_sustained > 0:
        return (
            "Sustained-tone false onsets were rejected.  "
            "Good for avoiding pad/keyboard sustain spam."
        )
    if d.rejected_by_hum > 0:
        return "Steady low-frequency hum was rejected."
    if d.rejected_by_noise_gate > 0:
        return (
            "Raw noise-like onsets were rejected by the noise gate.  "
            "Try a cleaner signal or raise min_event_energy."
        )
    return "Raw onsets were rejected.  Try a cleaner signal or raise min_event_energy."


# ── Detector ─────────────────────────────────────────────────────────


class Detector:
    """Input-side diagnostic detector.

    Accepts an audio array + sample rate, runs detection through the
    existing perception pipeline, and produces a DetectorDiagnostics
    snapshot with warnings, pulse estimates, density trends, and
    automated summary interpretation.

    This is additive — it does NOT modify any existing behaviour,
    MIDI output, or confidence engines.

    Parameters
    ----------
    config : DetectorConfig | None
        Optional configuration.  If None, conservative defaults are used.
    """

    def __init__(self, config: DetectorConfig | None = None) -> None:
        self._config = config or DetectorConfig()

    # ── Public API ────────────────────────────────────────────────

    def detect(
        self,
        signal: np.ndarray,
        sample_rate: int,
    ) -> DetectorResult:
        """Run detection and return full diagnostics.

        Parameters
        ----------
        signal : np.ndarray
            1-D or 2-D audio samples (float in [-1, 1] preferred).
        sample_rate : int
            Sample rate in Hz.

        Returns
        -------
        DetectorResult
        """
        cfg = self._config

        # --- signal-level statistics --------------------------------
        data = np.asarray(signal, dtype=np.float64)
        if data.ndim > 1:
            data = data.mean(axis=1)

        duration = len(data) / max(sample_rate, 1)
        signal_rms = float(np.sqrt(np.mean(data**2))) if data.size else 0.0

        # Noise floor: low-percentile of absolute amplitude
        noise_floor = (
            float(np.percentile(np.abs(data), cfg.noise_floor_percentile))
            if data.size > 0
            else 0.0
        )

        # Clipping fraction
        clipping_frac = (
            float(np.mean(np.abs(data) >= cfg.clipping_threshold))
            if data.size > 0
            else 0.0
        )

        # --- detect events via existing pipeline --------------------
        raw_events = detect_events_from_audio(
            signal,
            sample_rate,
            threshold_mult=cfg.threshold_mult,
            std_mult=cfg.std_mult,
            min_interval=cfg.min_interval,
        )

        # --- noise gate (Phase 2) -----------------------------------
        raw_count = len(raw_events)
        rejected_by_gate = 0
        if cfg.noise_gate_enabled and raw_count > 0:
            pre_gate = len(raw_events)
            events = self._apply_noise_gate(raw_events, noise_floor, cfg)
            rejected_by_gate = pre_gate - len(events)
        else:
            events = list(raw_events)
        rejected_count = rejected_by_gate

        # --- hum / sustained-tone rejection (Phase 6) ----------------
        rejected_by_hum = 0
        if cfg.hum_rejection_enabled and len(events) > 0:
            events, hum_count = self._reject_hum(events, cfg)
            rejected_by_hum = hum_count
            rejected_count += hum_count

        # --- sustained-tone rejection (Phase 8) ----------------------
        rejected_by_sustained = 0
        if cfg.sustained_rejection_enabled and len(events) > 0:
            events, sustained_count = self._reject_sustained(events, cfg)
            rejected_by_sustained = sustained_count
            rejected_count += sustained_count

        event_count = len(events)

        # --- event-level statistics ---------------------------------
        if events:
            strengths = [e.strength for e in events]
            energies = [e.energy for e in events]
            avg_strength = float(np.mean(strengths))
            peak_strength = float(np.max(strengths))
            avg_energy = float(np.mean(energies))
            peak_energy = float(np.max(energies))
            densities = [e.density for e in events]
            density_final = densities[-1] if densities else 0.0

            # Density trend: compare first half vs second half
            mid = len(densities) // 2
            first_half = densities[:mid] if mid > 0 else densities
            second_half = densities[mid:] if mid > 0 else densities
            density_mean_first = float(np.mean(first_half)) if first_half else 0.0
            density_mean_second = float(np.mean(second_half)) if second_half else 0.0

            if density_mean_second - density_mean_first > 0.05:
                density_trend = "rising"
            elif density_mean_first - density_mean_second > 0.05:
                density_trend = "falling"
            elif density_final < 0.05:
                density_trend = "sparse"
            else:
                density_trend = "flat"

            # Frequency region counts
            region_counts: dict[str, int] = {}
            for e in events:
                region_counts[e.frequency_region] = (
                    region_counts.get(e.frequency_region, 0) + 1
                )
        else:
            avg_strength = 0.0
            peak_strength = 0.0
            avg_energy = 0.0
            peak_energy = 0.0
            density_final = 0.0
            density_mean_first = 0.0
            density_mean_second = 0.0
            density_trend = "sparse"
            region_counts = {}

        # Event rate
        events_per_second = event_count / duration if duration > 0 else 0.0

        # --- pulse estimate (optional) ------------------------------
        pulse_bpm: float | None = None
        pulse_confidence: float = 0.0
        pulse_stability: str = "unknown"

        if event_count >= cfg.min_events_for_pulse:
            try:
                tracker = PulseTracker(
                    min_bpm=cfg.pulse_min_bpm,
                    max_bpm=cfg.pulse_max_bpm,
                )
                for event in events:
                    tracker.process_event(event)
                state = tracker.get_state()
                pulse_bpm = state.best_bpm
                pulse_confidence = state.confidence
                pulse_stability = state.stability
            except Exception:
                log.exception("PulseTracker raised an unexpected error")

        # --- timing diagnostics (Phase 10) --------------------------
        median_ioi_ms = 0.0
        ioi_spread_ms = 0.0
        ioi_cv = 0.0
        timing_feel = "unknown"
        timing_flags: list[str] = []

        if event_count >= 4:
            times = sorted(e.time_seconds for e in events)
            iois = np.diff(times)
            if len(iois) >= 2:
                median_ioi = float(np.median(iois))
                median_ioi_ms = median_ioi * 1000
                iqr_ioi = float(np.percentile(iois, 75)) - float(np.percentile(iois, 25))
                ioi_spread_ms = iqr_ioi * 1000
                mean_ioi = float(np.mean(iois))
                ioi_cv = (float(np.std(iois)) / mean_ioi) if mean_ioi > 0 else 0.0

                # Classify timing feel
                if ioi_cv < 0.03 and ioi_spread_ms < 20:
                    timing_feel = "tight"
                elif ioi_cv < 0.15 and ioi_spread_ms < 60:
                    timing_feel = "loose"
                elif ioi_cv < 0.30 and ioi_spread_ms < 120:
                    timing_feel = "unstable"
                else:
                    timing_feel = "drifting"

                # Timing flags
                if ioi_cv < 0.02 and ioi_spread_ms < 15:
                    timing_flags.append("TIGHT_TIMING")
                if 0.02 <= ioi_cv < 0.10 and ioi_spread_ms < 60:
                    timing_flags.append("LOOSE_BUT_STABLE")
                if ioi_cv >= 0.10:
                    timing_flags.append("HEAVY_TIMING_JITTER")
                if timing_feel == "drifting":
                    timing_flags.append("TEMPO_DRIFT")
                if timing_feel == "unstable":
                    timing_flags.append("UNSTABLE_TIMING")

        # --- warnings -----------------------------------------------
        warnings: list[str] = []

        if event_count == 0:
            warnings.append("NO_EVENTS")
        if signal_rms < cfg.low_signal_rms:
            warnings.append("LOW_SIGNAL")
        if events_per_second > cfg.spam_threshold_hz:
            warnings.append("EVENT_SPAM")
        if clipping_frac > 0.05:
            warnings.append("CLIPPING")
        if (
            event_count >= cfg.min_events_for_pulse
            and pulse_confidence < cfg.pulse_confidence_min
        ):
            warnings.append("NO_STABLE_PULSE")

        # Build partial diagnostics (without summary, added below)
        diagnostics = DetectorDiagnostics(
            duration_seconds=duration,
            event_count=event_count,
            events_per_second=events_per_second,
            rejected_count=rejected_count,
            avg_strength=avg_strength,
            peak_strength=peak_strength,
            avg_energy=avg_energy,
            peak_energy=peak_energy,
            estimated_noise_floor=noise_floor,
            signal_rms=signal_rms,
            clipping_fraction=clipping_frac,
            density_final=density_final,
            density_mean_first_half=density_mean_first,
            density_mean_second_half=density_mean_second,
            density_trend=density_trend,
            frequency_region_counts=region_counts,
            pulse_bpm=pulse_bpm,
            pulse_confidence=pulse_confidence,
            pulse_stability=pulse_stability,
            warnings=warnings,
            events=events,
            rejected_by_noise_gate=rejected_by_gate,
            rejected_by_hum=rejected_by_hum,
            rejected_by_sustained=rejected_by_sustained,
            median_ioi_ms=median_ioi_ms,
            ioi_spread_ms=ioi_spread_ms,
            ioi_cv=ioi_cv,
            timing_feel=timing_feel,
            timing_flags=timing_flags,
        )

        # --- automated summary (Phase 4) ----------------------------
        summary = self._build_summary(diagnostics, cfg)
        diagnostics.input_quality = summary["input_quality"]
        diagnostics.musical_state = summary["musical_state"]
        diagnostics.recommendation = summary["recommendation"]
        diagnostics.summary_flags = summary["summary_flags"]

        return DetectorResult(
            diagnostics=diagnostics,
            events=events,
            config=cfg,
        )

    # ── Noise Gate (Phase 2) ──────────────────────────────────────

    @staticmethod
    def _apply_noise_gate(
        events: list[MusicalEvent],
        noise_floor: float,
        cfg: DetectorConfig,
    ) -> list[MusicalEvent]:
        """Filter events whose energy is not meaningfully above the noise floor."""
        accepted: list[MusicalEvent] = []
        for event in events:
            if event.energy < cfg.min_event_energy:
                continue
            if event.strength < cfg.min_event_strength:
                continue
            gated_threshold = noise_floor * cfg.noise_floor_ratio
            if event.energy < gated_threshold:
                continue
            accepted.append(event)
        return accepted

    # ── Hum / Sustained-Tone Rejection (Phase 6) ──────────────────

    @staticmethod
    def _reject_hum(
        events: list[MusicalEvent],
        cfg: DetectorConfig,
    ) -> tuple[list[MusicalEvent], int]:
        """Filter events that appear to come from steady hum or sustained tones.

        Detection criteria (must satisfy ALL):
        1. Event rate exceeds hum_rate_threshold (dense, monotonic)
        2. Dominant frequency is in sub/low bands (low-frequency energy)
        3. Frequency regions are limited to ≤ hum_max_distinct_regions

        If hum is detected, ALL events are rejected — a sustained tone
        may produce intermittent onsets, but none are musically meaningful.

        Returns (filtered_events, rejected_count).
        """
        if not events:
            return events, 0

        # Compute rate from the event timestamps
        if len(events) < 2:
            return events, 0

        duration = events[-1].time_seconds - events[0].time_seconds
        if duration <= 0:
            return events, 0
        rate = len(events) / duration

        # Criterion 1: dense event rate
        if rate < cfg.hum_rate_threshold:
            return events, 0

        # Criterion 2: low-frequency dominance
        region_counts: dict[str, int] = {}
        for e in events:
            region_counts[e.frequency_region] = (
                region_counts.get(e.frequency_region, 0) + 1
            )
        low_count = region_counts.get("sub", 0) + region_counts.get("low", 0)
        low_ratio = low_count / len(events) if events else 0.0

        if low_ratio < cfg.hum_low_freq_ratio:
            return events, 0

        # Criterion 3: limited spectral diversity (monotonic signal)
        distinct_regions = len(region_counts)
        if distinct_regions > cfg.hum_max_distinct_regions:
            return events, 0

        # All criteria met — reject entire set as hum/sustained tone
        return [], len(events)

    # ── Sustained-Tone Rejection (Phase 8) ─────────────────────────

    @staticmethod
    def _reject_sustained(
        events: list[MusicalEvent],
        cfg: DetectorConfig,
    ) -> tuple[list[MusicalEvent], int]:
        """Filter events from sustained keyboard/pad tones.

        Detects dense, same-region events with near-constant energy
        (flat sustained tones) and rejects all but the first few.

        Detection criteria (must satisfy ALL to trigger):
        1. Event rate >= sustained_min_event_rate
        2. A single frequency region dominates (>= sustained_region_dominance_ratio)
        3. Energy variation within dominant region is low (IQR <= sustained_energy_variation_max)
        4. Consecutive events in dominant region have short gaps (<= sustained_max_gap_seconds)

        When triggered, ALL events in the dominant region are rejected.
        Returns (filtered_events, rejected_count).
        """
        if not events or len(events) < 4:
            return events, 0

        # Compute rate
        duration = events[-1].time_seconds - events[0].time_seconds
        if duration <= 0:
            return events, 0
        rate = len(events) / duration

        if rate < cfg.sustained_min_event_rate:
            return events, 0

        # Find dominant region
        region_counts: dict[str, int] = {}
        for e in events:
            region_counts[e.frequency_region] = (
                region_counts.get(e.frequency_region, 0) + 1
            )
        total = len(events)
        dominant_region = max(region_counts, key=region_counts.get)  # type: ignore[arg-type]
        dom_count = region_counts[dominant_region]

        if dom_count / total < cfg.sustained_region_dominance_ratio:
            return events, 0

        # Extract energies of events in dominant region
        dom_energies = [e.energy for e in events if e.frequency_region == dominant_region]
        if len(dom_energies) < 4:
            return events, 0

        # Check energy variation — low variation = flat sustained tone
        q1 = float(np.percentile(dom_energies, 25))
        q3 = float(np.percentile(dom_energies, 75))
        iqr = q3 - q1

        if iqr > cfg.sustained_energy_variation_max:
            return events, 0

        # Check inter-event gaps in dominant region.
        # Use median gap — sustained tones have mostly short gaps
        # even if there are occasional silences between notes.
        dom_times = sorted(e.time_seconds for e in events if e.frequency_region == dominant_region)
        if len(dom_times) > 2:
            gaps = np.array([dom_times[i] - dom_times[i - 1] for i in range(1, len(dom_times))])
            median_gap = float(np.median(gaps))
            if median_gap > cfg.sustained_max_gap_seconds:
                return events, 0
        else:
            return events, 0

        # All criteria met — sustained tone detected.
        # Keep events in non-dominant regions, reject dominant-region events.
        kept = [e for e in events if e.frequency_region != dominant_region]
        rejected = total - len(kept)
        return kept, rejected

    # ── Summary / Detector Doctor (Phase 4) ────────────────────────

    @staticmethod
    def _build_summary(
        d: DetectorDiagnostics,
        cfg: DetectorConfig,
    ) -> dict[str, object]:
        """Produce automated interpretation labels from diagnostics.

        Returns a dict with keys: input_quality, musical_state,
        recommendation, summary_flags.
        """
        flags: list[str] = []
        input_quality: str
        musical_state: str
        recommendation: str

        eps = d.events_per_second

        # ── Core classification ────────────────────────────────────

        # Case 1: All raw events were rejected → noise
        if d.rejected_count > 0 and d.event_count == 0:
            input_quality = "noisy"
            musical_state = "noise"
            # Phase 9: per-filter attribution
            if d.rejected_by_noise_gate > 0:
                flags.append("RAW_ONSET_SPAM_GATED")
                flags.append("NOISE_REJECTED")
            if d.rejected_by_hum > 0:
                flags.append("HUM_REJECTED")
            if d.rejected_by_sustained > 0:
                flags.append("SUSTAINED_TONE_REJECTED")

        # Case 2: No events at all → silence
        elif d.event_count == 0:
            input_quality = "unusable" if d.signal_rms < cfg.low_signal_rms else "weak"
            musical_state = "silence"
            flags.append("NO_USABLE_EVENTS")

        # Case 3: Stable pulse detected
        elif (
            d.pulse_confidence >= cfg.pulse_confidence_min
            and eps <= cfg.stable_rate_max
            and eps >= 0.5
        ):
            input_quality = "good"
            musical_state = "stable_pulse"
            flags.append("CLEAN_STABLE_PULSE")

        # Case 4: Stable pulse but unusual rate
        elif d.pulse_confidence >= cfg.pulse_confidence_min:
            input_quality = "usable"
            musical_state = "stable_pulse"

        # Case 5: Sparse hits
        elif eps <= cfg.sparse_rate_threshold:
            input_quality = "usable"
            musical_state = "sparse_hits"
            flags.append("SPARSE_INPUT")

        # Case 6: Dense / potentially spammy
        elif eps > cfg.dense_rate_threshold:
            input_quality = "weak"
            musical_state = "dense_activity"
            flags.append("DENSE_INPUT")
            flags.append("UNSTABLE_PULSE")

        # Case 7: Moderate density, no stable pulse
        elif eps > 2.0:
            input_quality = "usable"
            musical_state = "dense_activity"
            flags.append("DENSE_INPUT")

        # Case 8: Fallback
        else:
            input_quality = "weak"
            musical_state = "unstable"

        # ── Additional flags ────────────────────────────────
        if d.clipping_fraction > 0.05:
            flags.append("CLIPPING_RISK")
        if d.signal_rms < cfg.low_signal_rms:
            flags.append("LOW_SIGNAL")

        # ── Recommendation ──────────────────────────────────
        rec_map = {
            "stable_pulse": "Looks suitable for downstream pulse tracking.",
            "sparse_hits": "Events detected but too sparse for reliable pulse estimation.",
            "dense_activity": (
                "High event density.  Consider raising onset thresholds "
                "or checking for background noise."
            ),
            "silence": "No usable events found.  Check input gain or source.",
            "noise": _recommendation_for_noise(d),
            "unstable": "Events present but no stable pulse emerged.",
        }
        recommendation = rec_map.get(musical_state, "No specific recommendation.")

        return {
            "input_quality": input_quality,
            "musical_state": musical_state,
            "recommendation": recommendation,
            "summary_flags": flags,
        }

    # ── Report Formatting ──────────────────────────────────────────

    @staticmethod
    def format_report(result: DetectorResult) -> str:
        """Render a human-readable text report (ASCII-safe for Windows consoles)."""
        d = result.diagnostics
        lines: list[str] = []

        lines.append("=" * 62)
        lines.append("  DETECTOR DIAGNOSTICS REPORT")
        lines.append("=" * 62)
        lines.append(f"  Duration          : {d.duration_seconds:.3f} s")
        lines.append("")

        # ── Summary (Phase 4) ──────────────────────────────────────
        lines.append("-- Summary")
        lines.append(f"  Input quality    : {d.input_quality}")
        lines.append(f"  Musical state    : {d.musical_state}")
        lines.append(f"  Recommendation   : {d.recommendation}")
        if d.summary_flags:
            lines.append(f"  Flags            : {', '.join(d.summary_flags)}")
        lines.append("")

        lines.append("-- Rejection Breakdown")
        lines.append(f"  Noise gate        : {d.rejected_by_noise_gate}")
        lines.append(f"  Hum filter        : {d.rejected_by_hum}")
        lines.append(f"  Sustained filter  : {d.rejected_by_sustained}")
        lines.append("")

        lines.append("-- Events")
        lines.append(f"  Raw event count   : {d.event_count + d.rejected_count}")
        lines.append(f"  Accepted events   : {d.event_count}")
        lines.append(f"  Rejected (gate)   : {d.rejected_count}")
        lines.append(f"  Events / second   : {d.events_per_second:.1f}")
        lines.append("")

        lines.append("-- Energy")
        lines.append(f"  Avg strength      : {d.avg_strength:.3f}")
        lines.append(f"  Peak strength     : {d.peak_strength:.3f}")
        lines.append(f"  Avg energy        : {d.avg_energy:.3f}")
        lines.append(f"  Peak energy       : {d.peak_energy:.3f}")
        lines.append(f"  Signal RMS        : {d.signal_rms:.4f}")
        lines.append(f"  Noise floor (est) : {d.estimated_noise_floor:.4f}")
        lines.append(f"  Clipping fraction : {d.clipping_fraction:.3f}")
        lines.append("")

        lines.append("-- Density")
        lines.append(f"  Final density     : {d.density_final:.3f}")
        lines.append(f"  First-half mean   : {d.density_mean_first_half:.3f}")
        lines.append(f"  Second-half mean  : {d.density_mean_second_half:.3f}")
        lines.append(f"  Density trend     : {d.density_trend}")
        lines.append("")

        lines.append("-- Frequency Regions")
        if d.frequency_region_counts:
            for region, count in sorted(d.frequency_region_counts.items()):
                lines.append(f"  {region:<12} : {count}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("-- Timing Feel")
        if d.expected_bpm is not None:
            lines.append(f"  Expected BPM      : {d.expected_bpm:.1f}")
        lines.append(f"  Estimated BPM     : {d.pulse_bpm:.1f}" if d.pulse_bpm else "  Estimated BPM     : -")
        if d.expected_bpm is not None and d.pulse_bpm is not None:
            bpm_err = abs(d.expected_bpm - d.pulse_bpm)
            lines.append(f"  BPM error         : {bpm_err:.1f}")
        lines.append(f"  Median IOI        : {d.median_ioi_ms:.0f} ms")
        lines.append(f"  IOI spread        : {d.ioi_spread_ms:.0f} ms")
        lines.append(f"  Timing feel       : {d.timing_feel}")
        if d.timing_flags:
            lines.append(f"  Timing flags      : {', '.join(d.timing_flags)}")
        lines.append("")

        lines.append("-- Pulse Estimate")
        if d.pulse_bpm is not None:
            lines.append(f"  Best BPM          : {d.pulse_bpm:.1f}")
            lines.append(f"  Confidence        : {d.pulse_confidence:.3f}")
            lines.append(f"  Stability         : {d.pulse_stability}")
        else:
            lines.append("  (not enough events for pulse estimate)")
        lines.append("")

        lines.append("-- Warnings")
        if d.warnings:
            for w in d.warnings:
                lines.append(f"  !! {w}")
        else:
            lines.append("  (none)")
        lines.append("")

        lines.append("-- Event Timeline")
        if d.events:
            lines.append(
                f"  {'Time (s)':>10}  {'Strength':>8}  {'Region':>10}"
                f"  {'Energy':>8}  {'Density':>8}"
            )
            lines.append(f"  {'-' * 10}  {'-' * 8}  {'-' * 10}  {'-' * 8}  {'-' * 8}")
            for e in d.events[:50]:
                lines.append(
                    f"  {e.time_seconds:10.4f}"
                    f"  {e.strength:8.3f}"
                    f"  {e.frequency_region:>10}"
                    f"  {e.energy:8.3f}"
                    f"  {e.density:8.3f}"
                )
            if len(d.events) > 50:
                lines.append(f"  ... ({len(d.events) - 50} more events omitted)")
        else:
            lines.append("  (no events)")
        lines.append("")
        lines.append("=" * 62)
        lines.append("  END OF REPORT")
        lines.append("=" * 62)

        return "\n".join(lines)
