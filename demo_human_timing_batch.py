"""Batch demo: recognise BPM from human-ish pulse timing.

This expands the synthetic proof from a perfect click to batches of imperfect
human pulse input across tempos and timing feels.

Run:
    python demo_human_timing_batch.py
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass

from audio.onset_detector import detect_onsets_from_events
from pulse_tracker import estimate_tempo
from synthetic.noisy_pulse import HumanTimingProfile, make_human_pulse


@dataclass(frozen=True)
class HumanTimingCase:
    bpm: float
    profile: HumanTimingProfile
    seed: int
    bars: int = 6


@dataclass(frozen=True)
class HumanTimingResult:
    case: HumanTimingCase
    estimated_bpm: float
    bpm_error: float
    onset_count: int
    interval_jitter_ms: float
    passed: bool


DEFAULT_PROFILES: tuple[HumanTimingProfile, ...] = (
    HumanTimingProfile(name="tight", jitter_ms=10.0),
    HumanTimingProfile(name="normal_human", jitter_ms=25.0),
    HumanTimingProfile(name="loose", jitter_ms=45.0),
    HumanTimingProfile(name="pushes", jitter_ms=22.0, drift_ms_per_beat=-1.2),
    HumanTimingProfile(name="drags", jitter_ms=22.0, drift_ms_per_beat=1.2),
    HumanTimingProfile(name="lumpy_swing", jitter_ms=18.0, swing_ms=28.0),
    HumanTimingProfile(name="misses_one", jitter_ms=22.0, drop_every=9),
    HumanTimingProfile(name="extra_hits", jitter_ms=18.0, extra_offbeat_every=7),
)

DEFAULT_TEMPOS: tuple[float, ...] = (72.0, 90.0, 118.0, 140.0, 168.0)


def build_batch_cases(
    tempos: tuple[float, ...] = DEFAULT_TEMPOS,
    profiles: tuple[HumanTimingProfile, ...] = DEFAULT_PROFILES,
) -> list[HumanTimingCase]:
    """Build a deterministic cross-product of tempos and timing profiles."""
    cases: list[HumanTimingCase] = []
    seed = 100
    for bpm in tempos:
        for profile in profiles:
            cases.append(HumanTimingCase(bpm=bpm, profile=profile, seed=seed))
            seed += 1
    return cases


def analyse_case(
    case: HumanTimingCase,
    bpm_tolerance: float = 4.0,
) -> HumanTimingResult:
    """Run one human timing case through onset detection and tempo estimation."""
    pulses = make_human_pulse(
        bpm=case.bpm,
        bars=case.bars,
        profile=case.profile,
        seed=case.seed,
    )
    onsets = detect_onsets_from_events(pulses, min_interval=0.05)
    onset_times = [event.time_seconds for event in onsets]
    estimated_bpm = estimate_tempo(onset_times)
    bpm_error = estimated_bpm - case.bpm
    interval_jitter_ms = _interval_jitter_ms(onset_times)

    return HumanTimingResult(
        case=case,
        estimated_bpm=estimated_bpm,
        bpm_error=bpm_error,
        onset_count=len(onset_times),
        interval_jitter_ms=interval_jitter_ms,
        passed=abs(bpm_error) <= bpm_tolerance,
    )


def analyse_batch(cases: list[HumanTimingCase]) -> list[HumanTimingResult]:
    """Analyse all cases in a batch."""
    return [analyse_case(case) for case in cases]


def _interval_jitter_ms(onset_times: list[float]) -> float:
    """Return median absolute interval deviation in milliseconds."""
    intervals = [b - a for a, b in zip(onset_times, onset_times[1:]) if b > a]
    if len(intervals) < 2:
        return 0.0
    median_interval = statistics.median(intervals)
    deviations = [abs(interval - median_interval) for interval in intervals]
    return statistics.median(deviations) * 1000.0


def print_batch_report(results: list[HumanTimingResult]) -> None:
    """Print a compact pass/fail report for the batch."""
    print("\nHuman timing BPM recognition batch")
    print("=" * 80)
    print(
        f"{'Tempo':>6s}  {'Profile':<14s}  {'Est':>6s}  {'Err':>7s}  "
        f"{'Jitter':>8s}  {'Onsets':>6s}  Result"
    )
    print("-" * 80)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(
            f"{result.case.bpm:6.1f}  {result.case.profile.name:<14s}  "
            f"{result.estimated_bpm:6.1f}  {result.bpm_error:+7.2f}  "
            f"{result.interval_jitter_ms:7.1f}ms  {result.onset_count:6d}  "
            f"{status}"
        )

    passed = sum(1 for result in results if result.passed)
    max_error = max((abs(result.bpm_error) for result in results), default=0.0)
    print("-" * 80)
    print(f"Passed {passed}/{len(results)} cases. Worst BPM error: {max_error:.2f}")


def main() -> int:
    results = analyse_batch(build_batch_cases())
    print_batch_report(results)
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
