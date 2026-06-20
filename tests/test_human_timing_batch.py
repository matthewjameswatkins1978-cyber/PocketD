import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from demo_human_timing_batch import analyse_batch, analyse_case, build_batch_cases
from synthetic.noisy_pulse import HumanTimingProfile, make_human_pulse


def test_human_pulse_generation_is_deterministic() -> None:
    profile = HumanTimingProfile(
        name="normal_human",
        jitter_ms=25.0,
        drift_ms_per_beat=0.5,
        swing_ms=10.0,
    )

    first = make_human_pulse(bpm=118.0, bars=4, profile=profile, seed=42)
    second = make_human_pulse(bpm=118.0, bars=4, profile=profile, seed=42)

    assert first == second
    assert len(first) == 16


def test_batch_contains_multiple_tempos_and_profiles() -> None:
    cases = build_batch_cases()
    tempos = {case.bpm for case in cases}
    profiles = {case.profile.name for case in cases}

    assert len(tempos) >= 5
    assert len(profiles) >= 6
    assert len(cases) == len(tempos) * len(profiles)


def test_normal_human_timing_estimates_bpm_across_tempos() -> None:
    profile = HumanTimingProfile(name="normal_human", jitter_ms=25.0)
    cases = [
        case
        for case in build_batch_cases(
            tempos=(72.0, 90.0, 118.0, 140.0, 168.0),
            profiles=(profile,),
        )
    ]
    results = analyse_batch(cases)

    assert all(result.passed for result in results)
    assert max(abs(result.bpm_error) for result in results) <= 4.0


def test_messier_human_profiles_still_estimate_close_to_bpm() -> None:
    profiles = (
        HumanTimingProfile(name="loose", jitter_ms=45.0),
        HumanTimingProfile(name="pushes", jitter_ms=22.0, drift_ms_per_beat=-1.2),
        HumanTimingProfile(name="drags", jitter_ms=22.0, drift_ms_per_beat=1.2),
        HumanTimingProfile(name="lumpy_swing", jitter_ms=18.0, swing_ms=28.0),
        HumanTimingProfile(name="misses_one", jitter_ms=22.0, drop_every=9),
        HumanTimingProfile(name="extra_hits", jitter_ms=18.0, extra_offbeat_every=7),
    )
    cases = build_batch_cases(tempos=(90.0, 118.0, 140.0), profiles=profiles)
    results = analyse_batch(cases)

    assert all(result.passed for result in results)
    assert max(abs(result.bpm_error) for result in results) <= 4.0


def test_single_case_reports_jitter_and_onset_count() -> None:
    case = build_batch_cases(
        tempos=(118.0,),
        profiles=(HumanTimingProfile(name="loose", jitter_ms=45.0),),
    )[0]
    result = analyse_case(case)

    assert result.passed is True
    assert result.onset_count == 24
    assert result.interval_jitter_ms > 0.0
