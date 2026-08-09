from __future__ import annotations

from stage2.v5_2.phase_c_report import classify_phase_c_direction


def _decision(*, adopt: bool, wins: int, mean: float) -> dict[str, object]:
    return {
        "adopt": adopt,
        "low_support_target_wins": wins,
        "low_support_mean_relative_improvement": mean,
    }


def test_phase_c_pass_requires_frozen_adoption_gate() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=True, wins=3, mean=0.021), pace_guard_pass=True,
    ) == "PASS"


def test_phase_c_weak_is_metric_derived_not_hardcoded() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=3, mean=0.006), pace_guard_pass=True,
    ) == "WEAK"


def test_phase_c_fail_for_negative_or_unstable_direction() -> None:
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=2, mean=-0.01), pace_guard_pass=True,
    ) == "FAIL"
    assert classify_phase_c_direction(
        _decision(adopt=False, wins=3, mean=0.01), pace_guard_pass=False,
    ) == "FAIL"
