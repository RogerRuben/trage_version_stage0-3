from __future__ import annotations

import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_main_profiles_are_exogenous_and_monotone() -> None:
    doc = json.loads((REPO_ROOT / "stage4/config/vehicle_capability_profiles.json").read_text(encoding="utf-8"))
    names = doc["threshold_policy"]["main_profiles"]
    assert names == ["conservative_av", "moderate_av", "mature_av"]
    profiles = doc["profiles"]
    for name in names:
        profile = profiles[name]
        assert profile["profile_role"] == "main_scenario"
        assert profile["threshold_source_type"] == "explicit_exogenous_scenario"
        assert profile["test_day_used_for_thresholds"] is False
    for dimension in ["lcs", "pmis", "rts", "iis"]:
        ceilings = [profiles[name]["dimension_hard_threshold"][dimension] for name in names]
        assert ceilings == sorted(ceilings)
    uncertainty = [profiles[name]["uncertainty_tolerance"] for name in names]
    assert uncertainty == sorted(uncertainty)


def test_full_day_calibrated_profile_is_sensitivity_only() -> None:
    doc = json.loads((REPO_ROOT / "stage4/config/vehicle_capability_profiles.json").read_text(encoding="utf-8"))
    profile = doc["profiles"]["full_day_calibrated_sensitivity_av"]
    assert profile["profile_role"] == "sensitivity_only_test_day_derived"
    assert profile["test_day_used_for_thresholds"] is True
    assert profile["prohibited_as_main_scenario"] is True
