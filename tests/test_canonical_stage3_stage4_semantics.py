import json
from pathlib import Path

from stage3.scripts.build_stage3_canonical_smoke import TARGETS
from stage4.scripts.run_canonical_safe_o0_smoke import radius


def test_stage3_expected_names_are_not_quantile_aliases():
    schema = json.loads(Path("stage3/config/stage3_condition_vector_v2.schema.json").read_text())
    assert schema["expected_value_policy"].startswith("continuous regression")
    for target in TARGETS:
        assert f"{target}_expected" in schema["roles"]


def test_stage4_av_profile_is_exogenous():
    profile = json.loads(Path("stage4/config/canonical_smoke_av_profile.json").read_text())
    assert "not_test_day_calibration" in profile["threshold_source"]
    assert profile["status"] == "engineering_scenario_prior"


def test_dynamic_radius_schedule_is_monotone():
    values = [radius(wait)[1] for wait in (0, 119, 120, 239, 240, 359, 360, 480)]
    assert values == sorted(values)
    assert values[0] == 2000
    assert values[-1] == 6000
