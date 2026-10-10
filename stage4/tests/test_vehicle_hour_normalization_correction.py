import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage4.dispatch.fleet_normalization import (
    FLEET_REL,
    build_fleet_scenario,
    exact_baseline_vehicle_hours,
)
from stage4.dispatch.parameterization_diagnostics import (
    FAMILIES,
    fleet_vehicle_hour_scenarios,
    gamma_reference_regimes,
)

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2016-10-31 08:00:00", tz="Asia/Shanghai")
END = START + pd.Timedelta(hours=3)
OUTPUT = ROOT / "stage4/output/vehicle_hour_normalization_correction"


def _scenario(q_a: float):
    return build_fleet_scenario(
        ROOT,
        benchmark_start=START,
        simulation_end=END,
        requested_q_a=q_a,
        seed=20260824,
    )


def test_exact_baseline_equals_total_frozen_session_duration():
    template = pd.read_parquet(
        ROOT / FLEET_REL,
        columns=["availability_start_time", "availability_end_time"],
    )
    expected = (
        pd.to_datetime(template["availability_end_time"], utc=True)
        - pd.to_datetime(template["availability_start_time"], utc=True)
    ).dt.total_seconds().sum() / 3600.0
    assert exact_baseline_vehicle_hours(ROOT) == pytest.approx(expected, abs=1e-9)


def test_q0_selects_all_hv_sessions_and_zero_avs():
    scenario = _scenario(0.0)
    assert scenario.accounting["av_count"] == 0
    expected_hv_sessions = len(
        pd.read_parquet(ROOT / FLEET_REL, columns=["source_session_id"])
    )
    assert scenario.accounting["selected_hv_session_count"] == expected_hv_sessions
    assert scenario.accounting["achieved_hv_vehicle_hours"] == pytest.approx(
        scenario.accounting["h_base_exact"], abs=1e-9
    )
    assert scenario.accounting["vehicle_hour_error_pct"] == 0.0


def test_corrected_q025_uses_128_avs():
    scenario = _scenario(0.25)
    assert scenario.accounting["av_count"] == 128
    assert scenario.accounting["vehicle_hour_error_pct"] <= 2.0


def test_q1_negative_raw_residual_uses_zero_hv_sessions():
    scenario = _scenario(1.0)
    assert scenario.accounting["raw_hv_residual_vehicle_hours"] < 0.0
    assert scenario.accounting["target_hv_vehicle_hours"] == 0.0
    assert scenario.accounting["selected_hv_session_count"] == 0


def test_six_rows_share_exact_denominator_and_keep_bin_equivalent_separate():
    fleet = fleet_vehicle_hour_scenarios(ROOT)
    assert len(fleet) == 6
    assert fleet["H_base_exact"].nunique() == 1
    assert fleet["H_base_15min_equivalent"].nunique() == 1
    assert not np.isclose(
        fleet.iloc[0]["H_base_exact"], fleet.iloc[0]["H_base_15min_equivalent"]
    )
    assert (fleet["HV_vehicle_hour_error_pct"] <= 2.0).all()


def test_corrected_s4_neutral_reproduces_corrected_s3_when_products_exist():
    s3_path = OUTPUT / "corrected_q025_s3_summary.json"
    s4_path = OUTPUT / "corrected_q025_s4_summary.json"
    if not s3_path.is_file() or not s4_path.is_file():
        pytest.skip("two bounded corrected neutral runs execute after unit checks")
    s3 = json.loads(s3_path.read_text(encoding="utf-8"))
    s4 = json.loads(s4_path.read_text(encoding="utf-8"))
    fields = (
        "requests",
        "matched",
        "completed",
        "patience_expired",
        "first_window_matched",
        "carry_over_recovered",
        "critical_matched",
    )
    assert {field: s3[field] for field in fields} == {
        field: s4[field] for field in fields
    }


def test_gamma_refresh_matches_corrected_neutral_trajectory_when_product_exists():
    path_file = OUTPUT / "corrected_neutral_exposure_path.parquet"
    gamma_file = OUTPUT / "corrected_gamma_reference_regimes.json"
    if not path_file.is_file() or not gamma_file.is_file():
        pytest.skip("corrected Gamma products execute after unit checks")
    path = pd.read_parquet(path_file)
    observed = json.loads(gamma_file.read_text(encoding="utf-8"))
    expected = gamma_reference_regimes(path)
    for family in FAMILIES:
        assert observed["families"][family]["NEUTRAL_FINAL_MEAN"] == pytest.approx(
            expected["families"][family]["NEUTRAL_FINAL_MEAN"]
        )
        assert observed["families"][family]["NEUTRAL_PATH_ENVELOPE"] == pytest.approx(
            expected["families"][family]["NEUTRAL_PATH_ENVELOPE"]
        )
