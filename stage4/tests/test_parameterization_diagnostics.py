from pathlib import Path

import numpy as np
import pandas as pd

from stage4.dispatch.parameterization_diagnostics import (
    HV_TOLERANCE_PCT,
    fleet_vehicle_hour_scenarios,
    gamma_reference_regimes,
    exposure_values,
)

ROOT = Path(__file__).resolve().parents[2]


def _tiny_path() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "av_assignment_rank": [1, 2, 3],
            "assignment_time": pd.date_range(
                "2016-10-31", periods=3, freq="min", tz="UTC"
            ),
            "cumulative_mean_static_excess": [0.2, 0.1, 0.3],
            "cumulative_mean_dynamic_excess": [0.0, 0.4, 0.2],
            "cumulative_mean_speed_excess": [0.1, 0.05, 0.0],
        }
    )


def test_exposure_formula_preserves_zero_mass():
    result = exposure_values(pd.Series([0.2, 1.0, 1.5, 4.0]))
    assert result.tolist() == [0.0, 0.0, 0.5, 3.0]
    assert result.eq(0.0).sum() == 2


def test_gamma_mean_is_final_cumulative_mean():
    result = gamma_reference_regimes(_tiny_path())
    assert result["families"]["static"]["NEUTRAL_FINAL_MEAN"] == 0.3
    assert result["families"]["dynamic"]["NEUTRAL_FINAL_MEAN"] == 0.2


def test_gamma_path_is_maximum_cumulative_mean():
    result = gamma_reference_regimes(_tiny_path())
    assert result["families"]["static"]["NEUTRAL_PATH_ENVELOPE"] == 0.3
    assert result["families"]["dynamic"]["NEUTRAL_PATH_ENVELOPE"] == 0.4


def test_q025_reproduces_frozen_av_count_and_share():
    fleet = fleet_vehicle_hour_scenarios(ROOT)
    row = fleet.loc[np.isclose(fleet["requested_q_A"], 0.25)].iloc[0]
    assert int(row["AV_vehicle_count"]) == 150
    assert np.isclose(row["achieved_q_A"], 0.2505306378092488)


def test_q0_has_zero_avs():
    fleet = fleet_vehicle_hour_scenarios(ROOT)
    row = fleet.loc[np.isclose(fleet["requested_q_A"], 0.0)].iloc[0]
    assert int(row["AV_vehicle_count"]) == 0
    assert row["achieved_AV_vehicle_hours"] == 0.0


def test_applicable_hv_accounting_stays_within_frozen_tolerance():
    fleet = fleet_vehicle_hour_scenarios(ROOT)
    calibrated = fleet["requested_q_A"].isin([0.25, 0.50, 0.75])
    assert (
        fleet.loc[calibrated, "HV_vehicle_hour_error_pct"] <= HV_TOLERANCE_PCT
    ).all()
    q1 = fleet.loc[np.isclose(fleet["requested_q_A"], 1.0)].iloc[0]
    assert q1["target_HV_vehicle_hours"] < 0.0
    assert int(q1["selected_HV_session_count"]) == 0
    assert pd.isna(q1["HV_vehicle_hour_error_pct"])
