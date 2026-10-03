"""Minimal scientific contract for the S4 v2 Stage4 interface."""

from pathlib import Path
import inspect

import numpy as np

import stage3.odd_tod.operational_suitability as v2


def test_frozen_stage3_profile_not_changed():
    source = inspect.getsource(v2.build_operational_suitability)
    assert "profile_sha256_before" in source
    assert "profile_sha256_after" in source
    assert "frozen Stage3 profile changed" in source


def test_m3_checkpoint_not_changed():
    assert len(v2.M3_SHA256) == 64
    source = inspect.getsource(v2._input_audit)
    assert "frozen M3 checkpoint changed" in source


def test_no_future_information():
    source = inspect.getsource(v2._input_audit)
    assert '"decision_time_only"' in source
    assert '"realized_future_time_used"' in source
    assert '"realized_targets_persisted"' in source


def test_test31_only():
    assert v2.TEST_DATE == "20161031"
    assert v2.EXPECTED_ORDER_COUNT == 30_000
    assert v2.EXPECTED_ORDER_PROFILE_COUNT == 90_000


def test_no_route_replanning():
    source = Path(v2.__file__).read_text(encoding="utf-8").lower()
    assert "valhalla" not in source
    assert "k_shortest" not in source
    assert '"route_replanning": false' in source


def test_hard_state_logic():
    assert v2.hard_state_from_reasons(["KNOWN"], ["MISSING"]) == "INFEASIBLE"
    assert v2.hard_state_from_reasons([], ["MISSING"]) == "UNKNOWN"
    assert v2.hard_state_from_reasons([], []) == "FEASIBLE"


def test_rho_formula():
    assert v2.utilization_ratio(8.0, 4.0) == 2.0
    assert v2.overall_utilization(0.8, 1.2, 0.5) == 1.2
    assert np.isnan(v2.overall_utilization(0.8, np.nan, 0.5))


def test_no_weighted_average():
    source = inspect.getsource(v2.overall_utilization)
    assert ".max()" in source
    assert "average" not in source
    assert "dot" not in source


def test_output_schema():
    required = {
        "order_id", "profile_id", "hard_state", "rho_static", "rho_dynamic",
        "rho_speed", "rho_overall", "static_A_ratio", "static_M_ratio",
        "static_D_ratio", "static_L_ratio", "dynamic_12_ratios", "reason_codes",
        "original_route",
    }
    assert required.issubset(v2.OUTPUT_COLUMNS)
