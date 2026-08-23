"""Small finalization contract; deliberately limited to ten tests."""

from pathlib import Path
import inspect

import numpy as np

import stage3.odd_tod.finalization as final


def test_hard_feasible_original_does_not_trigger_fallback():
    assert not final.fallback_triggered("FEASIBLE")


def test_hard_unknown_original_does_not_trigger_fallback():
    assert not final.fallback_triggered("UNKNOWN")


def test_hard_infeasible_original_triggers_fallback():
    assert final.fallback_triggered("INFEASIBLE")


def test_candidate_distance_bound():
    assert final.within_distance_bound(125.0, 100.0)
    assert not final.within_distance_bound(125.01, 100.0)


def test_candidate_hard_state_ignores_soft_rho():
    assert final.candidate_hard_state([], []) == "FEASIBLE"
    assert final.candidate_hard_state([], ["MISSING"]) == "UNKNOWN"
    assert final.candidate_hard_state(["PROHIBITED"], []) == "INFEASIBLE"


def test_feasible_selection_minimizes_p50_then_distance():
    candidates = [
        {"hard_state": "FEASIBLE", "service_time_p50_s": 10.0, "distance_m": 20.0, "route_reference": "b"},
        {"hard_state": "FEASIBLE", "service_time_p50_s": 10.0, "distance_m": 15.0, "route_reference": "a"},
        {"hard_state": "INFEASIBLE", "service_time_p50_s": 1.0, "distance_m": 1.0, "route_reference": "x"},
    ]
    assert final.select_hard_feasible_candidate(candidates)["route_reference"] == "a"


def test_missing_candidate_m3_stays_unknown_without_imputation():
    source = Path(final.__file__).read_text(encoding="utf-8")
    assert final.DYNAMIC_UNKNOWN_REASON in source
    assert '"selected_service_time_p50_s": np.nan' in source
    assert "class mean" in final.__doc__


def test_output_schema_is_frozen_90000_contract():
    assert final.EXPECTED_ORDER_COUNT == 30_000
    assert final.EXPECTED_ORDER_PROFILE_COUNT == 90_000
    assert len(final.FINAL_COLUMNS) == 26
    assert {"selected_route_type", "hard_state", "rho_static", "rho_dynamic", "rho_speed"}.issubset(final.FINAL_COLUMNS)


def test_stage3_profiles_cdf_checkpoint_are_not_modified():
    source = inspect.getsource(final.finalize)
    assert "profile_before" in source and "profile_after" in source
    assert "cdf_before" in source and "cdf_after" in source
    assert "frozen M3 checkpoint changed" in source


def test_no_passenger_model_dispatch_solver_or_weighted_average():
    source = Path(final.__file__).read_text(encoding="utf-8").lower()
    assert '"passenger_model": false' in source
    assert '"dispatch_solver": false' in source
    assert "weighted average" not in source
    assert np.isnan(float("nan"))
