from pathlib import Path

import pandas as pd

from stage4.dispatch.candidate_graph import (
    SparseCandidateIndex,
    SparseValhallaMatrixAdapter,
    SpatialVehicle,
    search_radius_m,
)
from stage4.dispatch.fleet_normalization import build_fleet_scenario
from stage4.dispatch.rolling_or_control import patience_expired, patience_feasible
from stage4.dispatch.solver import AssignmentArc, solve_lexicographic

ROOT = Path(__file__).resolve().parents[2]
START = pd.Timestamp("2016-10-31 08:00:00", tz="Asia/Shanghai")


def test_first_attempt_uses_two_kilometre_radius():
    assert search_radius_m(0) == 2000.0


def test_carry_over_expands_radius_next_epoch():
    assert search_radius_m(1) == 3000.0
    assert search_radius_m(99) == 8000.0


def test_arc_beyond_remaining_patience_is_removed():
    assert patience_feasible(90.0, 90.0)
    assert not patience_feasible(90.1, 90.0)


def test_patience_expired_request_exits_at_deadline():
    assert not patience_expired(299, 300)
    assert patience_expired(300, 300)


def test_active_vehicle_hour_accounting_is_within_tolerance():
    scenario = build_fleet_scenario(
        ROOT,
        benchmark_start=START,
        simulation_end=START + pd.Timedelta(hours=3),
        requested_q_a=0.25,
        seed=20260824,
        max_hv_hour_error_pct=2.0,
    )
    assert scenario.accounting["vehicle_hour_error_pct"] <= 2.0


def test_av_full_horizon_and_hv_session_windows_are_distinct():
    end = START + pd.Timedelta(hours=3)
    scenario = build_fleet_scenario(
        ROOT,
        benchmark_start=START,
        simulation_end=end,
        requested_q_a=0.25,
        seed=20260824,
    )
    av = [item for item in scenario.native_fixtures if item.vehicle_type == "AV"]
    hv = [item for item in scenario.native_fixtures if item.vehicle_type == "HV"]
    assert av and hv
    assert all(
        item.availability_start_time == START and item.availability_end_time == end
        for item in av
    )
    assert any(
        item.availability_start_time != START or item.availability_end_time != end
        for item in hv
    )


def test_sparse_top_k_never_exceeds_twenty():
    vehicles = [
        SpatialVehicle(f"HV_{i:02d}", i, "HV", 108.94 + i * 0.00001, 34.26)
        for i in range(30)
    ]
    candidates, spatial_count = SparseCandidateIndex(vehicles).query(
        108.94, 34.26, 2000.0, 20, False
    )
    assert spatial_count == 30
    assert len(candidates) == 20

    class MatrixFailsRouteSucceeds:
        def matrix(self, request):
            del request
            return {"sources_to_targets": [[{"time": None, "distance": None}]]}

        def route(self, request):
            del request
            return {
                "trip": {
                    "status": 0,
                    "legs": [{}],
                    "summary": {"time": 45.0, "length": 0.5},
                }
            }

    adapter = SparseValhallaMatrixAdapter(ROOT, actor=MatrixFailsRouteSucceeds())
    fallback = adapter.estimate_many([vehicles[0]], 108.95, 34.27, START)
    assert vehicles[0].native_vehicle_id in fallback
    assert adapter.matrix_failed_arcs == 1
    assert adapter.route_fallback_successes == 1
    assert adapter.routing_failures == 0


def test_exact_lexicographic_solver_protects_critical_request():
    arcs = [
        AssignmentArc(1, 10, 100.0, True, True),
        AssignmentArc(1, 20, 1.0, False, False),
    ]
    result = solve_lexicographic(arcs)
    assert result.selected_indices == (0,)
    assert result.critical_matched == 1
    assert result.total_matched == 1
