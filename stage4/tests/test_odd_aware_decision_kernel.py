import json
from pathlib import Path

import numpy as np
import pytest

from stage4.dispatch.acceptance import passenger_acceptance
from stage4.dispatch.candidate_graph import SparseCandidateIndex, SpatialVehicle
from stage4.dispatch.exposure import (
    CumulativeExposureState,
    ExposureExcess,
    exposure_excess,
)
from stage4.dispatch.odd_aware_runner import EXPECTED, load_odd_config
from stage4.dispatch.solver import AssignmentArc, solve_lexicographic

ROOT = Path(__file__).resolve().parents[2]


def test_acceptance_gate_prunes_av_before_routing_and_is_stable():
    vehicles = [
        SpatialVehicle("HV", 1, "HV", 108.94, 34.26),
        SpatialVehicle("AV", 2, "AV", 108.9401, 34.26),
    ]
    index = SparseCandidateIndex(vehicles)
    rejected, _ = index.query(108.94, 34.26, 2000.0, 20, False)
    accepted, _ = index.query(108.94, 34.26, 2000.0, 20, True)
    assert [item[0].vehicle_type for item in rejected] == ["HV"]
    assert {item[0].vehicle_type for item in accepted} == {"HV", "AV"}
    assert passenger_acceptance("o1", 0.5, 7) == passenger_acceptance("o1", 0.5, 7)
    assert passenger_acceptance("o1", 1.0, 7).acceptance_source == "ALL_ACCEPT_AV"


def test_reference_envelope_excess_is_exact_and_unclipped():
    assert exposure_excess(0.5, 1.0, 3.25) == ExposureExcess(0.0, 0.0, 2.25)


def test_nonfinite_rho_removes_av_exposure_without_imputation():
    assert exposure_excess(np.nan, 1.0, 1.0) is None
    assert exposure_excess(1.0, np.inf, 1.0) is None


@pytest.mark.parametrize("family", ["static", "dynamic", "speed"])
def test_cumulative_gamma_selects_only_feasible_low_exposure(family):
    values = {"static": 0.0, "dynamic": 0.0, "speed": 0.0}
    low = {**values, family: 0.1}
    high = {**values, family: 0.9}
    arcs = [
        AssignmentArc(
            1,
            10,
            10.0,
            False,
            False,
            vehicle_type="AV",
            **{f"exposure_{name}": value for name, value in high.items()},
        ),
        AssignmentArc(
            1,
            20,
            20.0,
            False,
            False,
            vehicle_type="AV",
            **{f"exposure_{name}": value for name, value in low.items()},
        ),
    ]
    gammas = {"static": None, "dynamic": None, "speed": None, family: 0.5}
    result = solve_lexicographic(
        arcs, exposure_state=CumulativeExposureState(), gammas=gammas
    )
    assert result.selected_indices == (1,)
    assert result.enabled_gamma_constraint_count == 1


def test_cumulative_exposure_state_persists_across_epochs():
    state = CumulativeExposureState()
    state.update([ExposureExcess(0.2, 0.1, 0.0)])
    state.update([ExposureExcess(0.4, 0.3, 0.2)])
    assert state.av_assignments == 2
    assert state.static == pytest.approx(0.6)
    assert state.dynamic == pytest.approx(0.4)
    assert state.mean("speed") == pytest.approx(0.1)


def test_cost_level_breaks_tie_without_changing_higher_optima():
    arcs = [
        AssignmentArc(1, 10, 30.0, True, True, vehicle_type="HV", operating_cost=80),
        AssignmentArc(2, 10, 30.0, True, True, vehicle_type="AV", operating_cost=40),
    ]
    result = solve_lexicographic(arcs, cost_level_enabled=True)
    assert result.selected_indices == (1,)
    assert (
        result.critical_matched,
        result.total_matched,
        result.carry_over_matched,
    ) == (
        1,
        1,
        1,
    )
    assert result.cost_level_solved


def test_neutral_config_disables_gamma_and_cost():
    config = load_odd_config(ROOT, None)
    assert config["passenger_acceptance_rate"] == 1.0
    assert all(
        config[f"gamma_{name}"] is None for name in ("static", "dynamic", "speed")
    )
    assert config["cost_level_enabled"] is False
    assert config["pickup_cost_epsilon"] == 0.0


def test_neutral_canonical_product_reproduces_s3_aggregate_when_present():
    path = ROOT / "stage4/output/odd_aware_decision_kernel/kernel_summary.json"
    if not path.is_file():
        pytest.skip("canonical run is executed once after focused unit checks")
    summary = json.loads(path.read_text(encoding="utf-8"))
    assert summary["canonical_reproduction"] == EXPECTED
    assert summary["canonical_reproduction_pass"] is True
