from types import SimpleNamespace

import pandas as pd
import pytest

from stage4.dispatch.candidate_graph import SparseCandidateIndex, SpatialVehicle
from stage4.dispatch.gate_diagnostics import (
    GATE_COLUMNS,
    LOSS_COLUMNS,
    aggregate_gate_epochs,
    empty_gate_counts,
    evidence_contract_complete,
    structural_reason,
    validate_gate_counts,
)


def _request(**updates):
    values = {
        "selected_route_type": "ORIGINAL",
        "hard_state": "FEASIBLE",
        "evidence_complete": True,
        "predicted_service_time_s": 600.0,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_structural_reasons_are_mutually_exclusive() -> None:
    assert structural_reason(_request()) is None
    assert structural_reason(_request(selected_route_type="NONE")) == "NO_SELECTED_ROUTE"
    assert structural_reason(_request(hard_state="INFEASIBLE")) == "HARD_INFEASIBLE"
    assert structural_reason(_request(hard_state="UNKNOWN")) == "HARD_UNKNOWN"


def test_evidence_contract_requires_flag_exposure_and_positive_service_time() -> None:
    exposure = SimpleNamespace(static=0.0, dynamic=0.0, speed=0.0)
    assert evidence_contract_complete(_request(), exposure)
    assert not evidence_contract_complete(_request(evidence_complete=False), exposure)
    assert not evidence_contract_complete(_request(), None)
    assert not evidence_contract_complete(_request(predicted_service_time_s=float("nan")), exposure)


def test_gate_counts_reconcile_and_aggregate_to_15_minutes() -> None:
    row = empty_gate_counts()
    row.update(
        {
            "gate_av_n0_spatial": 100,
            "gate_av_n1_passenger_compatible": 70,
            "gate_av_n2_structurally_ready": 50,
            "gate_av_n3_evidence_complete": 40,
            "gate_av_n3a_shared_topk": 20,
            "gate_av_n3b_route_returned": 18,
            "gate_av_n4_pickup_within_patience": 12,
            "gate_av_n5_solver_eligible": 12,
            "gate_av_n6_selected": 2,
            "gate_av_loss_acceptance": 30,
            "gate_av_loss_no_selected_route": 5,
            "gate_av_loss_hard_infeasible": 4,
            "gate_av_loss_hard_unknown": 11,
            "gate_av_loss_evidence_incomplete": 10,
            "gate_av_loss_shared_topk": 20,
            "gate_av_loss_routing_failure": 2,
            "gate_av_loss_patience": 6,
            "gate_av_loss_other_arc_condition": 0,
            "gate_av_loss_dispatch_competition": 10,
        }
    )
    validate_gate_counts(row)
    epoch = pd.DataFrame(
        [
            {"timestamp": "2016-10-31T08:01:00+08:00", **row},
            {"timestamp": "2016-10-31T08:14:30+08:00", **row},
        ]
    )
    binned = aggregate_gate_epochs(epoch)
    assert len(binned) == 1
    assert int(binned.iloc[0]["gate_av_n0_spatial"]) == 200
    assert str(binned.iloc[0]["time_bin_start"].tz) == "Asia/Shanghai"


def test_invalid_non_nested_gate_counts_fail_closed() -> None:
    row = empty_gate_counts()
    row["gate_av_n1_passenger_compatible"] = 1
    with pytest.raises(ValueError, match="not nested"):
        validate_gate_counts(row)


def test_shadow_spatial_count_does_not_change_candidate_query() -> None:
    vehicles = [
        SpatialVehicle("HV", 1, "HV", 108.94, 34.26),
        SpatialVehicle("AV", 2, "AV", 108.9401, 34.26),
        SpatialVehicle("AV2", 3, "AV", 108.9402, 34.26),
    ]
    index = SparseCandidateIndex(vehicles)
    before, before_count = index.query(108.94, 34.26, 2_000.0, 20, True)
    shadow_count = index.count_vehicle_type_within(108.94, 34.26, 2_000.0, "AV")
    after, after_count = index.query(108.94, 34.26, 2_000.0, 20, True)
    assert shadow_count == 2
    assert before_count == after_count
    assert before == after
    assert set(GATE_COLUMNS).isdisjoint(LOSS_COLUMNS)
