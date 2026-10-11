"""Bounded integration checks; no simulation, routing Actor or inference."""
from collections import Counter, OrderedDict
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd

from stage4.analysis.capability_chain_instances import join_movement_phase
from stage4.analysis.scheme_a_two_scale_experiment import load_protocol
from stage4.analysis.type_conditioned_reposition_reference import aggregate_targets
from stage4.dispatch.request_decision_diagnostics import RequestDecisionDiagnostics
from stage4.dispatch.type_conditioned_movement import TypeConditionedHistoricalDeficitMovementManager as Manager
from stage4.dispatch.solver import AssignmentArc

ROOT = Path(__file__).resolve().parents[2]


def test_typed_counts_use_current_final_graph_not_old_nearby_opportunities():
    d = RequestDecisionDiagnostics(typed_candidates=True, deadline_semantics="REQUEST_TO_ASSIGNMENT")
    d.activate(1, "one", 0., 300., 0.)
    d.round(1, 0., 2, 1, 1)
    d.validated(1, "AV", 100.)
    d.round(1, 30., 2, 1, 1)
    d.validated(1, "HV", 50.)
    d.finish(1, 30., "ASSIGNED", vehicle_type="HV")
    d.activate(2, "two", 0., 300., 0.)
    d.round(2, 0., 2, 1, 1)
    d.validated(2, "AV", 90.)
    d.validated(2, "HV", 60.)
    d.reject(2, "UTURN_OUTSIDE_RESEARCH_PROFILE", stage="PICKUP_ROUTE", vehicle_type="AV")
    d.reject_stage(2, "DEPARTURE_JOIN", "AV", ["UTURN_OUTSIDE_RESEARCH_PROFILE"])
    d.HV_alternatives(2, idle=1, serving_other=0, moving=0)
    d.finish(2, 0., "ASSIGNED", vehicle_type="HV")
    rows = list(d.rows())
    assert rows[0]["typed_last_final_AV_count"] == 0
    assert rows[0]["typed_HV_assigned_with_final_AV_same_epoch"] is False
    assert rows[1]["typed_HV_assigned_with_final_AV_same_epoch"] is True
    assert d.summary()["HV_assigned_with_unoccupied_valid_AV"] == 1
    assert json.loads(rows[1]["rejection_counts_json"])["UTURN_OUTSIDE_RESEARCH_PROFILE"] == 1
    assert len(json.loads(rows[1]["typed_rejection_counts_json"])) == 2
    legacy = RequestDecisionDiagnostics()
    legacy.activate(3, "legacy", 0., 300., 0.)
    legacy.validated(3, "HV", 10.)
    assert not any(key.startswith("typed_") for key in next(legacy.rows()))


def test_join_phase_retains_multi_interface_and_repeated_occurrence_ambiguity():
    record = lambda a, b, internal=(): dict(incoming_stage3_edge_uid=a,
        outgoing_stage3_edge_uid=b, internal_stage3_edge_uids=json.dumps(list(internal)))
    assert join_movement_phase(["a", "b", "b", "c"], record("a", "b"), 1, 2, 1) == "DEPARTURE_JOIN"
    assert join_movement_phase(["a", "b", "b", "c"], record("b", "c"), 1, 2, 1) == "PICKUP_CUSTOMER_JOIN"
    assert join_movement_phase(["a", "i", "b"], record("a", "b", ["i"]), 1, 1, 1) == "MULTI_INTERFACE_JOIN"
    assert join_movement_phase(["a", "b", "a", "b"], record("a", "b"), 1, 2, 1) == "AMBIGUOUS_JOIN_BOUNDARY"


def test_exact_Stage1_target_join_does_not_turn_missing_nodes_into_C_failure():
    templates = pd.DataFrame(dict(date=["20161010"] * 3, order_id=["a", "b", "c"],
        compatible_C=[True, False, True], compatible_M=[True] * 3, compatible_A=[True] * 3))
    nodes = pd.DataFrame(dict(date=["20161010"] * 3, order_id=["a", "b", "c"], start_node=[10., 20., None],
        departure_time=[pd.Timestamp("2016-10-10T00:00:00+08:00").timestamp()] * 3, stage1_core_eligible=[True] * 3))
    reference = pd.DataFrame(dict(time_bin_index=[0, 0], node_id=[10, 20], lon_wgs84=[108.9, 108.91], lat_wgs84=[34.2, 34.2]))
    result, facts = aggregate_targets(templates, nodes, reference)
    assert facts["missing_or_invalid_start_node"] == 1
    assert facts["profiles"]["C"]["source_compatible_orders"] == 2
    assert facts["profiles"]["C"]["mapped_compatible_orders"] == 1
    assert result.loc[result.profile_id.eq("C"), "node_id"].tolist() == [10]
    assert result.groupby(["profile_id", "time_bin_index"]).demand_share.sum().eq(1.).all()


def test_typed_targets_use_full_physical_occupancy_support_and_one_shared_move_cap():
    A, B = (108.9, 34.2), (108.901, 34.2)
    base = pd.DataFrame(dict(node_id=[10, 20], lon_wgs84=[A[0], B[0]], lat_wgs84=[A[1], B[1]], demand_share=[.9, .1]))
    typed = base.iloc[[1]].copy()
    typed["demand_share"] = 1.
    vehicles = {0: NS(fixture=NS(vehicle_type="AV", native_id=0, vehicle_id="v0"),
                     native_vehicle=NS(pos=A, assign_vehicle_plan=lambda *a: None)),
                1: NS(fixture=NS(vehicle_type="HV", native_id=1, vehicle_id="v1"),
                     native_vehicle=NS(pos=B, assign_vehicle_plan=lambda *a: None)),
                2: NS(fixture=NS(vehicle_type="AV", native_id=2, vehicle_id="v2"),
                     native_vehicle=NS(pos=A, assign_vehicle_plan=lambda *a: None))}
    control = NS(config={"profile_id": "C"}, runtime_by_vid=vehicles,
        fixture_windows_s={v: (0., 86400.) for v in vehicles}, _available=lambda *a: True,
        event_calendar=NS(active_ids=lambda now: list(vehicles)),
        _timestamp=lambda now: pd.Timestamp("2016-10-31T00:00:00+08:00") + pd.Timedelta(seconds=now))
    manager = Manager.__new__(Manager)
    manager.control = control
    manager.network = NS(return_position_coordinates=lambda p: p,
        register_vehicle_leg=lambda *a: None, registry=NS(position_for=lambda x, y: (x, y)))
    manager.targets = {0: base}
    manager.targets_by_profile = {"HV": manager.targets, "C": {0: typed}}
    manager.active, manager.rows, manager.epochs, manager.counts = {}, [], [], Counter()
    manager._issued_routes, manager._latest_move = OrderedDict(), {}
    manager.geometry = NS(register=lambda *a: None)
    manager.bindings = NS(states=NS(REPOSITION=1), vehicle_route_leg=lambda *a: NS())
    manager.max_moves, manager.radius, manager.top_k = 1, 2000., 3
    manager.max_eta, manager.day_end_s, manager.wall_limit_s = 300., 86400, 10.
    manager.routing_wall_time_s = manager.total_decision_time_s = 0.
    manager._route_candidate = lambda r, d, t, x, y: dict(common_supported=True,
        compatible_profiles=["HV", "C"], duration_s=10., distance_m=100.,
        points=[(x, y), (float(d.lon_wgs84), float(d.lat_wgs84))], traffic_unknown_share=1.)
    surplus, contexts, _ = manager._typed_surplus(0)
    assert next(row for row in surplus if row[1].fixture.native_id == 0)[2] == 10
    assert contexts["C"][3][20] == 2  # C at A must not be counted as already at filtered target B.
    manager.after_normal_dispatch(control, 0)
    assert len(manager.rows) == manager.epochs[-1]["started"] == 1
    assert manager.rows[0]["target_profile_id"] == "C"
    assert manager.targets_by_profile["HV"] is manager.targets


def test_v5_is_opt_in_and_keeps_v4_scientific_resource_conditions():
    old = load_protocol(ROOT, Path("stage4/config/scheme_a_two_scale_city_v4.json"))
    new = load_protocol(ROOT, Path("stage4/config/scheme_a_two_scale_city_v5.json"))
    changed = {key for key in set(old) | set(new) if old.get(key) != new.get(key)}
    assert changed == {"version", "base_config", "base_config_sha256", "typed_request_diagnostics",
        "idle_movement_mode", "current_empty_distance_mode", "typed_reposition_reference", "typed_reposition_manifest"}
    assert new["max_match_wait_s"] == 300 and new["max_pickup_eta_s"] == 600
    assert old.get("current_empty_distance_mode", "LOCAL_STABLE") == "LOCAL_STABLE"


def test_common_movement_ablation_keeps_v5_except_target_rule_and_admin_identity():
    full = load_protocol(ROOT, Path("stage4/config/scheme_a_two_scale_city_v5.json"))
    common = load_protocol(ROOT, Path("stage4/config/scheme_a_two_scale_city_v5_common.json"))
    changed = {key for key in set(full) | set(common) if full.get(key) != common.get(key)}
    assert changed == {"version", "base_config", "base_config_sha256", "idle_movement_mode"}
    assert common["idle_movement_mode"] == "COMMON_HISTORICAL_DEFICIT"
    assert common["current_empty_distance_mode"] == "PRIMARY_FACE_MM"
    assert common["typed_request_diagnostics"] is True
    assert common["max_match_wait_s"] == 300 and common["max_pickup_eta_s"] == 600


def test_native_adapter_records_legal_AV_left_idle_when_HV_is_selected(monkeypatch):
    from stage4.dispatch import scheme_a_two_scale_native as native
    from stage4.tests.test_scheme_a_two_scale_native import fixture, Value
    c, cfg = fixture()
    cfg.update(idle_movement_mode="TYPE_CONDITIONED_HISTORICAL_DEFICIT", current_empty_distance_mode="PRIMARY_FACE_MM")
    c.repositioning_manager.active = {}
    c.request_decisions = RequestDecisionDiagnostics(typed_candidates=True)
    c.request_decisions.activate(11, "current", 900., 1200., 930.)
    c.request_decisions.round(11, 930., 2, 1, 1)
    monkeypatch.setattr(native, "predicted_vehicle_states", lambda *args: [])
    arcs = [AssignmentArc(vid, 11, eta, False, True,
        (c.runtime_by_vid[vid], c.request_by_rid[11], NS(route_distance_m=distance)), kind)
        for vid, kind, eta, distance in ((1, "AV", 60., 400.), (2, "HV", 30., 100.))]
    adapter = native.NativeTwoScaleAdapter("SERVICE_PRESERVING", Value(), None, None,
        lambda control, offered, now: offered, None, cfg)
    result = adapter.solve(c, arcs, [11], 930.)
    assert result.selected_indices == (1,)
    c.request_decisions.finish(11, 930., "ASSIGNED", vehicle_type="HV")
    row = next(c.request_decisions.rows())
    assert row["typed_last_final_AV_count"] == row["typed_last_final_HV_count"] == 1
    assert row["typed_AV_alternatives_idle_after_choice"] == 1
    assert row["typed_AV_alternatives_serving_other_request"] == row["typed_AV_alternatives_moving_after_choice"] == 0
    assert adapter.rows[-1]["secondary_optimality_proved"]
