"""Integrated mechanism checks only: no Test31 replay, routing Actor or GPU."""
import json
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace as NS

import pandas as pd
import pytest

from stage4.analysis.scheme_a_two_scale_experiment import group_configuration, load_protocol
from stage4.dispatch.historical_deficit_movement import HistoricalDeficitMovementManager
from stage4.dispatch.city_pickup_support import Test31PickupJoinEvidence as PickupJoinEvidence
from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors
from stage4.dispatch.request_decision_diagnostics import RequestDecisionDiagnostics
from stage4.dispatch.rolling_or_control import patience_expired, patience_feasible
from stage4.dispatch import scheme_a_two_scale_native as native
from stage4.dispatch.scheme_a_current_flow import solve_current_flow
from stage4.dispatch.solver import AssignmentArc, LexicographicResult
from stage4.tests.test_controlled_exit import controller
from stage4.tests.test_scheme_a_two_scale_native import Value, fixture
from stage4.tests.test_controlled_routes import encode


ROOT = Path(__file__).resolve().parents[2]
V3 = Path("stage4/config/scheme_a_two_scale_city_v3.json")


def test_v3_is_explicit_and_cannot_overwrite_old_protocol_or_products():
    old, new = load_protocol(ROOT), load_protocol(ROOT, V3)
    for key in ("patience_s", "requested_q_a", "passenger_acceptance_rate", "solver_time_limit_s",
                "planning_horizon_s", "coarse_value_max_pricing_rounds", "reposition_max_moves"):
        assert old[key] == new[key]
    assert "idle_movement_mode" not in old
    assert new["idle_movement_mode"] == "COMMON_HISTORICAL_DEFICIT"
    assert new["precise_request_deadline"] and new["coarse_nested_profile_paths"]
    assert group_configuration(old, "HOTSPOT_BASELINE")["disk_route_cache"] != (
        group_configuration(new, "HOTSPOT_BASELINE")["disk_route_cache"])


def test_common_deficit_moves_after_service_without_future_value_and_retain_direction_certificate():
    origin, target = (108.9, 34.2), (108.905, 34.2)
    ref = pd.DataFrame([dict(time_bin_index=1, node_id=i, lon_wgs84=point[0],
        lat_wgs84=point[1], demand_share=share) for i, point, share in
        ((1, origin, .1), (2, target, .9))])
    plans, queries = [], []
    runtimes = {i: NS(fixture=NS(native_id=i, vehicle_id=str(i), vehicle_type="AV" if i == 2 else "HV"),
        active_order_id=None, native_vehicle=NS(pos=(1, None, None),
            assign_vehicle_plan=lambda p, t: plans.append((p, t)))) for i in (1, 2, 3)}
    points = {1: origin, 2: target}
    network = NS(return_position_coordinates=lambda p: points[p[0]], move_along_route=lambda *a, **k: None,
        register_vehicle_leg=lambda *a: None, registry=NS(position_for=lambda *a: (2, None, None)))
    control = NS(routing_engine=network, bindings=NS(states=NS(REPOSITION=1),
        vehicle_route_leg=lambda *a: a), config=dict(profile_id="C"),
        runtime_by_vid=runtimes, fixture_windows_s={i: (0, 3600) for i in runtimes},
        _timestamp=lambda s: pd.Timestamp("2016-10-31", tz="Asia/Shanghai") + pd.Timedelta(seconds=s),
        event_calendar=NS(active_ids=lambda t: tuple(runtimes)),
        _assign=lambda *a: None, receive_status_update=lambda *a, **k: None)
    control._available = lambda r, t: r is not runtimes[3]  # Already committed by normal dispatch.
    manager = HistoricalDeficitMovementManager(control, ref, NS(), max_moves=50)
    control.repositioning_manager = manager
    def connector(state, destination, now, profile):
        queries.append((state, destination, now, profile))
        assert state.context["kind"] == "NATIVE_VEHICLE"
        assert destination["node_id"] == 2 and destination["require_geometry"]
        routed = dict(points=[origin, target], duration_s=60., distance_m=450.,
            traffic_unknown_share=1., connection_evidence_state="C_DIRECTED_SEAM_CERTIFIED" if profile == "C" else "HV_AUTO_ROUTABLE",
            geometry_timestamp="2016-10-31T00:00:00+08:00", shape_polyline6=encode([origin, target]),
            edges=[dict(stage3_edge_uid="issued", begin_shape_index=0, end_shape_index=1,
                trace_shape_polyline6=encode([origin, target]))])
        return dict(supported=True, routed=routed, connection_evidence_state=routed["connection_evidence_state"])
    manager.connectors = connector
    manager.after_normal_dispatch(control, 900)
    assert set(manager.active) == {1, 2} and len(plans) == len(queries) == 2
    assert {row["target_node_id"] for row in manager.rows} == {2}
    assert manager.summary()["independent_of_future_service_value"]
    assert manager.rows[1]["connection_evidence_state"] == "C_DIRECTED_SEAM_CERTIFIED"
    with pytest.raises(ValueError, match="also execute"):
        manager.queue_actions([dict(native_vehicle_id=1)], 900)
    manager.connectors = lambda *a: dict(supported=True, routed={}, connection_evidence_state=None)
    with pytest.raises(ValueError, match="departure direction"):
        manager._route_candidate(runtimes[2], ref.iloc[1], 1800, *origin)

    join = PickupJoinEvidence.__new__(PickupJoinEvidence)
    join._idle_routes = OrderedDict()
    join.departure_context_routing_time_s = 0.
    join.departure_context_route_count = 0
    def forbidden(*a):
        raise AssertionError("issued path must not be routed again at a different time")
    join.router = NS(base=NS(route=forbidden))
    token = manager.geometry.by_vehicle[2][2]
    runtimes[2].native_vehicle.pos = (1, 2, .5, token)
    context = join._idle_context(control, runtimes[2], 930, -1.)
    assert context["edges"][0]["stage3_edge_uid"] == "issued"
    assert context["source_progress_fraction"] == .5 and join.departure_context_route_count == 0
    row = manager.rows[manager.active.pop(2)]
    row.update(status="COMPLETED", end_time_s=960., actual_distance_m=450.)
    runtimes[2].native_vehicle.pos = (2, None, None)
    assert manager.has_movement_since(2, -1.)
    assert join._idle_context(control, runtimes[2], 960, -1.) is not None

    # If compact cache was evicted and reconstruction fails, a completed
    # MOVE must not make a never-served C vehicle appear newly admitted.
    manager._issued_routes.clear()
    stamps = []
    join.router.base.route = lambda *a: (stamps.append(a[-1]) or dict(common_supported=False))
    control.assignment_rows, control.completed_rids, control.request_by_rid = [], set(), {}
    control.sim_time = 960
    connector = SchemeACityConnectors.__new__(SchemeACityConnectors)
    connector.control, connector.actual_evidence = control, join
    connector._native_scope = None
    connector.native_context_cache, connector.native_signature_cache = OrderedDict(), OrderedDict()
    result = connector._native_source(dict(native_vehicle_id=2, timestamp_s=960), target)
    assert result == (None, None, "PREVIOUS_DIRECTION_CONTEXT_UNAVAILABLE")
    assert stamps == [pd.Timestamp("2016-10-31", tz="Asia/Shanghai")]


def test_release_based_fractional_deadline_and_competing_expiry_paths_do_not_reset_clock():
    c, runtime, request, _ = controller()
    c.config.update(precise_request_deadline=True)
    c.request_decisions = RequestDecisionDiagnostics()
    request.sim_time_s = 60.5
    c.user_request(request.native_request, 90)
    assert c.request_meta[0]["pickup_deadline_s"] == 360.5
    assert not patience_expired(360, 360.5) and patience_expired(361, 360.5)
    assert not patience_feasible(-1., 20.)
    def selected(control, arcs, waiting, now):
        assert len(arcs) == 1
        control.request_decisions.validated(arcs[0].request_id)
        return LexicographicResult((0,), 0., 0, 1, 0)
    # The lifecycle check does not initialize an unrelated HiGHS MILP in the
    # same process as the separately checked single-thread coarse LP.
    c.flexibility_adapter = NS(policy="SERVICE_PRESERVING", solve=selected)
    c.time_trigger(90)
    assert c.request_decisions.records[0].status == "ASSIGNED"
    with pytest.raises(ValueError, match="committed"):
        c._expire(0, 390)
    d = RequestDecisionDiagnostics()
    d.activate(2, "unserved", 60.5, 360.5, 90)
    d.round(2, 90, 5, 0, 0)
    d.round(2, 330, 7, 0, 0)
    with pytest.raises(ValueError, match="before"):
        d.finish(2, 360, "EXPIRED")
    d.finish(2, 390, "EXPIRED")
    d.finish(2, 390, "EXPIRED")
    row = next(d.rows())
    assert row["deadline_s"] == 360.5 and row["attempt_rounds"] == 2
    assert row["expiry_last_reached_stage"] == "NO_NEARBY_AVAILABLE_VEHICLE"
    assert json.loads(row["rejection_counts_json"]) == {}


def test_service_preserving_vs_defer_is_visible_and_common_moves_never_enter_current_flow(monkeypatch):
    c, cfg = fixture()
    cfg["idle_movement_mode"] = "COMMON_HISTORICAL_DEFICIT"
    c.request_decisions = RequestDecisionDiagnostics()
    c.request_decisions.activate(11, "current", 900, 1200, 900)
    value = Value()
    value.value = lambda state: 1. if state.position == (108.91, 34.2) else 3.
    monkeypatch.setattr(native, "predicted_vehicle_states", lambda *a: [])
    monkeypatch.setattr(native, "solve_current_flow", lambda actions, units, **kw:
        solve_current_flow(actions, units, backend="INTEGER_REFERENCE", **kw))
    def forbidden(*a):
        raise AssertionError("common deficit base must not query value-selected MOVE sites/routes")
    monkeypatch.setattr(native, "sites_for", forbidden)
    arc = AssignmentArc(1, 11, 60, False, True,
        (c.runtime_by_vid[1], c.request_by_rid[11], NS(route_distance_m=400)), "AV")
    deferred = native.NativeTwoScaleAdapter("CHAIN_DEFER", value, None, forbidden,
        lambda c, arcs, now: arcs, None, cfg)
    assert deferred.solve(c, [arc], [11], 900).selected_indices == ()
    assert c.request_decisions.records[11].withheld_idle_opportunity_rounds == 1
    assert c.request_decisions.records[11].validated_arc_attempts == 1
    preserving = native.NativeTwoScaleAdapter("SERVICE_PRESERVING", value, None, forbidden,
        lambda c, arcs, now: arcs, None, cfg)
    assert preserving.solve(c, [arc], [11], 900).selected_indices == (0,)
    assert c.repositioning_manager.queued[-1] == ([], 900)
    assert preserving.rows[-1]["current_move_proposed_count"] == 0
    assert preserving.rows[-1]["idle_movement_mode"] == "COMMON_HISTORICAL_DEFICIT"
