"""Three focused mocked checks; no Actor, production, or native replay run."""
from collections import OrderedDict
import json
from types import SimpleNamespace as NS

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stage4.analysis.capability_chain_instances import IDENTITY_COLUMNS, JoinEvidence
from stage4.analysis.flexibility_prepare import S3
from stage4.dispatch.city_pickup_support import OVERLAY_REL, S2A, _SmallJoinRouter
from stage4.dispatch.scheme_a_city_connections import SchemeACityConnectors, SelectedMovementManager
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate


ORIGIN, PICKUP = (108.9, 34.2), (108.901, 34.2)
MIDNIGHT = pd.Timestamp("2016-10-31", tz="Asia/Shanghai")


class ETA:
    def beta_for(self, timestamp):
        return int(timestamp.hour * 4 + timestamp.minute // 15), (1. if timestamp.minute < 15 else 2.)


class JoinRouter:
    def __init__(self):
        self.geometry_cache = OrderedDict()
        self._features = None
        self.prepared = []

    def prepare_geometry(self, uids):
        self.prepared.extend(sorted(uids))


class Router:
    def __init__(self, root, eta):
        self.root, self.eta_adapter, self.calls = root, eta, []

    def route(self, *args):
        self.calls.append(args)
        origin, target = args[:2], args[2:4]
        return dict(common_supported=True, compatible_profiles=("HV", "C"), reason_codes=[],
            profile_reason_codes=dict(C=[]), raw_time_s=100., duration_s=777., distance_m=300.,
            points=[origin, target], snap_gap_origin_m=0., snap_gap_target_m=0.,
            snapped_origin=origin, snapped_target=target, shape_polyline6="mock-only-shape",
            edges=[dict(stage3_edge_uid="EMPTY")], traffic_unknown_share=1.)


def tokens(date, order, *, reverse=False, length=12):
    return [dict(zip(IDENTITY_COLUMNS, [date, order, i,
        "reverse-canonical" if reverse and i == 0 else None,
        "HISTORICAL_REVERSE_OVERLAY" if reverse and i == 0 else "FULL_NETWORK_EDGE",
        None if reverse and i == 0 else f"{order}:{i}"])) for i in range(length)]


def control_and_evidence(root):
    eta = ETA()
    route = Router(root, eta)
    evidence = NS(router=JoinRouter(), tokens=OrderedDict(), _private_ranges={},
        _private_identity=None, _private_overlay=None,
        overlay=pd.DataFrame(columns=["canonical_edge_uid", "physical_forward_stage3_edge_uid"]))
    c = NS(eta_adapter=eta, sim_time=61200, request_by_rid={},
        _timestamp=lambda now: MIDNIGHT + pd.Timedelta(seconds=now),
        runtime_by_vid={}, assignment_rows=[], completed_rids=set(),
        routing_engine=NS(return_position_coordinates=lambda p: p))
    return c, route, evidence


def task(order, *, kind="FORECAST", date="20161010", profiles=("HV", "C")):
    return NS(source_date=date, source_order_id=order, source_kind=kind,
        pickup=PICKUP, dropoff=ORIGIN, compatible_profiles=profiles, deadline_s=90000.)


def test_frozen_train_provenance_full_bodies_and_departure_time_beta(tmp_path, monkeypatch):
    path = tmp_path / S3 / "cache/train/date=20161010/identity.parquet"
    path.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist(tokens("20161010", "source", reverse=True)
        + tokens("20161010", "target")), path)
    overlay = tmp_path / OVERLAY_REL
    overlay.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(canonical_edge_uid="reverse-canonical",
        physical_forward_stage3_edge_uid="physical-forward")]), overlay)
    c, route, evidence = control_and_evidence(tmp_path)
    geometry_path = tmp_path / S2A / "stage3_full_network_edges.parquet"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    geometry_rows = [dict(stage3_edge_uid=uid, geometry=json.dumps([ORIGIN, PICKUP]),
        from_stage3_node_uid="first", to_stage3_node_uid="last")
        for uid in ["physical-forward", "EMPTY"] + [f"{order}:{i}"
            for order in ("source", "target") for i in range(12)]]
    pq.write_table(pa.Table.from_pylist(geometry_rows), geometry_path)
    evidence.router = _SmallJoinRouter(route)
    templates = pd.DataFrame([dict(date="20161010", order_id=order, compatible_C=is_c)
        for order, is_c in (("source", True), ("target", True), ("hv-only", False))])
    observed = []
    def joined(self, previous, following, routed, tolerance):
        observed.append((len(self.tokens[previous]), len(self.tokens[following]), tolerance))
        assert self.tokens[previous].iloc[0].route_token_type == "HISTORICAL_REVERSE_OVERLAY"
        assert self.overlay_forward == {"reverse-canonical": "physical-forward"}
        return dict(supported=True, compatible_profiles=("HV", "C"), C_reason_codes=[])
    monkeypatch.setattr(JoinEvidence, "evaluate", joined)
    con = SchemeACityConnectors(c, route, NS(seam_evidence=evidence), templates,
        geometry_timestamp=MIDNIGHT + pd.Timedelta(hours=17))
    source = NS(position=ORIGIN, context=dict(kind="CUSTOMER", task=task("source")))
    first = con(source, task("target"), 62040., "C")
    second = con(source, task("target"), 62100., "C")
    assert first["travel_time_s"] == 100. and second["travel_time_s"] == 200.
    assert first["empty_distance_m"] == second["empty_distance_m"] == 300.
    assert observed == [(12, 12, 80.)]  # Static certificate reuse, not a new seam parse.
    assert len(route.calls) == 1 and route.calls[0][-1] == MIDNIGHT + pd.Timedelta(hours=17)
    assert second["arrival_context"] is None and second["routed"] is None
    assert con(source, task("hv-only", profiles=("HV",)), 62100., "HV")["supported"]
    not_revealed = con(source, task("future31", kind="ACTUAL_PENDING", date="20161031"), 62100., "C")
    assert not_revealed["reason_codes"] == ["TEST31_CUSTOMER_NOT_YET_REVEALED"]
    assert len(route.calls) == 1
    assert con.diagnostics()["counts"]["train_identity_projection_scans"] == 1
    assert con.diagnostics()["counts"]["certificate_cache_hits"] == 1
    con.join_router.small_geometry({"EMPTY"})
    geometry_metrics = con.diagnostics()
    assert geometry_metrics["geometry_rowgroup_reads"] == 1
    assert geometry_metrics["geometry_rowgroup_cache_hits"] == 1
    assert geometry_metrics["geometry_rowgroup_cache_mib"] <= 64.


def test_native_arrival_prefix_is_not_a_fabricated_future_customer(tmp_path, monkeypatch):
    c, route, evidence = control_and_evidence(tmp_path)
    record = NS(order_id="actual", sim_time_s=61190.)
    c.request_by_rid[1] = record
    c.runtime_by_vid[2] = NS(native_vehicle=NS(pos=ORIGIN), fixture=NS(native_id=2))
    evidence._private_identity = pa.Table.from_pylist(tokens("20161031", "actual"))
    evidence._private_ranges = dict(actual=(0, 12))
    evidence._idle_context = lambda *args: dict(point=ORIGIN,
        edges=[dict(stage3_edge_uid=f"actual-incoming:{i}") for i in range(14)],
        evidence_kind="ACTUAL_ROUTED_ARRIVAL_OR_PREFIX")
    def joined(self, previous, following, routed, tolerance):
        assert len(self.tokens[previous]) == 14
        assert self.tokens[previous].resolved_stage3_edge_uid.iloc[-1] == "actual-incoming:13"
        assert self.selected.loc[previous].end_lon_wgs84 == ORIGIN[0]
        assert self.tokens[following].order_id.unique().tolist() == ["actual"]
        return dict(supported=True, compatible_profiles=("C",), C_reason_codes=[])
    monkeypatch.setattr(JoinEvidence, "evaluate", joined)
    con = SchemeACityConnectors(c, route, NS(seam_evidence=evidence),
        pd.DataFrame(columns=["date", "order_id", "compatible_C"]), geometry_timestamp=MIDNIGHT)
    origin = NS(position=ORIGIN, context=dict(kind="NATIVE_VEHICLE", native_vehicle_id=2, timestamp_s=61200.))
    result = con(origin, task("actual", kind="ACTUAL_PENDING", date="20161031"), 61200., "C")
    assert result["supported"] and result["connection_evidence_state"] == "C_DIRECTED_SEAM_CERTIFIED"
    origin.context["timestamp_s"] = 61170.
    rejected = con(origin, task("actual", kind="ACTUAL_PENDING", date="20161031"), 61200., "C")
    assert rejected["reason_codes"] == ["NATIVE_DEPARTURE_CONTEXT_NOT_CURRENT"]


def test_only_selected_native_move_executes_and_hooks_account_one_leg(tmp_path):
    positions = {1: ORIGIN, 2: PICKUP}
    plans, registered = [], []
    network = NS(return_position_coordinates=lambda p: positions[p[0]],
        move_along_route=lambda *args, **kwargs: None,
        registry=NS(position_for=lambda *point: (2, None, None)),
        register_vehicle_leg=lambda *args: registered.append(args))
    runtime = NS(fixture=NS(native_id=7, vehicle_id="C7", vehicle_type="AV"),
        native_vehicle=NS(pos=(1, None, None),
            assign_vehicle_plan=lambda plan, now: plans.append((plan, now))), active_order_id=None)
    c = NS(routing_engine=network, bindings=NS(states=NS(REPOSITION="REPOSITION"),
            vehicle_route_leg=lambda status, destination, assignments: NS(status=status, destination_pos=destination)),
        _assign=lambda *args: None, receive_status_update=lambda *args: None,
        _available=lambda runtime, now: True, config=dict(profile_id="C"),
        fixture_windows_s={7: (0, 1800)}, runtime_by_vid={7: runtime})
    reference = pd.DataFrame([dict(time_bin_index=0, node_id=1, lon_wgs84=ORIGIN[0],
        lat_wgs84=ORIGIN[1], demand_share=1.)])
    manager = SelectedMovementManager(c, reference, NS(), day_end_s=1800)
    # No event calendar/demand heuristic exists in this mock: an empty queue
    # must not accidentally call the parent's automatic greedy rule.
    manager.after_normal_dispatch(c, 0)
    assert not manager.rows and not registered
    routed = dict(supported=True, duration_s=100., distance_m=300., points=[ORIGIN, PICKUP],
        origin_wgs84=ORIGIN, target_wgs84=PICKUP, departure_s=900.,
        connection_evidence_state="C_DIRECTED_SEAM_CERTIFIED", traffic_unknown_share=1.)
    selected = dict(native_vehicle_id=7, position=PICKUP, node_id=2, site_id="S2", routed=routed)
    with pytest.raises(ValueError, match="900-second"):
        manager.queue_actions([selected], 30)
    manager.queue_actions([selected], 900)
    manager.after_normal_dispatch(c, 900)
    assert len(manager.rows) == len(plans) == len(registered) == 1
    assert runtime.state == "NATIVE_REPOSITIONING" and manager.active == {7: 0}
    row = manager.rows[0]
    assert row["decision_source"] == "SCHEME_A_SHARED_MASTER_SELECTED_CURRENT_MOVE"
    assert row["native_vehicle_id"] == 7 and row["planned_distance_m"] == 300.
    runtime.native_vehicle.pos = (2, None, None)
    manager._record_end(7, 1000., 300., "COMPLETED")
    manager.finalize(1000)
    summary = manager.summary()
    assert summary["empty_distance_m"] == 300. and not summary["automatic_historical_deficit_rule_active"]
    assert summary["counts"]["STARTED"] == summary["counts"]["COMPLETED"] == 1


def test_scalar_certificate_retains_static_rejection_not_deadline_and_move_gets_geometry(tmp_path, monkeypatch):
    c, route, evidence = control_and_evidence(tmp_path)
    calls = []
    def joined(*args):
        calls.append(1)
        return dict(supported=True, compatible_profiles=("C",), C_reason_codes=[])
    monkeypatch.setattr(JoinEvidence, "evaluate", joined)
    original = route.route
    def routed(*args):
        result = original(*args)
        if args[2] == 108.902:
            result.update(common_supported=False, reason_codes=["STATIC_MAPPING_UNSUPPORTED"])
        return result
    route.route = routed
    templates = pd.DataFrame([dict(date="20161010", order_id="hv", compatible_C=False)])
    con = SchemeACityConnectors(c, route, NS(seam_evidence=evidence), templates, geometry_timestamp=MIDNIGHT)
    origin = NS(position=ORIGIN, context=None)
    site = dict(site_id="S1", position=PICKUP)
    first = con(origin, site, 62040., "C")
    cached = con(origin, site, 62100., "C")
    assert first["travel_time_s"] == 100. and cached["travel_time_s"] == 200.
    assert first["arrival_context"]["edge_uids"] == cached["arrival_context"]["edge_uids"] == ("EMPTY",)
    assert first["routed"] is cached["routed"] is None and len(calls) == 1
    executed = con(origin, dict(site, require_geometry=True), 62100., "C")
    assert executed["routed"]["points"] == [ORIGIN, PICKUP] and len(calls) == 2
    assert executed["routed"]["departure_s"] == 62100.
    rejected_site = dict(site_id="Sbad", position=(108.902, 34.2))
    assert con(origin, rejected_site, 62100., "C")["reason_codes"] == ["STATIC_MAPPING_UNSUPPORTED"]
    assert con(origin, rejected_site, 62100., "C")["reason_codes"] == ["STATIC_MAPPING_UNSUPPORTED"]
    assert len(route.calls) == 2  # Static pass and static failure both cached.
    c.config = dict(scheme_a_fast_forecast_hv=True)
    matrix_calls = []
    def estimate_many(vehicles, lon, lat, stamp):
        matrix_calls.append((vehicles[0], stamp))
        bin_index, beta = c.eta_adapter.beta_for(stamp)
        return {0: PickupEstimate(100., 100. * beta, 300., beta, bin_index, True)}
    c.eta_adapter.estimate_many = estimate_many
    late = task("hv", profiles=("HV",))
    late.deadline_s = 62150.
    assert con(origin, late, 62100., "HV")["reason_codes"] == ["CONNECTOR_EXCEEDS_ORIGINAL_PICKUP_DEADLINE"]
    late.deadline_s = 63000.
    accepted = con(origin, late, 62100., "HV")
    assert accepted["supported"] and accepted["travel_time_s"] == 200.
    assert accepted["connection_evidence_state"] == "HV_STATIC_AUTO_OD_ROUTABLE"
    assert len(matrix_calls) == 1 and len(route.calls) == 2
    metrics = con.diagnostics()
    assert metrics["counts"]["certificate_cache_hits"] == 3
    assert metrics["certificate_cache_entries"] == 3
    assert 0 < metrics["certificate_cache_mib"] < metrics["certificate_cache_byte_limit_mib"] == 32.
