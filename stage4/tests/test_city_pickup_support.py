import json
from types import SimpleNamespace as NS

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from stage4.tests.test_controlled_routes import encode

from stage4.analysis.capability_chain_instances import IDENTITY_COLUMNS
from stage4.dispatch.city_pickup_support import CityPickupSupportValidator, TEST31_IDENTITY, S2A
from stage4.dispatch.solver import AssignmentArc
from stage4.fleetpy_adapter.valhalla_time_adapter import PickupEstimate


MIDNIGHT = pd.Timestamp("2016-10-31", tz="Asia/Shanghai")
SNAPSHOT = MIDNIGHT + pd.Timedelta(hours=17)
ORIGIN, PICKUP = (108.9, 34.2), (108.901, 34.2)


class ETA:
    def beta_for(self, timestamp):
        return (68 if timestamp.minute < 15 else 69), (1. if timestamp.minute < 15 else 2.)


class Routes:
    def __init__(self, eta, *, compatible=True, supported=True):
        self.eta_adapter, self.calls = eta, []
        self.compatible, self.supported = compatible, supported

    def route(self, origin_lon, origin_lat, target_lon, target_lat, timestamp):
        self.calls.append((origin_lon, origin_lat, target_lon, target_lat, timestamp))
        points = [(origin_lon, origin_lat), (target_lon, target_lat)]
        return dict(common_supported=self.supported,
            compatible_profiles=frozenset(("HV", "A", "C")) if self.compatible else frozenset(("HV", "A")),
            reason_codes=() if self.supported else ("EDGE_OUTSIDE_FROZEN_FULL_NETWORK",),
            profile_reason_codes=dict(C=() if self.compatible else ("UTURN_OUTSIDE_RESEARCH_PROFILE",)),
            raw_time_s=100., duration_s=999., distance_m=300., points=points, geometry_points=points,
            snap_gap_origin_m=0., snap_gap_target_m=0., snapped_origin=points[0], snapped_target=points[-1],
            shape_polyline6=encode(points),
            edges=[dict(stage3_edge_uid="incoming", begin_shape_index=0, end_shape_index=1,
                trace_shape_polyline6=encode(points))], traffic_unknown_share=1., dynamic_evidence_complete=False)


class AllowedJoin:
    tolerance = 80.

    def prepare(self, c, arcs, now):
        pass

    def evaluate_pickup(self, c, runtime, request, route, now):
        return dict(supported=True, compatible_profiles=["HV", "C"], reason_codes=[],
            C_reason_codes=[], pickup_to_customer_checked=True, previous_to_pickup_checked=False)


def case(eta, *, vehicle_type="AV", vid=2, rid=100, dropoff=(108.902, 34.2)):
    runtime = NS(fixture=NS(native_id=vid, vehicle_type=vehicle_type), native_vehicle=NS(pos=ORIGIN))
    request = NS(native_id=rid, order_id=f"order{rid}", sim_time_s=61200., pickup_lon_wgs84=PICKUP[0], pickup_lat_wgs84=PICKUP[1],
        dropoff_lon_wgs84=dropoff[0], dropoff_lat_wgs84=dropoff[1], dropoff_position=dropoff)
    estimate = PickupEstimate(5., 5., 5., 1., 68, False)
    arc = AssignmentArc(vid, rid, 5., False, False, (runtime, request, estimate), vehicle_type)
    c = NS(eta_adapter=eta, _timestamp=lambda now: MIDNIGHT + pd.Timedelta(seconds=now),
        request_meta={rid: dict(pickup_deadline_s=62400., research_route_compatible=True)},
        request_by_rid={rid: request}, routing_engine=NS(return_position_coordinates=lambda pos: pos),
        assignment_rows=[], completed_rids=set())
    return c, arc


@pytest.mark.parametrize("supported", [True, False])
def test_empty_pickup_capability_is_checked_and_hv_arc_is_preserved(supported):
    eta = ETA()
    c, av = case(eta)
    _, hv = case(eta, vehicle_type="HV", vid=1, rid=101)
    c.request_meta[101] = dict(pickup_deadline_s=62400., research_route_compatible=False)
    routes = Routes(eta, compatible=False, supported=supported)
    validator = CityPickupSupportValidator(routes, geometry_timestamp=SNAPSHOT,
        routing_budget_s=10., seam_evidence=AllowedJoin())
    retained = validator(c, [av, hv], 62040.)
    assert retained == [hv] and retained[0] is hv
    assert len(routes.calls) == 1 and validator.last_diagnostics["original_indices"] == [1]
    assert validator.pickup_paths == {}
    validator.routing_budget_s = 0.
    retained = validator(c, [av, hv], 62040.)
    assert retained == [hv] and len(routes.calls) == 1
    assert validator.last_diagnostics["rejection_reason_counts"]["PICKUP_SUPPORT_ROUTING_BUDGET_EXHAUSTED"] == 1


def test_snapshot_geometry_cached_but_now_beta_and_original_deadline_are_rechecked():
    eta = ETA()
    c, arc = case(eta)
    routes = Routes(eta)
    validator = CityPickupSupportValidator(routes, geometry_timestamp=SNAPSHOT,
        routing_budget_s=10., seam_evidence=AllowedJoin())
    first = validator(c, [arc], 62040.)[0]  # 17:14, beta=1.
    second = validator(c, [arc], 62100.)[0]  # 17:15, beta=2.
    assert first.pickup_eta_s == 100. and second.pickup_eta_s == 200.
    assert first.payload[2].valhalla_time_s == second.payload[2].valhalla_time_s == 100.
    assert second.payload[2].route_distance_m == 300. and second.payload[2].beta == 2.
    assert second.payload[2].cache_hit and second.payload[2].time_bin_index == 69
    assert len(routes.calls) == 1 and routes.calls[0][-1] == SNAPSHOT
    assert arc.pickup_eta_s == 5.  # The solver maps returned replacements to its original list.
    assert validator.pickup_paths[(2, 100)]["duration_s"] == 200.
    assert c.city_pickup_paths[(2, 100)] == validator.pickup_paths[(2, 100)]["points"]
    assert validator.last_diagnostics["original_indices"] == [0]
    assert validator(c, [arc], 62220.) == []  # Only 180 s remain; no deadline extension.
    assert validator.pickup_paths == {} and len(routes.calls) == 1
    assert validator.last_diagnostics["rejection_reason_counts"] == dict(SCALAR_PICKUP_EXCEEDS_REMAINING_PATIENCE=1)
    assert validator.diagnostics()["routing_budget_is_separate_from_solver"]


@pytest.mark.parametrize("maneuver", ["STRAIGHT", "UTURN"])
def test_real_join_parser_catches_extra_customer_seam_action(tmp_path, maneuver):
    eta = ETA()
    outgoing_point = (108.902, 34.2) if maneuver == "STRAIGHT" else ORIGIN
    c, arc = case(eta, dropoff=outgoing_point)
    routes = Routes(eta)
    routes.root = tmp_path
    routes._boundary = pa.Table.from_pylist([dict(stage3_edge_uid="incoming", intersection_complex_uid="j", boundary_role="INCOMING"),
        dict(stage3_edge_uid="outgoing", intersection_complex_uid="j", boundary_role="OUTGOING")])
    routes._movements = pa.Table.from_pylist([dict(intersection_complex_uid="j", incoming_stage3_edge_uid="incoming",
        outgoing_stage3_edge_uid="outgoing", route_turn_type=maneuver,
        restriction_enforcement_certified=False, movement_legality_state="UNKNOWN")])
    routes._controls = pa.Table.from_pylist([dict(intersection_complex_uid="j", roundabout_evidence_present=False,
        signalized_research=False, control_basis="RESEARCH_ASSUMED_NON_SIGNAL_WITH_PROVENANCE")])
    identity = tmp_path / TEST31_IDENTITY
    identity.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(zip(IDENTITY_COLUMNS,
        ["20161031", "order100", 0, None, "FULL_NETWORK_EDGE", "outgoing"]))]), identity)
    geometry = tmp_path / S2A / "stage3_full_network_edges.parquet"
    geometry.parent.mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([dict(stage3_edge_uid="incoming", geometry=json.dumps([ORIGIN, PICKUP]),
        from_stage3_node_uid="origin", to_stage3_node_uid="junction"),
        dict(stage3_edge_uid="outgoing", geometry=json.dumps([PICKUP, outgoing_point]),
            from_stage3_node_uid="junction", to_stage3_node_uid="next")]), geometry)
    validator = CityPickupSupportValidator(routes, geometry_timestamp=SNAPSHOT, routing_budget_s=10.)
    prepared = validator.prepare_private_identity_store(["order100"])
    assert prepared["stored_order_count"] == prepared["stored_token_count"] == 1
    if maneuver == "STRAIGHT":
        arc.payload[0].native_vehicle.pos = (7, 8, .5, 99)
        c.routing_engine.return_position_coordinates = lambda pos: ((108.9005, 34.2) if pos == (7, 8, .5, 99) else pos)
        c.repositioning_manager = NS(active={2: 0}, rows=[dict(native_vehicle_id=2,
            start_time_s=61990, start_lon_wgs84=ORIGIN[0], start_lat_wgs84=ORIGIN[1],
            destination_position=PICKUP, shape_polyline6=encode([ORIGIN, PICKUP]), status="IN_PROGRESS")])
    retained = validator(c, [arc], 62040.)
    assert bool(retained) is (maneuver == "STRAIGHT")
    if maneuver == "UTURN":
        assert validator.last_diagnostics["rejection_reason_counts"] == dict(UTURN_OUTSIDE_RESEARCH_PROFILE=1)
    else:
        seam = validator.pickup_paths[(2, 100)]["seam"]
        assert seam["join_maneuvers"] == ["STRAIGHT"] and seam["pickup_to_customer_checked"]
        assert seam["previous_to_pickup_checked"]
        context = c.city_departure_contexts[2]
        assert context["source_progress_fraction"] == .5 and context["original_shape_verified"]
        assert [edge["stage3_edge_uid"] for edge in context["edges"]] == ["incoming"]
    assert validator.seam_evidence.identity_scan_count == 1
    assert "order100" in validator.seam_evidence.tokens
    validator(c, [arc], 62070.)
    assert validator.seam_evidence.identity_scan_count == 1  # No per-epoch million-token scan.
    assert len(routes.calls) == (2 if maneuver == "STRAIGHT" else 1)
    arc.payload[1].sim_time_s = 63000.
    with pytest.raises(ValueError, match="unrevealed"):
        validator(c, [arc], 62070.)
