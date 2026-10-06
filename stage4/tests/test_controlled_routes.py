from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from stage3.odd_tod.finalization import DYNAMIC_UNKNOWN_REASON
from stage3.odd_tod.network_foundation import stage3_edge_uid
from stage4.dispatch.controlled_routes import ControlledEmptyRouter, CONTROL_REL, S2A, S2B


STAMP = pd.Timestamp("2016-10-31 08:07:30", tz="Asia/Shanghai")
POINTS = [(108.9001, 34.2001), (108.901, 34.2001), (108.9011, 34.2002), (108.9011, 34.201)]
UIDS = [stage3_edge_uid(i) for i in (10, 11, 12)]


def encode(points):
    output, previous = [], (0, 0)
    for lon, lat in points:
        value = (round(lat * 1e6), round(lon * 1e6))
        for delta in (value[0] - previous[0], value[1] - previous[1]):
            number = ~(delta << 1) if delta < 0 else delta << 1
            while number >= 0x20:
                output.append(chr((0x20 | (number & 0x1f)) + 63))
                number >>= 5
            output.append(chr(number + 63))
        previous = value
    return "".join(output)


class Actor:
    def __init__(self, *, unknown_edge=False, missing_indices=False, incomplete_shape=False):
        self.route_requests, self.trace_requests = [], []
        self.unknown_edge, self.missing_indices = unknown_edge, missing_indices
        self.incomplete_shape = incomplete_shape

    def route(self, request):
        self.route_requests.append(request)
        return dict(trip=dict(status=0, summary=dict(time=100.0, length=0.25),
            legs=[dict(shape=encode(POINTS))]))

    def trace_attributes(self, request):
        self.trace_requests.append(request)
        edges = [dict(id=i, length=0.08, begin_shape_index=j, end_shape_index=j + 1)
            for j, i in enumerate((10, 11, 999 if self.unknown_edge else 12))]
        if self.missing_indices:
            edges[-1].pop("begin_shape_index")
        if self.incomplete_shape:
            edges.pop(1)
        return dict(shape=encode(POINTS), edges=edges)


def assets(tmp_path, *, maneuver="LEFT", signalized=False, certified=False,
        missing_movement=False, unresolved_geometry=False):
    def write(relative, rows):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows), path)

    geometries = [[POINTS[j], POINTS[j + 1]] for j in range(3)]
    if unresolved_geometry:
        geometries[0] = [POINTS[0], POINTS[0]]
    import json
    write(S2A / "stage3_full_network_edges.parquet", [dict(valhalla_directed_edge_id=i,
        stage3_edge_uid=uid, auto_routable=True, geometry=json.dumps(geometry))
        for i, uid, geometry in zip((10, 11, 12), UIDS, geometries)])
    write(S2B / "stage3_edge_complex_boundary_index.parquet", [dict(stage3_edge_uid=uid,
        intersection_complex_uid="junction", boundary_role=role)
        for uid, role in zip(UIDS, ("INCOMING", "INTERNAL", "OUTGOING"))])
    write(S2B / "stage3_route_movement_lookup.parquet", [dict(intersection_complex_uid="junction",
        incoming_stage3_edge_uid=UIDS[2] if missing_movement else UIDS[0],
        outgoing_stage3_edge_uid=UIDS[2], route_turn_type=maneuver,
        restriction_enforcement_certified=certified,
        movement_legality_state="CERTIFIED_PROHIBITED" if certified else "UNKNOWN")])
    write(CONTROL_REL, [dict(intersection_complex_uid="junction", roundabout_evidence_present=False,
        signalized_research=signalized,
        control_basis="NEW_699_POSITIVE_OVERLAY" if signalized else "RESEARCH_ASSUMED_NON_SIGNAL_WITH_PROVENANCE")])
    return tmp_path


def router(tmp_path, **asset_options):
    actor = Actor()
    eta = SimpleNamespace(actor=actor, beta_for=lambda timestamp: (32, 2.0))
    return ControlledEmptyRouter(assets(tmp_path, **asset_options), eta), actor


def test_scalar_tuple_actual_edge_parser_and_bounded_cache(tmp_path):
    routes, actor = router(tmp_path)
    result = routes.route(108.9, 34.2, 108.9012, 34.2011, STAMP)
    assert result["common_supported"]
    assert result["compatible_profiles"] == frozenset(("HV", "M", "A"))
    assert result["maneuver_types"] == ("LEFT",) and result["encounter_count"] == 1
    assert (result["raw_time_s"], result["duration_s"], result["distance_m"]) == (100., 200., 250.)
    assert result["points"] == result["geometry_points"] == POINTS
    assert result["snapped_origin"] == POINTS[0] and result["snap_gap_origin_m"] > 0
    assert result["snapped_target"] == POINTS[-1] and result["snap_gap_target_m"] > 0
    assert [edge["valhalla_directed_edge_id"] for edge in result["edges"]] == [10, 11, 12]
    assert [edge["stage3_edge_uid"] for edge in result["edges"]] == UIDS
    assert [(edge["begin_shape_index"], edge["end_shape_index"], edge["length_m"])
        for edge in result["edges"]] == [(0, 1, 80.), (1, 2, 80.), (2, 3, 80.)]
    assert result["traffic_unknown_share"] == 1.0 and not result["dynamic_evidence_complete"]
    assert result["dynamic_reason_codes"] == (DYNAMIC_UNKNOWN_REASON,)
    assert actor.route_requests[0]["costing"] == actor.trace_requests[0]["costing"] == "auto"
    assert actor.route_requests[0]["date_time"] == dict(type=1, value="2016-10-31T08:07")
    assert actor.trace_requests[0]["shape_match"] == "edge_walk"
    assert [tuple((p["lon"], p["lat"])) for p in actor.trace_requests[0]["shape"]] == POINTS
    result["points"].append((0., 0.))
    cached = routes.route(108.9, 34.2, 108.9012, 34.2011, STAMP)
    assert cached["cache_hit"] and cached["points"] == POINTS
    assert len(actor.route_requests) == len(actor.trace_requests) == 1
    routes._cache_limit = 2
    for minute in (1, 2, 3):
        routes.route(108.9, 34.2, 108.9012, 34.2011, STAMP + pd.Timedelta(minutes=minute))
    assert len(routes.cache) == 2
    diagnostic = routes.diagnostics()
    assert (diagnostic["route_requests"], diagnostic["scalar_route_calls"], diagnostic["edge_walk_calls"],
        diagnostic["route_cache_hits"]) == (5, 4, 4, 1)
    assert diagnostic["common_support_evaluations"] == dict(supported=4)
    assert diagnostic["common_failure_reason_counts"] == {}
    assert diagnostic["traffic_partition"] == dict(NC=0., ML=0., MR=0., CG=0., SC=0., U=1.)
    assert not diagnostic["dynamic_evidence_complete"]


@pytest.mark.parametrize("case", ["uturn", "outside_network", "missing_indices", "incomplete_shape", "unresolved"])
def test_uturn_and_incomplete_evidence_have_explicit_common_support(tmp_path, case):
    routes, actor = router(tmp_path, maneuver="UTURN", missing_movement=case == "unresolved",
        unresolved_geometry=case == "unresolved")
    actor.unknown_edge = case == "outside_network"
    actor.missing_indices = case == "missing_indices"
    actor.incomplete_shape = case == "incomplete_shape"
    result = routes.route(108.9, 34.2, 108.9012, 34.2011, STAMP)
    if case == "uturn":
        assert result["common_supported"]
        assert result["compatible_profiles"] == frozenset(("A", "HV"))
        assert result["maneuver_types"] == ("UTURN",)
        assert "UTURN_OUTSIDE_RESEARCH_PROFILE" in result["profile_reason_codes"]["M"]
    else:
        assert not result["common_supported"] and result["compatible_profiles"] == frozenset()
        assert result["reason_codes"] and "STRAIGHT" not in result["maneuver_types"]
        assert result["points"] == POINTS  # Failure does not replace the routed tuple.


@pytest.mark.parametrize("certified", [False, True])
def test_positive_control_common_prohibition_and_coincident_od(tmp_path, certified):
    routes, actor = router(tmp_path, signalized=True, certified=certified)
    result = routes.route(108.9, 34.2, 108.9012, 34.2011, STAMP)
    assert result["common_supported"] is not certified
    assert result["compatible_profiles"] == (frozenset() if certified else frozenset(("HV", "C", "M", "A")))
    if certified:
        assert result["reason_codes"] == ("CERTIFIED_COMMON_PROHIBITION",)
    same = routes.route(108.9, 34.2, 108.9, 34.2, STAMP)
    assert same["common_supported"] and same["coincident_od"]
    assert (same["raw_time_s"], same["duration_s"], same["distance_m"]) == (0., 0., 0.)
    assert len(actor.route_requests) == len(actor.trace_requests) == 1
