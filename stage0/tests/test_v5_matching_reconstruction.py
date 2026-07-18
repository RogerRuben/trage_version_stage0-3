from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from stage0.v5.matching import (
    BoundedSourceCache, Candidate, TransitionEngine, angular_difference,
    _ambiguity_flags, full_order_decision, local_failures_require_full_order, local_hmm_windows,
)
from stage0.v5.reconstruction import EdgeAwareRouter, build_movements, build_traversals, route_parts_frame
from stage0.v5.routing import CompactMovementRouter


def _edges():
    rows = []
    for uid, u, v, coords, parallel in [
        ("a", 1, 2, [(0, 0), (10, 0)], "p"),
        ("a2", 1, 2, [(0, 1), (10, 1)], "p"),
        ("b", 2, 3, [(10, 0), (20, 0)], None),
        ("c", 3, 4, [(20, 0), (30, 0)], None),
    ]:
        rows.append({"edge_uid": uid, "from_node": u, "to_node": v, "edge_key": uid, "geometry": LineString(coords), "length_m": 10.0, "candidate_penalty": 0.0, "parallel_group": parallel, "bridge": False, "tunnel": False, "layer": 0, "highway": "primary"})
    return gpd.GeoDataFrame(rows, geometry="geometry", crs=3857)


def _movements():
    return pd.DataFrame([
        {"from_edge_uid": "a", "via_node": 2, "to_edge_uid": "b", "movement_type": "straight", "turn_angle": 0.0, "restriction_status": "allowed", "layer_compatibility": True, "road_class_transition": "primary->primary", "merge_diverge_flag": False},
        {"from_edge_uid": "b", "via_node": 3, "to_edge_uid": "c", "movement_type": "straight", "turn_angle": 0.0, "restriction_status": "allowed", "layer_compatibility": True, "road_class_transition": "primary->primary", "merge_diverge_flag": False},
    ])


def _candidate(uid, index, position, heading=0.0):
    return Candidate(uid, index, position, 1.0, heading, 1.0, 1)


def test_candidate_preserves_edge_identity_and_heading_consistency():
    assert _candidate("a", 0, 2).edge_uid != _candidate("a2", 1, 2).edge_uid
    assert float(angular_difference(359, 1)) == 2.0


def test_local_hmm_window_has_anchors_and_merges_nearby_points():
    assert local_hmm_windows([False, True, False, True, False], 1, 2) == [(0, 5)]


def test_full_order_fallback_trigger_can_be_computed_from_ambiguity_share():
    flags = pd.Series([True, True, False, True]).to_numpy()
    assert full_order_decision(flags, {"full_order_ambiguity_share": 0.45, "full_order_min_windows": 4}) == (True, "raw_ambiguity_share")


def test_sparse_ambiguity_does_not_trigger_from_expanded_window_length():
    flags = pd.Series([False] * 20).to_numpy()
    flags[[2, 7, 12]] = True
    assert full_order_decision(flags, {"full_order_ambiguity_share": 0.45, "full_order_min_windows": 4, "full_order_min_window_share": 0.25}) == (False, "")


def test_one_local_failure_does_not_escalate_whole_order():
    assert not local_failures_require_full_order(1, {"full_order_min_windows": 4})
    assert local_failures_require_full_order(4, {"full_order_min_windows": 4})


def test_grade_signature_difference_requires_close_plausible_candidates():
    edges = _edges()
    edges.loc[1, "layer"] = 1
    rows = [[
        Candidate("a", 0, 2, 2, 5, 2, 1),
        Candidate("a2", 1, 2, 40, 5, 40, 2),
    ]]
    flags, reasons = _ambiguity_flags(rows, edges, distance_margin_m=5)
    assert not flags[0]
    assert "grade_separation" not in reasons[0]


def test_same_edge_transition_distance_uses_positions():
    engine = TransitionEngine(_edges(), _movements(), pd.DataFrame([{"from_node": 1, "to_node": 2, "routing_cost_m": 10.0}]), {"route_cache_sources": 2, "max_route_distance_m": 100.0})
    assert engine.distance(_candidate("a", 0, 2), _candidate("a", 0, 8)) == 6
    assert math.isinf(engine.distance(_candidate("a", 0, 8), _candidate("a", 0, 2)))


def test_same_edge_small_projection_jitter_is_allowed_with_penalty():
    engine = TransitionEngine(_edges(), _movements(), pd.DataFrame(), {"route_cache_sources": 2, "max_route_distance_m": 100.0, "same_edge_jitter_tolerance_m": 3.0, "same_edge_jitter_penalty_per_m": 0.5})
    distances, penalties = engine.transition_matrices([_candidate("a", 0, 8)], [_candidate("a", 0, 6)], 100)
    assert distances[0, 0] == 0
    assert penalties[0, 0] == 1.0


def test_adjacent_movement_distance_avoids_dijkstra():
    engine = TransitionEngine(_edges(), _movements(), pd.DataFrame(), {"route_cache_sources": 2, "max_route_distance_m": 100.0})
    assert engine.distance(_candidate("a", 0, 7), _candidate("b", 2, 4)) == 7
    assert engine.router.cache_size == 0


def test_bounded_routing_cache_only_materializes_requested_targets():
    import networkx as nx
    graph = nx.DiGraph([(1, 2, {"weight": 1.0}), (2, 3, {"weight": 1.0}), (3, 4, {"weight": 1.0})])
    cache = BoundedSourceCache(graph, max_sources=2, cutoff=10)
    cache.distances(1, [4]); cache.distances(2, [4]); cache.distances(3, [4])
    assert cache.size == 3


def test_multi_target_search_runs_once_per_source_and_stops_at_targets():
    router = CompactMovementRouter(_edges(), _movements(), {"max_route_distance_m": 100})
    result = router.multi_target_bridges(["a"], ["b", "c"], 100)
    assert result[("a", "b")][0] == ["a", "b"]
    assert result[("a", "c")][0] == ["a", "b", "c"]
    assert router.stats().calls == 1


def test_hmm_and_reconstruction_share_forbidden_movement_semantics():
    movements = _movements().iloc[[0]].copy()
    movements.loc[:, "restriction_status"] = "forbidden:no_left_turn"
    router = CompactMovementRouter(_edges(), movements, {"max_route_distance_m": 100})
    engine = TransitionEngine(_edges(), movements, pd.DataFrame(), {"max_route_distance_m": 100}, router)
    assert math.isinf(engine.distance(_candidate("a", 0, 5), _candidate("b", 2, 5), 100))
    assert EdgeAwareRouter(_edges(), movements, {"max_route_distance_m": 100}, router).bridge("a", "b") is None


def test_edge_aware_reconstruction_returns_concrete_inferred_edge():
    router = EdgeAwareRouter(_edges(), _movements(), {"u_turn_penalty_m": 500, "road_class_transition_penalty_m": 30, "max_route_distance_m": 100})
    matched = pd.DataFrame({"edge_uid": ["a", "c"]})
    route = router.reconstruct(matched)
    assert route.edge_uids == ["a", "b", "c"]
    assert route.observed == [True, False, True]


def test_inferred_path_has_no_realized_time_and_intervals_conserve():
    edges = _edges()
    route = EdgeAwareRouter(edges, _movements(), {"u_turn_penalty_m": 500, "road_class_transition_penalty_m": 30, "max_route_distance_m": 100}).reconstruct(pd.DataFrame({"edge_uid": ["a", "c"]}))
    parts = route_parts_frame("o", route, edges)
    matched = pd.DataFrame({"point_seq": [0, 1, 2], "timestamp": [0.0, 5.0, 12.0], "edge_uid": ["a", "a", "c"], "metric_x": [0.0, 5.0, 30.0], "metric_y": [0.0, 0.0, 0.0]})
    traversals = build_traversals("o", matched, parts, edges)
    turns = build_movements("o", parts, CompactMovementRouter(edges, _movements(), {"max_route_distance_m": 100}), matched)
    assert traversals.loc[traversals.is_interpolated, "travel_time_s"].isna().all()
    assert traversals.observed_interval_time_s.sum() + turns.observed_interval_time_s.sum() == 12.0
    assert abs(traversals.allocated_distance_m.sum() - 30.0) < 1e-9


def test_movement_output_uses_prebuilt_router_not_per_order_dataframe_index(monkeypatch):
    edges = _edges()
    movements = _movements()
    router = CompactMovementRouter(edges, movements, {"max_route_distance_m": 100})
    route = EdgeAwareRouter(edges, movements, {"max_route_distance_m": 100}, router).reconstruct(pd.DataFrame({"edge_uid": ["a", "c"]}))
    parts = route_parts_frame("o", route, edges)
    monkeypatch.setattr(pd.DataFrame, "set_index", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("per-order index build")))
    built = build_movements("o", parts, router)
    assert len(built) == 2
