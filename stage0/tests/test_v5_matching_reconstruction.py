from __future__ import annotations

import math

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from stage0.v5.matching import (
    Candidate, CandidateIndex, TransitionEngine, angular_difference,
    _ambiguity_flags, _gps_headings, _transition_ambiguity_flags, _viterbi,
    candidate_emission_cost, effective_ambiguity_flags, full_order_decision,
    local_failures_require_full_order, local_hmm_windows, transition_cutoff,
)
from stage0.v5.reconstruction import (
    EdgeAwareRouter,
    add_position_aware_route_distances,
    build_movements,
    build_traversals,
    build_unresolved_intervals,
    projected_route_distance_m,
    route_parts_frame,
)
from stage0.v5.routing import CompactMovementRouter
from stage0.v5.quality import evaluate_order_quality


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


def _matched_ac() -> pd.DataFrame:
    return pd.DataFrame({
        "point_seq": [0, 1],
        "edge_uid": ["a", "c"],
        "selected_path_json": ["", '["a","b","c"]'],
        "transition_cutoff_m": [math.nan, 100.0],
        "selected_path_distance_m": [math.nan, 30.0],
    })


def _quality_config() -> dict:
    return {
        "hard": {
            "maximum_heading_direction_difference_deg": 100,
            "same_edge_reverse_tolerance_m": 10,
            "same_edge_jitter_penalty_per_m": 0.25,
            "minimum_direction_displacement_m": 3,
            "minimum_consecutive_direction_conflicts": 2,
            "maximum_direction_violations": 0,
            "maximum_topology_gaps": 0,
            "maximum_unreasonable_detours": 0,
            "maximum_od_endpoint_error_m": 150,
            "maximum_illegal_u_turns": 0,
            "minimum_route_links": 1,
            "conservation_tolerance": 1e-6,
            "hmm_path_distance_tolerance_m": 1e-6,
            "maximum_layer_violations": 0,
        },
        "soft": {
            "maximum_route_length_ratio": 2.5,
            "maximum_fallback_share": 0.25,
            "maximum_p90_projection_distance_m": 60,
            "minimum_route_length_ratio": 0.5,
            "maximum_interpolated_distance_share": 0.5,
            "minimum_match_confidence": 0.35,
            "maximum_repeated_link_share": 0.2,
            "maximum_parallel_ambiguity_share": 0.25,
        },
    }


def test_candidate_preserves_edge_identity_and_heading_consistency():
    assert _candidate("a", 0, 2).edge_uid != _candidate("a2", 1, 2).edge_uid
    assert float(angular_difference(359, 1)) == 2.0


def test_local_hmm_window_has_anchors_and_merges_nearby_points():
    assert local_hmm_windows([False, True, False, True, False], 1, 2) == [(0, 5)]


def test_full_order_fallback_trigger_can_be_computed_from_ambiguity_share():
    flags = pd.Series([True, True, False, True]).to_numpy()
    assert full_order_decision(flags, {"full_order_ambiguity_share": 0.45, "full_order_min_windows": 4}) == (True, "raw_ambiguity_share")


def test_sparse_ambiguity_does_not_trigger_from_expanded_window_length():
    flags = pd.Series([False] * 20).to_numpy(copy=True)
    flags[[2, 7, 12]] = True
    assert full_order_decision(flags, {"full_order_ambiguity_share": 0.45, "full_order_min_windows": 4, "full_order_min_window_share": 0.25}) == (False, "")


def test_one_local_failure_does_not_escalate_whole_order():
    assert not local_failures_require_full_order(1, {"full_order_min_windows": 4})
    assert local_failures_require_full_order(4, {"full_order_min_windows": 4})


def test_grade_signature_difference_requires_close_plausible_candidates():
    edges = _edges()
    edges.loc[1, "layer"] = 1
    rows = [[
        Candidate("a", 0, 2, 2, 5, 2, 1, total_emission_cost=0.1),
        Candidate("a2", 1, 2, 40, 5, 40, 2, total_emission_cost=10.0),
    ]]
    flags, reasons = _ambiguity_flags(rows, edges, emission_margin_cost=0.5)
    assert not flags[0]
    assert "grade_separation" not in reasons[0]


def test_same_edge_transition_distance_uses_positions():
    engine = TransitionEngine(_edges(), _movements(), pd.DataFrame([{"from_node": 1, "to_node": 2, "routing_cost_m": 10.0}]), {"route_cache_sources": 2, "max_route_distance_m": 100.0})
    assert engine.distance(_candidate("a", 0, 2), _candidate("a", 0, 8)) == 6
    assert math.isinf(engine.distance(_candidate("a", 0, 8), _candidate("a", 0, 2)))


def test_same_edge_small_projection_jitter_is_allowed_with_penalty():
    engine = TransitionEngine(_edges(), _movements(), pd.DataFrame(), {"route_cache_sources": 2, "max_route_distance_m": 100.0, "same_edge_jitter_tolerance_m": 3.0, "same_edge_jitter_penalty_per_m": 0.5})
    distances, penalties, retry_attempted = engine.transition_matrices(
        [_candidate("a", 0, 8)], [_candidate("a", 0, 6)], 100
    )
    assert distances[0, 0] == 0
    assert penalties[0, 0] == 1.0
    assert not retry_attempted


def test_same_edge_selected_distance_and_jitter_are_independently_audited():
    edges = _edges()
    matched = pd.DataFrame({
        "point_seq": [0, 1],
        "timestamp": [0.0, 2.0],
        "edge_uid": ["a", "a"],
        "position_on_edge": [8.0, 6.0],
        "metric_x": [8.0, 6.0],
        "metric_y": [0.0, 0.0],
        "gps_to_edge_distance_m": [1.0, 1.0],
        "parallel_ambiguity": [False, False],
        "selected_path_json": ["", '["a"]'],
        "selected_path_distance_m": [math.nan, 0.0],
        # Two metres of reverse jitter at 0.5 cost/metre deliberately
        # disagrees with the frozen audit rate of 0.25.
        "selected_jitter_penalty_m": [math.nan, 1.0],
        "heading_reliable": [False, False],
        "edge_heading_difference_deg": [0.0, 0.0],
        "observed_step_m": [0.0, 2.0],
    })
    route = EdgeAwareRouter(
        edges, _movements(), {"max_route_distance_m": 100}
    ).reconstruct(matched)
    parts = route_parts_frame("o", route, edges)
    parts = add_position_aware_route_distances(matched, parts, edges)
    quality = evaluate_order_quality(
        "o",
        matched,
        parts,
        build_traversals("o", matched, parts, edges),
        pd.DataFrame(),
        edges,
        {"matching_mode": "local_hmm"},
        _quality_config(),
    )
    assert quality["hmm_path_distance_mismatch_count"] == 0
    assert quality["same_edge_jitter_mismatch_count"] == 1


def test_transition_ambiguity_is_not_masked_by_unreliable_heading():
    effective = effective_ambiguity_flags(
        pd.Series([True, False]).to_numpy(),
        pd.Series([False, True]).to_numpy(),
        pd.Series([False, False]).to_numpy(),
    )
    assert effective.tolist() == [False, True]


def test_context_expansion_remerges_overlapping_local_windows():
    flags = pd.Series([False] * 10).to_numpy(copy=True)
    flags[[2, 6]] = True
    assert local_hmm_windows(flags, context_points=2, merge_gap_points=2) == [
        (0, 9)
    ]


def test_transition_matrix_evidence_is_reused_within_an_order():
    engine = TransitionEngine(
        _edges(),
        _movements(),
        pd.DataFrame(),
        {"max_route_distance_m": 100, "same_edge_jitter_tolerance_m": 3},
    )
    engine.begin_order()
    previous = [_candidate("a", 0, 5)]
    current = [_candidate("c", 3, 5)]
    first = engine.transition_matrices(previous, current, 100)
    calls_after_first = engine.router.stats().path_calls
    second = engine.transition_matrices(previous, current, 100)
    assert first[0].tolist() == second[0].tolist()
    assert engine.router.stats().path_calls == calls_after_first
    assert engine.evidence_cache_stats() == (1, 1)


def test_viterbi_does_not_repeat_retry_already_done_by_matrix_builder(
    monkeypatch,
):
    engine = TransitionEngine(
        _edges(),
        _movements(),
        pd.DataFrame(),
        {
            "max_route_distance_m": 100,
            "transition_retry_max_m": 100,
            "transition_retry_multiplier": 1.5,
            "transition_retry_candidate_subset": 3,
        },
    )
    calls = 0
    original = engine.transition_matrices

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, "transition_matrices", counted)
    config = {
        "sigma_distance_m": 15.0,
        "beta_transition_m": 60.0,
        "beta_semantic_cost_m": 60.0,
        "transition_cutoff_min_m": 20.0,
        "transition_cutoff_max_m": 20.0,
        "transition_cutoff_alpha": 0.0,
        "transition_cutoff_base_m": 20.0,
        "transition_max_speed_mps": 40.0,
    }
    assert _viterbi(
        [[_candidate("c", 3, 5)], [_candidate("a", 0, 5)]],
        pd.Series([0.0, 1.0]).to_numpy(),
        pd.Series([0.0, 1.0]).to_numpy(),
        engine,
        config,
    ) is None
    assert calls == 1


def test_selected_transition_range_does_not_revalidate_whole_order():
    engine = TransitionEngine(
        _edges(),
        _movements(),
        pd.DataFrame(),
        {
            "max_route_distance_m": 100,
            "same_edge_jitter_tolerance_m": 3,
            "transition_cutoff_min_m": 100,
            "transition_cutoff_max_m": 100,
            "transition_cutoff_alpha": 0,
            "transition_cutoff_base_m": 100,
            "transition_max_speed_mps": 100,
        },
    )
    paths, _, failures = engine.selected_transition_paths(
        [
            _candidate("a", 0, 1),
            _candidate("a", 0, 5),
            _candidate("b", 2, 5),
            _candidate("c", 3, 5),
        ],
        pd.Series([0.0, 4.0, 10.0, 10.0]).to_numpy(),
        pd.Series([0.0, 1.0, 1.0, 1.0]).to_numpy(),
        engine.router.config,
        from_point_index=2,
        to_point_index=2,
    )
    assert not failures
    assert [path.point_index for path in paths] == [2]


def test_nonadjacent_pair_is_not_mislabeled_as_missing_topology():
    engine = TransitionEngine(
        _edges(), _movements(), pd.DataFrame(), {"max_route_distance_m": 100}
    )
    assert engine.raw_movement_status("a", "c") == (
        "no_direct_raw_movement_record"
    )


def test_adjacent_movement_distance_avoids_dijkstra():
    engine = TransitionEngine(_edges(), _movements(), pd.DataFrame(), {"route_cache_sources": 2, "max_route_distance_m": 100.0})
    assert engine.distance(_candidate("a", 0, 7), _candidate("b", 2, 4)) == 7
    assert engine.router.cache_size == 0


def test_multi_target_search_runs_once_per_source_and_stops_at_targets():
    router = CompactMovementRouter(_edges(), _movements(), {"max_route_distance_m": 100})
    result = router.multi_target_distances(["a"], ["b", "c"], 100)
    assert result[("a", "b")] == 0
    assert result[("a", "c")] == 10
    assert router.stats().distance_calls == 1
    assert router.stats().path_calls == 0


def test_transition_ambiguity_probe_uses_distance_frontier_not_path_search():
    engine = TransitionEngine(
        _edges(),
        _movements(),
        pd.DataFrame(),
        {
            "max_route_distance_m": 100.0,
            "transition_cutoff_alpha": 3.0,
            "transition_cutoff_base_m": 20.0,
            "transition_cutoff_max_m": 100.0,
            "transition_max_speed_mps": 40.0,
            "transition_ambiguity_path_gps_ratio": 3.0,
            "transition_ambiguity_inferred_m": 250.0,
        },
    )
    flags, _, evidence = _transition_ambiguity_flags(
        [_candidate("a", 0, 8), _candidate("c", 3, 2)],
        pd.Series([0.0, 20.0]).to_numpy(),
        pd.Series([0.0, 5.0]).to_numpy(),
        engine,
        engine.router.config,
    )
    assert not flags.any()
    assert evidence[0].edge_uids == ()
    assert engine.router.stats().distance_calls == 1
    assert engine.router.stats().path_calls == 0


def test_multi_target_path_search_stops_after_all_targets_are_settled():
    edges = _edges()
    extra_edges = []
    extra_movements = []
    previous = "a"
    previous_node = 2
    for index in range(20):
        uid = f"branch_{index}"
        next_node = 100 + index
        extra_edges.append({
            "edge_uid": uid,
            "from_node": previous_node,
            "to_node": next_node,
            "edge_key": uid,
            "geometry": LineString([(10 + index, 5), (11 + index, 5)]),
            "length_m": 50.0,
            "candidate_penalty": 100.0,
            "parallel_group": None,
            "bridge": False,
            "tunnel": False,
            "layer": 0,
            "highway": "service",
        })
        extra_movements.append({
            "from_edge_uid": previous,
            "via_node": previous_node,
            "to_edge_uid": uid,
            "movement_type": "straight",
            "turn_angle": 0.0,
            "restriction_status": "allowed",
            "layer_compatibility": True,
            "road_class_transition": "primary->service",
            "merge_diverge_flag": False,
        })
        previous = uid
        previous_node = next_node
    edges = pd.concat([
        edges,
        gpd.GeoDataFrame(extra_edges, geometry="geometry", crs=3857),
    ], ignore_index=True)
    movements = pd.concat(
        [_movements(), pd.DataFrame(extra_movements)], ignore_index=True
    )
    router = CompactMovementRouter(edges, movements, {
        "max_route_distance_m": 2000,
        "pareto_epsilon_m": 0.25,
        "pareto_max_labels_per_state": 12,
    })
    result = router.transition_paths_from_source("a", {"c": 100.0})
    assert result["c"] is not None
    assert list(result["c"].edge_uids) == ["a", "b", "c"]
    # The low-cost target is settled before the expensive service-road branch
    # is exhausted.  This assertion guards against reverting to a full-cutoff
    # Pareto expansion.
    assert router.stats().expanded_nodes < 10


def test_approximate_unresolved_path_is_not_negative_cached():
    router = CompactMovementRouter(
        _edges(),
        _movements(),
        {
            "max_route_distance_m": 100,
            "pareto_search_mode": "approximate",
            "pareto_epsilon_m": 0.25,
            "pareto_max_labels_per_state": 1,
        },
    )
    assert router.transition_paths_from_source("c", {"a": 100.0})["a"] is None
    assert not router._path_negative
    assert router.stats().approximate_unresolved == 1


def test_exact_unresolved_path_can_be_negative_cached():
    router = CompactMovementRouter(
        _edges(),
        _movements(),
        {
            "max_route_distance_m": 100,
            "pareto_search_mode": "exact",
            "pareto_epsilon_m": 0.25,
            "pareto_max_labels_per_state": 1,
        },
    )
    assert router.transition_paths_from_source("c", {"a": 100.0})["a"] is None
    assert router._path_negative
    assert router.stats().exact_path_calls == 1


def test_order_local_source_frontier_is_reused_for_new_targets():
    edges = pd.concat([
        _edges(),
        gpd.GeoDataFrame([{
            "edge_uid": "d", "from_node": 4, "to_node": 5, "edge_key": "d",
            "geometry": LineString([(30, 0), (40, 0)]), "length_m": 10.0,
            "candidate_penalty": 0.0, "parallel_group": None, "bridge": False,
            "tunnel": False, "layer": 0, "highway": "primary",
        }], geometry="geometry", crs=3857),
    ], ignore_index=True)
    movements = pd.concat([_movements(), pd.DataFrame([{
        "from_edge_uid": "c", "via_node": 4, "to_edge_uid": "d",
        "movement_type": "straight", "turn_angle": 0.0,
        "restriction_status": "allowed", "layer_compatibility": True,
        "road_class_transition": "primary->primary", "merge_diverge_flag": False,
    }])], ignore_index=True)
    router = CompactMovementRouter(edges, movements, {"max_route_distance_m": 100})
    router.begin_order()
    assert router.multi_target_distances(["a"], ["c"], 100)[("a", "c")] == 10
    first = router.stats()
    assert router.multi_target_distances(["a"], ["d"], 100)[("a", "d")] == 20
    second = router.stats()
    assert second.expanded_nodes - first.expanded_nodes <= 1
    router.begin_order()
    assert not router._distance_source_states


def test_negative_distance_cache_uses_actual_exhausted_cutoff():
    router = CompactMovementRouter(_edges(), _movements(), {"max_route_distance_m": 100})
    assert math.isinf(router.multi_target_distances(["a"], ["c"], 5)[("a", "c")])
    assert router.multi_target_distances(["a"], ["c"], 20)[("a", "c")] == 10


def test_positive_selected_path_cache_researches_larger_cutoff():
    router = CompactMovementRouter(_edges(), _movements(), {"max_route_distance_m": 100})
    assert router.bridge_path("a", "c", 20)[0] == ["a", "b", "c"]
    assert router.bridge_path("a", "c", 100)[0] == ["a", "b", "c"]
    assert router.stats().path_calls == 2


def test_negative_path_cache_can_expand_to_larger_cutoff():
    router = CompactMovementRouter(_edges(), _movements(), {"max_route_distance_m": 100})
    assert router.bridge_path("a", "c", 5) is None
    assert router.bridge_path("a", "c", 20)[0] == ["a", "b", "c"]


def _cutoff_choice_router() -> CompactMovementRouter:
    template = _edges().iloc[0].to_dict()
    specs = [
        ("a", 1, 2, 10.0, 1.0),
        ("short", 2, 3, 10.0, 10.0),
        ("long1", 2, 4, 20.0, 1.0),
        ("long2", 4, 3, 20.0, 1.0),
        ("d", 3, 5, 10.0, 1.0),
    ]
    rows = []
    for index, (uid, start, end, length, penalty) in enumerate(specs):
        rows.append({
            **template,
            "edge_uid": uid,
            "from_node": start,
            "to_node": end,
            "length_m": length,
            "routing_penalty": penalty,
            "geometry": LineString([(index * 10, 0), (index * 10 + length, 0)]),
        })
    edges = gpd.GeoDataFrame(rows, geometry="geometry", crs=3857)
    movements = pd.DataFrame([
        {
            "from_edge_uid": left, "via_node": via, "to_edge_uid": right,
            "movement_type": "straight", "turn_angle": 0.0,
            "restriction_status": "allowed", "layer_compatibility": True,
            "road_class_transition": "primary->primary",
            "merge_diverge_flag": False,
        }
        for left, via, right in [
            ("a", 2, "short"), ("short", 3, "d"),
            ("a", 2, "long1"), ("long1", 4, "long2"), ("long2", 3, "d"),
        ]
    ])
    return CompactMovementRouter(edges, movements, {"max_route_distance_m": 100})


def test_path_cache_small_then_large_matches_fresh_large():
    cached = _cutoff_choice_router()
    assert list(cached.transition_path("a", "d", 15).edge_uids) == ["a", "short", "d"]
    large_after_small = cached.transition_path("a", "d", 50)
    fresh_large = _cutoff_choice_router().transition_path("a", "d", 50)
    assert large_after_small == fresh_large
    assert list(large_after_small.edge_uids) == ["a", "long1", "long2", "d"]


def test_path_cache_large_then_small_matches_fresh_small():
    cached = _cutoff_choice_router()
    cached.transition_path("a", "d", 50)
    small_after_large = cached.transition_path("a", "d", 15)
    fresh_small = _cutoff_choice_router().transition_path("a", "d", 15)
    assert small_after_large == fresh_small


def test_hmm_and_output_use_the_same_generalized_transition_path():
    edges = gpd.GeoDataFrame([
        {**_edges().iloc[0].to_dict(), "edge_uid": "a", "from_node": 1, "to_node": 2, "length_m": 10.0, "routing_penalty": 1.0},
        {**_edges().iloc[0].to_dict(), "edge_uid": "low", "from_node": 2, "to_node": 3, "length_m": 5.0, "routing_penalty": 10.0},
        {**_edges().iloc[0].to_dict(), "edge_uid": "good1", "from_node": 2, "to_node": 4, "length_m": 7.0, "routing_penalty": 1.0},
        {**_edges().iloc[0].to_dict(), "edge_uid": "good2", "from_node": 4, "to_node": 3, "length_m": 7.0, "routing_penalty": 1.0},
        {**_edges().iloc[0].to_dict(), "edge_uid": "d", "from_node": 3, "to_node": 5, "length_m": 10.0, "routing_penalty": 1.0},
    ], geometry="geometry", crs=3857)
    movements = pd.DataFrame([
        {"from_edge_uid": left, "via_node": via, "to_edge_uid": right, "movement_type": "straight", "turn_angle": 0.0, "restriction_status": "allowed", "layer_compatibility": True, "road_class_transition": "primary->primary", "merge_diverge_flag": False}
        for left, via, right in [("a", 2, "low"), ("low", 3, "d"), ("a", 2, "good1"), ("good1", 4, "good2"), ("good2", 3, "d")]
    ])
    router = CompactMovementRouter(edges, movements, {"max_route_distance_m": 100})
    selected = router.transition_path("a", "d", 100)
    assert list(selected.edge_uids) == ["a", "good1", "good2", "d"]
    assert router.bridge_path("a", "d", 100)[0] == list(selected.edge_uids)
    assert router.bridge_path("a", "d", 100)[1] == selected.physical_distance_m


def test_batch_candidate_projection_deduplicates_candidate_alias(tmp_path):
    edges = _edges().iloc[:2].copy()
    edges["candidate_alias_uid"] = "a"
    config = {
        "spacing_complex_m": 6.0, "spacing_curve_m": 9.0, "spacing_straight_m": 22.5,
        "spacing_urban_m": 15.0, "radius_m": 80.0, "heading_weight_m": 15.0,
        "max_candidates": 10, "complex_candidates": 8, "dense_candidates": 5,
        "ordinary_candidates": 3,
    }
    index = CandidateIndex(edges, config, str(tmp_path / "candidate"), metric_crs="EPSG:3857")
    rows = index.candidates_batch(
        pd.Series([5.0]).to_numpy(), pd.Series([0.2]).to_numpy(), pd.Series([0.0]).to_numpy(),
    )
    assert len(rows[0]) == 1


def test_candidate_rank_and_truncation_use_hmm_emission_cost(tmp_path):
    template = _edges().iloc[0].to_dict()
    edges = gpd.GeoDataFrame([
        {
            **template,
            "edge_uid": "legacy_winner",
            "edge_key": "legacy_winner",
            "geometry": LineString([(0, 20), (20, 20)]),
            "length_m": 20.0,
            "candidate_penalty": 0.0,
            "parallel_group": None,
        },
        {
            **template,
            "edge_uid": "emission_winner",
            "edge_key": "emission_winner",
            "geometry": LineString([(0, 10), (20, 10)]),
            "length_m": 20.0,
            "candidate_penalty": 15.0,
            "parallel_group": None,
        },
    ], geometry="geometry", crs=3857)
    config = {
        "spacing_complex_m": 6.0,
        "spacing_curve_m": 9.0,
        "spacing_straight_m": 22.5,
        "spacing_urban_m": 15.0,
        "radius_m": 80.0,
        "heading_weight_m": 15.0,
        "heading_log_cost_max": 1.0,
        "edge_prior_scale_m": 30.0,
        "max_candidates": 1,
        "absolute_max_candidates": 20,
        "complex_candidates": 1,
        "dense_candidates": 1,
        "ordinary_candidates": 1,
    }
    index = CandidateIndex(
        edges,
        config,
        str(tmp_path / "emission_candidate"),
        metric_crs="EPSG:3857",
        emission_config={"sigma_distance_m": 15.0},
    )
    candidates = index.candidates_batch(
        pd.Series([5.0]).to_numpy(),
        pd.Series([0.0]).to_numpy(),
        pd.Series([0.0]).to_numpy(),
    )[0]
    assert candidates[0].edge_uid == "emission_winner"
    assert candidates[0].score > 20.0
    assert candidates[0].total_emission_cost < 0.8
    assert candidate_emission_cost(candidates[0], 15.0) == (
        candidates[0].total_emission_cost
    )


def test_recovery_candidate_override_is_not_clipped_by_normal_limit(tmp_path):
    template = _edges().iloc[0].to_dict()
    edges = gpd.GeoDataFrame([
        {
            **template,
            "edge_uid": f"edge_{index}",
            "edge_key": f"edge_{index}",
            "geometry": LineString([(0, index), (20, index)]),
            "length_m": 20.0,
            "candidate_penalty": 0.0,
            "parallel_group": None,
        }
        for index in range(15)
    ], geometry="geometry", crs=3857)
    config = {
        "spacing_complex_m": 6.0,
        "spacing_curve_m": 9.0,
        "spacing_straight_m": 22.5,
        "spacing_urban_m": 15.0,
        "radius_m": 80.0,
        "heading_weight_m": 15.0,
        "heading_log_cost_max": 1.0,
        "edge_prior_scale_m": 30.0,
        "max_candidates": 10,
        "absolute_max_candidates": 20,
        "complex_candidates": 8,
        "dense_candidates": 5,
        "ordinary_candidates": 3,
    }
    index = CandidateIndex(
        edges,
        config,
        str(tmp_path / "recovery_candidate"),
        metric_crs="EPSG:3857",
        emission_config={"sigma_distance_m": 15.0},
    )
    candidates = index.candidates_batch(
        pd.Series([5.0]).to_numpy(),
        pd.Series([0.0]).to_numpy(),
        pd.Series([0.0]).to_numpy(),
        candidate_limit_override=20,
    )[0]
    assert len(candidates) == 15


def test_stationary_heading_contributes_no_direction_evidence():
    headings, reliable = _gps_headings(
        pd.Series([0.0, 5.0, 5.1, 5.2, 10.0]).to_numpy(),
        pd.Series([0.0, 0.0, 0.0, 0.0, 0.0]).to_numpy(),
        2.0,
    )
    assert not reliable[2]
    assert headings[2] == 0.0
    emission_flags = pd.Series(
        [False, False, True, False, False]
    ).to_numpy()
    flags = effective_ambiguity_flags(
        emission_flags,
        pd.Series([False] * 5).to_numpy(),
        reliable,
    )
    assert full_order_decision(
        flags, {"full_order_ambiguity_share": 0.45}
    ) == (False, "")


def test_direct_movement_returns_complete_generalized_cost():
    edges = _edges()
    edges["routing_penalty"] = 1.0
    edges.loc[edges.edge_uid.eq("b"), "routing_penalty"] = 2.0
    router = CompactMovementRouter(edges, _movements(), {"max_route_distance_m": 100})
    direct = router.transition_path("a", "b", 0)
    assert direct is not None
    assert direct.physical_distance_m == 0.0
    assert direct.generalized_routing_cost == 20.0


def test_selected_direct_transition_respects_endpoint_cutoff():
    engine = TransitionEngine(
        _edges(), _movements(), pd.DataFrame(),
        {"max_route_distance_m": 100, "transition_cutoff_min_m": 5,
         "transition_cutoff_max_m": 5, "transition_cutoff_alpha": 0,
         "transition_cutoff_base_m": 5, "transition_max_speed_mps": 100},
    )
    selected = [_candidate("a", 0, 1), _candidate("b", 2, 9)]
    paths, _, failures = engine.selected_transition_paths(
        selected,
        pd.Series([0.0, 1.0]).to_numpy(),
        pd.Series([0.0, 1.0]).to_numpy(),
        {"max_route_distance_m": 100, "transition_cutoff_min_m": 5,
         "transition_cutoff_max_m": 5, "transition_cutoff_alpha": 0,
         "transition_cutoff_base_m": 5, "transition_max_speed_mps": 100},
    )
    assert not paths
    assert failures[0].reason == "endpoint_distance_exceeds_retry_cutoff"


def test_failed_match_position_and_inferred_audits_are_not_applicable():
    quality = evaluate_order_quality(
        "o",
        pd.DataFrame({
            "metric_x": [0.0, 1.0], "metric_y": [0.0, 0.0],
            "timestamp": [0.0, 1.0], "parallel_ambiguity": [False, False],
        }),
        pd.DataFrame(),
        pd.DataFrame(),
        pd.DataFrame(),
        _edges(),
        {"matching_mode": "failed_no_continuous_route"},
        {
            "hard": {
                "maximum_heading_direction_difference_deg": 100,
                "same_edge_reverse_tolerance_m": 10,
                "minimum_direction_displacement_m": 3,
                "minimum_consecutive_direction_conflicts": 2,
                "maximum_direction_violations": 0,
                "maximum_topology_gaps": 0,
                "maximum_unreasonable_detours": 0,
                "maximum_od_endpoint_error_m": 150,
                "maximum_illegal_u_turns": 0,
                "minimum_route_links": 1,
                "conservation_tolerance": 1e-6,
                "maximum_layer_violations": 0,
            },
            "soft": {
                "maximum_route_length_ratio": 2.5,
                "maximum_fallback_share": 0.25,
                "maximum_p90_projection_distance_m": 60,
                "minimum_route_length_ratio": 0.5,
                "maximum_interpolated_distance_share": 0.5,
                "minimum_match_confidence": 0.35,
                "maximum_repeated_link_share": 0.2,
                "maximum_parallel_ambiguity_share": 0.25,
            },
        },
    )
    assert quality["position_audit_applicable_order_count"] == 0
    assert quality["position_audit_not_applicable_match_failure_count"] == 1
    assert quality["actual_invalid_position_event_count"] == 0
    assert math.isnan(quality["interpolated_distance_share"])


def test_hmm_and_reconstruction_share_forbidden_movement_semantics():
    movements = _movements().iloc[[0]].copy()
    movements.loc[:, "restriction_status"] = "forbidden:no_left_turn"
    router = CompactMovementRouter(_edges(), movements, {"max_route_distance_m": 100})
    engine = TransitionEngine(_edges(), movements, pd.DataFrame(), {"max_route_distance_m": 100}, router)
    assert math.isinf(engine.distance(_candidate("a", 0, 5), _candidate("b", 2, 5), 100))
    assert EdgeAwareRouter(_edges(), movements, {"max_route_distance_m": 100}, router).bridge("a", "b") is None
    assert router.raw_movement("a", "b").restriction_status == "forbidden:no_left_turn"


def test_local_viterbi_respects_fixed_endpoint_anchors():
    edges = _edges()
    router = CompactMovementRouter(edges, _movements(), {"max_route_distance_m": 100})
    engine = TransitionEngine(
        edges, _movements(), pd.DataFrame(),
        {"max_route_distance_m": 100, "same_edge_jitter_tolerance_m": 10},
        router,
    )
    start = _candidate("a", 0, 2)
    end = _candidate("b", 2, 5)
    rows = [[start, _candidate("a2", 1, 2)], [end]]
    config = {
        "sigma_distance_m": 15.0, "beta_transition_m": 60.0,
        "beta_semantic_cost_m": 60.0, "beta_speed_difference_mps": 5.0,
        "max_route_distance_m": 100.0, "transition_cutoff_min_m": 100.0,
        "transition_cutoff_max_m": 100.0, "transition_cutoff_alpha": 3.0,
        "transition_cutoff_base_m": 20.0, "transition_max_speed_mps": 40.0,
        "anchor_position_tolerance_m": 1.0,
    }
    solved = _viterbi(
        rows, pd.Series([0.0, 13.0]).to_numpy(),
        pd.Series([0.0, 2.0]).to_numpy(), engine, config,
        start_anchor=start, end_anchor=end,
    )
    assert solved is not None
    assert solved[0][0].edge_uid == "a"
    assert solved[0][-1].edge_uid == "b"


def test_dynamic_transition_cutoff_records_spatial_and_time_bounds():
    config = {
        "max_route_distance_m": 6000.0,
        "transition_cutoff_alpha": 3.0,
        "transition_cutoff_base_m": 200.0,
        "transition_cutoff_min_m": 300.0,
        "transition_cutoff_max_m": 6000.0,
        "transition_max_speed_mps": 40.0,
    }
    assert transition_cutoff(50.0, 10.0, config) == 350.0
    assert transition_cutoff(5000.0, 1000.0, config) == 6000.0


def test_edge_aware_reconstruction_returns_concrete_inferred_edge():
    router = EdgeAwareRouter(_edges(), _movements(), {"u_turn_penalty_m": 500, "road_class_transition_penalty_m": 30, "max_route_distance_m": 100})
    matched = _matched_ac()
    route = router.reconstruct(matched)
    assert route.edge_uids == ["a", "b", "c"]
    assert route.observed == [True, False, True]


def test_inferred_path_has_no_realized_time_and_intervals_conserve():
    edges = _edges()
    route = EdgeAwareRouter(edges, _movements(), {"u_turn_penalty_m": 500, "road_class_transition_penalty_m": 30, "max_route_distance_m": 100}).reconstruct(_matched_ac())
    parts = route_parts_frame("o", route, edges)
    matched = pd.DataFrame({
        "point_seq": [0, 1, 2], "timestamp": [0.0, 5.0, 12.0],
        "edge_uid": ["a", "a", "c"], "position_on_edge": [0.0, 5.0, 10.0],
        "metric_x": [0.0, 5.0, 30.0], "metric_y": [0.0, 0.0, 0.0],
    })
    parts = add_position_aware_route_distances(matched, parts, edges)
    traversals = build_traversals("o", matched, parts, edges)
    movement_router = CompactMovementRouter(edges, _movements(), {"max_route_distance_m": 100})
    turns = build_movements("o", parts, movement_router, matched)
    unresolved = build_unresolved_intervals("o", parts, movement_router, matched)
    assert traversals.loc[traversals.is_interpolated, "travel_time_s"].isna().all()
    assert turns.observed_interval_time_s.sum() == 0.0
    assert unresolved.unresolved_interval_time_s.sum() == 7.0
    assert traversals.observed_interval_time_s.sum() + unresolved.unresolved_interval_time_s.sum() == 12.0
    assert abs(traversals.allocated_distance_m.sum() - 30.0) < 1e-9


def test_repeated_edge_visits_create_distinct_traversals():
    edges = gpd.GeoDataFrame([
        {"edge_uid": uid, "from_node": i, "to_node": i + 1, "edge_key": uid,
         "geometry": LineString([(i * 10, 0), ((i + 1) * 10, 0)]), "length_m": 10.0}
        for i, uid in enumerate(["a", "b", "c", "d"])
    ], geometry="geometry", crs=3857)
    parts = pd.DataFrame({
        "order_id": "o", "route_sequence": range(5), "edge_uid": ["a", "b", "c", "b", "d"],
        "is_interpolated": False, "route_source": "observed",
    })
    matched = pd.DataFrame({
        "point_seq": range(5), "timestamp": [0.0, 2.0, 5.0, 9.0, 14.0],
        "edge_uid": ["a", "b", "c", "b", "d"], "position_on_edge": [2.0, 3.0, 4.0, 6.0, 7.0],
        "metric_x": [2.0, 13.0, 24.0, 16.0, 37.0], "metric_y": 0.0,
    })
    parts = add_position_aware_route_distances(matched, parts, edges)
    traversals = build_traversals("o", matched, parts, edges)
    visits = traversals.loc[traversals.edge_uid.eq("b")]
    assert visits.traversal_id.tolist() == [1, 3]
    assert visits.route_sequence.tolist() == [1, 3]
    assert visits.enter_time.tolist() == [2.0, 9.0]


def test_first_and_last_edges_use_projected_positions():
    edges = _edges()
    route = EdgeAwareRouter(edges, _movements(), {"max_route_distance_m": 100}).reconstruct(
        _matched_ac()
    )
    parts = route_parts_frame("o", route, edges)
    matched = pd.DataFrame({
        "point_seq": [0, 1, 2], "timestamp": [0.0, 2.0, 8.0],
        "edge_uid": ["a", "a", "c"], "position_on_edge": [2.0, 8.0, 4.0],
        "metric_x": [2.0, 8.0, 24.0], "metric_y": 0.0,
    })
    parts = add_position_aware_route_distances(matched, parts, edges)
    assert parts.allocated_distance_m.tolist() == [8.0, 10.0, 4.0]
    assert projected_route_distance_m(matched, parts, edges) == 22.0
    traversals = build_traversals("o", matched, parts, edges)
    assert traversals.allocated_distance_m.sum() == 22.0


def test_same_edge_short_order_uses_position_delta():
    edges = _edges()
    route = EdgeAwareRouter(edges, _movements(), {"max_route_distance_m": 100}).reconstruct(
        pd.DataFrame({"edge_uid": ["a", "a"]})
    )
    parts = route_parts_frame("o", route, edges)
    matched = pd.DataFrame({
        "point_seq": [0, 1], "timestamp": [0.0, 3.0], "edge_uid": ["a", "a"],
        "position_on_edge": [2.0, 8.0], "metric_x": [2.0, 8.0], "metric_y": [0.0, 0.0],
    })
    parts = add_position_aware_route_distances(matched, parts, edges)
    assert parts.allocated_distance_m.tolist() == [6.0]


def test_movement_output_uses_prebuilt_router_not_per_order_dataframe_index(monkeypatch):
    edges = _edges()
    movements = _movements()
    router = CompactMovementRouter(edges, movements, {"max_route_distance_m": 100})
    route = EdgeAwareRouter(edges, movements, {"max_route_distance_m": 100}, router).reconstruct(_matched_ac())
    parts = route_parts_frame("o", route, edges)
    monkeypatch.setattr(pd.DataFrame, "set_index", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("per-order index build")))
    built = build_movements("o", parts, router)
    assert len(built) == 2
