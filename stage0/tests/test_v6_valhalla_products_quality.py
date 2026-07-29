from __future__ import annotations

import numpy as np
import pandas as pd

from stage0.v6.products import build_order_products
from stage0.v6.pipeline import _concat, _write_product
from stage0.v6.quality import (
    evaluate_dynamic_measurement_quality,
    evaluate_route_quality,
)


def _source(
    timestamps=(0.0, 10.0),
    subtraces=None,
    usable=None,
    distances=None,
):
    size = len(timestamps)
    return pd.DataFrame(
        {
            "order_id": ["o"] * size,
            "subtrace_id": subtraces or ["o:000"] * size,
            "original_point_seq": list(range(size)),
            "timestamp": list(timestamps),
            "step_distance_m": distances or [0.0] + [100.0] * (size - 1),
            "usable_subtrace": usable or [True] * size,
            "preprocess_break_before": [False] * size,
        }
    )


def _matched(edge_indices, timestamps=None, statuses=None, positions=None):
    size = len(edge_indices)
    timestamps = timestamps or [float(index * 10) for index in range(size)]
    return pd.DataFrame(
        {
            "order_id": ["o"] * size,
            "subtrace_id": ["o:000"] * size,
            "original_point_seq": list(range(size)),
            "timestamp": timestamps,
            "matched_point_status": statuses or ["matched"] * size,
            "edge_index": edge_indices,
            "percent_along": positions or np.linspace(0.1, 0.9, size).tolist(),
            "distance_from_trace_point_m": [2.0] * size,
            "route_discontinuity": [False] * size,
        }
    )


def _routes(edge_indices, sources=None, edge_uids=None):
    size = len(edge_indices)
    sources = sources or ["observed"] * size
    edge_uids = edge_uids or [f"e{index}" for index in edge_indices]
    return pd.DataFrame(
        {
            "order_id": ["o"] * size,
            "subtrace_id": ["o:000"] * size,
            "path_id": [0] * size,
            "route_sequence": list(range(size)),
            "valhalla_edge_index": edge_indices,
            "valhalla_edge_id": [f"v{index}" for index in edge_indices],
            "canonical_edge_uid": edge_uids,
            "canonical_from_node": list(range(1, size + 1)),
            "canonical_to_node": list(range(2, size + 2)),
            "entry_position_m": [0.0] * size,
            "exit_position_m": [100.0] * size,
            "length_m": [100.0] * size,
            "source_percent_along": [0.0] * size,
            "target_percent_along": [1.0] * size,
            "route_source": sources,
            "is_interpolated": [value == "inferred" for value in sources],
            "mapping_status": ["exact_edge_mapping"] * size,
            "engine_allocated_travel_time_s": [np.nan] * size,
            "valhalla_edge_elapsed_time_s": [np.nan] * size,
        }
    )


def _thresholds():
    return {
        "strict": {
            "minimum_matched_interval_share": 0.4,
            "maximum_snap_distance_p90_m": 10,
            "minimum_canonical_mapping_share": 0.99,
            "maximum_inferred_distance_share": 0.6,
            "maximum_od_endpoint_error_m": 10,
            "minimum_route_gps_ratio": 0.5,
            "maximum_route_gps_ratio": 3.0,
            "maximum_preprocess_break_time_share": 0.2,
        },
        "analysis": {
            "minimum_matched_interval_share": 0.25,
            "minimum_canonical_mapping_share": 0.5,
            "maximum_od_endpoint_error_m": 100,
            "minimum_route_gps_ratio": 0.25,
            "maximum_route_gps_ratio": 4.0,
            "maximum_preprocess_break_time_share": 0.5,
        },
        "dynamic_strict": {
            "minimum_direct_observed_interval_time_share": 0.7,
            "maximum_unresolved_time_share": 0.2,
        },
        "dynamic_partial": {
            "minimum_direct_observed_interval_time_share": 0.0,
        },
    }


def test_same_edge_consecutive_gps_points_create_direct_observed_time():
    products = build_order_products(_source(), _matched([0, 0]), _routes([0]))
    traversal = products["link_traversals"].iloc[0]
    assert traversal.measurement_source == "direct_observed"
    assert traversal.enter_time == 0.0
    assert traversal.exit_time == 10.0
    assert traversal.observed_travel_time_s == 10.0
    observation = products["link_interval_observations"].iloc[0]
    assert observation.gps_interval_id == 0
    assert observation.traversal_id == traversal.traversal_id
    assert observation.measurement_source == "direct_observed"
    assert observation.label_valid
    accounting = products["interval_accounting"].iloc[0]
    assert accounting.duplicate_interval_allocation_count == 0
    assert accounting.non_direct_observed_time_violation_count == 0
    assert accounting.distance_conservation_valid
    assert products["unresolved_intervals"].empty


def test_multi_edge_interval_does_not_allocate_direct_link_time():
    products = build_order_products(
        _source(), _matched([0, 1]), _routes([0, 1])
    )
    assert products["link_traversals"].observed_travel_time_s.isna().all()
    interval = products["interval_measurements"].iloc[0]
    assert interval.measurement_source == "interval_supported"
    assert interval.interval_reason == "multi_edge_interval_without_direct_timing"
    assert products["unresolved_intervals"].iloc[0].measurement_source == "interval_supported"


def test_intermediate_inferred_edge_makes_whole_interval_unresolved():
    products = build_order_products(
        _source(),
        _matched([0, 2]),
        _routes([0, 1, 2], ["observed", "inferred", "observed"]),
    )
    assert products["link_traversals"].observed_travel_time_s.isna().all()
    interval = products["interval_measurements"].iloc[0]
    assert interval.measurement_source == "unresolved"
    assert interval.interval_reason == "inferred_path_between_gps_anchors"
    assert interval.unresolved_time_s == 10.0


def test_unresolved_gap_does_not_shift_later_traversal_timestamp():
    source = _source((0.0, 10.0, 30.0, 40.0))
    matched = _matched(
        [0, 0, 2, 2],
        timestamps=[0.0, 10.0, 30.0, 40.0],
        positions=[0.1, 0.9, 0.1, 0.9],
    )
    products = build_order_products(
        source,
        matched,
        _routes([0, 1, 2], ["observed", "inferred", "observed"]),
    )
    later = products["link_traversals"].loc[
        products["link_traversals"].edge_uid.eq("e2")
    ].iloc[0]
    assert later.enter_time == 30.0
    assert later.exit_time == 40.0


def test_unusable_short_subtrace_is_unresolved():
    source = _source(usable=[False, False])
    products = build_order_products(
        source,
        _matched([pd.NA, pd.NA], statuses=["unmatched", "unmatched"]),
        _routes([]),
    )
    assert products["unresolved_intervals"].iloc[0].unresolved_reason == (
        "unusable_short_subtrace"
    )


def test_repeated_edge_visits_keep_distinct_traversal_ids():
    products = build_order_products(
        _source((0.0, 10.0, 20.0, 30.0, 40.0)),
        _matched([0, 1, 2, 3, 4]),
        _routes([0, 1, 2, 3, 4], edge_uids=["A", "B", "C", "B", "D"]),
    )
    visits = products["link_traversals"].loc[
        products["link_traversals"].edge_uid.eq("B")
    ]
    assert len(visits) == 2
    assert visits.traversal_id.nunique() == 2


def test_time_categories_are_mutually_exclusive_and_conserved():
    source = _source((0.0, 10.0, 20.0, 30.0))
    matched = _matched(
        [0, 0, 1, 3],
        timestamps=[0.0, 10.0, 20.0, 30.0],
        positions=[0.1, 0.9, 0.2, 0.8],
    )
    products = build_order_products(
        source,
        matched,
        _routes([0, 1, 2, 3], ["observed", "observed", "inferred", "observed"]),
    )
    accounting = products["interval_accounting"].iloc[0]
    assert accounting.time_conservation_valid
    assert abs(accounting.time_conservation_error_s) <= 1e-6
    categories = products["interval_measurements"][
        [
            "direct_observed_travel_time_s",
            "interval_supported_time_s",
            "engine_allocated_only_time_s",
            "unresolved_time_s",
        ]
    ].notna().sum(axis=1)
    assert categories.eq(1).all()
    assert products["link_interval_observations"].gps_interval_id.is_unique
    assert (
        products["link_interval_observations"].measurement_source
        .eq("direct_observed")
        .all()
    )


def test_route_quality_and_dynamic_quality_can_differ():
    products = build_order_products(
        _source(), _matched([0, 1]), _routes([0, 1])
    )
    route = evaluate_route_quality(
        _source(),
        _matched([0, 1]),
        products["route_parts"],
        products["interval_measurements"],
        _thresholds(),
    )
    dynamic = evaluate_dynamic_measurement_quality(
        products["route_parts"],
        products["link_traversals"],
        products["interval_measurements"],
        products["interval_accounting"],
        _thresholds(),
    )
    assert route["route_quality"] == "strict_core"
    assert dynamic["dynamic_measurement_quality"] == "dynamic_unusable"


def test_route_ratio_uses_resolved_subtrace_distance():
    source = _source(
        (0.0, 10.0, 1000.0),
        subtraces=["o:000", "o:000", "o:001"],
        distances=[0.0, 100.0, 10000.0],
    )
    source.loc[2, "preprocess_break_before"] = True
    matched = _matched([0, 0, pd.NA], timestamps=[0.0, 10.0, 1000.0])
    matched.loc[2, "subtrace_id"] = "o:001"
    matched.loc[2, "matched_point_status"] = "unmatched"
    breaks = pd.DataFrame(
        [
            {
                "from_original_point_seq": 1,
                "to_original_point_seq": 2,
                "break_reason": "preprocess_time_gap",
            }
        ]
    )
    products = build_order_products(
        source, matched, _routes([0]), preprocess_breaks=breaks
    )
    route = evaluate_route_quality(
        source,
        matched,
        products["route_parts"],
        products["interval_measurements"],
        _thresholds(),
    )
    assert route["resolved_subtrace_gps_distance_m"] == 100.0
    assert route["route_resolved_gps_distance_ratio"] == 1.0
    assert route["route_raw_gps_distance_ratio"] < 0.02


def test_movement_provenance_has_no_synthetic_delay():
    products = build_order_products(
        _source(), _matched([0, 1]), _routes([0, 1])
    )
    movement = products["turn_movements"].iloc[0]
    assert movement.movement_source == "directly_observed_transition"
    assert pd.isna(movement.movement_travel_time_s)
    assert pd.isna(movement.movement_delay_s)


def test_primary_topology_components_are_kept_without_cross_gap_movement():
    routes = _routes([0, 1])
    routes["valhalla_path_id"] = [0, 0]
    routes["path_id"] = [0, 1]
    matched = _matched([0, 1])
    matched["route_discontinuity"] = [True, True]
    products = build_order_products(_source(), matched, routes)
    assert products["route_parts"].canonical_edge_uid.tolist() == ["e0", "e1"]
    assert products["link_traversals"].edge_uid.tolist() == ["e0", "e1"]
    assert products["turn_movements"].empty


def test_inferred_and_unknown_dynamic_fields_are_nan_not_zero():
    products = build_order_products(
        _source(),
        _matched([0, 2]),
        _routes([0, 1, 2], ["observed", "inferred", "observed"]),
    )
    inferred = products["link_traversals"].loc[
        products["link_traversals"].measurement_source.eq("engine_interpolated")
    ]
    assert inferred.observed_travel_time_s.isna().all()
    assert inferred.engine_allocated_travel_time_s.isna().all()
    assert inferred.travel_time_s.isna().all()
    assert (
        products["interval_accounting"]
        .inferred_edge_observed_time_violation_count.eq(0)
        .all()
    )


def test_bucket_write_preserves_single_batch_columns_and_values(tmp_path):
    first = pd.DataFrame({"order_id": ["a"], "value": [1.0]})
    second = pd.DataFrame({"order_id": ["b"], "value": [2.0]})
    single_batch = pd.concat([first, second], ignore_index=True)
    bucket_batch = _concat([first, second])
    target = tmp_path / "day=20161010" / "part=000.parquet"
    _write_product(bucket_batch, target)
    restored = pd.read_parquet(target)
    assert restored.columns.tolist() == single_batch.columns.tolist()
    assert restored.dtypes.astype(str).tolist() == single_batch.dtypes.astype(str).tolist()
    pd.testing.assert_frame_equal(restored, single_batch)


def test_repeated_product_build_is_field_identical():
    arguments = (_source(), _matched([0, 1]), _routes([0, 1]))
    left = build_order_products(*arguments)
    right = build_order_products(*arguments)
    for product in left:
        pd.testing.assert_frame_equal(left[product], right[product])
