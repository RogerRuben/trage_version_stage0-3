from __future__ import annotations

import pandas as pd

from stage0.v6.products import build_order_products
from stage0.v6.quality import evaluate_order_quality


def _source_points():
    return pd.DataFrame(
        {
            "order_id": ["o"] * 3,
            "subtrace_id": ["o:000"] * 3,
            "timestamp": [0.0, 10.0, 20.0],
            "step_distance_m": [0.0, 100.0, 100.0],
        }
    )


def _matched_points():
    return pd.DataFrame(
        {
            "order_id": ["o"] * 3,
            "subtrace_id": ["o:000"] * 3,
            "original_point_seq": [0, 1, 2],
            "timestamp": [0.0, 10.0, 20.0],
            "matched_point_status": ["matched", "matched", "interpolated"],
            "edge_index": [0, 2, 2],
            "distance_from_trace_point_m": [2.0, 3.0, 4.0],
            "route_discontinuity": [False, False, False],
        }
    )


def _route_parts():
    return pd.DataFrame(
        {
            "order_id": ["o"] * 3,
            "subtrace_id": ["o:000"] * 3,
            "path_id": [0, 0, 0],
            "route_sequence": [0, 1, 2],
            "valhalla_edge_index": [0, 1, 2],
            "valhalla_edge_id": ["v0", "v1", "v2"],
            "canonical_edge_uid": ["e0", "e1", "e2"],
            "canonical_from_node": [1, 2, 3],
            "canonical_to_node": [2, 3, 4],
            "entry_position_m": [0.0, 0.0, 0.0],
            "exit_position_m": [50.0, 100.0, 50.0],
            "length_m": [50.0, 100.0, 50.0],
            "route_source": ["observed", "inferred", "observed"],
            "is_interpolated": [False, True, False],
            "mapping_status": ["exact_edge_mapping"] * 3,
        }
    )


def test_inferred_edges_receive_no_dynamic_time_and_unresolved_is_retained():
    products = build_order_products(_source_points(), _matched_points(), _route_parts())
    traversals = products["link_traversals"]
    assert traversals.loc[traversals.edge_uid.eq("e1"), "travel_time_s"].iloc[0] == 0
    assert traversals.loc[traversals.edge_uid.eq("e1"), "traversal_source"].iloc[0] == "inferred"
    assert products["unresolved_intervals"].empty


def test_quality_uses_only_simple_valhalla_product_metrics():
    products = build_order_products(_source_points(), _matched_points(), _route_parts())
    thresholds = {
        "strict": {
            "minimum_matched_interval_share": 0.4,
            "maximum_snap_distance_p90_m": 10,
            "minimum_canonical_mapping_share": 0.99,
            "maximum_inferred_distance_share": 0.6,
            "maximum_od_endpoint_error_m": 10,
            "minimum_route_gps_ratio": 0.5,
            "maximum_route_gps_ratio": 2.0,
        },
        "analysis": {
            "minimum_matched_interval_share": 0.25,
            "minimum_canonical_mapping_share": 0.5,
            "maximum_od_endpoint_error_m": 100,
            "minimum_route_gps_ratio": 0.25,
            "maximum_route_gps_ratio": 4.0,
        },
    }
    quality = evaluate_order_quality(
        _source_points(),
        _matched_points(),
        _route_parts(),
        products["unresolved_intervals"],
        thresholds,
    )
    assert quality["route_quality"] == "strict_core"
    assert quality["inferred_distance_share"] == 0.5
    assert quality["unresolved_time_share"] == 0.0
    assert quality["matched_interval_share"] == 1.0
