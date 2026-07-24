from __future__ import annotations

import pandas as pd

from stage0.v6.canonical_mapper import CanonicalEdgeMapper
from stage0.v6.parser import MATCHED_POINT_COLUMNS, ROUTE_PART_COLUMNS, parse_trace_attributes


def test_parser_emits_required_point_and_directed_edge_fields():
    source = pd.DataFrame(
        {
            "original_point_seq": [7, 8],
            "timestamp": [100, 103],
        }
    )
    raw = {
        "edges": [
            {
                "id": 99,
                "way_id": 10,
                "node_id": 1,
                "end_node": {"node_id": 2, "elapsed_time": 3.0},
                "forward": True,
                "length": 0.123,
                "source_percent_along": 0.25,
                "road_class": "primary",
                "bridge": True,
                "tunnel": False,
                "speed_limit": 50,
            }
        ],
        "matched_points": [
            {
                "type": "matched",
                "edge_index": 0,
                "distance_along_edge": 0.25,
                "distance_from_trace_point": 2.5,
                "lon": 108.0,
                "lat": 34.0,
            },
            {
                "type": "interpolated",
                "edge_index": 0,
                "distance_along_edge": 0.75,
                "distance_from_trace_point": 4.0,
                "lon": 108.1,
                "lat": 34.1,
                "end_route_discontinuity": True,
            },
        ],
    }
    matched, routes = parse_trace_attributes(
        raw, source, order_id="o", subtrace_id="o:000"
    )
    assert list(matched.columns) == MATCHED_POINT_COLUMNS
    assert list(routes.columns) == ROUTE_PART_COLUMNS
    assert matched.loc[0, "begin_osm_node_id"] == 1
    assert matched.loc[0, "end_osm_node_id"] == 2
    assert matched.loc[1, "route_discontinuity"]
    assert routes.loc[0, "length_m"] == 123.0
    assert routes.loc[0, "source_percent_along"] == 0.25
    assert routes.loc[0, "target_percent_along"] == 1.0
    assert routes.loc[0, "route_source"] == "observed"


def test_parser_normalizes_valhalla_uint64_edge_sentinel_to_missing():
    source = pd.DataFrame({"original_point_seq": [1], "timestamp": [100]})
    raw = {
        "edges": [{"id": 1, "way_id": 2, "node_id": 3, "end_node": {"node_id": 4}}],
        "matched_points": [
            {
                "type": "interpolated",
                "edge_index": 18_446_744_073_709_551_615,
            }
        ],
    }
    matched, _ = parse_trace_attributes(
        raw, source, order_id="o", subtrace_id="o:000"
    )
    assert pd.isna(matched.loc[0, "edge_index"])


def test_parser_converts_cumulative_elapsed_time_to_edge_increments():
    source = pd.DataFrame(
        {"original_point_seq": [0, 1], "timestamp": [0, 12]}
    )
    raw = {
        "edges": [
            {
                "id": index,
                "way_id": index,
                "node_id": index,
                "end_node": {"node_id": index + 1, "elapsed_time": elapsed},
            }
            for index, elapsed in enumerate([3, 8, 12])
        ],
        "matched_points": [
            {"type": "matched", "edge_index": 0},
            {"type": "matched", "edge_index": 2},
        ],
    }
    _, routes = parse_trace_attributes(
        raw, source, order_id="o", subtrace_id="o:000"
    )
    assert routes.valhalla_cumulative_elapsed_time_s.tolist() == [3, 8, 12]
    assert routes.valhalla_edge_elapsed_time_s.tolist() == [3, 5, 4]
    assert routes.engine_allocated_travel_time_s.isna().all()


def _canonical_edges():
    return pd.DataFrame(
        {
            "edge_uid": ["10:0:F", "10:1:F", "10:2:F", "20:0:F", "20:1:F"],
            "osm_way_id": [10, 10, 10, 20, 20],
            "segment_seq": [0, 1, 2, 0, 1],
            "direction": ["F"] * 5,
            "from_node": [1, 2, 3, 8, 8],
            "to_node": [2, 3, 4, 9, 9],
            "length_m": [10.0, 20.0, 30.0, 4.0, 5.0],
            "highway": ["primary"] * 5,
            "bridge": [False] * 5,
            "tunnel": [False] * 5,
        }
    )


def test_mapper_expands_unique_way_node_path_and_preserves_distance():
    mapper = CanonicalEdgeMapper(_canonical_edges())
    route = pd.DataFrame(
        [
            {
                "order_id": "o",
                "subtrace_id": "o:000",
                "path_id": 0,
                "route_sequence": 0,
                "valhalla_edge_index": 0,
                "valhalla_edge_id": "v",
                "osm_way_id": 10,
                "begin_osm_node_id": 1,
                "end_osm_node_id": 4,
                "forward": True,
                "source_percent_along": 0.0,
                "target_percent_along": 1.0,
                "length_m": 60.0,
                "road_class": "primary",
                "bridge": False,
                "tunnel": False,
                "speed_limit": 50,
                "edge_elapsed_time_s": 4,
                "is_interpolated": False,
                "route_source": "observed",
            }
        ]
    )
    mapped, summary = mapper.map_route_parts(route)
    assert mapped.canonical_edge_uid.tolist() == ["10:0:F", "10:1:F", "10:2:F"]
    assert mapped.mapping_status.eq("way_and_node_mapping").all()
    assert mapped.length_m.sum() == 60.0
    assert summary.status_counts == {"way_and_node_mapping": 1}


def test_mapper_refuses_ambiguous_way_and_unmapped_edge():
    mapper = CanonicalEdgeMapper(_canonical_edges())
    ambiguous = pd.Series(
        {
            "osm_way_id": 20,
            "begin_osm_node_id": 8,
            "end_osm_node_id": 9,
            "forward": True,
        }
    )
    missing = ambiguous.copy()
    missing["osm_way_id"] = 999
    assert mapper.resolve(ambiguous) == ("ambiguous_mapping", [])
    assert mapper.resolve(missing) == ("unmapped", [])
