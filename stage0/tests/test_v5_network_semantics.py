from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import LineString

from stage0.v5.network import (
    build_movement_graph,
    classify_parallel_edges,
    directed_directions,
    motor_vehicle_eligible,
    normalize_layer,
    normalize_oneway,
    stable_edge_uid,
)


def _edge(uid: str, way: int, u: int, v: int, coords, layer=0, bridge=False, tunnel=False, highway="primary"):
    return {
        "edge_uid": uid,
        "physical_way_id": str(way),
        "osm_way_id": way,
        "from_node": u,
        "to_node": v,
        "geometry": LineString(coords),
        "layer": layer,
        "bridge": bridge,
        "tunnel": tunnel,
        "highway": highway,
        "access": None,
        "service": None,
        "maxspeed": None,
    }


def test_stable_edge_uid_is_direction_specific():
    assert stable_edge_uid(42, 3, "F") == "42:3:F"
    assert stable_edge_uid(42, 3, "R") == "42:3:R"


def test_oneway_reverse_expands_only_reverse():
    assert normalize_oneway("-1") == "reverse"
    assert directed_directions("reverse") == ("R",)


def test_bidirectional_expands_both_directions():
    assert directed_directions(normalize_oneway("no")) == ("F", "R")


def test_true_parallel_edges_are_retained_and_classified():
    edges = gpd.GeoDataFrame([
        _edge("1:0:F", 1, 1, 2, [(0, 0), (10, 0)]),
        _edge("2:0:F", 2, 1, 2, [(0, 1), (10, 1)]),
    ], geometry="geometry", crs=3857)
    audit = classify_parallel_edges(edges)
    assert len(edges) == 2
    assert audit.parallel_category.iloc[0] == "true_parallel"
    assert not bool(audit.merge_allowed.iloc[0])


def test_bridge_entry_at_shared_node_is_legal_and_classified():
    edges = gpd.GeoDataFrame([
        _edge("1:0:F", 1, 1, 2, [(0, 0), (10, 0)], layer=1, bridge=True),
        _edge("2:0:F", 2, 2, 3, [(10, 0), (20, 0)], layer=0),
    ], geometry="geometry", crs=3857)
    movements = build_movement_graph(edges, pd.DataFrame())
    assert bool(movements.layer_compatibility.iloc[0])
    assert movements.level_transition_type.iloc[0] == "bridge_exit"


def test_tunnel_default_layer_and_explicit_layer_are_normalized():
    assert normalize_layer(None, tunnel="yes") == -1
    assert normalize_layer("-2", tunnel="yes") == -2


def test_service_access_filter_is_selective_not_global():
    excluded = {"footway", "cycleway"}
    assert motor_vehicle_eligible({"highway": "service", "motor_vehicle": "yes"}, excluded)
    assert not motor_vehicle_eligible({"highway": "service", "service": "driveway"}, excluded)
    assert not motor_vehicle_eligible({"highway": "footway"}, excluded)
