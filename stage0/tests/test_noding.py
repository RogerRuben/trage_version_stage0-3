from shapely.geometry import LineString, Point

import numpy as np

from stage0.canonical.noding import (
    cluster_endpoints,
    cluster_endpoints_by_level,
    connector_traversal_directions,
    grade_transition_connector_eligible,
    parse_bool,
    split_line_at_points,
    topology_level,
    topology_levels_compatible,
)
from stage0.scripts.build_noded_network_v3 import bool_value


def test_split_line_at_interior_intersection():
    line = LineString([(0, 0), (10, 0)])
    pieces = split_line_at_points(line, [Point(5, 0)])
    assert len(pieces) == 2
    assert sum(piece.length for piece in pieces) == 10


def test_endpoint_is_not_split_again():
    line = LineString([(0, 0), (10, 0)])
    assert len(split_line_at_points(line, [Point(0, 0), Point(10, 0)])) == 1


def test_endpoint_clustering_is_tolerance_bounded():
    node_ids, representatives = cluster_endpoints(
        __import__("numpy").array([[0.0, 0.0], [0.2, 0.0], [2.0, 0.0]]), 0.5
    )
    assert node_ids.tolist() == [0, 0, 1]
    assert len(representatives) == 2


def test_grade_separation_key_differs():
    assert topology_level(0, False, False) != topology_level(1, True, False)


def test_bool_value_true_encodings():
    for value in ["T", "TRUE", "1", "Y", "YES", True, "t", "true"]:
        assert bool_value(value) is True


def test_bool_value_false_encodings():
    for value in ["F", "FALSE", "0", "N", "NO", False, "f", "false"]:
        assert bool_value(value) is False


def test_bridge_tunnel_boolean_encodings():
    assert topology_level(0, "T", "F") == ("0", True, False)
    assert topology_level(0, "NO", "YES") == ("0", False, True)


def test_missing_bool_value():
    for value in [None, np.nan, "", "NA" if False else "nan"]:
        assert parse_bool(value) is False


def test_same_layer_endpoints_clustered():
    coordinates = np.array([[0.0, 0.0], [0.2, 0.0]])
    node_ids, _, _ = cluster_endpoints_by_level(
        coordinates, [("0", False, False), ("0", False, False)], 0.75
    )
    assert node_ids[0] == node_ids[1]


def test_cross_layer_endpoints_not_clustered():
    coordinates = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    levels = [("0", False, False), ("1", False, False), ("0", True, False), ("0", False, True)]
    node_ids, _, _ = cluster_endpoints_by_level(coordinates, levels, 0.75)
    assert len(set(node_ids)) == 4


def test_grade_separated_intersections_not_noded():
    ground = topology_level(0, False, False)
    assert not topology_levels_compatible(ground, topology_level(1, False, False))
    assert not topology_levels_compatible(ground, topology_level(0, True, False))
    assert not topology_levels_compatible(ground, topology_level(0, False, True))


def test_grade_transition_connector_requires_alignment_and_same_class():
    ground, bridge = topology_level(0, False, False), topology_level(1, True, False)
    assert grade_transition_connector_eligible(
        ground, bridge, (1, 0), (-1, 0), "primary", "primary"
    )
    assert not grade_transition_connector_eligible(
        ground, bridge, (1, 0), (0, 1), "primary", "primary"
    )
    assert not grade_transition_connector_eligible(
        ground, bridge, (1, 0), (-1, 0), "primary", "secondary"
    )
    assert grade_transition_connector_eligible(
        ground, bridge, (1, 0), (-1, 0), "primary", "trunk_link"
    )


def test_connector_directionality():
    assert connector_traversal_directions("F", False, "F", True) == (True, False)
    assert connector_traversal_directions("F", True, "F", False) == (False, True)
    assert connector_traversal_directions("B", True, "B", False) == (True, True)


def test_connector_candidate_ineligible_contract():
    connector = {"topology_connector": True, "candidate_eligible": False}
    assert connector["topology_connector"]
    assert not connector["candidate_eligible"]
