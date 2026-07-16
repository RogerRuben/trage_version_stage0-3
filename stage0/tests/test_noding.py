from shapely.geometry import LineString, Point

from stage0.canonical.noding import cluster_endpoints, split_line_at_points, topology_level


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
