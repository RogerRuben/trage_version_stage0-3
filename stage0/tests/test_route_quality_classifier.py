import networkx as nx

from stage0.scripts.classify_canonical_route_quality import bridge_gap


def test_bridge_gap_requires_bounded_directed_path():
    lookup = {
        "left": {"from_node": 1, "to_node": 2, "oneway_code": "F"},
        "right": {"from_node": 3, "to_node": 4, "oneway_code": "F"},
    }
    graph = nx.DiGraph()
    graph.add_edge(2, 3, length_m=100.0, link_id="bridge")
    evidence = bridge_gap("left", "right", lookup, graph, 200.0, 2, {})
    assert evidence["bridge_found"]
    assert evidence["bridge_repairable"]
    assert evidence["bridge_link_count"] == 1
    assert evidence["bridge_distance_m"] == 100.0


def test_bridge_gap_rejects_path_beyond_distance_cap():
    lookup = {
        "left": {"from_node": 1, "to_node": 2, "oneway_code": "F"},
        "right": {"from_node": 3, "to_node": 4, "oneway_code": "F"},
    }
    graph = nx.DiGraph()
    graph.add_edge(2, 3, length_m=300.0, link_id="bridge")
    evidence = bridge_gap("left", "right", lookup, graph, 200.0, 2, {})
    assert not evidence["bridge_found"]
    assert not evidence["bridge_repairable"]
