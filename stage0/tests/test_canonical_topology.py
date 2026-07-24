import networkx as nx

from stage0.canonical.topology import (
    build_multidigraph,
    is_directed_transition,
    minimum_parallel_edge,
)


def test_shared_node_is_not_sufficient_when_next_link_points_toward_junction():
    assert not is_directed_transition(1, 2, "F", 3, 2, "F")


def test_oneway_reverse_transition_is_rejected_and_forward_is_accepted():
    assert not is_directed_transition(1, 2, "F", 3, 1, "F")
    assert is_directed_transition(1, 2, "F", 2, 3, "F")


def test_legal_uturn_between_separate_directed_links_is_retained():
    assert is_directed_transition(1, 2, "F", 2, 1, "F")


def test_parallel_edges_are_not_collapsed_in_multidigraph():
    graph = build_multidigraph([
        {"road_idx": 10, "from_node": 1, "to_node": 2, "length": 120.0, "oneway_code": "F", "road_class": "main"},
        {"road_idx": 11, "from_node": 1, "to_node": 2, "length": 100.0, "oneway_code": "F", "road_class": "service"},
    ])
    assert isinstance(graph, nx.MultiDiGraph)
    assert graph.number_of_edges(1, 2) == 2
    assert minimum_parallel_edge(graph, 1, 2)["road_idx"] == 11

