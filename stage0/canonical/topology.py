"""Direction-aware topology primitives that retain parallel road edges."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import networkx as nx


def allows_forward(oneway_code: str | None) -> bool:
    return str(oneway_code or "B").upper() in {"F", "B"}


def allows_reverse(oneway_code: str | None) -> bool:
    return str(oneway_code or "B").upper() in {"T", "B"}


def legal_exits(from_node: int, to_node: int, oneway_code: str | None) -> set[int]:
    exits: set[int] = set()
    if allows_forward(oneway_code):
        exits.add(int(to_node))
    if allows_reverse(oneway_code):
        exits.add(int(from_node))
    return exits


def legal_entries(from_node: int, to_node: int, oneway_code: str | None) -> set[int]:
    entries: set[int] = set()
    if allows_forward(oneway_code):
        entries.add(int(from_node))
    if allows_reverse(oneway_code):
        entries.add(int(to_node))
    return entries


def is_directed_transition(
    previous_from: int,
    previous_to: int,
    previous_oneway: str | None,
    next_from: int,
    next_to: int,
    next_oneway: str | None,
) -> bool:
    """Return true only when a legal previous exit equals a legal next entry."""

    return bool(
        legal_exits(previous_from, previous_to, previous_oneway)
        & legal_entries(next_from, next_to, next_oneway)
    )


def build_multidigraph(edges: Iterable[dict[str, Any]]) -> nx.MultiDiGraph:
    """Build a directed multigraph without collapsing parallel physical links."""

    graph = nx.MultiDiGraph()
    for edge in edges:
        road_idx = int(edge["road_idx"])
        u = int(edge["from_node"])
        v = int(edge["to_node"])
        length = float(edge["length"])
        code = edge.get("oneway_code")
        attributes = {**edge, "weight": length, "road_idx": road_idx}
        if allows_forward(code):
            graph.add_edge(u, v, key=f"{road_idx}:F", **attributes, direction="F")
        if allows_reverse(code):
            graph.add_edge(v, u, key=f"{road_idx}:T", **attributes, direction="T")
    return graph


def minimum_parallel_edge(graph: nx.MultiDiGraph, u: int, v: int) -> dict[str, Any]:
    edges = graph.get_edge_data(u, v)
    if not edges:
        raise KeyError((u, v))
    return min(edges.values(), key=lambda data: (float(data["weight"]), int(data["road_idx"])))

