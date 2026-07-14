"""Sparse lexicographic matching helpers."""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx
import pandas as pd


SERVICE_CARDINALITY_BONUS = 1_000_000_000


@dataclass
class CandidateEdge:
    request_id: str
    vehicle_id: str
    pickup_eta_sec: float
    marginal_contribution: float
    passenger_gc: float
    driver_utility: float
    stress: float
    objective: float
    metadata: dict


def sparse_max_cardinality_match(edges: list[CandidateEdge], objective: str) -> tuple[list[CandidateEdge], dict]:
    if not edges:
        return [], {"matching_solver": "sparse_networkx_max_weight_matching", "matched_edges": 0, "matching_runtime_sec": 0.0}
    import time

    start = time.perf_counter()
    graph = nx.Graph()
    lookup: dict[tuple[str, str], CandidateEdge] = {}
    for edge in edges:
        if objective == "min_pickup":
            secondary = -edge.pickup_eta_sec
        else:
            secondary = edge.marginal_contribution
        graph.add_edge(f"o:{edge.request_id}", f"v:{edge.vehicle_id}", weight=SERVICE_CARDINALITY_BONUS + int(round(secondary * 1_000)))
        lookup[(edge.request_id, edge.vehicle_id)] = edge
    matching = nx.algorithms.matching.max_weight_matching(graph, maxcardinality=True, weight="weight")
    chosen: list[CandidateEdge] = []
    for a, b in matching:
        if a.startswith("o:"):
            key = (a[2:], b[2:])
        else:
            key = (b[2:], a[2:])
        if key in lookup:
            chosen.append(lookup[key])
    return chosen, {
        "matching_solver": "sparse_networkx_max_weight_matching",
        "matched_edges": len(chosen),
        "matching_runtime_sec": float(time.perf_counter() - start),
    }


def edges_to_frame(edges: list[CandidateEdge]) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "request_id": e.request_id,
            "vehicle_id": e.vehicle_id,
            "pickup_eta_sec": e.pickup_eta_sec,
            "marginal_contribution": e.marginal_contribution,
            "passenger_gc": e.passenger_gc,
            "driver_utility": e.driver_utility,
            "stress": e.stress,
            **e.metadata,
        }
        for e in edges
    ])

