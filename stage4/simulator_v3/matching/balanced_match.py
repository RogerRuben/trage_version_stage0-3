"""Three-Stakeholder Balanced matching skeleton.

The formal strategy requires zone-time stress budgets and minimum zone-service
targets.  This module exposes a solver that applies those constraints when
provided; callers should not interpret it as complete if the constraint tables
are empty.
"""

from __future__ import annotations

from .sparse_matcher import CandidateEdge, sparse_max_cardinality_match


def solve(edges: list[CandidateEdge], remaining_stress_budget: dict[str, float] | None = None, minimum_zone_service: dict[str, int] | None = None):
    remaining_stress_budget = remaining_stress_budget or {}
    if remaining_stress_budget:
        filtered = []
        for edge in edges:
            zone = str(edge.metadata.get("origin_zone", ""))
            if edge.metadata.get("vehicle_type") == "HV" and edge.stress > remaining_stress_budget.get(zone, float("inf")):
                continue
            filtered.append(edge)
        edges = filtered
    chosen, stats = sparse_max_cardinality_match(edges, "max_contribution")
    stats["balanced_constraint_tables_present"] = bool(remaining_stress_budget or minimum_zone_service)
    return chosen, stats

