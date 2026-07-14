"""ODD-Gated Price-Aware strategy."""

from __future__ import annotations

from .sparse_matcher import CandidateEdge, sparse_max_cardinality_match


def solve(edges: list[CandidateEdge]):
    return sparse_max_cardinality_match(edges, "max_contribution")

