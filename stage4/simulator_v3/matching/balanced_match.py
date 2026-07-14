"""Three-Stakeholder Balanced sparse matching.

The implementation keeps the same first stage as the other production
strategies (maximize served orders), then uses marginal operating contribution
as the secondary objective under additional stakeholder constraints.  Daily
fixed costs are intentionally absent from edge weights; they belong to scenario
accounting, not epoch-level edge choice.
"""

from __future__ import annotations

from collections import defaultdict

from .sparse_matcher import CandidateEdge, sparse_max_cardinality_match


def solve(
    edges: list[CandidateEdge],
    remaining_stress_budget: dict[str, float] | None = None,
    minimum_zone_service: dict[str, int] | None = None,
    passenger_gc_cap: float = 120.0,
    hv_utility_min: float = -2.0,
    default_zone_stress_budget: float = 2.5,
):
    remaining_stress_budget = remaining_stress_budget or {}
    minimum_zone_service = minimum_zone_service or {}
    filtered: list[CandidateEdge] = []
    gc_filtered = 0
    utility_filtered = 0
    for edge in edges:
        if edge.passenger_gc > passenger_gc_cap:
            gc_filtered += 1
            continue
        if edge.metadata.get("vehicle_type") == "HV" and edge.driver_utility < hv_utility_min:
            utility_filtered += 1
            continue
        filtered.append(edge)
    removed_keys: set[tuple[str, str]] = set()
    stress_relax_iterations = 0
    chosen: list[CandidateEdge] = []
    stats: dict = {}
    for _ in range(10):
        active_edges = [e for e in filtered if (e.request_id, e.vehicle_id) not in removed_keys]
        chosen, stats = sparse_max_cardinality_match(active_edges, "max_contribution")
        zone_stress = defaultdict(float)
        for edge in chosen:
            if edge.metadata.get("vehicle_type") == "HV":
                zone_stress[str(edge.metadata.get("origin_zone", "UNKNOWN"))] += edge.stress
        violating = {
            zone: total
            for zone, total in zone_stress.items()
            if total > remaining_stress_budget.get(zone, default_zone_stress_budget)
        }
        if not violating:
            break
        worst = None
        worst_score = -1.0
        for edge in chosen:
            zone = str(edge.metadata.get("origin_zone", "UNKNOWN"))
            if zone not in violating or edge.metadata.get("vehicle_type") != "HV":
                continue
            score = edge.stress
            if score > worst_score:
                worst = edge
                worst_score = score
        if worst is None:
            break
        removed_keys.add((worst.request_id, worst.vehicle_id))
        stress_relax_iterations += 1
    chosen_zones = defaultdict(int)
    for edge in chosen:
        chosen_zones[str(edge.metadata.get("origin_zone", "UNKNOWN"))] += 1
    zone_service_deficit = {
        zone: max(0, target - chosen_zones.get(zone, 0))
        for zone, target in minimum_zone_service.items()
    }
    stats["balanced_constraint_tables_present"] = True
    stats["stress_constraint_active"] = stress_relax_iterations > 0
    stats["passenger_gc_constraint_active"] = gc_filtered > 0
    stats["hv_utility_constraint_active"] = utility_filtered > 0
    stats["zone_service_constraint_active"] = bool(minimum_zone_service)
    stats["stress_filtered_edges"] = len(removed_keys)
    stats["passenger_gc_filtered_edges"] = gc_filtered
    stats["hv_utility_filtered_edges"] = utility_filtered
    stats["zone_service_deficit_total"] = int(sum(zone_service_deficit.values()))
    stats["constraint_relaxation_used"] = bool(zone_service_deficit)
    stats["stress_budget_repair_iterations"] = stress_relax_iterations
    return chosen, stats
