"""Stateful Three-Stakeholder Balanced sparse matching.

The solver receives explicit zone-time constraint tables.  It first obtains a
maximum-cardinality sparse match, then performs cardinality-preserving swaps
for zone service targets and aggregate HV stress budgets before maximizing
marginal operating contribution within that constrained feasible set.

Daily fixed costs never enter this module; they are scenario-level ledger
items and cannot affect an epoch edge choice.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from math import ceil

from .sparse_matcher import CandidateEdge, sparse_max_cardinality_match


def zone_time_key(zone: str, time_bin: int) -> str:
    return f"{zone}|{int(time_bin):02d}"


@dataclass
class BalancedEpochState:
    """Persistent, auditable zone-time stakeholder constraint state."""

    stress_budget_share_of_pending_load: float = 0.55
    minimum_zone_service_rate: float = 0.10
    maximum_zone_service_target_per_epoch: int = 5
    total_stress_budget: dict[str, float] = field(default_factory=dict)
    remaining_stress_budget: dict[str, float] = field(default_factory=dict)
    minimum_zone_service_target: dict[str, int] = field(default_factory=dict)
    served_zone_count: dict[str, int] = field(default_factory=dict)
    pending_zone_count: dict[str, int] = field(default_factory=dict)
    constraint_source: str = "observable_pending_load_scenario_policy_v1"

    def prepare_epoch(self, pending_requests: list, time_bin: int) -> dict[str, str]:
        pending_by_zone: dict[str, list[float]] = defaultdict(list)
        for req in pending_requests:
            pending_by_zone[str(req.origin_zone)].append(float(req.stress_value if req.condition_available else 0.0))
        active: dict[str, str] = {}
        for zone, stresses in pending_by_zone.items():
            key = zone_time_key(zone, time_bin)
            self.pending_zone_count[key] = max(
                self.pending_zone_count.get(key, 0), len(stresses)
            )
            # The budget is proportional to the currently observable pending
            # stress load.  It is not a hidden test-day outcome or a fixed
            # universal threshold.
            load = float(sum(stresses))
            target_budget = max(0.0, self.stress_budget_share_of_pending_load * load)
            previous_total = self.total_stress_budget.get(key, 0.0)
            if target_budget > previous_total:
                # Increase the period budget only when newly observable load
                # raises the auditable scenario allocation.  Previously
                # consumed stress is never reset at a later decision epoch.
                self.remaining_stress_budget[key] = (
                    self.remaining_stress_budget.get(key, 0.0)
                    + target_budget - previous_total
                )
                self.total_stress_budget[key] = target_budget
            else:
                self.remaining_stress_budget.setdefault(key, target_budget)
                self.total_stress_budget.setdefault(key, target_budget)
            target = min(
                self.maximum_zone_service_target_per_epoch,
                int(ceil(self.minimum_zone_service_rate * len(stresses))),
            )
            self.minimum_zone_service_target[key] = max(
                self.minimum_zone_service_target.get(key, 0), target
            )
            self.served_zone_count.setdefault(key, 0)
            active[zone] = key
        return active

    def consume(self, chosen: list[CandidateEdge], time_bin: int) -> None:
        for edge in chosen:
            zone = str(edge.metadata.get("origin_zone", "UNKNOWN"))
            key = zone_time_key(zone, time_bin)
            self.served_zone_count[key] = self.served_zone_count.get(key, 0) + 1
            if edge.metadata.get("vehicle_type") == "HV":
                self.remaining_stress_budget[key] = max(
                    0.0,
                    self.remaining_stress_budget.get(key, 0.0) - float(edge.stress),
                )

    def tables_for_epoch(self, time_bin: int) -> dict:
        suffix = f"|{int(time_bin):02d}"
        return {
            "remaining_stress_budget": {k.split("|")[0]: v for k, v in self.remaining_stress_budget.items() if k.endswith(suffix)},
            "total_stress_budget": {k.split("|")[0]: v for k, v in self.total_stress_budget.items() if k.endswith(suffix)},
            "minimum_zone_service_target": {
                k.split("|")[0]: max(0, v - self.served_zone_count.get(k, 0))
                for k, v in self.minimum_zone_service_target.items() if k.endswith(suffix)
            },
            "absolute_minimum_zone_service_target": {k.split("|")[0]: v for k, v in self.minimum_zone_service_target.items() if k.endswith(suffix)},
            "served_zone_count": {k.split("|")[0]: v for k, v in self.served_zone_count.items() if k.endswith(suffix)},
            "pending_zone_count": {k.split("|")[0]: v for k, v in self.pending_zone_count.items() if k.endswith(suffix)},
            "constraint_source": self.constraint_source,
        }


def _base_filter(edges: list[CandidateEdge], passenger_gc_cap: float, hv_utility_min: float) -> tuple[list[CandidateEdge], dict]:
    filtered: list[CandidateEdge] = []
    gc_filtered = utility_filtered = odd_filtered = 0
    for edge in edges:
        if edge.passenger_gc > passenger_gc_cap:
            gc_filtered += 1
            continue
        if edge.metadata.get("vehicle_type") == "HV" and edge.driver_utility < hv_utility_min:
            utility_filtered += 1
            continue
        if edge.metadata.get("vehicle_type") == "AV" and not bool(edge.metadata.get("combined_odd_feasible", False)):
            odd_filtered += 1
            continue
        filtered.append(edge)
    return filtered, {
        "passenger_gc_filtered_edges": gc_filtered,
        "hv_utility_filtered_edges": utility_filtered,
        "av_odd_filtered_edges": odd_filtered,
    }


def _zone_counts(chosen: list[CandidateEdge]) -> Counter:
    return Counter(str(e.metadata.get("origin_zone", "UNKNOWN")) for e in chosen)


def _zone_hv_stress(chosen: list[CandidateEdge]) -> dict[str, float]:
    out: dict[str, float] = defaultdict(float)
    for edge in chosen:
        if edge.metadata.get("vehicle_type") == "HV":
            out[str(edge.metadata.get("origin_zone", "UNKNOWN"))] += float(edge.stress)
    return dict(out)


def solve(
    edges: list[CandidateEdge],
    *,
    constraint_tables: dict,
    passenger_gc_cap: float = 120.0,
    hv_utility_min: float = -2.0,
):
    """Solve with explicit zone-time tables; missing tables are an error."""

    required = {
        "remaining_stress_budget",
        "minimum_zone_service_target",
        "served_zone_count",
        "pending_zone_count",
    }
    missing = required.difference(constraint_tables)
    if missing:
        raise ValueError(f"Balanced constraint tables missing: {sorted(missing)}")

    budgets = {str(k): float(v) for k, v in constraint_tables["remaining_stress_budget"].items()}
    targets = {str(k): int(v) for k, v in constraint_tables["minimum_zone_service_target"].items()}
    filtered, filter_stats = _base_filter(edges, passenger_gc_cap, hv_utility_min)

    # Remove only individually impossible HV edges. Aggregate budgets are
    # enforced below through swaps/re-optimization.
    admissible = [
        edge for edge in filtered
        if edge.metadata.get("vehicle_type") != "HV"
        or float(edge.stress) <= budgets.get(str(edge.metadata.get("origin_zone", "UNKNOWN")), 0.0) + 1e-12
    ]
    chosen, stats = sparse_max_cardinality_match(admissible, "max_contribution")
    maximum_cardinality = len(chosen)

    # Cardinality-preserving zone fairness swaps. A deficient-zone request can
    # replace an assignment using the same vehicle while keeping total service.
    fairness_swaps = 0
    for _ in range(maximum_cardinality + 1):
        counts = _zone_counts(chosen)
        deficit_zones = [z for z, target in targets.items() if counts.get(z, 0) < target]
        if not deficit_zones:
            break
        by_vehicle = {edge.vehicle_id: edge for edge in chosen}
        used_requests = {edge.request_id for edge in chosen}
        replacement = None
        for zone in deficit_zones:
            candidates = sorted(
                (e for e in admissible if str(e.metadata.get("origin_zone", "UNKNOWN")) == zone and e.request_id not in used_requests),
                key=lambda e: (-e.marginal_contribution, e.pickup_eta_sec, e.request_id, e.vehicle_id),
            )
            for candidate in candidates:
                incumbent = by_vehicle.get(candidate.vehicle_id)
                if incumbent is None:
                    continue
                incumbent_zone = str(incumbent.metadata.get("origin_zone", "UNKNOWN"))
                if counts.get(incumbent_zone, 0) > targets.get(incumbent_zone, 0):
                    replacement = (incumbent, candidate)
                    break
            if replacement:
                break
        if replacement is None:
            break
        incumbent, candidate = replacement
        chosen = [candidate if e is incumbent else e for e in chosen]
        fairness_swaps += 1

    # Aggregate stress repair uses lower-stress alternatives for the same
    # request first, then same-vehicle swaps. It can reduce cardinality only
    # when the stakeholder budget makes the maximum-cardinality solution
    # infeasible and no feasible replacement exists.
    stress_swaps = stress_drops = 0
    for _ in range(maximum_cardinality + 1):
        hv_stress = _zone_hv_stress(chosen)
        violating = [z for z, total in hv_stress.items() if total > budgets.get(z, 0.0) + 1e-12]
        if not violating:
            break
        zone = max(violating, key=lambda z: hv_stress[z] - budgets.get(z, 0.0))
        victim = max(
            (e for e in chosen if e.metadata.get("vehicle_type") == "HV" and str(e.metadata.get("origin_zone", "UNKNOWN")) == zone),
            key=lambda e: (e.stress, -e.marginal_contribution),
        )
        used_vehicles = {e.vehicle_id for e in chosen if e is not victim}
        alternatives = sorted(
            (
                e for e in admissible
                if e.request_id == victim.request_id
                and e.vehicle_id not in used_vehicles
                and (e.metadata.get("vehicle_type") != "HV" or e.stress < victim.stress)
            ),
            key=lambda e: (e.stress if e.metadata.get("vehicle_type") == "HV" else -1.0, -e.marginal_contribution),
        )
        if alternatives:
            chosen = [alternatives[0] if e is victim else e for e in chosen]
            stress_swaps += 1
        else:
            chosen = [e for e in chosen if e is not victim]
            stress_drops += 1

    counts = _zone_counts(chosen)
    deficits = {z: max(0, t - counts.get(z, 0)) for z, t in targets.items()}
    final_stress = _zone_hv_stress(chosen)
    budget_violations = {
        z: max(0.0, total - budgets.get(z, 0.0))
        for z, total in final_stress.items()
        if total > budgets.get(z, 0.0) + 1e-12
    }
    stats.update(filter_stats)
    stats.update({
        "balanced_constraint_tables_present": True,
        "balanced_constraint_source": constraint_tables.get("constraint_source", ""),
        "maximum_cardinality_before_constraints": maximum_cardinality,
        "matched_edges": len(chosen),
        "stress_constraint_active": bool(budgets),
        "zone_service_constraint_active": bool(targets),
        "stress_constraint_binding": bool(stress_swaps or stress_drops),
        "zone_service_constraint_binding": bool(fairness_swaps or any(deficits.values())),
        "stress_replacement_count": stress_swaps,
        "stress_cardinality_reduction_count": stress_drops,
        "zone_fairness_swap_count": fairness_swaps,
        "zone_service_deficit_total": int(sum(deficits.values())),
        "stress_budget_violation_total": float(sum(budget_violations.values())),
        "constraint_relaxation_used": bool(any(deficits.values())),
        "price_aware_equivalent": False,
        "edge_objective": "max_marginal_operating_contribution_after_max_cardinality",
        "constraint_model": "passenger_gc+hv_utility+av_odd+zone_time_hv_stress+minimum_zone_service",
    })
    return chosen, stats
