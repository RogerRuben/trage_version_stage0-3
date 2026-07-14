"""Two-stage sparse candidate generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.neighbors import BallTree

from ..entities import RequestState, VehicleState
from ..system_state import SystemState

EARTH_M = 6_371_000.0


@dataclass
class CandidatePolicy:
    candidate_policy: str = "adaptive"
    initial_candidates: int = 20
    second_stage_candidates: int = 40
    maximum_candidates: int = 80
    minimum_av_candidates: int = 10
    minimum_hv_candidates: int = 10
    minimum_feasible_candidates: int = 10


class CandidateGenerator:
    def __init__(self, policy: CandidatePolicy):
        self.policy = policy

    def controllable_vehicles(self, state: SystemState) -> list[VehicleState]:
        ids = sorted(state.idle_hv_ids | state.idle_av_ids)
        return [state.vehicles[vid] for vid in ids]

    def generate(self, requests: list[RequestState], vehicles: list[VehicleState], radius_by_order: dict[str, float]) -> tuple[dict[str, list[VehicleState]], dict]:
        if not requests or not vehicles:
            return {}, {"coarse_candidate_edges": 0, "candidate_truncation_rate": 0.0, "orders_hitting_candidate_cap": 0}
        coords = np.radians(np.array([[v.current_lat, v.current_lon] for v in vehicles], dtype=float))
        tree = BallTree(coords, metric="haversine")
        result: dict[str, list[VehicleState]] = {}
        total_possible = 0
        retained = 0
        hit_cap = 0
        for req in requests:
            radius = radius_by_order.get(req.order_id, 2_000.0)
            q = np.radians([[req.origin_lat, req.origin_lon]])
            idx, dist = tree.query_radius(q, r=radius / EARTH_M, return_distance=True, sort_results=True)
            ids = idx[0]
            total_possible += len(ids)
            if len(ids) > self.policy.maximum_candidates:
                hit_cap += 1
            selected = [vehicles[int(i)] for i in ids[: self.policy.maximum_candidates]]
            result[req.order_id] = selected
            retained += len(selected)
        trunc = 1.0 - retained / total_possible if total_possible else 0.0
        return result, {
            "coarse_candidate_edges": retained,
            "candidate_truncation_rate": max(0.0, trunc),
            "orders_hitting_candidate_cap": hit_cap,
        }
