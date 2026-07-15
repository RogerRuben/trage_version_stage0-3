from stage4.simulator_v3.matching.balanced_match import solve
from stage4.simulator_v3.matching.sparse_matcher import CandidateEdge


def edge(order, vehicle, stress, contribution, zone="z0_0"):
    return CandidateEdge(order, vehicle, 10.0, contribution, 20.0, 0.0, stress, contribution, {"vehicle_type": "HV", "origin_zone": zone})


def test_balanced_applies_stress_budget_and_differs_from_plain_contribution():
    chosen, stats = solve(
        [edge("o1", "v1", 3.0, 100.0), edge("o2", "v2", 0.1, 1.0)],
        constraint_tables={
            "remaining_stress_budget": {"z0_0": 1.0},
            "minimum_zone_service_target": {"z0_0": 1},
            "served_zone_count": {"z0_0": 0},
            "pending_zone_count": {"z0_0": 2},
            "constraint_source": "unit_test",
        },
    )
    assert stats["stress_constraint_active"] is True
    assert {e.request_id for e in chosen} == {"o2"}
