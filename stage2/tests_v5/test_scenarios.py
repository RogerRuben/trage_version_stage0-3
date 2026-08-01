from __future__ import annotations

import numpy as np

from stage2.v5.reference import aggregate_traversal_scenarios_reference
from stage2.v5.scenario import aggregate_traversal_scenarios, generate_route_scenarios


def test_scenarios_are_reproducible_and_sum_to_route() -> None:
    kwargs = dict(
        route_id=np.array(["a", "a", "b"]),
        pace_log_mu=np.log(np.array([0.2, 0.3, 0.4])),
        pace_log_scale=np.log(np.array([0.2, 0.25, 0.3])),
        allocated_distance_m=np.array([10.0, 20.0, 30.0]),
        scenario_count=1000,
        seed=17,
        model="shared_route_latent",
    )
    first = generate_route_scenarios(**kwargs)
    second = generate_route_scenarios(**kwargs)
    assert np.array_equal(first.traversal_time_s, second.traversal_time_s)
    assert np.array_equal(first.route_time_s, second.route_time_s)
    assert np.allclose(first.route_time_s[0], first.traversal_time_s[:2].sum(axis=0))


def test_correlation_does_not_change_lognormal_marginals_materially() -> None:
    base = dict(
        route_id=np.array(["a", "a"]),
        pace_log_mu=np.log(np.array([0.2, 0.3])),
        pace_log_scale=np.log(np.array([0.25, 0.25])),
        allocated_distance_m=np.ones(2),
        scenario_count=50000,
        seed=42,
    )
    independent = generate_route_scenarios(**base, model="independent")
    correlated = generate_route_scenarios(**base, model="shared_route_latent")
    for row in range(2):
        q1 = np.quantile(independent.traversal_time_s[row], [0.5, 0.9, 0.95])
        q2 = np.quantile(correlated.traversal_time_s[row], [0.5, 0.9, 0.95])
        assert np.allclose(q1, q2, rtol=0.03)


def test_optimized_scenario_aggregation_matches_reference() -> None:
    inverse = np.array([0, 0, 1, 2, 2, 2])
    traversal = np.arange(30, dtype=float).reshape(6, 5)
    actual = aggregate_traversal_scenarios(inverse, traversal)
    expected = aggregate_traversal_scenarios_reference(inverse, traversal)
    assert np.allclose(actual, expected, atol=1e-10, rtol=0)
