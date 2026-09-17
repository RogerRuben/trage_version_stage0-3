from __future__ import annotations

import numpy as np

from stage2.v5.scenario import (
    generate_cross_order_quantile_scenarios,
    generate_quantile_route_scenarios,
)


def test_quantile_scenarios_are_reproducible_bounded_and_correlated() -> None:
    route = np.array(["a", "a", "b"])
    p50 = np.array([0.20, 0.25, 0.30])
    p90 = np.array([0.30, 0.35, 0.42])
    p95 = np.array([0.34, 0.39, 0.47])
    distance = np.array([100.0, 200.0, 150.0])
    first = generate_quantile_route_scenarios(
        route,
        p50,
        p90,
        p95,
        distance,
        scenario_count=20_000,
        seed=17,
        model="shared_route_quantile",
        shared_route_rho=0.5,
    )
    second = generate_quantile_route_scenarios(
        route,
        p50,
        p90,
        p95,
        distance,
        scenario_count=20_000,
        seed=17,
        model="shared_route_quantile",
        shared_route_rho=0.5,
    )
    assert np.array_equal(first.route_time_s, second.route_time_s)
    marginal = first.traversal_time_s / distance[:, None]
    observed = np.quantile(marginal, [0.5, 0.9, 0.95], axis=1).T
    expected = np.column_stack((p50, p90, p95))
    assert np.allclose(observed, expected, atol=0.01)
    maximum = p95 + 4.0 * (p95 - p90)
    assert np.all(marginal <= maximum[:, None] + 1.0e-9)
    assert np.corrcoef(marginal[0], marginal[1])[0, 1] > 0.25


def test_cross_order_scenario_index_is_one_shared_system_state() -> None:
    route = np.array(["order_a", "order_b", "order_c"])
    result = generate_cross_order_quantile_scenarios(
        route,
        np.full(3, 0.2),
        np.full(3, 0.3),
        np.full(3, 0.35),
        np.full(3, 100.0),
        network_time_bin=np.array(["t1", "t1", "t1"]),
        region_time_bin=np.array(["r1", "r1", "r2"]),
        highway_time_bin=np.array(["h1", "h2", "h2"]),
        scenario_count=20_000,
        seed=23,
        network_weight=0.4,
        region_weight=0.1,
        highway_weight=0.05,
        route_weight=0.05,
    )
    assert np.array_equal(result.system_scenario_id, np.arange(20_000))
    assert np.array_equal(result.system_scenario_id, result.network_shock_id)
    first, second = result.scenario.route_time_s[:2]
    assert np.corrcoef(first, second)[0, 1] > 0.25
    repeated = generate_cross_order_quantile_scenarios(
        route,
        np.full(3, 0.2),
        np.full(3, 0.3),
        np.full(3, 0.35),
        np.full(3, 100.0),
        network_time_bin=np.array(["t1", "t1", "t1"]),
        region_time_bin=np.array(["r1", "r1", "r2"]),
        highway_time_bin=np.array(["h1", "h2", "h2"]),
        scenario_count=20_000,
        seed=23,
        network_weight=0.4,
        region_weight=0.1,
        highway_weight=0.05,
        route_weight=0.05,
    )
    assert np.array_equal(result.scenario.route_time_s, repeated.scenario.route_time_s)
