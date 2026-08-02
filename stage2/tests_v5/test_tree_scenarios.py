from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5.tree_scenarios_v5_1 import _tree_day_scenarios


def test_tree_residual_bootstrap_is_reproducible_and_cross_order_correlated() -> None:
    frame = pd.DataFrame(
        {
            "order_id": ["a", "a", "b", "b"],
            "route_sequence": [0, 1, 0, 1],
            "allocated_distance_m": [100.0, 150.0, 120.0, 130.0],
        }
    )
    pace = np.array([0.20, 0.22, 0.25, 0.24])
    residual = np.linspace(-0.08, 0.08, 1000)
    route_id, first = _tree_day_scenarios(
        frame, pace, residual, scenario_count=10_000, seed=29, route_batch_size=1
    )
    _, second = _tree_day_scenarios(
        frame, pace, residual, scenario_count=10_000, seed=29, route_batch_size=2
    )
    assert route_id.tolist() == ["a", "b"]
    assert np.array_equal(first, second)
    assert np.corrcoef(first[0], first[1])[0, 1] > 0.10
    assert np.isfinite(first).all() and np.all(first > 0)
