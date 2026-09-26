from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5.checkpoint_selection import evaluate_candidate, select_checkpoint


THRESHOLDS = {
    "maximum_pace_mean_s_per_m": 5.0,
    "maximum_mean_to_p50_ratio": 20.0,
    "maximum_p99_9_pace_mean_s_per_m": 1.0,
    "maximum_single_row_mae_contribution_share": 0.8,
    "maximum_single_row_rmse_contribution_share": 0.8,
}


def _frame(mean: np.ndarray, p50: np.ndarray) -> pd.DataFrame:
    count = len(mean)
    return pd.DataFrame(
        {
            "order_id": [f"o{i}" for i in range(count)],
            "traversal_id": np.arange(count),
            "pace_log_mu": np.log(p50),
            "pace_log_scale": -2.0,
            "pace_pred_mean": mean,
            "pace_pred_p50": p50,
            "pace_pred_p90": p50 + 0.03,
            "pace_pred_p95": p50 + 0.05,
            "service_time_availability_probability": 0.8,
            "lcs_availability_probability": 0.8,
            "rts_availability_probability": 0.8,
            "dynamics_availability_probability": 0.8,
            "pace_sec_per_m": 0.2,
            "pace_target_valid": True,
        }
    )


def test_lowest_loss_checkpoint_is_rejected_when_stability_fails() -> None:
    unstable = evaluate_candidate(
        _frame(np.array([0.2, 0.2, 100.0]), np.array([0.2, 0.2, 0.2])),
        checkpoint_id="lowest_loss",
        validation_distribution_nll=-10.0,
        thresholds=THRESHOLDS,
        route_scenario_smoke_pass=True,
    )
    stable = evaluate_candidate(
        _frame(np.array([0.2, 0.21, 0.19]), np.array([0.2, 0.21, 0.19])),
        checkpoint_id="stable",
        validation_distribution_nll=-1.0,
        thresholds=THRESHOLDS,
        route_scenario_smoke_pass=True,
    )
    result = select_checkpoint([unstable, stable])
    assert unstable["hard_gate_status"] == "FAIL"
    assert result["selected_checkpoint_id"] == "stable"


def test_stable_candidates_are_selected_by_frozen_metric_order() -> None:
    first = evaluate_candidate(
        _frame(np.array([0.22, 0.22, 0.22]), np.array([0.22, 0.22, 0.22])),
        checkpoint_id="better_nll",
        validation_distribution_nll=-2.0,
        thresholds=THRESHOLDS,
        route_scenario_smoke_pass=True,
    )
    second = evaluate_candidate(
        _frame(np.array([0.21, 0.21, 0.21]), np.array([0.2, 0.2, 0.2])),
        checkpoint_id="better_p50",
        validation_distribution_nll=-1.0,
        thresholds=THRESHOLDS,
        route_scenario_smoke_pass=True,
    )
    assert select_checkpoint([first, second])["selected_checkpoint_id"] == "better_p50"
