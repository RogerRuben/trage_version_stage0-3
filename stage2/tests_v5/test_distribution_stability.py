from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5.distribution_stability import summarize_traversals


def _frame(mean: np.ndarray, p50: np.ndarray, p90: np.ndarray, p95: np.ndarray) -> pd.DataFrame:
    count = len(mean)
    return pd.DataFrame(
        {
            "pace_log_mu": np.log(np.maximum(p50, 1.0e-6)),
            "pace_log_scale": -2.0,
            "pace_pred_mean": mean,
            "pace_pred_p50": p50,
            "pace_pred_p90": p90,
            "pace_pred_p95": p95,
            "service_time_availability_probability": 0.8,
            "lcs_availability_probability": 0.8,
            "rts_availability_probability": 0.8,
            "dynamics_availability_probability": 0.8,
            "pace_sec_per_m": np.full(count, 0.2),
            "pace_target_valid": True,
        }
    )


def test_distribution_audit_detects_ratio_nonfinite_crossing_and_row_contribution() -> None:
    summary = summarize_traversals(
        _frame(
            np.array([0.2, 0.2, 100.0, np.inf]),
            np.array([0.2, 0.2, 0.2, 0.2]),
            np.array([0.3, 0.3, 0.1, 0.3]),
            np.array([0.35, 0.35, 0.4, 0.35]),
        )
    )
    assert summary["mean_to_p50_ratio"]["max"] == 500.0
    assert summary["nonfinite_counts"]["pace_pred_mean"] == 1
    assert summary["non_monotonic_or_nonpositive_quantile_count"] == 1
    assert summary["mean_error"]["maximum_row_mae_contribution_share"] > 0.99
    assert summary["mean_error"]["maximum_row_rmse_contribution_share"] > 0.999
