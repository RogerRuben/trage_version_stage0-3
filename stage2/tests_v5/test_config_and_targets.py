from __future__ import annotations

import numpy as np

from stage2.v5.availability import service_time_target_arrays, stabilized_ipw
from stage2.v5.config import load_config


def test_frozen_config_and_service_time_masks() -> None:
    config = load_config()
    assert config.section("split")["final_test_dates"] == ["20161028", "20161029", "20161030"]
    arrays = service_time_target_arrays(
        np.array(["direct_observed", "engine_interpolated", "direct_observed"]),
        np.array([10.0, 12.0, 5.0]),
        np.array([20.0, 30.0, 0.0]),
    )
    assert arrays["travel_time_direct_valid"].tolist() == [True, False, True]
    assert arrays["travel_time_interpolated_valid"].tolist() == [False, False, False]
    assert arrays["pace_target_valid"].tolist() == [True, False, False]
    assert arrays["pace_target_sec_per_m"][0] == 0.5
    assert np.isnan(arrays["pace_target_sec_per_m"][1:]).all()


def test_stabilized_ipw_is_finite_clipped_and_ess_correct() -> None:
    mask = np.array([True, True, False, True])
    probability = np.array([0.5, 0.01, 0.8, 1.0])
    weights, diagnostics = stabilized_ipw(mask, probability, epsilon=0.05, maximum_weight=3.0)
    assert np.isfinite(weights).all()
    assert weights.max() <= 3.0
    assert diagnostics.clipped_count == 1
    positive = weights[mask]
    expected = positive.sum() ** 2 / np.square(positive).sum()
    assert np.isclose(diagnostics.effective_sample_size, expected)

