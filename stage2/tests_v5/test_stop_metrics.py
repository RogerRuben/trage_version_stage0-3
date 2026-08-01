from __future__ import annotations

import numpy as np

from stage2.v5.metrics import evaluate_stop_two_part


def test_stop_occurrence_and_positive_share_are_separate() -> None:
    result = evaluate_stop_two_part(
        np.array([0.0, 0.0, 0.2, 0.7]),
        np.array([0.1, 0.2, 0.8, 0.9]),
        np.array([0.4, 0.5, 0.25, 0.65]),
    )
    assert result["stop_occurrence_prevalence"] == 0.5
    assert result["positive_stop_share"]["count"] == 2
    assert result["expected_stop_share"]["count"] == 4
    assert np.isclose(result["always_zero_baseline"]["mae"], 0.225)
