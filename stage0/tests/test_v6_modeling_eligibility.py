from __future__ import annotations

import pandas as pd

from stage0.v6.eligibility import evaluate_modeling_eligibility


def _points(time_gaps, distances):
    count = len(time_gaps)
    return pd.DataFrame(
        {
            "order_id": ["o"] * count,
            "matching_lon": [108.0 + index * 0.001 for index in range(count)],
            "matching_lat": [34.0] * count,
            "time_gap_s": time_gaps,
            "step_distance_m": distances,
        }
    )


def _metrics(raw_distance, resolved_distance, *, count=10):
    return {
        "valid_point_count": count,
        "usable_subtrace_count": 1,
        "preprocess_break_count": 0,
        "raw_order_gps_distance_m": raw_distance,
        "resolved_subtrace_gps_distance_m": resolved_distance,
    }


def test_large_time_and_distance_gap_is_excluded():
    points = _points([0] + [10] * 8 + [99], [0] + [20] * 8 + [837])
    result = evaluate_modeling_eligibility(points, _metrics(997, 160))
    assert result["modeling_eligible"] is False
    assert result["unobserved_movement_gap_count"] == 1
    assert "LARGE_UNOBSERVED_MOVEMENT_GAP" in result["modeling_exclusion_reasons"]


def test_long_stationary_pause_is_not_a_missing_movement_gap():
    points = _points([0] + [10] * 8 + [600], [0] + [20] * 8 + [50])
    result = evaluate_modeling_eligibility(points, _metrics(210, 210))
    assert result["modeling_eligible"] is True
    assert result["unobserved_movement_gap_count"] == 0


def test_low_total_movement_is_excluded():
    points = _points([0] + [10] * 9, [0] + [2.5] * 9)
    result = evaluate_modeling_eligibility(points, _metrics(22.5, 22.5))
    assert result["modeling_eligible"] is False
    assert "LOW_TOTAL_MOVEMENT" in result["modeling_exclusion_reasons"]
