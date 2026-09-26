from __future__ import annotations

import pandas as pd

from stage0.v6.order_processor import stage1_core_decision
from stage0.v6.stage1_production import _prefilter_order


def _final(**overrides):
    value = {
        "route_status": "route_pass",
        "gps_status": "clean",
        "outlier_time_share": 0.0,
        "outlier_distance_share": 0.0,
        "canonical_status": "unique",
        "dynamic_status": "dynamic_partial",
    }
    value.update(overrides)
    return value


def _accounting(**overrides):
    value = {
        "time_conservation_valid": True,
        "distance_conservation_valid": True,
        "duplicate_interval_allocation_count": 0,
        "non_direct_observed_time_violation_count": 0,
        "valid_direct_interval_count": 8,
        "unique_timed_edge_count": 5,
    }
    value.update(overrides)
    return pd.DataFrame([value])


SETTINGS = {
    "maximum_local_outlier_time_share": 0.05,
    "maximum_local_outlier_distance_share": 0.05,
    "minimum_valid_direct_interval_count": 8,
    "minimum_unique_timed_edge_count": 5,
}


def test_stage1_core_accepts_resolved_partial_dynamic_order():
    accepted, reason = stage1_core_decision(
        _final(canonical_status="chain_resolved"),
        _accounting(),
        SETTINGS,
    )
    assert accepted
    assert reason == ""


def test_stage1_core_never_accepts_route_partial_or_conservation_failure():
    accepted, reason = stage1_core_decision(
        _final(route_status="route_partial"),
        _accounting(distance_conservation_valid=False),
        SETTINGS,
    )
    assert not accepted
    assert "ROUTE_NOT_PASS" in reason
    assert "DISTANCE_CONSERVATION_FAILURE" in reason


def test_prefilter_does_not_infer_timestamp_reversal_from_unordered_archive_rows():
    timestamps = [100, 109, 103, 124, 106, 121, 112, 118, 115, 127]
    frame = pd.DataFrame(
        {
            "order_id": ["example"] * len(timestamps),
            "timestamp": timestamps,
            "lon": [108.90 + index * 0.001 for index in range(len(timestamps))],
            "lat": [34.20] * len(timestamps),
        }
    )
    settings = {
        "minimum_valid_points": 10,
        "minimum_duration_s": 20,
        "maximum_invalid_coordinate_share": 0.1,
        "maximum_reverse_timestamp_share": 0.1,
        "maximum_duplicate_point_share": 0.5,
        "minimum_od_distance_m": 50,
        "impossible_speed_mps": 100,
        "maximum_impossible_speed_interval_share": 0.5,
    }

    result = _prefilter_order(frame, "20161009", "0000", settings)

    assert result["pre_match_eligible"]
    assert "TIMESTAMP_SEVERELY_REVERSED" not in result["pre_match_rejection_reason"]
