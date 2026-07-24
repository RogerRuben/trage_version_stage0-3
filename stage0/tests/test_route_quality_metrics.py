import math

from stage0.canonical.route_quality import (
    core_threshold_flags,
    projection_metrics,
    route_sequence_metrics,
)


def test_route_length_ratio_and_interpolated_distance_share():
    metrics = route_sequence_metrics(
        ["a", "b", "c"], [False, True, False], {"a": 100, "b": 50, "c": 150}, {}, 200
    )
    assert metrics["route_length_ratio"] == 1.5
    assert metrics["interpolated_distance_share"] == 1 / 6


def test_route_length_ratio():
    metrics = route_sequence_metrics(["a"], [False], {"a": 200}, {}, 100)
    assert metrics["route_length_ratio"] == 2.0


def test_interpolated_distance_share():
    metrics = route_sequence_metrics(["a", "b"], [True, False], {"a": 25, "b": 75}, {}, 100)
    assert metrics["interpolated_distance_share"] == 0.25


def test_unreasonable_detour_detection_and_core_threshold_application():
    row = {
        "direction_gap_count": 0, "unreasonable_detour_count": 1,
        "fallback_point_share": 0, "p90_projection_distance_m": 10,
        "route_length_ratio": 1, "interpolated_distance_share": 0.1,
        "origin_projection_error_m": 10, "destination_projection_error_m": 10,
        "mean_match_confidence": 0.8, "u_turn_count": 0, "repeated_link_share": 0,
    }
    cfg = {
        "maximum_direction_gaps": 0, "maximum_unreasonable_detour_count": 0,
        "maximum_fallback_point_share": 0.25, "maximum_p90_projection_distance_m": 60,
        "minimum_route_length_ratio": 0.5, "maximum_route_length_ratio": 2.5,
        "maximum_interpolated_distance_share": 0.5, "maximum_od_projection_error_m": 100,
        "minimum_mean_match_confidence": 0.35, "maximum_u_turn_count": 0,
        "maximum_repeated_link_share": 0.2,
    }
    flags = core_threshold_flags(row, cfg)
    assert not flags["core_no_unreasonable_detour"]
    assert not flags["core_all_thresholds_pass"]


def test_unreasonable_detour_detection():
    assert test_unreasonable_detour_detection_and_core_threshold_application() is None


def test_core_threshold_application():
    assert test_unreasonable_detour_detection_and_core_threshold_application() is None


def test_od_endpoint_error():
    metrics = projection_metrics([1, 2], [108.0, 108.1], [34.0, 34.1], [108.0, 108.1], [34.0, 34.1])
    assert math.isclose(metrics["origin_projection_error_m"], 0.0, abs_tol=1e-9)
    assert math.isclose(metrics["destination_projection_error_m"], 0.0, abs_tol=1e-9)
