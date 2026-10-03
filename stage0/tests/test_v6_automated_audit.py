from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import LineString

from stage0.v6.automated_audit import (
    AUDIT_COLUMNS,
    INDEX_COLUMNS,
    _identify_risk_windows,
    _canonical_audit,
    _clipped_route_lines,
    _v5_comparison,
    build_audit_features,
    classify_audit_features,
    select_review_cases,
    write_review_indexes,
)


def _feature_row(order_id: str, **overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "order_id": order_id,
        "date": "20161010",
        "successful_reconstruction": True,
        "route_quality": "strict_core",
        "dynamic_quality": "dynamic_partial",
        "dynamic_coverage_class": "low_dynamic_coverage",
        "snap_mean_m": 3.0,
        "snap_p90_m": 8.0,
        "snap_p99_m": 15.0,
        "snap_max_m": 25.0,
        "route_buffer_coverage_share": 1.0,
        "route_resolved_gps_ratio": 1.01,
        "route_raw_gps_ratio": 1.01,
        "od_endpoint_error_m": 5.0,
        "maximum_implied_speed_mps": 22.0,
        "speed_violation_count": 0,
        "canonical_mapping_share": 1.0,
        "topology_violation_count": 0,
        "direction_violation_count": 0,
        "osm_oneway_conflict_count": 0,
        "uturn_count": 0,
        "matched_point_share": 0.99,
        "matched_route_interval_share": 0.95,
        "direct_observed_time_share": 0.10,
        "direct_observed_distance_share": 0.15,
        "interval_supported_time_share": 0.10,
        "unresolved_time_share": 0.80,
        "timed_traversal_share": 0.30,
        "valid_timed_traversal_count": 12,
        "preprocess_break_count": 0,
        "valhalla_discontinuity_count": 0,
        "engine_interpolated_endpoint_time_share": 0.40,
        "canonical_not_unique_time_share": 0.30,
        "inferred_path_time_share": 0.05,
        "v5_v6_comparison_available": True,
        "v5_v6_edge_jaccard": 0.90,
        "v5_v6_route_distance_ratio": 1.02,
        "time_conservation_valid": True,
        "timestamp_anchor_valid": True,
        "modeling_eligible": True,
        "modeling_exclusion_reasons": "",
    }
    row.update(overrides)
    return row


def test_three_classes_score_and_reason_codes_are_explainable() -> None:
    features = pd.DataFrame(
        [
            _feature_row("pass"),
            _feature_row(
                "manual",
                dynamic_quality="dynamic_unusable",
                direct_observed_time_share=0.0,
                unresolved_time_share=1.0,
            ),
            _feature_row(
                "fail",
                successful_reconstruction=False,
                route_quality="rejected",
                canonical_mapping_share=0.0,
                route_resolved_gps_ratio=0.0,
            ),
        ]
    )

    audit = classify_audit_features(features, {})
    classes = audit.set_index("order_id").audit_class.to_dict()

    assert classes == {
        "fail": "auto_fail",
        "manual": "manual_review",
        "pass": "auto_pass",
    }
    assert audit.audit_score.notna().all()
    assert audit.reason_codes.str.len().gt(0).all()
    assert "NO_VALID_ROUTE" in audit.set_index("order_id").loc["fail", "reason_codes"]


def test_candidate_limit_priority_and_order_are_stable() -> None:
    rows = []
    for index in range(35):
        rows.append(
            {
                **_feature_row(f"m{index:02d}"),
                "audit_class": "manual_review",
                "audit_score": float(index),
                "reason_codes": "RISK",
            }
        )
    for index in range(15):
        rows.append(
            {
                **_feature_row(f"f{index:02d}"),
                "audit_class": "auto_fail",
                "audit_score": float(index + 100),
                "reason_codes": "FAIL",
            }
        )
    audit = pd.DataFrame(rows)

    first = select_review_cases(audit)
    second = select_review_cases(audit.sample(frac=1, random_state=7))

    assert len(first) == 40
    assert first.iloc[:30].audit_class.eq("manual_review").all()
    assert first.iloc[30:].audit_class.eq("auto_fail").all()
    assert first.case_index.tolist() == list(range(1, 41))
    assert first[["case_index", "order_id"]].equals(
        second[["case_index", "order_id"]]
    )


def test_information_poor_order_is_excluded_before_route_failure_rules() -> None:
    features = pd.DataFrame(
        [
            _feature_row(
                "sparse",
                modeling_eligible=False,
                modeling_exclusion_reasons="LARGE_UNOBSERVED_MOVEMENT_GAP",
                successful_reconstruction=False,
                route_quality="rejected",
            )
        ]
    )
    audit = classify_audit_features(features, {})
    assert audit.loc[0, "audit_class"] == "excluded_low_information"
    assert audit.loc[0, "audit_score"] == 0
    assert audit.loc[0, "reason_codes"] == "LARGE_UNOBSERVED_MOVEMENT_GAP"


def test_reverse_oneway_traversal_is_informational_not_direction_violation() -> None:
    routes = pd.DataFrame(
        {
            "order_id": ["o"],
            "subtrace_id": ["o:000"],
            "path_id": [0],
            "route_sequence": [0],
            "canonical_edge_uid": ["e"],
            "canonical_from_node": [2],
            "canonical_to_node": [1],
            "canonical_traversal_direction": ["R"],
            "traversed_against_osm_oneway": [True],
        }
    )
    canonical = pd.DataFrame(
        {
            "edge_uid": ["e"],
            "from_node": [1],
            "to_node": [2],
            "direction": ["F"],
        }
    )
    result = _canonical_audit(routes, canonical)
    assert result.loc[0, "direction_violation_count"] == 0
    assert result.loc[0, "osm_oneway_conflict_count"] == 1


def test_native_canonical_reverse_edge_is_already_oriented_for_traversal() -> None:
    routes = pd.DataFrame(
        {
            "order_id": ["o"],
            "subtrace_id": ["o:000"],
            "path_id": [0],
            "route_sequence": [0],
            "canonical_edge_uid": ["e:R"],
            "canonical_from_node": [2],
            "canonical_to_node": [1],
            "canonical_traversal_direction": ["R"],
            "traversed_against_osm_oneway": [False],
        }
    )
    canonical = pd.DataFrame(
        {
            "edge_uid": ["e:R"],
            "from_node": [2],
            "to_node": [1],
            "direction": ["R"],
        }
    )
    result = _canonical_audit(routes, canonical)
    assert result.loc[0, "direction_violation_count"] == 0
    assert result.loc[0, "osm_oneway_conflict_count"] == 0


def test_route_plot_geometry_is_clipped_and_preserves_reverse_direction() -> None:
    routes = pd.DataFrame(
        {
            "canonical_edge_uid": ["e"],
            "canonical_length_m": [100.0],
            "entry_position_m": [80.0],
            "exit_position_m": [20.0],
            "path_id": [0],
            "valhalla_topology_gap_before": [False],
        }
    )
    lines, gaps = _clipped_route_lines(
        routes, {"e": LineString([(0, 0), (10, 0)])}, "canonical_edge_uid"
    )
    assert gaps == []
    assert np.allclose(lines[0][0], [8, 0])
    assert np.allclose(lines[0][-1], [2, 0])


def test_review_index_files_are_generated(tmp_path: Path) -> None:
    row = {
        **_feature_row("order-a"),
        "case_index": 1,
        "audit_class": "manual_review",
        "audit_score": 42.0,
        "primary_reason": "HIGH_SNAP_P90",
        "secondary_reasons": "LOW_DYNAMIC_COVERAGE",
        "risk_window_from_seq": 10,
        "risk_window_to_seq": 40,
        "risk_window_reason": "highest_snap",
        "image_path": "images/case_001_order-a.png",
    }
    index = pd.DataFrame([row])

    write_review_indexes(index, tmp_path)

    written = pd.read_csv(tmp_path / "index.csv")
    assert list(written.columns) == INDEX_COLUMNS
    assert written.loc[0, "case_index"] == 1
    assert "case_001_order-a.png" in (tmp_path / "index.md").read_text(
        encoding="utf-8"
    )


def test_v5_missing_is_explicit_and_not_an_anomaly() -> None:
    orders = pd.DataFrame({"order_id": ["a", "b"]})
    v6 = pd.DataFrame(
        {
            "order_id": ["a"],
            "canonical_edge_uid": ["edge-a"],
            "length_m": [10.0],
        }
    )

    comparison = _v5_comparison(orders, v6, None, None)

    assert not comparison.v5_v6_comparison_available.any()
    assert comparison.v5_v6_edge_jaccard.isna().all()
    assert comparison.v5_v6_route_distance_ratio.isna().all()


def test_risk_windows_only_process_selected_candidates() -> None:
    matched = pd.DataFrame(
        {
            "order_id": ["selected", "selected", "ignored"],
            "original_point_seq": [0, 30, 50],
            "distance_from_trace_point_m": [2.0, 70.0, 90.0],
            "route_discontinuity": [False, False, True],
        }
    )
    intervals = pd.DataFrame(
        {
            "order_id": ["selected", "ignored"],
            "from_original_point_seq": [20, 40],
            "to_original_point_seq": [21, 41],
            "interval_duration_s": [1.0, 1.0],
            "gps_interval_distance_m": [10.0, 100.0],
            "measurement_source": ["unresolved", "unresolved"],
            "interval_reason": ["inferred_path_between_gps_anchors", "unmatched_endpoint"],
        }
    )

    windows = _identify_risk_windows({"selected"}, matched, intervals)

    assert windows.order_id.tolist() == ["selected"]
    assert windows.loc[0, "risk_window_from_seq"] <= 30
    assert windows.loc[0, "risk_window_to_seq"] >= 30


def test_build_features_has_complete_output_without_matcher_or_v5() -> None:
    products = {
        "order_base": pd.DataFrame({"order_id": ["a"], "date": ["20161010"]}),
        "route_quality": pd.DataFrame(
            {
                "order_id": ["a"],
                "successful_reconstruction": [True],
                "matched_point_share": [1.0],
                "matched_interval_share": [1.0],
                "preprocess_break_count": [0],
                "discontinuity_count": [0],
                "od_endpoint_error_m": [2.0],
                "route_distance_m": [200.0],
                "route_resolved_gps_distance_ratio": [1.0],
                "route_raw_gps_distance_ratio": [1.0],
                "canonical_edge_mapping_share": [1.0],
                "route_quality": ["strict_core"],
            }
        ),
        "dynamic_measurement_quality": pd.DataFrame(
            {
                "order_id": ["a"],
                "dynamic_measurement_quality": ["dynamic_partial"],
                "direct_observed_interval_time_share": [0.5],
                "direct_observed_distance_share": [0.5],
                "interval_supported_time_share": [0.2],
                "unresolved_time_share": [0.3],
                "valid_timed_traversal_count": [2],
                "timed_traversal_share": [0.5],
            }
        ),
        "matched_points": pd.DataFrame(
            {
                "order_id": ["a", "a"],
                "distance_from_trace_point_m": [2.0, 4.0],
            }
        ),
        "interval_measurements": pd.DataFrame(
            {
                "order_id": ["a"],
                "interval_duration_s": [10.0],
                "gps_interval_distance_m": [100.0],
            }
        ),
        "unresolved_intervals": pd.DataFrame(
            {
                "order_id": pd.Series(dtype=str),
                "unresolved_reason": pd.Series(dtype=str),
                "unresolved_interval_time_s": pd.Series(dtype=float),
            }
        ),
        "route_parts": pd.DataFrame(
            {
                "order_id": ["a", "a"],
                "subtrace_id": ["s", "s"],
                "path_id": [0, 0],
                "route_sequence": [0, 1],
                "canonical_edge_uid": ["e1", "e2"],
                "canonical_from_node": [1, 2],
                "canonical_to_node": [2, 3],
                "length_m": [100.0, 100.0],
            }
        ),
        "link_traversals": pd.DataFrame(
            {"order_id": ["a"], "traversal_id": [0]}
        ),
        "turn_movements": pd.DataFrame(
            {"order_id": ["a"], "movement_sequence": [0]}
        ),
        "interval_accounting": pd.DataFrame(
            {
                "order_id": ["a"],
                "time_conservation_valid": [True],
                "timestamp_anchor_valid": [True],
            }
        ),
    }
    canonical = pd.DataFrame(
        {
            "edge_uid": ["e1", "e2"],
            "from_node": [1, 2],
            "to_node": [2, 3],
        }
    )

    features = build_audit_features(products, canonical, {})
    audit = classify_audit_features(features, {})

    assert len(audit) == 1
    assert set(AUDIT_COLUMNS).issubset(audit.columns)
    assert audit.loc[0, "topology_violation_count"] == 0
    assert audit.loc[0, "direction_violation_count"] == 0
    assert audit.loc[0, "dynamic_coverage_class"] == "direct_time_observations"
    assert audit.loc[0, "v5_v6_comparison_available"] == np.False_
