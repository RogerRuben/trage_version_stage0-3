from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage1.v3.aggregation import aggregate_order_labels
from stage1.v3.config import load_config, validate_config, validate_split_config
from stage1.v3.histograms import (
    FixedBinHistogram,
    empirical_cdf_from_histogram,
)
from stage1.v3.input_adapter import (
    BucketRef,
    _enrich_direction_lineage,
    _fill_nullable_order_endpoints,
    _normalize_nullable_dtypes,
)
from stage1.v3.io import canonical_json_bytes, sha256_bytes
from stage1.v3.models import load_model_bundle, write_model_bundle
from stage1.v3.pipeline import _validate_output_frames
from stage1.v3.primitives import (
    build_interval_labels,
    build_traversal_primitives,
    select_direct_observations,
)
from stage1.v3.references import (
    SparseCohortHistograms,
    apply_reference_labels,
)
from stage1.v3.schema import (
    ContractError,
    OUTPUT_PRIMARY_KEYS,
    OUTPUT_REQUIRED_COLUMNS,
)
from stage1.v3.support import (
    apply_directed_support,
    build_directed_edge_catalog,
    fit_directed_support,
    fit_directed_support_from_observations,
)


TRAIN_DATES = [f"201610{day:02d}" for day in range(9, 25)]
VALIDATION_DATES = [f"201610{day:02d}" for day in range(25, 28)]


def _config_payload() -> dict:
    return {
        "schema_version": "stage1_label_schema_v3",
        "status": "review_candidate",
        "label_temporality": "retrospective_post_trip_realized_label",
        "core_composite_status": "disabled",
        "stage0_release": {
            "stage0_tag": "stage0-v6-final",
            "stage0_tag_commit": (
                "729275d81ec5dc224ac0967a6e600457764607b8"
            ),
            "stage0_source_content_hash": "a5e482f4a0d2b607",
        },
        "split": {
            "train_dates": TRAIN_DATES,
            "validation_dates": VALIDATION_DATES,
            "test_date": "20161031",
            "reference_fit_dates": TRAIN_DATES,
        },
        "time": {"timezone": "Asia/Shanghai"},
        "direct": {
            "duration_tolerance_s": 1e-6,
            "distance_identity_tolerance_m": 1e-6,
            "speed_tolerance_mps": 1e-6,
            "interval_time_rel": 1e-12,
            "interval_speed_rel": 1e-9,
            "dynamic_time_rel": 1e-12,
            "dynamic_distance_rel": 1e-12,
        },
        "lcs": {
            "minimum_direct_intervals_per_traversal": 3,
            "minimum_observed_time_s": 6.0,
            "minimum_direct_observed_distance_m": 10.0,
            "maximum_adjacent_gap_s": 6.0,
            "low_speed_mps": 5.0,
            "stop_speed_mps": 1.0,
            "speed_cv_scale": 1.0,
            "acceleration_rms_scale_mps2": 1.0,
            "minimum_acceleration_pairs": 2,
            "maximum_physical_speed_mps": 75.0,
            "maximum_absolute_acceleration_mps2": 8.0,
            "components": {
                "low_speed_time_share": {"weight": 0.25},
                "stop_time_share": {"weight": 0.25},
                "speed_cv_bounded": {"weight": 0.25},
                "acceleration_rms_bounded": {"weight": 0.25},
            },
        },
        "rts": {
            "minimum_direct_observed_time_s": 3.0,
            "minimum_direct_observed_distance_m": 10.0,
            "minimum_reference_sample_size": 1,
            "sec_per_m_clip": [0.01, 10.0],
            "tail_event_percentile_threshold": 0.90,
        },
        "reference": {
            "minimum_observed_distance_m": 10.0,
            "histogram_min_sec_per_m": 0.01,
            "histogram_max_sec_per_m": 10.0,
            "histogram_bins": 64,
            "quantile": 0.5,
            "minimum_cohort_support": 1,
        },
        "normalization": {
            "raw_bins": 100,
            "minimum_cohort_support": 1,
        },
        "aggregation": {"tail_percentile_threshold": 0.90},
        "cohort_reference": {
            "time_bin_minutes": 30,
            "minimum_sample_size": 1,
            "peak_windows_local": [
                ["07:00", "09:30"],
                ["17:00", "19:30"],
            ],
            "train_reference_application": "leave_one_out",
            "validation_test_reference_application": "full_train_frozen",
            "raw_cdf_application": "full_train_empirical_self_rank_for_train",
            "fallback": [
                {"name": name}
                for name in (
                    "edge_time_weekday",
                    "edge_peak",
                    "edge",
                    "highway_time_weekday",
                    "highway",
                    "global",
                )
            ],
            "fallback_policy": (
                "first_level_meeting_minimum_sample_size_"
                "else_global_if_nonempty_else_na"
            ),
        },
        "coverage": {
            "minimum_direct_interval_count": 2,
            "minimum_unique_timed_edge_count": 2,
            "minimum_observed_time_share": 0.50,
            "minimum_observed_distance_share": 0.50,
        },
        "support": {
            "threshold_status": "review_candidate",
            "fit_scope": "train_only",
            "minimum_edge_observations": 2,
            "minimum_edge_hour_observations": 1,
            "minimum_fallback_observations": 1,
            "fallback_order": [
                "road_class_hour",
                "spatial_neighbor",
                "upper_spatial_region",
                "global_hour",
            ],
            "validation_test_policy": "apply_frozen_train_support_only",
        },
        "preflight": {
            "expected_order_count": 220000,
            "expected_split_counts": {
                "train": 160000,
                "validation": 30000,
                "test": 30000,
            },
        },
        "gns": {
            "status": "external_static_extension",
            "core_label_role": "excluded",
        },
        "iis": {
            "status": "unavailable",
            "available": False,
            "unavailable_reason": (
                "STAGE0_V6_MOVEMENT_DYNAMIC_EVIDENCE_NOT_AVAILABLE"
            ),
        },
        "pmis": {
            "status": "unavailable",
            "available": False,
            "unavailable_reason": (
                "CURRENT_CANONICAL_POI_EXPOSURE_EXTENSION_NOT_FROZEN"
            ),
        },
        "outputs": {
            product: {
                "primary_key": list(OUTPUT_PRIMARY_KEYS[product]),
                "required_fields": sorted(required),
            }
            for product, required in OUTPUT_REQUIRED_COLUMNS.items()
        },
    }


def _load_config(tmp_path, payload: dict | None = None):
    path = tmp_path / "stage1_v3_test.json"
    path.write_text(
        json.dumps(payload or _config_payload(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return load_config(path)


def _observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1", "o1", "o1"],
            "gps_interval_id": [1, 2, 3, 4],
            "traversal_id": [10, 10, 11, 11],
            "canonical_edge_uid": [
                "segment1:F",
                "segment1:F",
                "segment2:F",
                "segment2:F",
            ],
            "interval_start_time": [0.0, 5.0, 10.0, 15.0],
            "interval_end_time": [5.0, 10.0, 15.0, 20.0],
            "observed_travel_time_s": [5.0, 5.0, 5.0, 5.0],
            "observed_distance_m": [25.0, 25.0, 25.0, 25.0],
            "observed_speed_mps": [5.0, 5.0, 5.0, 5.0],
            "measurement_source": [
                "direct_observed",
                "direct_observed",
                "interval_supported",
                "engine_interpolated",
            ],
            "label_valid": [True, False, True, True],
        }
    )


def _traversals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "traversal_id": [10, 11],
            "route_sequence": [0, 1],
            "canonical_edge_uid": ["segment1:F", "segment2:F"],
            "observed_directed_edge_uid": ["segment1:F", "segment2:F"],
            "observed_from_node": pd.Series([1, 2], dtype="Int64"),
            "observed_to_node": pd.Series([2, 3], dtype="Int64"),
            "observed_direction": ["F", "F"],
            "synthetic_reverse_edge": [False, False],
            "osm_direction_disagreement": [False, False],
            "canonical_mapping_available": [True, True],
            "mapping_status": ["unique", "unique"],
            "osm_oneway": [False, False],
            "measurement_source": ["direct_observed", "interval_supported"],
            "allocated_distance_m": [50.0, 50.0],
        }
    )


def _route_parts() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "route_sequence": [0, 1],
            "canonical_edge_uid": ["segment1:F", "segment2:F"],
            "observed_directed_edge_uid": ["segment1:F", "segment2:F"],
            "observed_from_node": pd.Series([1, 2], dtype="Int64"),
            "observed_to_node": pd.Series([2, 3], dtype="Int64"),
            "observed_direction": ["F", "F"],
            "synthetic_reverse_edge": [False, False],
            "osm_direction_disagreement": [False, False],
            "canonical_mapping_available": [True, True],
            "mapping_status": ["unique", "unique"],
            "osm_oneway": [False, False],
            "canonical_highway": ["primary", "primary"],
            "canonical_length_m": [50.0, 50.0],
            "length_m": [50.0, 50.0],
            "measurement_source": ["direct_observed", "interval_supported"],
        }
    )


def _aggregation_traversals() -> pd.DataFrame:
    frame = _traversals().copy()
    frame["measurement_source"] = "direct_observed"
    return frame


def _traversal_labels(
    *,
    direct_interval_count: tuple[int, int] = (1, 1),
    rts_missing: bool = False,
    interaction_missing: bool = True,
) -> pd.DataFrame:
    rts_raw = [np.nan, np.nan] if rts_missing else [0.2, 0.4]
    rts_pct = [np.nan, np.nan] if rts_missing else [0.3, 0.5]
    iis_raw = [np.nan, np.nan] if interaction_missing else [0.1, 0.2]
    iis_pct = [np.nan, np.nan] if interaction_missing else [0.2, 0.3]
    pmis_raw = [np.nan, np.nan] if interaction_missing else [0.1, 0.2]
    pmis_pct = [np.nan, np.nan] if interaction_missing else [0.2, 0.3]
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "traversal_id": [10, 11],
            "route_sequence": [0, 1],
            "canonical_edge_uid": ["segment1:F", "segment2:F"],
            "observed_directed_edge_uid": ["segment1:F", "segment2:F"],
            "lcs_available": [True, True],
            "lcs_unavailable_reason": ["", ""],
            "lcs_raw": [0.2, 0.4],
            "lcs_pct": [0.3, 0.5],
            "gns_available": [False, False],
            "gns_unavailable_reason": [
                "EDGE_STATIC_FEATURE_EXTENSION_NOT_FITTED",
                "EDGE_STATIC_FEATURE_EXTENSION_NOT_FITTED",
            ],
            "gns_raw": [np.nan, np.nan],
            "gns_pct": [np.nan, np.nan],
            "rts_available": [not rts_missing, not rts_missing],
            "rts_unavailable_reason": (
                ["NO_REFERENCE", "NO_REFERENCE"] if rts_missing else ["", ""]
            ),
            "rts_raw": rts_raw,
            "rts_pct": rts_pct,
            "iis_available": [False, False],
            "iis_unavailable_reason": [
                "STAGE0_V6_MOVEMENT_DYNAMIC_EVIDENCE_NOT_AVAILABLE",
                "STAGE0_V6_MOVEMENT_DYNAMIC_EVIDENCE_NOT_AVAILABLE",
            ],
            "iis_raw": iis_raw,
            "iis_pct": iis_pct,
            "pmis_available": [False, False],
            "pmis_unavailable_reason": [
                "CURRENT_CANONICAL_POI_EXPOSURE_EXTENSION_NOT_FROZEN",
                "CURRENT_CANONICAL_POI_EXPOSURE_EXTENSION_NOT_FROZEN",
            ],
            "pmis_raw": pmis_raw,
            "pmis_pct": pmis_pct,
            "direct_observed_time_s": [30.0, 30.0],
            "direct_observed_distance_m": [25.0, 25.0],
            "allocated_distance_m": [50.0, 50.0],
            "direct_interval_count": list(direct_interval_count),
            "label_valid": [True, True],
        }
    )


def _order_base() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1"],
            "date": ["20161009"],
            "split": ["train"],
            "departure_time": [0.0],
            "arrival_time": [100.0],
            "stage1_core_eligible": [True],
        }
    )


def test_select_direct_observations_is_strictly_direct_and_valid():
    selected = select_direct_observations(_observations())

    assert selected["gps_interval_id"].tolist() == [1]
    assert selected["measurement_source"].eq("direct_observed").all()
    assert selected["label_valid"].all()


def test_select_direct_observations_rejects_duplicate_interval_allocation():
    observations = _observations().iloc[[0, 0]].copy()

    with pytest.raises(ContractError, match="gps_interval_id|duplicate"):
        select_direct_observations(observations)


def test_select_direct_observations_rejects_string_boolean():
    observations = _observations().iloc[[0]].copy()
    observations["label_valid"] = "False"

    with pytest.raises(ContractError, match="boolean|label_valid"):
        select_direct_observations(observations)


def test_build_primitives_rejects_missing_traversal_foreign_key(tmp_path):
    config = _load_config(tmp_path)
    observations = _observations().iloc[[0]].copy()
    observations["traversal_id"] = 999

    with pytest.raises(ContractError, match="traversal|foreign|orphan"):
        build_traversal_primitives(
            observations,
            _traversals(),
            _route_parts(),
            config,
        )


def test_build_primitives_rejects_edge_identity_mismatch(tmp_path):
    config = _load_config(tmp_path)
    observations = _observations().iloc[[0]].copy()
    observations["canonical_edge_uid"] = "wrong-edge"

    with pytest.raises(ContractError, match="canonical_edge_uid|edge"):
        build_traversal_primitives(
            observations,
            _traversals(),
            _route_parts(),
            config,
        )


def test_lcs_rejects_discontinuous_direct_window(tmp_path):
    config = _load_config(tmp_path)
    observations = _observations().iloc[[0, 0, 0]].copy()
    observations["gps_interval_id"] = [1, 2, 3]
    observations["traversal_id"] = 10
    observations["measurement_source"] = "direct_observed"
    observations["label_valid"] = True
    observations["interval_start_time"] = [0.0, 2.0, 20.0]
    observations["interval_end_time"] = [2.0, 4.0, 22.0]
    observations["observed_travel_time_s"] = 2.0
    observations["observed_distance_m"] = 10.0
    observations["observed_speed_mps"] = 5.0

    result = build_traversal_primitives(
        observations,
        _traversals().iloc[[0]].copy(),
        _route_parts().iloc[[0]].copy(),
        config,
    )

    assert not bool(result.iloc[0]["lcs_available"])
    assert (
        result.iloc[0]["lcs_unavailable_reason"]
        == "DISCONTINUOUS_DIRECT_WINDOW"
    )
    assert bool(result.iloc[0]["discontinuous_direct_window"])


def test_lcs_continuous_window_uses_frozen_component_weights(tmp_path):
    config = _load_config(tmp_path)
    observations = _observations().iloc[[0, 0, 0]].copy()
    observations["gps_interval_id"] = [1, 2, 3]
    observations["traversal_id"] = 10
    observations["measurement_source"] = "direct_observed"
    observations["label_valid"] = True
    observations["interval_start_time"] = [0.0, 2.0, 4.0]
    observations["interval_end_time"] = [2.0, 4.0, 6.0]
    observations["observed_travel_time_s"] = 2.0
    observations["observed_distance_m"] = [8.0, 12.0, 8.0]
    observations["observed_speed_mps"] = [4.0, 6.0, 4.0]

    result = build_traversal_primitives(
        observations,
        _traversals().iloc[[0]].copy(),
        _route_parts().iloc[[0]].copy(),
        config,
    )

    row = result.iloc[0]
    expected = np.mean(
        [
            row["low_speed_time_share"],
            row["stop_time_share"],
            row["speed_cv_bounded"],
            row["acceleration_rms_bounded"],
        ]
    )
    assert bool(row["lcs_available"])
    assert row["lcs_raw"] == pytest.approx(expected)


def test_empty_primitives_keep_declared_logical_columns(tmp_path):
    config = _load_config(tmp_path)
    observations = _observations().iloc[0:0].copy()
    traversals = _traversals().iloc[0:0].copy()
    route_parts = _route_parts().iloc[0:0].copy()

    interval_labels = build_interval_labels(
        observations, traversals, route_parts, config
    )
    traversal_labels = build_traversal_primitives(
        observations, traversals, route_parts, config
    )

    assert (
        OUTPUT_REQUIRED_COLUMNS["interval_labels"] - {"split", "date"}
    ).issubset(interval_labels.columns)
    primitive_required = {
        "maximum_absolute_acceleration_mps2",
        "maximum_internal_gap_s",
        "discontinuous_direct_window",
        "direct_distance_exceeds_allocated",
    }
    assert primitive_required.issubset(traversal_labels.columns)


def test_empty_order_outputs_keep_declared_logical_columns(tmp_path):
    config = _load_config(tmp_path)

    labels, quality = aggregate_order_labels(
        _traversal_labels().iloc[0:0].copy(),
        _route_parts().iloc[0:0].copy(),
        _aggregation_traversals().iloc[0:0].copy(),
        _order_base().iloc[0:0].copy(),
        config,
    )

    assert OUTPUT_REQUIRED_COLUMNS["order_labels"].issubset(labels.columns)
    assert OUTPUT_REQUIRED_COLUMNS["order_label_quality"].issubset(
        quality.columns
    )


def test_output_contract_rejects_undeclared_columns():
    products = {
        product: pd.DataFrame(columns=sorted(columns))
        for product, columns in OUTPUT_REQUIRED_COLUMNS.items()
    }
    ref = BucketRef(
        split="train",
        date="20161009",
        bucket=0,
        path=Path("."),
        manifest={},
    )
    _validate_output_frames(products, ref)

    products["interval_labels"]["undeclared"] = pd.Series(dtype="object")
    with pytest.raises(ContractError, match="undeclared"):
        _validate_output_frames(products, ref)


def test_rts_missingness_is_preserved_and_blocks_core_composite(tmp_path):
    config = _load_config(tmp_path)

    labels, quality = aggregate_order_labels(
        _traversal_labels(rts_missing=True),
        _route_parts(),
        _aggregation_traversals(),
        _order_base(),
        config,
    )

    row = labels.iloc[0]
    assert not bool(row["rts_available"])
    assert np.isnan(row["rts_mean"])
    assert "core_composite_mean" not in labels.columns
    assert row["core_composite_status"] == "disabled"
    assert quality.iloc[0]["rts_missing_reason"] != ""


def test_iis_and_pmis_remain_unavailable_not_zero(tmp_path):
    config = _load_config(tmp_path)

    labels, _ = aggregate_order_labels(
        _traversal_labels(interaction_missing=True),
        _route_parts(),
        _aggregation_traversals(),
        _order_base(),
        config,
    )

    row = labels.iloc[0]
    assert not bool(row["iis_available"])
    assert not bool(row["pmis_available"])
    assert np.isnan(row["iis_mean"])
    assert np.isnan(row["pmis_mean"])
    assert "iis" not in row["composition_signature"].split("+")
    assert "pmis" not in row["composition_signature"].split("+")


@pytest.mark.parametrize("dimension", ["iis", "pmis", "gns"])
def test_unavailable_extension_value_is_rejected(tmp_path, dimension):
    config = _load_config(tmp_path)
    traversal_labels = _traversal_labels()
    traversal_labels[f"{dimension}_raw"] = 0.2
    traversal_labels[f"{dimension}_pct"] = 0.3

    with pytest.raises(ContractError, match=dimension.upper()):
        aggregate_order_labels(
            traversal_labels,
            _route_parts(),
            _aggregation_traversals(),
            _order_base(),
            config,
        )


def test_cdf_uses_explicit_tails_and_bin_midranks():
    histogram = FixedBinHistogram.empty(np.array([0.0, 1.0, 2.0]))
    histogram.update(np.array([0.25, 0.75, 1.25, 1.75]))
    query = np.array([-1.0, 0.5, 1.5, 3.0, np.nan])

    result = empirical_cdf_from_histogram(query, histogram)

    assert result[0] == 0.0
    assert result[1] == pytest.approx(0.25)
    assert result[2] == pytest.approx(0.75)
    assert result[3] == 1.0
    assert np.isnan(result[4])
    assert np.allclose(
        histogram.cdf(query[:-1]),
        result[:-1],
        equal_nan=True,
    )


def test_empty_cdf_support_returns_nan():
    histogram = FixedBinHistogram.empty(np.array([0.0, 1.0, 2.0]))

    result = empirical_cdf_from_histogram(
        np.array([-1.0, 0.5, 3.0]),
        histogram,
    )

    assert np.isnan(result).all()


def test_cdf_supported_zero_and_one_use_bin_midranks():
    histogram = FixedBinHistogram.empty(np.array([0.0, 0.5, 1.0]))
    histogram.update(np.array([0.0, 0.0, 1.0, 1.0]))

    result = empirical_cdf_from_histogram(np.array([0.0, 1.0]), histogram)

    assert result[0] == pytest.approx(0.25)
    assert result[1] == pytest.approx(0.75)


def _reference_frame(observed_sec_per_m: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "canonical_edge_uid": ["segment1:F"],
            "observed_directed_edge_uid": ["segment1:F"],
            "canonical_highway": ["primary"],
            "time_bin_30m": [0],
            "weekday_type": ["weekday"],
            "peak_offpeak": ["offpeak"],
            "observed_sec_per_m": [observed_sec_per_m],
        }
    )


def test_rts_missing_reference_remains_na(tmp_path):
    config = _load_config(tmp_path)
    model = SparseCohortHistograms.empty(np.array([0.5, 1.5]))

    result = apply_reference_labels(
        _reference_frame(1.0),
        model,
        config,
        reference_fit_manifest_id="fit",
    )

    assert not bool(result.iloc[0]["rts_available"])
    assert np.isnan(result.iloc[0]["rts_raw"])
    assert result.iloc[0]["rts_unavailable_reason"] == "REFERENCE_SUPPORT_UNAVAILABLE"


def test_rts_valid_no_excess_is_true_zero(tmp_path):
    config = _load_config(tmp_path)
    model = SparseCohortHistograms.empty(np.array([0.5, 1.5]))
    model.update(_reference_frame(1.0), "observed_sec_per_m")

    result = apply_reference_labels(
        _reference_frame(0.8),
        model,
        config,
        reference_fit_manifest_id="fit",
    )

    assert bool(result.iloc[0]["rts_available"])
    assert result.iloc[0]["rts_raw"] == pytest.approx(0.0)


def test_train_reference_leave_one_out_does_not_use_its_own_row(tmp_path):
    config = _load_config(tmp_path)
    model = SparseCohortHistograms.empty(np.array([0.5, 1.5]))
    frame = _reference_frame(1.0)
    model.update(frame, "observed_sec_per_m", clip=True)

    result = apply_reference_labels(
        frame,
        model,
        config,
        reference_fit_manifest_id="fit",
        leave_one_out=True,
    )

    assert not bool(result.iloc[0]["rts_available"])
    assert (
        result.iloc[0]["rts_unavailable_reason"]
        == "REFERENCE_SUPPORT_UNAVAILABLE"
    )


def test_final_split_config_is_valid_and_exact(tmp_path):
    config = _load_config(tmp_path)

    validate_split_config(config)
    validate_config(config)


@pytest.mark.parametrize("section", ["tolerances", "validation"])
def test_config_rejects_legacy_tolerance_overrides(tmp_path, section):
    payload = copy.deepcopy(_config_payload())
    payload[section] = {"dynamic_time_abs_s": 1000.0}
    config = _load_config(tmp_path, payload)

    with pytest.raises(ContractError, match="override|forbidden"):
        validate_config(config)


@pytest.mark.parametrize(
    "mutation",
    [
        "overlap",
        "fit_on_validation",
        "fit_on_test",
        "wrong_test_date",
        "missing_train_date",
    ],
)
def test_split_config_fails_closed_on_leakage_or_drift(tmp_path, mutation):
    payload = copy.deepcopy(_config_payload())
    if mutation == "overlap":
        payload["split"]["validation_dates"].append("20161024")
    elif mutation == "fit_on_validation":
        payload["split"]["reference_fit_dates"].append("20161025")
    elif mutation == "fit_on_test":
        payload["split"]["reference_fit_dates"].append("20161031")
    elif mutation == "wrong_test_date":
        payload["split"]["test_date"] = "20161030"
    elif mutation == "missing_train_date":
        payload["split"]["train_dates"].remove("20161017")

    config = _load_config(tmp_path, payload)

    with pytest.raises(ContractError):
        validate_split_config(config)


def test_aggregation_rejects_duplicate_order_base(tmp_path):
    config = _load_config(tmp_path)
    order_base = pd.concat([_order_base(), _order_base()], ignore_index=True)

    with pytest.raises(ContractError, match="order_id|duplicate"):
        aggregate_order_labels(
            _traversal_labels(),
            _route_parts(),
            _aggregation_traversals(),
            order_base,
            config,
        )


def test_aggregation_rejects_orphan_order(tmp_path):
    config = _load_config(tmp_path)
    traversal_labels = _traversal_labels()
    traversal_labels["order_id"] = "orphan"

    with pytest.raises(ContractError, match="order_id|foreign|orphan"):
        aggregate_order_labels(
            traversal_labels,
            _route_parts(),
            _aggregation_traversals(),
            _order_base(),
            config,
        )


def test_below_coverage_threshold_is_unavailable_and_has_no_composite(tmp_path):
    config = _load_config(tmp_path)
    labels_input = _traversal_labels(direct_interval_count=(1, 0))

    labels, quality = aggregate_order_labels(
        labels_input,
        _route_parts(),
        _aggregation_traversals(),
        _order_base(),
        config,
    )

    row = labels.iloc[0]
    quality_row = quality.iloc[0]
    assert quality_row["direct_interval_count"] == 1
    assert not bool(row["lcs_available"])
    assert not bool(row["rts_available"])
    assert "core_composite_mean" not in labels.columns
    assert row["composition_signature"] != "lcs+gns+rts"


def test_full_dimension_coverage_still_emits_no_core_composite(tmp_path):
    config = _load_config(tmp_path)

    labels, quality = aggregate_order_labels(
        _traversal_labels(),
        _route_parts(),
        _aggregation_traversals(),
        _order_base(),
        config,
    )

    row = labels.iloc[0]
    quality_row = quality.iloc[0]
    assert bool(row["lcs_available"])
    assert not bool(row["gns_available"])
    assert bool(row["rts_available"])
    assert row["core_composition_signature"] == "lcs+rts"
    assert "core_composite_mean" not in labels.columns
    assert "core_composite_tail" not in labels.columns
    assert row["core_composite_status"] == "disabled"
    assert quality_row["direct_interval_count"] == 2
    assert quality_row["unique_timed_edge_count"] == 2
    assert quality_row["observed_time_share"] == pytest.approx(0.60)
    assert quality_row["observed_distance_share"] == pytest.approx(0.50)


def _direction_products() -> dict[str, pd.DataFrame]:
    return {
        "order_base": pd.DataFrame(
            {"order_id": ["o1"], "start_node": [np.nan], "end_node": [np.nan]}
        ),
        "route_parts": pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o1"],
                "route_sequence": [0, 1, 2],
                "canonical_edge_uid": ["physical1:F", "physical2:R", pd.NA],
                "canonical_from_node": [20, 30, pd.NA],
                "canonical_to_node": [10, 40, pd.NA],
                "begin_osm_node_id": [20, 30, 40],
                "end_osm_node_id": [10, 40, 50],
                "canonical_traversal_direction": ["R", "R", pd.NA],
                "mapping_status": ["unique", "unique", "unmapped"],
                "osm_oneway": [True, True, False],
                "traversed_against_osm_oneway": [True, True, False],
                "canonical_highway": ["primary", "secondary", pd.NA],
                "canonical_length_m": [100.0, 80.0, pd.NA],
                "road_class": ["primary", "secondary", pd.NA],
                "bridge": [False, False, False],
                "tunnel": [False, False, False],
            }
        ),
        "link_traversals": pd.DataFrame(
            {
                "order_id": ["o1", "o1", "o1"],
                "traversal_id": [1, 2, 3],
                "route_sequence": [0, 1, 2],
                "canonical_edge_uid": ["physical1:F", "physical2:R", pd.NA],
            }
        ),
        "link_interval_observations": pd.DataFrame(
            {
                "order_id": ["o1", "o1"],
                "gps_interval_id": [1, 2],
                "traversal_id": [1, 2],
                "canonical_edge_uid": ["physical1:F", "physical2:R"],
            }
        ),
        "turn_movements": pd.DataFrame(
            {"via_node": pd.Series(dtype=float)}
        ),
    }


def test_actual_direction_identity_and_unmapped_lineage_are_independent():
    products = _direction_products()
    _normalize_nullable_dtypes(products)
    _enrich_direction_lineage(products)
    _fill_nullable_order_endpoints(products)

    route = products["route_parts"]
    first = route.iloc[0]
    assert first["observed_directed_edge_uid"] == "physical1:R"
    assert int(first["observed_from_node"]) == 20
    assert int(first["observed_to_node"]) == 10
    assert bool(first["synthetic_reverse_edge"])
    assert bool(first["osm_direction_disagreement"])

    already_reverse = route.iloc[1]
    assert already_reverse["observed_directed_edge_uid"] == "physical2:R"
    assert not bool(already_reverse["synthetic_reverse_edge"])
    assert bool(already_reverse["osm_direction_disagreement"])

    gap = route.iloc[2]
    assert not bool(gap["canonical_mapping_available"])
    assert gap["route_lineage_status"] == "unmapped_lineage_gap"
    assert not bool(gap["sequence_feature_mask"])
    assert pd.isna(gap["observed_directed_edge_uid"])
    assert products["order_base"]["start_node"].dtype == "Int64"
    assert products["order_base"]["end_node"].dtype == "Int64"
    assert int(products["order_base"].iloc[0]["start_node"]) == 20
    assert int(products["order_base"].iloc[0]["end_node"]) == 50
    assert len(products["link_interval_observations"]) == 2


def test_synthetic_reverse_is_a_real_graph_edge_with_inherited_static_data(
    tmp_path,
):
    products = _direction_products()
    _normalize_nullable_dtypes(products)
    _enrich_direction_lineage(products)
    catalog = build_directed_edge_catalog([products["route_parts"]])

    reverse = catalog.set_index("observed_directed_edge_uid").loc["physical1:R"]
    assert int(reverse["observed_from_node"]) == 20
    assert int(reverse["observed_to_node"]) == 10
    assert reverse["canonical_highway"] == "primary"
    assert float(reverse["canonical_length_m"]) == pytest.approx(100.0)
    assert bool(reverse["synthetic_reverse_edge"])

    primitive = pd.DataFrame(
        {
            "observed_directed_edge_uid": ["physical1:R"],
            "direct_interval_count": [3],
            "observation_window_start_time": [1475967600.0],
        }
    )
    model = fit_directed_support([primitive], catalog)
    config = _load_config(tmp_path)
    interval_support = fit_directed_support_from_observations(
        [
            pd.DataFrame(
                {
                    "order_id": ["o1", "o1", "o1"],
                    "traversal_id": [1, 1, 1],
                    "gps_interval_id": [1, 2, 3],
                    "observed_directed_edge_uid": ["physical1:R"] * 3,
                    "interval_start_time": [
                        1475967600.0,
                        1475967602.0,
                        1475967604.0,
                    ],
                }
            )
        ],
        catalog,
        config,
    )
    pd.testing.assert_frame_equal(model.counts, interval_support.counts)
    labelled = apply_directed_support(primitive, model, config)
    assert int(labelled.iloc[0]["edge_observation_count"]) == 3
    assert labelled.iloc[0]["edge_support_level"] == "edge"
    assert labelled.iloc[0]["edge_hour_support_level"] == "edge_hour"


def test_route_segment_component_distance_is_forbidden_for_features():
    payload = json.loads(
        Path("stage1/config/stage1_label_schema_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["upstream_contract"]["deprecated_for_features"][
        "route_segments.segment_route_distance_m"
    ] == "component_distance_not_local_segment_distance"


def test_model_bundle_round_trip_includes_directed_graph_and_train_support(
    tmp_path,
):
    config = _load_config(tmp_path)
    products = _direction_products()
    _normalize_nullable_dtypes(products)
    _enrich_direction_lineage(products)
    catalog = build_directed_edge_catalog([products["route_parts"]])
    primitive = pd.DataFrame(
        {
            "observed_directed_edge_uid": ["physical1:R"],
            "direct_interval_count": [3],
            "observation_window_start_time": [1475967600.0],
        }
    )
    support = fit_directed_support([primitive], catalog)
    reference = SparseCohortHistograms.empty(
        np.geomspace(0.01, 10.0, 65)
    )
    normalized = SparseCohortHistograms.empty(np.linspace(0.0, 1.0, 101))
    identities = {
        "train/20161009/00000": {
            "schemas": {"order_base": "schema"},
        }
    }
    source_id = sha256_bytes(canonical_json_bytes(identities))
    freeze_identity = {
        "manifest_sha": "manifest",
        "git_commit_sha": "commit",
        "config_sha": "config",
        "pbf_sha": "pbf",
        "valhalla_tiles_sha": "tiles",
        "fixed600_sample_sha": "sample",
        "fixed600_summary_sha": "summary",
    }
    target = tmp_path / "models"
    write_model_bundle(
        target,
        reference=reference,
        lcs=normalized,
        rts=normalized,
        support=support,
        config=config,
        source_manifest_id=source_id,
        input_bucket_identities=identities,
        upstream_identity={
            "code_sha": "code",
            "config_sha": "config",
            "tiles_sha": "tiles",
        },
        stage0_freeze_identity=freeze_identity,
        stage1_code_sha="stage1",
    )
    loaded = load_model_bundle(target, config)
    assert "physical1:R" in set(
        loaded.support.edge_catalog["observed_directed_edge_uid"]
    )
    assert loaded.manifest["support_fit_scope"] == "train_only"
    assert loaded.manifest["stage0_release"]["stage0_tag"] == "stage0-v6-final"
