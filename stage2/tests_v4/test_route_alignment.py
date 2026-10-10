from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from stage2.v4.contracts import Stage2V4ContractError
from stage2.v4.stage1_adapter import Stage1BucketRef, build_route_alignment


def _route() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train", "train"],
            "date": ["20161009", "20161009"],
            "order_id": ["o1", "o1"],
            "route_sequence": [0, 1],
            "canonical_edge_uid": ["c0", "c1"],
            "observed_directed_edge_uid": ["e0", "e1"],
            "observed_from_node": [1, 2],
            "observed_to_node": [2, 3],
            "observed_direction": ["forward", "forward"],
            "route_part_length_m": [10.0, 20.0],
            "canonical_highway": ["primary", "primary"],
            "road_class": ["primary", "primary"],
            "bridge": [False, False],
            "tunnel": [False, False],
            "synthetic_reverse_edge": [False, False],
            "osm_direction_disagreement": [False, False],
            "sequence_feature_mask": [True, True],
            "directed_edge_model_scope": ["seen", "seen"],
        }
    )


def _traversal() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train"],
            "date": ["20161009"],
            "order_id": ["o1"],
            "traversal_id": [7],
            "route_sequence": [1],
            "canonical_edge_uid": ["c1"],
            "observed_directed_edge_uid": ["e1"],
            "crawl_time_share": [0.2],
            "stop_time_share": [0.0],
            "speed_cv_bounded": [0.3],
            "acceleration_rms_bounded": [0.4],
            "acceleration_pair_count": [0],
            "acceleration_weight_s": [0.0],
            "direct_interval_count": [3],
            "direct_observed_time_s": [12.0],
            "observation_window_end_time": [101.0],
            "lcs_raw": [float("nan")],
            "lcs_pct": [float("nan")],
            "lcs_tail_event": pd.Series([pd.NA], dtype="boolean"),
            "lcs_available": [False],
            "lcs_unavailable_reason": ["INSUFFICIENT_ACCELERATION_PAIRS"],
            "rts_raw": [0.1],
            "rts_pct": [0.2],
            "rts_tail_event": pd.Series([False], dtype="boolean"),
            "rts_available": [True],
            "rts_measurement_available": [True],
            "rts_unavailable_reason": [""],
            "reference_model_id": ["model"],
            "label_schema_version": ["stage1_label_schema_v3"],
        }
    )


def _orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "split": ["train"],
            "date": ["20161009"],
            "order_id": ["o1"],
            "departure_time": [100.0],
            "start_node": [1],
            "end_node": [3],
            "stage1_core_eligible": [True],
        }
    )


def _physical() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "order_id": ["o1", "o1"],
            "traversal_id": [6, 7],
            "route_sequence": [0, 1],
            "route_sequence_end": [0, 1],
            "enter_time": [100.0, 101.0],
            "exit_time": [101.0, 103.0],
            "travel_time_s": [1.0, 2.0],
            "time_source": ["engine", "direct"],
            "time_observation_valid": [False, True],
            "measurement_source": ["engine_interpolated", "direct_observed"],
            "allocated_distance_m": [10.0, 20.0],
        }
    )


def _write_bucket(tmp_path: Path, traversals: pd.DataFrame | None = None) -> Stage1BucketRef:
    output = tmp_path / "out"
    source_input = tmp_path / "in"
    output.mkdir()
    source_input.mkdir()
    _route().to_parquet(output / "route_sequence_context.parquet", index=False)
    (traversals if traversals is not None else _traversal()).to_parquet(
        output / "traversal_labels.parquet",
        index=False,
    )
    _orders().to_parquet(source_input / "order_base.parquet", index=False)
    _physical().to_parquet(source_input / "link_traversals.parquet", index=False)
    return Stage1BucketRef(
        split="train",
        date="20161009",
        bucket=0,
        output_path=output,
        input_path=source_input,
    )


def test_route_skeleton_retains_unlabelled_token_and_order_context(tmp_path: Path) -> None:
    result = build_route_alignment(_write_bucket(tmp_path))
    assert len(result.route_tokens) == 2
    assert result.route_tokens["label_available"].tolist() == [False, True]
    assert result.route_tokens["decision_time"].tolist() == [100.0, 100.0]
    assert set(result.route_tokens["decision_time_source"]) == {
        "stage0_order_departure_time"
    }
    assert pd.isna(result.route_tokens.loc[0, "crawl_time_share"])
    assert pd.isna(result.route_tokens.loc[0, "lcs_tail_event"])


def test_component_masks_are_independent_of_lcs_scalar(tmp_path: Path) -> None:
    result = build_route_alignment(_write_bucket(tmp_path))
    labeled = result.route_tokens.loc[result.route_tokens["label_available"]].iloc[0]
    assert bool(labeled["crawl_target_valid"])
    assert bool(labeled["stop_target_valid"])
    assert bool(labeled["speed_cv_target_valid"])
    assert not bool(labeled["acceleration_rms_target_valid"])
    assert not bool(labeled["lcs_target_valid"])
    assert bool(labeled["rts_target_valid"])


def test_alignment_records_audited_one_to_one_span(tmp_path: Path) -> None:
    result = build_route_alignment(_write_bucket(tmp_path))
    row = result.traversal_alignment.iloc[0]
    assert row["traversal_span_start_sequence"] == 1
    assert row["traversal_span_end_sequence"] == 1
    assert row["traversal_span_length"] == 1
    assert row["alignment_status"] == "one_to_one"


def test_duplicate_label_route_join_fails(tmp_path: Path) -> None:
    traversals = pd.concat([_traversal(), _traversal().assign(traversal_id=8)])
    with pytest.raises(Stage2V4ContractError, match="route mapping"):
        build_route_alignment(_write_bucket(tmp_path, traversals))
