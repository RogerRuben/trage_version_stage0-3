from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.sparsity_diagnostic import classify_diagnostic
from stage2.v5_2.sparsity_support import (
    IDENTITY_COLUMNS, TARGETS, TARGET_VALID_COLUMNS, cluster_bootstrap_difference,
    fit_support_counts, positive_quantiles, spatial_high_temporal_sparse,
    support_groups, validate_prediction_alignment,
)


REPO = Path(__file__).resolve().parents[2]


def _train_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "split": ["train"] * 5,
        "date": ["20161009"] * 5,
        "order_id": ["a", "a", "b", "c", "d"],
        "traversal_id": [1, 2, 1, 1, 1],
        "observed_directed_edge_uid": ["e1", "e1", "e1", "e2", "e2"],
        "estimated_time_bin": [1, 1, 2, 1, 2],
        "crawl_target_valid": [True, False, True, True, True],
        "stop_target_valid": [True, True, True, True, True],
        "speed_cv_target_valid": [False, False, True, False, True],
        "acceleration_rms_target_valid": [False, False, False, True, False],
    })


def test_spatial_support_train_only() -> None:
    fitted = fit_support_counts([_train_frame()], expected_dates=("20161009",))
    assert fitted.spatial.to_dict() == {"e1": 3, "e2": 2}
    assert fitted.unique_physical_traversal_count == 5


def test_spatiotemporal_support_uses_existing_time_bin() -> None:
    fitted = fit_support_counts([_train_frame()], expected_dates=("20161009",))
    assert fitted.spatiotemporal.loc[("e1", 1)] == 2
    assert fitted.spatiotemporal.loc[("e1", 2)] == 1
    invalid = _train_frame()
    invalid.loc[0, "estimated_time_bin"] = 48
    with pytest.raises(Stage2V52ContractError, match="frozen 0-47"):
        fit_support_counts([invalid], expected_dates=("20161009",))


def test_target_specific_support_respects_target_valid_mask() -> None:
    fitted = fit_support_counts([_train_frame()], expected_dates=("20161009",))
    assert fitted.target_specific["crawl"].loc[("e1", 1)] == 1
    assert fitted.target_specific["stop"].loc[("e1", 1)] == 2
    assert ("e1", 1) not in fitted.target_specific["speed_cv"].index


def test_overlap_tokens_not_double_counted() -> None:
    frame = _train_frame()
    duplicated = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    fitted = fit_support_counts([duplicated], expected_dates=("20161009",))
    assert fitted.spatial.loc["e1"] == 3
    assert fitted.duplicate_removed_count == 1


def test_evaluation_rows_never_enter_support_fit() -> None:
    frame = _train_frame()
    frame["split"] = "evaluation"
    with pytest.raises(Stage2V52ContractError, match="evaluation rows"):
        fit_support_counts([frame], expected_dates=("20161009",))


def test_support_groups_use_train_only_quantiles() -> None:
    counts = pd.Series([1, 2, 3, 100], index=["a", "b", "c", "d"])
    quantiles = positive_quantiles(counts)
    assert np.array_equal(
        support_groups([0, 1, 2, 3, 100], quantiles),
        np.array(["unseen", "low", "medium", "medium", "high"]),
    )


def test_spatial_high_temporal_sparse_classification() -> None:
    result = spatial_high_temporal_sparse(
        ["high", "high", "medium", "high"], ["unseen", "low", "low", "high"],
    )
    assert result.tolist() == [True, True, False, False]


def _prediction_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "date": ["20161025", "20161025"], "order_id": ["a", "b"],
        "traversal_id": [1, 1],
    })
    for target in TARGETS:
        frame[f"{target}_valid"] = [True, False]
        frame[f"target_{target}"] = [0.2, np.nan]
    return frame


def test_prediction_identity_alignment() -> None:
    reference = _prediction_frame()
    validate_prediction_alignment({model: reference.copy() for model in ("M1", "M3", "M4")})


def test_m1_m3_m4_same_physical_traversal_pairing() -> None:
    reference = _prediction_frame()
    reordered = reference.iloc[::-1].reset_index(drop=True)
    with pytest.raises(Stage2V52ContractError, match="not exactly paired"):
        validate_prediction_alignment({"M1": reference, "M3": reordered, "M4": reference.copy()})


def test_bootstrap_seed_is_frozen() -> None:
    config = json.loads(
        (REPO / "stage2/config/stage2_v5_2_sparsity_diagnostic.json").read_text(encoding="utf-8")
    )
    assert config["bootstrap"] == {
        "cluster_unit": ["observed_directed_edge_uid", "estimated_time_bin"],
        "resamples": 1000, "confidence_level": 0.95, "seed": 20261009,
    }
    first = cluster_bootstrap_difference(
        [1.0, 2.0, 3.0], [0.5, 1.5, 2.5], resamples=1000,
        seed=config["bootstrap"]["seed"], confidence_level=0.95,
    )
    second = cluster_bootstrap_difference(
        [1.0, 2.0, 3.0], [0.5, 1.5, 2.5], resamples=1000,
        seed=config["bootstrap"]["seed"], confidence_level=0.95,
    )
    assert first == second


def test_rts_excluded_from_sparsity_main_analysis() -> None:
    assert "rts" not in TARGETS
    assert "rts" not in TARGET_VALID_COLUMNS
    config = json.loads(
        (REPO / "stage2/config/stage2_v5_2_sparsity_diagnostic.json").read_text(encoding="utf-8")
    )
    assert tuple(config["targets"]) == TARGETS
    assert "rts" not in config["targets"]


def test_identity_columns_are_physical_traversal_not_overlap_chunk() -> None:
    assert IDENTITY_COLUMNS == ("date", "order_id", "traversal_id")


def test_diag_b_uses_complete_spatiotemporal_sparse_groups() -> None:
    correlations = pd.DataFrame([
        {
            "scope": scope, "date": date, "target": target,
            "support_dimension": dimension,
            "spearman_log1p_support_vs_cell_mae": 0.01,
        }
        for target in TARGETS
        for scope, date in (("aggregate", "all"), ("daily", "20161025"), ("daily", "20161026"), ("daily", "20161027"))
        for dimension in ("spatial", "spatiotemporal", "target_specific")
    ])
    bootstrap = pd.DataFrame([
        {"target": target, "comparison": comparison, "effect": -0.01}
        for target in TARGETS
        for comparison in (
            "spatiotemporal_low_vs_high", "spatiotemporal_unseen_vs_high",
            "spatial_high_temporal_sparse_vs_high_high",
        )
    ])
    rows = []
    for target in TARGETS:
        positive = target != "stop"
        for group in ("unseen", "low"):
            for comparison in ("M3_vs_M1", "M4_vs_M3"):
                rows.append({
                    "scope": "aggregate", "date": "all", "support_dimension": "spatiotemporal",
                    "target": target, "support_group": group, "comparison": comparison,
                    "absolute_improvement": 0.01 if comparison == "M3_vs_M1" and positive else -0.01,
                })
        for date in ("20161025", "20161026", "20161027"):
            for comparison in ("M3_vs_M1", "M4_vs_M3"):
                rows.append({
                    "scope": "daily", "date": date, "support_dimension": "spatiotemporal",
                    "target": target, "support_group": "low", "comparison": comparison,
                    "absolute_improvement": 0.01 if comparison == "M3_vs_M1" and positive else -0.01,
                })
    result = classify_diagnostic(
        correlations=correlations, bootstrap=bootstrap,
        mismatch_summary={"evaluation_traversals": 100}, transfer_metrics=pd.DataFrame(rows),
    )
    assert result["classification"] == "DIAG-B"
    assert result["evidence"]["structured_transfer_positive_targets"] == [
        "crawl", "speed_cv", "acceleration_rms",
    ]
