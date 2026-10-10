from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage2.v4.stage1_adapter import _add_component_masks
from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.upstream_sampling_audit import _load_config, refresh_evidence, run_audit
from stage2.v5_2.upstream_sampling_support import (
    REJECTION_TOKEN_GROUPS,
    add_frozen_stage1_target_masks,
    aggregate_rank_decision,
    assert_disjoint_identity_sets,
    assert_distribution_sums_to_one,
    assert_selection_hash_contract,
    assign_support_groups,
    cluster_bootstrap_rate_effect,
    material_negative_rate_effect,
    normalized_selection_rank,
    positive_support_quantiles,
    rejection_mechanisms,
    selection_hex,
    validate_funnel_identity,
)


def _labels() -> pd.DataFrame:
    return pd.DataFrame({
        "direct_interval_count": [1, 1, 2, 2],
        "direct_observed_time_s": [10.0, 0.0, 12.0, 12.0],
        "crawl_time_share": [0.2, 0.2, 0.3, 1.2],
        "stop_time_share": [0.1, 0.1, 0.2, 0.2],
        "speed_cv_bounded": [0.2, 0.2, 0.3, 0.3],
        "acceleration_rms_bounded": [0.2, 0.2, 0.3, 0.3],
        "acceleration_pair_count": [1, 1, 0, 2],
        "acceleration_weight_s": [2.0, 2.0, 0.0, 4.0],
        "lcs_available": [False] * 4,
        "lcs_raw": [np.nan] * 4,
        "lcs_pct": [np.nan] * 4,
        "rts_available": [False] * 4,
        "rts_measurement_available": [False] * 4,
        "rts_raw": [np.nan] * 4,
        "rts_pct": [np.nan] * 4,
        "lcs_tail_event": pd.Series([pd.NA] * 4, dtype="boolean"),
        "rts_tail_event": pd.Series([pd.NA] * 4, dtype="boolean"),
    })


def test_raw_candidate_counts_are_not_accepted_counts() -> None:
    validate_funnel_identity(
        raw=100, processed=40, accepted=25, rejected=15,
        unprocessed_quota=60,
    )
    with pytest.raises(Stage2V52ContractError):
        validate_funnel_identity(
            raw=25, processed=40, accepted=25, rejected=15,
            unprocessed_quota=0,
        )


def test_selection_rank_uses_only_date_order_seed() -> None:
    date, seed = "20161009", 20261009
    frame = pd.DataFrame({
        "order_id": ["b", "a", "c"],
        "selection_hash": [
            selection_hex(date, value, seed) for value in ("b", "a", "c")
        ],
        "origin_grid": ["x", "y", "z"],
    })
    assert_selection_hash_contract(frame, date=date, seed=seed)
    changed = frame.copy()
    changed.loc[0, "selection_hash"] = selection_hex(date, "different", seed)
    with pytest.raises(Stage2V52ContractError):
        assert_selection_hash_contract(changed, date=date, seed=seed)
    ranks = normalized_selection_rank(frame)
    assert sorted(ranks.tolist()) == pytest.approx([1 / 6, 3 / 6, 5 / 6])


def test_quota_unprocessed_is_distinguished_from_rejected() -> None:
    validate_funnel_identity(
        raw=10, processed=6, accepted=4, rejected=2,
        unprocessed_quota=4,
    )
    with pytest.raises(Stage2V52ContractError):
        validate_funnel_identity(
            raw=10, processed=6, accepted=4, rejected=6,
            unprocessed_quota=0,
        )


def test_rejection_reason_mapping_is_exhaustive() -> None:
    for token in REJECTION_TOKEN_GROUPS:
        assert rejection_mechanisms(token)
    assert rejection_mechanisms(
        "ROUTE_NOT_PASS|CANONICAL_NOT_RESOLVED|INSUFFICIENT_TIMED_EDGES"
    ) == (
        "ROUTE/MAP-MATCH", "CANONICAL/NETWORK",
        "DYNAMIC/SUPERVISION-RELATED",
    )
    assert rejection_mechanisms("MATCH_REJECTION:RuntimeError:failed") == (
        "ROUTE/MAP-MATCH",
    )
    with pytest.raises(Stage2V52ContractError):
        rejection_mechanisms("UNKNOWN_NEW_REASON")


def test_raw_support_fit_is_train_only() -> None:
    train_counts = pd.Series([1, 2, 8], index=["a", "b", "c"])
    quantiles = positive_support_quantiles(train_counts)
    evaluation = pd.Series(["a", "future"])
    groups = assign_support_groups(evaluation, train_counts.to_dict(), quantiles)
    assert groups.iloc[1] == "unseen"
    assert "future" not in train_counts.index


def test_accepted_order_identity_reconciles_stage0_manifest() -> None:
    raw = {("20161009", "a"), ("20161009", "b"), ("20161009", "c")}
    accepted = {("20161009", "a")}
    rejected = {("20161009", "b")}
    assert_disjoint_identity_sets(raw, accepted, rejected)
    with pytest.raises(Stage2V52ContractError):
        assert_disjoint_identity_sets(raw, accepted, {("20161009", "a")})
    with pytest.raises(Stage2V52ContractError):
        assert_disjoint_identity_sets(raw, {("20161009", "missing")}, rejected)


def test_stage1_target_valid_attrition_uses_frozen_masks() -> None:
    source = _labels()
    expected = source.copy()
    _add_component_masks(expected)
    actual = add_frozen_stage1_target_masks(source)
    for column in (
        "crawl_target_valid", "stop_target_valid", "speed_cv_target_valid",
        "acceleration_rms_target_valid",
    ):
        assert actual[column].tolist() == expected[column].tolist()


def test_stage1_frozen_manifest_uses_engineering_status() -> None:
    source = inspect.getsource(run_audit.__globals__["_audit_stage1_labels"])
    assert 'payload.get("engineering_status") != "PASS"' in source
    assert 'payload.get("status") != "PASS"' not in source


def test_distribution_share_sums_to_one() -> None:
    assert_distribution_sums_to_one([0.2, 0.3, 0.5])
    with pytest.raises(Stage2V52ContractError):
        assert_distribution_sums_to_one([0.2, 0.3])


def test_rare_context_funnel_identity() -> None:
    daily = pd.DataFrame({
        "date": ["d1", "d1", "d2", "d2"],
        "comparison_group": ["rare", "common", "rare", "common"],
        "accepted": [2, 8, 3, 7],
        "processed": [5, 10, 6, 10],
    })
    result = cluster_bootstrap_rate_effect(
        daily, numerator="accepted", denominator="processed",
        replicates=20, seed=3,
    )
    assert result.rare_count == 5
    assert result.rare_denominator == 11
    assert result.common_count == 15
    assert result.common_denominator == 20


def test_material_effect_accepts_absolute_or_relative_threshold() -> None:
    absolute = {
        "rate_ratio": 0.89,
        "percentage_point_difference": -0.07,
        "difference_ci_high": -0.05,
    }
    assert material_negative_rate_effect(
        absolute, maximum_rate_ratio=0.85, minimum_absolute_gap=0.02
    )
    small = {
        "rate_ratio": 0.99,
        "percentage_point_difference": -0.005,
        "difference_ci_high": 0.001,
    }
    assert not material_negative_rate_effect(
        small, maximum_rate_ratio=0.85, minimum_absolute_gap=0.02
    )


def test_rank_decision_excludes_tiny_strata_and_aggregates_dates() -> None:
    frame = pd.DataFrame({
        "dimension": ["raw_sparsity_group"] * 4,
        "stratum": ["unseen", "high", "unseen", "high"],
        "count": [2, 100, 3, 120],
        "mean_rank": [0.9, 0.502, 0.1, 0.498],
    })
    aggregate, gap = aggregate_rank_decision(frame, minimum_count=30)
    assert aggregate["stratum"].tolist() == ["high"]
    assert gap < 0.01


def test_no_stage0_or_stage1_production_write() -> None:
    source = inspect.getsource(run_audit)
    assert "Stage0OrderProcessor" not in source
    assert "build_stage1" not in source
    assert "stage2/docs/v5_2/upstream_sampling_audit" in source
    assert "_production_snapshot" in source
    evidence_source = inspect.getsource(refresh_evidence)
    assert "Stage0OrderProcessor" not in evidence_source
    assert "_build_evidence" in evidence_source


def test_stress_test_authorization_remains_false(tmp_path: Path) -> None:
    payload = {
        "audit_scope": "DESCRIPTIVE_REPRESENTATIVENESS_AUDIT_ONLY",
        "authorizations": {
            "stage0_production": False, "stage1_production": False,
            "model_training": False, "model_inference": False,
            "tau_selection": False, "sparsity_stress_test": False,
            "transfer_v2": False, "phase_d": False, "stage3": False,
        },
        "time_bin": {
            "timezone": "Asia/Shanghai", "minutes": 30, "count": 48,
            "definition": "local_hour * 2 + (local_minute >= 30)",
        },
        "spatial_grid": {"cell_size_m": 1000},
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert not any(_load_config(path)["authorizations"].values())
    payload["authorizations"]["sparsity_stress_test"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(Stage2V52ContractError):
        _load_config(path)
