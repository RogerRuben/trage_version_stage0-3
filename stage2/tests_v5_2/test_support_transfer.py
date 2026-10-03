from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stage2.v5_2.support_transfer import (
    SupportAwareEdgeRepresentation,
    fit_train_support,
    lookup_train_support,
    _payload_hash,
    select_tau_once,
    support_gate,
)
from stage2.v5_2.structure_features import (
    build_static_structure_features,
    fit_static_structure_artifact,
)


def test_support_gate_is_zero_for_unseen_and_monotonic() -> None:
    values = support_gate(np.array([0, 1, 10, 100]), tau=10.0)
    assert values[0] == 0.0
    assert np.all(np.diff(values) > 0)


def test_support_lookup_never_uses_evaluation_counts() -> None:
    artifact = fit_train_support(["a", "a", "b", "c", "c", "c"], fit_dates=["20161009"]).to_payload()
    support, groups = lookup_train_support(["a", "unseen", "c"], artifact)
    assert support.tolist() == [2, 0, 3]
    assert groups[1] == "unseen"
    assert artifact["evaluation_support_used"] is False


def test_tau_selection_is_limited_to_train_quantile_candidates() -> None:
    artifact = fit_train_support(
        ["a", "b", "b", "c", "c", "c", "d", "d", "d", "d"],
        fit_dates=["20161009"],
    ).to_payload()
    candidates = artifact["tau_candidates"]
    targets = ("crawl", "stop", "speed_cv", "acceleration_rms")
    scores = {label: {target: float(index + 1) for target in targets} for index, label in enumerate(("p25", "p50", "p75"))}
    metrics = {
        "schema_version": "stage2_v5_2_tau_evaluation.2", "status": "PASS",
        "protocol_id": "transfer_tuning", "protocol_hash": "p" * 64,
        "train_dates": [f"201610{day:02d}" for day in range(9, 19)],
        "validation_dates": ["20161019", "20161020"],
        "support_artifact_embedded_sha256": artifact["artifact_sha256"],
        "support_artifact_sha256": "s" * 64, "feature_artifact_sha256": "f" * 64,
        "m1_source_checkpoint_sha256": "a" * 64, "m1_checkpoint_sha256": "b" * 64,
        "m1_evaluation_manifest_sha256": "c" * 64, "evaluation_code_sha256": "d" * 64,
        "evaluation_schema": "fixture", "m1_core_mae": {target: 1.0 for target in targets},
        "m4_candidates": {
            label: {"support_tau_candidate": label, "support_tau_value": artifact["positive_quantiles"][label], "core_mae": score}
            for label, score in scores.items()
        },
    }
    metrics["artifact_sha256"] = _payload_hash(metrics)
    selection = select_tau_once(metrics, artifact)
    assert selection["selected_tau"] in candidates
    assert selection["rolling_reselection_allowed"] is False


def test_structure_branch_computes_unseen_edge_representation() -> None:
    torch = pytest.importorskip("torch")
    module = SupportAwareEdgeRepresentation(
        edge_vocabulary_size=8,
        static_feature_count=3,
        embedding_dim=4,
        tau=5.0,
        mode="support_aware",
    )
    edge = torch.tensor([1])
    static = torch.tensor([[1.0, 0.0, 2.0]])
    support = torch.tensor([0.0])
    actual = module(edge, static, support)
    expected = module.structure_encoder(static)
    assert torch.allclose(actual, expected)


def test_static_structure_encoder_uses_train_vocabulary_for_unseen_category() -> None:
    train = pd.DataFrame({
        "split": ["train"], "date": ["20161009"], "row_id": [10],
        "order_id": ["a"], "route_sequence": [0], "canonical_highway": ["primary"],
        "road_class": ["major"], "observed_direction": ["forward"], "bridge": [False],
        "tunnel": [False], "synthetic_reverse_edge": [False], "osm_direction_disagreement": [False],
    })
    evaluation = train.copy()
    evaluation["canonical_highway"] = "unseen_class"
    artifact = fit_static_structure_artifact(
        [train], protocol_id="fixture", protocol_train_dates=["20161009"]
    )
    features, names, row_id = build_static_structure_features(evaluation, artifact)
    unseen_index = names.index("canonical_highway=__UNSEEN_OR_MISSING__")
    assert features[0, unseen_index] == 1.0
    assert row_id.tolist() == [10]
    assert artifact.to_payload()["evaluation_rows_used"] == 0
