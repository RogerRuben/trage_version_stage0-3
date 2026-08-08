from __future__ import annotations

import numpy as np
import pytest

from stage2.v5_2.support_transfer import (
    SupportAwareEdgeRepresentation,
    fit_train_support,
    lookup_train_support,
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
    scores = {float(value): float(index + 1) for index, value in enumerate(candidates)}
    selection = select_tau_once(scores, artifact)
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
    import pandas as pd

    train = pd.DataFrame({
        "order_id": ["a"], "route_sequence": [0], "canonical_highway": ["primary"],
        "road_class": ["major"], "observed_direction": ["forward"], "bridge": [False],
        "tunnel": [False], "synthetic_reverse_edge": [False], "osm_direction_disagreement": [False],
    })
    evaluation = train.copy()
    evaluation["canonical_highway"] = "unseen_class"
    artifact = fit_static_structure_artifact([train], fit_dates=["20161009"])
    features, names = build_static_structure_features(evaluation, artifact)
    unseen_index = names.index("canonical_highway=__UNSEEN_OR_MISSING__")
    assert features[0, unseen_index] == 1.0
    assert artifact.to_payload()["evaluation_rows_used"] == 0
