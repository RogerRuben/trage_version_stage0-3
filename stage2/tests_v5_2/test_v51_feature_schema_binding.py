from __future__ import annotations

import json

import pytest

from stage2.v5_2.contracts import Stage2V52ContractError
from stage2.v5_2.feature_binding import bind_v51_feature_schema


def _artifact(tmp_path, names, *, unseen_index=1):
    vocabularies = {}
    for name in names:
        vocabularies[name] = {"token_to_index": {
            "__PAD__": 0, "__UNSEEN__": unseen_index, "__RARE__": 2, "__MISSING__": 3,
        }}
    path = tmp_path / "features.json"
    path.write_text(json.dumps({"vocabularies": vocabularies}), encoding="utf-8")
    return path


def _state(torch, count):
    return {f"embeddings.{index}.weight": torch.zeros(4, 3) for index in range(count)}


def test_wrong_edge_categorical_slot_fails(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    names = ("highway", "edge", "time_bin", "position_bucket", "route_length_bucket")
    with pytest.raises(Stage2V52ContractError, match="categorical order"):
        bind_v51_feature_schema(_artifact(tmp_path, names), checkpoint_state=_state(torch, 5))


def test_wrong_unseen_index_fails(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    names = ("edge", "highway", "time_bin", "position_bucket", "route_length_bucket")
    with pytest.raises(Stage2V52ContractError, match="reserved-token indices"):
        bind_v51_feature_schema(
            _artifact(tmp_path, names, unseen_index=4), checkpoint_state=_state(torch, 5)
        )
