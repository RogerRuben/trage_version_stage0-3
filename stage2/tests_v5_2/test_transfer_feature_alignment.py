from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5_2.structure_features import (
    build_static_structure_features,
    fit_static_structure_artifact,
)
from stage2.v5_2.transfer_data import _payload_source_positions


def _frame() -> pd.DataFrame:
    return pd.DataFrame({
        "split": ["train", "train", "train"],
        "date": ["20161009"] * 3,
        "order_id": ["b", "a", "a"],
        "route_sequence": [0, 1, 0],
        "row_id": [30, 20, 10],
        "canonical_highway": ["secondary", "primary", "primary"],
        "road_class": ["minor", "major", "major"],
        "observed_direction": ["forward"] * 3,
        "bridge": [False] * 3,
        "tunnel": [False] * 3,
        "synthetic_reverse_edge": [False] * 3,
        "osm_direction_disagreement": [False] * 3,
    })


def test_internal_route_sort_scattered_back_to_explicit_row_id_order() -> None:
    frame = _frame()
    artifact = fit_static_structure_artifact(
        [frame], protocol_id="fixture", protocol_train_dates=("20161009",)
    )
    features, names, row_id = build_static_structure_features(frame, artifact)
    major = names.index("road_class=major")
    minor = names.index("road_class=minor")
    assert row_id.tolist() == [30, 20, 10]
    assert features[:, major].tolist() == [0.0, 1.0, 1.0]
    assert features[:, minor].tolist() == [1.0, 0.0, 0.0]


def test_payload_identity_keeps_full_split_and_date_strings() -> None:
    frame = pd.DataFrame({
        "split": ["train", "train"], "date": ["20161009", "20161009"],
        "order_id": ["order-a", "order-a"], "traversal_id": [0, 1],
    })
    payload = {
        "order_id": np.asarray(["order-a"]),
        "traversal_id": np.asarray([[0, 1, -1]], dtype=np.int64),
        "pad_mask": np.asarray([[False, False, True]]),
    }
    positions = _payload_source_positions(payload, frame)
    assert positions.tolist() == [[0, 1, -1]]
