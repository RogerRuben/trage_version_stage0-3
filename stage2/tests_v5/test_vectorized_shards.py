from __future__ import annotations

import numpy as np
import pandas as pd

from stage2.v5.data import PROFILE_PACE_COLUMNS, RECENT_PACE_COLUMNS
from stage2.v5.shards import CONTINUOUS_TARGETS, TARGET_MASKS, vectorized_chunk_payload


def test_vectorized_chunks_preserve_routes_and_unit_supervision() -> None:
    lengths = [3, 7, 11]
    frame = pd.DataFrame(
        {
            "order_id": np.repeat(["a", "b", "c"], lengths),
            "route_sequence": np.concatenate([np.arange(length) for length in lengths]),
            "traversal_id": np.arange(sum(lengths)),
            "route_part_length_m": 10.0,
            "allocated_distance_m": 10.0,
            "route_position_ratio": 0.5,
            "route_token_count": np.repeat(lengths, lengths),
            "observed_directed_edge_uid": "edge",
            "canonical_highway": "primary",
            "estimated_time_bin": 4,
            "feature_age_s": 30.0,
            "forecast_horizon_s": 60.0,
        }
    )
    for column in CONTINUOUS_TARGETS:
        frame[column] = 0.2
    for column in TARGET_MASKS:
        frame[column] = True
    frame["lcs_tail_event"] = False
    frame["rts_tail_event"] = False
    for column in RECENT_PACE_COLUMNS:
        frame[column] = 0.1
    for column in PROFILE_PACE_COLUMNS:
        frame[column] = 0.1
    frame["observed_sec_per_m_profile_count"] = 10
    tokens = {"__PAD__": 0, "__UNSEEN__": 1, "__RARE__": 2, "__MISSING__": 3, "edge": 4, "primary": 5, "4": 6, "10": 7, "2": 8}
    vocabulary = {"token_to_index": tokens, "seen_tokens": list(tokens)}
    artifacts = {
        "numeric_features": ["route_part_length_m"],
        "numeric_mean": [10.0],
        "numeric_std": [1.0],
        "vocabularies": {name: vocabulary for name in ("edge", "highway", "time_bin", "position_bucket", "route_length_bucket")},
    }
    payload = vectorized_chunk_payload(frame, artifacts, max_seq_len=5, overlap=2)
    identity = np.char.add(np.char.add(payload["order_id"][:, None], "|"), payload["traversal_id"].astype(str))
    valid = ~payload["pad_mask"]
    unique, inverse = np.unique(identity[valid], return_inverse=True)
    totals = np.bincount(inverse, weights=payload["supervision_weight"][valid])
    assert len(unique) == sum(lengths)
    assert np.allclose(totals, 1.0)
    assert np.all(payload["overlap_supervision_count"][valid] >= 1)
    assert not payload["tail_masks"].any()

    artifacts["percentile_supervision_allowed"] = True
    legacy_payload = vectorized_chunk_payload(frame, artifacts, max_seq_len=5, overlap=2)
    assert legacy_payload["tail_masks"][~legacy_payload["pad_mask"]].all()
