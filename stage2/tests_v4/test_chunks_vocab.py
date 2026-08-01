from __future__ import annotations

import pandas as pd

from stage2.v4.models.datasets import (
    MISSING_TOKEN,
    RARE_TOKEN,
    UNSEEN_TOKEN,
    _categorical_values,
    continuous_chunk_starts,
    encode_categorical,
)


def test_continuous_chunks_never_create_false_adjacency() -> None:
    starts = continuous_chunk_starts(301, max_seq_len=128, overlap=32)
    assert starts == (0, 96, 173)
    intervals = [(start, min(start + 128, 301)) for start in starts]
    assert all(right_start < left_end for (_, left_end), (right_start, _) in zip(intervals, intervals[1:]))
    covered = set()
    for start, end in intervals:
        covered.update(range(start, end))
    assert covered == set(range(301))


def test_unseen_and_train_rare_tokens_remain_distinct() -> None:
    vocabulary = {
        "token_to_index": {
            "__PAD__": 0,
            UNSEEN_TOKEN: 1,
            RARE_TOKEN: 2,
            MISSING_TOKEN: 3,
            "frequent": 4,
        },
        "seen_tokens": [MISSING_TOKEN, "frequent", "rare"],
    }
    encoded = encode_categorical(
        pd.Series(["frequent", "rare", "new", None]),
        vocabulary,
    )
    assert encoded.tolist() == [4, 2, 1, 3]


def test_categorical_values_normalize_mixed_missing_types_to_strings() -> None:
    frame = pd.DataFrame(
        {
            "observed_directed_edge_uid": ["edge", float("nan")],
            "canonical_highway": ["primary", None],
            "estimated_time_bin": [1, None],
            "route_position_ratio": [0.1, None],
            "route_token_count": [15, None],
        }
    )
    categories = _categorical_values(frame)
    assert all(
        all(isinstance(value, str) for value in values.tolist())
        for values in categories.values()
    )
    assert categories["edge"].tolist() == ["edge", MISSING_TOKEN]
    assert categories["highway"].tolist() == ["primary", MISSING_TOKEN]
