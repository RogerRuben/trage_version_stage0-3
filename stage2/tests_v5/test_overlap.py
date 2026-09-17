from __future__ import annotations

import numpy as np

from stage2.v5.chunks import chunk_supervision_arrays


def _physical_weight(length: int, max_seq_len: int, overlap: int) -> np.ndarray:
    total = np.zeros(length)
    for chunk in chunk_supervision_arrays(length, max_seq_len=max_seq_len, overlap=overlap):
        start = int(chunk["chunk_start"])
        local = chunk["supervision_weight"]
        usable = local > 0
        total[start : start + usable.sum()] += local[usable]
    return total


def test_overlap_supervision_total_weight_is_one() -> None:
    assert np.allclose(_physical_weight(301, 128, 32), 1.0)


def test_unique_token_weighted_loss_is_overlap_invariant() -> None:
    error = np.linspace(0.0, 1.0, 301)
    totals = []
    for overlap in (0, 16, 32, 64):
        weight = _physical_weight(301, 128, overlap)
        totals.append(np.sum(weight * error) / np.sum(weight))
    assert np.allclose(totals, totals[0], atol=1e-14)

