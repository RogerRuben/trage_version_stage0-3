"""Continuous chunk definitions with overlap-normalized supervision."""

from __future__ import annotations

import numpy as np

from .contracts import Stage2V5ContractError


def continuous_chunk_starts(length: int, *, max_seq_len: int, overlap: int) -> tuple[int, ...]:
    if length <= 0:
        return ()
    if not 0 <= overlap < max_seq_len:
        raise Stage2V5ContractError("chunk overlap must be in [0, max_seq_len)")
    if length <= max_seq_len:
        return (0,)
    stride = max_seq_len - overlap
    starts = list(range(0, length - max_seq_len + 1, stride))
    last = length - max_seq_len
    if starts[-1] != last:
        starts.append(last)
    return tuple(starts)


def chunk_supervision_arrays(length: int, *, max_seq_len: int, overlap: int) -> list[dict[str, np.ndarray]]:
    starts = continuous_chunk_starts(length, max_seq_len=max_seq_len, overlap=overlap)
    count = np.zeros(length, dtype=np.int32)
    for start in starts:
        count[start : min(start + max_seq_len, length)] += 1
    result = []
    for start in starts:
        end = min(start + max_seq_len, length)
        valid_length = end - start
        local_count = np.zeros(max_seq_len, dtype=np.int32)
        local_weight = np.zeros(max_seq_len, dtype=np.float64)
        local_count[:valid_length] = count[start:end]
        local_weight[:valid_length] = 1.0 / count[start:end]
        result.append(
            {
                "chunk_start": np.asarray(start, dtype=np.int64),
                "overlap_supervision_count": local_count,
                "supervision_weight": local_weight,
            }
        )
    return result
