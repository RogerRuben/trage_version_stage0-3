"""Overlap-aware supervision utilities.

Every physical route token contributes total weight one even when it appears in
several overlapping chunks.  The functions operate on integer token identity
codes and never scan the full input once per token.
"""

from __future__ import annotations

import numpy as np

from .contracts import Stage2V5ContractError


def overlap_supervision(
    order_id: np.ndarray,
    traversal_id: np.ndarray,
    valid: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.asarray(order_id).reshape(-1)
    traversal = np.asarray(traversal_id).reshape(-1)
    if order.shape != traversal.shape:
        raise Stage2V5ContractError("order and traversal identity shapes differ")
    usable = traversal >= 0 if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    if usable.shape != order.shape:
        raise Stage2V5ContractError("supervision validity shape differs")
    count = np.zeros(len(order), dtype=np.int32)
    weight = np.zeros(len(order), dtype=np.float64)
    if not usable.any():
        return count.reshape(np.shape(order_id)), weight.reshape(np.shape(order_id))
    identity = np.rec.fromarrays(
        [order[usable].astype(str), traversal[usable].astype(np.int64)],
        names=("order_id", "traversal_id"),
    )
    _, inverse = np.unique(identity, return_inverse=True)
    frequencies = np.bincount(inverse)
    local_count = frequencies[inverse].astype(np.int32)
    count[usable] = local_count
    weight[usable] = 1.0 / local_count
    return count.reshape(np.shape(order_id)), weight.reshape(np.shape(order_id))


def assert_unit_physical_weight(
    order_id: np.ndarray,
    traversal_id: np.ndarray,
    supervision_weight: np.ndarray,
    valid: np.ndarray | None = None,
    *,
    atol: float = 1e-12,
) -> None:
    order = np.asarray(order_id).reshape(-1)
    traversal = np.asarray(traversal_id).reshape(-1)
    weight = np.asarray(supervision_weight, dtype=np.float64).reshape(-1)
    usable = traversal >= 0 if valid is None else np.asarray(valid, dtype=bool).reshape(-1)
    identity = np.rec.fromarrays(
        [order[usable].astype(str), traversal[usable].astype(np.int64)],
        names=("order_id", "traversal_id"),
    )
    _, inverse = np.unique(identity, return_inverse=True)
    totals = np.bincount(inverse, weights=weight[usable])
    if not np.allclose(totals, 1.0, rtol=0.0, atol=atol):
        raise Stage2V5ContractError("overlap supervision does not sum to one")
