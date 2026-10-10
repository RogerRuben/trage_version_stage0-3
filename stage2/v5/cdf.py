"""Indexed empirical-CDF lookup with ordered fallback levels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from .contracts import Stage2V5ContractError


@dataclass(frozen=True)
class EmpiricalCDFIndex:
    levels: tuple[str, ...]
    samples: Mapping[str, Mapping[str, np.ndarray]]
    supports: Mapping[str, Mapping[str, int]]


def _percentile(sorted_sample: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.searchsorted(sorted_sample, values, side="right") / float(len(sorted_sample))


def map_empirical_cdf(
    values: np.ndarray,
    cohort_keys: Mapping[str, np.ndarray],
    index: EmpiricalCDFIndex,
    *,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw = np.asarray(values, dtype=np.float64)
    result = np.full(raw.shape, np.nan, dtype=np.float64)
    level_used = np.full(raw.shape, "unresolved", dtype=object)
    support_used = np.zeros(raw.shape, dtype=np.int64)
    unresolved = np.isfinite(raw)
    for level in index.levels:  # Fixed fallback levels, not a row/group loop.
        positions = np.flatnonzero(unresolved)
        if not len(positions):
            break
        if level not in cohort_keys:
            raise Stage2V5ContractError(f"missing CDF cohort keys for {level}")
        keys = np.asarray(cohort_keys[level]).astype(str)[positions]
        codes, unique = pd.factorize(keys, sort=False)
        order = np.argsort(codes, kind="stable")
        sizes = np.bincount(codes, minlength=len(unique))
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        level_samples = index.samples.get(level, {})
        level_support = index.supports.get(level, {})
        required = 1 if level == "global" else int(minimum_support)
        for code, key in enumerate(unique):  # O(K), over contiguous positions only.
            support = int(level_support.get(str(key), 0))
            sample = np.asarray(level_samples.get(str(key), ()), dtype=np.float64)
            if support < required or not len(sample):
                continue
            local = order[offsets[code] : offsets[code + 1]]
            target = positions[local]
            result[target] = _percentile(sample, raw[target])
            level_used[target] = level
            support_used[target] = support
            unresolved[target] = False
    return result, level_used, support_used


def map_empirical_cdf_reference(
    values: np.ndarray,
    cohort_keys: Mapping[str, np.ndarray],
    index: EmpiricalCDFIndex,
    *,
    minimum_support: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Clear row-wise reference retained for correctness tests only."""
    raw = np.asarray(values, dtype=np.float64)
    result = np.full(raw.shape, np.nan)
    level_used = np.full(raw.shape, "unresolved", dtype=object)
    support_used = np.zeros(raw.shape, dtype=np.int64)
    for row, value in enumerate(raw):
        if not np.isfinite(value):
            continue
        for level in index.levels:
            key = str(np.asarray(cohort_keys[level]).astype(str)[row])
            support = int(index.supports.get(level, {}).get(key, 0))
            required = 1 if level == "global" else minimum_support
            sample = np.asarray(index.samples.get(level, {}).get(key, ()), dtype=float)
            if support >= required and len(sample):
                result[row] = _percentile(sample, np.asarray([value]))[0]
                level_used[row] = level
                support_used[row] = support
                break
    return result, level_used, support_used

