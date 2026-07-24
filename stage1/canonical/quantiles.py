"""Partition-invariant, mergeable distribution summaries for Stage 1 v2.

The v1 implementation averaged partition medians, which is not a median of the
union and changes when files are repartitioned.  V2 uses a fixed-bin histogram:
each observation contributes to exactly one globally defined bin, summaries are
merged by integer addition, and therefore the fitted quantile is invariant to
partition count and processing order (up to the declared bin resolution).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class MergeableHistogram:
    """A deterministic mergeable quantile sketch with an explicit resolution."""

    edges: np.ndarray
    counts: np.ndarray

    @classmethod
    def empty(cls, edges: np.ndarray) -> "MergeableHistogram":
        clean_edges = np.asarray(edges, dtype=np.float64)
        if clean_edges.ndim != 1 or len(clean_edges) < 2:
            raise ValueError("edges must be a one-dimensional array of length >= 2")
        if not np.all(np.diff(clean_edges) > 0):
            raise ValueError("edges must be strictly increasing")
        return cls(clean_edges, np.zeros(len(clean_edges) - 1, dtype=np.int64))

    @property
    def sample_size(self) -> int:
        return int(self.counts.sum())

    @property
    def centers(self) -> np.ndarray:
        return (self.edges[:-1] + self.edges[1:]) / 2.0

    def update(self, values: np.ndarray) -> None:
        clean = np.asarray(values, dtype=np.float64)
        clean = clean[np.isfinite(clean)]
        if clean.size == 0:
            return
        # Histogram semantics include the rightmost edge in the last bin.
        addition, _ = np.histogram(clean, bins=self.edges)
        self.counts += addition.astype(np.int64, copy=False)

    def merge(self, other: "MergeableHistogram") -> None:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("cannot merge histograms with different global edges")
        self.counts += other.counts

    def quantile(self, probability: float) -> float:
        if not 0 <= probability <= 1:
            raise ValueError("probability must be in [0, 1]")
        if self.sample_size == 0:
            return float("nan")
        target = probability * max(self.sample_size - 1, 0)
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, target + 1, side="left"))
        index = min(index, len(self.counts) - 1)
        before = int(cumulative[index - 1]) if index else 0
        count = int(self.counts[index])
        if count <= 0:
            return float(self.centers[index])
        fraction = np.clip((target - before + 0.5) / count, 0.0, 1.0)
        return float(self.edges[index] + fraction * (self.edges[index + 1] - self.edges[index]))

    def to_records(self) -> dict[str, list[float] | list[int]]:
        return {"edges": self.edges.tolist(), "counts": self.counts.tolist()}

    @classmethod
    def from_records(cls, records: dict) -> "MergeableHistogram":
        result = cls.empty(np.asarray(records["edges"], dtype=np.float64))
        counts = np.asarray(records["counts"], dtype=np.int64)
        if counts.shape != result.counts.shape:
            raise ValueError("serialized counts do not match edges")
        result.counts[:] = counts
        return result


def empirical_cdf_interpolated(
    values: np.ndarray,
    support_values: np.ndarray,
    support_counts: np.ndarray,
) -> np.ndarray:
    """Evaluate a monotone empirical CDF with interpolation across empty bins.

    V1 assigned 0.5 whenever an exact histogram bin was absent.  That creates a
    discontinuity and maps both tails to the median.  V2 interpolates between
    observed ordered support points, returns 0 below support and 1 above support,
    and preserves missing inputs as missing.
    """

    query = np.asarray(values, dtype=np.float64)
    support = np.asarray(support_values, dtype=np.float64)
    counts = np.asarray(support_counts, dtype=np.float64)
    valid_support = np.isfinite(support) & np.isfinite(counts) & (counts > 0)
    support = support[valid_support]
    counts = counts[valid_support]
    if support.size == 0:
        return np.full(query.shape, np.nan, dtype=np.float64)
    order = np.argsort(support, kind="mergesort")
    support = support[order]
    counts = counts[order]
    unique, inverse = np.unique(support, return_inverse=True)
    merged_counts = np.zeros(len(unique), dtype=np.float64)
    np.add.at(merged_counts, inverse, counts)
    total = merged_counts.sum()
    midrank = (np.cumsum(merged_counts) - merged_counts / 2.0) / total
    result = np.interp(query, unique, midrank, left=0.0, right=1.0)
    result[~np.isfinite(query)] = np.nan
    return result
