"""Deterministic, mergeable histograms used by Stage 1 label schema v3.

The implementation deliberately evaluates the empirical CDF by histogram bin,
not by comparing a raw query with bin centres.  This distinction is important
for bounded labels: a supported value at exactly 0 or 1 receives the empirical
midrank of its bin instead of being treated as an unsupported tail value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FixedBinHistogram:
    """A fixed-edge histogram whose integer state can be merged losslessly."""

    edges: np.ndarray
    counts: np.ndarray
    invalid_count: int = 0
    underflow_count: int = 0
    overflow_count: int = 0

    @classmethod
    def empty(cls, edges: np.ndarray) -> "FixedBinHistogram":
        clean_edges = np.asarray(edges, dtype=np.float64)
        if clean_edges.ndim != 1 or clean_edges.size < 2:
            raise ValueError("histogram edges must be one-dimensional with length >= 2")
        if not np.isfinite(clean_edges).all():
            raise ValueError("histogram edges must be finite")
        if not np.all(np.diff(clean_edges) > 0):
            raise ValueError("histogram edges must be strictly increasing")
        return cls(
            edges=clean_edges,
            counts=np.zeros(clean_edges.size - 1, dtype=np.int64),
        )

    @property
    def sample_size(self) -> int:
        return int(self.counts.sum())

    @property
    def occupied_centres(self) -> np.ndarray:
        occupied = self.counts > 0
        centres = (self.edges[:-1] + self.edges[1:]) / 2.0
        return centres[occupied]

    def update(self, values: np.ndarray, *, clip: bool = False) -> None:
        """Add observations and record invalid/tail counts explicitly.

        With ``clip=True`` finite tail observations are assigned to the first or
        last bin.  With ``clip=False`` they are counted as under/overflow but do
        not silently enter the fitted distribution.
        """

        raw = np.asarray(values, dtype=np.float64).reshape(-1)
        finite = np.isfinite(raw)
        self.invalid_count += int((~finite).sum())
        clean = raw[finite]
        if clean.size == 0:
            return

        below = clean < self.edges[0]
        above = clean > self.edges[-1]
        self.underflow_count += int(below.sum())
        self.overflow_count += int(above.sum())
        if clip:
            clean = np.clip(clean, self.edges[0], self.edges[-1])
        else:
            clean = clean[~below & ~above]
        if clean.size == 0:
            return

        addition, _ = np.histogram(clean, bins=self.edges)
        self.counts += addition.astype(np.int64, copy=False)

    def merge(self, other: "FixedBinHistogram") -> None:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("cannot merge histograms with different edges")
        self.counts += other.counts
        self.invalid_count += int(other.invalid_count)
        self.underflow_count += int(other.underflow_count)
        self.overflow_count += int(other.overflow_count)

    def quantile(self, probability: float) -> float:
        """Return a within-bin interpolated quantile."""

        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be in [0, 1]")
        if self.sample_size == 0:
            return float("nan")
        target = probability * max(self.sample_size - 1, 0)
        cumulative = np.cumsum(self.counts)
        index = int(np.searchsorted(cumulative, target + 1.0, side="left"))
        index = min(index, self.counts.size - 1)
        before = int(cumulative[index - 1]) if index else 0
        count = int(self.counts[index])
        if count <= 0:
            return float((self.edges[index] + self.edges[index + 1]) / 2.0)
        fraction = float(np.clip((target - before + 0.5) / count, 0.0, 1.0))
        return float(
            self.edges[index]
            + fraction * (self.edges[index + 1] - self.edges[index])
        )

    def cdf(self, values: np.ndarray) -> np.ndarray:
        """Evaluate empirical midranks with interpolation across empty bins."""

        query = np.asarray(values, dtype=np.float64)
        result = np.full(query.shape, np.nan, dtype=np.float64)
        if self.sample_size == 0:
            return result

        occupied = self.counts > 0
        cumulative = np.cumsum(self.counts, dtype=np.float64)
        midranks = (cumulative - self.counts / 2.0) / float(self.sample_size)
        centres = (self.edges[:-1] + self.edges[1:]) / 2.0
        support_x = centres[occupied]
        support_y = midranks[occupied]

        flat_query = query.reshape(-1)
        flat_result = result.reshape(-1)
        finite_positions = np.flatnonzero(np.isfinite(flat_query))
        if finite_positions.size == 0:
            return result
        finite_values = flat_query[finite_positions]

        indices = np.searchsorted(self.edges, finite_values, side="right") - 1
        indices[finite_values == self.edges[-1]] = self.counts.size - 1
        in_range = (indices >= 0) & (indices < self.counts.size)
        supported = in_range.copy()
        supported[in_range] &= self.counts[indices[in_range]] > 0

        if supported.any():
            flat_result[finite_positions[supported]] = midranks[indices[supported]]

        unresolved = ~supported
        if unresolved.any():
            unresolved_values = finite_values[unresolved]
            interpolated = np.interp(
                unresolved_values,
                support_x,
                support_y,
                left=0.0,
                right=1.0,
            )
            flat_result[finite_positions[unresolved]] = interpolated
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "edges": self.edges.tolist(),
            "counts": self.counts.tolist(),
            "invalid_count": int(self.invalid_count),
            "underflow_count": int(self.underflow_count),
            "overflow_count": int(self.overflow_count),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FixedBinHistogram":
        histogram = cls.empty(np.asarray(payload["edges"], dtype=np.float64))
        raw_counts = np.asarray(payload["counts"])
        if (
            raw_counts.ndim != 1
            or not np.issubdtype(raw_counts.dtype, np.integer)
        ):
            raise ValueError("serialized histogram counts must be integers")
        counts = raw_counts.astype(np.int64, copy=False)
        if counts.shape != histogram.counts.shape:
            raise ValueError("serialized histogram counts do not match edges")
        if (counts < 0).any():
            raise ValueError("serialized histogram counts must be non-negative")
        histogram.counts[:] = counts
        for name in ("invalid_count", "underflow_count", "overflow_count"):
            value = payload.get(name, 0)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(
                    f"serialized histogram {name} must be a non-negative integer"
                )
            setattr(histogram, name, value)
        return histogram


def empirical_cdf_from_histogram(
    values: np.ndarray,
    histogram: FixedBinHistogram,
) -> np.ndarray:
    """Convenience wrapper used by the transform and unit-test surfaces."""

    return histogram.cdf(values)
