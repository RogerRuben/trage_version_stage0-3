"""Streaming cohort references and CDF normalization for Stage 1 v3."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable

import numpy as np
import pandas as pd

from .histograms import FixedBinHistogram
from .schema import ContractError

if TYPE_CHECKING:
    from .config import Stage1V3Config


COHORT_LEVELS = (
    "edge_time_weekday",
    "edge_peak",
    "edge",
    "highway_time_weekday",
    "highway",
    "global",
)


def add_cohort_keys(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "observed_directed_edge_uid",
        "canonical_highway",
        "time_bin_30m",
        "weekday_type",
        "peak_offpeak",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractError(f"cannot construct cohort keys; missing columns: {missing}")
    result = frame.copy()
    edge = result["observed_directed_edge_uid"].astype(str)
    highway = result["canonical_highway"].fillna("unknown").astype(str)
    time_bin = result["time_bin_30m"].astype(str)
    weekday = result["weekday_type"].astype(str)
    peak = result["peak_offpeak"].astype(str)
    result["edge_time_weekday"] = edge + "|" + time_bin + "|" + weekday
    result["edge_peak"] = edge + "|" + peak
    result["edge"] = edge
    result["highway_time_weekday"] = highway + "|" + time_bin + "|" + weekday
    result["highway"] = highway
    result["global"] = "global"
    return result


@dataclass
class SparseCohortHistograms:
    """Sparse fixed-bin histograms keyed by deterministic fallback cohorts.

    Each cohort stores only occupied bins.  This avoids allocating a dense
    4096-element array for every edge/time key while preserving exact integer
    merge semantics.
    """

    edges: np.ndarray
    counts: dict[str, dict[str, dict[int, int]]] = field(
        default_factory=lambda: {level: {} for level in COHORT_LEVELS}
    )
    invalid_count: int = 0
    underflow_count: int = 0
    overflow_count: int = 0

    @classmethod
    def empty(cls, edges: np.ndarray) -> "SparseCohortHistograms":
        clean = np.asarray(edges, dtype=np.float64)
        FixedBinHistogram.empty(clean)
        return cls(edges=clean)

    def update(
        self,
        frame: pd.DataFrame,
        value_column: str,
        *,
        clip: bool = False,
    ) -> None:
        if value_column not in frame:
            raise ContractError(f"missing histogram value column: {value_column}")
        if frame.empty:
            return
        keyed = add_cohort_keys(frame)
        raw = pd.to_numeric(keyed[value_column], errors="coerce")
        finite = np.isfinite(raw.to_numpy(dtype=np.float64))
        self.invalid_count += int((~finite).sum())
        if not finite.any():
            return
        keyed = keyed.loc[finite].copy()
        values = raw.loc[finite].to_numpy(dtype=np.float64)
        below = values < self.edges[0]
        above = values > self.edges[-1]
        self.underflow_count += int(below.sum())
        self.overflow_count += int(above.sum())
        if clip:
            values = np.clip(values, self.edges[0], self.edges[-1])
        else:
            keep = ~below & ~above
            keyed = keyed.loc[keep].copy()
            values = values[keep]
        if values.size == 0:
            return

        bin_index = np.searchsorted(self.edges, values, side="right") - 1
        bin_index[values == self.edges[-1]] = len(self.edges) - 2
        keyed["_histogram_bin"] = bin_index.astype(np.int32)
        for level in COHORT_LEVELS:
            grouped = (
                keyed.groupby([level, "_histogram_bin"], sort=False, dropna=False)
                .size()
                .reset_index(name="count")
            )
            level_counts = self.counts.setdefault(level, {})
            for key_value, bin_value, count_value in grouped[
                [level, "_histogram_bin", "count"]
            ].itertuples(index=False, name=None):
                key = str(key_value)
                index = int(bin_value)
                histogram = level_counts.setdefault(key, {})
                histogram[index] = histogram.get(index, 0) + int(count_value)

    def merge(self, other: "SparseCohortHistograms") -> None:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("cannot merge cohort histograms with different edges")
        for level in COHORT_LEVELS:
            target_level = self.counts.setdefault(level, {})
            for key, bins in other.counts.get(level, {}).items():
                target_bins = target_level.setdefault(key, {})
                for index, count in bins.items():
                    target_bins[int(index)] = (
                        target_bins.get(int(index), 0) + int(count)
                    )
        self.invalid_count += int(other.invalid_count)
        self.underflow_count += int(other.underflow_count)
        self.overflow_count += int(other.overflow_count)

    def support(self, level: str, key: str) -> int:
        return int(sum(self.counts.get(level, {}).get(str(key), {}).values()))

    def _dense(self, level: str, key: str) -> FixedBinHistogram:
        histogram = FixedBinHistogram.empty(self.edges)
        for index, count in self.counts.get(level, {}).get(str(key), {}).items():
            histogram.counts[int(index)] = int(count)
        return histogram

    def _sparse_quantile(
        self,
        level: str,
        key: str,
        probability: float,
        *,
        excluded_bin: int | None = None,
    ) -> float:
        bins = self.counts.get(level, {}).get(str(key), {})
        adjusted: list[tuple[int, int]] = []
        for index, count in sorted(bins.items()):
            value = int(count) - (1 if excluded_bin == int(index) else 0)
            if value > 0:
                adjusted.append((int(index), value))
        sample_size = sum(count for _, count in adjusted)
        if sample_size <= 0:
            return float("nan")
        target = probability * max(sample_size - 1, 0)
        cumulative = 0
        for index, count in adjusted:
            if cumulative + count >= target + 1.0:
                fraction = float(
                    np.clip((target - cumulative + 0.5) / count, 0.0, 1.0)
                )
                return float(
                    self.edges[index]
                    + fraction * (self.edges[index + 1] - self.edges[index])
                )
            cumulative += count
        return float("nan")

    def _sparse_cdf(
        self,
        level: str,
        key: str,
        values: np.ndarray,
    ) -> np.ndarray:
        query = np.asarray(values, dtype=np.float64)
        result = np.full(query.shape, np.nan, dtype=np.float64)
        bins = self.counts.get(level, {}).get(str(key), {})
        if not bins:
            return result
        occupied_indices = np.asarray(sorted(bins), dtype=np.int64)
        occupied_counts = np.asarray(
            [bins[int(index)] for index in occupied_indices],
            dtype=np.float64,
        )
        sample_size = float(occupied_counts.sum())
        cumulative = np.cumsum(occupied_counts)
        midranks = (cumulative - occupied_counts / 2.0) / sample_size
        centres = (
            self.edges[occupied_indices] + self.edges[occupied_indices + 1]
        ) / 2.0
        finite = np.isfinite(query)
        if not finite.any():
            return result
        clean = query[finite]
        query_bins = np.searchsorted(self.edges, clean, side="right") - 1
        query_bins[clean == self.edges[-1]] = len(self.edges) - 2
        midrank_lookup = {
            int(index): float(midrank)
            for index, midrank in zip(occupied_indices, midranks)
        }
        clean_result = np.interp(
            clean,
            centres,
            midranks,
            left=0.0,
            right=1.0,
        )
        for position, index in enumerate(query_bins):
            if int(index) in midrank_lookup:
                clean_result[position] = midrank_lookup[int(index)]
        result[finite] = clean_result
        return result

    def choose_quantile(
        self,
        frame: pd.DataFrame,
        *,
        probability: float,
        minimum_support: int,
        leave_one_out_value_column: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keyed = add_cohort_keys(frame)
        values = np.full(len(keyed), np.nan, dtype=np.float64)
        levels = np.full(len(keyed), "unresolved", dtype=object)
        supports = np.zeros(len(keyed), dtype=np.int64)
        unresolved = np.ones(len(keyed), dtype=bool)
        excluded_values: np.ndarray | None = None
        if leave_one_out_value_column is not None:
            if leave_one_out_value_column not in keyed:
                raise ContractError(
                    "leave-one-out reference column is missing: "
                    f"{leave_one_out_value_column}"
                )
            excluded_values = pd.to_numeric(
                keyed[leave_one_out_value_column], errors="coerce"
            ).to_numpy(dtype=np.float64)
        quantile_cache: dict[tuple[str, str, int | None], float] = {}
        for level in COHORT_LEVELS:
            if not unresolved.any():
                break
            positions = np.flatnonzero(unresolved)
            keys = keyed.iloc[positions][level].astype(str).to_numpy()
            required_support = 1 if level == "global" else minimum_support
            for position, key_value in zip(positions, keys):
                key = str(key_value)
                support = self.support(level, key)
                excluded_bin: int | None = None
                if excluded_values is not None:
                    excluded_value = float(excluded_values[position])
                    if not np.isfinite(excluded_value):
                        continue
                    clipped = float(
                        np.clip(excluded_value, self.edges[0], self.edges[-1])
                    )
                    excluded_bin = int(
                        np.searchsorted(self.edges, clipped, side="right") - 1
                    )
                    if clipped == self.edges[-1]:
                        excluded_bin = len(self.edges) - 2
                    bins = self.counts.get(level, {}).get(key, {})
                    if bins.get(excluded_bin, 0) <= 0:
                        raise ContractError(
                            "leave-one-out row is absent from its fitted cohort"
                        )
                    support -= 1
                if support < required_support:
                    continue
                cache_key = (level, key, excluded_bin)
                if cache_key not in quantile_cache:
                    quantile_cache[cache_key] = self._sparse_quantile(
                        level,
                        key,
                        probability,
                        excluded_bin=excluded_bin,
                    )
                values[position] = quantile_cache[cache_key]
                levels[position] = level
                supports[position] = support
                unresolved[position] = False
        return values, levels, supports

    def choose_cdf(
        self,
        frame: pd.DataFrame,
        value_column: str,
        *,
        minimum_support: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keyed = add_cohort_keys(frame)
        raw = pd.to_numeric(keyed[value_column], errors="coerce").to_numpy(
            dtype=np.float64
        )
        values = np.full(len(keyed), np.nan, dtype=np.float64)
        levels = np.full(len(keyed), "unresolved", dtype=object)
        supports = np.zeros(len(keyed), dtype=np.int64)
        unresolved = np.isfinite(raw)
        for level in COHORT_LEVELS:
            if not unresolved.any():
                break
            positions = np.flatnonzero(unresolved)
            keys = keyed.iloc[positions][level].astype(str).to_numpy()
            for key in np.unique(keys):
                key_positions = positions[keys == key]
                support = self.support(level, key)
                required_support = 1 if level == "global" else minimum_support
                if support < required_support:
                    continue
                values[key_positions] = self._sparse_cdf(
                    level,
                    key,
                    raw[key_positions],
                )
                levels[key_positions] = level
                supports[key_positions] = support
                unresolved[key_positions] = False
        return values, levels, supports

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for level in COHORT_LEVELS:
            for key in sorted(self.counts.get(level, {})):
                bins = self.counts[level][key]
                support = int(sum(bins.values()))
                for index in sorted(bins):
                    rows.append(
                        {
                            "level": level,
                            "key": key,
                            "bin_index": int(index),
                            "count": int(bins[index]),
                            "support_count": support,
                        }
                    )
        return pd.DataFrame(
            rows,
            columns=["level", "key", "bin_index", "count", "support_count"],
        )

    @classmethod
    def from_frame(
        cls,
        edges: np.ndarray,
        frame: pd.DataFrame,
        *,
        invalid_count: int = 0,
        underflow_count: int = 0,
        overflow_count: int = 0,
    ) -> "SparseCohortHistograms":
        required = {"level", "key", "bin_index", "count", "support_count"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ContractError(f"histogram model missing columns: {missing}")
        if frame[list(required)].isna().any().any():
            raise ContractError("histogram model contains null required values")
        if frame.duplicated(["level", "key", "bin_index"], keep=False).any():
            raise ContractError("histogram model contains duplicate bin records")
        for column, minimum in (
            ("bin_index", 0),
            ("count", 1),
            ("support_count", 1),
        ):
            numeric = pd.to_numeric(frame[column], errors="coerce")
            if (
                (~np.isfinite(numeric))
                | numeric.lt(minimum)
                | numeric.ne(np.floor(numeric))
            ).any():
                raise ContractError(
                    f"histogram model has invalid integer column {column}"
                )
        result = cls.empty(edges)
        result.invalid_count = int(invalid_count)
        result.underflow_count = int(underflow_count)
        result.overflow_count = int(overflow_count)
        for row in frame.itertuples(index=False):
            level = str(row.level)
            if level not in COHORT_LEVELS:
                raise ContractError(f"unknown cohort level in model: {level}")
            key = str(row.key)
            index = int(row.bin_index)
            count = int(row.count)
            if not 0 <= index < len(result.edges) - 1 or count <= 0:
                raise ContractError("invalid histogram bin record")
            bins = result.counts[level].setdefault(key, {})
            bins[index] = count
        if len(frame):
            reported = (
                frame.groupby(["level", "key"], sort=False)["support_count"]
                .nunique(dropna=False)
            )
            if reported.ne(1).any():
                raise ContractError(
                    "histogram support_count is inconsistent within a cohort"
                )
            actual = (
                frame.groupby(["level", "key"], sort=False)["count"].sum()
            )
            declared = (
                frame.groupby(["level", "key"], sort=False)[
                    "support_count"
                ].first()
            )
            if not pd.to_numeric(declared, errors="coerce").eq(actual).all():
                raise ContractError(
                    "histogram support_count does not equal the bin total"
                )
        return result


def fit_reference_histograms(
    primitive_batches: Iterable[pd.DataFrame],
    config: "Stage1V3Config",
) -> SparseCohortHistograms:
    settings = config.section("reference")
    minimum = float(settings["histogram_min_sec_per_m"])
    maximum = float(settings["histogram_max_sec_per_m"])
    bins = int(settings["histogram_bins"])
    if not 0 < minimum < maximum or bins < 2:
        raise ContractError("invalid reference histogram configuration")
    model = SparseCohortHistograms.empty(np.geomspace(minimum, maximum, bins + 1))
    for batch in primitive_batches:
        model.update(batch, "observed_sec_per_m", clip=True)
    return model


def fit_label_histograms(
    label_batches: Iterable[pd.DataFrame],
    config: "Stage1V3Config",
) -> dict[str, SparseCohortHistograms]:
    settings = config.section("normalization")
    bins = int(settings["raw_bins"])
    if bins < 2:
        raise ContractError("normalization.raw_bins must be at least 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    models = {
        "lcs": SparseCohortHistograms.empty(edges),
        "rts": SparseCohortHistograms.empty(edges),
    }
    for batch in label_batches:
        models["lcs"].update(batch, "lcs_raw")
        models["rts"].update(batch, "rts_raw")
    return models


def apply_reference_labels(
    primitives: pd.DataFrame,
    reference_model: SparseCohortHistograms,
    config: "Stage1V3Config",
    *,
    reference_fit_manifest_id: str,
    reference_model_id: str | None = None,
    leave_one_out: bool = False,
) -> pd.DataFrame:
    """Calculate RTS without converting an unavailable reference into zero."""

    result = primitives.copy()
    settings = config.section("reference")
    minimum_support = int(
        config.section("rts")["minimum_reference_sample_size"]
    )
    if minimum_support != int(settings["minimum_cohort_support"]):
        raise ContractError(
            "RTS and reference cohort support thresholds differ"
        )
    probability = float(settings["quantile"])
    reference, level, support = reference_model.choose_quantile(
        result,
        probability=probability,
        minimum_support=minimum_support,
        leave_one_out_value_column=(
            "observed_sec_per_m" if leave_one_out else None
        ),
    )
    pace = pd.to_numeric(result["observed_sec_per_m"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    valid_observation = np.isfinite(pace) & (pace > 0)
    valid_reference = np.isfinite(reference) & (reference > 0)
    available = valid_observation & valid_reference
    excess = np.full(len(result), np.nan, dtype=np.float64)
    excess[available] = np.maximum(pace[available] / reference[available] - 1.0, 0.0)
    rts = np.full(len(result), np.nan, dtype=np.float64)
    rts[available] = excess[available] / (1.0 + excess[available])

    reason = np.full(len(result), "", dtype=object)
    reason[~valid_observation] = "INVALID_OR_INSUFFICIENT_DIRECT_PACE"
    speed_valid = result.get(
        "rts_direct_speed_valid",
        pd.Series(True, index=result.index, dtype=bool),
    ).fillna(False).astype(bool)
    invalid_speed = ~speed_valid
    reason[invalid_speed.to_numpy(dtype=bool)] = "IMPOSSIBLE_DIRECT_SPEED"
    reason[valid_observation & ~valid_reference] = "REFERENCE_SUPPORT_UNAVAILABLE"
    result["reference_sec_per_m"] = reference
    result["reference_level_used"] = level
    result["reference_sample_size"] = support
    result["reference_fit_manifest_id"] = reference_fit_manifest_id
    result["reference_model_id"] = (
        reference_model_id
        if reference_model_id is not None
        else reference_fit_manifest_id
    )
    result["excess_time_ratio"] = excess
    result["rts_raw"] = rts
    result["rts_available"] = available
    result["rts_unavailable_reason"] = reason
    return result


def apply_percentile_labels(
    labels: pd.DataFrame,
    models: dict[str, SparseCohortHistograms],
    config: "Stage1V3Config",
) -> pd.DataFrame:
    result = labels.copy()
    minimum_support = int(config.section("normalization")["minimum_cohort_support"])
    for dimension in ("lcs", "rts"):
        if dimension not in models:
            raise ContractError(f"missing normalization model: {dimension}")
        raw = pd.to_numeric(
            result[f"{dimension}_raw"], errors="coerce"
        ).to_numpy(dtype=np.float64)
        flag_series = result[f"{dimension}_available"]
        strict_boolean = flag_series.map(
            lambda value: isinstance(value, (bool, np.bool_))
        )
        if not strict_boolean.all():
            raise ContractError(
                f"{dimension}_available must contain only non-null booleans"
            )
        source_available = flag_series.eq(True).to_numpy(dtype=bool)
        finite_raw = np.isfinite(raw)
        if np.any(source_available != finite_raw):
            raise ContractError(
                f"{dimension} raw value and availability flag are inconsistent"
            )
        percentile, level, support = models[dimension].choose_cdf(
            result,
            f"{dimension}_raw",
            minimum_support=minimum_support,
        )
        available = source_available & np.isfinite(percentile)
        lost_support = source_available & ~available
        percentile[~available] = np.nan
        if lost_support.any():
            result.loc[lost_support, f"{dimension}_raw"] = np.nan
            result.loc[lost_support, f"{dimension}_available"] = False
            result.loc[
                lost_support, f"{dimension}_unavailable_reason"
            ] = "CDF_SUPPORT_UNAVAILABLE"
        if np.isfinite(percentile).any() and (
            (percentile[np.isfinite(percentile)] < 0.0).any()
            or (percentile[np.isfinite(percentile)] > 1.0).any()
        ):
            raise ContractError(f"{dimension} percentile is outside [0, 1]")
        result[f"{dimension}_pct"] = percentile
        result[f"{dimension}_cdf_level_used"] = level
        result[f"{dimension}_cdf_sample_size"] = support
        result.loc[~available, f"{dimension}_cdf_level_used"] = "unresolved"
        result.loc[~available, f"{dimension}_cdf_sample_size"] = 0
    rts_tail_threshold = float(
        config.section("rts")["tail_event_percentile_threshold"]
    )
    result["rts_tail_event"] = (
        result["rts_available"].fillna(False)
        & pd.to_numeric(result["rts_pct"], errors="coerce").ge(
            rts_tail_threshold
        )
    )
    for unavailable in ("gns", "iis", "pmis"):
        for suffix in ("raw", "pct"):
            column = f"{unavailable}_{suffix}"
            if column in result and pd.to_numeric(
                result[column], errors="coerce"
            ).notna().any():
                raise ContractError(
                    f"{unavailable.upper()} cannot enter the dynamic v3 pipeline"
                )
        available_column = f"{unavailable}_available"
        if available_column in result:
            strict = result[available_column].map(
                lambda value: isinstance(value, (bool, np.bool_))
            )
            if not strict.all() or result[available_column].eq(True).any():
                raise ContractError(
                    f"{unavailable.upper()} must remain unavailable here"
                )

    result["iis_raw"] = np.nan
    result["iis_pct"] = np.nan
    result["iis_available"] = False
    result["iis_unavailable_reason"] = str(
        config.section("iis")["unavailable_reason"]
    )
    result["pmis_raw"] = np.nan
    result["pmis_pct"] = np.nan
    result["pmis_available"] = False
    result["pmis_unavailable_reason"] = str(
        config.section("pmis")["unavailable_reason"]
    )
    result["gns_raw"] = np.nan
    result["gns_pct"] = np.nan
    result["gns_available"] = False
    result["gns_unavailable_reason"] = "EDGE_STATIC_FEATURE_EXTENSION_NOT_FITTED"
    return result
