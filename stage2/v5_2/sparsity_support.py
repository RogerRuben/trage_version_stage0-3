"""Train-only support primitives for the Stage 2 v5.2 sparsity diagnostic."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .contracts import Stage2V52ContractError, require_columns


IDENTITY_COLUMNS = ("date", "order_id", "traversal_id")
TARGETS = ("crawl", "stop", "speed_cv", "acceleration_rms")
TARGET_VALID_COLUMNS = {
    "crawl": "crawl_target_valid",
    "stop": "stop_target_valid",
    "speed_cv": "speed_cv_target_valid",
    "acceleration_rms": "acceleration_rms_target_valid",
}
EDGE_COLUMN = "observed_directed_edge_uid"
TIME_BIN_COLUMN = "estimated_time_bin"
GROUPS = ("unseen", "low", "medium", "high")
SUPPORT_REQUIRED_COLUMNS = (
    "split", *IDENTITY_COLUMNS, EDGE_COLUMN, TIME_BIN_COLUMN, *TARGET_VALID_COLUMNS.values(),
)


@dataclass(frozen=True)
class SupportCounts:
    spatial: pd.Series
    spatiotemporal: pd.Series
    target_specific: dict[str, pd.Series]
    source_row_count: int
    unique_physical_traversal_count: int
    duplicate_removed_count: int
    missing_edge_count: int
    fit_dates_observed: tuple[str, ...]


def unique_physical_traversals(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Collapse overlap copies only when their support-defining fields agree."""
    require_columns(frame.columns, SUPPORT_REQUIRED_COLUMNS, product="sparsity support source")
    duplicated = frame.duplicated(list(IDENTITY_COLUMNS), keep=False)
    if not duplicated.any():
        return frame, 0
    defining = (EDGE_COLUMN, TIME_BIN_COLUMN, *TARGET_VALID_COLUMNS.values())
    disagreement = (
        frame.loc[duplicated]
        .groupby(list(IDENTITY_COLUMNS), sort=False, observed=True, dropna=False)[list(defining)]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )
    if disagreement.any():
        raise Stage2V52ContractError("overlap copies disagree on sparsity support identity")
    unique = frame.drop_duplicates(list(IDENTITY_COLUMNS), keep="first")
    return unique, int(len(frame) - len(unique))


def _sum_series(parts: list[pd.Series], *, names: Sequence[str]) -> pd.Series:
    if not parts:
        index = pd.MultiIndex.from_arrays([[] for _ in names], names=list(names)) if len(names) > 1 else pd.Index([], name=names[0])
        return pd.Series([], index=index, dtype=np.int64)
    combined = pd.concat(parts)
    levels: int | list[int] = list(range(len(names))) if len(names) > 1 else 0
    result = combined.groupby(level=levels, sort=True, observed=True).sum().astype(np.int64)
    result.index.names = list(names)
    return result.sort_index()


def fit_support_counts(
    train_frames: Iterable[pd.DataFrame], *, expected_dates: Sequence[str],
) -> SupportCounts:
    """Fit spatial, edge-time, and target-valid edge-time counts from Train only."""
    spatial_parts: list[pd.Series] = []
    temporal_parts: list[pd.Series] = []
    target_parts: dict[str, list[pd.Series]] = {target: [] for target in TARGETS}
    source_rows = unique_rows = duplicate_rows = missing_edges = 0
    observed_dates: set[str] = set()
    for frame in train_frames:
        require_columns(frame.columns, SUPPORT_REQUIRED_COLUMNS, product="sparsity Train source")
        source_rows += len(frame)
        if not frame["split"].astype(str).eq("train").all():
            raise Stage2V52ContractError("evaluation rows cannot enter sparsity support fit")
        unique, removed = unique_physical_traversals(frame)
        duplicate_rows += removed
        unique_rows += len(unique)
        observed_dates.update(unique["date"].astype(str).unique())
        bins = pd.to_numeric(unique[TIME_BIN_COLUMN], errors="coerce")
        if bins.isna().any() or not bins.between(0, 47).all():
            raise Stage2V52ContractError("sparsity support must use the frozen 0-47 time_bin")
        valid_edge = unique[EDGE_COLUMN].notna() & unique[EDGE_COLUMN].astype(str).ne("")
        missing_edges += int((~valid_edge).sum())
        usable = unique.loc[valid_edge].copy()
        usable[EDGE_COLUMN] = usable[EDGE_COLUMN].astype(str)
        usable[TIME_BIN_COLUMN] = pd.to_numeric(usable[TIME_BIN_COLUMN]).astype(np.int16)
        spatial_parts.append(usable.groupby(EDGE_COLUMN, sort=True, observed=True).size())
        key = [EDGE_COLUMN, TIME_BIN_COLUMN]
        temporal_parts.append(usable.groupby(key, sort=True, observed=True).size())
        for target, validity in TARGET_VALID_COLUMNS.items():
            valid = usable[validity].fillna(False).astype(bool)
            target_parts[target].append(usable.loc[valid].groupby(key, sort=True, observed=True).size())
    expected = tuple(str(value) for value in expected_dates)
    if tuple(sorted(observed_dates)) != tuple(sorted(expected)):
        raise Stage2V52ContractError("sparsity support dates differ from frozen Train dates")
    return SupportCounts(
        spatial=_sum_series(spatial_parts, names=(EDGE_COLUMN,)),
        spatiotemporal=_sum_series(temporal_parts, names=(EDGE_COLUMN, TIME_BIN_COLUMN)),
        target_specific={
            target: _sum_series(parts, names=(EDGE_COLUMN, TIME_BIN_COLUMN))
            for target, parts in target_parts.items()
        },
        source_row_count=source_rows,
        unique_physical_traversal_count=unique_rows,
        duplicate_removed_count=duplicate_rows,
        missing_edge_count=missing_edges,
        fit_dates_observed=tuple(sorted(observed_dates)),
    )


def positive_quantiles(counts: pd.Series) -> dict[str, float]:
    positive = counts.to_numpy(np.float64)
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if not len(positive):
        raise Stage2V52ContractError("support counts contain no positive observations")
    values = np.quantile(positive, (0.25, 0.50, 0.75, 0.90))
    return {name: float(value) for name, value in zip(("p25", "p50", "p75", "p90"), values)}


def support_groups(counts: Sequence[int | float], quantiles: Mapping[str, float]) -> np.ndarray:
    values = np.asarray(counts, dtype=np.float64)
    groups = np.full(len(values), "high", dtype=object)
    groups[values <= float(quantiles["p75"])] = "medium"
    groups[values <= float(quantiles["p25"])] = "low"
    groups[~np.isfinite(values) | (values <= 0)] = "unseen"
    return groups.astype(str)


def lookup_edge_time(counts: pd.Series, edges: pd.Series, time_bins: pd.Series) -> np.ndarray:
    index = pd.MultiIndex.from_arrays(
        [edges.fillna("").astype(str), pd.to_numeric(time_bins, errors="coerce").fillna(-1).astype(np.int16)],
        names=[EDGE_COLUMN, TIME_BIN_COLUMN],
    )
    return counts.reindex(index, fill_value=0).to_numpy(np.int64)


def count_records_sha256(counts: pd.Series) -> str:
    """Hash sorted support records without embedding a large lookup in JSON."""
    digest = hashlib.sha256()
    ordered = counts.sort_index()
    if isinstance(ordered.index, pd.MultiIndex):
        iterator = ((*index, int(value)) for index, value in ordered.items())
    else:
        iterator = ((index, int(value)) for index, value in ordered.items())
    for record in iterator:
        digest.update(("\x1f".join(str(value) for value in record) + "\n").encode("utf-8"))
    return digest.hexdigest()


def support_group_record_counts(counts: pd.Series, quantiles: Mapping[str, float]) -> dict[str, int]:
    labels = support_groups(counts.to_numpy(np.int64), quantiles)
    return {group: int((labels == group).sum()) for group in GROUPS}


def spatial_high_temporal_sparse(spatial_group: Sequence[str], temporal_group: Sequence[str]) -> np.ndarray:
    spatial = np.asarray(spatial_group).astype(str)
    temporal = np.asarray(temporal_group).astype(str)
    return (spatial == "high") & np.isin(temporal, ("unseen", "low"))


def validate_prediction_alignment(frames: Mapping[str, pd.DataFrame]) -> None:
    """Require exact physical traversal pairing and identical truths/masks."""
    required_models = ("M1", "M3", "M4")
    if tuple(frames) != required_models:
        raise Stage2V52ContractError("sparsity diagnostic requires paired M1/M3/M4 predictions")
    reference = frames["M1"]
    require_columns(reference.columns, IDENTITY_COLUMNS, product="M1 diagnostic predictions")
    if reference.duplicated(list(IDENTITY_COLUMNS)).any():
        raise Stage2V52ContractError("M1 diagnostic predictions duplicate a physical traversal")
    for model in required_models[1:]:
        candidate = frames[model]
        require_columns(candidate.columns, IDENTITY_COLUMNS, product=f"{model} diagnostic predictions")
        if candidate.duplicated(list(IDENTITY_COLUMNS)).any():
            raise Stage2V52ContractError(f"{model} diagnostic predictions duplicate a physical traversal")
        if not reference.loc[:, IDENTITY_COLUMNS].reset_index(drop=True).equals(
            candidate.loc[:, IDENTITY_COLUMNS].reset_index(drop=True)
        ):
            raise Stage2V52ContractError("M1/M3/M4 physical traversal identities are not exactly paired")
        for target in TARGETS:
            mask = f"{target}_valid"
            truth = f"target_{target}"
            if not np.array_equal(reference[mask].to_numpy(bool), candidate[mask].to_numpy(bool)):
                raise Stage2V52ContractError(f"paired models disagree on {target} validity")
            left = reference[truth].to_numpy(float)
            right = candidate[truth].to_numpy(float)
            if not np.allclose(left, right, equal_nan=True, atol=1.0e-7, rtol=0):
                raise Stage2V52ContractError(f"paired models disagree on {target} truth")


def cluster_bootstrap_difference(
    left: Sequence[float], right: Sequence[float], *, resamples: int, seed: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Cell-cluster bootstrap of mean(left) - mean(right)."""
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    left_values = left_values[np.isfinite(left_values)]
    right_values = right_values[np.isfinite(right_values)]
    if not len(left_values) or not len(right_values):
        return {
            "status": "INSUFFICIENT_SUPPORT", "left_cell_count": int(len(left_values)),
            "right_cell_count": int(len(right_values)), "effect": None,
            "ci_lower": None, "ci_upper": None,
        }
    rng = np.random.default_rng(int(seed))
    distribution = np.empty(int(resamples), dtype=np.float64)
    for index in range(int(resamples)):
        left_mean = left_values[rng.integers(0, len(left_values), len(left_values))].mean()
        right_mean = right_values[rng.integers(0, len(right_values), len(right_values))].mean()
        distribution[index] = left_mean - right_mean
    alpha = (1.0 - float(confidence_level)) / 2.0
    return {
        "status": "PASS", "left_cell_count": int(len(left_values)),
        "right_cell_count": int(len(right_values)),
        "effect": float(left_values.mean() - right_values.mean()),
        "ci_lower": float(np.quantile(distribution, alpha)),
        "ci_upper": float(np.quantile(distribution, 1.0 - alpha)),
    }
