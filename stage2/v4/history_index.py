"""Vectorized strict-cutoff queries over the Stage 2 v4 event store."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from .contracts import Stage2V4ContractError, require_columns
from .history_store import HISTORY_VALUE_COLUMNS, validate_history_store


LEVEL_KEYS: dict[str, tuple[str, ...]] = {
    "edge_time": (
        "observed_directed_edge_uid",
        "profile_time_bin",
        "profile_weekday_type",
    ),
    "edge": ("observed_directed_edge_uid",),
    "highway_time": (
        "canonical_highway",
        "profile_time_bin",
        "profile_weekday_type",
    ),
    "highway": ("canonical_highway",),
    "global": (),
}
METRIC_AVAILABILITY = {
    "lcs_raw": "lcs_available",
    "lcs_tail_event": "lcs_available",
    "rts_raw": "rts_available",
    "rts_tail_event": "rts_available",
}


@dataclass(frozen=True)
class HistoryQuery:
    mean: np.ndarray
    std: np.ndarray
    count: np.ndarray
    maximum_event_time: np.ndarray


def _normalise_key(value: Any) -> Any:
    if isinstance(value, tuple):
        return tuple(
            item.item() if isinstance(item, np.generic) else item for item in value
        )
    return value.item() if isinstance(value, np.generic) else value


class TemporalHistoryIndex:
    """In-memory group index; all cutoffs use searchsorted(side='left')."""

    def __init__(self, events: pd.DataFrame):
        required = {
            "availability_timestamp",
            "observed_directed_edge_uid",
            "canonical_highway",
            "profile_time_bin",
            "profile_weekday_type",
            "lcs_available",
            "rts_available",
            *HISTORY_VALUE_COLUMNS,
        }
        require_columns(events.columns, required, "causal history events")
        times = pd.to_numeric(events["availability_timestamp"], errors="coerce")
        if not np.isfinite(times.to_numpy(dtype=float, na_value=np.nan)).all():
            raise Stage2V4ContractError("history index contains a missing timestamp")
        self.events = events.reset_index(drop=True)
        self._times = times.to_numpy(dtype=np.float64)
        self._group_cache: dict[str, dict[Any, np.ndarray]] = {}

    @classmethod
    def from_store(
        cls,
        root: str | Path,
        config: Any,
    ) -> "TemporalHistoryIndex":
        source = Path(root)
        summary = validate_history_store(source, config)
        columns = [
            "availability_timestamp",
            "observed_directed_edge_uid",
            "canonical_highway",
            "profile_time_bin",
            "profile_weekday_type",
            "lcs_available",
            "rts_available",
            *HISTORY_VALUE_COLUMNS,
        ]
        frames = [
            pd.read_parquet(
                source / "events" / f"day={date}.parquet",
                columns=columns,
            )
            for date in sorted(summary["event_files"])
        ]
        events = pd.concat(frames, ignore_index=True)
        del frames
        events.sort_values("availability_timestamp", kind="stable", inplace=True)
        events.reset_index(drop=True, inplace=True)
        return cls(events)

    def _groups(self, level: str) -> dict[Any, np.ndarray]:
        if level not in LEVEL_KEYS:
            raise Stage2V4ContractError(f"unknown history fallback level: {level}")
        if level in self._group_cache:
            return self._group_cache[level]
        keys = LEVEL_KEYS[level]
        if not keys:
            groups = {"__global__": np.arange(len(self.events), dtype=np.int64)}
        else:
            grouped = self.events.groupby(
                list(keys),
                sort=False,
                observed=True,
                dropna=False,
            ).indices
            groups = {
                _normalise_key(key): np.asarray(indices, dtype=np.int64)
                for key, indices in grouped.items()
            }
        self._group_cache[level] = groups
        return groups

    def _query_groups(self, queries: pd.DataFrame, level: str) -> dict[Any, np.ndarray]:
        keys = LEVEL_KEYS[level]
        require_columns(queries.columns, {"decision_time", *keys}, "history queries")
        if not keys:
            return {"__global__": np.arange(len(queries), dtype=np.int64)}
        grouped = queries.groupby(
            list(keys),
            sort=False,
            observed=True,
            dropna=False,
        ).indices
        return {
            _normalise_key(key): np.asarray(indices, dtype=np.int64)
            for key, indices in grouped.items()
        }

    def query_metric(
        self,
        queries: pd.DataFrame,
        *,
        level: str,
        metric: str,
        window_s: float | None = None,
    ) -> HistoryQuery:
        if metric != "__event__" and metric not in self.events:
            raise Stage2V4ContractError(f"history metric is missing: {metric}")
        decision = pd.to_numeric(queries["decision_time"], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if not np.isfinite(decision).all():
            raise Stage2V4ContractError("history query has a missing decision_time")
        n_rows = len(queries)
        means = np.full(n_rows, np.nan, dtype=np.float64)
        stds = np.full(n_rows, np.nan, dtype=np.float64)
        counts = np.zeros(n_rows, dtype=np.int64)
        maxima = np.full(n_rows, np.nan, dtype=np.float64)
        event_groups = self._groups(level)
        query_groups = self._query_groups(queries, level)
        availability_column = METRIC_AVAILABILITY.get(metric)
        availability = (
            self.events[availability_column].fillna(False).to_numpy(dtype=bool)
            if availability_column
            else None
        )
        raw_values = (
            np.ones(len(self.events), dtype=np.float64)
            if metric == "__event__"
            else pd.to_numeric(self.events[metric], errors="coerce").to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
        )
        for key, query_positions in query_groups.items():
            event_positions = event_groups.get(key)
            if event_positions is None or not len(event_positions):
                continue
            event_times = self._times[event_positions]
            values = raw_values[event_positions]
            valid = np.isfinite(values)
            if availability is not None:
                valid &= availability[event_positions]
            if not valid.any():
                continue
            metric_times = event_times[valid]
            metric_values = values[valid]
            if len(metric_times) > 1 and np.any(metric_times[1:] < metric_times[:-1]):
                order = np.argsort(metric_times, kind="stable")
                metric_times = metric_times[order]
                metric_values = metric_values[order]
            cutoff = decision[query_positions]
            right = np.searchsorted(metric_times, cutoff, side="left")
            if window_s is None:
                left = np.zeros_like(right)
            else:
                left = np.searchsorted(
                    metric_times,
                    cutoff - float(window_s),
                    side="left",
                )
            prefix = np.concatenate(([0.0], np.cumsum(metric_values, dtype=np.float64)))
            prefix_sq = np.concatenate(
                ([0.0], np.cumsum(metric_values * metric_values, dtype=np.float64))
            )
            count = right - left
            total = prefix[right] - prefix[left]
            total_sq = prefix_sq[right] - prefix_sq[left]
            usable = count > 0
            target = query_positions[usable]
            local_count = count[usable]
            local_mean = total[usable] / local_count
            local_variance = np.maximum(
                total_sq[usable] / local_count - local_mean * local_mean,
                0.0,
            )
            means[target] = local_mean
            stds[target] = np.sqrt(local_variance)
            counts[target] = local_count
            maxima[target] = metric_times[right[usable] - 1]
        if np.any(maxima[np.isfinite(maxima)] >= decision[np.isfinite(maxima)]):
            raise Stage2V4ContractError(
                "history query leaked an event at/after decision_time"
            )
        return HistoryQuery(means, stds, counts, maxima)

    def query_fallback(
        self,
        queries: pd.DataFrame,
        *,
        metrics: Iterable[str],
        minimum_observations: int,
        levels: tuple[str, ...] = (
            "edge_time",
            "edge",
            "highway_time",
            "highway",
            "global",
        ),
    ) -> pd.DataFrame:
        metric_names = tuple(metrics)
        storage: dict[str, dict[str, np.ndarray]] = {}
        for metric in metric_names:
            storage[metric] = {
                "mean": np.full(len(queries), np.nan),
                "std": np.full(len(queries), np.nan),
                "count": np.zeros(len(queries), dtype=np.int64),
                "maximum": np.full(len(queries), np.nan),
                "chosen": np.full(len(queries), "", dtype=object),
            }
        for level in levels:
            level_results = self._query_level_metrics(
                queries,
                level=level,
                metrics=metric_names,
            )
            threshold = 1 if level == "global" else int(minimum_observations)
            for metric in metric_names:
                target = storage[metric]
                result = level_results[metric]
                use = (target["chosen"] == "") & (result.count >= threshold)
                target["mean"][use] = result.mean[use]
                target["std"][use] = result.std[use]
                target["count"][use] = result.count[use]
                target["maximum"][use] = result.maximum_event_time[use]
                target["chosen"][use] = level

        output_columns: dict[str, Any] = {}
        for metric in metric_names:
            target = storage[metric]
            prefix = f"{metric}_profile"
            output_columns[f"{prefix}_mean"] = target["mean"]
            output_columns[f"{prefix}_std"] = target["std"]
            output_columns[f"{prefix}_count"] = target["count"]
            output_columns[f"{prefix}_maximum_event_time"] = target["maximum"]
            output_columns[f"{prefix}_fallback_level"] = pd.Series(
                target["chosen"],
                index=queries.index,
                dtype="string",
            ).replace("", pd.NA)
        return pd.DataFrame(output_columns, index=queries.index)

    def _query_level_metrics(
        self,
        queries: pd.DataFrame,
        *,
        level: str,
        metrics: tuple[str, ...],
        window_s: float | None = None,
    ) -> dict[str, HistoryQuery]:
        decision = pd.to_numeric(queries["decision_time"], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if not np.isfinite(decision).all():
            raise Stage2V4ContractError("history query has a missing decision_time")
        n_rows = len(queries)
        arrays = {
            metric: {
                "mean": np.full(n_rows, np.nan, dtype=np.float64),
                "std": np.full(n_rows, np.nan, dtype=np.float64),
                "count": np.zeros(n_rows, dtype=np.int64),
                "maximum": np.full(n_rows, np.nan, dtype=np.float64),
            }
            for metric in metrics
        }
        raw_values = {
            metric: pd.to_numeric(self.events[metric], errors="coerce").to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
            for metric in metrics
        }
        availability = {
            metric: (
                self.events[METRIC_AVAILABILITY[metric]]
                .fillna(False)
                .to_numpy(dtype=bool)
                if metric in METRIC_AVAILABILITY
                else None
            )
            for metric in metrics
        }
        event_groups = self._groups(level)
        query_groups = self._query_groups(queries, level)
        for key, query_positions in query_groups.items():
            event_positions = event_groups.get(key)
            if event_positions is None or not len(event_positions):
                continue
            event_times = self._times[event_positions]
            cutoff = decision[query_positions]
            for metric in metrics:
                values = raw_values[metric][event_positions]
                valid = np.isfinite(values)
                available = availability[metric]
                if available is not None:
                    valid &= available[event_positions]
                if not valid.any():
                    continue
                metric_times = event_times[valid]
                metric_values = values[valid]
                if len(metric_times) > 1 and np.any(
                    metric_times[1:] < metric_times[:-1]
                ):
                    order = np.argsort(metric_times, kind="stable")
                    metric_times = metric_times[order]
                    metric_values = metric_values[order]
                right = np.searchsorted(metric_times, cutoff, side="left")
                left = (
                    np.zeros_like(right)
                    if window_s is None
                    else np.searchsorted(
                        metric_times,
                        cutoff - float(window_s),
                        side="left",
                    )
                )
                count = right - left
                usable = count > 0
                if not usable.any():
                    continue
                prefix = np.concatenate(
                    ([0.0], np.cumsum(metric_values, dtype=np.float64))
                )
                prefix_sq = np.concatenate(
                    (
                        [0.0],
                        np.cumsum(metric_values * metric_values, dtype=np.float64),
                    )
                )
                total = prefix[right] - prefix[left]
                total_sq = prefix_sq[right] - prefix_sq[left]
                target_positions = query_positions[usable]
                local_count = count[usable]
                local_mean = total[usable] / local_count
                local_variance = np.maximum(
                    total_sq[usable] / local_count - local_mean * local_mean,
                    0.0,
                )
                target = arrays[metric]
                target["mean"][target_positions] = local_mean
                target["std"][target_positions] = np.sqrt(local_variance)
                target["count"][target_positions] = local_count
                target["maximum"][target_positions] = metric_times[right[usable] - 1]
        results = {
            metric: HistoryQuery(
                values["mean"],
                values["std"],
                values["count"],
                values["maximum"],
            )
            for metric, values in arrays.items()
        }
        for result in results.values():
            finite = np.isfinite(result.maximum_event_time)
            if np.any(result.maximum_event_time[finite] >= decision[finite]):
                raise Stage2V4ContractError(
                    "history query leaked an event at/after decision_time"
                )
        return results

    def _legacy_query_fallback(
        self,
        queries: pd.DataFrame,
        *,
        metrics: Iterable[str],
        minimum_observations: int,
        levels: tuple[str, ...],
    ) -> pd.DataFrame:
        """Reference implementation retained only for audit comparisons."""
        output = pd.DataFrame(index=queries.index)
        for metric in metrics:
            mean = np.full(len(queries), np.nan)
            std = np.full(len(queries), np.nan)
            count = np.zeros(len(queries), dtype=np.int64)
            maximum = np.full(len(queries), np.nan)
            chosen = np.full(len(queries), "", dtype=object)
            for level in levels:
                result = self.query_metric(
                    queries,
                    level=level,
                    metric=metric,
                )
                threshold = 1 if level == "global" else int(minimum_observations)
                use = (chosen == "") & (result.count >= threshold)
                mean[use] = result.mean[use]
                std[use] = result.std[use]
                count[use] = result.count[use]
                maximum[use] = result.maximum_event_time[use]
                chosen[use] = level
            prefix = f"{metric}_profile"
            output[f"{prefix}_mean"] = mean
            output[f"{prefix}_std"] = std
            output[f"{prefix}_count"] = count
            output[f"{prefix}_maximum_event_time"] = maximum
            output[f"{prefix}_fallback_level"] = pd.Series(
                chosen,
                index=queries.index,
                dtype="string",
            ).replace("", pd.NA)
        return output

    def query_window_features(
        self,
        queries: pd.DataFrame,
        *,
        level: str,
        metrics: Iterable[str],
        windows_minutes: Iterable[int],
    ) -> pd.DataFrame:
        metric_names = tuple(metrics)
        windows = tuple(int(value) for value in windows_minutes)
        decision = pd.to_numeric(queries["decision_time"], errors="coerce").to_numpy(
            dtype=np.float64,
            na_value=np.nan,
        )
        if not np.isfinite(decision).all():
            raise Stage2V4ContractError("history query has a missing decision_time")
        n_rows = len(queries)
        event_arrays = {
            minutes: {
                "count": np.zeros(n_rows, dtype=np.int64),
                "maximum": np.full(n_rows, np.nan),
            }
            for minutes in windows
        }
        metric_arrays = {
            (minutes, metric): {
                "mean": np.full(n_rows, np.nan),
                "std": np.full(n_rows, np.nan),
                "count": np.zeros(n_rows, dtype=np.int64),
            }
            for minutes in windows
            for metric in metric_names
        }
        raw_values = {
            metric: pd.to_numeric(self.events[metric], errors="coerce").to_numpy(
                dtype=np.float64,
                na_value=np.nan,
            )
            for metric in metric_names
        }
        availability = {
            metric: (
                self.events[METRIC_AVAILABILITY[metric]]
                .fillna(False)
                .to_numpy(dtype=bool)
                if metric in METRIC_AVAILABILITY
                else None
            )
            for metric in metric_names
        }
        event_groups = self._groups(level)
        query_groups = self._query_groups(queries, level)
        for key, query_positions in query_groups.items():
            event_positions = event_groups.get(key)
            if event_positions is None or not len(event_positions):
                continue
            event_times = self._times[event_positions]
            if len(event_times) > 1 and np.any(event_times[1:] < event_times[:-1]):
                order = np.argsort(event_times, kind="stable")
                event_times = event_times[order]
                event_positions = event_positions[order]
            cutoff = decision[query_positions]
            event_right = np.searchsorted(event_times, cutoff, side="left")
            for minutes in windows:
                event_left = np.searchsorted(
                    event_times,
                    cutoff - float(minutes) * 60.0,
                    side="left",
                )
                count = event_right - event_left
                usable = count > 0
                target = event_arrays[minutes]
                target["count"][query_positions] = count
                target["maximum"][query_positions[usable]] = event_times[
                    event_right[usable] - 1
                ]

            for metric in metric_names:
                values = raw_values[metric][event_positions]
                valid = np.isfinite(values)
                available = availability[metric]
                if available is not None:
                    valid &= available[event_positions]
                if not valid.any():
                    continue
                metric_times = event_times[valid]
                metric_values = values[valid]
                right = np.searchsorted(metric_times, cutoff, side="left")
                prefix_sum = np.concatenate(
                    ([0.0], np.cumsum(metric_values, dtype=np.float64))
                )
                prefix_sq = np.concatenate(
                    (
                        [0.0],
                        np.cumsum(metric_values * metric_values, dtype=np.float64),
                    )
                )
                for minutes in windows:
                    left = np.searchsorted(
                        metric_times,
                        cutoff - float(minutes) * 60.0,
                        side="left",
                    )
                    count = right - left
                    usable = count > 0
                    if not usable.any():
                        continue
                    total = prefix_sum[right] - prefix_sum[left]
                    total_sq = prefix_sq[right] - prefix_sq[left]
                    local_count = count[usable]
                    mean = total[usable] / local_count
                    variance = np.maximum(
                        total_sq[usable] / local_count - mean * mean,
                        0.0,
                    )
                    target = metric_arrays[(minutes, metric)]
                    target_positions = query_positions[usable]
                    target["mean"][target_positions] = mean
                    target["std"][target_positions] = np.sqrt(variance)
                    target["count"][target_positions] = local_count

        output_columns: dict[str, Any] = {}
        for minutes in windows:
            prefix = f"{level}_{minutes}m"
            event = event_arrays[minutes]
            output_columns[f"{prefix}_completed_traversal_count"] = event["count"]
            output_columns[f"{prefix}_maximum_event_time"] = event["maximum"]
            output_columns[f"{prefix}_feature_age_s"] = decision - event["maximum"]
            output_columns[f"{prefix}_feature_available"] = event["count"] > 0
            for metric in metric_names:
                values = metric_arrays[(minutes, metric)]
                output_columns[f"{prefix}_{metric}_mean"] = values["mean"]
                output_columns[f"{prefix}_{metric}_std"] = values["std"]
                output_columns[
                    f"{prefix}_{metric}_available_label_count"
                ] = values["count"]
        for minutes in windows:
            maximum = event_arrays[minutes]["maximum"]
            finite = np.isfinite(maximum)
            if np.any(maximum[finite] >= decision[finite]):
                raise Stage2V4ContractError(
                    "history window leaked an event at/after decision_time"
                )
        return pd.DataFrame(output_columns, index=queries.index)
