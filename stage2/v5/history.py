"""Indexed causal history lookup used by v5 performance-sensitive paths."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .contracts import Stage2V5ContractError


def causal_window_mean(
    event_key: np.ndarray,
    event_time: np.ndarray,
    event_value: np.ndarray,
    query_key: np.ndarray,
    decision_time: np.ndarray,
    *,
    window_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    event_key = np.asarray(event_key).astype(str)
    event_time = np.asarray(event_time, dtype=np.float64)
    event_value = np.asarray(event_value, dtype=np.float64)
    query_key = np.asarray(query_key).astype(str)
    decision_time = np.asarray(decision_time, dtype=np.float64)
    if not (len(event_key) == len(event_time) == len(event_value)):
        raise Stage2V5ContractError("history event array lengths differ")
    if len(query_key) != len(decision_time):
        raise Stage2V5ContractError("history query array lengths differ")
    event_order = np.lexsort((event_time, event_key))
    query_order = np.lexsort((decision_time, query_key))
    sorted_event_key = event_key[event_order]
    sorted_event_time = event_time[event_order]
    sorted_event_value = event_value[event_order]
    sorted_query_key = query_key[query_order]
    sorted_decision = decision_time[query_order]
    all_keys, event_codes = np.unique(sorted_event_key, return_inverse=True)
    query_codes = np.searchsorted(all_keys, sorted_query_key)
    known = (query_codes < len(all_keys)) & (all_keys[np.minimum(query_codes, max(len(all_keys) - 1, 0))] == sorted_query_key) if len(all_keys) else np.zeros(len(query_codes), dtype=bool)
    sorted_mean = np.full(len(query_key), np.nan)
    sorted_count = np.zeros(len(query_key), dtype=np.int64)
    event_sizes = np.bincount(event_codes, minlength=len(all_keys))
    event_offsets = np.concatenate(([0], np.cumsum(event_sizes)))
    query_positions = np.flatnonzero(known)
    if len(query_positions):
        known_codes = query_codes[query_positions]
        order = np.argsort(known_codes, kind="stable")
        grouped_positions = query_positions[order]
        grouped_codes = known_codes[order]
        sizes = np.bincount(grouped_codes, minlength=len(all_keys))
        offsets = np.concatenate(([0], np.cumsum(sizes)))
        for code in np.flatnonzero(sizes):  # O(K), slices only the indexed cohort.
            queries = grouped_positions[offsets[code] : offsets[code + 1]]
            left_event = event_offsets[code]
            right_event = event_offsets[code + 1]
            times = sorted_event_time[left_event:right_event]
            values = sorted_event_value[left_event:right_event]
            finite = np.isfinite(values)
            times = times[finite]
            values = values[finite]
            if not len(times):
                continue
            cutoff = sorted_decision[queries]
            right = np.searchsorted(times, cutoff, side="left")
            left = np.searchsorted(times, cutoff - float(window_s), side="left")
            prefix = np.concatenate(([0.0], np.cumsum(values)))
            count = right - left
            usable = count > 0
            sorted_count[queries] = count
            sorted_mean[queries[usable]] = (prefix[right[usable]] - prefix[left[usable]]) / count[usable]
    mean = np.empty_like(sorted_mean)
    count = np.empty_like(sorted_count)
    mean[query_order] = sorted_mean
    count[query_order] = sorted_count
    return mean, count

