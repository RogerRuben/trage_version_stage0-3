"""Vectorized token and original-route micro-condition products."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .contracts import (
    STATIC_COMPLEXITY_COLUMNS,
    TOKEN_REQUIRED_COLUMNS,
    Stage2V52ContractError,
    require_columns,
)
from .support_transfer import lookup_train_support


ORDER_KEYS = ("split", "date", "order_id")
PREDICTION_ALIASES = {
    "pred_crawl_share": "pred_crawl_time_share",
    "pred_stop_share": "pred_stop_time_share",
    "pred_speed_cv_bounded": "pred_speed_cv_bounded",
    "pred_acceleration_rms_bounded": "pred_acceleration_rms_bounded",
    "pred_rts_raw": "pred_rts_raw",
    "pred_pace_p50": "pace_pred_p50",
}
AVAILABILITY_ALIASES = {
    "crawl_target_available": "crawl_target_valid",
    "stop_target_available": "stop_target_valid",
    "speed_cv_target_available": "speed_cv_target_valid",
    "acceleration_target_available": "acceleration_rms_target_valid",
    "rts_target_available": "rts_target_valid",
}
DIMENSIONS = {
    "crawl": "pred_crawl_share",
    "stop": "pred_stop_share",
    "speed_cv": "pred_speed_cv_bounded",
    "acceleration": "pred_acceleration_rms_bounded",
    "rts": "pred_rts_raw",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def build_micro_condition_tokens(
    predictions: pd.DataFrame,
    route_context: pd.DataFrame,
    *,
    support_artifact: Mapping[str, Any],
    prediction_source: str,
    model_id: str,
    model_hash: str,
) -> pd.DataFrame:
    """Create one formal prediction row per physical traversal."""
    join = [*ORDER_KEYS, "traversal_id"]
    require_columns(predictions.columns, (*join, "allocated_distance_m"), product="traversal predictions")
    require_columns(route_context.columns, join, product="route context")
    cutoff_inputs = {"feature_cutoff_time", "decision_time", "feature_age_s"} & set(route_context.columns)
    if "feature_cutoff_time" not in cutoff_inputs and not {"decision_time", "feature_age_s"} <= cutoff_inputs:
        raise Stage2V52ContractError(
            "route context requires feature_cutoff_time or decision_time plus feature_age_s"
        )
    context_required = (
        *join, "route_sequence", "observed_directed_edge_uid", "canonical_edge_uid",
    )
    require_columns(route_context.columns, context_required, product="route context")
    if predictions.duplicated(join).any() or route_context.duplicated(join).any():
        raise Stage2V52ContractError("token identity is not one-to-one")
    keep = list(dict.fromkeys([
        *context_required, "history_support", "observed_sec_per_m_profile_count",
        "feature_cutoff_time", "decision_time", "feature_age_s",
        "route_part_length_m", "canonical_highway", "road_class", "bridge", "tunnel",
    ]))
    keep = [column for column in keep if column in route_context.columns]
    merged = predictions.merge(route_context.loc[:, keep], on=join, how="left", validate="one_to_one")
    if merged["observed_directed_edge_uid"].isna().any():
        raise Stage2V52ContractError("prediction row is missing its original-route edge identity")
    result = merged.loc[:, [column for column in ORDER_KEYS if column in merged.columns]].copy()
    for column in ("order_id", "route_sequence", "traversal_id", "observed_directed_edge_uid", "canonical_edge_uid"):
        result[column] = merged[column]
    for target, source in PREDICTION_ALIASES.items():
        if source not in merged:
            raise Stage2V52ContractError(f"prediction is missing {source}")
        result[target] = pd.to_numeric(merged[source], errors="coerce")
    distance = pd.to_numeric(merged["allocated_distance_m"], errors="coerce")
    result["allocated_distance_m"] = distance
    result["estimated_travel_time_p50_s"] = result["pred_pace_p50"] * distance
    for target, source in AVAILABILITY_ALIASES.items():
        result[target] = merged[source].fillna(False).astype(bool) if source in merged else False
    history_source = "history_support" if "history_support" in merged else "observed_sec_per_m_profile_count"
    result["history_support"] = pd.to_numeric(merged[history_source], errors="coerce").fillna(0).astype("int64")
    support, group = lookup_train_support(result["observed_directed_edge_uid"], support_artifact)
    result["edge_train_support"] = support
    result["edge_seen_in_train"] = support > 0
    result["support_group"] = group
    result["prediction_source"] = str(prediction_source)
    result["model_id"] = str(model_id)
    result["model_hash"] = str(model_hash)
    if "feature_cutoff_time" in merged:
        cutoff = pd.to_numeric(merged["feature_cutoff_time"], errors="coerce")
    else:
        decision = pd.to_numeric(merged["decision_time"], errors="coerce")
        age = pd.to_numeric(merged["feature_age_s"], errors="coerce")
        cutoff = decision - age
        if ((age <= 0) | age.isna()).any():
            raise Stage2V52ContractError("derived feature cutoff requires strictly positive feature_age_s")
    result["feature_cutoff_time"] = cutoff
    if result["feature_cutoff_time"].isna().any():
        raise Stage2V52ContractError("feature_cutoff_time cannot be missing")
    for optional in ("route_part_length_m", "canonical_highway", "road_class", "bridge", "tunnel"):
        if optional in merged:
            result[optional] = merged[optional]
    require_columns(result.columns, TOKEN_REQUIRED_COLUMNS, product="micro_condition_tokens")
    return result.sort_values([*ORDER_KEYS, "route_sequence"], kind="stable", ignore_index=True)


def fit_train_cdf_thresholds(
    train_tokens: pd.DataFrame,
    *,
    quantile: float = 0.90,
) -> dict[str, Any]:
    """Freeze the empirical Train CDF cut corresponding to F_train(z)>=q."""
    if not 0 < quantile < 1:
        raise Stage2V52ContractError("CDF quantile must be between zero and one")
    thresholds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for name, column in DIMENSIONS.items():
        require_columns(train_tokens.columns, (column,), product="Train micro tokens")
        values = pd.to_numeric(train_tokens[column], errors="coerce").to_numpy(np.float64)
        values = values[np.isfinite(values)]
        if not len(values):
            raise Stage2V52ContractError(f"Train CDF has no valid {name} values")
        thresholds[name] = float(np.quantile(values, quantile, method="inverted_cdf"))
        counts[name] = int(len(values))
    return {
        "schema_version": "stage2_v5_2_train_micro_cdf.1",
        "fit_split": "train",
        "evaluation_rows_used": 0,
        "quantile": float(quantile),
        "thresholds": thresholds,
        "valid_counts": counts,
    }


def weighted_quantile_by_group(
    group_codes: np.ndarray,
    values: np.ndarray,
    weights: np.ndarray,
    group_count: int,
    *,
    quantile: float,
) -> np.ndarray:
    """Vectorized weighted empirical quantile with missing values preserved."""
    codes = np.asarray(group_codes, dtype=np.int64)
    value = np.asarray(values, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    valid = (codes >= 0) & np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    result = np.full(group_count, np.nan, dtype=np.float64)
    if not valid.any():
        return result
    order = np.lexsort((value[valid], codes[valid]))
    local_codes = codes[valid][order]
    local_values = value[valid][order]
    local_weights = weight[valid][order]
    cumulative = np.cumsum(local_weights)
    starts = np.r_[True, local_codes[1:] != local_codes[:-1]]
    base = np.zeros(group_count, dtype=np.float64)
    start_index = np.flatnonzero(starts)
    base[local_codes[start_index]] = np.where(start_index > 0, cumulative[start_index - 1], 0.0)
    total = np.bincount(local_codes, weights=local_weights, minlength=group_count)
    reached = cumulative - base[local_codes] >= quantile * total[local_codes]
    first = np.full(group_count, len(local_codes), dtype=np.int64)
    np.minimum.at(first, local_codes[reached], np.flatnonzero(reached))
    present = first < len(local_codes)
    result[present] = local_values[first[present]]
    return result


def _divide(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    return np.divide(numerator, denominator, out=np.full(len(numerator), np.nan), where=denominator > 0)


def _maximum_consecutive_share(
    codes: np.ndarray,
    high: np.ndarray,
    weights: np.ndarray,
    total_weight: np.ndarray,
) -> np.ndarray:
    group_count = len(total_weight)
    previous_group = np.r_[-1, codes[:-1]]
    previous_high = np.r_[False, high[:-1]]
    run_start = high & ((codes != previous_group) | ~previous_high)
    run_id = np.cumsum(run_start, dtype=np.int64) - 1
    maximum = np.zeros(group_count, dtype=np.float64)
    if high.any():
        run_frame = pd.DataFrame({"group": codes[high], "run": run_id[high], "weight": weights[high]})
        run_weights = run_frame.groupby(["group", "run"], sort=False, observed=True)["weight"].sum()
        group_max = run_weights.groupby(level=0, sort=False).max()
        maximum[group_max.index.to_numpy(np.int64)] = group_max.to_numpy(np.float64)
    return _divide(maximum, total_weight)


def aggregate_original_route_micro_conditions(
    tokens: pd.DataFrame,
    train_cdf: Mapping[str, Any],
    *,
    minimum_coverage: float = 0.80,
) -> pd.DataFrame:
    """Aggregate traversal predictions over each historical original route."""
    required = (
        *ORDER_KEYS, "route_sequence", "estimated_travel_time_p50_s", "allocated_distance_m",
        "edge_train_support", "support_group", *DIMENSIONS.values(),
    )
    require_columns(tokens.columns, required, product="micro tokens for route aggregation")
    if train_cdf.get("fit_split") != "train" or train_cdf.get("evaluation_rows_used") != 0:
        raise Stage2V52ContractError("high exposure requires a frozen Train-only CDF")
    thresholds = train_cdf.get("thresholds", {})
    working = tokens.sort_values([*ORDER_KEYS, "route_sequence"], kind="stable").reset_index(drop=True)
    grouped = working.groupby(list(ORDER_KEYS), sort=False, observed=True, dropna=False)
    codes = grouped.ngroup().to_numpy(np.int64)
    group_count = int(codes.max() + 1) if len(codes) else 0
    result = working.loc[:, ORDER_KEYS].groupby(codes, sort=False, observed=True).first().reset_index(drop=True)
    result["route_identity"] = "historical_original_service_route"
    time_weight = pd.to_numeric(working["estimated_travel_time_p50_s"], errors="coerce").to_numpy(np.float64)
    distance_weight = pd.to_numeric(working["allocated_distance_m"], errors="coerce").to_numpy(np.float64)
    physical_time = np.isfinite(time_weight) & (time_weight > 0)
    total_time = np.bincount(codes, weights=np.where(physical_time, time_weight, 0.0), minlength=group_count)
    result["travel_time_p50_s"] = total_time
    common_valid = physical_time.copy()
    for column in DIMENSIONS.values():
        common_valid &= np.isfinite(pd.to_numeric(working[column], errors="coerce").to_numpy(np.float64))
    covered = np.bincount(codes, weights=np.where(common_valid, time_weight, 0.0), minlength=group_count)
    result["micro_condition_coverage"] = _divide(covered, total_time)
    support = pd.to_numeric(working["edge_train_support"], errors="coerce").to_numpy(np.float64)
    support_valid = physical_time & np.isfinite(support)
    support_num = np.bincount(codes, weights=np.where(support_valid, support * time_weight, 0.0), minlength=group_count)
    support_den = np.bincount(codes, weights=np.where(support_valid, time_weight, 0.0), minlength=group_count)
    result["support_weighted_mean"] = _divide(support_num, support_den)
    groups = working["support_group"].astype(str).to_numpy()
    for name, mask in (("low_support_route_share", groups == "low"), ("unseen_edge_route_share", groups == "unseen")):
        numerator = np.bincount(codes, weights=np.where(physical_time & mask, time_weight, 0.0), minlength=group_count)
        result[name] = _divide(numerator, total_time)
    for name, column in DIMENSIONS.items():  # Fixed five-dimension loop, never per route.
        value = pd.to_numeric(working[column], errors="coerce").to_numpy(np.float64)
        valid = physical_time & np.isfinite(value)
        valid_weight = np.where(valid, time_weight, 0.0)
        denominator = np.bincount(codes, weights=valid_weight, minlength=group_count)
        numerator = np.bincount(codes, weights=np.where(valid, value * time_weight, 0.0), minlength=group_count)
        result[f"{name}_weighted_mean"] = _divide(numerator, denominator)
        result[f"{name}_weighted_p90"] = weighted_quantile_by_group(
            codes, value, valid_weight, group_count, quantile=0.90
        )
        if name not in thresholds:
            raise Stage2V52ContractError(f"Train CDF threshold missing for {name}")
        high = valid & (value >= float(thresholds[name]))
        high_weight = np.bincount(codes, weights=np.where(high, time_weight, 0.0), minlength=group_count)
        result[f"{name}_high_exposure_share"] = _divide(high_weight, denominator)
        if name in {"crawl", "stop"}:
            result[f"{name}_max_consecutive_high_share"] = _maximum_consecutive_share(
                codes, high, np.where(high, time_weight, 0.0), total_time
            )
        if name == "rts":
            distance_valid = np.isfinite(value) & np.isfinite(distance_weight) & (distance_weight > 0)
            distance = np.where(distance_valid, distance_weight, 0.0)
            distance_den = np.bincount(codes, weights=distance, minlength=group_count)
            distance_num = np.bincount(codes, weights=np.where(distance_valid, value * distance_weight, 0.0), minlength=group_count)
            result["rts_distance_weighted_mean"] = _divide(distance_num, distance_den)
            result["rts_distance_weighted_p90"] = weighted_quantile_by_group(
                codes, value, distance, group_count, quantile=0.90
            )
    result["unknown_flag"] = (
        ~np.isfinite(result["micro_condition_coverage"].to_numpy(float))
        | (result["micro_condition_coverage"].to_numpy(float) < minimum_coverage)
    )
    return result


def aggregate_static_route_complexity(tokens: pd.DataFrame) -> pd.DataFrame:
    """Build a separate product; unavailable dimensions remain NA, never zero."""
    require_columns(
        tokens.columns,
        (*ORDER_KEYS, "route_sequence", "allocated_distance_m", "road_class", "bridge", "tunnel"),
        product="static complexity input",
    )
    working = tokens.sort_values([*ORDER_KEYS, "route_sequence"], kind="stable").reset_index(drop=True)
    grouped = working.groupby(list(ORDER_KEYS), sort=False, observed=True, dropna=False)
    codes = grouped.ngroup().to_numpy(np.int64)
    group_count = int(codes.max() + 1) if len(codes) else 0
    result = working.loc[:, ORDER_KEYS].groupby(codes, sort=False, observed=True).first().reset_index(drop=True)
    distance = pd.to_numeric(working["allocated_distance_m"], errors="coerce").to_numpy(np.float64)
    physical = np.isfinite(distance) & (distance > 0)
    total = np.bincount(codes, weights=np.where(physical, distance, 0.0), minlength=group_count)
    for source, output in (("bridge", "bridge_exposure_share"), ("tunnel", "tunnel_exposure_share")):
        values = working[source].astype("boolean")
        known = values.notna().to_numpy() & physical
        exposed = values.fillna(False).to_numpy(bool) & known
        known_weight = np.bincount(codes, weights=np.where(known, distance, 0.0), minlength=group_count)
        exposed_weight = np.bincount(codes, weights=np.where(exposed, distance, 0.0), minlength=group_count)
        result[output] = _divide(exposed_weight, known_weight)
    road_class = working["road_class"].astype("string")
    known = road_class.notna().to_numpy()
    same_group = codes == np.r_[-1, codes[:-1]]
    prior_known = np.r_[False, known[:-1]]
    transition_eligible = same_group & known & prior_known
    changed = transition_eligible & (road_class.astype(str).to_numpy() != np.r_["", road_class.astype(str).to_numpy()[:-1]])
    eligible_count = np.bincount(codes, weights=transition_eligible.astype(float), minlength=group_count)
    changed_count = np.bincount(codes, weights=changed.astype(float), minlength=group_count)
    result["road_class_transition_rate"] = _divide(changed_count, eligible_count)
    for column in (
        "intersection_exposure_share", "signal_exposure_share", "merge_exposure_share",
        "turn_exposure_share", "ramp_exposure_share",
    ):
        result[column] = pd.Series(pd.array([pd.NA] * group_count, dtype="Float64"))
    result["static_field_status"] = (
        "bridge,tunnel,road_class=AVAILABLE; intersection,signal,merge,turn,ramp=NA_UPSTREAM_UNAVAILABLE"
    )
    require_columns(result.columns, STATIC_COMPLEXITY_COLUMNS, product="static route complexity")
    return result


def write_partition_products(
    token_frame: pd.DataFrame,
    route_frame: pd.DataFrame,
    static_frame: pd.DataFrame,
    *,
    output_root: str | Path,
) -> dict[str, Any]:
    """Atomically write one bounded split/date partition and its manifest."""
    for frame, name in ((token_frame, "tokens"), (route_frame, "routes"), (static_frame, "static")):
        require_columns(frame.columns, ("split", "date"), product=name)
        if frame[["split", "date"]].drop_duplicates().shape[0] != 1:
            raise Stage2V52ContractError("writer accepts exactly one split/date partition")
    split = str(token_frame["split"].iloc[0])
    date = str(token_frame["date"].iloc[0])
    root = Path(output_root)
    paths = {
        "micro_condition_tokens": root / "micro_condition_tokens" / f"split={split}" / f"date={date}" / "part.parquet",
        "original_route_micro_conditions": root / "original_route_micro_conditions" / f"split={split}" / f"date={date}" / "part.parquet",
        "static_route_complexity": root / "static_route_complexity" / f"split={split}" / f"date={date}" / "part.parquet",
    }
    for path, frame in zip(paths.values(), (token_frame, route_frame, static_frame)):
        _atomic_parquet(path, frame)
    manifest = {
        "schema_version": "stage2_v5_2_micro_partition.1",
        "split": split,
        "date": date,
        "route_semantics": "historical_original_service_route",
        "row_counts": {
            "micro_condition_tokens": int(len(token_frame)),
            "original_route_micro_conditions": int(len(route_frame)),
            "static_route_complexity": int(len(static_frame)),
        },
        "files": {name: {"path": path.as_posix(), "sha256": _sha256_file(path)} for name, path in paths.items()},
    }
    _atomic_json(root / "manifests" / f"split={split}" / f"date={date}.json", manifest)
    return manifest
