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
CORE_DEPLOYABLE_DIMENSIONS = {
    "crawl": "pred_crawl_share",
    "stop": "pred_stop_share",
    "speed_cv": "pred_speed_cv_bounded",
    "acceleration": "pred_acceleration_rms_bounded",
}
DIAGNOSTIC_DIMENSIONS = {
    "rts": "pred_rts_raw",
}
DIMENSIONS = {**CORE_DEPLOYABLE_DIMENSIONS, **DIAGNOSTIC_DIMENSIONS}
ALLOWED_ROUTE_PROVENANCE = frozenset({
    (
        "revealed_route_proxy",
        "frozen_stage2_v4_revealed_route_proxy",
        "stage2_v4_route_conditioned_dataset.1",
    ),
    (
        "historical_original_service_route",
        "frozen_stage1_route_parts",
        "stage1_v3_route_sequence_context.1",
    ),
})


def _canonical_hash(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


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
    protocol_id: str,
    prediction_source: str,
    model_id: str,
    model_hash: str,
) -> pd.DataFrame:
    """Create one formal prediction row per physical traversal."""
    join = [*ORDER_KEYS, "traversal_id"]
    require_columns(predictions.columns, (*join, "allocated_distance_m"), product="traversal predictions")
    require_columns(route_context.columns, join, product="route context")
    context_required = (
        *join, "route_sequence", "observed_directed_edge_uid", "canonical_edge_uid",
        "decision_time", "feature_cutoff_time", "route_track", "route_source", "route_product_version",
    )
    require_columns(route_context.columns, context_required, product="route context")
    if predictions.duplicated(join).any() or route_context.duplicated(join).any():
        raise Stage2V52ContractError("token identity is not one-to-one")
    keep = list(dict.fromkeys([
        *context_required, "history_support", "observed_sec_per_m_profile_count",
        "feature_cutoff_time", "decision_time",
        "route_part_length_m", "canonical_highway", "road_class", "bridge", "tunnel",
    ]))
    keep = [column for column in keep if column in route_context.columns]
    merged = predictions.merge(route_context.loc[:, keep], on=join, how="left", validate="one_to_one")
    if merged["observed_directed_edge_uid"].isna().any():
        raise Stage2V52ContractError("prediction row is missing its original-route edge identity")
    provenance = merged[["route_track", "route_source", "route_product_version"]].astype(str)
    observed_provenance = {
        tuple(row) for row in provenance.drop_duplicates().to_numpy()
    }
    if len(observed_provenance) != 1 or not observed_provenance <= ALLOWED_ROUTE_PROVENANCE:
        raise Stage2V52ContractError(
            f"route context is not a frozen original-route product: {sorted(observed_provenance)}"
        )
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
    if "history_support" in merged:
        history_source = "history_support"
    elif "observed_sec_per_m_profile_count" in merged:
        history_source = "observed_sec_per_m_profile_count"
    else:
        raise Stage2V52ContractError("route context has no frozen history support field")
    result["history_support"] = pd.to_numeric(merged[history_source], errors="coerce").fillna(0).astype("int64")
    decision = pd.to_numeric(merged["decision_time"], errors="coerce")
    cutoff = pd.to_numeric(merged["feature_cutoff_time"], errors="coerce")
    age = decision - cutoff
    if decision.isna().any() or cutoff.isna().any() or (age <= 0).any():
        raise Stage2V52ContractError("every feature_cutoff_time must be strictly earlier than decision_time")
    support, group = lookup_train_support(result["observed_directed_edge_uid"], support_artifact)
    result["edge_train_support"] = support
    result["edge_seen_in_train"] = support > 0
    result["support_group"] = group
    result["protocol_id"] = str(protocol_id)
    result["prediction_source"] = str(prediction_source)
    result["model_id"] = str(model_id)
    result["model_hash"] = str(model_hash)
    result["decision_time"] = decision
    result["feature_cutoff_time"] = cutoff
    result["feature_age_s"] = age
    for column in ("route_track", "route_source", "route_product_version"):
        result[column] = merged[column].astype(str)
    for optional in ("route_part_length_m", "canonical_highway", "road_class", "bridge", "tunnel"):
        if optional in merged:
            result[optional] = merged[optional]
    require_columns(result.columns, TOKEN_REQUIRED_COLUMNS, product="micro_condition_tokens")
    return result.sort_values([*ORDER_KEYS, "route_sequence"], kind="stable", ignore_index=True)


def fit_train_cdf_thresholds(
    train_tokens: pd.DataFrame,
    *,
    protocol_id: str,
    protocol_train_dates: Sequence[str],
    input_sha256: str,
    quantile: float = 0.90,
) -> dict[str, Any]:
    """Freeze the empirical Train CDF cut corresponding to F_train(z)>=q."""
    if not 0 < quantile < 1:
        raise Stage2V52ContractError("CDF quantile must be between zero and one")
    metadata = ("split", "date", "protocol_id", "model_id", "prediction_source")
    require_columns(train_tokens.columns, metadata, product="Train micro CDF input")
    if not train_tokens["split"].astype(str).eq("train").all():
        raise Stage2V52ContractError("micro CDF input contains non-Train rows")
    observed_dates = tuple(sorted(train_tokens["date"].astype(str).unique()))
    expected_dates = tuple(sorted(str(value) for value in protocol_train_dates))
    if observed_dates != expected_dates:
        raise Stage2V52ContractError(
            f"micro CDF dates {observed_dates} differ from protocol Train dates {expected_dates}"
        )
    if set(train_tokens["protocol_id"].astype(str).unique()) != {str(protocol_id)}:
        raise Stage2V52ContractError("micro CDF protocol_id is mixed or incorrect")
    model_ids = tuple(sorted(train_tokens["model_id"].astype(str).unique()))
    sources = tuple(sorted(train_tokens["prediction_source"].astype(str).unique()))
    if len(model_ids) != 1 or len(sources) != 1:
        raise Stage2V52ContractError("micro CDF requires one model_id and prediction_source")
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
    payload = {
        "schema_version": "stage2_v5_2_train_micro_cdf.1",
        "fit_split": "train",
        "evaluation_rows_used": 0,
        "protocol_id": str(protocol_id),
        "fit_dates_observed": list(observed_dates),
        "model_id": model_ids[0],
        "prediction_source": sources[0],
        "input_sha256": str(input_sha256),
        "quantile": float(quantile),
        "thresholds": thresholds,
        "valid_counts": counts,
    }
    payload["artifact_sha256"] = _canonical_hash(payload)
    return payload


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
    service_time_complete_threshold: float = 0.999,
) -> pd.DataFrame:
    """Aggregate traversal predictions over each historical original route."""
    if not 0 < minimum_coverage <= service_time_complete_threshold <= 1:
        raise Stage2V52ContractError("coverage thresholds must satisfy 0 < minimum <= complete <= 1")
    required = (
        *ORDER_KEYS, "route_sequence", "estimated_travel_time_p50_s", "allocated_distance_m",
        "edge_train_support", "support_group", "protocol_id", "model_id", "prediction_source",
        "route_track", "route_source", "route_product_version", *DIMENSIONS.values(),
    )
    require_columns(tokens.columns, required, product="micro tokens for route aggregation")
    if train_cdf.get("fit_split") != "train" or train_cdf.get("evaluation_rows_used") != 0:
        raise Stage2V52ContractError("high exposure requires a frozen Train-only CDF")
    token_protocols = set(tokens["protocol_id"].astype(str).unique())
    if token_protocols != {str(train_cdf.get("protocol_id"))}:
        raise Stage2V52ContractError("route aggregation CDF protocol differs from token protocol")
    if set(tokens["model_id"].astype(str).unique()) != {str(train_cdf.get("model_id"))}:
        raise Stage2V52ContractError("route aggregation CDF model differs from token model")
    if set(tokens["prediction_source"].astype(str).unique()) != {
        str(train_cdf.get("prediction_source"))
    }:
        raise Stage2V52ContractError("route aggregation CDF source differs from token source")
    thresholds = train_cdf.get("thresholds", {})
    working = tokens.sort_values([*ORDER_KEYS, "route_sequence"], kind="stable").reset_index(drop=True)
    grouped = working.groupby(list(ORDER_KEYS), sort=False, observed=True, dropna=False)
    codes = grouped.ngroup().to_numpy(np.int64)
    group_count = int(codes.max() + 1) if len(codes) else 0
    result = working.loc[:, ORDER_KEYS].groupby(codes, sort=False, observed=True).first().reset_index(drop=True)
    for column in ("route_track", "route_source", "route_product_version"):
        result[column] = working[column].groupby(codes, sort=False, observed=True).first().to_numpy()
    result["route_identity"] = result["route_track"]
    time_weight = pd.to_numeric(working["estimated_travel_time_p50_s"], errors="coerce").to_numpy(np.float64)
    distance_weight = pd.to_numeric(working["allocated_distance_m"], errors="coerce").to_numpy(np.float64)
    physical_distance = np.isfinite(distance_weight) & (distance_weight > 0)
    total_distance = np.bincount(
        codes, weights=np.where(physical_distance, distance_weight, 0.0), minlength=group_count
    )
    result["route_total_distance_m"] = total_distance
    pace_valid = physical_distance & np.isfinite(time_weight) & (time_weight > 0)
    partial_time = np.bincount(codes, weights=np.where(pace_valid, time_weight, 0.0), minlength=group_count)
    pace_distance = np.bincount(
        codes, weights=np.where(pace_valid, distance_weight, 0.0), minlength=group_count
    )
    pace_coverage = _divide(pace_distance, total_distance)
    result["partial_travel_time_p50_s"] = partial_time
    result["pace_prediction_coverage_distance"] = pace_coverage
    result["travel_time_p50_s"] = np.where(pace_coverage >= minimum_coverage, partial_time, np.nan)
    result["service_time_complete_flag"] = pace_coverage >= service_time_complete_threshold
    result["service_time_status"] = np.where(
        result["service_time_complete_flag"], "complete",
        np.where(pace_coverage >= minimum_coverage, "partial_coverage_estimate", "unavailable"),
    )
    physical_time = pace_valid
    total_time = partial_time
    common_valid = pace_valid.copy()
    for column in CORE_DEPLOYABLE_DIMENSIONS.values():
        common_valid &= np.isfinite(pd.to_numeric(working[column], errors="coerce").to_numpy(np.float64))
    covered_distance = np.bincount(
        codes, weights=np.where(common_valid, distance_weight, 0.0), minlength=group_count
    )
    result["micro_prediction_coverage_distance"] = _divide(covered_distance, total_distance)
    result["micro_condition_coverage"] = result["micro_prediction_coverage_distance"]
    support = pd.to_numeric(working["edge_train_support"], errors="coerce").to_numpy(np.float64)
    support_valid = physical_distance & np.isfinite(support)
    support_num = np.bincount(codes, weights=np.where(support_valid, support * distance_weight, 0.0), minlength=group_count)
    support_den = np.bincount(codes, weights=np.where(support_valid, distance_weight, 0.0), minlength=group_count)
    result["support_weighted_mean"] = _divide(support_num, support_den)
    groups = working["support_group"].astype(str).to_numpy()
    for name, mask in (("low_support_route_share", groups == "low"), ("unseen_edge_route_share", groups == "unseen")):
        numerator = np.bincount(codes, weights=np.where(physical_distance & mask, distance_weight, 0.0), minlength=group_count)
        result[name] = _divide(numerator, total_distance)
    for name, column in DIMENSIONS.items():  # Fixed predicted-target loop, never per route.
        value = pd.to_numeric(working[column], errors="coerce").to_numpy(np.float64)
        valid = physical_time & np.isfinite(value)
        dimension_distance = np.bincount(
            codes,
            weights=np.where(physical_distance & np.isfinite(value), distance_weight, 0.0),
            minlength=group_count,
        )
        coverage_column = f"{name}_prediction_coverage"
        result[coverage_column] = _divide(dimension_distance, total_distance)
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
        insufficient = result[coverage_column].to_numpy(float) < minimum_coverage
        for output in (f"{name}_weighted_mean", f"{name}_weighted_p90", f"{name}_high_exposure_share"):
            result.loc[insufficient, output] = np.nan
        if name in {"crawl", "stop"}:
            result.loc[insufficient, f"{name}_max_consecutive_high_share"] = np.nan
        if name == "rts":
            result.loc[insufficient, ["rts_distance_weighted_mean", "rts_distance_weighted_p90"]] = np.nan
    result["unknown_flag"] = (
        ~np.isfinite(result["pace_prediction_coverage_distance"].to_numpy(float))
        | (result["pace_prediction_coverage_distance"].to_numpy(float) < minimum_coverage)
        | ~np.isfinite(result["micro_condition_coverage"].to_numpy(float))
        | (result["micro_condition_coverage"].to_numpy(float) < minimum_coverage)
    )
    return result


def aggregate_static_route_complexity(tokens: pd.DataFrame) -> pd.DataFrame:
    """Build a separate product; unavailable dimensions remain NA, never zero."""
    require_columns(
        tokens.columns,
        (
            *ORDER_KEYS, "route_sequence", "allocated_distance_m", "canonical_highway",
            "road_class", "bridge", "tunnel",
        ),
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
    road_class_text = road_class.astype(str).to_numpy()
    changed = transition_eligible & (
        road_class_text != np.concatenate((np.array([""], dtype=object), road_class_text[:-1]))
    )
    eligible_count = np.bincount(codes, weights=transition_eligible.astype(float), minlength=group_count)
    changed_count = np.bincount(codes, weights=changed.astype(float), minlength=group_count)
    result["road_class_transition_rate"] = _divide(changed_count, eligible_count)
    highway = working["canonical_highway"].astype("string")
    highway_known = highway.notna().to_numpy()
    prior_highway_known = np.r_[False, highway_known[:-1]]
    highway_transition_eligible = same_group & highway_known & prior_highway_known
    highway_text = highway.astype(str).to_numpy()
    highway_changed = highway_transition_eligible & (
        highway_text != np.concatenate((np.array([""], dtype=object), highway_text[:-1]))
    )
    highway_eligible_count = np.bincount(
        codes, weights=highway_transition_eligible.astype(float), minlength=group_count
    )
    highway_changed_count = np.bincount(
        codes, weights=highway_changed.astype(float), minlength=group_count
    )
    result["canonical_highway_transition_rate"] = _divide(
        highway_changed_count, highway_eligible_count
    )
    category_frame = pd.DataFrame({
        "group": codes[physical & highway_known],
        "highway": highway_text[physical & highway_known],
        "weight": distance[physical & highway_known],
    })
    entropy = np.full(group_count, np.nan, dtype=np.float64)
    if not category_frame.empty:
        category_weight = category_frame.groupby(
            ["group", "highway"], sort=False, observed=True
        )["weight"].sum().reset_index()
        category_total = category_weight.groupby("group", sort=False, observed=True)["weight"].transform("sum")
        probability = category_weight["weight"].to_numpy(float) / category_total.to_numpy(float)
        category_weight["entropy_term"] = -probability * np.log(probability)
        entropy_by_group = category_weight.groupby("group", sort=False, observed=True)["entropy_term"].sum()
        entropy[entropy_by_group.index.to_numpy(np.int64)] = entropy_by_group.to_numpy(float)
    result["canonical_highway_entropy"] = entropy
    normalized_highway = np.char.lower(highway_text.astype(str))
    for output, categories in (
        ("motorway_trunk_exposure_share", ("motorway", "motorway_link", "trunk", "trunk_link")),
        ("primary_secondary_exposure_share", ("primary", "primary_link", "secondary", "secondary_link")),
    ):
        exposed = physical & highway_known & np.isin(normalized_highway, categories)
        known_weight = np.bincount(
            codes, weights=np.where(physical & highway_known, distance, 0.0), minlength=group_count
        )
        exposed_weight = np.bincount(
            codes, weights=np.where(exposed, distance, 0.0), minlength=group_count
        )
        result[output] = _divide(exposed_weight, known_weight)
    for column in (
        "intersection_exposure_share", "signal_exposure_share", "merge_exposure_share",
        "turn_exposure_share", "ramp_exposure_share",
    ):
        result[column] = pd.Series(pd.array([pd.NA] * group_count, dtype="Float64"))
    result["static_field_status"] = (
        "canonical_highway,bridge,tunnel,road_class=AVAILABLE; "
        "intersection,signal,merge,turn,ramp=NA_SCHEMA_AUDITED_NO_STABLE_JOINABLE_FIELD"
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
        "status": "PASS",
        "split": split,
        "date": date,
        "route_semantics": "historical_original_service_route",
        "coverage_semantics": {
            "micro_condition_coverage": list(CORE_DEPLOYABLE_DIMENSIONS),
            "unknown_flag_micro_dimensions": list(CORE_DEPLOYABLE_DIMENSIONS),
            "diagnostic_dimensions_with_independent_coverage": list(DIAGNOSTIC_DIMENSIONS),
        },
        "product_schema_versions": {
            "micro_condition_tokens": "stage2_v5_2_micro_condition_tokens.2",
            "original_route_micro_conditions": "stage2_v5_2_original_route_micro_conditions.2",
            "static_route_complexity": "stage2_v5_2_static_route_complexity.2",
        },
        "route_source_hash": _canonical_hash(
            token_frame[["route_track", "route_source", "route_product_version"]]
            .drop_duplicates()
            .sort_values(["route_track", "route_source", "route_product_version"], kind="stable")
            .to_dict(orient="records")
        ),
        "row_counts": {
            "micro_condition_tokens": int(len(token_frame)),
            "original_route_micro_conditions": int(len(route_frame)),
            "static_route_complexity": int(len(static_frame)),
        },
        "files": {name: {"path": path.as_posix(), "sha256": _sha256_file(path)} for name, path in paths.items()},
    }
    _atomic_json(root / "manifests" / f"split={split}" / f"date={date}.json", manifest)
    return manifest
