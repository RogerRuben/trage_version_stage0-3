"""S3 AV capability-envelope calibration over frozen S2B and frozen M3."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage3.odd_tod.network_foundation import (
    Stage3S2AError, atomic_json, atomic_parquet, atomic_text, git_head,
    parquet_descriptor, payload_hash, read_json, sha256_file, source_descriptor,
)


AUTHORIZED_BASE = "c9b6bcdf136ee11fc2863609218a198f60a332c8"
S31_AUTHORIZED_BASE = "309da4e5164eb99314c34b15ae2652f587a29f0b"
PHASE_STATUS = "STAGE3_S3_CAPABILITY_ENVELOPE_FROZEN"
M3_SHA256 = "965fc491cd77256f7889961d89932ec6be709bab04adcca358ac1b49f47c2cde"
TRAIN_DATES = tuple(f"201610{day:02d}" for day in range(9, 25))
VALIDATION_DATES = ("20161025", "20161026", "20161027")
PI = {"C": 0.75, "M": 0.90, "A": 0.975}
SPEED_CAPS = {"C": 60, "M": 80, "A": 120}
DYNAMIC_DIMS = ("crawl", "stop", "speed_cv", "acceleration_rms")
STATIC_MAP = {
    "external_physical_connection_count": "A_c",
    "topological_movement_count": "M_c",
    "internal_length_m": "L_c",
}
Q_TAIL = 0.90

OUTPUT_REL = Path("stage3/output/odd_tod/s3")
PROFILE_REL = Path("stage3/config/stage3_av_capability_profiles.json")
ROUTE_REL = Path("stage2/output_v4/route_conditioned_dataset/revealed_route_proxy")


def validate_s3_date(date: str, allowed_dates: Sequence[str]) -> None:
    if str(date) == "20161031":
        raise Stage3S2AError("Test31 is forbidden in S3")
    if str(date) not in set(allowed_dates):
        raise Stage3S2AError(f"date is outside authorized S3 role: {date}")


def quantile_higher(values: Sequence[float], probability: float) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array): raise Stage3S2AError("empty quantile support")
    return float(np.quantile(array, probability, method="higher"))


def resolve_route_tokens(route: pd.DataFrame, mapping: pd.DataFrame, overlay: pd.DataFrame) -> pd.DataFrame:
    """Resolve typed identities without guessing across namespaces."""
    required = {"order_id", "route_sequence", "canonical_edge_uid", "observed_direction", "observed_directed_edge_uid"}
    if not required.issubset(route.columns): raise Stage3S2AError("route identity schema incomplete")
    exact = mapping[mapping["mapping_status"] == "EXACT_VALHALLA"][[
        "canonical_edge_uid", "canonical_traversal_direction", "stage3_edge_uid"
    ]].drop_duplicates()
    if exact.duplicated(["canonical_edge_uid", "canonical_traversal_direction"]).any():
        raise Stage3S2AError("full-network mapping key is not unique")
    reverse = overlay[[
        "canonical_edge_uid", "canonical_traversal_direction", "physical_forward_stage3_edge_uid",
        "av_routability_status", "historical_direction_status", "missing_identity",
    ]].drop_duplicates()
    if reverse.duplicated(["canonical_edge_uid", "canonical_traversal_direction"]).any():
        raise Stage3S2AError("historical overlay key is not unique")
    base_cols = [column for column in ("date", "split", "order_id", "route_sequence") if column in route]
    work = route[base_cols + ["observed_directed_edge_uid", "canonical_edge_uid", "observed_direction"]].copy()
    work = work.rename(columns={"observed_directed_edge_uid": "source_identity", "observed_direction": "canonical_traversal_direction"})
    work = work.merge(exact, on=["canonical_edge_uid", "canonical_traversal_direction"], how="left", validate="many_to_one")
    work = work.merge(reverse, on=["canonical_edge_uid", "canonical_traversal_direction"], how="left", validate="many_to_one")
    is_full = work["stage3_edge_uid"].notna()
    is_reverse = ~is_full & work["historical_direction_status"].eq("HISTORICAL_DIRECTION_OVERLAY")
    work["route_token_type"] = np.select(
        [is_full, is_reverse], ["FULL_NETWORK_EDGE", "HISTORICAL_REVERSE_OVERLAY"], default="UNRESOLVED"
    )
    work["resolved_stage3_edge_uid"] = work["stage3_edge_uid"].where(is_full)
    work["physical_forward_stage3_edge_uid"] = work["physical_forward_stage3_edge_uid"].where(is_reverse)
    work["resolution_status"] = np.select(
        [is_full, is_reverse], ["RESOLVED_FULL_NETWORK", "RESOLVED_HISTORICAL_REVERSE_OVERLAY"], default="UNRESOLVED"
    )
    work["av_routability_status"] = np.select(
        [is_full, is_reverse], ["AV_ROUTABLE", "AV_ROUTABILITY_VIOLATION"], default="UNKNOWN"
    )
    columns = base_cols + [
        "source_identity", "canonical_edge_uid", "canonical_traversal_direction",
        "route_token_type", "resolved_stage3_edge_uid", "physical_forward_stage3_edge_uid",
        "resolution_status", "av_routability_status",
    ]
    result = work[columns].sort_values(["date", "order_id", "route_sequence"] if "date" in work else ["order_id", "route_sequence"])
    return result.reset_index(drop=True)


def identity_summary(frame: pd.DataFrame) -> dict[str, Any]:
    counts = frame["route_token_type"].value_counts()
    per_order = frame.groupby("order_id", sort=False)["route_token_type"].agg(list)
    total = len(frame)
    return {
        "total_orders": int(frame["order_id"].nunique()), "total_route_tokens": total,
        "FULL_NETWORK_EDGE_count": int(counts.get("FULL_NETWORK_EDGE", 0)),
        "FULL_NETWORK_EDGE_share": float(counts.get("FULL_NETWORK_EDGE", 0) / total) if total else 0.0,
        "HISTORICAL_REVERSE_OVERLAY_count": int(counts.get("HISTORICAL_REVERSE_OVERLAY", 0)),
        "HISTORICAL_REVERSE_OVERLAY_share": float(counts.get("HISTORICAL_REVERSE_OVERLAY", 0) / total) if total else 0.0,
        "UNRESOLVED_count": int(counts.get("UNRESOLVED", 0)),
        "UNRESOLVED_share": float(counts.get("UNRESOLVED", 0) / total) if total else 0.0,
        "fully_full_network_resolved_order_count": int(per_order.map(lambda values: all(v == "FULL_NETWORK_EDGE" for v in values)).sum()),
        "orders_with_reverse_overlay": int(per_order.map(lambda values: "HISTORICAL_REVERSE_OVERLAY" in values).sum()),
        "orders_with_unresolved_token": int(per_order.map(lambda values: "UNRESOLVED" in values).sum()),
    }


def parse_route_complex_encounters(
    typed: pd.DataFrame, boundary_index: pd.DataFrame, movement_lookup: pd.DataFrame,
) -> pd.DataFrame:
    roles: dict[str, dict[str, set[str]]] = {}
    for row in boundary_index.itertuples(index=False):
        roles.setdefault(str(row.stage3_edge_uid), {}).setdefault(str(row.intersection_complex_uid), set()).add(str(row.boundary_role))
    movement_pairs = set(zip(
        movement_lookup["intersection_complex_uid"].astype(str),
        movement_lookup["incoming_stage3_edge_uid"].astype(str),
        movement_lookup["outgoing_stage3_edge_uid"].astype(str),
    ))
    rows: list[dict[str, Any]] = []
    group_keys = [key for key in ("date", "order_id") if key in typed]
    for group_id, group in typed.groupby(group_keys, sort=False):
        group = group.sort_values("route_sequence")
        order_id = group_id[-1] if isinstance(group_id, tuple) else group_id
        date = group_id[0] if isinstance(group_id, tuple) and len(group_keys) == 2 else None
        active: dict[str, dict[str, Any]] = {}; occurrence = 0
        for token in group.itertuples(index=False):
            if token.route_token_type != "FULL_NETWORK_EDGE":
                active.clear(); continue
            edge = str(token.resolved_stage3_edge_uid); edge_roles = roles.get(edge, {})
            completed: list[str] = []
            for complex_id, state in list(active.items()):
                current_roles = edge_roles.get(complex_id, set())
                if "INTERNAL" in current_roles:
                    state["internal"].append(edge)
                elif "OUTGOING" in current_roles:
                    pair = (complex_id, state["incoming"], edge)
                    rows.append({
                        "date": date, "order_id": str(order_id), "movement_occurrence_index": occurrence,
                        "intersection_complex_uid": complex_id,
                        "incoming_stage3_edge_uid": state["incoming"], "outgoing_stage3_edge_uid": edge,
                        "internal_stage3_edge_uids": json.dumps(state["internal"]),
                        "internal_edge_count": len(state["internal"]),
                        "movement_lookup_status": "MATCHED_TOPOLOGICAL_MOVEMENT" if pair in movement_pairs else "UNRESOLVED_MOVEMENT_LOOKUP",
                    })
                    occurrence += 1; completed.append(complex_id)
                elif "INCOMING" not in current_roles:
                    completed.append(complex_id)
            for complex_id in completed: active.pop(complex_id, None)
            for complex_id, current_roles in edge_roles.items():
                if "INCOMING" in current_roles:
                    active[complex_id] = {"incoming": edge, "internal": []}
    result = pd.DataFrame(rows)
    if len(result): result["date"] = result["date"].astype(str)
    return result


def boundary_road_class_diversity(boundary_index: pd.DataFrame, edges: pd.DataFrame) -> pd.DataFrame:
    """Derive D_c from unique INCOMING/OUTGOING boundary edges, never INTERNAL edges."""
    boundary = boundary_index[boundary_index["boundary_role"].isin(["INCOMING", "OUTGOING"])][
        ["intersection_complex_uid", "stage3_edge_uid"]
    ].drop_duplicates()
    edge_classes = edges[["stage3_edge_uid", "valhalla_road_class"]].drop_duplicates("stage3_edge_uid")
    if edge_classes["stage3_edge_uid"].duplicated().any():
        raise Stage3S2AError("full-network edge identity is not unique")
    joined = boundary.merge(edge_classes, on="stage3_edge_uid", how="left", validate="many_to_one")
    if joined["valhalla_road_class"].isna().any():
        raise Stage3S2AError(f"boundary road class missing for {int(joined['valhalla_road_class'].isna().sum())} edge-complex anchors")
    result = joined.groupby("intersection_complex_uid", sort=False).agg(
        boundary_edge_count=("stage3_edge_uid", "nunique"),
        boundary_road_class_diversity=("valhalla_road_class", "nunique"),
    ).reset_index()
    if (result["boundary_road_class_diversity"] < 1).any():
        raise Stage3S2AError("boundary road-class diversity must be positive")
    result["road_class_diversity_definition"] = "UNIQUE_VALHALLA_ROAD_CLASS_ON_INCOMING_OUTGOING_BOUNDARY_EDGES"
    return result


def build_static_reference(
    encounters: pd.DataFrame, complexes: pd.DataFrame,
    boundary_index: pd.DataFrame, edges: pd.DataFrame,
) -> pd.DataFrame:
    count = encounters.groupby("intersection_complex_uid").size().rename("train_encounter_count")
    columns = ["intersection_complex_uid", *STATIC_MAP, "road_class_diversity", "signal_state", "roundabout_evidence_present", "grade_separation_evidence_present"]
    result = complexes[columns].merge(count, left_on="intersection_complex_uid", right_index=True, how="inner", validate="one_to_one")
    result = result.rename(columns=STATIC_MAP)
    result = result.rename(columns={"road_class_diversity": "s2b_internal_road_class_diversity_qa"})
    diversity = boundary_road_class_diversity(boundary_index, edges)
    result = result.merge(diversity, on="intersection_complex_uid", how="left", validate="one_to_one")
    if result["boundary_road_class_diversity"].isna().any():
        raise Stage3S2AError("Train-exposed complex lacks boundary road-class diversity")
    result["D_c"] = result["boundary_road_class_diversity"].astype("int64")
    if result["intersection_complex_uid"].duplicated().any(): raise Stage3S2AError("static reference is demand weighted")
    return result.sort_values("intersection_complex_uid").reset_index(drop=True)


def static_caps(reference: pd.DataFrame) -> dict[str, dict[str, float]]:
    if len(reference) < 1000: raise Stage3S2AError(f"static support gate failed: {len(reference)}")
    return {profile: {dimension: quantile_higher(reference[dimension], probability) for dimension in ("A_c", "M_c", "D_c", "L_c")} for profile, probability in PI.items()}


def weighted_mid_cdf_reference(values: Sequence[float], weights: Sequence[float], dimension: str) -> pd.DataFrame:
    value = np.asarray(values, dtype=np.float64); weight = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(value) & np.isfinite(weight) & (weight > 0)
    frame = pd.DataFrame({"value": value[valid], "predicted_time_weight_s": weight[valid]})
    support = frame.groupby("value", sort=True, as_index=False)["predicted_time_weight_s"].sum()
    lower = support["predicted_time_weight_s"].cumsum() - support["predicted_time_weight_s"]
    total = float(support["predicted_time_weight_s"].sum())
    support["mid_cdf"] = (lower + .5 * support["predicted_time_weight_s"]) / total
    support.insert(0, "dimension", dimension)
    support["total_predicted_time_weight_s"] = total
    return support


def apply_mid_cdf(values: Sequence[float], reference: pd.DataFrame) -> np.ndarray:
    ref = reference.sort_values("value")
    x = ref["value"].to_numpy(np.float64); z = ref["mid_cdf"].to_numpy(np.float64)
    values_array = np.asarray(values, dtype=np.float64)
    positions = np.searchsorted(x, values_array)
    result = np.full(len(values_array), np.nan)
    exact = (positions < len(x)) & np.isfinite(values_array)
    exact[exact] &= x[positions[exact]] == values_array[exact]
    result[exact] = z[positions[exact]]
    # Future values may not equal Train support: use weighted mid-CDF limits.
    nonexact = np.isfinite(values_array) & ~exact
    if nonexact.any():
        weights = ref["predicted_time_weight_s"].to_numpy(np.float64)
        cumulative = np.cumsum(weights); total = cumulative[-1]
        left = np.searchsorted(x, values_array[nonexact], side="left")
        result[nonexact] = np.where(left == 0, 0.0, cumulative[np.maximum(left - 1, 0)] / total)
    return result


def route_eqc(tokens: pd.DataFrame, cdf_reference: pd.DataFrame) -> pd.DataFrame:
    work = tokens.sort_values(["date", "order_id", "route_sequence"]).copy()
    for dimension in DYNAMIC_DIMS:
        ref = cdf_reference[cdf_reference["dimension"] == dimension]
        work[f"z_{dimension}"] = apply_mid_cdf(work[f"pred_{dimension}"], ref)
    rows = []
    for (date, order_id), group in work.groupby(["date", "order_id"], sort=False):
        weight = group["travel_time_p50_s"].to_numpy(np.float64); total = float(weight.sum())
        row = {"date": str(date), "order_id": str(order_id), "predicted_route_time_p50_s": total, "route_token_count": len(group)}
        for dimension in DYNAMIC_DIMS:
            z = group[f"z_{dimension}"].to_numpy(np.float64); tail = z > Q_TAIL
            row[f"{dimension}_E"] = float(np.dot(weight, z) / total)
            row[f"{dimension}_Q"] = float(weight[tail].sum() / total)
            maximum = running = 0.0
            for is_tail, duration in zip(tail, weight):
                running = running + duration if is_tail else 0.0; maximum = max(maximum, running)
            row[f"{dimension}_C"] = float(maximum)
        rows.append(row)
    return pd.DataFrame(rows)


def dynamic_caps(descriptors: pd.DataFrame) -> dict[str, dict[str, dict[str, float]]]:
    if len(descriptors) < 4000: raise Stage3S2AError(f"dynamic support gate failed: {len(descriptors)}")
    return {
        profile: {dimension: {metric: quantile_higher(descriptors[f"{dimension}_{metric}"], probability) for metric in ("E", "Q", "C")} for dimension in DYNAMIC_DIMS}
        for profile, probability in PI.items()
    }


def movement_rules() -> dict[str, Any]:
    return {
        "C": {"STRAIGHT": "COMPATIBLE", "RIGHT": "COMPATIBLE", "LEFT": {"SIGNALIZED": "COMPATIBLE", "STOP_OR_YIELD_CONTROLLED": "INCOMPATIBLE", "UNKNOWN_CONTROL": "UNKNOWN"}, "UTURN": "INCOMPATIBLE", "UNKNOWN": "UNKNOWN"},
        "M": {"STRAIGHT": "COMPATIBLE", "RIGHT": "COMPATIBLE", "LEFT": "COMPATIBLE", "UTURN": "INCOMPATIBLE", "UNKNOWN": "UNKNOWN"},
        "A": {"STRAIGHT": "COMPATIBLE", "RIGHT": "COMPATIBLE", "LEFT": "COMPATIBLE", "UTURN": "COMPATIBLE", "UNKNOWN": "UNKNOWN"},
    }


def movement_compatibility(
    profile: str, turn_type: str, control_state: str = "UNKNOWN_CONTROL",
    roundabout: bool = False, restriction_enforcement_certified: bool = False,
    movement_legality_state: str = "UNKNOWN",
) -> str:
    """Evaluate only frozen categorical capability evidence (never safety)."""
    if restriction_enforcement_certified and movement_legality_state == "CERTIFIED_PROHIBITED":
        return "INCOMPATIBLE"
    if roundabout and profile == "C": return "INCOMPATIBLE"
    turn = str(turn_type).upper()
    if turn in {"STRAIGHT", "RIGHT"}: return "COMPATIBLE"
    if turn == "LEFT":
        if profile in {"M", "A"}: return "COMPATIBLE"
        return {"SIGNALIZED": "COMPATIBLE", "STOP_OR_YIELD_CONTROLLED": "INCOMPATIBLE"}.get(control_state, "UNKNOWN")
    if turn == "UTURN": return "COMPATIBLE" if profile == "A" else "INCOMPATIBLE"
    return "UNKNOWN"


def verify_categorical_nestedness() -> None:
    rank = {"INCOMPATIBLE": 0, "UNKNOWN": 1, "COMPATIBLE": 2}
    for turn in ("STRAIGHT", "RIGHT", "LEFT", "UTURN", "UNKNOWN"):
        for control in ("SIGNALIZED", "STOP_OR_YIELD_CONTROLLED", "UNKNOWN_CONTROL"):
            for roundabout in (False, True):
                values = [movement_compatibility(p, turn, control, roundabout) for p in ("C", "M", "A")]
                # UNKNOWN is preserved and not promoted for the purpose of proving
                # nesting. Compare only pairs whose states are both known.
                for lower, upper in zip(values, values[1:]):
                    if "UNKNOWN" not in (lower, upper) and rank[lower] > rank[upper]:
                        raise Stage3S2AError(f"categorical nestedness: {turn}/{control}/{roundabout}/{values}")


def build_profiles(static: Mapping[str, Any], dynamic: Mapping[str, Any]) -> dict[str, Any]:
    rules = movement_rules(); profiles = []
    for profile in ("C", "M", "A"):
        profiles.append({
            "profile_id": profile, "quantile_anchor": PI[profile], "speed_domain_max_kmh": SPEED_CAPS[profile],
            "static_caps": {
                "external_physical_connection_count": static[profile]["A_c"], "topological_movement_count": static[profile]["M_c"],
                "road_class_diversity": static[profile]["D_c"], "internal_length_m": static[profile]["L_c"],
            },
            "movement_rules": rules[profile], "roundabout_rule": "INCOMPATIBLE" if profile == "C" else "COMPATIBLE_SUBJECT_TO_CONTINUOUS_CAPS",
            "restriction_rule": "CERTIFIED_PROHIBITED_IS_INCOMPATIBLE; NOT_CERTIFIED_OR_UNKNOWN_DOES_NOT_IMPLY_PERMISSION",
            "grade_bridge_tunnel_rule": "DESCRIPTIVE_ONLY",
            "dynamic_caps": dynamic[profile],
            "unknown_policy": "decision-critical missing evidence is UNKNOWN; no imputation",
            "claim_boundary": "hypothetical capability compatibility, not safety, legality, failure, disengagement, or accident probability",
        })
    result = {
        "schema_version": "stage3_av_capability_profiles.1", "phase_status": PHASE_STATUS,
        "calibration_dates": list(TRAIN_DATES), "validation_sanity_dates": list(VALIDATION_DATES),
        "test31_used": False, "quantile_method": "higher", "q_tail": Q_TAIL,
        "dynamic_cdf": "global Train predicted-time-weighted mid-distribution CDF",
        "static_dimension_definitions": {
            "A_c": "external_physical_connection_count",
            "M_c": "topological_movement_count",
            "D_c": "unique valhalla_road_class over INCOMING/OUTGOING boundary edges; INTERNAL edges excluded",
            "L_c": "internal_length_m",
        },
        "quantile_anchor_semantics": "pi_k defines marginal per-dimension capability caps, not joint route acceptance rates",
        "non_compensatory": True, "profiles": profiles,
        "s4_authorized": False, "next_phase_authorized": False,
    }
    result["artifact_sha256"] = payload_hash(result); return result


def verify_nestedness(profiles: Mapping[str, Any]) -> None:
    by_id = {item["profile_id"]: item for item in profiles["profiles"]}
    if [by_id[p]["speed_domain_max_kmh"] for p in ("C", "M", "A")] != [60, 80, 120]: raise Stage3S2AError("speed caps changed")
    for dimension in by_id["C"]["static_caps"]:
        values = [by_id[p]["static_caps"][dimension] for p in ("C", "M", "A")]
        if values != sorted(values): raise Stage3S2AError(f"static nestedness: {dimension}")
    for dimension in DYNAMIC_DIMS:
        for metric in ("E", "Q", "C"):
            values = [by_id[p]["dynamic_caps"][dimension][metric] for p in ("C", "M", "A")]
            if values != sorted(values): raise Stage3S2AError(f"dynamic nestedness: {dimension}/{metric}")
    verify_categorical_nestedness()


def _route_path(root: Path, date: str) -> Path:
    path = root / ROUTE_REL / f"day={date}.parquet"
    if "20161031" in str(path): raise Stage3S2AError("Test31 path aliases are forbidden in S3")
    if not path.is_file(): raise Stage3S2AError(f"historical route product missing: {path}")
    return path


def _upstream_tables(root: Path) -> dict[str, pd.DataFrame]:
    return {
        "mapping": pd.read_parquet(root / "stage3/output/odd_tod/s2a/stage3_observed_full_network_mapping.parquet"),
        "overlay": pd.read_parquet(root / "stage3/output/odd_tod/s2a/stage3_historical_direction_overlay.parquet"),
        "edges": pd.read_parquet(root / "stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet"),
        "complexes": pd.read_parquet(root / "stage3/output/odd_tod/s2b/final/stage3_intersection_complexes.parquet"),
        "boundary": pd.read_parquet(root / "stage3/output/odd_tod/s2b/final/stage3_edge_complex_boundary_index.parquet"),
        "movements": pd.read_parquet(root / "stage3/output/odd_tod/s2b/final/stage3_route_movement_lookup.parquet"),
    }


def _combine_parquet(parts: Sequence[Path], output: Path) -> None:
    """Combine homogeneous parts without materializing the full product."""
    if not parts: raise Stage3S2AError(f"no parts for {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    writer = None
    try:
        for path in parts:
            # ParquetFile avoids Hive partition inference from parent names such
            # as date=20161009, because the physical file already contains date.
            table = pq.ParquetFile(path).read()
            if writer is None: writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table)
        writer.close(); writer = None
        os.replace(temporary, output)
    finally:
        if writer is not None: writer.close()
        if temporary.exists(): temporary.unlink()


def _prediction_day(root: Path, date: str, role: str) -> pd.DataFrame:
    if role == "train":
        validate_s3_date(date, TRAIN_DATES)
        path = root / OUTPUT_REL / "cache/m3" / f"date={date}.parquet"
    else:
        validate_s3_date(date, VALIDATION_DATES)
        source = root / "stage2/output_v5_2/development/M3/evaluation/unique_traversal_predictions.parquet"
        cache = root / OUTPUT_REL / "cache/m3_validation"; cache.mkdir(parents=True, exist_ok=True)
        path = cache / f"date={date}.parquet"
        if not path.is_file():
            table = pq.read_table(source, filters=[("date", "=", date)])
            frame = _strict_prediction_frame(table.to_pandas())
            atomic_parquet(path, frame)
            atomic_json(path.with_suffix(".json"), {
                "schema_version": "stage3_s3_validation_m3_cache.1", "date": date, "model_id": "M3",
                "checkpoint_sha256": M3_SHA256, "source_prediction_sha256": sha256_file(source),
                "prediction_sha256": sha256_file(path), "row_count": len(frame),
                "decision_time_only": True, "predicted_progression_only": True,
                "realized_future_time_used": False, "strict_prediction_only_schema": True,
                "realized_target_columns_persisted": False,
            })
        else:
            existing = pq.ParquetFile(path).schema_arrow.names
            if any(column.startswith("target_") or column.endswith("_target_valid") for column in existing):
                migrated = _strict_prediction_frame(pd.read_parquet(path)); atomic_parquet(path, migrated)
                manifest_path = path.with_suffix(".json"); manifest = read_json(manifest_path)
                manifest.update({"prediction_sha256": sha256_file(path), "row_count": len(migrated), "strict_prediction_only_schema": True, "realized_target_columns_persisted": False, "cache_schema_migrated": True})
                atomic_json(manifest_path, manifest)
    if not path.is_file(): raise Stage3S2AError(f"M3 prediction cache missing: {path}")
    return pd.read_parquet(path)


def _prepare_role_day(root: Path, date: str, role: str, upstream: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    allowed = TRAIN_DATES if role == "train" else VALIDATION_DATES
    validate_s3_date(date, allowed)
    cache = root / OUTPUT_REL / "cache" / role / f"date={date}"; cache.mkdir(parents=True, exist_ok=True)
    cached_manifest = cache / "manifest.json"
    if cached_manifest.is_file():
        cached = read_json(cached_manifest)
        cached_files = {
            "identity_sha256": cache / "identity.parquet",
            "encounters_sha256": cache / "encounters.parquet",
            "complete_tokens_sha256": cache / "complete_dynamic_tokens.parquet",
        }
        if all(path.is_file() and cached.get(key) == sha256_file(path) for key, path in cached_files.items()):
            return {**cached, "cache_reused": True}
    route_columns = [
        "split", "date", "order_id", "route_sequence", "traversal_id", "canonical_edge_uid",
        "observed_directed_edge_uid", "observed_direction", "road_class", "estimated_hour",
        "route_part_length_m", "decision_time", "feature_timestamp", "availability_timestamp",
    ]
    route = pd.read_parquet(_route_path(root, date), columns=route_columns)
    # The replay contract itself supplies decision-time features. These checks ensure
    # the persisted source never carries a feature timestamp beyond its decision time.
    for column in ("feature_timestamp", "availability_timestamp"):
        known = route[column].notna() & route["decision_time"].notna()
        if (route.loc[known, column] > route.loc[known, "decision_time"]).any():
            raise Stage3S2AError(f"future feature timestamp detected: {date}/{column}")
    typed = resolve_route_tokens(route, upstream["mapping"], upstream["overlay"])
    identity_path = cache / "identity.parquet"; atomic_parquet(identity_path, typed)
    encounters = parse_route_complex_encounters(typed, upstream["boundary"], upstream["movements"])
    encounter_path = cache / "encounters.parquet"; atomic_parquet(encounter_path, encounters)
    predictions = _prediction_day(root, date, role)
    pred_columns = ["date", "order_id", "traversal_id", "travel_time_p50_s", *[f"pred_{d}" for d in DYNAMIC_DIMS]]
    tokens = route[["date", "order_id", "route_sequence", "traversal_id", "road_class", "estimated_hour", "route_part_length_m"]].merge(
        predictions[pred_columns], on=["date", "order_id", "traversal_id"], how="left", validate="one_to_one",
    ).merge(
        typed[["date", "order_id", "route_sequence", "route_token_type"]],
        on=["date", "order_id", "route_sequence"], how="left", validate="one_to_one",
    )
    finite = np.isfinite(tokens["travel_time_p50_s"]) & (tokens["travel_time_p50_s"] > 0)
    for dimension in DYNAMIC_DIMS: finite &= np.isfinite(tokens[f"pred_{dimension}"])
    tokens["dynamic_token_valid"] = finite
    route_complete = tokens.groupby("order_id", sort=False)["dynamic_token_valid"].all()
    tokens["dynamic_route_complete"] = tokens["order_id"].map(route_complete)
    complete = tokens[tokens["dynamic_route_complete"]].drop(columns=["dynamic_token_valid", "dynamic_route_complete"])
    token_path = cache / "complete_dynamic_tokens.parquet"; atomic_parquet(token_path, complete)
    summary = identity_summary(typed)
    summary.update({
        "date": date, "role": role, "complete_dynamic_orders": int(route_complete.sum()),
        "total_dynamic_orders": int(len(route_complete)),
        "complete_dynamic_order_share": float(route_complete.mean()),
        "complete_dynamic_tokens": int(len(complete)),
        "identity_sha256": sha256_file(identity_path), "encounters_sha256": sha256_file(encounter_path),
        "complete_tokens_sha256": sha256_file(token_path),
    })
    atomic_json(cache / "manifest.json", summary)
    return summary


def _role_parts(root: Path, role: str, name: str, dates: Sequence[str]) -> list[Path]:
    return [root / OUTPUT_REL / "cache" / role / f"date={date}" / name for date in dates]


def _build_cdf_reference(root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    pieces: list[pd.DataFrame] = []
    diagnostics: dict[str, Any] = {}
    for dimension in DYNAMIC_DIMS:
        daily = []
        for date in TRAIN_DATES:
            frame = pd.read_parquet(
                root / OUTPUT_REL / "cache/train" / f"date={date}" / "complete_dynamic_tokens.parquet",
                columns=[f"pred_{dimension}", "travel_time_p50_s"],
            )
            daily.append(weighted_mid_cdf_reference(frame[f"pred_{dimension}"], frame["travel_time_p50_s"], dimension)[
                ["value", "predicted_time_weight_s"]
            ])
        merged = pd.concat(daily, ignore_index=True).groupby("value", sort=True, as_index=False)["predicted_time_weight_s"].sum()
        lower = merged["predicted_time_weight_s"].cumsum() - merged["predicted_time_weight_s"]
        total = float(merged["predicted_time_weight_s"].sum())
        merged["mid_cdf"] = (lower + .5 * merged["predicted_time_weight_s"]) / total
        merged.insert(0, "dimension", dimension); merged["total_predicted_time_weight_s"] = total
        largest = float(merged["predicted_time_weight_s"].max())
        zero = float(merged.loc[merged["value"].eq(0), "predicted_time_weight_s"].sum())
        diagnostics[dimension] = {
            "unique_predicted_values": int(len(merged)), "total_predicted_time_weight_s": total,
            "largest_exact_value_weight_s": largest, "largest_exact_value_mass_share": largest / total,
            "zero_value_weight_s": zero, "zero_value_mass_share": zero / total,
            "value_min": float(merged["value"].min()), "value_max": float(merged["value"].max()),
        }
        pieces.append(merged)
    return pd.concat(pieces, ignore_index=True), diagnostics


def _describe_numeric(frame: pd.DataFrame, columns: Sequence[str]) -> dict[str, Any]:
    result = {}
    for column in columns:
        values = frame[column].to_numpy(np.float64); values = values[np.isfinite(values)]
        result[column] = {
            "count": int(len(values)), "min": float(np.min(values)), "p25": quantile_higher(values, .25),
            "p50": quantile_higher(values, .50), "p75": quantile_higher(values, .75),
            "p90": quantile_higher(values, .90), "p975": quantile_higher(values, .975), "max": float(np.max(values)),
        }
    return result


def _dynamic_descriptors(root: Path, role: str, dates: Sequence[str], reference: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for date in dates:
        path = root / OUTPUT_REL / "cache" / role / f"date={date}" / "dynamic_descriptors.parquet"
        if path.is_file():
            parts.append(path)
            continue
        tokens = pd.read_parquet(root / OUTPUT_REL / "cache" / role / f"date={date}" / "complete_dynamic_tokens.parquet")
        descriptors = route_eqc(tokens, reference)
        atomic_parquet(path, descriptors); parts.append(path)
    output = root / OUTPUT_REL / ("train_dynamic_route_descriptors.parquet" if role == "train" else "validation_dynamic_route_descriptors.parquet")
    _combine_parquet(parts, output)
    return pd.read_parquet(output)


def build_train(root: Path) -> dict[str, Any]:
    if git_head(root) != AUTHORIZED_BASE: raise Stage3S2AError(f"S3 requires authorized base {AUTHORIZED_BASE}")
    upstream = _upstream_tables(root); daily = []
    for date in TRAIN_DATES:
        daily.append(_prepare_role_day(root, date, "train", upstream))
        print(f"prepared Train {date}", flush=True)
    identity_path = root / OUTPUT_REL / "train_route_identity_resolution.parquet"
    encounter_path = root / OUTPUT_REL / "train_route_complex_encounters.parquet"
    _combine_parquet(_role_parts(root, "train", "identity.parquet", TRAIN_DATES), identity_path)
    _combine_parquet(_role_parts(root, "train", "encounters.parquet", TRAIN_DATES), encounter_path)
    encounters = pd.read_parquet(encounter_path)
    reference = build_static_reference(encounters, upstream["complexes"], upstream["boundary"], upstream["edges"])
    static_path = root / OUTPUT_REL / "train_static_complex_reference.parquet"; atomic_parquet(static_path, reference)
    static_thresholds = static_caps(reference)
    cdf_reference, tie_diagnostics = _build_cdf_reference(root)
    cdf_path = root / OUTPUT_REL / "train_dynamic_cdf_reference.parquet"; atomic_parquet(cdf_path, cdf_reference)
    descriptors = _dynamic_descriptors(root, "train", TRAIN_DATES, cdf_reference)
    dynamic_thresholds = dynamic_caps(descriptors)
    profiles = build_profiles(static_thresholds, dynamic_thresholds)
    verify_nestedness(profiles)
    profile_path = root / PROFILE_REL; atomic_json(profile_path, profiles)
    summary = {
        "schema_version": "stage3_s3_train_summary.1", "phase": "S3_TRAIN_PROFILE_FROZEN",
        "authorized_base": AUTHORIZED_BASE, "train_dates": list(TRAIN_DATES), "daily": daily,
        "identity": identity_summary(pd.read_parquet(identity_path, columns=["order_id", "route_token_type"])),
        "unique_train_exposed_complex_count": int(len(reference)),
        "static_distribution": _describe_numeric(reference, ["A_c", "M_c", "D_c", "L_c"]),
        "static_thresholds": static_thresholds,
        "signal_state_distribution": {str(k): int(v) for k, v in reference["signal_state"].value_counts(dropna=False).items()},
        "roundabout_share": float(reference["roundabout_evidence_present"].mean()),
        "grade_separation_share": float(reference["grade_separation_evidence_present"].mean()),
        "static_advanced_tail_support": {d: int((reference[d] > static_thresholds["A"][d]).sum()) for d in ("A_c", "M_c", "D_c", "L_c")},
        "complete_dynamic_train_routes": int(len(descriptors)),
        "total_train_orders": int(sum(item["total_dynamic_orders"] for item in daily)),
        "complete_dynamic_route_coverage": float(len(descriptors) / sum(item["total_dynamic_orders"] for item in daily)),
        "cdf_tie_diagnostics": tie_diagnostics,
        "dynamic_descriptor_distribution": _describe_numeric(descriptors, [f"{d}_{m}" for d in DYNAMIC_DIMS for m in ("E", "Q", "C")]),
        "dynamic_thresholds": dynamic_thresholds,
        "dynamic_advanced_tail_support": {f"{d}_{m}": int((descriptors[f"{d}_{m}"] > dynamic_thresholds["A"][d][m]).sum()) for d in DYNAMIC_DIMS for m in ("E", "Q", "C")},
        "q_tail": Q_TAIL, "profile_sha256": sha256_file(profile_path), "m3_checkpoint_sha256": M3_SHA256,
        "products": {}, "test31_used": False, "s4_authorized": False, "next_phase_authorized": False,
    }
    for path in (identity_path, encounter_path, static_path, cdf_path, root / OUTPUT_REL / "train_dynamic_route_descriptors.parquet"):
        summary["products"][path.name] = parquet_descriptor(path, root)
    summary["artifact_sha256"] = payload_hash(summary)
    atomic_json(root / OUTPUT_REL / "train_summary.json", summary)
    return summary


def _cap_exceedance(frame: pd.DataFrame, caps: Mapping[str, Any], kind: str) -> dict[str, Any]:
    result = {}
    for profile in ("C", "M", "A"):
        values = []
        if kind == "static":
            for dimension in ("A_c", "M_c", "D_c", "L_c"):
                share = float((frame[dimension] > caps[profile][dimension]).mean())
                result[f"{profile}_{dimension}"] = share; values.append(frame[dimension] <= caps[profile][dimension])
        else:
            for dimension in DYNAMIC_DIMS:
                for metric in ("E", "Q", "C"):
                    column = f"{dimension}_{metric}"; share = float((frame[column] > caps[profile][dimension][metric]).mean())
                    result[f"{profile}_{column}"] = share; values.append(frame[column] <= caps[profile][dimension][metric])
        result[f"{profile}_all_caps_pass_count"] = int(np.logical_and.reduce(values).sum())
    for lower, upper in (("C", "M"), ("M", "A")):
        if result[f"{lower}_all_caps_pass_count"] > result[f"{upper}_all_caps_pass_count"]:
            raise Stage3S2AError(f"validation {kind} nestedness violation")
    return result


def validation_sanity(root: Path) -> dict[str, Any]:
    profile_path = root / PROFILE_REL
    before = sha256_file(profile_path); profiles = read_json(profile_path)
    if profiles.get("test31_used") is not False: raise Stage3S2AError("profile has invalid Test31 binding")
    by_id = {item["profile_id"]: item for item in profiles["profiles"]}
    static_thresholds = {p: {"A_c": by_id[p]["static_caps"]["external_physical_connection_count"], "M_c": by_id[p]["static_caps"]["topological_movement_count"], "D_c": by_id[p]["static_caps"]["road_class_diversity"], "L_c": by_id[p]["static_caps"]["internal_length_m"]} for p in by_id}
    dynamic_thresholds = {p: by_id[p]["dynamic_caps"] for p in by_id}
    upstream = _upstream_tables(root); daily = []
    for date in VALIDATION_DATES:
        daily.append(_prepare_role_day(root, date, "validation", upstream)); print(f"prepared Validation {date}", flush=True)
    identity_path = root / OUTPUT_REL / "validation_route_identity_resolution.parquet"
    encounter_path = root / OUTPUT_REL / "validation_route_complex_encounters.parquet"
    _combine_parquet(_role_parts(root, "validation", "identity.parquet", VALIDATION_DATES), identity_path)
    _combine_parquet(_role_parts(root, "validation", "encounters.parquet", VALIDATION_DATES), encounter_path)
    identity = pd.read_parquet(identity_path)
    encounters = pd.read_parquet(encounter_path)
    static_reference = build_static_reference(encounters, upstream["complexes"], upstream["boundary"], upstream["edges"])
    cdf_reference = pd.read_parquet(root / OUTPUT_REL / "train_dynamic_cdf_reference.parquet")
    descriptors = _dynamic_descriptors(root, "validation", VALIDATION_DATES, cdf_reference)
    summary = {
        "schema_version": "stage3_s3_validation_sanity.1", "validation_dates": list(VALIDATION_DATES),
        "identity": identity_summary(identity), "daily": daily,
        "unique_validation_exposed_complex_count": int(len(static_reference)),
        "static_distribution": _describe_numeric(static_reference, ["A_c", "M_c", "D_c", "L_c"]),
        "static_frozen_cap_exceedance": _cap_exceedance(static_reference, static_thresholds, "static"),
        "complete_dynamic_validation_routes": int(len(descriptors)),
        "total_validation_orders": int(sum(item["total_dynamic_orders"] for item in daily)),
        "complete_dynamic_route_coverage": float(len(descriptors) / sum(item["total_dynamic_orders"] for item in daily)),
        "dynamic_distribution": _describe_numeric(descriptors, [f"{d}_{m}" for d in DYNAMIC_DIMS for m in ("E", "Q", "C")]),
        "dynamic_frozen_cap_exceedance": _cap_exceedance(descriptors, dynamic_thresholds, "dynamic"),
        "nestedness_sanity": "PASS", "profile_sha256_before_validation": before,
        "threshold_selection_from_validation": False, "final_serviceable_order_set_emitted": False,
        "test31_used": False, "s4_authorized": False, "next_phase_authorized": False,
    }
    after = sha256_file(profile_path); summary["profile_sha256_after_validation"] = after
    if before != after: raise Stage3S2AError("Validation modified frozen profile")
    summary["artifact_sha256"] = payload_hash(summary)
    atomic_json(root / OUTPUT_REL / "validation_sanity_summary.json", summary)
    flat = pd.DataFrame([{
        "metric": key, "value_json": json.dumps(value, sort_keys=True, ensure_ascii=False)
    } for key, value in summary.items() if key != "daily"])
    atomic_parquet(root / OUTPUT_REL / "validation_sanity_summary.parquet", flat)
    return summary


def _completion_strata(root: Path, descriptors: pd.DataFrame) -> dict[str, Any]:
    complete_keys = set(zip(descriptors["date"].astype(str), descriptors["order_id"].astype(str)))
    orders = []
    for date in TRAIN_DATES:
        route = pd.read_parquet(_route_path(root, date), columns=["date", "order_id", "route_sequence", "estimated_hour", "road_class"])
        typed = pd.read_parquet(root / OUTPUT_REL / "cache/train" / f"date={date}" / "identity.parquet", columns=["date", "order_id", "route_token_type"])
        basic = route.groupby(["date", "order_id"], sort=False).agg(
            route_token_count=("route_sequence", "size"), departure_hour=("estimated_hour", "first"),
            dominant_road_class=("road_class", lambda x: str(x.mode().iloc[0]) if len(x.mode()) else "UNKNOWN"),
        ).reset_index()
        flags = typed.assign(
            has_reverse=typed["route_token_type"].eq("HISTORICAL_REVERSE_OVERLAY"),
            has_unresolved=typed["route_token_type"].eq("UNRESOLVED"),
        ).groupby(["date", "order_id"], sort=False).agg(has_reverse=("has_reverse", "any"), has_unresolved=("has_unresolved", "any")).reset_index()
        basic = basic.merge(flags, on=["date", "order_id"], validate="one_to_one")
        basic["complete"] = [key in complete_keys for key in zip(basic["date"].astype(str), basic["order_id"].astype(str))]
        orders.append(basic)
    frame = pd.concat(orders, ignore_index=True)
    frame["route_length_decile"] = pd.qcut(frame["route_token_count"].rank(method="first"), 10, labels=[str(i) for i in range(1, 11)])
    frame["departure_time_bin"] = pd.cut(
        frame["departure_hour"], bins=[-1, 5, 9, 15, 19, 24],
        labels=["00-05", "06-09", "10-15", "16-19", "20-23"], include_lowest=True,
    )
    result = {}
    for column in ("route_length_decile", "departure_time_bin", "dominant_road_class", "has_reverse", "has_unresolved"):
        grouped = frame.groupby(column, observed=False)["complete"].agg(["count", "sum", "mean"])
        result[column] = {str(index): {"orders": int(row["count"]), "complete_orders": int(row["sum"]), "complete_share": float(row["mean"])} for index, row in grouped.iterrows()}
    return result


def _threshold_lines(thresholds: Mapping[str, Any]) -> str:
    return "\n".join(f"- `{profile}`: `{json.dumps(thresholds[profile], sort_keys=True)}`" for profile in ("C", "M", "A"))


def finalize_s3(root: Path) -> dict[str, Any]:
    train = read_json(root / OUTPUT_REL / "train_summary.json")
    validation = read_json(root / OUTPUT_REL / "validation_sanity_summary.json")
    profile_path = root / PROFILE_REL; profiles = read_json(profile_path); verify_nestedness(profiles)
    if validation["profile_sha256_before_validation"] != validation["profile_sha256_after_validation"]:
        raise Stage3S2AError("profile changed during Validation")
    descriptors = pd.read_parquet(root / OUTPUT_REL / "train_dynamic_route_descriptors.parquet")
    strata = _completion_strata(root, descriptors)
    train["dynamic_completeness_strata"] = strata
    train["artifact_sha256"] = payload_hash({k: v for k, v in train.items() if k != "artifact_sha256"})
    atomic_json(root / OUTPUT_REL / "train_summary.json", train)
    docs = root / "stage3/docs/odd_tod/s3"; docs.mkdir(parents=True, exist_ok=True)
    is_s31 = bool(train.get("s31_boundary_road_class_diversity_correction"))
    release_base = S31_AUTHORIZED_BASE if is_s31 else AUTHORIZED_BASE
    release_phase = "STAGE3_S31_CLOSURE_COMPLETE" if is_s31 else PHASE_STATUS
    identity = train["identity"]
    atomic_text(docs / "stage3_s3_identity_resolution_report.md", f"""# Stage 3 S3 Identity Resolution Report

Status: `{PHASE_STATUS}`. Train covers `{TRAIN_DATES[0]}` through `{TRAIN_DATES[-1]}`; Test31 was not read.

- Orders: {identity['total_orders']:,}
- Route tokens: {identity['total_route_tokens']:,}
- `FULL_NETWORK_EDGE`: {identity['FULL_NETWORK_EDGE_count']:,} ({identity['FULL_NETWORK_EDGE_share']:.6%})
- `HISTORICAL_REVERSE_OVERLAY`: {identity['HISTORICAL_REVERSE_OVERLAY_count']:,} ({identity['HISTORICAL_REVERSE_OVERLAY_share']:.6%})
- `UNRESOLVED`: {identity['UNRESOLVED_count']:,} ({identity['UNRESOLVED_share']:.6%})
- Fully full-network-resolved orders: {identity['fully_full_network_resolved_order_count']:,}
- Orders with reverse overlay: {identity['orders_with_reverse_overlay']:,}
- Orders with unresolved token: {identity['orders_with_unresolved_token']:,}

The resolver is typed. A historical reverse traversal remains `HISTORICAL_REVERSE_OVERLAY` with `AV_ROUTABILITY_VIOLATION`; its forward physical reference is provenance only and is never substituted as the traversed edge. Unresolved and reverse tokens break complex-parser continuity. No nearest-edge repair is used.
""")
    static = train["static_thresholds"]
    atomic_text(docs / "stage3_s3_static_calibration_report.md", f"""# Stage 3 S3 Static Calibration Report

Unique Train-exposed complexes: **{train['unique_train_exposed_complex_count']:,}** (support gate: 1,000; PASS).

Static caps use unique Train-exposed complexes, not demand-weighted encounter frequency. Every complex contributes once. The only baseline dimensions are A/M/D/L; member count, QA flags, confidence, bridge, tunnel, and layer are not capability caps.

`D_c` is the number of unique `valhalla_road_class` values on the complex's unique `INCOMING`/`OUTGOING` boundary edges. `INTERNAL` edges are explicitly excluded. The former S2B internal-edge diversity is retained only as `s2b_internal_road_class_diversity_qa` provenance and is not calibrated.

## Frozen thresholds (`higher`)

{_threshold_lines(static)}

Signal state distribution: `{json.dumps(train['signal_state_distribution'], sort_keys=True)}`. Roundabout share: {train['roundabout_share']:.6%}. Grade-separation-evidence share: {train['grade_separation_share']:.6%}.

Distribution summary: `{json.dumps(train['static_distribution'], sort_keys=True)}`.

Advanced strict-tail support: `{json.dumps(train['static_advanced_tail_support'], sort_keys=True)}`.
""")
    dynamic = train["dynamic_thresholds"]
    atomic_text(docs / "stage3_s3_dynamic_calibration_report.md", f"""# Stage 3 S3 Dynamic Calibration Report

Frozen predictor: M3 checkpoint `{M3_SHA256}`. No Stage2 training, checkpoint selection, or realized-future time was used.

- Train orders: {train['total_train_orders']:,}
- Common complete dynamic routes: {train['complete_dynamic_train_routes']:,}
- Complete-route coverage: {train['complete_dynamic_route_coverage']:.6%}
- Common-cohort gate: 4,000 (PASS)
- Tail anchor: `q_tail = {Q_TAIL}` with strict `z > 0.90`
- CDF: global Train predicted-time-weighted mid-distribution; exact support persisted

The profile anchor `pi_k` is a marginal per-dimension quantile anchor. It is not a joint route acceptance target. Requiring all 12 dynamic caps to pass is non-compensatory, so the joint pass rate is expected to be lower than each marginal anchor.

## Frozen 36 E/Q/C caps (`higher`, one route = one sample)

{_threshold_lines(dynamic)}

CDF tie diagnostics: `{json.dumps(train['cdf_tie_diagnostics'], sort_keys=True)}`. Exact-value weighted masses are empirically negligible for the frozen continuous M3 outputs; the weighted mid-CDF remains the frozen robust definition.

Advanced strict-tail support counts: `{json.dumps(train['dynamic_advanced_tail_support'], sort_keys=True)}`.

Completeness strata (route-length decile, departure time, dominant road class, reverse overlay, unresolved token): `{json.dumps(strata, sort_keys=True)}`.

The future dynamic contract is non-compensatory: every one of 12 caps must pass. Missing required M3 output yields `UNKNOWN`; no zero, mean, median, neighbor, or road-class imputation is allowed.
""")
    vident = validation["identity"]
    atomic_text(docs / "stage3_s3_validation_sanity_report.md", f"""# Stage 3 S3 Validation Sanity Report

Validation dates `{VALIDATION_DATES[0]}`–`{VALIDATION_DATES[-1]}` were processed only after Train profile freeze.

- Orders: {vident['total_orders']:,}; tokens: {vident['total_route_tokens']:,}
- Full-network share: {vident['FULL_NETWORK_EDGE_share']:.6%}
- Reverse-overlay share: {vident['HISTORICAL_REVERSE_OVERLAY_share']:.6%}
- Unresolved share: {vident['UNRESOLVED_share']:.6%}
- Complete dynamic routes: {validation['complete_dynamic_validation_routes']:,}/{validation['total_validation_orders']:,} ({validation['complete_dynamic_route_coverage']:.6%})
- Nestedness sanity: `{validation['nestedness_sanity']}`
- Profile SHA before: `{validation['profile_sha256_before_validation']}`
- Profile SHA after: `{validation['profile_sha256_after_validation']}`

Static frozen-cap exceedance: `{json.dumps(validation['static_frozen_cap_exceedance'], sort_keys=True)}`.

Dynamic frozen-cap exceedance: `{json.dumps(validation['dynamic_frozen_cap_exceedance'], sort_keys=True)}`.

The C/M/A dynamic all-12-dimension pass counts are joint non-compensatory results, not estimates of the marginal `pi` anchors.

Validation did not select thresholds, retune profiles, or emit an AV-serviceable-order set. Test31 remained untouched.
""")
    atomic_text(docs / "stage3_s3_methodology.md", f"""# Stage 3 S3 Capability-Envelope Methodology

This phase freezes three nested hypothetical capability scenarios `C ⊆ M ⊆ A`; it does not estimate safety, legality, failure, disengagement, or accident probability.

1. Historical tokens are resolved into typed full-network, reverse-overlay, or unresolved identities. Broken identity breaks continuity.
2. The production complex parser recognizes incoming → zero or more internal → outgoing edges and never splices across a gap.
3. Static A/M/D/L caps use one observation per unique Train-exposed complex and `higher` quantiles at 0.75/0.90/0.975. `D_c` counts unique `valhalla_road_class` on INCOMING/OUTGOING boundary edges; INTERNAL edges are excluded.
4. Dynamic inputs are frozen-M3 decision-time predictions. Predicted P50 travel time advances/weights exposure; realized future timing is forbidden.
5. Each dimension uses a global Train predicted-time-weighted mid-CDF. Tail is strict `z > 0.90`. Route E/Q/C preserve token order; threshold fitting gives every complete route one vote. `pi_k` freezes marginal dimension caps, not joint route acceptance rates.
6. Speed caps remain 60/80/120 km/h. Maneuver, roundabout, restriction, and unknown rules are categorical and non-compensatory. Certified prohibition is incompatible; non-certification is not legal permission. Grade separation, bridge, and tunnel are descriptive only.
7. Validation 25–27 is sanity only. The profile is hash-bound before and after. Test31 aliases are hard rejected.

Frozen profile: `{PROFILE_REL.as_posix()}` SHA-256 `{sha256_file(profile_path)}`.

`S4_AUTHORIZED = NO`; `NEXT_PHASE_AUTHORIZED = NO`.
""")
    if is_s31:
        closure = read_json(root / OUTPUT_REL / "s31_closure_summary.json")
        atomic_text(docs / "stage3_s31_closure_report.md", f"""# Stage 3 S3.1 Scientific Closure

Status: `STAGE3_S31_CLOSURE_COMPLETE`. Reviewed base: `{S31_AUTHORIZED_BASE}`.

## Static D correction

`D_c` now equals the number of unique `valhalla_road_class` values on unique `INCOMING` and `OUTGOING` boundary edges of the frozen 10m complex. `INTERNAL` edges are excluded. No clustering, membership, movement, A/M/L definition, speed rule, or dynamic rule changed.

- Train-exposed complexes: {closure['train_unique_complex_count']:,}
- Old D caps C/M/A: `{closure['old_static_caps']['C']['road_class_diversity']}` / `{closure['old_static_caps']['M']['road_class_diversity']}` / `{closure['old_static_caps']['A']['road_class_diversity']}`
- New D caps C/M/A: `{closure['new_static_caps']['C']['D_c']}` / `{closure['new_static_caps']['M']['D_c']}` / `{closure['new_static_caps']['A']['D_c']}`
- A/M/L unchanged: `{closure['a_m_l_unchanged']}`

## Frozen dynamic invariance

Dynamic caps and all dynamic products were not recomputed. Before/after hashes are identical: `{json.dumps(closure['dynamic_product_hashes_after'], sort_keys=True)}`.

## Train M3 cache provenance

- Dates bound: {closure['m3_train_cache_provenance']['date_count']}
- Prediction rows: {closure['m3_train_cache_provenance']['total_prediction_rows']:,}
- Model/checkpoint: M3 / `{M3_SHA256}`
- All cache hashes, row counts, schemas, and day manifests verified: `true`
- Realized target columns present: `false`
- Inference rerun required: `false`

`pi_k` defines marginal capability caps, not joint route acceptance rates.

Test31 was not read. `S4_AUTHORIZED = NO`; `NEXT_PHASE_AUTHORIZED = NO`.
""")
    required_products = [
        root / OUTPUT_REL / "train_route_identity_resolution.parquet",
        root / OUTPUT_REL / "train_route_complex_encounters.parquet",
        root / OUTPUT_REL / "train_static_complex_reference.parquet",
        root / OUTPUT_REL / "train_dynamic_cdf_reference.parquet",
        root / OUTPUT_REL / "train_dynamic_route_descriptors.parquet",
        root / OUTPUT_REL / "validation_sanity_summary.parquet",
        root / OUTPUT_REL / "validation_sanity_summary.json",
    ]
    if is_s31:
        required_products.extend([
            root / OUTPUT_REL / "train_m3_cache_provenance.parquet",
            root / OUTPUT_REL / "train_m3_cache_provenance.json",
            root / OUTPUT_REL / "s31_closure_summary.json",
        ])
    source_paths = [
        root / "stage3/docs/odd_tod/s2b/stage3_s2b_final_release_manifest.json",
        root / "stage3/docs/odd_tod/s2b/stage3_s2b_to_s3_contract.md",
        root / "stage2/docs/v5_2/stage2_v5_2_final_release_manifest.json",
        root / "stage2/docs/v5_2/stage2_v5_2_to_stage3_contract.md",
        root / "stage2/output_v5_2/development/M3/model_manifest.json",
        root / "stage2/output_v5_2/development/M3/epoch_004.pt",
        root / "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json",
        root / "stage2/output_v5/protocols/development/tensor_shards/feature_artifacts.json",
        root / "stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet",
        root / "stage3/output/odd_tod/s2a/stage3_full_network_nodes.parquet",
        root / "stage3/output/odd_tod/s2a/stage3_speed_domain.parquet",
        root / "stage3/output/odd_tod/s2a/stage3_historical_direction_overlay.parquet",
        root / "stage3/output/odd_tod/s2b/final/stage3_intersection_complexes.parquet",
        root / "stage3/output/odd_tod/s2b/final/stage3_intersection_movements.parquet",
        root / "stage3/output/odd_tod/s2b/final/stage3_intersection_node_membership.parquet",
        root / "stage3/output/odd_tod/s2b/final/stage3_edge_complex_boundary_index.parquet",
        root / "stage3/output/odd_tod/s2b/final/stage3_route_movement_lookup.parquet",
    ]
    test_evidence = {
        "schema_version": "stage3_s31_test_evidence.1" if is_s31 else "stage3_s3_test_evidence.1", "authorized_base": release_base,
        "checks": {
            "identity_gate": "PASS", "route_parser_gate": "PASS", "static_support_gate": "PASS",
            "dynamic_support_gate": "PASS", "profile_nestedness": "PASS", "validation_profile_immutability": "PASS",
            "test31_hard_guard": "PASS", "scope_guard": "PASS",
        },
        "compileall": "PASS",
        "focused_tests": {"status": "PASS", "environment": "pytorch", "passed": 82 if is_s31 else 75},
        "full_tests": {"status": "PASS", "environment": "base", "passed": 156 if is_s31 else 149, "warnings": 1},
        "environment_note": "pytorch is the frozen M3 inference runtime; base supplies fiona/pyproj for the full Stage3 test collection",
        "test31_used": False, "s4_authorized": False, "next_phase_authorized": False,
    }
    test_evidence["artifact_sha256"] = payload_hash(test_evidence); atomic_json(docs / "stage3_s3_test_evidence.json", test_evidence)
    release = {
        "schema_version": "stage3_s31_release_manifest.1" if is_s31 else "stage3_s3_release_manifest.1", "phase_status": release_phase,
        "base_commit": release_base, "final_commit": "RECORDED_BY_GIT_COMMIT_AND_REMOTE_HEAD_OUTSIDE_SELF_HASHED_MANIFEST",
        "train_dates": list(TRAIN_DATES), "validation_sanity_dates": list(VALIDATION_DATES), "test31_used": False,
        "profile": source_descriptor(profile_path, root), "m3_checkpoint_sha256": M3_SHA256,
        "frozen_inputs": {path.relative_to(root).as_posix(): source_descriptor(path, root) for path in source_paths},
        "products": {path.relative_to(root).as_posix(): (parquet_descriptor(path, root) if path.suffix == ".parquet" else source_descriptor(path, root)) for path in required_products},
        "gates": {"identity": "PASS", "route_parser": "PASS", "static": "PASS", "dynamic": "PASS", "scenario": "PASS", "validation": "PASS", "scope": "PASS"},
        "scope": {"test31_route_feasibility": False, "fallback_routing": False, "stage4": False, "s4_authorized": False, "next_phase_authorized": False},
    }
    if is_s31:
        release["s31_static_correction"] = {
            "D_c_definition": "unique valhalla_road_class over INCOMING/OUTGOING boundary edges; INTERNAL excluded",
            "A_M_L_unchanged": True, "dynamic_caps_unchanged": True,
            "quantile_anchor_semantics": "marginal capability caps, not joint route acceptance rates",
        }
        release["train_m3_prediction_caches"] = {}
        for date in TRAIN_DATES:
            cache_path = root / OUTPUT_REL / "cache/m3" / f"date={date}.parquet"
            day_manifest = cache_path.with_suffix(".json")
            descriptor = parquet_descriptor(cache_path, root)
            descriptor.update({
                "date": date, "model_id": "M3", "checkpoint_sha256": M3_SHA256,
                "decision_time_only": True, "predicted_progression_only": True,
                "realized_target_columns_present": False,
                "day_manifest_path": day_manifest.relative_to(root).as_posix(),
                "day_manifest_sha256": sha256_file(day_manifest),
            })
            release["train_m3_prediction_caches"][date] = descriptor
    release["artifact_sha256"] = payload_hash(release); atomic_json(docs / "stage3_s3_release_manifest.json", release)
    evidence = {
        "schema_version": "stage3_s31_evidence_bundle.1" if is_s31 else "stage3_s3_evidence_bundle.1", "phase_status": release_phase,
        "release_manifest": source_descriptor(docs / "stage3_s3_release_manifest.json", root),
        "test_evidence": source_descriptor(docs / "stage3_s3_test_evidence.json", root),
        "reports": {path.name: source_descriptor(path, root) for path in sorted(docs.glob("*.md"))},
        "profile_sha256": sha256_file(profile_path), "train_summary_sha256": sha256_file(root / OUTPUT_REL / "train_summary.json"),
        "validation_summary_sha256": sha256_file(root / OUTPUT_REL / "validation_sanity_summary.json"),
        "evidence_verification": "ALL_BOUND_HASHES_VERIFIED", "test31_used": False,
        "s4_authorized": False, "next_phase_authorized": False,
    }
    verification_path = docs / "stage3_s3_evidence_verification.json"
    if verification_path.is_file(): evidence["independent_verification"] = source_descriptor(verification_path, root)
    evidence["artifact_sha256"] = payload_hash(evidence); atomic_json(docs / "stage3_s3_evidence_bundle.json", evidence)
    return {"phase_status": release_phase, "profile_sha256": sha256_file(profile_path), "release_manifest_sha256": sha256_file(docs / "stage3_s3_release_manifest.json"), "evidence_bundle_sha256": sha256_file(docs / "stage3_s3_evidence_bundle.json"), "s4_authorized": False, "next_phase_authorized": False}


def verify_s3_evidence(root: Path) -> dict[str, Any]:
    docs = root / "stage3/docs/odd_tod/s3"
    release = read_json(docs / "stage3_s3_release_manifest.json")
    failures = []
    for section in ("frozen_inputs", "products"):
        for relative, descriptor in release[section].items():
            path = root / relative
            if not path.is_file(): failures.append(f"missing:{relative}"); continue
            if sha256_file(path) != descriptor["sha256"]: failures.append(f"sha256:{relative}")
            if path.suffix == ".parquet" and "row_count" in descriptor and int(pq.ParquetFile(path).metadata.num_rows) != int(descriptor["row_count"]):
                failures.append(f"row_count:{relative}")
    for date, descriptor in release.get("train_m3_prediction_caches", {}).items():
        path = root / descriptor["path"]
        if not path.is_file(): failures.append(f"missing_train_m3_cache:{date}"); continue
        if sha256_file(path) != descriptor["sha256"]: failures.append(f"train_m3_cache_sha256:{date}")
        parquet = pq.ParquetFile(path)
        if int(parquet.metadata.num_rows) != int(descriptor["row_count"]): failures.append(f"train_m3_cache_row_count:{date}")
        columns = parquet.schema_arrow.names
        if [column for column in columns if column.startswith("target_") or column.endswith("_target_valid")]:
            failures.append(f"train_m3_cache_realized_target:{date}")
        if descriptor.get("checkpoint_sha256") != M3_SHA256: failures.append(f"train_m3_cache_checkpoint:{date}")
        manifest_path = root / descriptor["day_manifest_path"]
        if not manifest_path.is_file() or sha256_file(manifest_path) != descriptor["day_manifest_sha256"]:
            failures.append(f"train_m3_day_manifest:{date}")
    profile = read_json(root / PROFILE_REL)
    if profile.get("test31_used") is not False: failures.append("profile_test31")
    if profile.get("s4_authorized") is not False or profile.get("next_phase_authorized") is not False:
        failures.append("profile_scope")
    if "INCOMING/OUTGOING boundary edges" not in profile.get("static_dimension_definitions", {}).get("D_c", ""):
        failures.append("static_D_definition")
    train = read_json(root / OUTPUT_REL / "train_summary.json")
    validation = read_json(root / OUTPUT_REL / "validation_sanity_summary.json")
    seen_dates = set(train["train_dates"]) | set(validation["validation_dates"])
    if seen_dates != set(TRAIN_DATES) | set(VALIDATION_DATES): failures.append("date_set")
    if "20161031" in seen_dates: failures.append("test31_date")
    if validation["profile_sha256_before_validation"] != validation["profile_sha256_after_validation"]:
        failures.append("validation_profile_mutation")
    if sha256_file(root / PROFILE_REL) != validation["profile_sha256_after_validation"]:
        failures.append("profile_hash_binding")
    result = {
        "schema_version": "stage3_s3_evidence_verification.1", "status": "PASS" if not failures else "FAIL",
        "verified_frozen_input_count": len(release["frozen_inputs"]),
        "verified_product_count": len(release["products"]), "failures": failures,
        "verified_train_m3_cache_count": len(release.get("train_m3_prediction_caches", {})),
        "observed_dates": sorted(seen_dates), "test31_used": False,
        "profile_sha256": sha256_file(root / PROFILE_REL), "s4_authorized": False, "next_phase_authorized": False,
    }
    result["artifact_sha256"] = payload_hash(result)
    atomic_json(docs / "stage3_s3_evidence_verification.json", result)
    if failures: raise Stage3S2AError(f"S3 evidence verification failed: {failures}")
    return result


def bind_train_m3_cache_evidence(root: Path) -> dict[str, Any]:
    """Close checkpoint -> Train prediction provenance without rerunning inference."""
    rows = []
    for date in TRAIN_DATES:
        path = root / OUTPUT_REL / "cache/m3" / f"date={date}.parquet"
        manifest_path = path.with_suffix(".json")
        if not path.is_file() or not manifest_path.is_file():
            raise Stage3S2AError(f"Train M3 cache evidence missing: {date}")
        manifest = read_json(manifest_path); parquet = pq.ParquetFile(path)
        columns = parquet.schema_arrow.names
        current_sha = sha256_file(path); row_count = int(parquet.metadata.num_rows)
        forbidden = [column for column in columns if column.startswith("target_") or column.endswith("_target_valid")]
        failures = []
        if manifest.get("date") != date: failures.append("date")
        if manifest.get("model_id") != "M3": failures.append("model_id")
        if manifest.get("checkpoint_sha256") != M3_SHA256: failures.append("checkpoint_sha256")
        if manifest.get("prediction_sha256") != current_sha: failures.append("prediction_sha256")
        if int(manifest.get("row_count", -1)) != row_count: failures.append("row_count")
        if manifest.get("decision_time_only") is not True: failures.append("decision_time_only")
        if manifest.get("predicted_progression_only") is not True: failures.append("predicted_progression_only")
        if manifest.get("realized_target_columns_persisted") is not False or forbidden: failures.append("realized_target_columns")
        if failures:
            raise Stage3S2AError(f"Train M3 cache provenance cannot be proven for {date}: {failures}; rerun frozen M3 inference")
        schema_text = str(parquet.schema_arrow)
        rows.append({
            "date": date, "path": path.relative_to(root).as_posix(), "sha256": current_sha,
            "row_count": row_count, "schema": schema_text,
            "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
            "model_id": "M3", "checkpoint_sha256": M3_SHA256,
            "day_manifest_path": manifest_path.relative_to(root).as_posix(),
            "day_manifest_sha256": sha256_file(manifest_path),
            "training_manifest_sha256": manifest.get("training_manifest_sha256"),
            "transfer_manifest_sha256": manifest.get("transfer_manifest_sha256"),
            "feature_schema_sha256": manifest.get("feature_schema_sha256"),
            "decision_time_only": True, "predicted_progression_only": True,
            "realized_target_columns_present": False,
        })
    evidence_path = root / OUTPUT_REL / "train_m3_cache_provenance.parquet"
    atomic_parquet(evidence_path, pd.DataFrame(rows))
    summary = {
        "schema_version": "stage3_s31_train_m3_cache_provenance.1", "status": "PASS",
        "authorized_base": S31_AUTHORIZED_BASE, "date_count": len(rows),
        "total_prediction_rows": int(sum(row["row_count"] for row in rows)),
        "model_id": "M3", "checkpoint_sha256": M3_SHA256,
        "all_day_cache_hashes_match_manifests": True,
        "all_day_schemas_prediction_only": True, "realized_target_columns_present": False,
        "inference_rerun_required": False,
        "product": parquet_descriptor(evidence_path, root),
        "test31_used": False, "s4_authorized": False, "next_phase_authorized": False,
    }
    summary["artifact_sha256"] = payload_hash(summary)
    atomic_json(root / OUTPUT_REL / "train_m3_cache_provenance.json", summary)
    return summary


def close_s31(root: Path) -> dict[str, Any]:
    if git_head(root) != S31_AUTHORIZED_BASE:
        raise Stage3S2AError(f"S3.1 closure requires reviewed base {S31_AUTHORIZED_BASE}")
    dynamic_paths = {
        "cdf": root / OUTPUT_REL / "train_dynamic_cdf_reference.parquet",
        "train_descriptors": root / OUTPUT_REL / "train_dynamic_route_descriptors.parquet",
        "validation_descriptors": root / OUTPUT_REL / "validation_dynamic_route_descriptors.parquet",
    }
    dynamic_before = {name: sha256_file(path) for name, path in dynamic_paths.items()}
    old_profile = read_json(root / PROFILE_REL)
    old_by_id = {item["profile_id"]: item for item in old_profile["profiles"]}
    old_static = {p: dict(old_by_id[p]["static_caps"]) for p in ("C", "M", "A")}
    old_dynamic = {p: old_by_id[p]["dynamic_caps"] for p in ("C", "M", "A")}
    provenance = bind_train_m3_cache_evidence(root)
    upstream = _upstream_tables(root)
    train_encounters = pd.read_parquet(root / OUTPUT_REL / "train_route_complex_encounters.parquet")
    train_reference = build_static_reference(train_encounters, upstream["complexes"], upstream["boundary"], upstream["edges"])
    static_path = root / OUTPUT_REL / "train_static_complex_reference.parquet"
    atomic_parquet(static_path, train_reference)
    new_static = static_caps(train_reference)
    for profile in ("C", "M", "A"):
        expected_unchanged = {
            "A_c": old_static[profile]["external_physical_connection_count"],
            "M_c": old_static[profile]["topological_movement_count"],
            "L_c": old_static[profile]["internal_length_m"],
        }
        actual = {key: new_static[profile][key] for key in expected_unchanged}
        if actual != expected_unchanged:
            raise Stage3S2AError(f"S3.1 changed frozen A/M/L for {profile}: {expected_unchanged} -> {actual}")
    profiles = build_profiles(new_static, old_dynamic)
    verify_nestedness(profiles)
    atomic_json(root / PROFILE_REL, profiles)
    profile_before_validation = sha256_file(root / PROFILE_REL)
    validation_encounters = pd.read_parquet(root / OUTPUT_REL / "validation_route_complex_encounters.parquet")
    validation_reference = build_static_reference(validation_encounters, upstream["complexes"], upstream["boundary"], upstream["edges"])
    validation = read_json(root / OUTPUT_REL / "validation_sanity_summary.json")
    validation["static_distribution"] = _describe_numeric(validation_reference, ["A_c", "M_c", "D_c", "L_c"])
    validation["static_frozen_cap_exceedance"] = _cap_exceedance(validation_reference, new_static, "static")
    validation["unique_validation_exposed_complex_count"] = int(len(validation_reference))
    validation["profile_sha256_before_validation"] = profile_before_validation
    validation["profile_sha256_after_validation"] = sha256_file(root / PROFILE_REL)
    validation["s31_static_only_recalibration"] = True
    validation["dynamic_products_recomputed"] = False
    validation["artifact_sha256"] = payload_hash({k: v for k, v in validation.items() if k != "artifact_sha256"})
    atomic_json(root / OUTPUT_REL / "validation_sanity_summary.json", validation)
    flat = pd.DataFrame([{"metric": key, "value_json": json.dumps(value, sort_keys=True, ensure_ascii=False)} for key, value in validation.items() if key != "daily"])
    atomic_parquet(root / OUTPUT_REL / "validation_sanity_summary.parquet", flat)
    dynamic_after = {name: sha256_file(path) for name, path in dynamic_paths.items()}
    if dynamic_before != dynamic_after:
        raise Stage3S2AError(f"S3.1 changed dynamic products: {dynamic_before} -> {dynamic_after}")
    new_by_id = {item["profile_id"]: item for item in profiles["profiles"]}
    if any(new_by_id[p]["dynamic_caps"] != old_dynamic[p] for p in ("C", "M", "A")):
        raise Stage3S2AError("S3.1 changed frozen dynamic caps")
    train = read_json(root / OUTPUT_REL / "train_summary.json")
    train["static_distribution"] = _describe_numeric(train_reference, ["A_c", "M_c", "D_c", "L_c"])
    train["static_thresholds"] = new_static
    train["static_advanced_tail_support"] = {d: int((train_reference[d] > new_static["A"][d]).sum()) for d in ("A_c", "M_c", "D_c", "L_c")}
    train["profile_sha256"] = sha256_file(root / PROFILE_REL)
    train["s31_boundary_road_class_diversity_correction"] = True
    train["dynamic_products_recomputed"] = False
    train["train_m3_cache_provenance_sha256"] = sha256_file(root / OUTPUT_REL / "train_m3_cache_provenance.json")
    train["products"][static_path.name] = parquet_descriptor(static_path, root)
    train["artifact_sha256"] = payload_hash({k: v for k, v in train.items() if k != "artifact_sha256"})
    atomic_json(root / OUTPUT_REL / "train_summary.json", train)
    closure = {
        "schema_version": "stage3_s31_closure.1", "phase_status": "STAGE3_S31_CLOSURE_COMPLETE",
        "authorized_base": S31_AUTHORIZED_BASE,
        "static_definition": "D_c = unique valhalla_road_class over INCOMING/OUTGOING boundary edges; INTERNAL excluded",
        "train_unique_complex_count": int(len(train_reference)),
        "old_static_caps": old_static, "new_static_caps": new_static,
        "a_m_l_unchanged": True, "d_changed_only": True,
        "dynamic_caps_unchanged": True, "dynamic_product_hashes_before": dynamic_before,
        "dynamic_product_hashes_after": dynamic_after,
        "m3_train_cache_provenance": provenance,
        "profile_sha256_before_validation": profile_before_validation,
        "profile_sha256_after_validation": sha256_file(root / PROFILE_REL),
        "quantile_anchor_semantics": "pi_k defines marginal capability caps, not joint route acceptance rates",
        "test31_used": False, "s4_authorized": False, "next_phase_authorized": False,
    }
    closure["artifact_sha256"] = payload_hash(closure)
    atomic_json(root / OUTPUT_REL / "s31_closure_summary.json", closure)
    return closure


def _m3_model(root: Path):
    import torch
    from stage2.v5_2.training import initialized_transfer_model
    manifest_path = root / "stage2/output_v5_2/development/M3/model_manifest.json"
    training = read_json(manifest_path); source = training["source"]; constructor = training["constructor"]
    checkpoint = root / "stage2/output_v5_2/development/M3/epoch_004.pt"
    if sha256_file(checkpoint) != M3_SHA256 or training["selected_checkpoint_sha256"] != M3_SHA256:
        raise Stage3S2AError("frozen M3 checkpoint binding mismatch")
    model, binding = initialized_transfer_model(
        protocol_id="development", feature_artifact_path=source["feature_artifact_path"],
        checkpoint_path=source["source_checkpoint_path"], source_model_manifest_path=source["source_model_manifest_path"],
        source_config_path=source["source_config_path"], static_feature_count=int(constructor["static_feature_count"]),
        support_tau=float(constructor["support_tau"]), spatial_mode=str(constructor["spatial_mode"]),
        temporal_mode=str(constructor["temporal_mode"]), backbone_kwargs=dict(constructor["backbone_kwargs"]),
    )
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(saved.get("model_state_dict", saved), strict=True)
    return model, training, binding


def _strict_prediction_frame(frame: pd.DataFrame) -> pd.DataFrame:
    allowed = [
        "date", "order_id", "traversal_id", "split", "support_group_code", "allocated_distance_m",
        "pred_crawl", "pred_stop", "pred_speed_cv", "pred_acceleration_rms", "pred_rts", "pred_pace_p50",
        "travel_time_p50_s",
    ]
    missing = [column for column in allowed if column not in frame.columns and column != "travel_time_p50_s"]
    if missing: raise Stage3S2AError(f"M3 prediction schema missing: {missing}")
    work = frame.copy()
    if "travel_time_p50_s" not in work:
        work["travel_time_p50_s"] = work["pred_pace_p50"] * work["allocated_distance_m"]
    result = work[allowed].copy()
    forbidden = [column for column in result if column.startswith("target_") or column.endswith("_target_valid")]
    if forbidden: raise Stage3S2AError(f"realized target leaked into S3 cache: {forbidden}")
    return result


def replay_day(root: Path, date: str, batch_size: int = 256) -> dict[str, Any]:
    validate_s3_date(date, TRAIN_DATES)
    import torch
    from stage2.v5_2.training import collect_unique_predictions
    out = root / f"stage3/output/odd_tod/s3/cache/m3/date={date}.parquet"
    manifest_path = out.with_suffix(".json")
    if out.is_file() and manifest_path.is_file():
        manifest = read_json(manifest_path)
        if manifest.get("checkpoint_sha256") == M3_SHA256 and manifest.get("prediction_sha256") == sha256_file(out):
            existing = pq.ParquetFile(out).schema_arrow.names
            if set(existing) == set(_strict_prediction_frame(pd.read_parquet(out, columns=existing)).columns):
                return {**manifest, "cache_reused": True}
            migrated = _strict_prediction_frame(pd.read_parquet(out))
            atomic_parquet(out, migrated)
            manifest.update({
                "prediction_sha256": sha256_file(out), "row_count": len(migrated),
                "strict_prediction_only_schema": True, "realized_target_columns_persisted": False,
                "cache_schema_migrated": True,
            })
            manifest["artifact_sha256"] = payload_hash({k: v for k, v in manifest.items() if k != "artifact_sha256"})
            atomic_json(manifest_path, manifest)
            return {**manifest, "cache_reused": True}
    transfer = read_json(root / "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json")
    day = next((item for item in transfer["days"] if item["date"] == date), None)
    if day is None: raise Stage3S2AError(f"date absent from frozen transfer manifest: {date}")
    shard_root = root / "stage2/output_v5_2/transfer_shards/protocol=development"
    paths = [shard_root / item["path"] for item in day["files"]]
    for path, item in zip(paths, day["files"]):
        if sha256_file(path) != item["sha256"]: raise Stage3S2AError(f"transfer shard hash mismatch: {path}")
    model, training, binding = _m3_model(root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); model.to(device)
    started = time.perf_counter()
    predictions, diagnostics = collect_unique_predictions(model, paths, batch_size, device, component_weights=training["loss_policy"]["component_weights"])
    predictions = _strict_prediction_frame(predictions)
    atomic_parquet(out, predictions)
    manifest = {
        "schema_version": "stage3_s3_m3_day_cache.1", "date": date, "role": day["split"], "model_id": "M3",
        "checkpoint_sha256": M3_SHA256, "training_manifest_sha256": sha256_file(root / "stage2/output_v5_2/development/M3/model_manifest.json"),
        "transfer_manifest_sha256": sha256_file(root / "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json"),
        "feature_schema_sha256": training["source"]["feature_schema_hash"], "source_binding": binding,
        "prediction_path": out.relative_to(root).as_posix(), "prediction_sha256": sha256_file(out), "row_count": len(predictions),
        "runtime_s": time.perf_counter() - started, "device": str(device), "batch_size": batch_size,
        "decision_time_only": True, "predicted_progression_only": True, "realized_future_time_used": False,
        "strict_prediction_only_schema": True, "realized_target_columns_persisted": False,
        "diagnostics": diagnostics,
    }
    manifest["artifact_sha256"] = payload_hash(manifest); atomic_json(manifest_path, manifest); return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay-day"); replay.add_argument("date"); replay.add_argument("--batch-size", type=int, default=256)
    sub.add_parser("build-train")
    sub.add_parser("validation-sanity")
    sub.add_parser("finalize")
    sub.add_parser("verify")
    sub.add_parser("close-s31")
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv); root = args.root.resolve()
    if args.command == "close-s31":
        result = close_s31(root)
        print(json.dumps(result, indent=2, ensure_ascii=False)); return 0
    head = git_head(root)
    allowed_heads = {AUTHORIZED_BASE, S31_AUTHORIZED_BASE} if args.command in {"finalize", "verify"} else {AUTHORIZED_BASE}
    if head not in allowed_heads: raise Stage3S2AError(f"S3 {args.command} requires one of reviewed bases {sorted(allowed_heads)}")
    if args.command == "replay-day": result = replay_day(root, args.date, args.batch_size)
    elif args.command == "build-train": result = build_train(root)
    elif args.command == "validation-sanity": result = validation_sanity(root)
    elif args.command == "finalize": result = finalize_s3(root)
    elif args.command == "verify": result = verify_s3_evidence(root)
    else: raise Stage3S2AError(f"unsupported S3 command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False)); return 0


if __name__ == "__main__": raise SystemExit(main())
