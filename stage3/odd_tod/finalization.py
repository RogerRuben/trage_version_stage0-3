"""Minimal bounded AV fallback and final Stage3-to-Stage4 interface.

Only structurally hard-infeasible historical routes trigger routing.  The
fallback uses one deterministic Valhalla ``auto`` route and asks Valhalla to
attribute its own encoded route shape with ``edge_walk``.  Every returned
directed edge must map exactly to the frozen Stage3 edge namespace.

The frozen M3 contract requires historical, decision-time feature rows that
do not exist for a newly proposed route.  Consequently fallback dynamic
evidence is explicitly unknown; no historical-route prediction, neighbour
prediction, class mean, or realised Test31 state is substituted.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from stage0.v6.coordinates import gcj02_to_wgs84
from stage3.odd_tod.capability_envelope import DYNAMIC_DIMS, parse_route_complex_encounters
from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
    atomic_json,
    atomic_parquet,
    atomic_text,
    parquet_descriptor,
    payload_hash,
    read_json,
    sha256_file,
    source_descriptor,
    stage3_edge_uid,
)
from stage3.odd_tod.operational_suitability import (
    HARD_REASON_CODES,
    STATIC_DIMENSIONS,
    _distribution,
    _profile_map,
    _ratio_vectors,
)
from stage3.odd_tod.original_route_suitability import (
    CDF_REL,
    EXPECTED_ORDER_COUNT,
    EXPECTED_ORDER_PROFILE_COUNT,
    M3_SHA256,
    PROFILE_REL,
    PROFILES,
    ROUTE_REL,
    TEST_DATE,
    UPSTREAM_REL,
    _encounter_descriptor,
    _encounter_detail,
    _full_static_reference,
    _speed_descriptor,
    evaluate_movement_atomic_checks,
)


AUTHORIZED_BASE = "514db0933493578463de0c37b9cfd31cc730ed7a"
PHASE_STATUS = "STAGE3_FINAL_FROZEN"
CONFIG_REL = Path("stage3/config/stage3_finalization.json")
OUTPUT_REL = Path("stage3/output/odd_tod/final")
DOCS_REL = Path("stage3/docs/odd_tod/final")
ORIGINAL_SUITABILITY_REL = Path(
    "stage3/output/odd_tod/s4/test31_av_operational_suitability.parquet"
)
ORIGINAL_DESCRIPTOR_REL = Path(
    "stage3/output/odd_tod/s4/test31_original_route_descriptors.parquet"
)
OD_MANIFEST_REL = Path(
    "stage0/work_v6_final/candidate_manifests/date=20161031.parquet"
)
FALLBACK_ROUTE_REL = OUTPUT_REL / "test31_fallback_route_edges.parquet"
FALLBACK_STATUS_REL = OUTPUT_REL / "test31_fallback_candidates.parquet"
INTERFACE_REL = OUTPUT_REL / "test31_stage3_to_stage4_interface.parquet"
SUMMARY_REL = OUTPUT_REL / "stage3_final_summary.json"
MANIFEST_REL = DOCS_REL / "stage3_final_manifest.json"
METHODOLOGY_REL = DOCS_REL / "stage3_final_methodology.md"
CONTRACT_REL = DOCS_REL / "stage3_to_stage4_contract.md"
REPORT_REL = DOCS_REL / "stage3_final_summary.md"

FINAL_COLUMNS = (
    "date",
    "order_id",
    "profile_id",
    "selected_route_type",
    "hard_state",
    "evidence_complete",
    "fallback_attempted",
    "fallback_candidate_count",
    "fallback_hard_feasible_count",
    "fallback_search_state",
    "selected_route_distance_m",
    "selected_service_time_p50_s",
    "rho_static",
    "rho_dynamic",
    "rho_speed",
    "rho_overall",
    "static_vector",
    "dynamic_vector",
    "hard_reason_codes",
    "unknown_reason_codes",
    "soft_reason_codes",
    "original_route_hard_state",
    "original_route_hard_reason_codes",
    "original_route_rho_static",
    "original_route_rho_dynamic",
    "original_route_rho_speed",
    "fallback_distance_ratio",
    "selected_route_reference",
)

DYNAMIC_UNKNOWN_REASON = "FROZEN_M3_EXACT_INPUT_CONTRACT_UNAVAILABLE_FOR_NEW_ROUTE"
LIMITED_SEARCH_NOT_ESTABLISHED_REASON = (
    "NO_HARD_FEASIBLE_FALLBACK_FOUND_UNDER_LIMITED_K1_SEARCH"
)
STRUCTURAL_UNKNOWN_REASONS = frozenset(
    {
        "MOVEMENT_LOOKUP_UNRESOLVED",
        "TURN_GEOMETRY_UNKNOWN",
        "CONSERVATIVE_LEFT_UNKNOWN_CONTROL",
    }
)


def _json_list(value: Any) -> list[str]:
    if value is None or (not isinstance(value, (str, list, tuple)) and pd.isna(value)):
        return []
    parsed = json.loads(value) if isinstance(value, str) else list(value)
    if not isinstance(parsed, list):
        raise Stage3S2AError("reason-code field must be a JSON list")
    return sorted({str(item) for item in parsed})


def _json(values: Iterable[str]) -> str:
    return json.dumps(sorted({str(value) for value in values}), separators=(",", ":"))


def decode_polyline6(encoded: str) -> list[tuple[float, float]]:
    """Decode a Valhalla precision-6 polyline as ``(lon, lat)`` pairs."""
    coordinates: list[tuple[float, float]] = []
    index = latitude = longitude = 0
    while index < len(encoded):
        deltas: list[int] = []
        for _ in range(2):
            shift = result = 0
            while True:
                if index >= len(encoded):
                    raise Stage3S2AError("truncated Valhalla encoded shape")
                value = ord(encoded[index]) - 63
                index += 1
                result |= (value & 0x1F) << shift
                shift += 5
                if value < 0x20:
                    break
            deltas.append(~(result >> 1) if result & 1 else result >> 1)
        latitude += deltas[0]
        longitude += deltas[1]
        coordinates.append((longitude / 1_000_000.0, latitude / 1_000_000.0))
    if len(coordinates) < 2:
        raise Stage3S2AError("Valhalla route shape has fewer than two points")
    return coordinates


def fallback_triggered(original_hard_state: str) -> bool:
    return str(original_hard_state) == "INFEASIBLE"


def within_distance_bound(candidate_m: float, original_m: float, bound: float = 1.25) -> bool:
    return bool(
        np.isfinite(candidate_m)
        and np.isfinite(original_m)
        and original_m > 0
        and candidate_m <= bound * original_m
    )


def candidate_hard_state(
    hard_reasons: Sequence[str], structural_unknown_reasons: Sequence[str]
) -> str:
    """Soft envelope and dynamic evidence never enter this state reducer."""
    if hard_reasons:
        return "INFEASIBLE"
    if structural_unknown_reasons:
        return "UNKNOWN"
    return "FEASIBLE"


def final_hard_state(original_hard_state: str, fallback_selected: bool) -> str:
    """Existence is provable by one success; K=1 failure remains unknown."""
    if str(original_hard_state) != "INFEASIBLE":
        return str(original_hard_state)
    return "FEASIBLE" if fallback_selected else "UNKNOWN"


def select_hard_feasible_candidate(candidates: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Select by frozen M3 P50 then distance; one unknown-P50 candidate is allowed."""
    eligible = [row for row in candidates if row.get("hard_state") == "FEASIBLE"]
    if not eligible:
        return None
    timed = [
        row
        for row in eligible
        if pd.notna(row.get("service_time_p50_s"))
        and np.isfinite(float(row["service_time_p50_s"]))
    ]
    if timed:
        return min(
            timed,
            key=lambda row: (
                float(row["service_time_p50_s"]),
                float(row["distance_m"]),
                str(row.get("route_reference", "")),
            ),
        )
    if len(eligible) == 1:
        return eligible[0]
    # The frozen baseline emits K=1.  Refuse to invent a ranking if a future
    # configuration supplies multiple candidates without frozen M3 evidence.
    return None


class ValhallaFallbackRouter:
    """One deterministic auto route plus exact same-engine edge attribution."""

    def __init__(self, config_path: Path, valid_edge_ids: set[int]) -> None:
        from valhalla import Actor

        self.actor = Actor(str(config_path.resolve()))
        self.valid_edge_ids = valid_edge_ids

    def _edge_walk(self, shape: str) -> list[int]:
        points = decode_polyline6(shape)
        request = {
            "shape": [{"lon": lon, "lat": lat} for lon, lat in points],
            "costing": "auto",
            "shape_match": "edge_walk",
            "filters": {
                "action": "include",
                "attributes": ["edge.id", "edge.length"],
            },
        }
        response = self.actor.trace_attributes(request)
        edges = response.get("edges") or []
        identifiers = [int(edge["id"]) for edge in edges if edge.get("id") is not None]
        if not identifiers or len(identifiers) != len(edges):
            raise Stage3S2AError("Valhalla edge_walk did not return complete edge IDs")
        if any(identifier not in self.valid_edge_ids for identifier in identifiers):
            raise Stage3S2AError("Valhalla candidate contains an edge outside frozen Stage3 network")
        return identifiers

    def route(
        self,
        *,
        order_id: str,
        origin_lon: float,
        origin_lat: float,
        destination_lon: float,
        destination_lat: float,
        decision_time: float,
        original_distance_m: float,
        distance_bound: float,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        local_time = datetime.fromtimestamp(float(decision_time), tz=timezone.utc).astimezone(
            ZoneInfo("Asia/Shanghai")
        )
        request = {
            "locations": [
                {"lon": float(origin_lon), "lat": float(origin_lat), "type": "break"},
                {
                    "lon": float(destination_lon),
                    "lat": float(destination_lat),
                    "type": "break",
                },
            ],
            "costing": "auto",
            "units": "kilometers",
            "directions_type": "none",
            "date_time": {"type": 1, "value": local_time.strftime("%Y-%m-%dT%H:%M")},
        }
        try:
            response = self.actor.route(request)
            trip = response["trip"]
            legs = trip["legs"]
            summary = trip["summary"]
            if len(legs) != 1 or int(trip.get("status", 0)) != 0:
                raise Stage3S2AError("Valhalla did not return one successful route leg")
            distance_m = float(summary["length"]) * 1000.0
            engine_time_s = float(summary["time"])
            if not within_distance_bound(distance_m, original_distance_m, distance_bound):
                return {
                    "order_id": order_id,
                    "candidate_status": "DISTANCE_BOUND_REJECTED",
                    "candidate_count": 0,
                    "candidate_distance_m": distance_m,
                    "fallback_distance_ratio": distance_m / original_distance_m,
                    "valhalla_route_time_s": engine_time_s,
                    "route_reference": None,
                    "failure_reason": "CANDIDATE_DISTANCE_RATIO_EXCEEDS_BOUND",
                }, []
            identifiers = self._edge_walk(str(legs[0]["shape"]))
            uids = [stage3_edge_uid(identifier) for identifier in identifiers]
            digest = hashlib.sha256("|".join(uids).encode()).hexdigest()[:20]
            reference = f"FALLBACK:{order_id}:{digest}"
            route_rows = [
                {
                    "date": TEST_DATE,
                    "order_id": order_id,
                    "route_reference": reference,
                    "route_sequence": sequence,
                    "stage3_edge_uid": uid,
                    "valhalla_directed_edge_id": identifier,
                }
                for sequence, (uid, identifier) in enumerate(zip(uids, identifiers, strict=True))
            ]
            return {
                "order_id": order_id,
                "candidate_status": "RESOLVED_BOUNDED",
                "candidate_count": 1,
                "candidate_distance_m": distance_m,
                "fallback_distance_ratio": distance_m / original_distance_m,
                "valhalla_route_time_s": engine_time_s,
                "route_reference": reference,
                "failure_reason": None,
            }, route_rows
        except Exception as exc:
            return {
                "order_id": order_id,
                "candidate_status": "UNRESOLVED",
                "candidate_count": 0,
                "candidate_distance_m": np.nan,
                "fallback_distance_ratio": np.nan,
                "valhalla_route_time_s": np.nan,
                "route_reference": None,
                "failure_reason": f"{type(exc).__name__}:{exc}",
            }, []


def _original_distances(root: Path) -> pd.DataFrame:
    route = pd.read_parquet(
        root / ROUTE_REL,
        columns=["order_id", "route_part_length_m", "decision_time"],
    )
    result = route.groupby("order_id", sort=False).agg(
        original_route_distance_m=("route_part_length_m", "sum"),
        decision_time=("decision_time", "first"),
    ).reset_index()
    result["order_id"] = result["order_id"].astype(str)
    return result


def _trigger_orders(root: Path) -> pd.DataFrame:
    original = pd.read_parquet(
        root / ORIGINAL_SUITABILITY_REL,
        columns=["order_id", "profile_id", "hard_state"],
    )
    triggered = original.loc[original["hard_state"].eq("INFEASIBLE"), "order_id"].astype(str).drop_duplicates()
    distances = _original_distances(root)
    od = pd.read_parquet(
        root / OD_MANIFEST_REL,
        columns=["order_id", "start_lon", "start_lat", "end_lon", "end_lat", "start_time"],
    )
    od["order_id"] = od["order_id"].astype(str)
    result = pd.DataFrame({"order_id": triggered}).merge(
        distances, on="order_id", how="left", validate="one_to_one"
    ).merge(od, on="order_id", how="left", validate="one_to_one")
    if len(result) != len(triggered) or result.isna().any().any():
        raise Stage3S2AError("fallback trigger orders do not reconcile with frozen OD manifest")
    lon, lat = gcj02_to_wgs84(result["start_lon"].to_numpy(), result["start_lat"].to_numpy())
    result["origin_lon"] = lon
    result["origin_lat"] = lat
    lon, lat = gcj02_to_wgs84(result["end_lon"].to_numpy(), result["end_lat"].to_numpy())
    result["destination_lon"] = lon
    result["destination_lat"] = lat
    if not np.allclose(result["decision_time"], result["start_time"], atol=1.0):
        raise Stage3S2AError("Test31 decision time does not agree with frozen raw-order start time")
    return result.sort_values("order_id", kind="stable").reset_index(drop=True)


def generate_candidates(root: Path, config: Mapping[str, Any], *, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    status_path = root / FALLBACK_STATUS_REL
    route_path = root / FALLBACK_ROUTE_REL
    if status_path.is_file() and route_path.is_file() and not force:
        return pd.read_parquet(status_path), pd.read_parquet(route_path)
    triggers = _trigger_orders(root)
    edges = pd.read_parquet(
        root / UPSTREAM_REL["full_network_edges"],
        columns=["stage3_edge_uid", "valhalla_directed_edge_id", "auto_routable"],
    )
    if not edges["auto_routable"].all() or edges["valhalla_directed_edge_id"].duplicated().any():
        raise Stage3S2AError("frozen full-network edge identity/routability invariant failed")
    valid_ids = set(edges["valhalla_directed_edge_id"].astype(np.int64))
    router = ValhallaFallbackRouter(Path(config["valhalla_config"]), valid_ids)
    output = root / OUTPUT_REL
    batch_root = output / "_fallback_batches"
    if force and batch_root.exists():
        shutil.rmtree(batch_root)
    batch_root.mkdir(parents=True, exist_ok=True)
    completed: set[str] = set()
    for path in batch_root.glob("status_*.parquet"):
        completed.update(pd.read_parquet(path, columns=["order_id"])["order_id"].astype(str))
    pending = triggers.loc[~triggers["order_id"].isin(completed)]
    batch_size = int(config.get("batch_size", 250))
    batch_number = len(list(batch_root.glob("status_*.parquet")))
    for start in range(0, len(pending), batch_size):
        status_rows: list[dict[str, Any]] = []
        route_rows: list[dict[str, Any]] = []
        for row in pending.iloc[start : start + batch_size].itertuples(index=False):
            status, routed = router.route(
                order_id=str(row.order_id),
                origin_lon=float(row.origin_lon),
                origin_lat=float(row.origin_lat),
                destination_lon=float(row.destination_lon),
                destination_lat=float(row.destination_lat),
                decision_time=float(row.decision_time),
                original_distance_m=float(row.original_route_distance_m),
                distance_bound=float(config["maximum_fallback_distance_ratio"]),
            )
            status_rows.append(status)
            route_rows.extend(routed)
        atomic_parquet(batch_root / f"status_{batch_number:05d}.parquet", pd.DataFrame(status_rows))
        if route_rows:
            atomic_parquet(batch_root / f"routes_{batch_number:05d}.parquet", pd.DataFrame(route_rows))
        batch_number += 1
    status_parts = sorted(batch_root.glob("status_*.parquet"))
    route_parts = sorted(batch_root.glob("routes_*.parquet"))
    statuses = pd.concat([pd.read_parquet(path) for path in status_parts], ignore_index=True)
    routes = (
        pd.concat([pd.read_parquet(path) for path in route_parts], ignore_index=True)
        if route_parts
        else pd.DataFrame(
            columns=[
                "date",
                "order_id",
                "route_reference",
                "route_sequence",
                "stage3_edge_uid",
                "valhalla_directed_edge_id",
            ]
        )
    )
    statuses = statuses.drop_duplicates("order_id", keep="last").sort_values("order_id")
    if len(statuses) != len(triggers):
        raise Stage3S2AError("fallback candidate generation did not reconcile trigger orders")
    if len(routes) and routes.duplicated(["order_id", "route_sequence"]).any():
        raise Stage3S2AError("fallback route edge identity is duplicated")
    atomic_parquet(status_path, statuses.reset_index(drop=True))
    atomic_parquet(route_path, routes.sort_values(["order_id", "route_sequence"]).reset_index(drop=True))
    shutil.rmtree(batch_root)
    return statuses.reset_index(drop=True), routes.reset_index(drop=True)


def _candidate_profile_table(
    root: Path, statuses: pd.DataFrame, routes: pd.DataFrame
) -> pd.DataFrame:
    accepted = statuses[statuses["candidate_status"].eq("RESOLVED_BOUNDED")].copy()
    if accepted.empty:
        return pd.DataFrame(columns=["order_id", "profile_id", "hard_state"])
    accepted_ids = set(accepted["order_id"].astype(str))
    route = routes[routes["order_id"].astype(str).isin(accepted_ids)].copy()
    typed = route[["date", "order_id", "route_sequence", "stage3_edge_uid"]].rename(
        columns={"stage3_edge_uid": "resolved_stage3_edge_uid"}
    )
    typed["route_token_type"] = "FULL_NETWORK_EDGE"
    boundary = pd.read_parquet(root / UPSTREAM_REL["edge_complex_boundary_index"])
    movements = pd.read_parquet(root / UPSTREAM_REL["route_movement_lookup"])
    encounters = parse_route_complex_encounters(typed, boundary, movements)
    static = _full_static_reference(root)
    detail = _encounter_detail(root, encounters, static)
    encounter_descriptor = _encounter_descriptor(
        accepted["order_id"].astype(str), detail
    )
    speed = pd.read_parquet(root / UPSTREAM_REL["speed_domain"])
    speed_descriptor, _ = _speed_descriptor(typed, speed)
    token_counts = typed.groupby("order_id", sort=False).size().rename("route_token_count").reset_index()
    descriptor = accepted[[
        "order_id",
        "candidate_distance_m",
        "fallback_distance_ratio",
        "route_reference",
    ]].merge(token_counts, on="order_id", validate="one_to_one").merge(
        speed_descriptor, on="order_id", validate="one_to_one"
    ).merge(encounter_descriptor, on="order_id", validate="one_to_one")
    descriptor["full_network_edge_token_count"] = descriptor["route_token_count"]
    descriptor["dynamic_complete"] = False
    for dimension in DYNAMIC_DIMS:
        for metric in ("E", "Q", "C"):
            descriptor[f"{dimension}_{metric}"] = np.nan
    profile_rows = descriptor.loc[descriptor.index.repeat(len(PROFILES))].reset_index(drop=True)
    profile_rows["profile_id"] = np.tile(np.asarray(PROFILES), len(descriptor))
    profiles = _profile_map(read_json(root / PROFILE_REL))
    profile_rows = _ratio_vectors(profile_rows, profiles)

    hard_map: dict[tuple[str, str], tuple[str, str, str]] = {}
    detail_groups = {str(key): value for key, value in detail.groupby("order_id", sort=False)}
    for order_id in descriptor["order_id"].astype(str):
        group = detail_groups.get(order_id, detail.iloc[0:0])
        for profile_id in PROFILES:
            hard: set[str] = set()
            unknown: set[str] = set()
            for encounter in group.to_dict("records"):
                for atomic in evaluate_movement_atomic_checks(encounter, profile_id):
                    reason = atomic.get("reason_code")
                    if not reason:
                        continue
                    if reason in HARD_REASON_CODES:
                        hard.add(str(reason))
                    elif reason in STRUCTURAL_UNKNOWN_REASONS:
                        unknown.add(str(reason))
            hard_map[(order_id, profile_id)] = (
                candidate_hard_state(sorted(hard), sorted(unknown)),
                _json(hard),
                _json(unknown),
            )
    states = [hard_map[(str(row.order_id), str(row.profile_id))] for row in profile_rows.itertuples()]
    profile_rows["hard_state"] = [value[0] for value in states]
    profile_rows["hard_reason_codes"] = [value[1] for value in states]
    profile_rows["structural_unknown_reason_codes"] = [value[2] for value in states]
    soft_codes: list[str] = []
    for row in profile_rows.itertuples():
        values = {
            "SOFT_STATIC_A_ENVELOPE_EXCEEDED": row.static_A_ratio,
            "SOFT_STATIC_M_ENVELOPE_EXCEEDED": row.static_M_ratio,
            "SOFT_STATIC_D_ENVELOPE_EXCEEDED": row.static_D_ratio,
            "SOFT_STATIC_L_ENVELOPE_EXCEEDED": row.static_L_ratio,
            "SOFT_SPEED_ENVELOPE_EXCEEDED": row.rho_speed,
        }
        soft_codes.append(_json(code for code, value in values.items() if pd.notna(value) and float(value) > 1.0))
    profile_rows["soft_reason_codes"] = soft_codes
    profile_rows["dynamic_evidence_state"] = "UNKNOWN"
    profile_rows["service_time_p50_s"] = np.nan
    return profile_rows


def _build_interface(
    root: Path,
    statuses: pd.DataFrame,
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    original = pd.read_parquet(root / ORIGINAL_SUITABILITY_REL).copy()
    descriptors = pd.read_parquet(
        root / ORIGINAL_DESCRIPTOR_REL,
        columns=["order_id", "predicted_route_time_p50_s"],
    )
    distances = _original_distances(root)[["order_id", "original_route_distance_m"]]
    original = original.merge(descriptors, on="order_id", validate="many_to_one").merge(
        distances, on="order_id", validate="many_to_one"
    )
    status_columns = [
        "order_id",
        "candidate_count",
        "candidate_distance_m",
        "fallback_distance_ratio",
        "route_reference",
        "candidate_status",
    ]
    original = original.merge(statuses[status_columns], on="order_id", how="left", validate="many_to_one")
    candidate_columns = [
        "order_id",
        "profile_id",
        "hard_state",
        "rho_static",
        "rho_dynamic",
        "rho_speed",
        "rho_overall",
        "static_vector",
        "dynamic_vector",
        "hard_reason_codes",
        "structural_unknown_reason_codes",
        "soft_reason_codes",
        "service_time_p50_s",
    ]
    renamed = {column: f"candidate_{column}" for column in candidate_columns if column not in {"order_id", "profile_id"}}
    original = original.merge(
        candidates[candidate_columns].rename(columns=renamed),
        on=["order_id", "profile_id"],
        how="left",
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for row in original.itertuples(index=False):
        attempted = fallback_triggered(row.hard_state)
        candidate_available = attempted and int(row.candidate_count or 0) == 1
        candidate_record = None
        if candidate_available and pd.notna(row.candidate_hard_state):
            candidate_record = {
                "hard_state": row.candidate_hard_state,
                "service_time_p50_s": row.candidate_service_time_p50_s,
                "distance_m": row.candidate_distance_m,
                "route_reference": row.route_reference,
            }
        selected = select_hard_feasible_candidate(
            [candidate_record] if candidate_record is not None else []
        )
        base = {
            "date": TEST_DATE,
            "order_id": str(row.order_id),
            "profile_id": str(row.profile_id),
            "fallback_attempted": attempted,
            "fallback_candidate_count": int(row.candidate_count or 0) if attempted else 0,
            "fallback_hard_feasible_count": int(selected is not None),
            "original_route_hard_state": str(row.hard_state),
            "original_route_hard_reason_codes": row.hard_reason_codes,
            "original_route_rho_static": row.rho_static,
            "original_route_rho_dynamic": row.rho_dynamic,
            "original_route_rho_speed": row.rho_speed,
            "fallback_distance_ratio": row.fallback_distance_ratio if attempted else np.nan,
        }
        if not attempted:
            complete = row.hard_state != "UNKNOWN" and all(
                pd.notna(value) and np.isfinite(float(value))
                for value in (row.rho_static, row.rho_dynamic, row.rho_speed)
            )
            rows.append({
                **base,
                "selected_route_type": "ORIGINAL",
                "hard_state": final_hard_state(row.hard_state, False),
                "fallback_search_state": "NOT_ATTEMPTED_ORIGINAL_RETAINED",
                "evidence_complete": bool(complete),
                "selected_route_distance_m": row.original_route_distance_m,
                "selected_service_time_p50_s": row.predicted_route_time_p50_s,
                "rho_static": row.rho_static,
                "rho_dynamic": row.rho_dynamic,
                "rho_speed": row.rho_speed,
                "rho_overall": row.rho_overall,
                "static_vector": row.static_vector,
                "dynamic_vector": row.dynamic_vector,
                "hard_reason_codes": row.hard_reason_codes,
                "unknown_reason_codes": row.unknown_reason_codes,
                "soft_reason_codes": row.soft_exceedance_reason_codes,
                "selected_route_reference": f"ORIGINAL:{row.order_id}",
            })
        elif selected is not None:
            unknown = set(_json_list(row.candidate_structural_unknown_reason_codes))
            unknown.add(DYNAMIC_UNKNOWN_REASON)
            rows.append({
                **base,
                "selected_route_type": "FALLBACK",
                "hard_state": final_hard_state(row.hard_state, True),
                "fallback_search_state": "HARD_FEASIBLE_FOUND",
                "evidence_complete": False,
                "selected_route_distance_m": row.candidate_distance_m,
                "selected_service_time_p50_s": np.nan,
                "rho_static": row.candidate_rho_static,
                "rho_dynamic": np.nan,
                "rho_speed": row.candidate_rho_speed,
                "rho_overall": np.nan,
                "static_vector": row.candidate_static_vector,
                "dynamic_vector": row.candidate_dynamic_vector,
                "hard_reason_codes": row.candidate_hard_reason_codes,
                "unknown_reason_codes": _json(unknown),
                "soft_reason_codes": row.candidate_soft_reason_codes,
                "selected_route_reference": row.route_reference,
            })
        else:
            unknown = set(_json_list(row.unknown_reason_codes))
            unknown.add(LIMITED_SEARCH_NOT_ESTABLISHED_REASON)
            if candidate_available and str(row.candidate_hard_state) == "UNKNOWN":
                unknown.update(_json_list(row.candidate_structural_unknown_reason_codes))
            rows.append({
                **base,
                "selected_route_type": "NONE",
                "hard_state": final_hard_state(row.hard_state, False),
                "fallback_search_state": "NOT_ESTABLISHED_UNDER_LIMITED_K1_SEARCH",
                "evidence_complete": False,
                "selected_route_distance_m": np.nan,
                "selected_service_time_p50_s": np.nan,
                "rho_static": np.nan,
                "rho_dynamic": np.nan,
                "rho_speed": np.nan,
                "rho_overall": np.nan,
                "static_vector": None,
                "dynamic_vector": None,
                "hard_reason_codes": "[]",
                "unknown_reason_codes": _json(unknown),
                "soft_reason_codes": "[]",
                "selected_route_reference": None,
            })
    result = pd.DataFrame(rows).loc[:, FINAL_COLUMNS]
    if (
        len(result) != EXPECTED_ORDER_PROFILE_COUNT
        or result["order_id"].nunique() != EXPECTED_ORDER_COUNT
        or result.duplicated(["order_id", "profile_id"]).any()
        or set(result["profile_id"]) != set(PROFILES)
    ):
        raise Stage3S2AError("final Stage3 interface is not exactly 30,000 x 3")
    return result.sort_values(["order_id", "profile_id"], kind="stable").reset_index(drop=True)


def _reason_family(code: str) -> str:
    if code == "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE":
        return "reverse_direction"
    if code == "CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE":
        return "roundabout"
    if code == "CERTIFIED_MOVEMENT_PROHIBITION":
        return "certified_restriction"
    return "maneuver"


def _summary(interface: pd.DataFrame, original: pd.DataFrame) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile_id in PROFILES:
        frame = interface[interface["profile_id"].eq(profile_id)]
        source = original[original["profile_id"].eq(profile_id)]
        original_counts = source["hard_state"].value_counts()
        final_counts = frame["hard_state"].value_counts()
        recovery: dict[str, dict[str, int]] = {}
        for family in ("reverse_direction", "maneuver", "roundabout", "certified_restriction"):
            mask = source["hard_reason_codes"].map(
                lambda value, f=family: any(_reason_family(code) == f for code in _json_list(value))
            )
            ids = set(source.loc[mask, "order_id"].astype(str))
            recovered = frame["order_id"].astype(str).isin(ids) & frame["selected_route_type"].eq("FALLBACK")
            recovery[family] = {"triggered": len(ids), "recovered": int(recovered.sum())}
        profiles[profile_id] = {
            "original_hard_state_counts": {
                state: int(original_counts.get(state, 0))
                for state in ("FEASIBLE", "UNKNOWN", "INFEASIBLE")
            },
            "fallback_attempted_count": int(frame["fallback_attempted"].sum()),
            "fallback_found_hard_feasible_count": int(frame["fallback_hard_feasible_count"].sum()),
            "final_hard_state_counts": {
                state: int(final_counts.get(state, 0))
                for state in ("FEASIBLE", "UNKNOWN", "INFEASIBLE")
            },
            "selected_route_type_counts": {
                key: int(frame["selected_route_type"].value_counts().get(key, 0))
                for key in ("ORIGINAL", "FALLBACK", "NONE")
            },
            "fallback_not_established_under_limited_search_count": int(
                frame["fallback_search_state"].eq(
                    "NOT_ESTABLISHED_UNDER_LIMITED_K1_SEARCH"
                ).sum()
            ),
            "fallback_distance_ratio": _distribution(
                frame.loc[frame["selected_route_type"].eq("FALLBACK"), "fallback_distance_ratio"]
            ),
            "rho_static": _distribution(frame["rho_static"]),
            "rho_dynamic": _distribution(frame["rho_dynamic"]),
            "rho_speed": _distribution(frame["rho_speed"]),
            "dynamic_evidence_complete_share": float(frame["rho_dynamic"].notna().mean()),
            "hard_infeasibility_recovery_by_original_reason_family": recovery,
        }
    return {
        "schema_version": "stage3_final_summary.2",
        "phase_status": PHASE_STATUS,
        "date": TEST_DATE,
        "order_count": int(interface["order_id"].nunique()),
        "order_profile_row_count": int(len(interface)),
        "profiles": profiles,
        "stage3_final_frozen": True,
        "stage3_to_stage4_interface_frozen": True,
        "stage4_dispatch_authorized": False,
    }


def _write_docs(root: Path, summary: Mapping[str, Any], manifest: Mapping[str, Any]) -> None:
    methodology = """# Stage3 Final Methodology\n\nStage3 finalization preserves the frozen M3 checkpoint, Train weighted mid-CDF, C/M/A profiles, A/M/D/L definitions, movement rules, restrictions, roundabout semantics, speed caps, and all 36 dynamic caps.\n\nOnly an original-route structural `INFEASIBLE` state triggers fallback. A single deterministic Valhalla `auto` route is requested from the frozen raw-GPS OD at the original decision time. The routed shape is passed back to the same Valhalla engine with `edge_walk`; every returned directed edge ID must exist in the frozen Stage3 full-network table. No reverse overlay, synthetic reverse, geometry-nearest repair, or forward substitution is permitted. Candidates beyond the fixed 1.25 distance ratio are rejected.\n\nCandidate movement evidence reuses the production intersection-complex parser and frozen movement/control/restriction rules. Static, dynamic, and speed envelope exceedance never changes hard feasibility. Because a new route lacks the exact frozen M3 historical feature rows, fallback dynamic evidence and service time remain null; no imputation or historical-route prediction copying is performed.\n\nCandidate selection is frozen as minimum M3 P50 followed by distance. The baseline produces one candidate, so a hard-feasible candidate with unavailable soft dynamic evidence may still be selected, with `evidence_complete=false`.\n"""
    contract = """# Frozen Stage3 → Stage4 Contract\n\nThe canonical input is `stage3/output/odd_tod/final/test31_stage3_to_stage4_interface.parquet`, exactly 30,000 Test31 orders × three C/M/A profiles.\n\n- `hard_state == INFEASIBLE`: Stage4 must forbid the AV assignment.\n- `hard_state == UNKNOWN`: Stage4 chooses whether baseline policy excludes or allows it.\n- `rho_static`, `rho_dynamic`, and `rho_speed` remain separate continuous capability-utilization families. They are not a safety score and must not be collapsed into a Stage3 binary label.\n- `selected_service_time_p50_s == null`: Stage4 may exclude that AV arc under its baseline policy; Stage3 does not impute it.\n- Passenger acceptance is supplied separately by Stage4. Stage3 contains no passenger model or dispatch solver.\n- `selected_route_reference` resolves either to the frozen historical route (`ORIGINAL:<order_id>`) or `test31_fallback_route_edges.parquet` (`FALLBACK:<order_id>:<digest>`).\n\nA fallback means only that a bounded route exists on the frozen AV-routable network under the hypothetical capability profile. It is not AV safety, legal, or commercial certification.\n"""
    methodology += (
        "\nThe frozen baseline produced exactly one candidate and every fallback M3 P50 was unavailable. "
        "Therefore no time ranking occurred: the sole candidate was selected only when structurally hard-feasible. K=1 failure is recorded as not established, never as proof of OD-level infeasibility.\n"
    )
    contract += (
        "\n## Limited-search and Stage4 eligibility semantics\n\n"
        "`selected_route_type=NONE` with `fallback_search_state=NOT_ESTABLISHED_UNDER_LIMITED_K1_SEARCH` means the frozen bounded K=1 procedure did not establish a hard-feasible AV route. It is `hard_state=UNKNOWN`, not proof that no AV route exists for the OD.\n\n"
        "Stage4 should distinguish structural route availability from evidence completeness. Under the conservative baseline, an AV arc is dispatch-ready only when a hard-feasible route is selected and static, dynamic, speed, and service-time evidence are complete; passenger acceptance is then applied separately.\n"
    )
    lines = [
        "# Stage3 Final Summary",
        "",
        "Stage3 finalization is complete. Stage4 dispatch remains unauthorized.",
        "",
        "| Profile | Original F/U/I | Fallback attempted | Hard-feasible recovered | Final F/U/I | ORIGINAL/FALLBACK/NONE | Dynamic complete |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for profile_id in PROFILES:
        item = summary["profiles"][profile_id]
        before = item["original_hard_state_counts"]
        after = item["final_hard_state_counts"]
        selected = item["selected_route_type_counts"]
        lines.append(
            f"| {profile_id} | {before['FEASIBLE']:,}/{before['UNKNOWN']:,}/{before['INFEASIBLE']:,} | "
            f"{item['fallback_attempted_count']:,} | {item['fallback_found_hard_feasible_count']:,} | "
            f"{after['FEASIBLE']:,}/{after['UNKNOWN']:,}/{after['INFEASIBLE']:,} | "
            f"{selected['ORIGINAL']:,}/{selected['FALLBACK']:,}/{selected['NONE']:,} | "
            f"{item['dynamic_evidence_complete_share']:.2%} |"
        )
    lines.extend(
        [
            "",
            "## Fallback distance and selected suitability",
            "",
            "| Profile | Distance ratio p50/p90/p99 | rho static p50/p90 | rho dynamic p50/p90 | rho speed p50/p90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for profile_id in PROFILES:
        item = summary["profiles"][profile_id]
        distance = item["fallback_distance_ratio"]
        static = item["rho_static"]
        dynamic = item["rho_dynamic"]
        speed = item["rho_speed"]
        lines.append(
            f"| {profile_id} | {distance['p50']:.4f}/{distance['p90']:.4f}/{distance['p99']:.4f} | "
            f"{static['p50']:.4f}/{static['p90']:.4f} | "
            f"{dynamic['p50']:.4f}/{dynamic['p90']:.4f} | "
            f"{speed['p50']:.4f}/{speed['p90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recovery by original hard-reason family",
            "",
            "| Profile | Reverse direction | Maneuver | Roundabout | Certified restriction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for profile_id in PROFILES:
        recovery = summary["profiles"][profile_id][
            "hard_infeasibility_recovery_by_original_reason_family"
        ]
        lines.append(
            f"| {profile_id} | {recovery['reverse_direction']['recovered']:,}/{recovery['reverse_direction']['triggered']:,} | "
            f"{recovery['maneuver']['recovered']:,}/{recovery['maneuver']['triggered']:,} | "
            f"{recovery['roundabout']['recovered']:,}/{recovery['roundabout']['triggered']:,} | "
            f"{recovery['certified_restriction']['recovered']:,}/{recovery['certified_restriction']['triggered']:,} |"
        )
    lines.extend(
        [
            "",
            "Fallback dynamic evidence is intentionally unknown when the exact frozen M3 input contract cannot be constructed. This is a declared limitation, not imputed evidence.",
            "",
            f"Manifest artifact hash: `{manifest['artifact_sha256']}`",
            "",
        ]
    )
    atomic_text(root / METHODOLOGY_REL, methodology)
    atomic_text(root / CONTRACT_REL, contract)
    atomic_text(root / REPORT_REL, "\n".join(lines))


def finalize(root: str | Path, *, force: bool = False) -> dict[str, Any]:
    root = Path(root).resolve()
    config = read_json(root / CONFIG_REL)
    if config.get("authorized_base") != AUTHORIZED_BASE or config.get("test_date") != TEST_DATE:
        raise Stage3S2AError("Stage3 finalization configuration is not the authorized freeze")
    if int(config.get("maximum_candidates", -1)) != 1:
        raise Stage3S2AError("minimal baseline must use exactly one deterministic candidate")
    profile_before = sha256_file(root / PROFILE_REL)
    cdf_before = sha256_file(root / CDF_REL)
    checkpoint = root / "stage2/output_v5_2/development/M3/epoch_004.pt"
    if sha256_file(checkpoint) != M3_SHA256:
        raise Stage3S2AError("frozen M3 checkpoint changed")
    statuses, routes = generate_candidates(root, config, force=force)
    candidates = _candidate_profile_table(root, statuses, routes)
    interface = _build_interface(root, statuses, candidates)
    atomic_parquet(root / INTERFACE_REL, interface)
    original = pd.read_parquet(
        root / ORIGINAL_SUITABILITY_REL,
        columns=["order_id", "profile_id", "hard_state", "hard_reason_codes"],
    )
    summary = _summary(interface, original)
    atomic_json(root / SUMMARY_REL, summary)
    profile_after = sha256_file(root / PROFILE_REL)
    cdf_after = sha256_file(root / CDF_REL)
    if profile_before != profile_after or cdf_before != cdf_after:
        raise Stage3S2AError("frozen Stage3 profile or CDF changed during finalization")
    major_inputs = {
        "configuration": source_descriptor(root / CONFIG_REL, root),
        "original_suitability": parquet_descriptor(root / ORIGINAL_SUITABILITY_REL, root),
        "original_route": parquet_descriptor(root / ROUTE_REL, root),
        "raw_order_od_manifest": parquet_descriptor(root / OD_MANIFEST_REL, root),
        "full_network_edges": parquet_descriptor(root / UPSTREAM_REL["full_network_edges"], root),
        "intersection_complexes": parquet_descriptor(root / UPSTREAM_REL["intersection_complexes"], root),
        "intersection_movements": parquet_descriptor(root / UPSTREAM_REL["intersection_movements"], root),
        "profiles": source_descriptor(root / PROFILE_REL, root),
        "train_weighted_mid_cdf": parquet_descriptor(root / CDF_REL, root),
        "frozen_m3_checkpoint": source_descriptor(checkpoint, root),
    }
    manifest = {
        "schema_version": "stage3_final_manifest.2",
        "authorized_base": AUTHORIZED_BASE,
        "phase_status": PHASE_STATUS,
        "frozen_state": {
            "stage0": "FROZEN",
            "stage1": "FROZEN",
            "stage2": "FROZEN",
            "stage3_network": "FROZEN",
            "stage3_intersections": "FROZEN",
            "stage3_capability_profiles": "FROZEN",
            "stage3_original_route_suitability": "FROZEN",
            "stage3_minimal_fallback": "FROZEN",
            "stage3_to_stage4_interface": "FROZEN",
            "stage3_final_frozen": True,
            "stage4_dispatch_authorized": False,
        },
        "policy": {
            "maximum_candidates": 1,
            "maximum_fallback_distance_ratio": float(config["maximum_fallback_distance_ratio"]),
            "hard_trigger_only": True,
            "unknown_does_not_trigger": True,
            "fallback_success_proves_route_existence": True,
            "limited_k1_failure_semantics": "UNKNOWN_NOT_ESTABLISHED",
            "limited_k1_failure_proves_od_infeasible": False,
            "soft_rho_does_not_trigger": True,
            "fallback_dynamic_imputation": False,
            "passenger_model": False,
            "dispatch_solver": False,
        },
        "major_inputs": major_inputs,
        "canonical_products": {
            "interface": parquet_descriptor(root / INTERFACE_REL, root),
            "fallback_candidate_status": parquet_descriptor(root / FALLBACK_STATUS_REL, root),
            "fallback_route_edges": parquet_descriptor(root / FALLBACK_ROUTE_REL, root),
            "summary": source_descriptor(root / SUMMARY_REL, root),
        },
        "profile_sha256_before": profile_before,
        "profile_sha256_after": profile_after,
        "cdf_sha256_before": cdf_before,
        "cdf_sha256_after": cdf_after,
        "summary": summary,
    }
    manifest["artifact_sha256"] = payload_hash(manifest)
    atomic_json(root / MANIFEST_REL, manifest)
    _write_docs(root, summary, manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)
    manifest = finalize(args.root, force=args.force)
    print(json.dumps({
        "phase_status": manifest["phase_status"],
        "interface": manifest["canonical_products"]["interface"],
        "summary": manifest["summary"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
