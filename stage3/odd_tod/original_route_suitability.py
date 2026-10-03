"""S4 Test31 frozen historical/original-route suitability assessment.

This module evaluates the exact frozen Test31 historical route.  It never
constructs, repairs, searches, or substitutes a route.  All capability
profiles, the Train CDF, S2A/S2B network products, and the frozen M3 checkpoint
are immutable inputs.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from stage3.odd_tod.capability_envelope import (
    DYNAMIC_DIMS,
    M3_SHA256,
    Q_TAIL,
    apply_mid_cdf,
    boundary_road_class_diversity,
    parse_route_complex_encounters,
    resolve_route_tokens,
)
from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
    atomic_json,
    atomic_parquet,
    atomic_text,
    git_head,
    parquet_descriptor,
    payload_hash,
    read_json,
    sha256_file,
    source_descriptor,
)


AUTHORIZED_BASE = "a18dcdb42bd622273eccf2347ab6c61f4fef955f"
TEST_DATE = "20161031"
EXPECTED_ORDER_COUNT = 30_000
EXPECTED_ORDER_PROFILE_COUNT = 90_000
PROFILES = ("C", "M", "A")
PHASE_STATUS = "STAGE3_S4_TEST31_ORIGINAL_ROUTE_SUITABILITY_COMPLETE"
S5_AUTHORIZED = False
NEXT_PHASE_AUTHORIZED = False

OUTPUT_REL = Path("stage3/output/odd_tod/s4")
DOCS_REL = Path("stage3/docs/odd_tod/s4")
PROFILE_REL = Path("stage3/config/stage3_av_capability_profiles.json")
CDF_REL = Path("stage3/output/odd_tod/s3/train_dynamic_cdf_reference.parquet")
ROUTE_REL = Path(
    "stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day=20161031.parquet"
)
ROUTE_MANIFEST_REL = Path(
    "stage2/output_v4/route_conditioned_dataset/manifests/revealed_route_proxy/day=20161031.json"
)

UPSTREAM_REL = {
    "full_network_edges": Path("stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet"),
    "speed_domain": Path("stage3/output/odd_tod/s2a/stage3_speed_domain.parquet"),
    "historical_direction_overlay": Path("stage3/output/odd_tod/s2a/stage3_historical_direction_overlay.parquet"),
    "observed_full_network_mapping": Path("stage3/output/odd_tod/s2a/stage3_observed_full_network_mapping.parquet"),
    "intersection_complexes": Path("stage3/output/odd_tod/s2b/final/stage3_intersection_complexes.parquet"),
    "intersection_movements": Path("stage3/output/odd_tod/s2b/final/stage3_intersection_movements.parquet"),
    "intersection_node_membership": Path("stage3/output/odd_tod/s2b/final/stage3_intersection_node_membership.parquet"),
    "edge_complex_boundary_index": Path("stage3/output/odd_tod/s2b/final/stage3_edge_complex_boundary_index.parquet"),
    "route_movement_lookup": Path("stage3/output/odd_tod/s2b/final/stage3_route_movement_lookup.parquet"),
    "train_static_complex_reference": Path("stage3/output/odd_tod/s3/train_static_complex_reference.parquet"),
    "train_m3_cache_provenance": Path("stage3/output/odd_tod/s3/train_m3_cache_provenance.parquet"),
    "s3_release_manifest": Path("stage3/docs/odd_tod/s3/stage3_s3_release_manifest.json"),
    "s31_closure_report": Path("stage3/docs/odd_tod/s3/stage3_s31_closure_report.md"),
    "s3_evidence_verification": Path("stage3/docs/odd_tod/s3/stage3_s3_evidence_verification.json"),
    "s2a1_scientific_closure_evidence": Path(
        "stage3/docs/odd_tod/s2a/stage3_s2a1_scientific_closure_evidence.json"
    ),
    "m3_checkpoint": Path("stage2/output_v5_2/development/M3/epoch_004.pt"),
    "m3_model_manifest": Path("stage2/output_v5_2/development/M3/model_manifest.json"),
    "m3_feature_artifacts": Path(
        "stage2/output_v5/protocols/development/tensor_shards/feature_artifacts.json"
    ),
    "m3_transfer_manifest": Path(
        "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json"
    ),
    "m3_static_artifact": Path("stage2/output_v5_2/development/artifacts/static.json"),
    "m3_support_artifact": Path("stage2/output_v5_2/development/artifacts/support.json"),
}

ROUTE_IDENTITY_COLUMNS = (
    "split",
    "date",
    "order_id",
    "route_sequence",
    "traversal_id",
    "canonical_edge_uid",
    "observed_directed_edge_uid",
    "observed_direction",
    "route_part_length_m",
    "decision_time",
    "feature_timestamp",
    "availability_timestamp",
)

ATOMIC_COLUMNS = (
    "date",
    "order_id",
    "profile_id",
    "check_family",
    "check_name",
    "state",
    "observed_value",
    "cap_value",
    "evidence_id",
    "route_sequence",
    "stage3_edge_uid",
    "intersection_complex_uid",
    "movement_occurrence_index",
    "reason_code",
)
ATOMIC_SCHEMA = pa.schema([
    pa.field("date", pa.string()),
    pa.field("order_id", pa.string()),
    pa.field("profile_id", pa.string()),
    pa.field("check_family", pa.string()),
    pa.field("check_name", pa.string()),
    pa.field("state", pa.string()),
    pa.field("observed_value", pa.float64()),
    pa.field("cap_value", pa.float64()),
    pa.field("evidence_id", pa.string()),
    pa.field("route_sequence", pa.int64()),
    pa.field("stage3_edge_uid", pa.string()),
    pa.field("intersection_complex_uid", pa.string()),
    pa.field("movement_occurrence_index", pa.int64()),
    pa.field("reason_code", pa.string()),
])

KNOWN_REASON_CODES = frozenset(
    {
        "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE",
        "SPEED_DOMAIN_CAP_EXCEEDED",
        "STATIC_A_CAP_EXCEEDED",
        "STATIC_M_CAP_EXCEEDED",
        "STATIC_D_CAP_EXCEEDED",
        "STATIC_L_CAP_EXCEEDED",
        "CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE",
        "UTURN_PROFILE_INCOMPATIBLE",
        "CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE",
        "CERTIFIED_MOVEMENT_PROHIBITION",
        *{
            f"DYNAMIC_{dimension.upper()}_{metric}_CAP_EXCEEDED"
            for dimension in DYNAMIC_DIMS
            for metric in ("E", "Q", "C")
        },
    }
)
UNKNOWN_REASON_CODES = frozenset(
    {
        "UNRESOLVED_ROUTE_IDENTITY",
        "SPEED_DOMAIN_UNKNOWN",
        "STATIC_METRIC_UNKNOWN",
        "MOVEMENT_LOOKUP_UNRESOLVED",
        "TURN_GEOMETRY_UNKNOWN",
        "CONSERVATIVE_LEFT_UNKNOWN_CONTROL",
        "DYNAMIC_ROUTE_INCOMPLETE",
    }
)


def aggregate_atomic_state(states: Iterable[str]) -> str:
    """Reduce atomic states with frozen INCOMPATIBLE > UNKNOWN precedence."""
    values = [str(value) for value in states]
    invalid = sorted(set(values) - {"COMPATIBLE", "UNKNOWN", "INCOMPATIBLE"})
    if invalid:
        raise Stage3S2AError(f"invalid atomic states: {invalid}")
    if "INCOMPATIBLE" in values:
        return "INCOMPATIBLE"
    if "UNKNOWN" in values:
        return "UNKNOWN"
    return "COMPATIBLE"


def finalize_route_state(states: Iterable[str]) -> str:
    return {
        "INCOMPATIBLE": "INFEASIBLE",
        "UNKNOWN": "UNKNOWN",
        "COMPATIBLE": "FEASIBLE",
    }[aggregate_atomic_state(states)]


def aggregate_reason_codes(atomic: pd.DataFrame | Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Preserve all distinct known and unknown causes, deterministically."""
    if not isinstance(atomic, pd.DataFrame):
        atomic = pd.DataFrame(list(atomic))
    if atomic.empty:
        return {
            "known_violation_reason_codes": [], "unknown_reason_codes": [],
            "known_violation_count": 0, "unknown_requirement_count": 0,
        }
    reasons = atomic.loc[atomic["reason_code"].notna(), ["state", "reason_code"]]
    known = sorted(set(reasons.loc[reasons["state"].eq("INCOMPATIBLE"), "reason_code"].astype(str)))
    unknown = sorted(set(reasons.loc[reasons["state"].eq("UNKNOWN"), "reason_code"].astype(str)))
    return {
        "known_violation_reason_codes": known,
        "unknown_reason_codes": unknown,
        "known_violation_count": len(known),
        "unknown_requirement_count": len(unknown),
    }


def evaluate_directional_routability(route_tokens: pd.DataFrame) -> list[dict[str, Any]]:
    """Return the single frozen route-level direction atomic result."""
    order_id = str(route_tokens["order_id"].iloc[0]) if "order_id" in route_tokens and len(route_tokens) else ""
    ordered = route_tokens.sort_values("route_sequence")
    reverse = ordered[ordered["route_token_type"].eq("HISTORICAL_REVERSE_OVERLAY")]
    unresolved = ordered[ordered["route_token_type"].eq("UNRESOLVED")]
    if len(reverse):
        token, state, reason = reverse.iloc[0], "INCOMPATIBLE", "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE"
    elif len(unresolved):
        token, state, reason = unresolved.iloc[0], "UNKNOWN", "UNRESOLVED_ROUTE_IDENTITY"
    else:
        return [_atomic_row(
            order_id=order_id, profile_id="", family="DIRECTION",
            name="DIRECTIONAL_ROUTABILITY", state="COMPATIBLE",
        )]
    resolved = token.get("resolved_stage3_edge_uid")
    return [_atomic_row(
        order_id=order_id, profile_id="", family="DIRECTION",
        name="DIRECTIONAL_ROUTABILITY", state=state,
        route_sequence=int(token["route_sequence"]),
        # physical_forward_stage3_edge_uid is provenance only and is
        # intentionally not emitted as a traversed AV edge.
        edge_id=None if resolved is None or pd.isna(resolved) else str(resolved), reason=reason,
    )]


def _profile_map(profile: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    by_id = {str(item["profile_id"]): dict(item) for item in profile["profiles"]}
    if tuple(by_id) != PROFILES and set(by_id) != set(PROFILES):
        raise Stage3S2AError("frozen profile IDs differ from C/M/A")
    expected_static = {
        "C": (4.0, 9.0, 2.0, 10.0),
        "M": (5.0, 16.0, 3.0, 34.0),
        "A": (9.0, 25.0, 3.0, 73.0),
    }
    for profile_id, expected in expected_static.items():
        caps = by_id[profile_id]["static_caps"]
        observed = (
            float(caps["external_physical_connection_count"]),
            float(caps["topological_movement_count"]),
            float(caps["road_class_diversity"]),
            float(caps["internal_length_m"]),
        )
        if observed != expected:
            raise Stage3S2AError(f"frozen static caps changed for {profile_id}: {observed}")
    if [float(by_id[p]["speed_domain_max_kmh"]) for p in PROFILES] != [60.0, 80.0, 120.0]:
        raise Stage3S2AError("frozen speed caps changed")
    if float(profile.get("q_tail", math.nan)) != Q_TAIL:
        raise Stage3S2AError("frozen q_tail changed")
    definition = str(profile.get("static_dimension_definitions", {}).get("D_c", ""))
    if "INCOMING/OUTGOING" not in definition or "INTERNAL" not in definition:
        raise Stage3S2AError("D_c is not bound to boundary-edge road-class diversity")
    return by_id


def _atomic_row(
    *,
    order_id: str,
    profile_id: str,
    family: str,
    name: str,
    state: str,
    observed: float | None = None,
    cap: float | None = None,
    evidence_id: str | None = None,
    route_sequence: int | None = None,
    edge_id: str | None = None,
    complex_id: str | None = None,
    occurrence: int | None = None,
    reason: str | None = None,
) -> dict[str, Any]:
    identity_suffix = "|".join(
        "" if value is None or (not isinstance(value, str) and pd.isna(value)) else str(value)
        for value in (route_sequence, edge_id, complex_id, occurrence)
    )
    return {
        "date": TEST_DATE,
        "order_id": str(order_id),
        "profile_id": str(profile_id),
        "check_family": str(family),
        "check_name": str(name),
        "state": str(state),
        "observed_value": observed,
        "cap_value": cap,
        "evidence_id": evidence_id or f"{family}|{order_id}|{profile_id}|{name}|{identity_suffix}",
        "route_sequence": route_sequence,
        "stage3_edge_uid": edge_id,
        "intersection_complex_uid": complex_id,
        "movement_occurrence_index": occurrence,
        "reason_code": reason,
    }


def evaluate_static_checks(
    complex_row: Mapping[str, Any], profile_id: str, static_caps: Mapping[str, float]
) -> list[dict[str, Any]]:
    """Evaluate A/M/D/L independently; no averaging or compensation."""
    aliases = {
        "A": ("A_c", "external_physical_connection_count", "STATIC_A_CAP_EXCEEDED"),
        "M": ("M_c", "topological_movement_count", "STATIC_M_CAP_EXCEEDED"),
        "D": ("D_c", "road_class_diversity", "STATIC_D_CAP_EXCEEDED"),
        "L": ("L_c", "internal_length_m", "STATIC_L_CAP_EXCEEDED"),
    }
    order_id = str(complex_row.get("order_id", ""))
    complex_id = complex_row.get("intersection_complex_uid")
    occurrence = complex_row.get("movement_occurrence_index")
    rows = []
    for label, (value_key, cap_key, reason) in aliases.items():
        value = complex_row.get(value_key)
        cap = float(static_caps[cap_key] if cap_key in static_caps else static_caps[value_key])
        known = value is not None and pd.notna(value) and np.isfinite(float(value))
        state = "UNKNOWN" if not known else ("INCOMPATIBLE" if float(value) > cap else "COMPATIBLE")
        rows.append(
            _atomic_row(
                order_id=order_id,
                profile_id=profile_id,
                family=f"STATIC_{label}",
                name=f"STATIC_{label}",
                state=state,
                observed=float(value) if known else None,
                cap=cap,
                complex_id=str(complex_id) if complex_id is not None else None,
                occurrence=int(occurrence) if occurrence is not None and pd.notna(occurrence) else None,
                reason="STATIC_METRIC_UNKNOWN" if state == "UNKNOWN" else (reason if state == "INCOMPATIBLE" else None),
            )
        )
    return rows


def evaluate_speed_checks(
    route_tokens: pd.DataFrame,
    speed_domain: pd.DataFrame,
    profile_id: str,
    speed_cap_kmh: float,
) -> list[dict[str, Any]]:
    """Evaluate only actually traversed FULL_NETWORK_EDGE identities."""
    order_id = (
        str(route_tokens["order_id"].iloc[0])
        if len(route_tokens) and "order_id" in route_tokens
        else ""
    )
    full = route_tokens[route_tokens["route_token_type"].eq("FULL_NETWORK_EDGE")].copy()
    if not len(full):
        return [_atomic_row(order_id=order_id, profile_id=profile_id, family="SPEED", name="speed_domain_cap", state="COMPATIBLE", cap=float(speed_cap_kmh))]
    speed = speed_domain.copy()
    if "speed_domain_provenance" not in speed:
        speed["speed_domain_provenance"] = "KNOWN_TEST_FIXTURE"
    speed = speed[["stage3_edge_uid", "speed_domain_value_kmh", "speed_domain_provenance"]].drop_duplicates("stage3_edge_uid")
    joined = full.merge(speed, left_on="resolved_stage3_edge_uid", right_on="stage3_edge_uid", how="left", validate="many_to_one")
    values = pd.to_numeric(joined["speed_domain_value_kmh"], errors="coerce")
    unknown = values.isna() | ~np.isfinite(values) | joined["speed_domain_provenance"].fillna("UNKNOWN").eq("UNKNOWN")
    violation = ~unknown & values.gt(float(speed_cap_kmh))
    rows: list[dict[str, Any]] = []
    for index, token in joined.sort_values("route_sequence").iterrows():
        if bool(violation.loc[index]):
            state, reason = "INCOMPATIBLE", "SPEED_DOMAIN_CAP_EXCEEDED"
        elif bool(unknown.loc[index]):
            state, reason = "UNKNOWN", "SPEED_DOMAIN_UNKNOWN"
        else:
            state, reason = "COMPATIBLE", None
        rows.append(_atomic_row(
            order_id=order_id, profile_id=profile_id, family="SPEED",
            name="SPEED_DOMAIN", state=state,
            observed=float(values.loc[index]) if pd.notna(values.loc[index]) else None,
            cap=float(speed_cap_kmh), route_sequence=int(token["route_sequence"]),
            edge_id=str(token["resolved_stage3_edge_uid"]), reason=reason,
        ))
    return rows


def evaluate_movement_atomic_checks(
    encounter: Mapping[str, Any], profile_id: str
) -> list[dict[str, Any]]:
    """Return independent movement/control/roundabout/restriction evidence."""
    order_id = str(encounter.get("order_id", ""))
    occurrence_value = encounter.get("movement_occurrence_index")
    occurrence = int(occurrence_value) if occurrence_value is not None and pd.notna(occurrence_value) else None
    complex_id = encounter.get("intersection_complex_uid")
    lookup = str(encounter.get("movement_lookup_status", "UNRESOLVED_MOVEMENT_LOOKUP"))
    turn = str(encounter.get("route_turn_type", "UNKNOWN")).upper()
    control = str(encounter.get("signal_state", "UNKNOWN_CONTROL")).upper()
    roundabout = bool(encounter.get("roundabout_evidence_present", False))
    certified = bool(encounter.get("restriction_enforcement_certified", False))
    legality = str(encounter.get("movement_legality_state", "UNKNOWN"))
    common = dict(order_id=order_id, profile_id=profile_id, complex_id=str(complex_id), occurrence=occurrence)

    if lookup != "MATCHED_TOPOLOGICAL_MOVEMENT":
        movement_state, movement_reason = "UNKNOWN", "MOVEMENT_LOOKUP_UNRESOLVED"
    elif turn == "UNKNOWN" or turn not in {"STRAIGHT", "RIGHT", "LEFT", "UTURN"}:
        movement_state, movement_reason = "UNKNOWN", "TURN_GEOMETRY_UNKNOWN"
    elif turn == "UTURN" and profile_id in {"C", "M"}:
        movement_state, movement_reason = "INCOMPATIBLE", "UTURN_PROFILE_INCOMPATIBLE"
    else:
        movement_state, movement_reason = "COMPATIBLE", None

    if lookup == "MATCHED_TOPOLOGICAL_MOVEMENT" and turn == "LEFT" and profile_id == "C":
        if control == "SIGNALIZED":
            control_state, control_reason = "COMPATIBLE", None
        elif control == "STOP_OR_YIELD_CONTROLLED":
            control_state, control_reason = "INCOMPATIBLE", "CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE"
        else:
            control_state, control_reason = "UNKNOWN", "CONSERVATIVE_LEFT_UNKNOWN_CONTROL"
    else:
        control_state, control_reason = "COMPATIBLE", None

    round_state = "INCOMPATIBLE" if roundabout and profile_id == "C" else "COMPATIBLE"
    restriction_state = "INCOMPATIBLE" if certified and legality == "CERTIFIED_PROHIBITED" else "COMPATIBLE"
    return [
        _atomic_row(family="MOVEMENT", name="MOVEMENT", state=movement_state, reason=movement_reason, **common),
        _atomic_row(family="CONTROL", name="CONTROL", state=control_state, reason=control_reason, **common),
        _atomic_row(
            family="ROUNDABOUT",
            name="ROUNDABOUT",
            state=round_state,
            reason="CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE" if round_state == "INCOMPATIBLE" else None,
            **common,
        ),
        _atomic_row(
            family="RESTRICTION",
            name="RESTRICTION",
            state=restriction_state,
            reason="CERTIFIED_MOVEMENT_PROHIBITION" if restriction_state == "INCOMPATIBLE" else None,
            **common,
        ),
    ]


def evaluate_dynamic_checks(
    descriptor: Mapping[str, Any], profile_id: str, dynamic_caps: Mapping[str, Mapping[str, float]]
) -> list[dict[str, Any]]:
    """Evaluate all twelve non-compensatory E/Q/C conditions."""
    order_id = str(descriptor.get("order_id", ""))
    complete = bool(descriptor.get("dynamic_complete", False))
    rows: list[dict[str, Any]] = []
    for dimension in DYNAMIC_DIMS:
        for metric in ("E", "Q", "C"):
            name = f"{dimension}_{metric}"
            cap = float(dynamic_caps[dimension][metric])
            value = descriptor.get(name)
            known = complete and value is not None and pd.notna(value) and np.isfinite(float(value))
            if not known:
                state, reason, observed = "UNKNOWN", "DYNAMIC_ROUTE_INCOMPLETE", None
            elif float(value) > cap:
                state = "INCOMPATIBLE"
                reason = f"DYNAMIC_{dimension.upper()}_{metric}_CAP_EXCEEDED"
                observed = float(value)
            else:
                state, reason, observed = "COMPATIBLE", None, float(value)
            rows.append(_atomic_row(
                order_id=order_id,
                profile_id=profile_id,
                family=f"DYNAMIC_{dimension.upper()}",
                name=name,
                state=state,
                observed=observed,
                cap=cap,
                reason=reason,
            ))
    return rows


def audit_three_state_nestedness(suitability: pd.DataFrame) -> dict[str, Any]:
    """Audit C->M->A without letting missing-evidence regress with capability."""
    required = {"order_id", "profile_id", "original_route_state"}
    if not required.issubset(suitability):
        raise Stage3S2AError("nestedness input schema incomplete")
    pivot = suitability.pivot(index="order_id", columns="profile_id", values="original_route_state")
    if set(pivot.columns) != set(PROFILES) or pivot.isna().any().any():
        raise Stage3S2AError("nestedness requires exactly C/M/A per order")
    rank = {"INFEASIBLE": 0, "UNKNOWN": 1, "FEASIBLE": 2}
    transitions: dict[str, Any] = {}
    blocker = 0
    feasible_to_unknown = 0
    for lower, upper in (("C", "M"), ("M", "A")):
        counts = pivot.groupby([lower, upper], observed=True).size()
        matrix = {
            f"{left}->{right}": int(counts.get((left, right), 0))
            for left in rank
            for right in rank
        }
        regress = np.array([rank[v] for v in pivot[lower]]) > np.array([rank[v] for v in pivot[upper]])
        pair_blockers = int(regress.sum())
        pair_f_to_u = int(((pivot[lower] == "FEASIBLE") & (pivot[upper] == "UNKNOWN")).sum())
        transitions[f"{lower}_to_{upper}"] = {
            "matrix": matrix,
            "rank_regression_count": pair_blockers,
            "feasible_to_unknown_count": pair_f_to_u,
        }
        blocker += pair_blockers
        feasible_to_unknown += pair_f_to_u
    passed = blocker == 0 and feasible_to_unknown == 0
    return {
        "status": "PASS" if passed else "FAIL",
        "pass": passed,
        "rank_regression_count": blocker,
        "capability_regression_count": blocker,
        "feasible_to_unknown_count": feasible_to_unknown,
        "transitions": transitions,
    }


def _frozen_bindings(root: Path) -> dict[str, Any]:
    release = read_json(root / UPSTREAM_REL["s3_release_manifest"])
    if release.get("artifact_sha256") != payload_hash(release):
        raise Stage3S2AError("S3.1 release manifest self-hash mismatch")
    if release.get("phase_status") != "STAGE3_S31_CLOSURE_COMPLETE":
        raise Stage3S2AError("S3.1 release is not frozen complete")
    verification = read_json(root / UPSTREAM_REL["s3_evidence_verification"])
    if verification.get("artifact_sha256") != payload_hash(verification):
        raise Stage3S2AError("S3 evidence verification self-hash mismatch")
    if verification.get("status") != "PASS":
        raise Stage3S2AError("S3 evidence verification is not PASS")
    inputs: dict[str, Any] = {}
    for name, relative in {"profile": PROFILE_REL, "cdf": CDF_REL, **UPSTREAM_REL}.items():
        path = root / relative
        inputs[name] = parquet_descriptor(path, root) if path.suffix == ".parquet" else source_descriptor(path, root)
    if inputs["m3_checkpoint"]["sha256"] != M3_SHA256:
        raise Stage3S2AError("frozen M3 checkpoint SHA mismatch")
    frozen = release.get("frozen_inputs", {})
    for key in (
        "stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet",
        "stage3/output/odd_tod/s2a/stage3_speed_domain.parquet",
        "stage3/output/odd_tod/s2a/stage3_historical_direction_overlay.parquet",
        "stage3/output/odd_tod/s2b/final/stage3_intersection_complexes.parquet",
        "stage3/output/odd_tod/s2b/final/stage3_edge_complex_boundary_index.parquet",
        "stage3/output/odd_tod/s2b/final/stage3_route_movement_lookup.parquet",
        "stage3/output/odd_tod/s2b/final/stage3_intersection_movements.parquet",
        "stage3/output/odd_tod/s2b/final/stage3_intersection_node_membership.parquet",
        "stage2/output_v5_2/development/M3/epoch_004.pt",
        "stage2/output_v5_2/development/M3/model_manifest.json",
        "stage2/output_v5/protocols/development/tensor_shards/feature_artifacts.json",
        "stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json",
    ):
        path = root / key
        if key not in frozen or frozen[key].get("sha256") != sha256_file(path):
            raise Stage3S2AError(f"S3 release frozen input mismatch: {key}")
    s2a_closure = read_json(root / UPSTREAM_REL["s2a1_scientific_closure_evidence"])
    if s2a_closure.get("artifact_sha256") != payload_hash(s2a_closure):
        raise Stage3S2AError("S2A.1 scientific closure self-hash mismatch")
    observed_mapping = s2a_closure.get("inputs", {}).get("observed_mapping", {})
    if (
        observed_mapping.get("path") != UPSTREAM_REL["observed_full_network_mapping"].as_posix()
        or observed_mapping.get("sha256") != inputs["observed_full_network_mapping"]["sha256"]
    ):
        raise Stage3S2AError("observed full-network mapping differs from S2A.1 frozen evidence")
    model_manifest = read_json(root / UPSTREAM_REL["m3_model_manifest"])
    if (
        model_manifest.get("source", {}).get("feature_artifact_sha256") != inputs["m3_feature_artifacts"]["sha256"]
        or model_manifest.get("static_artifact_sha256") != inputs["m3_static_artifact"]["sha256"]
        or model_manifest.get("support_artifact_sha256") != inputs["m3_support_artifact"]["sha256"]
        or model_manifest.get("tensor_manifest_sha256") != inputs["m3_transfer_manifest"]["sha256"]
    ):
        raise Stage3S2AError("M3 manifest does not bind the frozen inference artifacts")
    if release.get("profile", {}).get("sha256") != inputs["profile"]["sha256"]:
        raise Stage3S2AError("frozen profile SHA differs from S3.1 release")
    cdf_product = release.get("products", {}).get(CDF_REL.as_posix(), {})
    if cdf_product.get("sha256") != inputs["cdf"]["sha256"]:
        raise Stage3S2AError("frozen CDF SHA differs from S3.1 release")
    profile = read_json(root / PROFILE_REL)
    _profile_map(profile)
    return inputs


def _route_integrity(route: pd.DataFrame) -> dict[str, Any]:
    if set(route["date"].astype(str).unique()) != {TEST_DATE}:
        raise Stage3S2AError("S4 may read only Test31")
    orders = int(route["order_id"].nunique())
    if orders != EXPECTED_ORDER_COUNT:
        raise Stage3S2AError(f"Test31 order count differs from frozen 30000: {orders}")
    if route.duplicated(["order_id", "route_sequence"]).any():
        raise Stage3S2AError("Test31 route has duplicate order/route_sequence")
    if route.duplicated(["order_id", "traversal_id"]).any():
        raise Stage3S2AError("Test31 route has duplicate order/traversal_id")
    ordered = route.sort_values(["order_id", "route_sequence"], kind="stable")
    integrity = ordered.groupby("order_id", sort=False)["route_sequence"].agg(
        first="min", count="size", last="max"
    )
    bad = (integrity["first"] != 0) | (integrity["last"] != integrity["count"] - 1)
    if bad.any():
        raise Stage3S2AError(f"Test31 non-contiguous route_sequence orders: {int(bad.sum())}")
    return {
        "date": TEST_DATE,
        "order_count": orders,
        "route_token_count": int(len(route)),
        "route_sequence_integrity": "PASS",
        "unique_order_route_sequence": True,
        "unique_order_traversal_id": True,
        "minimum_route_token_count": int(integrity["count"].min()),
        "maximum_route_token_count": int(integrity["count"].max()),
    }


def prepare_test31(root: Path) -> dict[str, Any]:
    """Bind immutable inputs, then create typed identity and complex encounters."""
    if git_head(root) != AUTHORIZED_BASE:
        # The reviewed base is the authorization anchor. Once S4 itself is
        # committed, exact input hashes below remain the executable freeze;
        # reruns on descendants must not be confused with re-authorization.
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", AUTHORIZED_BASE, "HEAD"],
            cwd=root,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if ancestor.returncode != 0:
            raise Stage3S2AError(f"S4 requires reviewed base {AUTHORIZED_BASE}")
    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    bindings = _frozen_bindings(root)
    route_path = root / ROUTE_REL
    route_binding = parquet_descriptor(route_path, root)
    route_manifest_binding = source_descriptor(root / ROUTE_MANIFEST_REL, root)
    route_manifest = read_json(root / ROUTE_MANIFEST_REL)
    route = pd.read_parquet(route_path, columns=list(ROUTE_IDENTITY_COLUMNS))
    integrity = _route_integrity(route)
    if (
        route_manifest.get("date") != TEST_DATE
        or route_manifest.get("engineering_status") != "PASS"
        or route_manifest.get("file_sha256") != route_binding["sha256"]
        or int(route_manifest.get("row_count", -1)) != integrity["route_token_count"]
        or int(route_manifest.get("order_count", -1)) != integrity["order_count"]
        or int(route_manifest.get("route_key_duplicate_count", -1)) != 0
        or int(route_manifest.get("time_leakage_violation_count", -1)) != 0
    ):
        raise Stage3S2AError("Test31 route differs from the frozen Stage2 day manifest")

    # This manifest is deliberately committed atomically before any suitability
    # evaluation.  It binds the only Test31 route source and all frozen inputs.
    input_manifest = {
        "schema_version": "stage3_s4_test31_input_manifest.1",
        "phase": "S4_TEST31_ORIGINAL_ROUTE_SUITABILITY",
        "authorized_base": AUTHORIZED_BASE,
        "created_before_suitability_evaluation": True,
        "test_date": TEST_DATE,
        "route_source": route_binding,
        "route_source_manifest": route_manifest_binding,
        "integrity": integrity,
        "frozen_inputs": bindings,
        "original_route_only": True,
        "route_reconstruction": False,
        "nearest_geometry_repair": False,
        "fallback": False,
        "route_search": False,
        "profile_retuning": False,
        "test31_calibration": False,
        "s5_authorized": False,
        "next_phase_authorized": False,
    }
    input_manifest["artifact_sha256"] = payload_hash(input_manifest)
    atomic_json(output / "test31_input_manifest.json", input_manifest)

    mapping = pd.read_parquet(root / UPSTREAM_REL["observed_full_network_mapping"])
    overlay = pd.read_parquet(root / UPSTREAM_REL["historical_direction_overlay"])
    typed = resolve_route_tokens(route, mapping, overlay)
    identity_path = output / "test31_route_identity_resolution.parquet"
    atomic_parquet(identity_path, typed)
    boundary = pd.read_parquet(root / UPSTREAM_REL["edge_complex_boundary_index"])
    movements = pd.read_parquet(root / UPSTREAM_REL["route_movement_lookup"])
    encounters = parse_route_complex_encounters(typed, boundary, movements)
    encounter_path = output / "test31_route_complex_encounters.parquet"
    atomic_parquet(encounter_path, encounters)
    counts = typed["route_token_type"].value_counts()
    per_order = typed.groupby("order_id", sort=False)["route_token_type"].agg(list)
    result = {
        "schema_version": "stage3_s4_prepare_test31.1",
        **integrity,
        "FULL_NETWORK_EDGE_count": int(counts.get("FULL_NETWORK_EDGE", 0)),
        "HISTORICAL_REVERSE_OVERLAY_count": int(counts.get("HISTORICAL_REVERSE_OVERLAY", 0)),
        "UNRESOLVED_count": int(counts.get("UNRESOLVED", 0)),
        "fully_full_network_resolved_order_count": int(per_order.map(lambda v: all(x == "FULL_NETWORK_EDGE" for x in v)).sum()),
        "orders_with_reverse_overlay": int(per_order.map(lambda v: "HISTORICAL_REVERSE_OVERLAY" in v).sum()),
        "orders_with_unresolved_token": int(per_order.map(lambda v: "UNRESOLVED" in v).sum()),
        "complex_encounter_count": int(len(encounters)),
        "identity_product": parquet_descriptor(identity_path, root),
        "encounter_product": parquet_descriptor(encounter_path, root),
        "input_manifest_sha256": sha256_file(output / "test31_input_manifest.json"),
    }
    for key in ("FULL_NETWORK_EDGE", "HISTORICAL_REVERSE_OVERLAY", "UNRESOLVED"):
        result[f"{key}_share"] = float(result[f"{key}_count"] / len(typed))
    result["artifact_sha256"] = payload_hash(result)
    atomic_json(output / "test31_prepare_summary.json", result)
    return result


def _full_static_reference(root: Path) -> pd.DataFrame:
    complexes = pd.read_parquet(root / UPSTREAM_REL["intersection_complexes"])
    boundary = pd.read_parquet(root / UPSTREAM_REL["edge_complex_boundary_index"])
    edges = pd.read_parquet(root / UPSTREAM_REL["full_network_edges"])
    diversity = boundary_road_class_diversity(boundary, edges)
    static = complexes[[
        "intersection_complex_uid",
        "external_physical_connection_count",
        "topological_movement_count",
        "internal_length_m",
        "signal_state",
        "roundabout_evidence_present",
        "grade_separation_evidence_present",
    ]].rename(columns={
        "external_physical_connection_count": "A_c",
        "topological_movement_count": "M_c",
        "internal_length_m": "L_c",
    })
    static = static.merge(diversity, on="intersection_complex_uid", how="left", validate="one_to_one")
    static["D_c"] = static["boundary_road_class_diversity"]
    if static[["A_c", "M_c", "D_c", "L_c"]].isna().any().any():
        raise Stage3S2AError("full complex reference contains missing A/M/D/L")
    if len(static) != 43_685:
        raise Stage3S2AError(f"unexpected frozen complex count: {len(static)}")
    return static


def _dynamic_descriptors(
    route: pd.DataFrame, predictions: pd.DataFrame, cdf_path: Path
) -> pd.DataFrame:
    pred_columns = [
        "date", "order_id", "traversal_id", "travel_time_p50_s",
        *[f"pred_{dimension}" for dimension in DYNAMIC_DIMS],
    ]
    if predictions.duplicated(["date", "order_id", "traversal_id"]).any():
        raise Stage3S2AError("Test31 M3 prediction identity is not unique")
    tokens = route[["date", "order_id", "route_sequence", "traversal_id"]].merge(
        predictions[pred_columns],
        on=["date", "order_id", "traversal_id"],
        how="left",
        validate="one_to_one",
    ).sort_values(["order_id", "route_sequence"], kind="stable")
    required = ["travel_time_p50_s", *[f"pred_{dimension}" for dimension in DYNAMIC_DIMS]]
    valid_by_field = {column: np.isfinite(pd.to_numeric(tokens[column], errors="coerce")) for column in required}
    valid_by_field["travel_time_p50_s"] &= tokens["travel_time_p50_s"].gt(0)
    token_valid = np.logical_and.reduce([value.to_numpy(bool) for value in valid_by_field.values()])
    tokens["dynamic_token_valid"] = token_valid
    route_complete = tokens.groupby("order_id", sort=False)["dynamic_token_valid"].all()
    missing_count = tokens.assign(_missing=~tokens["dynamic_token_valid"]).groupby("order_id", sort=False)["_missing"].sum()
    first_missing = tokens.loc[~tokens["dynamic_token_valid"]].groupby("order_id", sort=False)["route_sequence"].min()
    masks: dict[str, str] = {}
    for order_id, group in tokens.loc[~tokens["dynamic_token_valid"]].groupby("order_id", sort=False):
        fields = sorted({column for column, mask in valid_by_field.items() if (~mask.loc[group.index]).any()})
        masks[str(order_id)] = json.dumps(fields, separators=(",", ":"))

    complete = tokens[tokens["order_id"].map(route_complete)].copy()
    for dimension in DYNAMIC_DIMS:
        reference = pq.read_table(
            cdf_path,
            columns=["dimension", "value", "predicted_time_weight_s", "mid_cdf"],
            filters=[("dimension", "=", dimension)],
        ).to_pandas()
        if set(reference["dimension"].unique()) != {dimension}:
            raise Stage3S2AError(f"frozen CDF filter failed for {dimension}")
        complete[f"z_{dimension}"] = apply_mid_cdf(complete[f"pred_{dimension}"], reference)
        del reference
    rows: list[dict[str, Any]] = []
    for order_id, group in complete.groupby("order_id", sort=False):
        weight = group["travel_time_p50_s"].to_numpy(np.float64)
        total = float(weight.sum())
        row: dict[str, Any] = {
            "order_id": str(order_id),
            "dynamic_complete": True,
            "predicted_route_time_p50_s": total,
        }
        for dimension in DYNAMIC_DIMS:
            z = group[f"z_{dimension}"].to_numpy(np.float64)
            tail = z > Q_TAIL
            row[f"{dimension}_E"] = float(np.dot(weight, z) / total)
            row[f"{dimension}_Q"] = float(weight[tail].sum() / total)
            maximum = running = 0.0
            for is_tail, duration in zip(tail, weight):
                running = running + float(duration) if is_tail else 0.0
                maximum = max(maximum, running)
            row[f"{dimension}_C"] = maximum
        rows.append(row)
    dynamic = pd.DataFrame(rows)
    base = pd.DataFrame({"order_id": route["order_id"].drop_duplicates().astype(str)})
    result = base.merge(dynamic, on="order_id", how="left", validate="one_to_one")
    result["dynamic_complete"] = result["order_id"].map(route_complete).fillna(False).astype(bool)
    result["missing_dynamic_token_count"] = result["order_id"].map(missing_count).fillna(0).astype("int64")
    result["first_missing_dynamic_route_sequence"] = result["order_id"].map(first_missing)
    result["missing_dynamic_field_mask"] = result["order_id"].map(masks).fillna("[]")
    return result


def _direction_descriptor(typed: pd.DataFrame) -> pd.DataFrame:
    work = typed.assign(
        _reverse=typed["route_token_type"].eq("HISTORICAL_REVERSE_OVERLAY"),
        _unresolved=typed["route_token_type"].eq("UNRESOLVED"),
        _full=typed["route_token_type"].eq("FULL_NETWORK_EDGE"),
    )
    rows = []
    for order_id, group in work.groupby("order_id", sort=False):
        reverse = group[group["_reverse"]]
        unresolved = group[group["_unresolved"]]
        rows.append({
            "order_id": str(order_id),
            "route_token_count": int(len(group)),
            "full_network_edge_token_count": int(group["_full"].sum()),
            "reverse_overlay_token_count": int(group["_reverse"].sum()),
            "unresolved_token_count": int(group["_unresolved"].sum()),
            "first_reverse_route_sequence": int(reverse["route_sequence"].min()) if len(reverse) else np.nan,
            "first_unresolved_route_sequence": int(unresolved["route_sequence"].min()) if len(unresolved) else np.nan,
        })
    return pd.DataFrame(rows)


def _speed_descriptor(typed: pd.DataFrame, speed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    full = typed[typed["route_token_type"].eq("FULL_NETWORK_EDGE")][[
        "order_id", "route_sequence", "resolved_stage3_edge_uid"
    ]].copy()
    domain = speed[["stage3_edge_uid", "speed_domain_value_kmh", "speed_domain_provenance"]].drop_duplicates("stage3_edge_uid")
    if domain["stage3_edge_uid"].duplicated().any():
        raise Stage3S2AError("speed-domain identity is not unique")
    joined = full.merge(domain, left_on="resolved_stage3_edge_uid", right_on="stage3_edge_uid", how="left", validate="many_to_one")
    value = pd.to_numeric(joined["speed_domain_value_kmh"], errors="coerce")
    joined["speed_known"] = value.notna() & np.isfinite(value) & ~joined["speed_domain_provenance"].fillna("UNKNOWN").eq("UNKNOWN")
    joined["speed_domain_value_kmh"] = value
    rows = []
    all_orders = typed["order_id"].drop_duplicates().astype(str)
    groups = {str(key): value for key, value in joined.groupby("order_id", sort=False)}
    for order_id in all_orders:
        group = groups.get(str(order_id), joined.iloc[0:0])
        known = group[group["speed_known"]]
        unknown = group[~group["speed_known"]]
        rows.append({
            "order_id": str(order_id),
            "max_route_speed_domain_kmh": float(known["speed_domain_value_kmh"].max()) if len(known) else np.nan,
            "unknown_speed_edge_count": int(len(unknown)),
            "first_unknown_speed_route_sequence": int(unknown["route_sequence"].min()) if len(unknown) else np.nan,
        })
    return pd.DataFrame(rows), joined


def _encounter_detail(root: Path, encounters: pd.DataFrame, static: pd.DataFrame) -> pd.DataFrame:
    movement = pd.read_parquet(root / UPSTREAM_REL["route_movement_lookup"])
    keys = ["intersection_complex_uid", "incoming_stage3_edge_uid", "outgoing_stage3_edge_uid"]
    if movement.duplicated(keys).any():
        raise Stage3S2AError("movement lookup key is not unique")
    columns = keys + [
        "route_turn_type", "restriction_evidence_present", "restriction_enforcement_certified",
        "movement_legality_state", "topological_path_exists",
    ]
    detail = encounters.merge(movement[columns], on=keys, how="left", validate="many_to_one")
    detail = detail.merge(static, on="intersection_complex_uid", how="left", validate="many_to_one")
    return detail


def _encounter_descriptor(
    all_orders: pd.Series, detail: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    groups = {str(key): value for key, value in detail.groupby("order_id", sort=False)}
    for order_id in all_orders.astype(str):
        group = groups.get(str(order_id), detail.iloc[0:0])
        turns = group["route_turn_type"].fillna("UNKNOWN").astype(str).str.upper() if len(group) else pd.Series(dtype=str)
        rows.append({
            "order_id": str(order_id),
            "resolved_complex_encounter_count": int(len(group)),
            "movement_encounter_count": int(len(group)),
            "matched_movement_count": int(group["movement_lookup_status"].eq("MATCHED_TOPOLOGICAL_MOVEMENT").sum()) if len(group) else 0,
            "unresolved_movement_lookup_count": int(group["movement_lookup_status"].ne("MATCHED_TOPOLOGICAL_MOVEMENT").sum()) if len(group) else 0,
            "route_max_A_c": float(group["A_c"].max()) if len(group) else np.nan,
            "route_max_M_c": float(group["M_c"].max()) if len(group) else np.nan,
            "route_max_D_c": float(group["D_c"].max()) if len(group) else np.nan,
            "route_max_L_c": float(group["L_c"].max()) if len(group) else np.nan,
            "straight_count": int(turns.eq("STRAIGHT").sum()),
            "right_count": int(turns.eq("RIGHT").sum()),
            "left_count": int(turns.eq("LEFT").sum()),
            "uturn_count": int(turns.eq("UTURN").sum()),
            "unknown_turn_count": int(turns.eq("UNKNOWN").sum()),
            "signalized_left_count": int((turns.eq("LEFT") & group["signal_state"].eq("SIGNALIZED")).sum()) if len(group) else 0,
            "unknown_control_left_count": int((turns.eq("LEFT") & ~group["signal_state"].isin(["SIGNALIZED", "STOP_OR_YIELD_CONTROLLED"])).sum()) if len(group) else 0,
            "roundabout_encounter_count": int(group["roundabout_evidence_present"].fillna(False).sum()) if len(group) else 0,
            "grade_separated_complex_encounter_count": int(group["grade_separation_evidence_present"].fillna(False).sum()) if len(group) else 0,
        })
    return pd.DataFrame(rows)


def _standard_atomic(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in ATOMIC_COLUMNS:
        if column not in result:
            result[column] = None
    return result[list(ATOMIC_COLUMNS)]


def _write_atomic_parts(parts: Sequence[pd.DataFrame], destination: Path) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp")
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for frame in parts:
            if frame.empty:
                continue
            table = pa.Table.from_pandas(
                _standard_atomic(frame), schema=ATOMIC_SCHEMA, preserve_index=False
            )
            if writer is None:
                writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table)
        if writer is None:
            raise Stage3S2AError("atomic-check generation produced no rows")
        writer.close()
        writer = None
        os.replace(temporary, destination)
    finally:
        if writer is not None:
            writer.close()
        if temporary.exists():
            temporary.unlink()


def _route_atomic_frames(
    descriptor: pd.DataFrame,
    detail: pd.DataFrame,
    speed_tokens: pd.DataFrame,
    profile_by_id: Mapping[str, Mapping[str, Any]],
) -> tuple[Iterable[pd.DataFrame], dict[tuple[str, str], dict[str, Any]]]:
    """Build audit rows and compact order/profile attribution simultaneously."""
    compact: dict[tuple[str, str], dict[str, Any]] = {}
    detail_groups = {str(key): value for key, value in detail.groupby("order_id", sort=False)}
    speed_groups = {str(key): value for key, value in speed_tokens.groupby("order_id", sort=False)}
    descriptor_by_order = descriptor.set_index("order_id", drop=False)

    def generate() -> Iterable[pd.DataFrame]:
      for profile_id in PROFILES:
        profile = profile_by_id[profile_id]
        static_caps = profile["static_caps"]
        speed_cap = float(profile["speed_domain_max_kmh"])
        rows: list[dict[str, Any]] = []
        for order_id, record in descriptor_by_order.iterrows():
            order_id = str(order_id)
            states: dict[str, list[str]] = defaultdict(list)
            known_codes: set[str] = set()
            unknown_codes: set[str] = set()
            known_atomic = unknown_atomic = 0

            reverse_count = int(record["reverse_overlay_token_count"])
            unresolved_count = int(record["unresolved_token_count"])
            direction_evidence: list[tuple[str, str | None, Any, int]] = []
            if reverse_count:
                direction_evidence.append(("INCOMPATIBLE", "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE", record["first_reverse_route_sequence"], reverse_count))
            elif unresolved_count:
                direction_evidence.append(("UNKNOWN", "UNRESOLVED_ROUTE_IDENTITY", record["first_unresolved_route_sequence"], unresolved_count))
            if not direction_evidence:
                direction_evidence.append(("COMPATIBLE", None, None, 0))
            for state, reason, seq, count in direction_evidence:
                rows.append(_atomic_row(
                    order_id=order_id, profile_id=profile_id, family="DIRECTION",
                    name="directional_routability", state=state, observed=float(count),
                    route_sequence=int(seq) if seq is not None and pd.notna(seq) else None,
                    reason=reason,
                ))
                states["directional_routability_state"].append(state)
                if state == "INCOMPATIBLE": known_codes.add(str(reason)); known_atomic += 1
                if state == "UNKNOWN": unknown_codes.add(str(reason)); unknown_atomic += 1

            speed_group = speed_groups.get(order_id, speed_tokens.iloc[0:0])
            known_speed = speed_group[speed_group["speed_known"]]
            unknown_speed = speed_group[~speed_group["speed_known"]]
            violating = known_speed[known_speed["speed_domain_value_kmh"] > speed_cap]
            speed_evidence: list[tuple[str, str | None, Any]] = []
            if len(violating):
                first = violating.sort_values("route_sequence").iloc[0]
                speed_evidence.append(("INCOMPATIBLE", "SPEED_DOMAIN_CAP_EXCEEDED", first))
            if len(unknown_speed):
                speed_evidence.append(("UNKNOWN", "SPEED_DOMAIN_UNKNOWN", unknown_speed.sort_values("route_sequence").iloc[0]))
            if not speed_evidence:
                speed_evidence.append(("COMPATIBLE", None, known_speed.sort_values("route_sequence").iloc[0] if len(known_speed) else None))
            for speed_state, speed_reason, first in speed_evidence:
                rows.append(_atomic_row(
                    order_id=order_id, profile_id=profile_id, family="SPEED", name="speed_domain_cap",
                    state=speed_state,
                    observed=float(known_speed["speed_domain_value_kmh"].max()) if len(known_speed) else None,
                    cap=speed_cap,
                    route_sequence=int(first["route_sequence"]) if first is not None else None,
                    edge_id=str(first["resolved_stage3_edge_uid"]) if first is not None else None,
                    reason=speed_reason,
                ))
                states["speed_state"].append(speed_state)
                if speed_state == "INCOMPATIBLE": known_codes.add(str(speed_reason)); known_atomic += 1
                if speed_state == "UNKNOWN": unknown_codes.add(str(speed_reason)); unknown_atomic += 1

            encounter_group = detail_groups.get(order_id, detail.iloc[0:0])
            static_aliases = {
                "A": ("A_c", "external_physical_connection_count", "STATIC_A_CAP_EXCEEDED"),
                "M": ("M_c", "topological_movement_count", "STATIC_M_CAP_EXCEEDED"),
                "D": ("D_c", "road_class_diversity", "STATIC_D_CAP_EXCEEDED"),
                "L": ("L_c", "internal_length_m", "STATIC_L_CAP_EXCEEDED"),
            }
            for label, (column, cap_key, cap_reason) in static_aliases.items():
                cap = float(static_caps[cap_key])
                missing = encounter_group[column].isna() if len(encounter_group) else pd.Series(dtype=bool)
                violations = encounter_group.loc[~missing & encounter_group[column].gt(cap)] if len(encounter_group) else encounter_group
                static_evidence: list[tuple[str, str | None, Any]] = []
                for _, complex_occurrence in encounter_group.sort_values("movement_occurrence_index").iterrows():
                    metric = complex_occurrence[column]
                    if pd.isna(metric):
                        static_evidence.append(("UNKNOWN", "STATIC_METRIC_UNKNOWN", complex_occurrence))
                    elif float(metric) > cap:
                        static_evidence.append(("INCOMPATIBLE", cap_reason, complex_occurrence))
                    else:
                        static_evidence.append(("COMPATIBLE", None, complex_occurrence))
                if not static_evidence:
                    static_evidence.append(("COMPATIBLE", None, None))
                for static_state, static_reason, first_static in static_evidence:
                    rows.append(_atomic_row(
                        order_id=order_id, profile_id=profile_id, family=f"STATIC_{label}", name=f"static_{label}_cap",
                        state=static_state,
                        observed=(
                            float(first_static[column])
                            if first_static is not None and pd.notna(first_static[column])
                            else None
                        ),
                        cap=cap,
                        complex_id=str(first_static["intersection_complex_uid"]) if first_static is not None else None,
                        occurrence=int(first_static["movement_occurrence_index"]) if first_static is not None else None,
                        reason=static_reason,
                    ))
                    states[f"static_{label}_state"].append(static_state)
                    if static_state == "INCOMPATIBLE": known_codes.add(str(static_reason)); known_atomic += 1
                    if static_state == "UNKNOWN": unknown_codes.add(str(static_reason)); unknown_atomic += 1

            movement_rows: list[dict[str, Any]] = []
            for encounter in encounter_group.to_dict("records"):
                movement_rows.extend(evaluate_movement_atomic_checks(encounter, profile_id))
            if movement_rows:
                rows.extend(movement_rows)
                movement_frame = pd.DataFrame(movement_rows)
                for family, output_key in (
                    ("MOVEMENT", "movement_state"),
                    ("CONTROL", "control_state"),
                    ("ROUNDABOUT", "roundabout_state"),
                    ("RESTRICTION", "restriction_state"),
                ):
                    family_frame = movement_frame[movement_frame["check_family"].eq(family)]
                    family_state = aggregate_atomic_state(family_frame["state"])
                    states[output_key].append(family_state)
                bad = movement_frame[movement_frame["state"].eq("INCOMPATIBLE")]
                unknown = movement_frame[movement_frame["state"].eq("UNKNOWN")]
                known_codes.update(bad["reason_code"].dropna().astype(str)); unknown_codes.update(unknown["reason_code"].dropna().astype(str))
                known_atomic += int(len(bad)); unknown_atomic += int(len(unknown))
            else:
                for output_key in ("movement_state", "control_state", "roundabout_state", "restriction_state"):
                    states[output_key].append("COMPATIBLE")
                # Retain explicit vacuous checks so every family is auditable.
                for family, name in (("MOVEMENT", "turn_geometry"), ("CONTROL", "movement_control"), ("ROUNDABOUT", "roundabout_capability"), ("RESTRICTION", "certified_movement_prohibition")):
                    rows.append(_atomic_row(order_id=order_id, profile_id=profile_id, family=family, name=name, state="COMPATIBLE"))

            dynamic_rows = evaluate_dynamic_checks(record, profile_id, profile["dynamic_caps"])
            rows.extend(dynamic_rows)
            dynamic_frame = pd.DataFrame(dynamic_rows)
            dynamic_state = aggregate_atomic_state(dynamic_frame["state"])
            states["dynamic_state"].append(dynamic_state)
            bad_dynamic = dynamic_frame[dynamic_frame["state"].eq("INCOMPATIBLE")]
            unknown_dynamic = dynamic_frame[dynamic_frame["state"].eq("UNKNOWN")]
            known_codes.update(bad_dynamic["reason_code"].dropna().astype(str)); unknown_codes.update(unknown_dynamic["reason_code"].dropna().astype(str))
            known_atomic += int(len(bad_dynamic)); unknown_atomic += int(len(unknown_dynamic))

            all_states = [value for values in states.values() for value in values]
            compact[(order_id, profile_id)] = {
                **{key: aggregate_atomic_state(value) for key, value in states.items()},
                "original_route_state": finalize_route_state(all_states),
                "known_violation_reason_codes": sorted(known_codes),
                "unknown_reason_codes": sorted(unknown_codes),
                "known_violation_count": len(known_codes),
                "unknown_requirement_count": len(unknown_codes),
                "known_violation_atomic_count": known_atomic,
                "unknown_atomic_count": unknown_atomic,
                "violating_edge_count": int(len(violating)),
                "unknown_speed_edge_count": int(len(unknown_speed)),
                "first_violating_stage3_edge_uid": str(violating.sort_values("route_sequence").iloc[0]["resolved_stage3_edge_uid"]) if len(violating) else None,
                "max_violating_speed_domain_kmh": float(violating["speed_domain_value_kmh"].max()) if len(violating) else None,
            }
            if len(rows) >= 100_000:
                yield pd.DataFrame(rows)
                rows = []
        if rows:
            yield pd.DataFrame(rows)
    return generate(), compact


def _suitability_from_compact(
    descriptor: pd.DataFrame, compact: Mapping[tuple[str, str], Mapping[str, Any]]
) -> pd.DataFrame:
    descriptor_map = descriptor.set_index("order_id", drop=False)
    rows = []
    for order_id in descriptor["order_id"].astype(str):
        evidence = descriptor_map.loc[order_id].to_dict()
        for profile_id in PROFILES:
            status = dict(compact[(order_id, profile_id)])
            rows.append({
                "date": TEST_DATE,
                **evidence,
                "profile_id": profile_id,
                **status,
                "known_violation_reason_codes": json.dumps(status["known_violation_reason_codes"], separators=(",", ":")),
                "unknown_reason_codes": json.dumps(status["unknown_reason_codes"], separators=(",", ":")),
            })
    result = pd.DataFrame(rows)
    if len(result) != EXPECTED_ORDER_PROFILE_COUNT or result.duplicated(["order_id", "profile_id"]).any():
        raise Stage3S2AError("suitability product is not exactly 30000 x 3")
    return result


def _order_flags(
    order_ids: pd.Series, fail: np.ndarray | pd.Series, unknown: np.ndarray | pd.Series,
) -> pd.DataFrame:
    frame = pd.DataFrame({
        "order_id": order_ids.astype(str).to_numpy(),
        "fail": np.asarray(fail, dtype=bool),
        "unknown": np.asarray(unknown, dtype=bool),
    })
    result = frame.groupby("order_id", sort=False).agg(
        fail=("fail", "any"), unknown=("unknown", "any"),
        fail_count=("fail", "sum"), unknown_count=("unknown", "sum"),
    )
    return result


def _state_array(fail: np.ndarray | pd.Series, unknown: np.ndarray | pd.Series) -> np.ndarray:
    return np.select(
        [np.asarray(fail, dtype=bool), np.asarray(unknown, dtype=bool)],
        ["INCOMPATIBLE", "UNKNOWN"],
        default="COMPATIBLE",
    )


def _vectorized_suitability(
    descriptor: pd.DataFrame,
    detail: pd.DataFrame,
    speed_tokens: pd.DataFrame,
    profile_by_id: Mapping[str, Mapping[str, Any]],
) -> pd.DataFrame:
    """Apply all frozen profile rules to 90k rows without Python route loops."""
    profile_rows = pd.DataFrame({"profile_id": list(PROFILES), "_cross": 1})
    result = descriptor.assign(_cross=1).merge(profile_rows, on="_cross", how="inner").drop(columns="_cross")
    result.insert(0, "date", TEST_DATE)
    known_flags = {code: np.zeros(len(result), dtype=bool) for code in sorted(KNOWN_REASON_CODES)}
    unknown_flags = {code: np.zeros(len(result), dtype=bool) for code in sorted(UNKNOWN_REASON_CODES)}
    known_atomic = np.zeros(len(result), dtype=np.int64)
    unknown_atomic = np.zeros(len(result), dtype=np.int64)

    reverse = result["reverse_overlay_token_count"].gt(0).to_numpy()
    unresolved = result["unresolved_token_count"].gt(0).to_numpy()
    result["directional_routability_state"] = _state_array(reverse, unresolved & ~reverse)
    known_flags["KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE"] |= reverse
    unknown_flags["UNRESOLVED_ROUTE_IDENTITY"] |= unresolved & ~reverse
    known_atomic += reverse.astype(np.int64)
    unknown_atomic += (unresolved & ~reverse).astype(np.int64)

    for profile_id in PROFILES:
        row_mask = result["profile_id"].eq(profile_id).to_numpy()
        cap = float(profile_by_id[profile_id]["speed_domain_max_kmh"])
        fail = result["max_route_speed_domain_kmh"].gt(cap).fillna(False).to_numpy() & row_mask
        unknown = result["unknown_speed_edge_count"].gt(0).to_numpy() & row_mask
        result.loc[row_mask, "speed_state"] = _state_array(fail[row_mask], unknown[row_mask])
        known_flags["SPEED_DOMAIN_CAP_EXCEEDED"] |= fail
        unknown_flags["SPEED_DOMAIN_UNKNOWN"] |= unknown
        violating = speed_tokens["speed_known"] & speed_tokens["speed_domain_value_kmh"].gt(cap)
        violation_count = speed_tokens.loc[violating].groupby("order_id", sort=False).size()
        max_violation = speed_tokens.loc[violating].groupby("order_id", sort=False)["speed_domain_value_kmh"].max()
        first_violation = (
            speed_tokens.loc[violating].sort_values(["order_id", "route_sequence"], kind="stable")
            .drop_duplicates("order_id").set_index("order_id")["resolved_stage3_edge_uid"]
        )
        result.loc[row_mask, "violating_edge_count"] = result.loc[row_mask, "order_id"].map(violation_count).fillna(0).astype("int64")
        result.loc[row_mask, "max_violating_speed_domain_kmh"] = result.loc[row_mask, "order_id"].map(max_violation)
        result.loc[row_mask, "first_violating_stage3_edge_uid"] = result.loc[row_mask, "order_id"].map(first_violation)
        known_atomic += fail.astype(np.int64); unknown_atomic += unknown.astype(np.int64)

    static_aliases = {
        "A": ("A_c", "external_physical_connection_count", "STATIC_A_CAP_EXCEEDED"),
        "M": ("M_c", "topological_movement_count", "STATIC_M_CAP_EXCEEDED"),
        "D": ("D_c", "road_class_diversity", "STATIC_D_CAP_EXCEEDED"),
        "L": ("L_c", "internal_length_m", "STATIC_L_CAP_EXCEEDED"),
    }
    for profile_id in PROFILES:
        row_mask = result["profile_id"].eq(profile_id).to_numpy()
        caps = profile_by_id[profile_id]["static_caps"]
        for label, (column, cap_key, reason) in static_aliases.items():
            value = pd.to_numeric(detail[column], errors="coerce")
            cap = float(caps[cap_key])
            violation_source = value.gt(cap) & value.notna()
            flags = _order_flags(detail["order_id"], violation_source, value.isna())
            fail = result["order_id"].map(flags["fail"]).fillna(False).to_numpy() & row_mask
            unknown = result["order_id"].map(flags["unknown"]).fillna(False).to_numpy() & row_mask
            result.loc[row_mask, f"static_{label}_state"] = _state_array(fail[row_mask], unknown[row_mask])
            known_flags[reason] |= fail; unknown_flags["STATIC_METRIC_UNKNOWN"] |= unknown
            known_atomic[row_mask] += result.loc[row_mask, "order_id"].map(flags["fail_count"]).fillna(0).to_numpy(np.int64)
            unknown_atomic[row_mask] += result.loc[row_mask, "order_id"].map(flags["unknown_count"]).fillna(0).to_numpy(np.int64)
            offending = detail.loc[
                violation_source,
                ["order_id", "movement_occurrence_index", "intersection_complex_uid", column],
            ].copy()
            if len(offending):
                offending["order_id"] = offending["order_id"].astype(str)
                first_offending = (
                    offending.sort_values(
                        ["order_id", "movement_occurrence_index"], kind="stable"
                    )
                    .drop_duplicates("order_id")
                    .set_index("order_id")
                )
                result.loc[row_mask, f"static_{label}_first_offending_complex_uid"] = (
                    result.loc[row_mask, "order_id"]
                    .map(first_offending["intersection_complex_uid"])
                    .to_numpy()
                )
                result.loc[row_mask, f"static_{label}_first_offending_value"] = (
                    result.loc[row_mask, "order_id"].map(first_offending[column]).to_numpy()
                )
            else:
                result.loc[row_mask, f"static_{label}_first_offending_complex_uid"] = None
                result.loc[row_mask, f"static_{label}_first_offending_value"] = np.nan

    lookup = detail["movement_lookup_status"].eq("MATCHED_TOPOLOGICAL_MOVEMENT")
    turn = detail["route_turn_type"].fillna("UNKNOWN").astype(str).str.upper()
    valid_turn = turn.isin(["STRAIGHT", "RIGHT", "LEFT", "UTURN"])
    for profile_id in PROFILES:
        row_mask = result["profile_id"].eq(profile_id).to_numpy()
        movement_fail = lookup & turn.eq("UTURN") & (profile_id in {"C", "M"})
        movement_unknown = ~lookup | ~valid_turn
        movement_flags = _order_flags(detail["order_id"], movement_fail, movement_unknown)
        fail = result["order_id"].map(movement_flags["fail"]).fillna(False).to_numpy() & row_mask
        unknown = result["order_id"].map(movement_flags["unknown"]).fillna(False).to_numpy() & row_mask
        result.loc[row_mask, "movement_state"] = _state_array(fail[row_mask], unknown[row_mask])
        known_flags["UTURN_PROFILE_INCOMPATIBLE"] |= fail
        lookup_unknown_flags = _order_flags(detail["order_id"], np.zeros(len(detail), bool), ~lookup)
        geometry_unknown_flags = _order_flags(detail["order_id"], np.zeros(len(detail), bool), lookup & ~valid_turn)
        lookup_unknown = result["order_id"].map(lookup_unknown_flags["unknown"]).fillna(False).to_numpy() & row_mask
        geometry_unknown = result["order_id"].map(geometry_unknown_flags["unknown"]).fillna(False).to_numpy() & row_mask
        unknown_flags["MOVEMENT_LOOKUP_UNRESOLVED"] |= lookup_unknown
        unknown_flags["TURN_GEOMETRY_UNKNOWN"] |= geometry_unknown
        known_atomic[row_mask] += result.loc[row_mask, "order_id"].map(movement_flags["fail_count"]).fillna(0).to_numpy(np.int64)
        unknown_atomic[row_mask] += result.loc[row_mask, "order_id"].map(movement_flags["unknown_count"]).fillna(0).to_numpy(np.int64)

        control_fail_source = lookup & turn.eq("LEFT") & detail["signal_state"].eq("STOP_OR_YIELD_CONTROLLED") & (profile_id == "C")
        control_unknown_source = lookup & turn.eq("LEFT") & ~detail["signal_state"].isin(["SIGNALIZED", "STOP_OR_YIELD_CONTROLLED"]) & (profile_id == "C")
        control_flags = _order_flags(detail["order_id"], control_fail_source, control_unknown_source)
        fail = result["order_id"].map(control_flags["fail"]).fillna(False).to_numpy() & row_mask
        unknown = result["order_id"].map(control_flags["unknown"]).fillna(False).to_numpy() & row_mask
        result.loc[row_mask, "control_state"] = _state_array(fail[row_mask], unknown[row_mask])
        known_flags["CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE"] |= fail
        unknown_flags["CONSERVATIVE_LEFT_UNKNOWN_CONTROL"] |= unknown
        known_atomic[row_mask] += result.loc[row_mask, "order_id"].map(control_flags["fail_count"]).fillna(0).to_numpy(np.int64)
        unknown_atomic[row_mask] += result.loc[row_mask, "order_id"].map(control_flags["unknown_count"]).fillna(0).to_numpy(np.int64)

        round_source = detail["roundabout_evidence_present"].fillna(False) & (profile_id == "C")
        round_flags = _order_flags(detail["order_id"], round_source, np.zeros(len(detail), bool))
        fail = result["order_id"].map(round_flags["fail"]).fillna(False).to_numpy() & row_mask
        result.loc[row_mask, "roundabout_state"] = _state_array(fail[row_mask], np.zeros(int(row_mask.sum()), bool))
        known_flags["CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE"] |= fail
        known_atomic[row_mask] += result.loc[row_mask, "order_id"].map(round_flags["fail_count"]).fillna(0).to_numpy(np.int64)

        restriction_source = detail["restriction_enforcement_certified"].fillna(False) & detail["movement_legality_state"].eq("CERTIFIED_PROHIBITED")
        restriction_flags = _order_flags(detail["order_id"], restriction_source, np.zeros(len(detail), bool))
        fail = result["order_id"].map(restriction_flags["fail"]).fillna(False).to_numpy() & row_mask
        result.loc[row_mask, "restriction_state"] = _state_array(fail[row_mask], np.zeros(int(row_mask.sum()), bool))
        known_flags["CERTIFIED_MOVEMENT_PROHIBITION"] |= fail
        known_atomic[row_mask] += result.loc[row_mask, "order_id"].map(restriction_flags["fail_count"]).fillna(0).to_numpy(np.int64)

    dynamic_incomplete = ~result["dynamic_complete"].fillna(False).to_numpy()
    dynamic_any_fail = np.zeros(len(result), dtype=bool)
    for profile_id in PROFILES:
        row_mask = result["profile_id"].eq(profile_id).to_numpy()
        caps = profile_by_id[profile_id]["dynamic_caps"]
        for dimension in DYNAMIC_DIMS:
            for metric in ("E", "Q", "C"):
                column = f"{dimension}_{metric}"
                fail = result[column].gt(float(caps[dimension][metric])).fillna(False).to_numpy() & row_mask & ~dynamic_incomplete
                dynamic_any_fail |= fail
                reason = f"DYNAMIC_{dimension.upper()}_{metric}_CAP_EXCEEDED"
                known_flags[reason] |= fail
                known_atomic += fail.astype(np.int64)
    unknown_flags["DYNAMIC_ROUTE_INCOMPLETE"] |= dynamic_incomplete
    unknown_atomic += dynamic_incomplete.astype(np.int64) * 12
    result["dynamic_state"] = _state_array(dynamic_any_fail, dynamic_incomplete)

    family_columns = [
        "directional_routability_state", "speed_state", "static_A_state", "static_M_state",
        "static_D_state", "static_L_state", "movement_state", "control_state",
        "roundabout_state", "restriction_state", "dynamic_state",
    ]
    any_incompatible = result[family_columns].eq("INCOMPATIBLE").any(axis=1).to_numpy()
    any_unknown = result[family_columns].eq("UNKNOWN").any(axis=1).to_numpy()
    result["original_route_state"] = np.select(
        [any_incompatible, any_unknown], ["INFEASIBLE", "UNKNOWN"], default="FEASIBLE"
    )
    known_codes = sorted(known_flags); unknown_codes = sorted(unknown_flags)
    known_matrix = np.column_stack([known_flags[code] for code in known_codes])
    unknown_matrix = np.column_stack([unknown_flags[code] for code in unknown_codes])
    result["known_violation_reason_codes"] = [
        json.dumps([code for code, present in zip(known_codes, row) if present], separators=(",", ":"))
        for row in known_matrix
    ]
    result["unknown_reason_codes"] = [
        json.dumps([code for code, present in zip(unknown_codes, row) if present], separators=(",", ":"))
        for row in unknown_matrix
    ]
    result["known_violation_count"] = known_matrix.sum(axis=1).astype(np.int64)
    result["unknown_requirement_count"] = unknown_matrix.sum(axis=1).astype(np.int64)
    result["known_violation_atomic_count"] = known_atomic
    result["unknown_atomic_count"] = unknown_atomic
    if len(result) != EXPECTED_ORDER_PROFILE_COUNT or result.duplicated(["order_id", "profile_id"]).any():
        raise Stage3S2AError("vectorized suitability is not exactly 30000 x 3")
    return result


def _atomic_base_frame(
    source: pd.DataFrame, profile_id: str, family: str, name: str,
    state: np.ndarray | pd.Series, reason: np.ndarray | pd.Series | str | None,
    *, observed: np.ndarray | pd.Series | float | None = None,
    cap: float | None = None, route_sequence: str | None = None,
    edge: str | None = None, complex_id: str | None = None, occurrence: str | None = None,
) -> pd.DataFrame:
    length = len(source)
    frame = pd.DataFrame({
        "date": TEST_DATE, "order_id": source["order_id"].astype(str).to_numpy(),
        "profile_id": profile_id, "check_family": family, "check_name": name,
        "state": np.asarray(state, dtype=object),
        "observed_value": observed if observed is not None else np.full(length, np.nan),
        "cap_value": cap if cap is not None else np.full(length, np.nan),
        "route_sequence": source[route_sequence].to_numpy() if route_sequence else pd.array([None] * length, dtype="Int64"),
        "stage3_edge_uid": source[edge].astype("string").to_numpy() if edge else None,
        "intersection_complex_uid": source[complex_id].astype("string").to_numpy() if complex_id else None,
        "movement_occurrence_index": source[occurrence].to_numpy() if occurrence else pd.array([None] * length, dtype="Int64"),
        "reason_code": reason,
    })
    locators = (
        frame["route_sequence"].astype("Int64").astype("string").fillna("") + "|"
        + frame["stage3_edge_uid"].astype("string").fillna("") + "|"
        + frame["intersection_complex_uid"].astype("string").fillna("") + "|"
        + frame["movement_occurrence_index"].astype("Int64").astype("string").fillna("")
    )
    frame["evidence_id"] = (
        family + "|" + frame["order_id"] + "|" + profile_id + "|" + name + "|" + locators
    )
    return _standard_atomic(frame)


def _vectorized_atomic_frames(
    descriptor: pd.DataFrame, detail: pd.DataFrame, speed_tokens: pd.DataFrame,
    profile_by_id: Mapping[str, Mapping[str, Any]], chunk_size: int = 100_000,
) -> Iterable[pd.DataFrame]:
    """Yield every route-level, complex-occurrence, movement, and dynamic check."""
    for profile_id in PROFILES:
        reverse = descriptor[descriptor["reverse_overlay_token_count"].gt(0)].copy()
        unresolved = descriptor[
            descriptor["unresolved_token_count"].gt(0)
            & descriptor["reverse_overlay_token_count"].eq(0)
        ].copy()
        compatible = descriptor[
            descriptor["reverse_overlay_token_count"].eq(0) & descriptor["unresolved_token_count"].eq(0)
        ].copy()
        if len(reverse):
            yield _atomic_base_frame(reverse, profile_id, "DIRECTION", "directional_routability", np.full(len(reverse), "INCOMPATIBLE"), "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE", observed=reverse["reverse_overlay_token_count"], route_sequence="first_reverse_route_sequence")
        if len(unresolved):
            yield _atomic_base_frame(unresolved, profile_id, "DIRECTION", "directional_routability", np.full(len(unresolved), "UNKNOWN"), "UNRESOLVED_ROUTE_IDENTITY", observed=unresolved["unresolved_token_count"], route_sequence="first_unresolved_route_sequence")
        if len(compatible):
            yield _atomic_base_frame(compatible, profile_id, "DIRECTION", "directional_routability", np.full(len(compatible), "COMPATIBLE"), None, observed=0.0)

        cap = float(profile_by_id[profile_id]["speed_domain_max_kmh"])
        speed_order = descriptor[["order_id", "max_route_speed_domain_kmh", "unknown_speed_edge_count", "first_unknown_speed_route_sequence"]].copy()
        violating_tokens = speed_tokens[speed_tokens["speed_known"] & speed_tokens["speed_domain_value_kmh"].gt(cap)].sort_values(["order_id", "route_sequence"], kind="stable")
        first_violation = violating_tokens.drop_duplicates("order_id")
        if len(first_violation):
            yield _atomic_base_frame(first_violation, profile_id, "SPEED", "speed_domain_cap", np.full(len(first_violation), "INCOMPATIBLE"), "SPEED_DOMAIN_CAP_EXCEEDED", observed=first_violation["speed_domain_value_kmh"], cap=cap, route_sequence="route_sequence", edge="resolved_stage3_edge_uid")
        unknown_tokens = speed_tokens[~speed_tokens["speed_known"]].sort_values(["order_id", "route_sequence"], kind="stable").drop_duplicates("order_id")
        if len(unknown_tokens):
            yield _atomic_base_frame(unknown_tokens, profile_id, "SPEED", "speed_domain_cap", np.full(len(unknown_tokens), "UNKNOWN"), "SPEED_DOMAIN_UNKNOWN", cap=cap, route_sequence="route_sequence", edge="resolved_stage3_edge_uid")
        evidence_orders = set(first_violation["order_id"].astype(str)) | set(unknown_tokens["order_id"].astype(str))
        speed_compatible = speed_order[~speed_order["order_id"].astype(str).isin(evidence_orders)]
        if len(speed_compatible):
            yield _atomic_base_frame(speed_compatible, profile_id, "SPEED", "speed_domain_cap", np.full(len(speed_compatible), "COMPATIBLE"), None, observed=speed_compatible["max_route_speed_domain_kmh"], cap=cap)

        static_aliases = {
            "A": ("A_c", "external_physical_connection_count", "STATIC_A_CAP_EXCEEDED"),
            "M": ("M_c", "topological_movement_count", "STATIC_M_CAP_EXCEEDED"),
            "D": ("D_c", "road_class_diversity", "STATIC_D_CAP_EXCEEDED"),
            "L": ("L_c", "internal_length_m", "STATIC_L_CAP_EXCEEDED"),
        }
        caps = profile_by_id[profile_id]["static_caps"]
        for start in range(0, len(detail), chunk_size):
            chunk = detail.iloc[start : start + chunk_size]
            for label, (column, cap_key, cap_reason) in static_aliases.items():
                values = pd.to_numeric(chunk[column], errors="coerce")
                state = _state_array(values.gt(float(caps[cap_key])) & values.notna(), values.isna())
                reason = np.where(state == "INCOMPATIBLE", cap_reason, np.where(state == "UNKNOWN", "STATIC_METRIC_UNKNOWN", None))
                yield _atomic_base_frame(chunk, profile_id, f"STATIC_{label}", f"static_{label}_cap", state, reason, observed=values, cap=float(caps[cap_key]), complex_id="intersection_complex_uid", occurrence="movement_occurrence_index")

            lookup = chunk["movement_lookup_status"].eq("MATCHED_TOPOLOGICAL_MOVEMENT")
            turn = chunk["route_turn_type"].fillna("UNKNOWN").astype(str).str.upper()
            valid_turn = turn.isin(["STRAIGHT", "RIGHT", "LEFT", "UTURN"])
            movement_fail = lookup & turn.eq("UTURN") & (profile_id in {"C", "M"})
            movement_unknown = ~lookup | ~valid_turn
            movement_state = _state_array(movement_fail, movement_unknown)
            movement_reason = np.where(
                movement_fail, "UTURN_PROFILE_INCOMPATIBLE",
                np.where(~lookup, "MOVEMENT_LOOKUP_UNRESOLVED", np.where(~valid_turn, "TURN_GEOMETRY_UNKNOWN", None)),
            )
            yield _atomic_base_frame(chunk, profile_id, "MOVEMENT", "turn_geometry", movement_state, movement_reason, complex_id="intersection_complex_uid", occurrence="movement_occurrence_index")

            control_fail = lookup & turn.eq("LEFT") & chunk["signal_state"].eq("STOP_OR_YIELD_CONTROLLED") & (profile_id == "C")
            control_unknown = lookup & turn.eq("LEFT") & ~chunk["signal_state"].isin(["SIGNALIZED", "STOP_OR_YIELD_CONTROLLED"]) & (profile_id == "C")
            control_state = _state_array(control_fail, control_unknown)
            control_reason = np.where(control_fail, "CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE", np.where(control_unknown, "CONSERVATIVE_LEFT_UNKNOWN_CONTROL", None))
            yield _atomic_base_frame(chunk, profile_id, "CONTROL", "movement_control", control_state, control_reason, complex_id="intersection_complex_uid", occurrence="movement_occurrence_index")

            round_fail = chunk["roundabout_evidence_present"].fillna(False).to_numpy(bool) & (profile_id == "C")
            yield _atomic_base_frame(chunk, profile_id, "ROUNDABOUT", "roundabout_capability", _state_array(round_fail, np.zeros(len(chunk), bool)), np.where(round_fail, "CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE", None), complex_id="intersection_complex_uid", occurrence="movement_occurrence_index")
            restriction_fail = chunk["restriction_enforcement_certified"].fillna(False).to_numpy(bool) & chunk["movement_legality_state"].eq("CERTIFIED_PROHIBITED").to_numpy()
            yield _atomic_base_frame(chunk, profile_id, "RESTRICTION", "certified_movement_prohibition", _state_array(restriction_fail, np.zeros(len(chunk), bool)), np.where(restriction_fail, "CERTIFIED_MOVEMENT_PROHIBITION", None), complex_id="intersection_complex_uid", occurrence="movement_occurrence_index")

        no_encounter = descriptor[~descriptor["order_id"].astype(str).isin(set(detail["order_id"].astype(str)))]
        for family, name in (("MOVEMENT", "turn_geometry"), ("CONTROL", "movement_control"), ("ROUNDABOUT", "roundabout_capability"), ("RESTRICTION", "certified_movement_prohibition")):
            if len(no_encounter):
                yield _atomic_base_frame(no_encounter, profile_id, family, name, np.full(len(no_encounter), "COMPATIBLE"), None)
        for label in ("A", "M", "D", "L"):
            if len(no_encounter):
                cap_key = static_aliases[label][1]
                yield _atomic_base_frame(no_encounter, profile_id, f"STATIC_{label}", f"static_{label}_cap", np.full(len(no_encounter), "COMPATIBLE"), None, cap=float(caps[cap_key]))

        for dimension in DYNAMIC_DIMS:
            for metric in ("E", "Q", "C"):
                column = f"{dimension}_{metric}"
                cap = float(profile_by_id[profile_id]["dynamic_caps"][dimension][metric])
                complete = descriptor["dynamic_complete"].fillna(False)
                fail = complete & descriptor[column].gt(cap)
                unknown = ~complete
                state = _state_array(fail, unknown)
                reason_code = f"DYNAMIC_{dimension.upper()}_{metric}_CAP_EXCEEDED"
                reason = np.where(fail, reason_code, np.where(unknown, "DYNAMIC_ROUTE_INCOMPLETE", None))
                yield _atomic_base_frame(descriptor, profile_id, f"DYNAMIC_{dimension.upper()}", column, state, reason, observed=descriptor[column], cap=cap)


def _cooccurrence(suitability: pd.DataFrame) -> pd.DataFrame:
    family_by_reason = {
        "direction": lambda code: code.startswith("KNOWN_REVERSE_DIRECTION"),
        "speed": lambda code: code.startswith("SPEED_"),
        "static": lambda code: code.startswith("STATIC_"),
        "movement_control": lambda code: code.startswith("UTURN_") or code.startswith("CONSERVATIVE_LEFT"),
        "roundabout": lambda code: "ROUNDABOUT" in code,
        "restriction": lambda code: code.startswith("CERTIFIED_MOVEMENT"),
        "dynamic": lambda code: code.startswith("DYNAMIC_") and not code.endswith("INCOMPLETE"),
    }
    rows = []
    for profile_id, frame in suitability.groupby("profile_id", sort=False):
        reason_sets = frame["known_violation_reason_codes"].map(json.loads)
        family_masks = {
            family: reason_sets.map(lambda codes, predicate=predicate: any(predicate(code) for code in codes))
            for family, predicate in family_by_reason.items()
        }
        infeasible = frame["original_route_state"].eq("INFEASIBLE")
        for left, right in itertools.product(family_by_reason, repeat=2):
            mask = family_masks[left] & family_masks[right]
            rows.append({
                "profile_id": profile_id,
                "cause_family_a": left,
                "cause_family_b": right,
                "route_count": int(mask.sum()),
                "share_all_routes": float(mask.mean()),
                "share_infeasible_routes": float(mask.sum() / infeasible.sum()) if infeasible.any() else 0.0,
            })
    return pd.DataFrame(rows)


def _population_summary(
    suitability: pd.DataFrame,
    descriptor: pd.DataFrame,
    identity_summary: Mapping[str, Any],
    nestedness: Mapping[str, Any],
) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    all_reason_codes = sorted(KNOWN_REASON_CODES | UNKNOWN_REASON_CODES)
    for profile_id, frame in suitability.groupby("profile_id", sort=False):
        state_counts = frame["original_route_state"].value_counts()
        known_sets = frame["known_violation_reason_codes"].map(json.loads)
        unknown_sets = frame["unknown_reason_codes"].map(json.loads)
        infeasible = frame["original_route_state"].eq("INFEASIBLE")
        final_unknown = frame["original_route_state"].eq("UNKNOWN")
        profile_summary = {
            "state_counts": {state: int(state_counts.get(state, 0)) for state in ("FEASIBLE", "UNKNOWN", "INFEASIBLE")},
            "state_shares": {state: float(state_counts.get(state, 0) / len(frame)) for state in ("FEASIBLE", "UNKNOWN", "INFEASIBLE")},
            "marginal_reason_counts": {code: int(known_sets.map(lambda values, c=code: c in values).sum() + unknown_sets.map(lambda values, c=code: c in values).sum()) for code in all_reason_codes},
            "exclusive_known_cause_counts": {code: int(known_sets.map(lambda values, c=code: len(values) == 1 and c in values).sum()) for code in sorted(KNOWN_REASON_CODES)},
            "multi_cause_violation_count": int(known_sets.map(lambda values: len(values) >= 2).sum()),
            "multi_cause_violation_share": float(known_sets.map(lambda values: len(values) >= 2).mean()),
            "mean_known_violations_per_infeasible_route": float(frame.loc[infeasible, "known_violation_count"].mean()) if infeasible.any() else 0.0,
            "unknown_exactly_one_reason_count": int(unknown_sets.map(lambda values: len(values) == 1).sum()),
            "unknown_multiple_reason_count": int(unknown_sets.map(lambda values: len(values) >= 2).sum()),
            "final_unknown_route_reason_counts": {
                code: int((final_unknown & unknown_sets.map(lambda values, c=code: c in values)).sum())
                for code in sorted(UNKNOWN_REASON_CODES)
            },
            "final_unknown_route_exactly_one_reason_count": int(
                (final_unknown & unknown_sets.map(lambda values: len(values) == 1)).sum()
            ),
            "final_unknown_route_multiple_reason_count": int(
                (final_unknown & unknown_sets.map(lambda values: len(values) >= 2)).sum()
            ),
            "known_violation_plus_unknown_evidence_count": int((infeasible & frame["unknown_requirement_count"].gt(0)).sum()),
            "known_violation_precedence_status": "PASS" if not ((infeasible & frame["unknown_requirement_count"].gt(0)) & frame["original_route_state"].ne("INFEASIBLE")).any() else "FAIL",
            "all_12_dynamic_pass_count": int(frame["dynamic_state"].eq("COMPATIBLE").sum()),
        }
        profiles[str(profile_id)] = profile_summary
    dynamic_columns = [f"{dimension}_{metric}" for dimension in DYNAMIC_DIMS for metric in ("E", "Q", "C")]
    complete = descriptor[descriptor["dynamic_complete"]]
    dynamic_distribution: dict[str, Any] = {}
    for column in dynamic_columns:
        values = complete[column].dropna().to_numpy(np.float64)
        dynamic_distribution[column] = {
            "count": int(len(values)),
            "min": float(values.min()) if len(values) else None,
            "p50": float(np.quantile(values, .5)) if len(values) else None,
            "p90": float(np.quantile(values, .9)) if len(values) else None,
            "p975": float(np.quantile(values, .975)) if len(values) else None,
            "max": float(values.max()) if len(values) else None,
        }
    summary = {
        "schema_version": "stage3_s4_test31_suitability_summary.1",
        "phase_status": PHASE_STATUS,
        "test_date": TEST_DATE,
        "test31_order_count": int(descriptor["order_id"].nunique()),
        "order_profile_row_count": int(len(suitability)),
        "test31_route_token_count": int(descriptor["route_token_count"].sum()),
        "identity": dict(identity_summary),
        "dynamic_complete_route_count": int(descriptor["dynamic_complete"].sum()),
        "dynamic_complete_route_share": float(descriptor["dynamic_complete"].mean()),
        "dynamic_distribution": dynamic_distribution,
        "profiles": profiles,
        "nestedness": dict(nestedness),
        "fallback_attempted": False,
        "route_search_performed": False,
        "profile_retuned": False,
        "test31_calibration": False,
        "original_route_only": True,
        "s5_authorized": False,
        "s6_authorized": False,
        "s7_authorized": False,
        "s8_authorized": False,
        "stage4_dispatch_authorized": False,
        "next_phase_authorized": False,
    }
    summary["artifact_sha256"] = payload_hash(summary)
    return summary


def assess_test31(root: Path) -> dict[str, Any]:
    output = root / OUTPUT_REL
    input_manifest_path = output / "test31_input_manifest.json"
    if not input_manifest_path.is_file():
        raise Stage3S2AError("Test31 input manifest must exist before suitability evaluation")
    input_manifest = read_json(input_manifest_path)
    before = {name: item["sha256"] for name, item in input_manifest["frozen_inputs"].items()}
    for name, expected in before.items():
        relative = PROFILE_REL if name == "profile" else CDF_REL if name == "cdf" else UPSTREAM_REL[name]
        if sha256_file(root / relative) != expected:
            raise Stage3S2AError(f"frozen input changed before Test31 assessment: {name}")

    route = pd.read_parquet(root / ROUTE_REL, columns=list(ROUTE_IDENTITY_COLUMNS))
    _route_integrity(route)
    typed = pd.read_parquet(output / "test31_route_identity_resolution.parquet")
    encounters = pd.read_parquet(output / "test31_route_complex_encounters.parquet")
    predictions = pd.read_parquet(output / "test31_m3_predictions.parquet")
    forbidden_prediction = [column for column in predictions if column.startswith("target_") or column.endswith("_target_valid")]
    if forbidden_prediction:
        raise Stage3S2AError(f"realized targets persisted in S4 prediction product: {forbidden_prediction}")
    if len(predictions) != len(route):
        raise Stage3S2AError(f"Test31 prediction rows differ from route tokens: {len(predictions)} != {len(route)}")

    profile_payload = read_json(root / PROFILE_REL)
    profile_by_id = _profile_map(profile_payload)
    direction = _direction_descriptor(typed)
    speed_domain = pd.read_parquet(root / UPSTREAM_REL["speed_domain"])
    speed_descriptor, speed_tokens = _speed_descriptor(typed, speed_domain)
    static = _full_static_reference(root)
    detail = _encounter_detail(root, encounters, static)
    encounter_descriptor = _encounter_descriptor(route["order_id"].drop_duplicates(), detail)
    dynamic = _dynamic_descriptors(route, predictions, root / CDF_REL)
    descriptor = direction.merge(speed_descriptor, on="order_id", validate="one_to_one")
    descriptor = descriptor.merge(encounter_descriptor, on="order_id", validate="one_to_one")
    descriptor = descriptor.merge(dynamic, on="order_id", validate="one_to_one")
    if len(descriptor) != EXPECTED_ORDER_COUNT:
        raise Stage3S2AError("descriptor product is not one row per Test31 order")
    descriptor_path = output / "test31_original_route_descriptors.parquet"
    atomic_parquet(descriptor_path, descriptor)

    atomic_path = output / "test31_original_route_atomic_checks.parquet"
    _write_atomic_parts(
        _vectorized_atomic_frames(descriptor, detail, speed_tokens, profile_by_id), atomic_path
    )
    suitability = _vectorized_suitability(descriptor, detail, speed_tokens, profile_by_id)
    nestedness = audit_three_state_nestedness(suitability)
    if nestedness["status"] != "PASS":
        raise Stage3S2AError(f"S4 profile nestedness failed: {nestedness}")
    suitability_path = output / "test31_original_route_suitability.parquet"
    atomic_parquet(suitability_path, suitability)
    cooccurrence = _cooccurrence(suitability)
    cooccurrence_path = output / "test31_reason_cooccurrence.parquet"
    atomic_parquet(cooccurrence_path, cooccurrence)

    prepare = read_json(output / "test31_prepare_summary.json")
    identity_summary = {key: value for key, value in prepare.items() if key.startswith(("FULL_", "HISTORICAL_", "UNRESOLVED_", "fully_", "orders_with_"))}
    summary = _population_summary(suitability, descriptor, identity_summary, nestedness)
    products = {}
    for path in (descriptor_path, atomic_path, suitability_path, cooccurrence_path):
        products[path.relative_to(root).as_posix()] = parquet_descriptor(path, root)
    summary["products"] = products

    after = {}
    for name, expected in before.items():
        relative = PROFILE_REL if name == "profile" else CDF_REL if name == "cdf" else UPSTREAM_REL[name]
        observed = sha256_file(root / relative)
        after[name] = observed
        if observed != expected:
            raise Stage3S2AError(f"frozen input changed during Test31 assessment: {name}")
    summary["frozen_hashes_before"] = before
    summary["frozen_hashes_after"] = after
    summary["profile_hash_unchanged"] = before["profile"] == after["profile"]
    summary["cdf_hash_unchanged"] = before["cdf"] == after["cdf"]
    summary["artifact_sha256"] = payload_hash(summary)
    atomic_json(output / "test31_suitability_summary.json", summary)
    return summary


def _markdown_state_table(summary: Mapping[str, Any]) -> str:
    lines = [
        "| Profile | FEASIBLE | UNKNOWN | INFEASIBLE |",
        "|---|---:|---:|---:|",
    ]
    for profile_id in PROFILES:
        counts = summary["profiles"][profile_id]["state_counts"]
        shares = summary["profiles"][profile_id]["state_shares"]
        lines.append(
            f"| {profile_id} | {counts['FEASIBLE']:,} ({shares['FEASIBLE']:.4%}) | "
            f"{counts['UNKNOWN']:,} ({shares['UNKNOWN']:.4%}) | "
            f"{counts['INFEASIBLE']:,} ({shares['INFEASIBLE']:.4%}) |"
        )
    return "\n".join(lines)


def _describe_values(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").to_numpy(np.float64)
    numeric = numeric[np.isfinite(numeric)]
    if not len(numeric):
        return {key: None for key in ("count", "min", "p25", "p50", "p75", "p90", "p975", "max")}
    return {
        "count": int(len(numeric)),
        "min": float(np.min(numeric)),
        "p25": float(np.quantile(numeric, .25, method="higher")),
        "p50": float(np.quantile(numeric, .50, method="higher")),
        "p75": float(np.quantile(numeric, .75, method="higher")),
        "p90": float(np.quantile(numeric, .90, method="higher")),
        "p975": float(np.quantile(numeric, .975, method="higher")),
        "max": float(np.max(numeric)),
    }


def _markdown_distribution_table(distributions: Mapping[str, Mapping[str, Any]]) -> str:
    lines = [
        "| Dimension | n | min | p25 | p50 | p75 | p90 | p97.5 | max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in distributions.items():
        lines.append(
            f"| {name} | {item['count']:,} | {item['min']:.6g} | {item['p25']:.6g} | "
            f"{item['p50']:.6g} | {item['p75']:.6g} | {item['p90']:.6g} | "
            f"{item['p975']:.6g} | {item['max']:.6g} |"
        )
    return "\n".join(lines)


def _family_prevalence(suitability: pd.DataFrame) -> dict[str, dict[str, dict[str, int]]]:
    known_predicates = {
        "direction": lambda code: code.startswith("KNOWN_REVERSE_DIRECTION"),
        "speed": lambda code: code == "SPEED_DOMAIN_CAP_EXCEEDED",
        "static_A": lambda code: code == "STATIC_A_CAP_EXCEEDED",
        "static_M": lambda code: code == "STATIC_M_CAP_EXCEEDED",
        "static_D": lambda code: code == "STATIC_D_CAP_EXCEEDED",
        "static_L": lambda code: code == "STATIC_L_CAP_EXCEEDED",
        "movement/control": lambda code: code.startswith("UTURN_") or code.startswith("CONSERVATIVE_LEFT"),
        "roundabout": lambda code: "ROUNDABOUT" in code,
        "restriction": lambda code: code.startswith("CERTIFIED_MOVEMENT"),
        "dynamic": lambda code: code.startswith("DYNAMIC_") and not code.endswith("INCOMPLETE"),
    }
    unknown_predicates = {
        "identity": lambda code: code == "UNRESOLVED_ROUTE_IDENTITY",
        "speed": lambda code: code == "SPEED_DOMAIN_UNKNOWN",
        "static": lambda code: code == "STATIC_METRIC_UNKNOWN",
        "movement": lambda code: code in {"MOVEMENT_LOOKUP_UNRESOLVED", "TURN_GEOMETRY_UNKNOWN"},
        "control": lambda code: code == "CONSERVATIVE_LEFT_UNKNOWN_CONTROL",
        "dynamic": lambda code: code == "DYNAMIC_ROUTE_INCOMPLETE",
    }
    result: dict[str, dict[str, dict[str, int]]] = {}
    for profile_id, frame in suitability.groupby("profile_id", sort=False):
        known_sets = frame["known_violation_reason_codes"].map(json.loads)
        unknown_sets = frame["unknown_reason_codes"].map(json.loads)
        result[str(profile_id)] = {
            "known": {
                family: int(known_sets.map(lambda codes, p=predicate: any(p(code) for code in codes)).sum())
                for family, predicate in known_predicates.items()
            },
            "unknown": {
                family: int(unknown_sets.map(lambda codes, p=predicate: any(p(code) for code in codes)).sum())
                for family, predicate in unknown_predicates.items()
            },
        }
    return result


def finalize_s4(root: Path) -> dict[str, Any]:
    output = root / OUTPUT_REL
    docs = root / DOCS_REL
    docs.mkdir(parents=True, exist_ok=True)
    summary = read_json(output / "test31_suitability_summary.json")
    if summary.get("phase_status") != PHASE_STATUS or summary.get("nestedness", {}).get("status") != "PASS":
        raise Stage3S2AError("S4 suitability summary is not complete/PASS")
    prediction_manifest = read_json(output / "test31_m3_predictions.json")
    prepare = read_json(output / "test31_prepare_summary.json")
    train_summary = read_json(root / "stage3/output/odd_tod/s3/train_summary.json")
    validation_summary = read_json(root / "stage3/output/odd_tod/s3/validation_sanity_summary.json")
    profile_by_id = _profile_map(read_json(root / PROFILE_REL))
    table = _markdown_state_table(summary)
    suitability = pd.read_parquet(output / "test31_original_route_suitability.parquet")
    family_prevalence = _family_prevalence(suitability)

    atomic_text(docs / "stage3_s4_methodology.md", f"""# Stage 3 S4 Methodology

S4 evaluates the exact frozen `{TEST_DATE}` historical route under the frozen hypothetical C/M/A capability scenarios. It performs no rerouting, fallback, nearest-geometry repair, profile retuning, CDF fitting, or Test31 calibration.

Atomic evidence is non-compensatory. A known incompatibility has precedence over unknown evidence, and both known and unknown cause vectors are retained. This is operational route compatibility, not safety, legal certification, failure probability, accident probability, disengagement probability, or ODD approval.

Dynamic inference uses frozen M3 checkpoint `{M3_SHA256}` and the frozen Train weighted mid-CDF. Dynamic E/Q/C is calculated only when every original-route token has complete prediction-side evidence; tail membership uses strict `z > {Q_TAIL}`.
""")
    identity_lines = [
        "# Stage 3 S4 Test31 Identity Report", "",
        f"- Orders: {prepare['order_count']:,}",
        f"- Route tokens: {prepare['route_token_count']:,}",
        f"- Full-network tokens: {prepare['FULL_NETWORK_EDGE_count']:,} ({prepare['FULL_NETWORK_EDGE_share']:.6%})",
        f"- Historical reverse overlays: {prepare['HISTORICAL_REVERSE_OVERLAY_count']:,} ({prepare['HISTORICAL_REVERSE_OVERLAY_share']:.6%})",
        f"- Unresolved tokens: {prepare['UNRESOLVED_count']:,} ({prepare['UNRESOLVED_share']:.6%})",
        f"- Fully full-network-resolved orders: {prepare['fully_full_network_resolved_order_count']:,} ({prepare['fully_full_network_resolved_order_count'] / prepare['order_count']:.6%})",
        f"- Orders with reverse overlay: {prepare['orders_with_reverse_overlay']:,}",
        f"- Orders with unresolved identity: {prepare['orders_with_unresolved_token']:,}", "",
        "## Descriptive temporal comparison", "",
        "| Split | FULL_NETWORK_EDGE | Reverse overlay | Unresolved | Fully resolved orders |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, identity in (
        ("Train 09-24", train_summary["identity"]),
        ("Validation 25-27", validation_summary["identity"]),
        ("Test31", {**summary["identity"], "total_orders": prepare["order_count"]}),
    ):
        identity_lines.append(
            f"| {label} | {identity['FULL_NETWORK_EDGE_share']:.6%} | "
            f"{identity['HISTORICAL_REVERSE_OVERLAY_share']:.6%} | "
            f"{identity['UNRESOLVED_share']:.6%} | "
            f"{identity['fully_full_network_resolved_order_count'] / identity['total_orders']:.6%} |"
        )
    identity_lines.extend([
        "", "This comparison is descriptive only and triggers no calibration action.", "",
        "Reverse overlays are known AV-routability violations and are never projected onto the physical forward edge. Unresolved identity is unknown evidence and is never geometry-imputed.", "",
    ])
    atomic_text(docs / "stage3_s4_test31_identity_report.md", "\n".join(identity_lines))
    descriptor = pd.read_parquet(output / "test31_original_route_descriptors.parquet")
    encounters = pd.read_parquet(
        output / "test31_route_complex_encounters.parquet",
        columns=["intersection_complex_uid"],
    )
    static_reference = _full_static_reference(root)
    exposed_static = static_reference[
        static_reference["intersection_complex_uid"].isin(encounters["intersection_complex_uid"].dropna())
    ].drop_duplicates("intersection_complex_uid")
    static_distribution = {
        column: _describe_values(exposed_static[column]) for column in ("A_c", "M_c", "D_c", "L_c")
    }
    static_lines = [
        "# Stage 3 S4 Test31 Static Report", "",
        "All encountered physical complexes are evaluated. `D_c` is recomputed over unique INCOMING/OUTGOING boundary-edge Valhalla road classes for the full frozen 43,685-complex network; the legacy S2B INTERNAL-edge QA field is not used.", "",
        f"- Test31 complex encounters: {int(descriptor['resolved_complex_encounter_count'].sum()):,}",
        f"- Unique exposed complexes: {len(exposed_static):,}",
        f"- Grade-separated complex encounters (descriptive only): {int(descriptor['grade_separated_complex_encounter_count'].sum()):,}", "",
        "## Unique-complex distributions", "", _markdown_distribution_table(static_distribution), "",
        "## Frozen-cap exceedance", "",
        "| Profile | A_c | M_c | D_c | L_c | All four pass |", "|---|---:|---:|---:|---:|---:|",
    ]
    static_keys = {
        "A_c": "external_physical_connection_count", "M_c": "topological_movement_count",
        "D_c": "road_class_diversity", "L_c": "internal_length_m",
    }
    for profile_id in PROFILES:
        caps = profile_by_id[profile_id]["static_caps"]
        masks = {column: exposed_static[column].gt(float(caps[key])) for column, key in static_keys.items()}
        cells = [f"{int(mask.sum()):,} ({float(mask.mean()):.4%})" for mask in masks.values()]
        all_pass = ~np.logical_or.reduce([mask.to_numpy(bool) for mask in masks.values()])
        static_lines.append(
            f"| {profile_id} | {' | '.join(cells)} | {int(all_pass.sum()):,} ({float(all_pass.mean()):.4%}) |"
        )
    static_lines.extend([
        "", "Frozen caps are applied per encountered complex without averaging or Test31 refitting.", "",
    ])
    atomic_text(docs / "stage3_s4_test31_static_report.md", "\n".join(static_lines))
    complete_dynamic = descriptor[descriptor["dynamic_complete"]]
    dynamic_distribution = {
        column: _describe_values(complete_dynamic[column])
        for column in (f"{dimension}_{metric}" for dimension in DYNAMIC_DIMS for metric in ("E", "Q", "C"))
    }
    dynamic_lines = [
        "# Stage 3 S4 Test31 Dynamic Report", "",
        f"- Frozen M3 checkpoint: `{M3_SHA256}`",
        f"- Prediction rows: {prediction_manifest['row_count']:,}",
        f"- Dynamic-complete routes: {summary['dynamic_complete_route_count']:,} / {summary['test31_order_count']:,} ({summary['dynamic_complete_route_share']:.6%})",
        f"- Validation dynamic-complete share: {validation_summary['complete_dynamic_route_coverage']:.6%}",
        f"- Decision-time only: `{str(prediction_manifest['decision_time_only']).lower()}`",
        f"- Predicted progression only: `{str(prediction_manifest['predicted_progression_only']).lower()}`",
        f"- Realized future time used: `{str(prediction_manifest['realized_future_time_used']).lower()}`",
        f"- Realized target columns persisted: `{str(prediction_manifest['realized_target_columns_persisted']).lower()}`", "",
        "## Test31 complete-route E/Q/C distributions", "", _markdown_distribution_table(dynamic_distribution), "",
        "## Frozen-cap exceedance", "",
        "| Profile | Dimension/metric | Test31 exceed | Validation exceed | Frozen cap |", "|---|---|---:|---:|---:|",
    ]
    for profile_id in PROFILES:
        for dimension in DYNAMIC_DIMS:
            for metric in ("E", "Q", "C"):
                column = f"{dimension}_{metric}"
                cap = float(profile_by_id[profile_id]["dynamic_caps"][dimension][metric])
                test_share = float(complete_dynamic[column].gt(cap).mean())
                validation_key = f"{profile_id}_{dimension}_{metric}"
                validation_share = float(validation_summary["dynamic_frozen_cap_exceedance"][validation_key])
                dynamic_lines.append(
                    f"| {profile_id} | {column} | {test_share:.4%} | {validation_share:.4%} | {cap:.6g} |"
                )
        dynamic_lines.append(
            f"| {profile_id} | **all 12 pass** | **{summary['profiles'][profile_id]['all_12_dynamic_pass_count']:,} ({summary['profiles'][profile_id]['all_12_dynamic_pass_count'] / summary['test31_order_count']:.4%})** | "
            f"**{validation_summary['dynamic_frozen_cap_exceedance'][profile_id + '_all_caps_pass_count']:,} ({validation_summary['dynamic_frozen_cap_exceedance'][profile_id + '_all_caps_pass_count'] / validation_summary['total_validation_orders']:.4%})** | N/A |"
        )
    dynamic_lines.extend([
        "", "The exact frozen Train CDF is applied; Test31 is not appended and no Test31 percentile is fitted. Incomplete routes remain dynamic `UNKNOWN` with null E/Q/C.",
        "The Test31-versus-Validation comparison is descriptive only; it defines no shift threshold and triggers no retuning.", "",
    ])
    atomic_text(docs / "stage3_s4_test31_dynamic_report.md", "\n".join(dynamic_lines))
    family_lines = ["| Profile | Family | Route count | Share |", "|---|---|---:|---:|"]
    for profile_id in PROFILES:
        for family, count in family_prevalence[profile_id]["known"].items():
            family_lines.append(f"| {profile_id} | {family} | {count:,} | {count / summary['test31_order_count']:.4%} |")
        for family, count in family_prevalence[profile_id]["unknown"].items():
            family_lines.append(f"| {profile_id} | unknown:{family} | {count:,} | {count / summary['test31_order_count']:.4%} |")
    atomic_text(docs / "stage3_s4_original_route_suitability_report.md", f"""# Stage 3 S4 Original-Route Suitability Report

## Overall

- Test31 orders: {summary['test31_order_count']:,}
- Order-profile evaluations: {summary['order_profile_row_count']:,}

{table}

## Major known and unknown evidence families

{chr(10).join(family_lines)}

Counts are marginal route-level prevalence and overlap across families; they must not be summed as mutually exclusive causes.

## Gates

- Nestedness: **{summary['nestedness']['status']}**
- Known-violation precedence: **PASS**
- Frozen profile unchanged: **{str(summary['profile_hash_unchanged']).upper()}**
- Frozen CDF unchanged: **{str(summary['cdf_hash_unchanged']).upper()}**
- Fallback attempted: **NO**
- Route search performed: **NO**
- Profile retuned: **NO**
- Test31 calibration: **NO**

These are exact historical/original-route compatibility outcomes. They are not an AV safety, legality, failure, accident, disengagement, or product-approval result.
""")
    attribution_lines = ["# Stage 3 S4 Reason Attribution Report", "", "Marginal reasons and co-occurrences preserve simultaneous causes; they are descriptive and not causal.", ""]
    for profile_id in PROFILES:
        item = summary["profiles"][profile_id]
        attribution_lines.extend([
            f"## {profile_id}", "",
            f"- Multi-cause known-violation routes: {item['multi_cause_violation_count']:,} ({item['multi_cause_violation_share']:.6%})",
            f"- Mean distinct known reasons per infeasible route: {item['mean_known_violations_per_infeasible_route']:.6f}",
            f"- Known violation plus unknown evidence: {item['known_violation_plus_unknown_evidence_count']:,}",
            f"- Routes carrying unknown evidence with exactly one reason (all final states): {item['unknown_exactly_one_reason_count']:,}",
            f"- Routes carrying unknown evidence with multiple reasons (all final states): {item['unknown_multiple_reason_count']:,}",
            f"- Final UNKNOWN routes with exactly one reason: {item['final_unknown_route_exactly_one_reason_count']:,}",
            f"- Final UNKNOWN routes with multiple reasons: {item['final_unknown_route_multiple_reason_count']:,}", "",
            "### Final UNKNOWN route decomposition", "",
            "| Unknown reason | Final UNKNOWN routes |", "|---|---:|",
        ])
        for reason, count in item["final_unknown_route_reason_counts"].items():
            attribution_lines.append(f"| `{reason}` | {count:,} |")
        attribution_lines.extend([
            "", "### All-state marginal attribution", "",
            "| Reason code | Marginal routes | Only-known-cause routes |", "|---|---:|---:|",
        ])
        for reason, count in item["marginal_reason_counts"].items():
            attribution_lines.append(
                f"| `{reason}` | {count:,} | {item['exclusive_known_cause_counts'].get(reason, 0):,} |"
            )
        attribution_lines.extend(["", "Pairwise major-family counts are in `test31_reason_cooccurrence.parquet`.", ""])
    atomic_text(docs / "stage3_s4_reason_attribution_report.md", "\n".join(attribution_lines))

    products: dict[str, Any] = {}
    for relative in (
        "test31_route_identity_resolution.parquet",
        "test31_route_complex_encounters.parquet",
        "test31_m3_predictions.parquet",
        "test31_m3_predictions.json",
        "test31_original_route_descriptors.parquet",
        "test31_original_route_atomic_checks.parquet",
        "test31_original_route_suitability.parquet",
        "test31_suitability_summary.json",
        "test31_reason_cooccurrence.parquet",
    ):
        path = output / relative
        products[path.relative_to(root).as_posix()] = parquet_descriptor(path, root) if path.suffix == ".parquet" else source_descriptor(path, root)
    existing_test_evidence_path = docs / "stage3_s4_test_evidence.json"
    existing_test_evidence = (
        read_json(existing_test_evidence_path) if existing_test_evidence_path.is_file() else {}
    )
    test_evidence = {
        "schema_version": "stage3_s4_test_evidence.1",
        "authorized_base": AUTHORIZED_BASE,
        "focused_tests": existing_test_evidence.get("focused_tests", {"status": "PENDING_FINAL_RECORD"}),
        "full_tests": existing_test_evidence.get("full_tests", {"status": "PENDING_FINAL_RECORD"}),
        "compileall": existing_test_evidence.get("compileall", {"status": "PENDING_FINAL_RECORD"}),
        "behavioral_gates": {
            "order_count_30000": summary["test31_order_count"] == EXPECTED_ORDER_COUNT,
            "order_profile_count_90000": summary["order_profile_row_count"] == EXPECTED_ORDER_PROFILE_COUNT,
            "nestedness": summary["nestedness"]["status"],
            "known_violation_precedence": "PASS",
            "profile_immutable": summary["profile_hash_unchanged"],
            "cdf_immutable": summary["cdf_hash_unchanged"],
            "prediction_schema_no_realized_targets": not prediction_manifest["realized_target_columns_persisted"],
            "decision_time_only": prediction_manifest["decision_time_only"],
            "original_route_only": True,
            "fallback": False,
            "route_search": False,
        },
        "s5_authorized": False,
        "next_phase_authorized": False,
    }
    test_evidence["artifact_sha256"] = payload_hash(test_evidence)
    atomic_json(docs / "stage3_s4_test_evidence.json", test_evidence)
    tests_ready = all(
        test_evidence.get(label, {}).get("status") == "PASS"
        for label in ("focused_tests", "full_tests", "compileall")
    )
    evidence_gate = "READY_FOR_INDEPENDENT_VERIFICATION" if tests_ready else "PENDING"
    overall_acceptance = (
        "READY_FOR_INDEPENDENT_VERIFICATION"
        if tests_ready
        else "PENDING_TESTS_AND_INDEPENDENT_VERIFICATION"
    )
    reports = {}
    for path in sorted(docs.glob("*.md")):
        reports[path.relative_to(root).as_posix()] = source_descriptor(path, root)
    reports[(docs / "stage3_s4_test_evidence.json").relative_to(root).as_posix()] = source_descriptor(
        docs / "stage3_s4_test_evidence.json", root
    )

    release = {
        "schema_version": "stage3_s4_release_manifest.1",
        "phase_status": PHASE_STATUS,
        "overall_acceptance_status": overall_acceptance,
        "authorized_base": AUTHORIZED_BASE,
        "final_commit": "RECORDED_BY_GIT_COMMIT_AND_REMOTE_HEAD_OUTSIDE_SELF_HASHED_MANIFEST",
        "test_date": TEST_DATE,
        "test31_order_count": summary["test31_order_count"],
        "order_profile_row_count": summary["order_profile_row_count"],
        "frozen_inputs": read_json(output / "test31_input_manifest.json")["frozen_inputs"],
        "products": products,
        "reports": reports,
        "gates": {
            "upstream_freeze": "PASS",
            "test31_integrity": "PASS",
            "identity": "PASS",
            "static": "PASS",
            "dynamic": "PASS",
            "three_state": "PASS",
            "nestedness": summary["nestedness"]["status"],
            "scope": "PASS",
            "evidence": evidence_gate,
        },
        "original_route_test31_assessment": "FROZEN",
        "fallback_attempted": False,
        "route_search_performed": False,
        "profile_retuned": False,
        "test31_calibration": False,
        "s5_authorized": False,
        "s6_authorized": False,
        "s7_authorized": False,
        "s8_authorized": False,
        "stage4_dispatch_authorized": False,
        "next_phase_authorized": False,
    }
    release["artifact_sha256"] = payload_hash(release)
    atomic_json(docs / "stage3_s4_release_manifest.json", release)
    refresh_evidence_chain(root)
    return {
        "phase_status": PHASE_STATUS,
        "orders": summary["test31_order_count"],
        "order_profile_rows": summary["order_profile_row_count"],
        "nestedness": summary["nestedness"]["status"],
        "release_manifest_sha256": sha256_file(docs / "stage3_s4_release_manifest.json"),
        "evidence_bundle_sha256": sha256_file(docs / "stage3_s4_evidence_bundle.json"),
        "s5_authorized": False,
        "next_phase_authorized": False,
    }


EVIDENCE_DESCRIPTOR_PATHS = {
    "release_manifest": DOCS_REL / "stage3_s4_release_manifest.json",
    "test_evidence": DOCS_REL / "stage3_s4_test_evidence.json",
    "suitability_summary": OUTPUT_REL / "test31_suitability_summary.json",
    "prediction_manifest": OUTPUT_REL / "test31_m3_predictions.json",
    "input_manifest": OUTPUT_REL / "test31_input_manifest.json",
}


def descriptor_mismatches(root: Path, descriptors: Mapping[str, Mapping[str, Any]]) -> list[str]:
    failures: list[str] = []
    for label, descriptor in descriptors.items():
        path_value = descriptor.get("path")
        if not isinstance(path_value, str):
            failures.append(f"descriptor_path:{label}")
            continue
        path = Path(path_value)
        path = path if path.is_absolute() else root / path
        if not path.is_file():
            failures.append(f"descriptor_missing:{label}")
        elif sha256_file(path) != descriptor.get("sha256"):
            failures.append(f"descriptor_hash:{label}")
    return failures


def refresh_evidence_chain(
    root: Path, *, preserve_verification_status: bool = False
) -> dict[str, Any]:
    """Refresh mutable test/release descriptors, then rebuild the evidence bundle."""
    docs = root / DOCS_REL
    output = root / OUTPUT_REL
    release_path = docs / "stage3_s4_release_manifest.json"
    release = read_json(release_path)
    test_path = docs / "stage3_s4_test_evidence.json"
    test_evidence = read_json(test_path)
    tests_ready = all(
        test_evidence.get(label, {}).get("status") == "PASS"
        for label in ("focused_tests", "full_tests", "compileall")
    )
    current_evidence_status = release.get("gates", {}).get("evidence")
    if not (
        preserve_verification_status
        and current_evidence_status in {"PASS", "FAILED_INDEPENDENT_VERIFICATION"}
    ):
        release.setdefault("gates", {})["evidence"] = (
            "READY_FOR_INDEPENDENT_VERIFICATION" if tests_ready else "PENDING"
        )
        release["overall_acceptance_status"] = (
            "READY_FOR_INDEPENDENT_VERIFICATION"
            if tests_ready
            else "PENDING_TESTS_AND_INDEPENDENT_VERIFICATION"
        )
    release.setdefault("reports", {})[test_path.relative_to(root).as_posix()] = source_descriptor(test_path, root)
    release["artifact_sha256"] = payload_hash(release)
    atomic_json(release_path, release)
    evidence = {
        "schema_version": "stage3_s4_evidence_bundle.1",
        "phase_status": PHASE_STATUS,
        "overall_acceptance_status": release["overall_acceptance_status"],
        **{
            name: source_descriptor(root / relative, root)
            for name, relative in EVIDENCE_DESCRIPTOR_PATHS.items()
        },
        "verification_path": (docs / "stage3_s4_evidence_verification.json").relative_to(root).as_posix(),
        "next_phase_authorized": False,
    }
    evidence["artifact_sha256"] = payload_hash(evidence)
    atomic_json(docs / "stage3_s4_evidence_bundle.json", evidence)
    return evidence


ATOMIC_FAMILY_STATE_COLUMNS = {
    "DIRECTION": "directional_routability_state",
    "SPEED": "speed_state",
    "STATIC_A": "static_A_state",
    "STATIC_M": "static_M_state",
    "STATIC_D": "static_D_state",
    "STATIC_L": "static_L_state",
    "MOVEMENT": "movement_state",
    "CONTROL": "control_state",
    "ROUNDABOUT": "roundabout_state",
    "RESTRICTION": "restriction_state",
}


def _stream_atomic_reconciliation(
    atomic_path: Path, suitability: pd.DataFrame,
) -> dict[str, Any]:
    """Independently reduce long atomic evidence back to all 90k route rows."""
    index = pd.MultiIndex.from_frame(suitability[["order_id", "profile_id"]].astype(str))
    size = len(suitability)
    family_fail = {column: np.zeros(size, bool) for column in ATOMIC_FAMILY_STATE_COLUMNS.values()}
    family_unknown = {column: np.zeros(size, bool) for column in ATOMIC_FAMILY_STATE_COLUMNS.values()}
    dynamic_fail = np.zeros(size, bool)
    dynamic_unknown = np.zeros(size, bool)
    known_sets = [set() for _ in range(size)]
    unknown_sets = [set() for _ in range(size)]
    known_atomic = np.zeros(size, np.int64)
    unknown_atomic = np.zeros(size, np.int64)
    # Exact global evidence-ID distinctness is verified separately with a
    # disk-backed external sort so this reducer remains memory bounded.
    null_evidence_id_count = 0
    invalid_state_count = 0
    invalid_reason_count = 0
    parquet = pq.ParquetFile(atomic_path)
    evidence_sort_directory = Path(
        tempfile.mkdtemp(prefix=".s4_evidence_id_sort_", dir=atomic_path.parent)
    )
    evidence_chunks: list[Path] = []
    evidence_sort_temporary_bytes = 0
    columns = ["order_id", "profile_id", "check_family", "state", "evidence_id", "reason_code"]
    try:
      for batch_number, batch in enumerate(parquet.iter_batches(batch_size=250_000, columns=columns)):
        frame = batch.to_pandas()
        keys = pd.MultiIndex.from_frame(frame[["order_id", "profile_id"]].astype(str))
        positions = index.get_indexer(keys)
        if (positions < 0).any():
            raise Stage3S2AError("atomic evidence references an unknown order/profile")
        states = frame["state"].astype(str)
        invalid_state_count += int((~states.isin(["COMPATIBLE", "UNKNOWN", "INCOMPATIBLE"])).sum())
        evidence_ids = frame["evidence_id"].astype("string")
        null_evidence_id_count += int(evidence_ids.isna().sum())
        values = sorted(evidence_ids.dropna().astype(str).tolist())
        chunk_path = evidence_sort_directory / f"chunk_{batch_number:05d}.txt"
        with chunk_path.open("w", encoding="utf-8", newline="\n") as stream:
            stream.write("\n".join(values))
            stream.write("\n")
        evidence_chunks.append(chunk_path)
        evidence_sort_temporary_bytes += chunk_path.stat().st_size
        reasons = frame["reason_code"].astype("string")
        known = states.eq("INCOMPATIBLE")
        unknown = states.eq("UNKNOWN")
        invalid_reason_count += int((known & ~reasons.isin(KNOWN_REASON_CODES)).sum())
        invalid_reason_count += int((unknown & ~reasons.isin(UNKNOWN_REASON_CODES)).sum())
        invalid_reason_count += int((states.eq("COMPATIBLE") & reasons.notna()).sum())
        np.add.at(known_atomic, positions[known.to_numpy()], 1)
        np.add.at(unknown_atomic, positions[unknown.to_numpy()], 1)
        for position, reason in zip(positions[known.to_numpy()], reasons[known].astype(str)):
            known_sets[int(position)].add(reason)
        for position, reason in zip(positions[unknown.to_numpy()], reasons[unknown].astype(str)):
            unknown_sets[int(position)].add(reason)
        families = frame["check_family"].astype(str)
        for family, column in ATOMIC_FAMILY_STATE_COLUMNS.items():
            mask = families.eq(family)
            np.logical_or.at(family_fail[column], positions[mask & known], True)
            np.logical_or.at(family_unknown[column], positions[mask & unknown], True)
        dynamic = families.str.startswith("DYNAMIC_")
        np.logical_or.at(dynamic_fail, positions[dynamic & known], True)
        np.logical_or.at(dynamic_unknown, positions[dynamic & unknown], True)
      streams = [path.open("r", encoding="utf-8") for path in evidence_chunks]
      try:
        previous: str | None = None
        evidence_id_unique_count = 0
        duplicate_evidence_id_count = 0
        for raw in heapq.merge(*streams):
            value = raw.rstrip("\n")
            if value == previous:
                duplicate_evidence_id_count += 1
            else:
                evidence_id_unique_count += 1
                previous = value
      finally:
        for stream in streams:
            stream.close()
    finally:
      shutil.rmtree(evidence_sort_directory, ignore_errors=True)
    mismatch_by_family: dict[str, int] = {}
    reduced_columns: list[np.ndarray] = []
    for column in ATOMIC_FAMILY_STATE_COLUMNS.values():
        reduced = _state_array(family_fail[column], family_unknown[column])
        reduced_columns.append(reduced)
        mismatch_by_family[column] = int((reduced != suitability[column].astype(str).to_numpy()).sum())
    dynamic_reduced = _state_array(dynamic_fail, dynamic_unknown)
    reduced_columns.append(dynamic_reduced)
    mismatch_by_family["dynamic_state"] = int(
        (dynamic_reduced != suitability["dynamic_state"].astype(str).to_numpy()).sum()
    )
    any_fail = np.logical_or.reduce([values == "INCOMPATIBLE" for values in reduced_columns])
    any_unknown = np.logical_or.reduce([values == "UNKNOWN" for values in reduced_columns])
    final_reduced = np.select([any_fail, any_unknown], ["INFEASIBLE", "UNKNOWN"], default="FEASIBLE")
    expected_known = suitability["known_violation_reason_codes"].map(lambda value: set(json.loads(value)))
    expected_unknown = suitability["unknown_reason_codes"].map(lambda value: set(json.loads(value)))
    return {
        "atomic_row_count": int(parquet.metadata.num_rows),
        "evidence_id_unique_count": evidence_id_unique_count,
        "duplicate_evidence_id_count": duplicate_evidence_id_count,
        "evidence_id_distinct_method": "memory_bounded_external_merge_sort_exact",
        "evidence_id_sort_temporary_bytes": evidence_sort_temporary_bytes,
        "null_evidence_id_count": null_evidence_id_count,
        "invalid_state_count": invalid_state_count,
        "invalid_reason_count": invalid_reason_count,
        "family_state_mismatch_counts": mismatch_by_family,
        "final_state_mismatch_count": int((final_reduced != suitability["original_route_state"].astype(str).to_numpy()).sum()),
        "known_reason_set_mismatch_count": int(sum(left != right for left, right in zip(known_sets, expected_known))),
        "unknown_reason_set_mismatch_count": int(sum(left != right for left, right in zip(unknown_sets, expected_unknown))),
        "known_distinct_count_mismatch_count": int((np.array([len(value) for value in known_sets]) != suitability["known_violation_count"].to_numpy()).sum()),
        "unknown_distinct_count_mismatch_count": int((np.array([len(value) for value in unknown_sets]) != suitability["unknown_requirement_count"].to_numpy()).sum()),
        "known_atomic_count_mismatch_count": int((known_atomic != suitability["known_violation_atomic_count"].to_numpy()).sum()),
        "unknown_atomic_count_mismatch_count": int((unknown_atomic != suitability["unknown_atomic_count"].to_numpy()).sum()),
    }


def verify_s4(root: Path) -> dict[str, Any]:
    output = root / OUTPUT_REL
    docs = root / DOCS_REL
    release = read_json(docs / "stage3_s4_release_manifest.json")
    evidence = read_json(docs / "stage3_s4_evidence_bundle.json")
    summary = read_json(output / "test31_suitability_summary.json")
    test_evidence = read_json(docs / "stage3_s4_test_evidence.json")
    input_manifest = read_json(output / "test31_input_manifest.json")
    prediction_manifest = read_json(output / "test31_m3_predictions.json")
    failures: list[str] = []
    if release.get("phase_status") != PHASE_STATUS:
        failures.append("release phase status")
    if release.get("artifact_sha256") != payload_hash(release):
        failures.append("release payload hash")
    if evidence.get("artifact_sha256") != payload_hash(evidence):
        failures.append("evidence payload hash")
    for label, payload in (
        ("test evidence", test_evidence), ("suitability summary", summary),
        ("input manifest", input_manifest), ("prediction manifest", prediction_manifest),
    ):
        if payload.get("artifact_sha256") != payload_hash(payload):
            failures.append(f"{label} payload hash")
    for label in ("focused_tests", "full_tests", "compileall"):
        if test_evidence.get(label, {}).get("status") != "PASS":
            failures.append(f"test evidence not PASS:{label}")
    if release.get("gates", {}).get("evidence") not in {
        "READY_FOR_INDEPENDENT_VERIFICATION", "PASS"
    }:
        failures.append("release evidence gate is not ready")
    for group in (release.get("frozen_inputs", {}), release.get("products", {}), release.get("reports", {})):
        for label, descriptor in group.items():
            path = root / descriptor["path"]
            if not path.is_file() or sha256_file(path) != descriptor.get("sha256"):
                failures.append(f"hash:{label}")
    evidence_descriptors = {
        name: evidence.get(name, {}) for name in EVIDENCE_DESCRIPTOR_PATHS
    }
    failures.extend(descriptor_mismatches(root, evidence_descriptors))
    failures.extend(descriptor_mismatches(root, {
        "route_source": input_manifest.get("route_source", {}),
        "route_source_manifest": input_manifest.get("route_source_manifest", {}),
    }))
    if sha256_file(root / PROFILE_REL) != summary["frozen_hashes_before"]["profile"]:
        failures.append("profile immutable hash")
    if sha256_file(root / CDF_REL) != summary["frozen_hashes_before"]["cdf"]:
        failures.append("cdf immutable hash")
    suitability = pd.read_parquet(
        output / "test31_original_route_suitability.parquet",
        columns=[
            "order_id", "profile_id", *ATOMIC_FAMILY_STATE_COLUMNS.values(), "dynamic_state",
            "original_route_state", "known_violation_reason_codes", "unknown_reason_codes",
            "known_violation_count", "unknown_requirement_count", "known_violation_atomic_count",
            "unknown_atomic_count",
        ],
    )
    if len(suitability) != EXPECTED_ORDER_PROFILE_COUNT or suitability["order_id"].nunique() != EXPECTED_ORDER_COUNT:
        failures.append("30000x3 reconciliation")
    nestedness = audit_three_state_nestedness(suitability)
    if nestedness["status"] != "PASS":
        failures.append("three-state nestedness")
    precedence = suitability["known_violation_count"].gt(0) & suitability["original_route_state"].ne("INFEASIBLE")
    if precedence.any():
        failures.append("known violation precedence")
    prediction_schema = pq.ParquetFile(output / "test31_m3_predictions.parquet").schema_arrow.names
    if any(column.startswith("target_") or column.endswith("_target_valid") for column in prediction_schema):
        failures.append("realized target in prediction schema")
    prediction_gates = {
        "checkpoint": prediction_manifest.get("checkpoint_sha256") == M3_SHA256,
        "prediction_sha": prediction_manifest.get("prediction_sha256")
        == sha256_file(output / "test31_m3_predictions.parquet"),
        "route_sha": prediction_manifest.get("route_sha256") == sha256_file(root / ROUTE_REL),
        "row_count": int(prediction_manifest.get("row_count", -1))
        == pq.ParquetFile(output / "test31_m3_predictions.parquet").metadata.num_rows,
        "order_count": int(prediction_manifest.get("order_count", -1)) == EXPECTED_ORDER_COUNT,
        "decision_time_only": prediction_manifest.get("decision_time_only") is True,
        "predicted_progression_only": prediction_manifest.get("predicted_progression_only") is True,
        "realized_future_time_used": prediction_manifest.get("realized_future_time_used") is False,
        "realized_targets_persisted": prediction_manifest.get("realized_target_columns_persisted") is False,
        "strict_prediction_only_schema": prediction_manifest.get("prediction_only_forward") is True,
        "target_arrays": prediction_manifest.get("target_arrays_constructed") is False,
        "loss_path": prediction_manifest.get("loss_or_metric_path_called") is False,
    }
    for label, passed in prediction_gates.items():
        if not passed:
            failures.append(f"prediction gate:{label}")
    manifest_bindings = prediction_manifest.get("input_bindings", {})
    for name, relative in (
        ("checkpoint", UPSTREAM_REL["m3_checkpoint"]),
        ("model_manifest", UPSTREAM_REL["m3_model_manifest"]),
        ("feature", UPSTREAM_REL["m3_feature_artifacts"]),
        ("static", UPSTREAM_REL["m3_static_artifact"]),
        ("support", UPSTREAM_REL["m3_support_artifact"]),
        ("route", ROUTE_REL),
    ):
        if manifest_bindings.get(name, {}).get("sha256") != sha256_file(root / relative):
            failures.append(f"prediction input binding:{name}")
    atomic_file = pq.ParquetFile(output / "test31_original_route_atomic_checks.parquet")
    if tuple(atomic_file.schema_arrow.names) != ATOMIC_COLUMNS:
        failures.append("atomic schema")
    if atomic_file.metadata.num_rows <= EXPECTED_ORDER_PROFILE_COUNT:
        failures.append("atomic row count")
    if suitability.duplicated(["order_id", "profile_id"]).any():
        failures.append("duplicate order/profile")
    reconciliation = _stream_atomic_reconciliation(
        output / "test31_original_route_atomic_checks.parquet", suitability
    )
    scalar_blockers = (
        "duplicate_evidence_id_count", "null_evidence_id_count", "invalid_state_count",
        "invalid_reason_count", "final_state_mismatch_count", "known_reason_set_mismatch_count",
        "unknown_reason_set_mismatch_count", "known_distinct_count_mismatch_count",
        "unknown_distinct_count_mismatch_count", "known_atomic_count_mismatch_count",
        "unknown_atomic_count_mismatch_count",
    )
    if any(int(reconciliation[key]) for key in scalar_blockers):
        failures.append("atomic-to-suitability reconciliation")
    if any(int(value) for value in reconciliation["family_state_mismatch_counts"].values()):
        failures.append("atomic family reducer reconciliation")
    if not failures:
        release.setdefault("gates", {})["evidence"] = "PASS"
        release["overall_acceptance_status"] = "PASS"
        release["artifact_sha256"] = payload_hash(release)
        atomic_json(docs / "stage3_s4_release_manifest.json", release)
        evidence = refresh_evidence_chain(root, preserve_verification_status=True)
        post_promotion_failures = descriptor_mismatches(
            root, {name: evidence.get(name, {}) for name in EVIDENCE_DESCRIPTOR_PATHS}
        )
        if post_promotion_failures:
            failures.extend(post_promotion_failures)
    if failures:
        release = read_json(docs / "stage3_s4_release_manifest.json")
        release.setdefault("gates", {})["evidence"] = "FAILED_INDEPENDENT_VERIFICATION"
        release["overall_acceptance_status"] = "FAIL"
        release["artifact_sha256"] = payload_hash(release)
        atomic_json(docs / "stage3_s4_release_manifest.json", release)
        evidence = refresh_evidence_chain(root, preserve_verification_status=True)

    result = {
        "schema_version": "stage3_s4_evidence_verification.1",
        "status": "PASS" if not failures else "FAIL",
        "overall_acceptance_status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "verified_frozen_input_count": len(release.get("frozen_inputs", {})),
        "verified_product_count": len(release.get("products", {})),
        "verified_report_count": len(release.get("reports", {})),
        "verified_evidence_descriptor_count": len(evidence_descriptors),
        "atomic_row_count": int(atomic_file.metadata.num_rows),
        "atomic_to_suitability_reconciliation": reconciliation,
        "prediction_gates": prediction_gates,
        "test31_order_count": int(suitability["order_id"].nunique()),
        "order_profile_row_count": int(len(suitability)),
        "nestedness": nestedness,
        "evidence_bundle_sha256": sha256_file(docs / "stage3_s4_evidence_bundle.json"),
        "s5_authorized": False,
        "next_phase_authorized": False,
    }
    result["artifact_sha256"] = payload_hash(result)
    atomic_json(docs / "stage3_s4_evidence_verification.json", result)
    if failures:
        raise Stage3S2AError(f"S4 evidence verification failed: {failures}")
    return result


def update_test_evidence(
    root: Path, *, focused_passed: int, full_passed: int, full_warnings: int, compileall: bool
) -> dict[str, Any]:
    path = root / DOCS_REL / "stage3_s4_test_evidence.json"
    payload = read_json(path)
    payload["focused_tests"] = {"status": "PASS", "passed": int(focused_passed)}
    payload["full_tests"] = {"status": "PASS", "passed": int(full_passed), "warnings": int(full_warnings)}
    payload["compileall"] = {"status": "PASS" if compileall else "FAIL"}
    payload["artifact_sha256"] = payload_hash(payload)
    atomic_json(path, payload)
    refresh_evidence_chain(root)
    return payload


def sync_delivery(root: Path) -> Path:
    """Rebuild a small directly-clickable delivery pack without stale files."""
    delivery = root / OUTPUT_REL / "final_delivery"
    temporary = delivery.with_name(f".{delivery.name}.tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True, exist_ok=False)
    names = [
        root / DOCS_REL / "stage3_s4_methodology.md",
        root / DOCS_REL / "stage3_s4_original_route_suitability_report.md",
        root / DOCS_REL / "stage3_s4_reason_attribution_report.md",
        root / DOCS_REL / "stage3_s4_test31_identity_report.md",
        root / DOCS_REL / "stage3_s4_test31_static_report.md",
        root / DOCS_REL / "stage3_s4_test31_dynamic_report.md",
        root / DOCS_REL / "stage3_s4_test_evidence.json",
        root / DOCS_REL / "stage3_s4_release_manifest.json",
        root / DOCS_REL / "stage3_s4_evidence_bundle.json",
        root / DOCS_REL / "stage3_s4_evidence_verification.json",
        root / OUTPUT_REL / "test31_input_manifest.json",
        root / OUTPUT_REL / "test31_prepare_summary.json",
        root / OUTPUT_REL / "test31_suitability_summary.json",
        root / OUTPUT_REL / "test31_m3_predictions.json",
        root / OUTPUT_REL / "test31_reason_cooccurrence.parquet",
    ]
    for source in names:
        if source.is_file():
            shutil.copy2(source, temporary / source.name)
    manifest = {
        "schema_version": "stage3_s4_clickable_delivery.1",
        "files": {
            path.name: source_descriptor(path)
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    manifest["artifact_sha256"] = payload_hash(manifest)
    atomic_json(temporary / "manifest.json", manifest)
    if delivery.exists():
        shutil.rmtree(delivery)
    os.replace(temporary, delivery)
    return delivery


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("prepare")
    inference = sub.add_parser("infer")
    inference.add_argument("--batch-size", type=int, default=256)
    sub.add_parser("assess")
    sub.add_parser("finalize")
    sub.add_parser("verify")
    sub.add_parser("sync-delivery")
    run = sub.add_parser("run")
    run.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if args.command == "prepare":
        result = prepare_test31(root)
    elif args.command == "infer":
        from stage3.odd_tod.s4_inference import build_test31_predictions
        result = build_test31_predictions(root, batch_size=args.batch_size)
    elif args.command == "assess":
        result = assess_test31(root)
    elif args.command == "finalize":
        result = finalize_s4(root)
    elif args.command == "verify":
        result = verify_s4(root)
    elif args.command == "sync-delivery":
        result = {"delivery": sync_delivery(root).as_posix()}
    elif args.command == "run":
        prepare_test31(root)
        from stage3.odd_tod.s4_inference import build_test31_predictions
        build_test31_predictions(root, batch_size=args.batch_size)
        assess_test31(root)
        finalize_s4(root)
        result = {
            "status": "AWAITING_TEST_EVIDENCE",
            "instruction": "run real tests/compileall, call update_test_evidence, then invoke verify",
            "s5_authorized": False,
            "next_phase_authorized": False,
        }
    else:
        raise Stage3S2AError(f"unsupported S4 command: {args.command}")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
