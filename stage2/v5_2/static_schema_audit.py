"""Bounded schema-only audit of frozen Stage 0/1/2 static complexity sources."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import Stage2V52ContractError


ROUTE_STATIC_CANDIDATES = (
    "canonical_highway", "road_class", "bridge", "tunnel", "ramp",
    "speed_limit", "lane_information",
)
MOVEMENT_STATIC_CANDIDATES = (
    "turning_type", "turn_type", "node_type", "signal", "signalized",
    "merge", "is_merge", "ramp",
)


def schema_names(paths: Iterable[str | Path]) -> set[str]:
    names: set[str] = set()
    found = False
    for value in paths:  # Bounded metadata scan; parquet rows are never read.
        path = Path(value)
        if not path.is_file():
            continue
        found = True
        names.update(pq.read_schema(path).names)
    if not found:
        raise Stage2V52ContractError("bounded schema audit found no frozen product files")
    return names


def audit_static_schema(
    *,
    route_columns: Iterable[str],
    movement_columns: Iterable[str],
) -> dict[str, Any]:
    route = set(route_columns)
    movement = set(movement_columns)
    available_route = sorted(set(ROUTE_STATIC_CANDIDATES) & route)
    available_movement = sorted(set(MOVEMENT_STATIC_CANDIDATES) & movement)
    movement_joinable = {
        "order_id", "movement_sequence", "from_edge_uid", "to_edge_uid"
    } <= movement
    physical_movement_semantics = bool(available_movement)
    return {
        "schema_version": "stage2_v5_2_static_schema_audit.1",
        "status": "PASS",
        "audit_type": "bounded_frozen_schema_only",
        "stage0_or_stage1_rerun": False,
        "route_static_fields_available": available_route,
        "movement_candidate_fields_available": available_movement,
        "movement_identity_joinable": movement_joinable,
        "movement_physical_semantics_available": physical_movement_semantics,
        "formal_static_complexity": {
            "canonical_highway": "AVAILABLE" if "canonical_highway" in route else "NA",
            "road_class": "AVAILABLE" if "road_class" in route else "NA",
            "bridge": "AVAILABLE" if "bridge" in route else "NA",
            "tunnel": "AVAILABLE" if "tunnel" in route else "NA",
            "ramp": "AVAILABLE" if "ramp" in route else "NA",
            "turn_signal_merge": "AVAILABLE" if physical_movement_semantics else "NA",
        },
        "na_policy": "NA_not_zero_and_no_reconstruction",
    }
