"""Phase B0 metadata-only audit of the frozen Stage 1/v4 inputs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import Stage2V52ContractError
from .protocols import get_protocol, protocol_role_dates


ROUTE_REQUIRED = {
    "order_id", "traversal_id", "route_sequence", "observed_directed_edge_uid",
    "canonical_highway", "decision_time", "forecast_horizon_s", "feature_age_s",
}
TRAVERSAL_REQUIRED = {
    "order_id", "traversal_id", "measurement_source", "observed_travel_time_s",
    "observed_distance_m", "allocated_distance_m",
}
LABEL_REQUIRED = {"order_id", "traversal_id", "observed_sec_per_m", "rts_measurement_available"}


def _input_split(date: str) -> str:
    day = int(date[-2:])
    return "train" if day <= 24 else "validation" if day <= 27 else "test"


def _schema(path: Path) -> set[str]:
    return set(pq.read_schema(path).names)


def audit_input_metadata(
    *, protocol_id: str, route_feature_root: str | Path,
    stage1_input_root: str | Path, stage1_output_root: str | Path,
) -> dict[str, Any]:
    """Inspect file metadata only; never read rows or launch a model."""
    get_protocol(protocol_id)
    route_root, input_root, output_root = map(Path, (route_feature_root, stage1_input_root, stage1_output_root))
    findings: list[dict[str, Any]] = []
    dates = tuple(date for values in protocol_role_dates(protocol_id).values() for date in values)
    for date in dates:
        route_path = route_root / f"day={date}.parquet"
        if not route_path.is_file():
            findings.append({"date": date, "product": "route_features", "status": "MISSING_FILE", "path": route_path.as_posix()})
        else:
            missing = sorted(ROUTE_REQUIRED - _schema(route_path))
            if missing:
                findings.append({"date": date, "product": "route_features", "status": "MISSING_FIELDS", "fields": missing})
        day_root = input_root / f"split={_input_split(date)}" / f"date={date}"
        traversal_paths = sorted(day_root.glob("bucket=*/link_traversals.parquet"))
        if not traversal_paths:
            findings.append({"date": date, "product": "link_traversals", "status": "MISSING_FILE"})
        for path in traversal_paths:
            missing = sorted(TRAVERSAL_REQUIRED - _schema(path))
            if missing:
                findings.append({"date": date, "product": "link_traversals", "status": "MISSING_FIELDS", "fields": missing, "path": path.as_posix()})
            label = output_root / path.relative_to(input_root).parent / "traversal_labels.parquet"
            if not label.is_file():
                findings.append({"date": date, "product": "traversal_labels", "status": "MISSING_FILE", "path": label.as_posix()})
            else:
                missing = sorted(LABEL_REQUIRED - _schema(label))
                if missing:
                    findings.append({"date": date, "product": "traversal_labels", "status": "MISSING_FIELDS", "fields": missing, "path": label.as_posix()})
    return {
        "schema_version": "stage2_v5_2_phase_b0_metadata_audit.1",
        "status": "PASS" if not findings else "FAIL", "protocol_id": protocol_id,
        "audit_mode": "parquet_metadata_only", "dates": list(dates), "findings": findings,
    }


def write_metadata_audit(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)
