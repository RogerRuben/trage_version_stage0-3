"""Static contract, availability, manifest, and smoke-config preflight."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .manifest import ManifestError, load_manifest, require_canonical_input


STAGES = ["stage0", "stage1", "stage2", "stage3", "stage4"]
REQUIRED_STAGE_KEYS = {
    "stage0": {"contract", "matcher_version", "road_network_version", "input_manifest", "output_manifest"},
    "stage1": {"contract", "label_schema", "fit_dates", "normalization_version", "input_manifest", "output_manifest"},
    "stage2": {"contract", "prediction_mode", "route_source", "folds", "input_manifest", "output_manifest"},
    "stage3": {"contract", "model", "calibration", "missing_modality_policy", "input_manifest", "output_manifest"},
    "stage4": {"contract", "mode", "service_time_source", "strategy", "operation", "preassignment", "idle_movement", "replication", "demand_manifest", "supply_manifest", "output_manifest"},
}


def validate_config(config: dict[str, Any], workspace: Path) -> list[str]:
    errors: list[str] = []
    if config.get("pipeline_version") != "stage0_4_rebaseline_v2":
        errors.append("pipeline_version_must_be_stage0_4_rebaseline_v2")
    for stage in STAGES:
        section = config.get(stage)
        if not isinstance(section, dict):
            errors.append(f"missing_section:{stage}")
            continue
        missing = REQUIRED_STAGE_KEYS[stage] - set(section)
        errors.extend(f"missing_config_key:{stage}.{key}" for key in sorted(missing))
        contract = workspace / str(section.get("contract", ""))
        if not contract.is_file():
            errors.append(f"missing_contract:{stage}:{contract}")
    smoke = config.get("smoke", {})
    orders = int(smoke.get("orders_per_day", 0))
    stage4_orders = int(smoke.get("stage4_orders", 0))
    if not int(smoke.get("minimum_orders_per_day", 1000)) <= orders <= int(smoke.get("maximum_orders_per_day", 5000)):
        errors.append("smoke_orders_per_day_out_of_range")
    if not int(smoke.get("stage4_minimum_orders", 500)) <= stage4_orders <= int(smoke.get("stage4_maximum_orders", 1000)):
        errors.append("stage4_smoke_orders_out_of_range")
    stage4 = config.get("stage4", {})
    expected = {
        "mode": "counterfactual_smoke",
        "strategy": "Safe GlobalMatch-MinPickup",
        "operation": "O0",
        "preassignment": False,
        "idle_movement": "Stay",
        "replication": 1,
        "service_time_source": "predicted_distribution",
    }
    for key, value in expected.items():
        if stage4.get(key) != value:
            errors.append(f"stage4_smoke_contract_mismatch:{key}")
    return errors


def validate_field_registry(path: Path) -> list[str]:
    required = {
        "field_name", "stage_created", "event_time", "availability_time", "source",
        "version", "allowed_products", "forbidden_products", "is_realized_outcome",
        "is_test_population_statistic",
    }
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ["field_registry_empty"]
    missing = required - set(rows[0])
    errors = [f"field_registry_missing_column:{name}" for name in sorted(missing)]
    names = [row.get("field_name", "") for row in rows]
    if len(names) != len(set(names)):
        errors.append("field_registry_duplicate_field")
    for row in rows:
        if not row.get("availability_time"):
            errors.append(f"undefined_availability:{row.get('field_name', '')}")
    return errors


def validate_declared_inputs(
    config: dict[str, Any], workspace: Path, schema_path: Path, start_stage: str
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    digests: list[str] = []
    start_index = STAGES.index(start_stage)
    for stage in STAGES[start_index:]:
        section = config[stage]
        roles = ["input_manifest"]
        if stage == "stage4":
            roles = ["demand_manifest", "supply_manifest"]
        for role in roles:
            path = workspace / section[role]
            try:
                manifest = load_manifest(path, schema_path, workspace)
                require_canonical_input(manifest)
                digests.append(manifest.digest)
            except ManifestError as exc:
                errors.append(f"{stage}.{role}:{exc}")
    return errors, digests

