"""Validated immutable configuration for Stage 2 v4."""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .contracts import STAGE2_V4_SCHEMA_VERSION, Stage2V4ContractError
from .io import canonical_json_bytes, sha256_bytes


TRAIN_DATES = tuple(f"201610{day:02d}" for day in range(9, 25))
VALIDATION_MODEL_DATES = ("20161025", "20161026")
CALIBRATION_DATES = ("20161027",)
TEST_DATES = ("20161031",)


@dataclass(frozen=True)
class Stage2V4Config:
    data: dict[str, Any]
    digest: str = field(init=False)

    def __post_init__(self) -> None:
        copied = copy.deepcopy(self.data)
        object.__setattr__(self, "data", copied)
        object.__setattr__(
            self,
            "digest",
            sha256_bytes(canonical_json_bytes(copied)),
        )
        validate_config(self)

    def section(self, name: str) -> dict[str, Any]:
        value = self.data.get(name)
        if not isinstance(value, dict):
            raise Stage2V4ContractError(f"config section {name!r} must be a mapping")
        return copy.deepcopy(value)

    @property
    def schema_version(self) -> str:
        return str(self.data.get("schema_version", ""))


def _dates(section: dict[str, Any], key: str) -> tuple[str, ...]:
    value = section.get(key)
    if not isinstance(value, list) or not all(
        isinstance(item, str) and len(item) == 8 for item in value
    ):
        raise Stage2V4ContractError(f"split.{key} must be a list of YYYYMMDD strings")
    return tuple(value)


def validate_config(config: Stage2V4Config) -> None:
    if config.schema_version != STAGE2_V4_SCHEMA_VERSION:
        raise Stage2V4ContractError(
            f"schema_version must be {STAGE2_V4_SCHEMA_VERSION!r}"
        )
    split = config.section("split")
    expected = {
        "train_dates": TRAIN_DATES,
        "validation_model_dates": VALIDATION_MODEL_DATES,
        "calibration_dates": CALIBRATION_DATES,
        "test_dates": TEST_DATES,
    }
    actual = {key: _dates(split, key) for key in expected}
    failures = [
        f"{key}: expected {value!r}, got {actual[key]!r}"
        for key, value in expected.items()
        if actual[key] != value
    ]
    if failures:
        raise Stage2V4ContractError(
            "Stage 2 v4 temporal split is frozen; " + "; ".join(failures)
        )
    flattened = [date for dates in actual.values() for date in dates]
    if len(flattened) != len(set(flattened)):
        raise Stage2V4ContractError("Stage 2 v4 split dates overlap")

    release = config.section("stage1_release")
    required_release = {
        "release_tag",
        "release_commit",
        "release_manifest_sha256",
        "engineering_status",
        "config_sha256",
        "code_sha",
        "model_id",
        "model_schema_version",
        "output_summary_schema_version",
        "output_bucket_schema_version",
        "label_schema_version",
        "bucket_count",
        "accepted_order_count",
    }
    missing = sorted(required_release - set(release))
    if missing:
        raise Stage2V4ContractError(
            f"stage1_release is missing frozen identity fields: {missing}"
        )
    if release["engineering_status"] != "PASS":
        raise Stage2V4ContractError("Stage 1 frozen engineering status must be PASS")
    if release["bucket_count"] != 196 or release["accepted_order_count"] != 220000:
        raise Stage2V4ContractError("Stage 1 frozen scale must be 196 buckets/220000 orders")

    causality = config.section("causality")
    if causality.get("decision_time_source") != "stage0_order_departure_time":
        raise Stage2V4ContractError(
            "this release requires decision_time from Stage 0 order departure_time"
        )
    if causality.get("availability_operator") != "<":
        raise Stage2V4ContractError("history availability must be strictly < decision_time")


def load_config(path: str | Path) -> Stage2V4Config:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2V4ContractError(f"cannot read Stage 2 v4 config: {source}") from exc
    if not isinstance(payload, dict):
        raise Stage2V4ContractError("Stage 2 v4 config root must be a mapping")
    return Stage2V4Config(payload)
