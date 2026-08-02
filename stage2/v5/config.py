"""Load and validate the frozen Stage 2 v5 configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Stage2V5ContractError


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_inherited_payload(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    base_path = payload.pop("base_config", None)
    if base_path is None:
        return payload
    resolved = Path(base_path)
    if not resolved.is_absolute():
        resolved = config_path.resolve().parents[2] / resolved
    return _deep_merge(load_inherited_payload(resolved), payload)


@dataclass(frozen=True)
class Stage2V5Config:
    payload: dict[str, Any]
    path: Path
    digest: str

    def section(self, name: str) -> dict[str, Any]:
        value = self.payload.get(name)
        if not isinstance(value, dict):
            raise Stage2V5ContractError(f"missing config section: {name}")
        return value


def _validate(payload: dict[str, Any]) -> None:
    if payload.get("schema_version") != "stage2_v5.1":
        raise Stage2V5ContractError("unsupported Stage 2 v5 schema")
    release = payload.get("stage2_v4_release", {})
    if release.get("tag") != "stage2-v4-final":
        raise Stage2V5ContractError("v5 must inherit stage2-v4-final")
    if release.get("commit") != "70cb70265cbb95e5fc9981024a554de28ee2be85":
        raise Stage2V5ContractError("v4 frozen commit mismatch")
    split = payload.get("split", {})
    groups = {
        name: tuple(split.get(name, ()))
        for name in (
            "train_dates",
            "validation_model_dates",
            "calibration_dates",
            "evaluation_dates",
            "legacy_test_dates",
        )
    }
    flattened = [date for dates in groups.values() for date in dates]
    if len(flattened) != len(set(flattened)):
        raise Stage2V5ContractError("v5 temporal split dates overlap")
    expected_development = {
        "train_dates": tuple(f"201610{day:02d}" for day in range(9, 22)),
        "validation_model_dates": ("20161022", "20161023"),
        "calibration_dates": ("20161024",),
        "evaluation_dates": ("20161025", "20161026", "20161027"),
        "legacy_test_dates": (),
    }
    protocol_name = split.get("protocol_name")
    expected_folds = (
        (range(9, 19), (19, 20), (21,), (22, 23)),
        (range(9, 21), (21, 22), (23,), (24, 25)),
        (range(9, 23), (23, 24), (25,), (26, 27)),
    )
    folds = payload.get("rolling_folds", ())
    if len(folds) != len(expected_folds):
        raise Stage2V5ContractError("exactly three rolling folds are required")
    for index, (fold, expected) in enumerate(zip(folds, expected_folds), start=1):
        if fold.get("fold_id") != f"fold_{index}":
            raise Stage2V5ContractError("rolling fold identity mismatch")
        actual = tuple(
            tuple(fold.get(name, ()))
            for name in ("train_dates", "validation_model_dates", "calibration_dates", "evaluation_dates")
        )
        wanted = tuple(tuple(f"201610{day:02d}" for day in values) for values in expected)
        if actual != wanted:
            raise Stage2V5ContractError(f"rolling fold {index} differs from the frozen protocol")
        dates = [date for group in actual for date in group]
        if dates != sorted(dates) or len(dates) != len(set(dates)):
            raise Stage2V5ContractError(f"rolling fold {index} is not strictly temporal")
    legacy = payload.get("legacy_benchmark_fit", {})
    if tuple(legacy.get("benchmark_dates", ())) != ("20161031",):
        raise Stage2V5ContractError("legacy benchmark identity mismatch")
    if payload.get("selected_history_mode") != "ordinary_concatenation":
        raise Stage2V5ContractError("history mode differs from frozen development selection")
    if protocol_name == "development_temporal_evaluation":
        expected_active = expected_development
    elif protocol_name in {fold["fold_id"] for fold in folds}:
        fold = next(item for item in folds if item["fold_id"] == protocol_name)
        expected_active = {
            "train_dates": tuple(fold["train_dates"]),
            "validation_model_dates": tuple(fold["validation_model_dates"]),
            "calibration_dates": tuple(fold["calibration_dates"]),
            "evaluation_dates": tuple(fold["evaluation_dates"]),
            "legacy_test_dates": (),
        }
    elif protocol_name == "legacy_frozen_benchmark":
        expected_active = {
            "train_dates": tuple(legacy["train_dates"]),
            "validation_model_dates": tuple(legacy["validation_model_dates"]),
            "calibration_dates": tuple(legacy["calibration_dates"]),
            "evaluation_dates": (),
            "legacy_test_dates": tuple(legacy["benchmark_dates"]),
        }
    else:
        raise Stage2V5ContractError("unknown Stage 2 v5 temporal protocol")
    if groups != expected_active:
        raise Stage2V5ContractError(f"active split differs from frozen {protocol_name} protocol")
    target = payload.get("service_time_target", {})
    if target.get("primary") != "direct_observed_sec_per_m":
        raise Stage2V5ContractError("v5 primary physical target must be direct pace")
    if target.get("missing_policy") != "nan_plus_mask":
        raise Stage2V5ContractError("missing labels must remain NaN plus a mask")
    quantiles = tuple(payload.get("distribution", {}).get("quantiles", ()))
    if quantiles != (0.5, 0.9, 0.95):
        raise Stage2V5ContractError("required service-time quantiles are P50/P90/P95")
    selection = payload.get("selection", {})
    epsilon = float(selection.get("epsilon", 0.0))
    clip = float(selection.get("maximum_weight", 0.0))
    if not (0.0 < epsilon < 1.0 and clip >= 1.0):
        raise Stage2V5ContractError("invalid stabilized IPW controls")


def load_config(path: str | Path = "stage2/config/stage2_v5.json") -> Stage2V5Config:
    config_path = Path(path)
    raw = config_path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    _validate(payload)
    return Stage2V5Config(
        payload=payload,
        path=config_path.resolve(),
        digest=hashlib.sha256(raw).hexdigest(),
    )
