"""Load and validate the frozen Stage 2 v5 configuration."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .contracts import Stage2V5ContractError


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
            "final_test_dates",
            "legacy_test_dates",
        )
    }
    flattened = [date for dates in groups.values() for date in dates]
    if len(flattened) != len(set(flattened)):
        raise Stage2V5ContractError("v5 temporal split dates overlap")
    if groups["final_test_dates"] != ("20161028", "20161029", "20161030"):
        raise Stage2V5ContractError("new final test split differs from split freeze")
    if groups["legacy_test_dates"] != ("20161031",):
        raise Stage2V5ContractError("20161031 must remain legacy-only")
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

