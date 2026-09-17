from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from stage2.v4.config import Stage2V4Config, load_config
from stage2.v4.contracts import Stage2V4ContractError, require_columns
from stage2.v4.release import validate_release_manifest_payload


REPO = Path(__file__).resolve().parents[2]


def test_stage1_release_mismatch_fails_closed() -> None:
    config = load_config(REPO / "stage2/config/stage2_v4.json")
    manifest = json.loads(
        (REPO / "stage1/docs/stage1_v3_release_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    manifest["stage1_identity"]["model_id"] = "wrong-model"
    with pytest.raises(Stage2V4ContractError, match="model ID"):
        validate_release_manifest_payload(manifest, config)


def test_schema_mismatch_fails_closed() -> None:
    with pytest.raises(Stage2V4ContractError, match="missing required columns"):
        require_columns(["order_id"], ["order_id", "route_sequence"], "route")


def test_split_overlap_or_change_fails_closed() -> None:
    original = json.loads(
        (REPO / "stage2/config/stage2_v4.json").read_text(encoding="utf-8")
    )
    changed = copy.deepcopy(original)
    changed["split"]["calibration_dates"] = ["20161026"]
    with pytest.raises(Stage2V4ContractError, match="temporal split is frozen"):
        Stage2V4Config(changed)
