from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stage3.v5_1.reader import (
    FormalScenarioReader,
    ScenarioSourceRegistry,
    Stage3V51ContractError,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _product(tmp_path: Path, *, stable: bool = True, source: str = "deep_scenario") -> Path:
    frame = pd.DataFrame(
        {
            "order_id": ["a", "b"],
            "route_service_time_p50_s": [100.0, 90.0],
            "route_service_time_p90_s": [140.0, 130.0],
            "route_service_time_p95_s": [160.0, 150.0],
            "route_service_time_mean_s": [120.0, 110.0],
        }
    )
    parquet = tmp_path / "route_service_predictions.parquet"
    frame.to_parquet(parquet, index=False)
    scenario = tmp_path / "route_scenario_samples.npz"
    np.savez_compressed(scenario, route_id=np.array(["a", "b"]), route_time_s=np.ones((2, 4)))
    manifest = {
        "eligible_for_stage3": stable,
        "stability_status": "PASS" if stable else "FAIL",
        "stability_check_status": "PASS" if stable else "FAIL",
        "prediction_source": source,
        "scenario_seed": 7,
        "field_eligibility": {
            "route_service_time_p50_s": "ELIGIBLE",
            "route_service_time_p90_s": "ELIGIBLE",
            "route_service_time_p95_s": "ELIGIBLE",
            "route_service_time_mean_s": "BLOCKED",
        },
        "files": {parquet.name: _sha(parquet), scenario.name: _sha(scenario)},
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return tmp_path


def test_reader_accepts_quantiles_and_external_provenance(tmp_path: Path) -> None:
    reader = FormalScenarioReader(_product(tmp_path))
    assert reader.rank_candidates(quantile="p90")["order_id"].tolist() == ["b", "a"]
    result = reader.compare_external_threshold(120.0, provenance="dispatch_sla_v1", quantile="p90")
    assert result["p90_exceeds_external_threshold"].tolist() == [True, True]


def test_reader_blocks_ineligible_field(tmp_path: Path) -> None:
    reader = FormalScenarioReader(_product(tmp_path))
    with pytest.raises(Stage3V51ContractError, match="not ELIGIBLE"):
        reader.read_fields(["route_service_time_mean_s"])


def test_reader_fails_closed_on_unstable_or_hash_mismatch(tmp_path: Path) -> None:
    unstable = tmp_path / "unstable"
    unstable.mkdir()
    with pytest.raises(Stage3V51ContractError, match="not eligible"):
        FormalScenarioReader(_product(unstable, stable=False))
    stable = tmp_path / "stable"
    stable.mkdir()
    product = _product(stable)
    (product / "route_service_predictions.parquet").write_bytes(b"corrupt")
    with pytest.raises(Stage3V51ContractError, match="hash mismatch"):
        FormalScenarioReader(product)


def test_reader_requires_explicit_stability_check_status(tmp_path: Path) -> None:
    product = _product(tmp_path)
    manifest_path = product / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.pop("stability_check_status")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(Stage3V51ContractError, match="stability check"):
        FormalScenarioReader(product)


def test_reader_rejects_label_derived_threshold(tmp_path: Path) -> None:
    reader = FormalScenarioReader(_product(tmp_path))
    with pytest.raises(Stage3V51ContractError, match="external"):
        reader.compare_external_threshold(120.0, provenance="truth")


def test_prediction_source_registry_switches_and_fails_closed(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    deep = tmp_path / "deep"
    deep.mkdir()
    registry = ScenarioSourceRegistry(
        {
            "tree": _product(tree, source="tree"),
            "deep_scenario": _product(deep, source="deep_scenario"),
        }
    )
    assert registry.open("tree").manifest["prediction_source"] == "tree"
    assert registry.open("deep_scenario").manifest["prediction_source"] == "deep_scenario"
    with pytest.raises(Stage3V51ContractError, match="unavailable"):
        registry.open("deep_p50")
    with pytest.raises(Stage3V51ContractError, match="upper bound"):
        registry.open("oracle")
