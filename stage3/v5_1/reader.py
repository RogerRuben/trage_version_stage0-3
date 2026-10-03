"""Fail-closed reader for eligible Stage 2 v5.1 route quantiles."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


class Stage3V51ContractError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class FormalScenarioReader:
    def __init__(self, product_directory: str | Path, *, prediction_source: str = "deep_scenario"):
        self.root = Path(product_directory)
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise Stage3V51ContractError("formal product manifest is missing")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if self.manifest.get("eligible_for_stage3") is not True:
            raise Stage3V51ContractError("product is not eligible for the Stage 3 prototype")
        if self.manifest.get("stability_status") != "PASS":
            raise Stage3V51ContractError("product stability status is not PASS")
        if self.manifest.get("stability_check_status") != "PASS":
            raise Stage3V51ContractError("product stability check status is not PASS")
        declared_source = self.manifest.get("prediction_source", "deep_scenario")
        if prediction_source != declared_source:
            raise Stage3V51ContractError(f"requested prediction source {prediction_source!r} is unavailable")
        for name, expected in self.manifest.get("files", {}).items():
            path = self.root / name
            if not path.is_file() or _sha256(path) != expected:
                raise Stage3V51ContractError(f"formal product hash mismatch: {name}")
        seed = self.manifest.get("scenario_seed")
        if not isinstance(seed, int) or seed < 0:
            raise Stage3V51ContractError("scenario seed is missing or invalid")

    def read_fields(self, fields: Iterable[str]) -> pd.DataFrame:
        requested = list(dict.fromkeys(["order_id", *fields]))
        eligibility = self.manifest.get("field_eligibility", {})
        blocked = [field for field in requested if field != "order_id" and eligibility.get(field) != "ELIGIBLE"]
        if blocked:
            raise Stage3V51ContractError(f"fields are not ELIGIBLE: {blocked}")
        path = self.root / "route_service_predictions.parquet"
        try:
            frame = pd.read_parquet(path, columns=requested)
        except Exception as exc:
            raise Stage3V51ContractError(f"formal product schema mismatch: {exc}") from exc
        return frame

    def rank_candidates(self, *, quantile: str = "p50") -> pd.DataFrame:
        if quantile not in {"p50", "p90", "p95"}:
            raise Stage3V51ContractError("only eligible P50/P90/P95 ranking is allowed")
        column = f"route_service_time_{quantile}_s"
        frame = self.read_fields([column])
        return frame.sort_values(column, kind="stable", ignore_index=True)

    def compare_external_threshold(self, threshold_s: float, *, provenance: str, quantile: str = "p90") -> pd.DataFrame:
        if not provenance or provenance.strip().lower() in {"truth", "actual_route_time", "label"}:
            raise Stage3V51ContractError("an external, non-label timeout threshold provenance is required")
        if not float(threshold_s) > 0:
            raise Stage3V51ContractError("external timeout threshold must be positive")
        column = f"route_service_time_{quantile}_s"
        frame = self.read_fields([column])
        frame["external_timeout_threshold_s"] = float(threshold_s)
        frame["external_timeout_threshold_provenance"] = provenance
        frame[f"{quantile}_exceeds_external_threshold"] = frame[column] > float(threshold_s)
        return frame


class ScenarioSourceRegistry:
    """Explicit prediction-source switch; unavailable sources fail closed."""

    ALLOWED = {"tree", "deep_p50", "deep_scenario"}

    def __init__(self, products: dict[str, str | Path]):
        unknown = set(products) - self.ALLOWED
        if unknown:
            raise Stage3V51ContractError(f"unsupported prediction sources: {sorted(unknown)}")
        self.products = {name: Path(path) for name, path in products.items()}

    def open(self, prediction_source: str) -> FormalScenarioReader:
        if prediction_source == "oracle":
            raise Stage3V51ContractError("oracle is a diagnostic upper bound, not a deployable source")
        if prediction_source not in self.ALLOWED:
            raise Stage3V51ContractError(f"unknown prediction source: {prediction_source}")
        root = self.products.get(prediction_source)
        if root is None:
            raise Stage3V51ContractError(
                f"requested prediction source {prediction_source!r} is unavailable"
            )
        return FormalScenarioReader(root, prediction_source=prediction_source)
