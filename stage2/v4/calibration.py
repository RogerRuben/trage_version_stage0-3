"""Calibration-day tail calibration and split-conformal intervals."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .config import Stage2V4Config
from .contracts import Stage2V4ContractError
from .io import (
    atomic_write_json,
    atomic_write_parquet,
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    stage2_v4_code_identity,
)
from .metrics import binary_metrics
from .prediction_io import merge_chunk_predictions


CALIBRATION_SCHEMA_VERSION = "stage2_v4_calibration.1"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2V4ContractError(f"manifest is not an object: {path}")
    return value


def _tail_fit(
    score: np.ndarray,
    truth: np.ndarray,
) -> tuple[dict[str, Any], dict[str, Any]]:
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    truth = np.asarray(truth, dtype=int)
    if len(np.unique(truth)) < 2:
        probability = float(truth.mean())
        model = {"method": "constant", "probability": probability}
        metrics = binary_metrics(truth, np.full(len(truth), probability))
        return model, {"constant": metrics, "selected_method": "constant"}
    logit = np.log(score / (1.0 - score)).reshape(-1, 1)
    platt = LogisticRegression(random_state=0, max_iter=1000).fit(logit, truth)
    platt_probability = platt.predict_proba(logit)[:, 1]
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(score, truth)
    isotonic_probability = isotonic.predict(score)
    candidates = {
        "platt": (platt, binary_metrics(truth, platt_probability)),
        "isotonic": (isotonic, binary_metrics(truth, isotonic_probability)),
    }
    selected = min(
        candidates,
        key=lambda name: (
            candidates[name][1]["brier"],
            candidates[name][1]["ece"],
        ),
    )
    return {
        "method": selected,
        "estimator": candidates[selected][0],
    }, {
        name: details[1] for name, details in candidates.items()
    } | {"selected_method": selected}


def apply_tail_calibrator(model: dict[str, Any], score: np.ndarray) -> np.ndarray:
    score = np.clip(np.asarray(score, dtype=float), 1e-6, 1 - 1e-6)
    if model["method"] == "constant":
        return np.full(len(score), float(model["probability"]))
    if model["method"] == "platt":
        logit = np.log(score / (1.0 - score)).reshape(-1, 1)
        return model["estimator"].predict_proba(logit)[:, 1]
    if model["method"] == "isotonic":
        return model["estimator"].predict(score)
    raise Stage2V4ContractError(f"unknown calibration method: {model['method']}")


def calibrate_predictions(
    model_root: str | Path,
    calibration_date: str,
    output_root: str | Path,
    config: Stage2V4Config,
    *,
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    expected_date = config.section("split")["calibration_dates"]
    if expected_date != [calibration_date]:
        raise Stage2V4ContractError(
            f"calibration may read only {expected_date}, got {calibration_date}"
        )
    model_manifest = _read_json(Path(model_root) / "model_manifest.json")
    if (
        model_manifest.get("engineering_status") != "PASS"
        or model_manifest.get("stage2_config_sha256") != config.digest
    ):
        raise Stage2V4ContractError("deep model is not bound to the current config")
    output = Path(output_root)
    bundle_path = output / "calibration_bundle.joblib"
    manifest_path = output / "calibration_manifest.json"
    if bundle_path.exists() or manifest_path.exists():
        if resume and bundle_path.is_file() and manifest_path.is_file():
            manifest = _read_json(manifest_path)
            if (
                manifest.get("engineering_status") == "PASS"
                and manifest.get("stage2_config_sha256") == config.digest
                and manifest.get("bundle_sha256") == sha256_file(bundle_path)
            ):
                return manifest
        if not force:
            raise Stage2V4ContractError("calibration output exists; use --resume or --force")
    predictions = merge_chunk_predictions(
        model_manifest["prediction_root"],
        split="calibration",
        date=calibration_date,
    )
    calibrators: dict[str, dict[str, Any]] = {}
    calibration_metrics: dict[str, Any] = {}
    for dimension in ("lcs", "rts"):
        valid = predictions[f"{dimension}_tail_valid"].astype(bool)
        score = predictions.loc[valid, f"{dimension}_tail_score"].to_numpy(float)
        truth = predictions.loc[valid, f"{dimension}_tail_event"].to_numpy(int)
        calibrators[dimension], calibration_metrics[dimension] = _tail_fit(score, truth)
        predictions[f"{dimension}_tail_probability_calibrated"] = np.nan
        predictions.loc[
            valid,
            f"{dimension}_tail_probability_calibrated",
        ] = apply_tail_calibrator(calibrators[dimension], score)

    conformal: dict[str, float] = {}
    for dimension in ("lcs", "rts"):
        mask = predictions[f"{dimension}_target_valid"].astype(bool)
        residual = np.abs(
            predictions.loc[mask, f"{dimension}_raw"].to_numpy(float)
            - predictions.loc[mask, f"pred_{dimension}_raw"].to_numpy(float)
        )
        if not len(residual):
            raise Stage2V4ContractError(f"no calibration residuals for {dimension}")
        conformal[dimension] = float(
            np.quantile(residual, 0.9, method="higher")
        )
        prediction = predictions[f"pred_{dimension}_raw"].to_numpy(float)
        predictions[f"pred_{dimension}_lower_90"] = np.clip(
            prediction - conformal[dimension],
            0.0,
            1.0,
        )
        predictions[f"pred_{dimension}_upper_90"] = np.clip(
            prediction + conformal[dimension],
            0.0,
            1.0,
        )
    bundle = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "stage2_config_sha256": config.digest,
        "deep_model_id": model_manifest["model_id"],
        "fit_date": calibration_date,
        "tail_calibrators": calibrators,
        "conformal_absolute_residual_q90": conformal,
    }
    output.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=".calibration_bundle.tmp-",
        suffix=".joblib",
        dir=output,
    )
    os.close(descriptor)
    temporary = Path(name)
    try:
        joblib.dump(bundle, temporary, compress=3)
        os.replace(temporary, bundle_path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    atomic_write_parquet(
        predictions,
        output / f"calibration_predictions_{calibration_date}.parquet",
    )
    bundle_sha = sha256_file(bundle_path)
    calibration_model_id = sha256_bytes(
        canonical_json_bytes(
            {
                "bundle_sha256": bundle_sha,
                "deep_model_id": model_manifest["model_id"],
                "fit_date": calibration_date,
            }
        )
    )
    manifest = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": stage2_v4_code_identity(
            (
                "stage2/v4/calibration.py",
                "stage2/v4/prediction_io.py",
                "stage2/v4/metrics.py",
            )
        ),
        "deep_model_id": model_manifest["model_id"],
        "calibration_model_id": calibration_model_id,
        "fit_dates": [calibration_date],
        "test_rows_read": 0,
        "row_count": int(len(predictions)),
        "tail_metrics": calibration_metrics,
        "conformal_absolute_residual_q90": conformal,
        "bundle_sha256": bundle_sha,
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(manifest_path, manifest)
    return manifest
