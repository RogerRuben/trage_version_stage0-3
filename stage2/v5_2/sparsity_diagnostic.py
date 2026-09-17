"""Read-only Stage 2 v5.2 spatiotemporal sparsity diagnostic.

The command consumes frozen Phase C predictions and Train/evaluation route
products.  It never trains a model, selects a checkpoint, or performs inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .contracts import Stage2V52ContractError, require_columns
from .feature_binding import sha256_path
from .sparsity_support import (
    EDGE_COLUMN, GROUPS, IDENTITY_COLUMNS, TARGETS, TARGET_VALID_COLUMNS,
    TIME_BIN_COLUMN, SupportCounts, cluster_bootstrap_difference,
    count_records_sha256, fit_support_counts, lookup_edge_time, positive_quantiles,
    spatial_high_temporal_sparse, support_group_record_counts, support_groups,
    validate_prediction_alignment,
)


REPORT_SCHEMA = "stage2_v5_2_spatiotemporal_sparsity_report.1"
EVIDENCE_SCHEMA = "stage2_v5_2_spatiotemporal_sparsity_evidence_bundle.1"
SUPPORT_SCHEMA = "stage2_v5_2_sparsity_support.1"
PHASE_C_STATUS = "PHASE_C_FAIL_FROZEN"
TARGET_ROUTE_COLUMNS = {
    "crawl": "crawl_time_share",
    "stop": "stop_time_share",
    "speed_cv": "speed_cv_bounded",
    "acceleration_rms": "acceleration_rms_bounded",
}
STRUCTURE_CATEGORICAL = (
    "canonical_highway", "road_class", "observed_direction",
    "upstream_road_class", "downstream_road_class",
)
STRUCTURE_BOOLEAN = (
    "bridge", "tunnel", "synthetic_reverse_edge", "osm_direction_disagreement",
)
ROUTE_COLUMNS = (
    "split", *IDENTITY_COLUMNS, "route_sequence", EDGE_COLUMN, TIME_BIN_COLUMN,
    "canonical_highway", "road_class", "observed_direction", *STRUCTURE_BOOLEAN,
    *TARGET_ROUTE_COLUMNS.values(), *TARGET_VALID_COLUMNS.values(),
)
PREDICTION_COLUMNS = (
    *IDENTITY_COLUMNS, "split",
    *(column for target in TARGETS for column in (
        f"target_{target}", f"{target}_valid", f"pred_{target}",
    )),
)


def _canonical_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2V52ContractError(f"missing sparsity diagnostic input: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise Stage2V52ContractError(f"diagnostic JSON is not an object: {path}")
    return payload


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    os.replace(temporary, path)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    _atomic_text(path, frame.to_csv(index=False, lineterminator="\n"))


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    if not path.is_file():
        raise Stage2V52ContractError(f"cannot bind missing diagnostic artifact: {path}")
    return {
        "path": path.resolve().relative_to(root.resolve()).as_posix(),
        "sha256": sha256_path(path),
        "size_bytes": int(path.stat().st_size),
    }


def _validate_descriptor(descriptor: Mapping[str, Any], root: Path) -> None:
    path = root / str(descriptor.get("path"))
    if (
        not path.is_file()
        or sha256_path(path) != descriptor.get("sha256")
        or int(path.stat().st_size) != int(descriptor.get("size_bytes", -1))
    ):
        raise Stage2V52ContractError(f"diagnostic descriptor does not resolve: {path}")


def _validate_embedded_hash(payload: Mapping[str, Any], *, name: str) -> None:
    expected = payload.get("artifact_sha256")
    body = {key: value for key, value in payload.items() if key != "artifact_sha256"}
    if not isinstance(expected, str) or expected != _canonical_hash(body):
        raise Stage2V52ContractError(f"{name} embedded hash does not resolve")


def _check_frozen_state(root: Path, config: Mapping[str, Any]) -> None:
    phase = _read_json(root / "stage2/config/stage2_v5_2.json")
    status = _read_json(root / "stage2/docs/v5_2/stage2_v5_2_status_manifest.json")
    boundaries = config.get("execution_boundaries", {})
    if (
        phase.get("phase") != "PHASE_C_COMPLETE_FROZEN"
        or phase.get("execution_authorization") != "NONE_POST_C"
        or phase.get("current_status") != PHASE_C_STATUS
        or status.get("status") != PHASE_C_STATUS
        or phase.get("phase_d_authorized") is not False
        or phase.get("m5_authorized") is not False
        or any(boundaries.get(key) is not False for key in (
            "retraining_allowed", "reinference_allowed", "tau_reselection_allowed",
            "phase_d_authorized", "transfer_v2_authorized", "stage3_authorized",
        ))
    ):
        raise Stage2V52ContractError("sparsity diagnostic requires the frozen post-Phase-C state")


def _route_path(root: Path, date: str) -> Path:
    return root / f"stage2/output_v4/route_conditioned_dataset/revealed_route_proxy/day={date}.parquet"


def _prediction_inputs(root: Path) -> dict[str, tuple[Path, Path]]:
    bases = {
        "M1": root / "stage2/output_v5_2/development/M1/evaluation.json",
        "M3": root / "stage2/output_v5_2/development/M3/evaluation",
        "M4": root / "stage2/output_v5_2/development/M4/evaluation",
    }
    return {
        model: (base / "evaluation_manifest.json", base / "unique_traversal_predictions.parquet")
        for model, base in bases.items()
    }


def audit_prediction_availability(
    root: Path, *, evaluation_dates: Sequence[str],
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    expected_dates = tuple(str(value) for value in evaluation_dates)
    expected_rows: int | None = None
    for model, (manifest_path, prediction_path) in _prediction_inputs(root).items():
        manifest = _read_json(manifest_path)
        if (
            manifest.get("status") != "PASS"
            or manifest.get("model_id") != model
            or manifest.get("role") != "evaluation"
            or tuple(str(value) for value in manifest.get("evaluation_dates", ())) != expected_dates
            or manifest.get("prediction_sha256") != sha256_path(prediction_path)
        ):
            raise Stage2V52ContractError(f"{model} prediction-level artifact is not formally bound")
        rows = int(manifest.get("unique_traversal_count", 0))
        if rows <= 0 or (expected_rows is not None and rows != expected_rows):
            raise Stage2V52ContractError("paired prediction artifacts differ in row count")
        expected_rows = rows
        models[model] = {
            "manifest": _descriptor(manifest_path, root),
            "predictions": _descriptor(prediction_path, root),
            "unique_physical_traversal_count": rows,
        }
    return {
        "status": "PASS", "reinference_required": False, "retraining_required": False,
        "evaluation_dates": list(expected_dates), "models": models,
        "paired_unique_physical_traversal_count": expected_rows,
    }


def _load_train_route(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(ROUTE_COLUMNS))
    frame["split"] = "train"
    return frame


def _load_prediction_day(path: Path, date: str) -> pd.DataFrame:
    frame = pd.read_parquet(path, columns=list(PREDICTION_COLUMNS), filters=[("date", "==", date)])
    frame["date"] = frame["date"].astype(str)
    frame["order_id"] = frame["order_id"].astype(str)
    frame["traversal_id"] = pd.to_numeric(frame["traversal_id"]).astype(np.int64)
    if len(frame) == 0 or not frame["date"].eq(date).all():
        raise Stage2V52ContractError(f"prediction artifact does not contain frozen date {date}")
    return frame


def _attach_structure_signature(frame: pd.DataFrame) -> pd.DataFrame:
    original = frame.copy()
    working = frame.sort_values(
        ["date", "order_id", "route_sequence", "traversal_id"], kind="stable",
    ).copy()
    route = working.groupby(["date", "order_id"], sort=False, observed=True, dropna=False)["road_class"]
    working["upstream_road_class"] = route.shift(1)
    working["downstream_road_class"] = route.shift(-1)
    pieces: list[pd.Series] = []
    for column in STRUCTURE_CATEGORICAL:
        pieces.append(working[column].astype("string").fillna("__UNSEEN_OR_MISSING__"))
    for column in STRUCTURE_BOOLEAN:
        values = working[column].astype("boolean")
        pieces.extend((values.fillna(False).astype(str), values.notna().astype(str)))
    signature = pieces[0]
    for values in pieces[1:]:
        signature = signature.str.cat(values, sep="\x1f")
    working["structure_signature"] = signature
    original["structure_signature"] = working["structure_signature"].reindex(original.index)
    return original


class ErrorAccumulator:
    def __init__(self) -> None:
        self._values: dict[tuple[str, str, str], list[np.ndarray]] = defaultdict(list)

    def add(self, *, dimension: str, target: str, groups: np.ndarray, errors: np.ndarray) -> None:
        for group in GROUPS:
            selected = errors[groups == group]
            if len(selected):
                self._values[(dimension, target, group)].append(selected.astype(np.float32, copy=False))

    def rows(self, *, scope: str = "aggregate", date: str = "all") -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for dimension in ("spatial", "spatiotemporal", "target_specific"):
            for target in TARGETS:
                for group in GROUPS:
                    parts = self._values.get((dimension, target, group), [])
                    values = np.concatenate(parts).astype(np.float64) if parts else np.array([], dtype=float)
                    rows.append({
                        "scope": scope, "date": date, "weighting": "traversal",
                        "support_dimension": dimension, "target": target, "support_group": group,
                        "n": int(len(values)),
                        "mae": float(values.mean()) if len(values) else None,
                        "median_absolute_error": float(np.median(values)) if len(values) else None,
                        "p90_absolute_error": float(np.quantile(values, 0.90)) if len(values) else None,
                    })
        return rows


class MeanAccumulator:
    def __init__(self) -> None:
        self._stats: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0.0])

    def add(self, key: Sequence[str], values: np.ndarray) -> None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if len(finite):
            state = self._stats[tuple(key)]
            state[0] += float(finite.sum())
            state[1] += float(len(finite))

    def mean(self, key: Sequence[str]) -> tuple[int, float | None]:
        total, count = self._stats.get(tuple(key), (0.0, 0.0))
        return int(count), float(total / count) if count else None


class PairAccumulator:
    def __init__(self) -> None:
        self._stats: dict[tuple[str, ...], list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])

    def add(
        self, key: Sequence[str], baseline_error: np.ndarray, candidate_error: np.ndarray,
    ) -> None:
        baseline = np.asarray(baseline_error, dtype=np.float64)
        candidate = np.asarray(candidate_error, dtype=np.float64)
        valid = np.isfinite(baseline) & np.isfinite(candidate)
        if valid.any():
            state = self._stats[tuple(key)]
            state[0] += float(baseline[valid].sum())
            state[1] += float(candidate[valid].sum())
            state[2] += float(valid.sum())

    def metrics(self, key: Sequence[str]) -> dict[str, Any]:
        baseline_sum, candidate_sum, count = self._stats.get(tuple(key), (0.0, 0.0, 0.0))
        if not count:
            return {
                "n": 0, "baseline_mae": None, "candidate_mae": None,
                "absolute_improvement": None, "relative_improvement": None,
            }
        baseline = baseline_sum / count
        candidate = candidate_sum / count
        return {
            "n": int(count), "baseline_mae": float(baseline), "candidate_mae": float(candidate),
            "absolute_improvement": float(baseline - candidate),
            "relative_improvement": float((baseline - candidate) / baseline) if baseline > 0 else None,
        }


def _support_artifact(
    *, kind: str, counts: pd.Series, quantiles: Mapping[str, float], support: SupportCounts,
    config: Mapping[str, Any], source_inputs: Sequence[Mapping[str, Any]], target: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": SUPPORT_SCHEMA,
        "status": "PASS",
        "support_kind": kind,
        "target": target,
        "fit_scope": "train_only",
        "fit_dates": list(config["train_dates"]),
        "fit_dates_observed": list(support.fit_dates_observed),
        "protocol_id": config["protocol_id"],
        "source_row_count": support.source_row_count,
        "unique_physical_traversal_count": support.unique_physical_traversal_count,
        "duplicate_removed_count": support.duplicate_removed_count,
        "missing_edge_count": support.missing_edge_count,
        "evaluation_rows_used": 0,
        "positive_quantiles": dict(quantiles),
        "support_group_rule": dict(config["support_groups"]),
        "count_record_count": int(len(counts)),
        "count_sum": int(counts.sum()),
        "count_minimum": int(counts.min()) if len(counts) else None,
        "count_maximum": int(counts.max()) if len(counts) else None,
        "count_records_sha256": count_records_sha256(counts),
        "support_group_record_counts": support_group_record_counts(counts, quantiles),
        "source_inputs_sha256": _canonical_hash({"inputs": list(source_inputs)}),
    }
    if kind != "spatial":
        payload["time_bin"] = dict(config["time_bin"])
    payload["artifact_sha256"] = _canonical_hash(payload)
    return payload


def _fit_train_support(
    root: Path, config: Mapping[str, Any], output_root: Path,
) -> tuple[SupportCounts, dict[str, dict[str, float]], pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    source_inputs: list[dict[str, Any]] = []
    edge_signature_parts: list[pd.DataFrame] = []

    def frames() -> Iterable[pd.DataFrame]:
        for date in config["train_dates"]:
            path = _route_path(root, str(date))
            source_inputs.append(_descriptor(path, root))
            frame = _load_train_route(path)
            if not frame["date"].astype(str).eq(str(date)).all():
                raise Stage2V52ContractError("Train route product date differs from frozen protocol")
            structured = _attach_structure_signature(frame)
            pairs = structured.loc[structured[EDGE_COLUMN].notna(), [EDGE_COLUMN, "structure_signature"]]
            edge_signature_parts.append(pairs.drop_duplicates().copy())
            yield frame

    support = fit_support_counts(frames(), expected_dates=config["train_dates"])
    quantiles: dict[str, dict[str, float]] = {
        "spatial": positive_quantiles(support.spatial),
        "spatiotemporal": positive_quantiles(support.spatiotemporal),
        **{
            f"target_specific:{target}": positive_quantiles(support.target_specific[target])
            for target in TARGETS
        },
    }
    spatial_artifact = _support_artifact(
        kind="spatial", counts=support.spatial, quantiles=quantiles["spatial"],
        support=support, config=config, source_inputs=source_inputs,
    )
    temporal_artifact = _support_artifact(
        kind="spatiotemporal", counts=support.spatiotemporal,
        quantiles=quantiles["spatiotemporal"], support=support, config=config,
        source_inputs=source_inputs,
    )
    target_artifact: dict[str, Any] = {
        "schema_version": "stage2_v5_2_target_specific_sparsity_support.1",
        "status": "PASS", "fit_scope": "train_only",
        "fit_dates": list(config["train_dates"]), "protocol_id": config["protocol_id"],
        "source_row_count": support.source_row_count,
        "unique_physical_traversal_count": support.unique_physical_traversal_count,
        "duplicate_removed_count": support.duplicate_removed_count,
        "evaluation_rows_used": 0, "time_bin": dict(config["time_bin"]),
        "targets": {
            target: _support_artifact(
                kind="target_specific_spatiotemporal", counts=support.target_specific[target],
                quantiles=quantiles[f"target_specific:{target}"], support=support,
                config=config, source_inputs=source_inputs, target=target,
            )
            for target in TARGETS
        },
        "rts_excluded": True,
        "source_inputs_sha256": _canonical_hash({"inputs": source_inputs}),
    }
    target_artifact["artifact_sha256"] = _canonical_hash(target_artifact)
    _atomic_json(output_root / "spatial_support.json", spatial_artifact)
    _atomic_json(output_root / "spatiotemporal_support.json", temporal_artifact)
    _atomic_json(output_root / "target_specific_support.json", target_artifact)
    edge_signatures = pd.concat(edge_signature_parts, ignore_index=True).drop_duplicates()
    artifacts = {
        "spatial": spatial_artifact,
        "spatiotemporal": temporal_artifact,
        "target_specific": target_artifact,
    }
    return support, quantiles, edge_signatures, source_inputs, artifacts


def _read_and_align_evaluation_day(
    root: Path, date: str, prediction_paths: Mapping[str, Path], support: SupportCounts,
    quantiles: Mapping[str, Mapping[str, float]],
) -> tuple[pd.DataFrame, dict[str, int]]:
    predictions = {
        model: _load_prediction_day(path, date) for model, path in prediction_paths.items()
    }
    validate_prediction_alignment(predictions)
    reference = predictions["M1"].copy()
    for model in ("M3", "M4"):
        for target in TARGETS:
            reference[f"pred_{model}_{target}"] = predictions[model][f"pred_{target}"].to_numpy(np.float32)
    for target in TARGETS:
        reference.rename(columns={f"pred_{target}": f"pred_M1_{target}"}, inplace=True)
    route_path = _route_path(root, date)
    route = pd.read_parquet(route_path, columns=list(ROUTE_COLUMNS))
    route["date"] = route["date"].astype(str)
    route["order_id"] = route["order_id"].astype(str)
    route["traversal_id"] = pd.to_numeric(route["traversal_id"]).astype(np.int64)
    if route.duplicated(list(IDENTITY_COLUMNS)).any():
        raise Stage2V52ContractError("evaluation route product duplicates a physical traversal")
    route = _attach_structure_signature(route)
    keep = [
        *IDENTITY_COLUMNS, EDGE_COLUMN, TIME_BIN_COLUMN, "structure_signature",
        *TARGET_ROUTE_COLUMNS.values(), *TARGET_VALID_COLUMNS.values(),
    ]
    merged = reference.merge(route.loc[:, keep], on=list(IDENTITY_COLUMNS), how="left", validate="one_to_one")
    if len(merged) != len(reference) or merged[TIME_BIN_COLUMN].isna().any():
        raise Stage2V52ContractError("prediction identity does not align to evaluation route product")
    for target in TARGETS:
        pred_valid = merged[f"{target}_valid"].to_numpy(bool)
        route_valid = merged[TARGET_VALID_COLUMNS[target]].to_numpy(bool)
        if not np.array_equal(pred_valid, route_valid):
            raise Stage2V52ContractError(f"prediction and route products disagree on {target} valid mask")
        predicted_truth = merged[f"target_{target}"].to_numpy(float)
        route_truth = merged[TARGET_ROUTE_COLUMNS[target]].to_numpy(float)
        if not np.allclose(predicted_truth[pred_valid], route_truth[pred_valid], atol=1.0e-7, rtol=0):
            raise Stage2V52ContractError(f"prediction and route products disagree on {target} truth")
    edges = merged[EDGE_COLUMN].astype("string")
    merged["spatial_support"] = edges.map(support.spatial).fillna(0).astype(np.int64)
    merged["spatiotemporal_support"] = lookup_edge_time(
        support.spatiotemporal, edges, merged[TIME_BIN_COLUMN],
    )
    merged["spatial_group"] = support_groups(merged["spatial_support"], quantiles["spatial"])
    merged["spatiotemporal_group"] = support_groups(
        merged["spatiotemporal_support"], quantiles["spatiotemporal"],
    )
    for target in TARGETS:
        support_column = f"target_specific_support_{target}"
        group_column = f"target_specific_group_{target}"
        merged[support_column] = lookup_edge_time(
            support.target_specific[target], edges, merged[TIME_BIN_COLUMN],
        )
        merged[group_column] = support_groups(
            merged[support_column], quantiles[f"target_specific:{target}"],
        )
    merged["spatial_high_temporal_sparse"] = spatial_high_temporal_sparse(
        merged["spatial_group"], merged["spatiotemporal_group"],
    )
    return merged, {
        "route_row_count": int(len(route)), "prediction_row_count": int(len(reference)),
        "joined_row_count": int(len(merged)),
    }


def _dimension_groups(frame: pd.DataFrame, target: str) -> dict[str, np.ndarray]:
    return {
        "spatial": frame["spatial_group"].astype(str).to_numpy(),
        "spatiotemporal": frame["spatiotemporal_group"].astype(str).to_numpy(),
        "target_specific": frame[f"target_specific_group_{target}"].astype(str).to_numpy(),
    }


def _add_error_metrics(
    frame: pd.DataFrame, *, date: str, aggregate: ErrorAccumulator,
) -> list[dict[str, Any]]:
    daily = ErrorAccumulator()
    for target in TARGETS:
        valid = frame[f"{target}_valid"].to_numpy(bool)
        truth = frame[f"target_{target}"].to_numpy(np.float64)
        prediction = frame[f"pred_M1_{target}"].to_numpy(np.float64)
        valid &= np.isfinite(truth) & np.isfinite(prediction)
        errors = np.abs(truth[valid] - prediction[valid])
        for dimension, groups in _dimension_groups(frame, target).items():
            selected_groups = groups[valid]
            daily.add(dimension=dimension, target=target, groups=selected_groups, errors=errors)
            aggregate.add(dimension=dimension, target=target, groups=selected_groups, errors=errors)
    return daily.rows(scope="daily", date=date)


def _add_transfer_metrics(
    frame: pd.DataFrame, *, date: str, aggregate: PairAccumulator,
) -> list[dict[str, Any]]:
    daily = PairAccumulator()
    comparisons = {
        "M3_vs_M1": ("M1", "M3"),
        "M4_vs_M1": ("M1", "M4"),
        "M4_vs_M3": ("M3", "M4"),
    }
    for target in TARGETS:
        valid = frame[f"{target}_valid"].to_numpy(bool)
        truth = frame[f"target_{target}"].to_numpy(np.float64)
        errors = {
            model: np.abs(truth - frame[f"pred_{model}_{target}"].to_numpy(np.float64))
            for model in ("M1", "M3", "M4")
        }
        valid &= np.isfinite(truth)
        valid &= np.logical_and.reduce([np.isfinite(values) for values in errors.values()])
        for dimension, groups in _dimension_groups(frame, target).items():
            for group in GROUPS:
                mask = valid & (groups == group)
                for comparison, (baseline, candidate) in comparisons.items():
                    key = (dimension, target, group, comparison)
                    daily.add(key, errors[baseline][mask], errors[candidate][mask])
                    aggregate.add(key, errors[baseline][mask], errors[candidate][mask])
    rows: list[dict[str, Any]] = []
    for dimension in ("spatial", "spatiotemporal", "target_specific"):
        for target in TARGETS:
            for group in GROUPS:
                for comparison in comparisons:
                    rows.append({
                        "scope": "daily", "date": date, "support_dimension": dimension,
                        "target": target, "support_group": group, "comparison": comparison,
                        **daily.metrics((dimension, target, group, comparison)),
                    })
    return rows


def _cell_aggregates(frame: pd.DataFrame, *, date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_parts: list[pd.DataFrame] = []
    structure_parts: list[pd.DataFrame] = []
    edge = frame[EDGE_COLUMN].astype("string").fillna("__MISSING_EDGE__")
    for target in TARGETS:
        valid = frame[f"{target}_valid"].to_numpy(bool)
        truth = frame[f"target_{target}"].to_numpy(np.float64)
        predictions = {
            model: frame[f"pred_{model}_{target}"].to_numpy(np.float64)
            for model in ("M1", "M3", "M4")
        }
        valid &= np.isfinite(truth)
        valid &= np.logical_and.reduce([np.isfinite(values) for values in predictions.values()])
        if not valid.any():
            continue
        working = pd.DataFrame({
            EDGE_COLUMN: edge[valid].astype(str).to_numpy(),
            TIME_BIN_COLUMN: frame.loc[valid, TIME_BIN_COLUMN].to_numpy(np.int16),
            "spatial_support": frame.loc[valid, "spatial_support"].to_numpy(np.int64),
            "spatiotemporal_support": frame.loc[valid, "spatiotemporal_support"].to_numpy(np.int64),
            "target_specific_support": frame.loc[valid, f"target_specific_support_{target}"].to_numpy(np.int64),
            "spatial_group": frame.loc[valid, "spatial_group"].astype(str).to_numpy(),
            "spatiotemporal_group": frame.loc[valid, "spatiotemporal_group"].astype(str).to_numpy(),
            "target_specific_group": frame.loc[valid, f"target_specific_group_{target}"].astype(str).to_numpy(),
            "mismatch": frame.loc[valid, "spatial_high_temporal_sparse"].to_numpy(bool),
            "target_value": truth[valid],
            "m1_error": np.abs(truth[valid] - predictions["M1"][valid]),
            "m3_error": np.abs(truth[valid] - predictions["M3"][valid]),
            "m4_error": np.abs(truth[valid] - predictions["M4"][valid]),
            "structure_signature": frame.loc[valid, "structure_signature"].astype(str).to_numpy(),
        })
        keys = [
            EDGE_COLUMN, TIME_BIN_COLUMN, "spatial_support", "spatiotemporal_support",
            "target_specific_support", "spatial_group", "spatiotemporal_group",
            "target_specific_group", "mismatch",
        ]
        cells = working.groupby(keys, sort=False, observed=True, dropna=False).agg(
            traversal_count=("m1_error", "size"),
            target_sum=("target_value", "sum"),
            m1_error_sum=("m1_error", "sum"),
            m3_error_sum=("m3_error", "sum"),
            m4_error_sum=("m4_error", "sum"),
        ).reset_index()
        cells.insert(0, "target", target)
        cells.insert(0, "date", date)
        cell_parts.append(cells)
        structure = working.groupby(
            ["structure_signature", EDGE_COLUMN, TIME_BIN_COLUMN],
            sort=False, observed=True, dropna=False,
        ).agg(
            traversal_count=("m1_error", "size"),
            target_sum=("target_value", "sum"),
            m1_error_sum=("m1_error", "sum"),
        ).reset_index()
        structure.insert(0, "target", target)
        structure.insert(0, "date", date)
        structure_parts.append(structure)
    return pd.concat(cell_parts, ignore_index=True), pd.concat(structure_parts, ignore_index=True)


def _distribution_rows(
    frame: pd.DataFrame, *, date: str,
) -> tuple[
    list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
    list[dict[str, Any]], list[dict[str, Any]],
]:
    spatial_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    target_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    total = len(frame)
    edge_values = frame[EDGE_COLUMN].astype("string").fillna("__MISSING_EDGE__")
    cell = pd.Series(
        list(zip(edge_values.astype(str), frame[TIME_BIN_COLUMN].astype(int))), index=frame.index,
    )
    for group in GROUPS:
        spatial_mask = frame["spatial_group"].astype(str).eq(group)
        temporal_mask = frame["spatiotemporal_group"].astype(str).eq(group)
        spatial_rows.append({
            "scope": "daily", "date": date, "support_group": group,
            "unique_edges": int(edge_values[spatial_mask].nunique()),
            "evaluation_traversals": int(spatial_mask.sum()),
            "evaluation_share": float(spatial_mask.mean()),
        })
        temporal_rows.append({
            "scope": "daily", "date": date, "support_group": group,
            "unique_edge_time_cells": int(cell[temporal_mask].nunique()),
            "evaluation_traversals": int(temporal_mask.sum()),
            "evaluation_share": float(temporal_mask.mean()),
        })
        for target in TARGETS:
            valid = frame[f"{target}_valid"].to_numpy(bool)
            mask = valid & frame[f"target_specific_group_{target}"].astype(str).eq(group).to_numpy()
            target_rows.append({
                "scope": "daily", "date": date, "target": target, "support_group": group,
                "target_valid_traversals": int(mask.sum()),
                "target_valid_share": float(mask.sum() / valid.sum()) if valid.sum() else None,
                "unique_edge_time_cells": int(cell[mask].nunique()),
            })
    for spatial_group in GROUPS:
        for temporal_group in GROUPS:
            mask = (
                frame["spatial_group"].astype(str).eq(spatial_group)
                & frame["spatiotemporal_group"].astype(str).eq(temporal_group)
            )
            row: dict[str, Any] = {
                "scope": "daily", "date": date, "spatial_group": spatial_group,
                "spatiotemporal_group": temporal_group,
                "unique_edge_time_cells": int(cell[mask].nunique()),
                "evaluation_traversals": int(mask.sum()),
                "evaluation_share": float(mask.sum() / total) if total else None,
            }
            for target in TARGETS:
                row[f"{target}_valid_count"] = int((mask & frame[f"{target}_valid"].to_numpy(bool)).sum())
            cross_rows.append(row)
    mismatch = frame["spatial_high_temporal_sparse"].to_numpy(bool)
    mismatch_rows = [{
        "scope": "daily", "date": date,
        "unique_edge_time_cells": int(cell[mismatch].nunique()),
        "evaluation_traversals": int(mismatch.sum()),
        "evaluation_share": float(mismatch.mean()),
        **{
            f"{target}_valid_count": int((mismatch & frame[f"{target}_valid"].to_numpy(bool)).sum())
            for target in TARGETS
        },
    }]
    return spatial_rows, temporal_rows, target_rows, cross_rows, mismatch_rows


class DistributionAccumulator:
    def __init__(self) -> None:
        self.total = 0
        self.spatial_counts = defaultdict(int)
        self.temporal_counts = defaultdict(int)
        self.spatial_edges: dict[str, set[str]] = defaultdict(set)
        self.temporal_cells: dict[str, set[tuple[str, int]]] = defaultdict(set)
        self.target_counts = defaultdict(int)
        self.target_cells: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
        self.cross_counts = defaultdict(int)
        self.cross_cells: dict[tuple[str, str], set[tuple[str, int]]] = defaultdict(set)
        self.cross_valid = defaultdict(int)
        self.mismatch_count = 0
        self.mismatch_cells: set[tuple[str, int]] = set()
        self.mismatch_valid = defaultdict(int)

    def add(self, frame: pd.DataFrame) -> None:
        self.total += len(frame)
        edges = frame[EDGE_COLUMN].astype("string").fillna("__MISSING_EDGE__").astype(str)
        bins = frame[TIME_BIN_COLUMN].astype(int)
        cells = list(zip(edges, bins))
        spatial = frame["spatial_group"].astype(str).to_numpy()
        temporal = frame["spatiotemporal_group"].astype(str).to_numpy()
        for group in GROUPS:
            smask = spatial == group
            tmask = temporal == group
            self.spatial_counts[group] += int(smask.sum())
            self.temporal_counts[group] += int(tmask.sum())
            self.spatial_edges[group].update(edges[smask].tolist())
            self.temporal_cells[group].update(cell for cell, keep in zip(cells, tmask) if keep)
            for target in TARGETS:
                valid = frame[f"{target}_valid"].to_numpy(bool)
                target_group = frame[f"target_specific_group_{target}"].astype(str).to_numpy()
                mask = valid & (target_group == group)
                self.target_counts[(target, group)] += int(mask.sum())
                self.target_cells[(target, group)].update(cell for cell, keep in zip(cells, mask) if keep)
        for spatial_group in GROUPS:
            for temporal_group in GROUPS:
                mask = (spatial == spatial_group) & (temporal == temporal_group)
                key = (spatial_group, temporal_group)
                self.cross_counts[key] += int(mask.sum())
                self.cross_cells[key].update(cell for cell, keep in zip(cells, mask) if keep)
                for target in TARGETS:
                    valid = frame[f"{target}_valid"].to_numpy(bool)
                    self.cross_valid[(spatial_group, temporal_group, target)] += int((mask & valid).sum())
        mismatch = frame["spatial_high_temporal_sparse"].to_numpy(bool)
        self.mismatch_count += int(mismatch.sum())
        self.mismatch_cells.update(cell for cell, keep in zip(cells, mismatch) if keep)
        for target in TARGETS:
            self.mismatch_valid[target] += int((mismatch & frame[f"{target}_valid"].to_numpy(bool)).sum())

    def rows(
        self, *, support: SupportCounts, quantiles: Mapping[str, Mapping[str, float]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        spatial_train_groups = support_groups(support.spatial.to_numpy(), quantiles["spatial"])
        temporal_train_groups = support_groups(
            support.spatiotemporal.to_numpy(), quantiles["spatiotemporal"],
        )
        spatial_rows = []
        temporal_rows = []
        target_rows = []
        cross_rows = []
        for group in GROUPS:
            spatial_rows.append({
                "scope": "aggregate", "date": "all", "support_group": group,
                "train_unique_edges": int((spatial_train_groups == group).sum()),
                "evaluation_unique_edges": int(len(self.spatial_edges[group])),
                "evaluation_traversals": int(self.spatial_counts[group]),
                "evaluation_share": float(self.spatial_counts[group] / self.total) if self.total else None,
            })
            temporal_rows.append({
                "scope": "aggregate", "date": "all", "support_group": group,
                "train_unique_edge_time_cells": int((temporal_train_groups == group).sum()),
                "evaluation_unique_edge_time_cells": int(len(self.temporal_cells[group])),
                "evaluation_traversals": int(self.temporal_counts[group]),
                "evaluation_share": float(self.temporal_counts[group] / self.total) if self.total else None,
            })
            for target in TARGETS:
                target_total = sum(self.target_counts[(target, candidate)] for candidate in GROUPS)
                target_rows.append({
                    "scope": "aggregate", "date": "all", "target": target,
                    "support_group": group,
                    "train_unique_edge_time_cells": int((support_groups(
                        support.target_specific[target].to_numpy(),
                        quantiles[f"target_specific:{target}"],
                    ) == group).sum()),
                    "evaluation_unique_edge_time_cells": int(len(self.target_cells[(target, group)])),
                    "target_valid_traversals": int(self.target_counts[(target, group)]),
                    "target_valid_share": float(self.target_counts[(target, group)] / target_total) if target_total else None,
                })
        for spatial_group in GROUPS:
            for temporal_group in GROUPS:
                key = (spatial_group, temporal_group)
                row: dict[str, Any] = {
                    "scope": "aggregate", "date": "all", "spatial_group": spatial_group,
                    "spatiotemporal_group": temporal_group,
                    "unique_edge_time_cells": int(len(self.cross_cells[key])),
                    "evaluation_traversals": int(self.cross_counts[key]),
                    "evaluation_share": float(self.cross_counts[key] / self.total) if self.total else None,
                }
                for target in TARGETS:
                    row[f"{target}_valid_count"] = int(self.cross_valid[(spatial_group, temporal_group, target)])
                cross_rows.append(row)
        mismatch = {
            "scope": "aggregate", "date": "all",
            "unique_edge_time_cells": int(len(self.mismatch_cells)),
            "evaluation_traversals": int(self.mismatch_count),
            "evaluation_share": float(self.mismatch_count / self.total) if self.total else None,
            **{f"{target}_valid_count": int(self.mismatch_valid[target]) for target in TARGETS},
        }
        return spatial_rows, temporal_rows, target_rows, cross_rows, mismatch


class MismatchAccumulator:
    def __init__(self) -> None:
        self.stats = defaultdict(lambda: [0.0, 0.0])

    def add(self, frame: pd.DataFrame) -> None:
        categories = {
            "spatial_high_temporal_sparse": frame["spatial_high_temporal_sparse"].to_numpy(bool),
            "spatial_high_temporal_high": (
                frame["spatial_group"].astype(str).eq("high")
                & frame["spatiotemporal_group"].astype(str).eq("high")
            ).to_numpy(),
        }
        for target in TARGETS:
            truth = frame[f"target_{target}"].to_numpy(np.float64)
            valid = frame[f"{target}_valid"].to_numpy(bool) & np.isfinite(truth)
            for category, category_mask in categories.items():
                for model in ("M1", "M3", "M4"):
                    prediction = frame[f"pred_{model}_{target}"].to_numpy(np.float64)
                    mask = valid & category_mask & np.isfinite(prediction)
                    error = np.abs(truth[mask] - prediction[mask])
                    if len(error):
                        state = self.stats[(category, target, model)]
                        state[0] += float(error.sum())
                        state[1] += float(len(error))

    def rows(self, *, scope: str, date: str) -> list[dict[str, Any]]:
        rows = []
        for category in ("spatial_high_temporal_sparse", "spatial_high_temporal_high"):
            for target in TARGETS:
                maes: dict[str, float | None] = {}
                counts: dict[str, int] = {}
                for model in ("M1", "M3", "M4"):
                    total, count = self.stats[(category, target, model)]
                    maes[model] = float(total / count) if count else None
                    counts[model] = int(count)
                m1, m3, m4 = maes["M1"], maes["M3"], maes["M4"]
                rows.append({
                    "scope": scope, "date": date, "group": category, "target": target,
                    "n": counts["M1"], "m1_mae": m1, "m3_mae": m3, "m4_mae": m4,
                    "m3_vs_m1_absolute_improvement": (m1 - m3) if m1 is not None and m3 is not None else None,
                    "m3_vs_m1_relative_improvement": ((m1 - m3) / m1) if m1 and m3 is not None else None,
                    "m4_vs_m1_absolute_improvement": (m1 - m4) if m1 is not None and m4 is not None else None,
                    "m4_vs_m1_relative_improvement": ((m1 - m4) / m1) if m1 and m4 is not None else None,
                    "m4_vs_m3_absolute_improvement": (m3 - m4) if m3 is not None and m4 is not None else None,
                    "m4_vs_m3_relative_improvement": ((m3 - m4) / m3) if m3 and m4 is not None else None,
                })
        return rows


def _finalize_cells(parts: Sequence[pd.DataFrame]) -> pd.DataFrame:
    combined = pd.concat(parts, ignore_index=True)
    keys = [
        "target", EDGE_COLUMN, TIME_BIN_COLUMN, "spatial_support",
        "spatiotemporal_support", "target_specific_support", "spatial_group",
        "spatiotemporal_group", "target_specific_group", "mismatch",
    ]
    aggregate = combined.groupby(keys, sort=False, observed=True, dropna=False).agg(
        traversal_count=("traversal_count", "sum"),
        target_sum=("target_sum", "sum"),
        m1_error_sum=("m1_error_sum", "sum"),
        m3_error_sum=("m3_error_sum", "sum"),
        m4_error_sum=("m4_error_sum", "sum"),
    ).reset_index()
    return aggregate


def _with_cell_means(cells: pd.DataFrame) -> pd.DataFrame:
    result = cells.copy()
    count = result["traversal_count"].to_numpy(np.float64)
    for source, output in (
        ("target_sum", "target_mean"), ("m1_error_sum", "m1_cell_mae"),
        ("m3_error_sum", "m3_cell_mae"), ("m4_error_sum", "m4_cell_mae"),
    ):
        result[output] = result[source].to_numpy(np.float64) / count
    return result


def _cell_equal_error_rows(cells: pd.DataFrame, *, scope: str, date: str) -> list[dict[str, Any]]:
    values = _with_cell_means(cells)
    rows: list[dict[str, Any]] = []
    dimensions = {
        "spatial": "spatial_group",
        "spatiotemporal": "spatiotemporal_group",
        "target_specific": "target_specific_group",
    }
    for target in TARGETS:
        target_cells = values.loc[values["target"].eq(target)]
        for dimension, group_column in dimensions.items():
            for group in GROUPS:
                selected = target_cells.loc[target_cells[group_column].astype(str).eq(group), "m1_cell_mae"].to_numpy(float)
                rows.append({
                    "scope": scope, "date": date, "weighting": "cell_equal",
                    "support_dimension": dimension, "target": target, "support_group": group,
                    "n": int(len(selected)),
                    "mae": float(selected.mean()) if len(selected) else None,
                    "median_absolute_error": float(np.median(selected)) if len(selected) else None,
                    "p90_absolute_error": float(np.quantile(selected, 0.90)) if len(selected) else None,
                })
    return rows


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 2 or np.unique(left[valid]).size < 2 or np.unique(right[valid]).size < 2:
        return None
    return float(spearmanr(left[valid], right[valid]).statistic)


def _correlation_rows(cells: pd.DataFrame, *, scope: str, date: str) -> list[dict[str, Any]]:
    values = _with_cell_means(cells)
    rows: list[dict[str, Any]] = []
    support_columns = {
        "spatial": "spatial_support",
        "spatiotemporal": "spatiotemporal_support",
        "target_specific": "target_specific_support",
    }
    for target in TARGETS:
        target_cells = values.loc[values["target"].eq(target)]
        error = target_cells["m1_cell_mae"].to_numpy(float)
        for dimension, support_column in support_columns.items():
            support_values = target_cells[support_column].to_numpy(float)
            rows.append({
                "scope": scope, "date": date, "target": target,
                "support_dimension": dimension, "cell_count": int(len(target_cells)),
                "spearman_log1p_support_vs_cell_mae": _spearman(np.log1p(support_values), error),
                "causal_interpretation": False,
            })
    return rows


def _bootstrap_rows(
    cells: pd.DataFrame, *, config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    values = _with_cell_means(cells)
    bootstrap = config["bootstrap"]
    comparisons = (
        ("spatiotemporal_low_vs_high", "spatiotemporal_group", "low", "high"),
        ("spatiotemporal_unseen_vs_high", "spatiotemporal_group", "unseen", "high"),
        ("spatial_high_temporal_sparse_vs_high_high", "mismatch_comparison", "sparse", "high_high"),
    )
    rows: list[dict[str, Any]] = []
    for target_index, target in enumerate(TARGETS):
        target_cells = values.loc[values["target"].eq(target)].copy()
        target_cells["mismatch_comparison"] = np.where(
            target_cells["mismatch"].to_numpy(bool), "sparse",
            np.where(
                target_cells["spatial_group"].astype(str).eq("high")
                & target_cells["spatiotemporal_group"].astype(str).eq("high"),
                "high_high", "other",
            ),
        )
        for comparison_index, (name, column, left_group, right_group) in enumerate(comparisons):
            left = target_cells.loc[target_cells[column].astype(str).eq(left_group), "m1_cell_mae"]
            right = target_cells.loc[target_cells[column].astype(str).eq(right_group), "m1_cell_mae"]
            seed = int(bootstrap["seed"]) + (target_index * 100) + comparison_index
            rows.append({
                "target": target, "comparison": name, "cluster_unit": "edge_time_cell",
                "resamples": int(bootstrap["resamples"]), "seed": seed,
                "confidence_level": float(bootstrap["confidence_level"]),
                **cluster_bootstrap_difference(
                    left, right, resamples=int(bootstrap["resamples"]), seed=seed,
                    confidence_level=float(bootstrap["confidence_level"]),
                ),
            })
    return rows


def _structure_diagnostic(
    edge_signatures: pd.DataFrame, structure_cells: Sequence[pd.DataFrame],
    *, config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    pairs = edge_signatures.drop_duplicates([EDGE_COLUMN, "structure_signature"])
    signature_edges = pairs.groupby("structure_signature", sort=False, observed=True)[EDGE_COLUMN].nunique()
    edge_signatures_count = pairs.groupby(EDGE_COLUMN, sort=False, observed=True)["structure_signature"].nunique()
    top = signature_edges.sort_values(ascending=False).head(20)
    top_rows = [
        {"rank": rank, "structure_signature_sha256": hashlib.sha256(str(signature).encode("utf-8")).hexdigest(), "unique_edge_count": int(count)}
        for rank, (signature, count) in enumerate(top.items(), start=1)
    ]
    minimum_n = int(config["structure_diagnostic"]["minimum_target_valid_traversals_per_cell"])
    minimum_cells = int(config["structure_diagnostic"]["minimum_cells_per_signature"])
    combined = pd.concat(structure_cells, ignore_index=True)
    aggregated = combined.groupby(
        ["target", "structure_signature", EDGE_COLUMN, TIME_BIN_COLUMN],
        sort=False, observed=True, dropna=False,
    ).agg(
        traversal_count=("traversal_count", "sum"),
        target_sum=("target_sum", "sum"),
        m1_error_sum=("m1_error_sum", "sum"),
    ).reset_index()
    aggregated = aggregated.loc[aggregated["traversal_count"].ge(minimum_n)].copy()
    aggregated["target_mean"] = aggregated["target_sum"] / aggregated["traversal_count"]
    aggregated["m1_cell_mae"] = aggregated["m1_error_sum"] / aggregated["traversal_count"]
    heterogeneity_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        target_cells = aggregated.loc[aggregated["target"].eq(target)]
        by_signature = target_cells.groupby("structure_signature", sort=False, observed=True).agg(
            cell_count=("m1_cell_mae", "size"),
            unique_edge_count=(EDGE_COLUMN, "nunique"),
            target_mean_std=("target_mean", "std"),
            target_mean_range=("target_mean", lambda values: float(values.max() - values.min())),
            m1_mae_std=("m1_cell_mae", "std"),
            m1_mae_range=("m1_cell_mae", lambda values: float(values.max() - values.min())),
        ).reset_index()
        eligible = by_signature.loc[by_signature["cell_count"].ge(minimum_cells)]
        heterogeneity_rows.append({
            "target": target, "minimum_valid_traversals_per_cell": minimum_n,
            "minimum_cells_per_signature": minimum_cells,
            "eligible_signature_count": int(len(eligible)),
            "eligible_cell_count": int(eligible["cell_count"].sum()),
            "median_within_signature_target_mean_std": float(eligible["target_mean_std"].median()) if len(eligible) else None,
            "median_within_signature_target_mean_range": float(eligible["target_mean_range"].median()) if len(eligible) else None,
            "median_within_signature_m1_mae_std": float(eligible["m1_mae_std"].median()) if len(eligible) else None,
            "median_within_signature_m1_mae_range": float(eligible["m1_mae_range"].median()) if len(eligible) else None,
        })
    summary = {
        "unique_edge_count": int(pairs[EDGE_COLUMN].nunique()),
        "unique_structure_signature_count": int(pairs["structure_signature"].nunique()),
        "edge_signature_pair_count": int(len(pairs)),
        "edge_to_signature_ratio": float(pairs[EDGE_COLUMN].nunique() / pairs["structure_signature"].nunique()),
        "mean_edges_per_signature": float(signature_edges.mean()),
        "median_edges_per_signature": float(signature_edges.median()),
        "maximum_edges_per_signature": int(signature_edges.max()),
        "collision_signature_share": float(signature_edges.gt(1).mean()),
        "mean_signatures_per_edge": float(edge_signatures_count.mean()),
        "median_signatures_per_edge": float(edge_signatures_count.median()),
    }
    return summary, top_rows, heterogeneity_rows


def _aggregate_transfer_rows(aggregate: PairAccumulator) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dimension in ("spatial", "spatiotemporal", "target_specific"):
        for target in TARGETS:
            for group in GROUPS:
                for comparison in ("M3_vs_M1", "M4_vs_M1", "M4_vs_M3"):
                    rows.append({
                        "scope": "aggregate", "date": "all", "support_dimension": dimension,
                        "target": target, "support_group": group, "comparison": comparison,
                        **aggregate.metrics((dimension, target, group, comparison)),
                    })
    return rows


def classify_diagnostic(
    *, correlations: pd.DataFrame, bootstrap: pd.DataFrame,
    mismatch_summary: Mapping[str, Any], transfer_metrics: pd.DataFrame,
) -> dict[str, Any]:
    """Apply the predeclared direction/consistency rubric without tuning an effect threshold."""
    sparsity_targets: list[str] = []
    misalignment_targets: list[str] = []
    target_specific_targets: list[str] = []
    structured_positive_targets: list[str] = []
    m4_increment_targets: list[str] = []
    for target in TARGETS:
        aggregate_corr = correlations.loc[
            correlations["scope"].eq("aggregate") & correlations["target"].eq(target)
        ].set_index("support_dimension")["spearman_log1p_support_vs_cell_mae"]
        daily_temporal = correlations.loc[
            correlations["scope"].eq("daily")
            & correlations["target"].eq(target)
            & correlations["support_dimension"].eq("spatiotemporal"),
            "spearman_log1p_support_vs_cell_mae",
        ].dropna()
        low_effect = bootstrap.loc[
            bootstrap["target"].eq(target)
            & bootstrap["comparison"].eq("spatiotemporal_low_vs_high"), "effect",
        ]
        temporal_corr = aggregate_corr.get("spatiotemporal")
        spatial_corr = aggregate_corr.get("spatial")
        target_corr = aggregate_corr.get("target_specific")
        if (
            pd.notna(temporal_corr) and float(temporal_corr) < 0
            and int((daily_temporal < 0).sum()) >= 2
            and len(low_effect) and pd.notna(low_effect.iloc[0]) and float(low_effect.iloc[0]) > 0
        ):
            sparsity_targets.append(target)
        mismatch_effect = bootstrap.loc[
            bootstrap["target"].eq(target)
            & bootstrap["comparison"].eq("spatial_high_temporal_sparse_vs_high_high"), "effect",
        ]
        if (
            len(mismatch_effect) and pd.notna(mismatch_effect.iloc[0]) and float(mismatch_effect.iloc[0]) > 0
            and pd.notna(temporal_corr) and pd.notna(spatial_corr)
            and abs(float(temporal_corr)) > abs(float(spatial_corr))
        ):
            misalignment_targets.append(target)
        if pd.notna(target_corr) and pd.notna(temporal_corr) and abs(float(target_corr)) > abs(float(temporal_corr)):
            target_specific_targets.append(target)
        aggregate_sparse_transfer = transfer_metrics.loc[
            transfer_metrics["scope"].eq("aggregate")
            & transfer_metrics["support_dimension"].eq("spatiotemporal")
            & transfer_metrics["target"].eq(target)
            & transfer_metrics["support_group"].isin(("unseen", "low"))
        ]
        daily_low_transfer = transfer_metrics.loc[
            transfer_metrics["scope"].eq("daily")
            & transfer_metrics["support_dimension"].eq("spatiotemporal")
            & transfer_metrics["target"].eq(target)
            & transfer_metrics["support_group"].eq("low")
        ]
        for comparison, destination in (
            ("M3_vs_M1", structured_positive_targets),
            ("M4_vs_M3", m4_increment_targets),
        ):
            aggregate_values = aggregate_sparse_transfer.loc[
                aggregate_sparse_transfer["comparison"].eq(comparison), "absolute_improvement",
            ].dropna()
            daily_values = daily_low_transfer.loc[
                daily_low_transfer["comparison"].eq(comparison), "absolute_improvement",
            ].dropna()
            if (
                len(aggregate_values) == 2
                and aggregate_values.gt(0).all()
                and int(daily_values.gt(0).sum()) >= 2
            ):
                destination.append(target)
    sparsity_real = len(sparsity_targets) >= 3
    misalignment_strong = (
        int(mismatch_summary.get("evaluation_traversals", 0)) > 0
        and (len(misalignment_targets) >= 3 or len(target_specific_targets) >= 3)
    )
    structured_positive = len(structured_positive_targets) >= 3
    gate_increment_supported = len(m4_increment_targets) >= 3
    if misalignment_strong and structured_positive:
        classification = "DIAG-A"
        conclusion = "CURRENT_EDGE_SUPPORT_MISALIGNED_WITH_SPATIOTEMPORAL_SPARSITY"
    elif structured_positive and not gate_increment_supported:
        classification = "DIAG-B"
        conclusion = "STRUCTURED_TRANSFER_POSITIVE; SUPPORT_AWARE_GATE_NOT_SUPPORTED"
    elif sparsity_real:
        classification = "DIAG-C"
        conclusion = "SPARSITY_IS_REAL_BUT_CURRENT_STRUCTURE_FEATURES_ARE_INSUFFICIENT"
    else:
        classification = "DIAG-D"
        conclusion = "SPARSITY_NOT_A_PRIMARY_ERROR_DRIVER"
    return {
        "classification": classification, "conclusion": conclusion,
        "rubric": {
            "uses_fixed_effect_threshold": False,
            "sparsity_requires_negative_aggregate_trend_positive_low_high_effect_and_two_of_three_daily_directions": True,
            "misalignment_requires_positive_mismatch_effect_and_stronger_temporal_or_target_specific_relationship": True,
            "structured_signal_requires_positive_M3_vs_M1_in_aggregate_low_and_unseen_plus_two_of_three_low_days": True,
        },
        "evidence": {
            "sparsity_error_targets": sparsity_targets,
            "edge_support_misalignment_targets": misalignment_targets,
            "target_specific_stronger_targets": target_specific_targets,
            "structured_transfer_positive_targets": structured_positive_targets,
            "m4_increment_positive_targets": m4_increment_targets,
            "sparsity_real": sparsity_real,
            "support_misalignment_strong": misalignment_strong,
            "structured_transfer_positive": structured_positive,
            "m4_gate_increment_supported": gate_increment_supported,
        },
        "authorizations": {
            "transfer_v2_authorized": False, "phase_d_authorized": False,
            "stage3_authorized": False,
        },
    }


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fig.savefig(temporary, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    os.replace(temporary, path)


def _binned_cell_curve(cells: pd.DataFrame, support_column: str) -> pd.DataFrame:
    values = _with_cell_means(cells)
    parts = []
    for target in TARGETS:
        subset = values.loc[values["target"].eq(target), [support_column, "m1_cell_mae"]].copy()
        ranks = subset[support_column].rank(method="first")
        bins = pd.qcut(ranks, q=min(20, max(2, len(subset))), labels=False, duplicates="drop")
        subset["quantile_bin"] = bins
        grouped = subset.groupby("quantile_bin", observed=True).agg(
            support_median=(support_column, "median"), m1_cell_mae=("m1_cell_mae", "mean"),
            cell_count=("m1_cell_mae", "size"),
        ).reset_index()
        grouped["target"] = target
        parts.append(grouped)
    return pd.concat(parts, ignore_index=True)


def _make_figures(
    *, support: SupportCounts, cells: pd.DataFrame, transfer: pd.DataFrame,
    cross: pd.DataFrame, output_root: Path,
) -> dict[str, Path]:
    figure_root = output_root / "figures"
    outputs: dict[str, Path] = {}
    for name, counts, title in (
        ("spatial_support_distribution", support.spatial, "Train spatial support"),
        ("spatiotemporal_support_distribution", support.spatiotemporal, "Train edge-time support"),
    ):
        fig, axis = plt.subplots(figsize=(7, 4))
        axis.hist(np.log1p(counts.to_numpy(np.float64)), bins=50, color="#315b7d", alpha=0.9)
        axis.set(xlabel="log1p(unique physical traversal support)", ylabel="support units", title=title)
        path = figure_root / f"{name}.png"
        _save_figure(fig, path)
        outputs[name] = path
    for name, column, title in (
        ("m1_mae_vs_spatiotemporal_support", "spatiotemporal_support", "M1 cell MAE vs edge-time support"),
        ("m1_mae_vs_target_specific_support", "target_specific_support", "M1 cell MAE vs target-valid support"),
    ):
        curve = _binned_cell_curve(cells, column)
        fig, axis = plt.subplots(figsize=(8, 5))
        for target in TARGETS:
            subset = curve.loc[curve["target"].eq(target)]
            axis.plot(np.log1p(subset["support_median"]), subset["m1_cell_mae"], marker="o", label=target)
        axis.set(xlabel="log1p(support), equal-frequency cell bins", ylabel="cell-equal M1 MAE", title=title)
        axis.legend(fontsize=8)
        path = figure_root / f"{name}.png"
        _save_figure(fig, path)
        outputs[name] = path
    selected = transfer.loc[
        transfer["scope"].eq("aggregate")
        & transfer["support_dimension"].eq("spatiotemporal")
        & transfer["comparison"].isin(("M3_vs_M1", "M4_vs_M1", "M4_vs_M3"))
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for axis, target in zip(axes.flat, TARGETS):
        subset = selected.loc[selected["target"].eq(target)]
        for comparison in ("M3_vs_M1", "M4_vs_M1", "M4_vs_M3"):
            ordered = subset.loc[subset["comparison"].eq(comparison)].set_index("support_group").reindex(GROUPS)
            axis.plot(GROUPS, ordered["relative_improvement"] * 100.0, marker="o", label=comparison)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set_title(target)
        axis.set_ylabel("relative MAE improvement (%)")
    axes.flat[0].legend(fontsize=7)
    path = figure_root / "transfer_improvement_vs_support.png"
    _save_figure(fig, path)
    outputs["transfer_improvement_vs_support"] = path
    aggregate_cross = cross.loc[cross["scope"].eq("aggregate")]
    matrix = aggregate_cross.pivot(index="spatial_group", columns="spatiotemporal_group", values="evaluation_share").reindex(index=GROUPS, columns=GROUPS)
    fig, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(matrix.to_numpy(float), cmap="Blues")
    axis.set_xticks(range(len(GROUPS)), GROUPS, rotation=30)
    axis.set_yticks(range(len(GROUPS)), GROUPS)
    axis.set(xlabel="spatiotemporal support", ylabel="spatial support", title="Evaluation traversal share")
    for row in range(len(GROUPS)):
        for column_index in range(len(GROUPS)):
            axis.text(column_index, row, f"{matrix.iloc[row, column_index]:.2%}", ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=axis, label="share")
    path = figure_root / "spatial_temporal_support_heatmap.png"
    _save_figure(fig, path)
    outputs["spatial_temporal_support_heatmap"] = path
    return outputs


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def _format_markdown_value(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value).replace("|", "\\|")


def _markdown_table(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    if frame.empty:
        return "_No rows._"
    lines = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for row in frame.loc[:, list(columns)].to_numpy(dtype=object):
        lines.append("| " + " | ".join(_format_markdown_value(value) for value in row) + " |")
    return "\n".join(lines)


def render_markdown(report: Mapping[str, Any]) -> str:
    tables = report["summaries"]
    spatial = pd.DataFrame(tables["spatial_sparsity"])
    temporal = pd.DataFrame(tables["spatiotemporal_sparsity"])
    mismatch = pd.DataFrame(tables["spatial_high_temporal_sparse"])
    correlations = pd.DataFrame(tables["error_support_correlations"])
    transfer = pd.DataFrame(tables["transfer_spatiotemporal"])
    structure = report["structure_diagnostic"]
    lines = [
        "# Stage 2 v5.2 spatiotemporal sparsity diagnostic", "",
        f"- Diagnostic classification: `{report['diagnostic_classification']['classification']}`",
        f"- Conclusion: `{report['diagnostic_classification']['conclusion']}`",
        f"- Phase C correction: `{report['phase_c_correction']['status']}` / `{report['phase_c_correction']['direction']}`",
        f"- Prediction-level artifact availability: `{report['prediction_level_artifact_availability']['status']}`",
        "- Reinference/retraining: `NO / NO`",
        "- Transfer-v2 / Phase D / Stage 3 authorized: `NO / NO / NO`", "",
        "## Frozen scope", "",
        f"- Train-only support: `{report['protocol']['train_dates'][0]}-{report['protocol']['train_dates'][-1]}`",
        f"- Diagnostic evaluation: `{report['protocol']['evaluation_dates'][0]}-{report['protocol']['evaluation_dates'][-1]}`",
        "- Existing time bin: Xi'an local 30-minute `estimated_time_bin` (0-47)",
        "- Unit: unique physical traversal `(date, order_id, traversal_id)`",
        "- RTS is excluded from the diagnostic main line.", "",
        "## Table 1 — Spatial sparsity (aggregate)", "",
        _markdown_table(spatial, (
            "support_group", "train_unique_edges", "evaluation_unique_edges",
            "evaluation_traversals", "evaluation_share",
        )), "",
        "## Table 2 — Spatiotemporal sparsity (aggregate)", "",
        _markdown_table(temporal, (
            "support_group", "train_unique_edge_time_cells", "evaluation_unique_edge_time_cells",
            "evaluation_traversals", "evaluation_share",
        )), "",
        "## Table 6 — Spatial-high / temporal-sparse", "",
        _markdown_table(mismatch, (
            "target", "n", "m1_mae", "m3_mae", "m4_mae",
            "m3_vs_m1_relative_improvement", "m4_vs_m3_relative_improvement",
        )), "",
        "## M1 cell-level support relationship", "",
        _markdown_table(correlations, (
            "target", "support_dimension", "cell_count",
            "spearman_log1p_support_vs_cell_mae",
        )), "",
        "## M3/M4 improvement by spatiotemporal support", "",
        _markdown_table(transfer, (
            "target", "support_group", "comparison", "n", "baseline_mae",
            "candidate_mae", "relative_improvement",
        )), "",
        "## Structure representation diagnostic", "",
        f"- Unique edges: `{structure['summary']['unique_edge_count']}`",
        f"- Unique signatures: `{structure['summary']['unique_structure_signature_count']}`",
        f"- Mean edges/signature: `{structure['summary']['mean_edges_per_signature']:.3f}`",
        f"- Collision-signature share: `{structure['summary']['collision_signature_share']:.3%}`",
        f"- Maximum edges sharing a signature: `{structure['summary']['maximum_edges_per_signature']}`", "",
        "## Diagnostic interpretation", "",
        report["interpretation"], "",
        "This is descriptive evidence, not a causal estimate. Effect sizes, sample counts, cluster-bootstrap uncertainty, and per-day direction are all preserved in the bound CSV tables.", "",
        "## Stop state", "",
        "No training, inference, checkpoint selection, tau selection, M5/M6, rolling Phase D, 20161028-30 data, or Stage 0/1 rebuild was performed.",
        "Part B remains user-gated. `TRANSFER_V2_AUTHORIZED=NO`, `PHASE_D_AUTHORIZED=NO`, `STAGE3_AUTHORIZED=NO`.", "",
    ]
    return "\n".join(lines)


def _build_evidence(
    *, root: Path, output_root: Path, config_path: Path, report_path: Path,
    report_markdown_path: Path,
    prediction_audit: Mapping[str, Any], train_inputs: Sequence[Mapping[str, Any]],
    evaluation_inputs: Sequence[Mapping[str, Any]], support_paths: Mapping[str, Path],
    table_paths: Mapping[str, Path], figure_paths: Mapping[str, Path],
) -> dict[str, Any]:
    phase_c_evidence_path = root / "stage2/docs/v5_2/stage2_v5_2_phase_c_evidence_bundle.json"
    phase_c_evidence = _read_json(phase_c_evidence_path)
    selected_checkpoints = {
        model: phase_c_evidence["models"][model]["selected_checkpoint"]
        for model in ("M1", "M3", "M4")
    }
    bundle: dict[str, Any] = {
        "schema_version": EVIDENCE_SCHEMA, "status": "PASS",
        "diagnostic_scope": "PART_A_ONLY",
        "phase_c_frozen": {
            "report": _descriptor(root / "stage2/docs/v5_2/stage2_v5_2_phase_c_report.json", root),
            "evidence": _descriptor(phase_c_evidence_path, root),
            "status_manifest": _descriptor(root / "stage2/docs/v5_2/stage2_v5_2_status_manifest.json", root),
            "selected_checkpoints": selected_checkpoints,
        },
        "prediction_level_artifacts": prediction_audit["models"],
        "train_traversal_inputs": list(train_inputs),
        "evaluation_route_inputs": list(evaluation_inputs),
        "time_bin_definition": {
            "column": TIME_BIN_COLUMN,
            "source": _descriptor(root / "stage2/v4/entry_time.py", root),
        },
        "target_valid_mask_sources": {
            "route_dataset_schema": _descriptor(root / "stage2/v5/data.py", root),
            "transfer_adapter": _descriptor(root / "stage2/v5_2/transfer_data.py", root),
            "prediction_overlap_merge": _descriptor(root / "stage2/v5_2/training.py", root),
        },
        "support_artifacts": {name: _descriptor(path, root) for name, path in support_paths.items()},
        "analysis": {
            "config": _descriptor(config_path, root),
            "support_code": _descriptor(root / "stage2/v5_2/sparsity_support.py", root),
            "diagnostic_code": _descriptor(root / "stage2/v5_2/sparsity_diagnostic.py", root),
        },
        "tables": {name: _descriptor(path, root) for name, path in table_paths.items()},
        "figures": {name: _descriptor(path, root) for name, path in figure_paths.items()},
        "reports": {
            "json": _descriptor(report_path, root),
            "markdown": _descriptor(report_markdown_path, root),
        },
        "bootstrap": _read_json(config_path)["bootstrap"],
        "rts_excluded": True,
        "authorizations": {
            "transfer_v2_authorized": False, "phase_d_authorized": False,
            "stage3_authorized": False,
        },
    }
    bundle["artifact_sha256"] = _canonical_hash(bundle)
    return bundle


def verify_evidence_bundle(payload: Mapping[str, Any], *, repo_root: str | Path) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if (
        payload.get("schema_version") != EVIDENCE_SCHEMA
        or payload.get("status") != "PASS"
        or payload.get("diagnostic_scope") != "PART_A_ONLY"
        or payload.get("rts_excluded") is not True
        or any(payload.get("authorizations", {}).get(key) is not False for key in (
            "transfer_v2_authorized", "phase_d_authorized", "stage3_authorized",
        ))
    ):
        raise Stage2V52ContractError("invalid sparsity diagnostic evidence identity")
    _validate_embedded_hash(payload, name="sparsity diagnostic evidence")
    resolved = 0

    def visit(value: Any) -> None:
        nonlocal resolved
        if isinstance(value, Mapping):
            if {"path", "sha256", "size_bytes"}.issubset(value):
                _validate_descriptor(value, root)
                resolved += 1
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(payload)
    report = _read_json(root / str(payload["reports"]["json"]["path"]))
    _validate_embedded_hash(report, name="sparsity diagnostic report")
    if (
        report.get("schema_version") != REPORT_SCHEMA
        or report.get("diagnostic_classification", {}).get("authorizations") != payload["authorizations"]
    ):
        raise Stage2V52ContractError("diagnostic report/evidence relationship differs")
    for descriptor in payload["support_artifacts"].values():
        support = _read_json(root / str(descriptor["path"]))
        _validate_embedded_hash(support, name="sparsity support artifact")
        if support.get("fit_scope") != "train_only" or support.get("evaluation_rows_used") != 0:
            raise Stage2V52ContractError("support artifact is not Train-only")
    forbidden = ("20161022", "20161023", "20161024", "20161028", "20161029", "20161030")
    train_paths = tuple(str(item["path"]) for item in payload["train_traversal_inputs"])
    if any(date in path for date in forbidden for path in train_paths):
        raise Stage2V52ContractError("non-Train date entered support evidence")
    return {
        "status": "PASS", "resolved_artifact_count": resolved,
        "diagnostic_classification": report["diagnostic_classification"]["classification"],
        "transfer_v2_authorized": False, "phase_d_authorized": False,
        "stage3_authorized": False,
    }


def rebuild_existing_evidence(repo_root: str | Path, *, config_path: str | Path) -> dict[str, Any]:
    """Refresh only the hash bundle after report/evidence code review; never rescan model data."""
    root = Path(repo_root).resolve()
    config_file = (root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path).resolve()
    config = _read_json(config_file)
    _check_frozen_state(root, config)
    output_root = root / "stage2/docs/v5_2/sparsity_diagnostic"
    evidence_path = output_root / "stage2_v5_2_spatiotemporal_sparsity_evidence_bundle.json"
    existing = _read_json(evidence_path)
    report_path = output_root / "stage2_v5_2_spatiotemporal_sparsity_report.json"
    markdown_path = output_root / "stage2_v5_2_spatiotemporal_sparsity_report.md"
    support_paths = {
        name: root / str(descriptor["path"])
        for name, descriptor in existing["support_artifacts"].items()
    }
    table_paths = {
        name: root / str(descriptor["path"])
        for name, descriptor in existing["tables"].items()
    }
    figure_paths = {
        name: root / str(descriptor["path"])
        for name, descriptor in existing["figures"].items()
    }
    evidence = _build_evidence(
        root=root, output_root=output_root, config_path=config_file,
        report_path=report_path, report_markdown_path=markdown_path,
        prediction_audit={"models": existing["prediction_level_artifacts"]},
        train_inputs=existing["train_traversal_inputs"],
        evaluation_inputs=existing["evaluation_route_inputs"],
        support_paths=support_paths, table_paths=table_paths, figure_paths=figure_paths,
    )
    _atomic_json(evidence_path, evidence)
    verification = verify_evidence_bundle(evidence, repo_root=root)
    return {
        "status": "PASS", "mode": "EVIDENCE_ONLY",
        "evidence_sha256": sha256_path(evidence_path),
        "resolved_artifact_count": verification["resolved_artifact_count"],
        "diagnostic_classification": verification["diagnostic_classification"],
        "transfer_v2_authorized": False, "phase_d_authorized": False,
        "stage3_authorized": False,
    }


def _target_specific_table(
    distribution: pd.DataFrame, errors: pd.DataFrame, transfer: pd.DataFrame,
) -> pd.DataFrame:
    base = distribution.loc[distribution["scope"].eq("aggregate")].copy()
    error = errors.loc[
        errors["scope"].eq("aggregate")
        & errors["weighting"].eq("traversal")
        & errors["support_dimension"].eq("target_specific"),
        ["target", "support_group", "n", "mae", "median_absolute_error", "p90_absolute_error"],
    ].rename(columns={"n": "error_n", "mae": "m1_mae"})
    selected = transfer.loc[
        transfer["scope"].eq("aggregate")
        & transfer["support_dimension"].eq("target_specific"),
        ["target", "support_group", "comparison", "relative_improvement"],
    ]
    pivot = selected.pivot_table(
        index=["target", "support_group"], columns="comparison",
        values="relative_improvement", aggfunc="first",
    ).reset_index().rename(columns={
        "M3_vs_M1": "m3_vs_m1_relative_improvement",
        "M4_vs_M1": "m4_vs_m1_relative_improvement",
        "M4_vs_M3": "m4_vs_m3_relative_improvement",
    })
    return base.merge(error, on=["target", "support_group"], how="left").merge(
        pivot, on=["target", "support_group"], how="left",
    )


def _structure_table(
    summary: Mapping[str, Any], top: Sequence[Mapping[str, Any]],
    heterogeneity: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    rows = [
        {"section": "summary", "target": None, "metric": key, "value": value}
        for key, value in summary.items()
    ]
    rows.extend(
        {
            "section": "top_signature", "target": None,
            "metric": f"rank_{row['rank']}_unique_edge_count", "value": row["unique_edge_count"],
            "structure_signature_sha256": row["structure_signature_sha256"],
        }
        for row in top
    )
    for row in heterogeneity:
        for key, value in row.items():
            if key not in {"target", "minimum_valid_traversals_per_cell", "minimum_cells_per_signature"}:
                rows.append({
                    "section": "within_signature_heterogeneity", "target": row["target"],
                    "metric": key, "value": value,
                })
    return pd.DataFrame(rows)


def run_diagnostic(repo_root: str | Path, *, config_path: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(repo_root).resolve()
    config_file = (root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path).resolve()
    config = _read_json(config_file)
    if (
        config.get("schema_version") != "stage2_v5_2_sparsity_diagnostic_config.1"
        or tuple(config.get("targets", ())) != TARGETS
        or config.get("protocol_id") != "development"
        or config.get("time_bin", {}).get("column") != TIME_BIN_COLUMN
        or config.get("bootstrap", {}).get("resamples") != 1000
    ):
        raise Stage2V52ContractError("invalid frozen sparsity diagnostic config")
    _check_frozen_state(root, config)
    output_root = root / "stage2/docs/v5_2/sparsity_diagnostic"
    output_root.mkdir(parents=True, exist_ok=True)
    prediction_audit = audit_prediction_availability(
        root, evaluation_dates=config["evaluation_dates"],
    )
    support_started = time.perf_counter()
    support, quantiles, edge_signatures, train_inputs, support_payloads = _fit_train_support(
        root, config, output_root,
    )
    support_runtime = time.perf_counter() - support_started
    prediction_paths = {
        model: prediction_path for model, (_manifest, prediction_path) in _prediction_inputs(root).items()
    }
    aggregate_errors = ErrorAccumulator()
    aggregate_transfer = PairAccumulator()
    aggregate_mismatch = MismatchAccumulator()
    distribution = DistributionAccumulator()
    error_rows: list[dict[str, Any]] = []
    transfer_rows: list[dict[str, Any]] = []
    mismatch_metric_rows: list[dict[str, Any]] = []
    spatial_rows: list[dict[str, Any]] = []
    temporal_rows: list[dict[str, Any]] = []
    target_distribution_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    mismatch_count_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    cell_parts: list[pd.DataFrame] = []
    structure_cell_parts: list[pd.DataFrame] = []
    evaluation_inputs: list[dict[str, Any]] = []
    identity_audit: dict[str, Any] = {}
    evaluation_started = time.perf_counter()
    for date in config["evaluation_dates"]:
        date = str(date)
        frame, day_identity = _read_and_align_evaluation_day(
            root, date, prediction_paths, support, quantiles,
        )
        identity_audit[date] = day_identity
        evaluation_inputs.append(_descriptor(_route_path(root, date), root))
        distribution.add(frame)
        day_spatial, day_temporal, day_target, day_cross, day_mismatch_count = _distribution_rows(
            frame, date=date,
        )
        spatial_rows.extend(day_spatial)
        temporal_rows.extend(day_temporal)
        target_distribution_rows.extend(day_target)
        cross_rows.extend(day_cross)
        mismatch_count_rows.extend(day_mismatch_count)
        error_rows.extend(_add_error_metrics(frame, date=date, aggregate=aggregate_errors))
        transfer_rows.extend(_add_transfer_metrics(frame, date=date, aggregate=aggregate_transfer))
        day_mismatch = MismatchAccumulator()
        day_mismatch.add(frame)
        aggregate_mismatch.add(frame)
        mismatch_metric_rows.extend(day_mismatch.rows(scope="daily", date=date))
        day_cells, day_structure_cells = _cell_aggregates(frame, date=date)
        cell_parts.append(day_cells)
        structure_cell_parts.append(day_structure_cells)
        error_rows.extend(_cell_equal_error_rows(day_cells, scope="daily", date=date))
        correlation_rows.extend(_correlation_rows(day_cells, scope="daily", date=date))
        del frame
    evaluation_runtime = time.perf_counter() - evaluation_started
    aggregate_spatial, aggregate_temporal, aggregate_target, aggregate_cross, mismatch_summary = distribution.rows(
        support=support, quantiles=quantiles,
    )
    spatial_rows.extend(aggregate_spatial)
    temporal_rows.extend(aggregate_temporal)
    target_distribution_rows.extend(aggregate_target)
    cross_rows.extend(aggregate_cross)
    mismatch_count_rows.append(mismatch_summary)
    error_rows.extend(aggregate_errors.rows(scope="aggregate", date="all"))
    transfer_rows.extend(_aggregate_transfer_rows(aggregate_transfer))
    mismatch_metric_rows.extend(aggregate_mismatch.rows(scope="aggregate", date="all"))
    aggregate_cells = _finalize_cells(cell_parts)
    error_rows.extend(_cell_equal_error_rows(aggregate_cells, scope="aggregate", date="all"))
    correlation_rows.extend(_correlation_rows(aggregate_cells, scope="aggregate", date="all"))
    bootstrap_rows = _bootstrap_rows(aggregate_cells, config=config)
    structure_summary, structure_top, structure_heterogeneity = _structure_diagnostic(
        edge_signatures, structure_cell_parts, config=config,
    )
    spatial_frame = pd.DataFrame(spatial_rows)
    temporal_frame = pd.DataFrame(temporal_rows)
    target_distribution_frame = pd.DataFrame(target_distribution_rows)
    cross_frame = pd.DataFrame(cross_rows)
    mismatch_count_frame = pd.DataFrame(mismatch_count_rows)
    errors_frame = pd.DataFrame(error_rows)
    transfer_frame = pd.DataFrame(transfer_rows)
    mismatch_frame = pd.DataFrame(mismatch_metric_rows)
    correlations_frame = pd.DataFrame(correlation_rows)
    bootstrap_frame = pd.DataFrame(bootstrap_rows)
    target_specific_frame = _target_specific_table(
        target_distribution_frame, errors_frame, transfer_frame,
    )
    aggregate_mismatch_rows = mismatch_frame.loc[
        mismatch_frame["scope"].eq("aggregate")
        & mismatch_frame["group"].eq("spatial_high_temporal_sparse")
    ].copy()
    for key, value in mismatch_summary.items():
        if key not in {"scope", "date"}:
            aggregate_mismatch_rows[key] = value
    structure_frame = _structure_table(structure_summary, structure_top, structure_heterogeneity)
    tables: dict[str, pd.DataFrame] = {
        "table_1_spatial_sparsity": spatial_frame,
        "table_2_spatiotemporal_sparsity": temporal_frame,
        "table_3_spatial_temporal_cross_tab": cross_frame,
        "table_4_m1_error_by_support": errors_frame,
        "table_5_transfer_improvement_by_spatiotemporal_support": transfer_frame.loc[
            transfer_frame["support_dimension"].eq("spatiotemporal")
        ],
        "transfer_improvement_by_all_support_dimensions": transfer_frame,
        "table_6_spatial_high_temporal_sparse": aggregate_mismatch_rows,
        "table_7_target_specific_sparsity": target_specific_frame,
        "table_8_structure_representation": structure_frame,
        "daily_spatial_high_temporal_sparse_counts": mismatch_count_frame,
        "daily_and_aggregate_mismatch_metrics": mismatch_frame,
        "support_error_correlations": correlations_frame,
        "cluster_bootstrap_uncertainty": bootstrap_frame,
        "structure_top_signatures": pd.DataFrame(structure_top),
        "structure_within_signature_heterogeneity": pd.DataFrame(structure_heterogeneity),
    }
    table_root = output_root / "tables"
    table_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = table_root / f"{name}.csv"
        _atomic_csv(path, frame)
        table_paths[name] = path
    figures = _make_figures(
        support=support, cells=aggregate_cells, transfer=transfer_frame,
        cross=cross_frame, output_root=output_root,
    )
    classification = classify_diagnostic(
        correlations=correlations_frame, bootstrap=bootstrap_frame,
        mismatch_summary=mismatch_summary, transfer_metrics=transfer_frame,
    )
    interpretations = {
        "DIAG-A": "The edge-only support is materially misaligned with the observed edge-time/target-valid sparsity pattern, and M3 retains a repeatable positive signal in the affected group. This permits only a Transfer-v2 methodology proposal after review; it does not authorize training.",
        "DIAG-B": "Structured transfer has a repeatable positive signal in sparse edge-time cells, while M4 adds no stable increment over M3. Stop support-aware expansion and review M1 versus M3 for the engineering candidate.",
        "DIAG-C": "Sparsity is associated with higher M1 error, but the current static structure representation does not transfer reliably. Do not redesign the support gate inside this phase.",
        "DIAG-D": "Spatiotemporal and target-specific support show no stable relationship with M1 error across the three evaluation days. Stop transfer expansion and compare M1 with M3 on simplicity and overall performance.",
    }
    report_path = output_root / "stage2_v5_2_spatiotemporal_sparsity_report.json"
    markdown_path = output_root / "stage2_v5_2_spatiotemporal_sparsity_report.md"
    support_paths = {
        "spatial": output_root / "spatial_support.json",
        "spatiotemporal": output_root / "spatiotemporal_support.json",
        "target_specific": output_root / "target_specific_support.json",
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA, "status": "PASS", "scope": "PART_A_ONLY",
        "phase_c_correction": {
            "status": "PASS", "direction": "FAIL", "frozen_status": PHASE_C_STATUS,
        },
        "prediction_level_artifact_availability": prediction_audit,
        "protocol": {
            "protocol_id": config["protocol_id"], "train_dates": list(config["train_dates"]),
            "evaluation_dates": list(config["evaluation_dates"]),
            "time_bin": dict(config["time_bin"]),
            "support_fit_scope": "train_only", "evaluation_rows_used_for_support": 0,
            "physical_traversal_identity": list(IDENTITY_COLUMNS), "rts_excluded": True,
        },
        "support_quantiles": quantiles,
        "support_artifacts": {name: _descriptor(path, root) for name, path in support_paths.items()},
        "identity_alignment_audit": identity_audit,
        "spatial_high_temporal_sparse": mismatch_summary,
        "structure_diagnostic": {
            "summary": structure_summary, "top_signatures": structure_top,
            "within_signature_heterogeneity": structure_heterogeneity,
        },
        "diagnostic_classification": classification,
        "interpretation": interpretations[classification["classification"]],
        "summaries": {
            "spatial_sparsity": _records(spatial_frame.loc[spatial_frame["scope"].eq("aggregate")]),
            "spatiotemporal_sparsity": _records(temporal_frame.loc[temporal_frame["scope"].eq("aggregate")]),
            "spatial_high_temporal_sparse": _records(aggregate_mismatch_rows),
            "error_support_correlations": _records(correlations_frame.loc[correlations_frame["scope"].eq("aggregate")]),
            "transfer_spatiotemporal": _records(transfer_frame.loc[
                transfer_frame["scope"].eq("aggregate")
                & transfer_frame["support_dimension"].eq("spatiotemporal")
            ]),
            "target_specific_sparsity": _records(target_specific_frame),
            "bootstrap_uncertainty": _records(bootstrap_frame),
            "daily_correlations": _records(correlations_frame.loc[correlations_frame["scope"].eq("daily")]),
            "daily_mismatch_metrics": _records(mismatch_frame.loc[mismatch_frame["scope"].eq("daily")]),
        },
        "tables": {name: _descriptor(path, root) for name, path in table_paths.items()},
        "figures": {name: _descriptor(path, root) for name, path in figures.items()},
        "runtime": {
            "train_support_seconds": support_runtime,
            "evaluation_analysis_seconds": evaluation_runtime,
            "total_seconds_before_report_write": time.perf_counter() - started,
        },
        "execution_boundaries": dict(config["execution_boundaries"]),
    }
    report["artifact_sha256"] = _canonical_hash(report)
    _atomic_json(report_path, report)
    _atomic_text(markdown_path, render_markdown(report))
    evidence = _build_evidence(
        root=root, output_root=output_root, config_path=config_file,
        report_path=report_path, report_markdown_path=markdown_path,
        prediction_audit=prediction_audit,
        train_inputs=train_inputs, evaluation_inputs=evaluation_inputs,
        support_paths=support_paths, table_paths=table_paths, figure_paths=figures,
    )
    evidence_path = output_root / "stage2_v5_2_spatiotemporal_sparsity_evidence_bundle.json"
    _atomic_json(evidence_path, evidence)
    verification = verify_evidence_bundle(evidence, repo_root=root)
    return {
        "status": "PASS", "diagnostic_classification": classification["classification"],
        "prediction_level_artifact_availability": "PASS",
        "report_sha256": sha256_path(report_path),
        "evidence_sha256": sha256_path(evidence_path),
        "resolved_artifact_count": verification["resolved_artifact_count"],
        "transfer_v2_authorized": False, "phase_d_authorized": False,
        "stage3_authorized": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config", default="stage2/config/stage2_v5_2_sparsity_diagnostic.json",
    )
    parser.add_argument("--evidence-only", action="store_true")
    args = parser.parse_args(argv)
    result = (
        rebuild_existing_evidence(args.repo_root, config_path=args.config)
        if args.evidence_only
        else run_diagnostic(args.repo_root, config_path=args.config)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
