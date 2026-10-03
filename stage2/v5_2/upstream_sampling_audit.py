"""Read-only Stage 0/1 upstream sampling representativeness audit.

This module deliberately reads only frozen manifests/products.  It never invokes
Stage 0/1 production, map matching, model training, inference, or tau selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer

from .contracts import Stage2V52ContractError
from .upstream_sampling_support import (
    REJECTION_GROUPS,
    SUPPORT_GROUPS,
    TARGETS,
    TARGET_VALID_COLUMNS,
    add_frozen_stage1_target_masks,
    aggregate_rank_decision,
    assert_disjoint_identity_sets,
    assert_selection_hash_contract,
    assign_support_groups,
    classify_upstream,
    cluster_bootstrap_rate_effect,
    jensen_shannon_divergence,
    local_time_bin,
    material_negative_rate_effect,
    normalized_selection_rank,
    positive_support_quantiles,
    probability_distribution,
    rejection_mechanisms,
    split_rejection_tokens,
    total_variation,
    validate_funnel_identity,
)


REPORT_SCHEMA = "stage0_stage1_upstream_representativeness_report.1"
EVIDENCE_SCHEMA = "stage0_stage1_upstream_representativeness_evidence.1"
IDENTITY = ("date", "order_id")
RAW_REQUIRED = {
    "order_id", "date", "selection_hash", "start_time", "start_lon",
    "start_lat", "end_lon", "end_lat", "eligible",
}
REJECTION_REQUIRED = {
    "order_id", "date", "selection_hash", "pre_match_status",
    "rejection_reason",
}
ACCEPTED_REQUIRED = {
    "order_id", "date", "selection_hash", "stage1_core_eligible",
}
LABEL_COLUMNS = (
    "date", "order_id", "traversal_id", "observed_directed_edge_uid",
    "time_bin_30m", "direct_interval_count", "direct_observed_time_s",
    "crawl_time_share", "stop_time_share", "speed_cv_bounded",
    "acceleration_rms_bounded", "acceleration_pair_count",
    "acceleration_weight_s",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return value.as_posix()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_text(
        path,
        json.dumps(
            payload, indent=2, sort_keys=True, ensure_ascii=False,
            default=_json_default,
        ) + "\n",
    )


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, lineterminator="\n")
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _descriptor(path: Path, root: Path) -> dict[str, Any]:
    resolved = path.resolve()
    if not resolved.is_file():
        raise Stage2V52ContractError(f"evidence file is missing: {path}")
    return {
        "path": resolved.relative_to(root.resolve()).as_posix(),
        "bytes": resolved.stat().st_size,
        "sha256": _sha256_file(resolved),
    }


def _descriptor_set(paths: Sequence[Path], root: Path) -> dict[str, Any]:
    entries = [_descriptor(path, root) for path in sorted(paths)]
    digest = hashlib.sha256()
    for item in entries:
        digest.update(item["path"].encode("utf-8"))
        digest.update(bytes.fromhex(item["sha256"]))
    return {
        "file_count": len(entries),
        "total_bytes": int(sum(item["bytes"] for item in entries)),
        "set_sha256": digest.hexdigest(),
        "files": entries,
    }


def _git_head(root: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True,
        capture_output=True, text=True,
    ).stdout.strip()


def _load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_scope") != "DESCRIPTIVE_REPRESENTATIVENESS_AUDIT_ONLY":
        raise Stage2V52ContractError("audit scope is not frozen to descriptive-only")
    authorizations = payload.get("authorizations", {})
    if not authorizations or any(bool(value) for value in authorizations.values()):
        raise Stage2V52ContractError("all execution authorizations must remain false")
    if payload["time_bin"] != {
        "timezone": "Asia/Shanghai", "minutes": 30, "count": 48,
        "definition": "local_hour * 2 + (local_minute >= 30)",
    }:
        raise Stage2V52ContractError("time-bin contract differs from frozen Stage 2")
    if payload["spatial_grid"]["cell_size_m"] != 1000:
        raise Stage2V52ContractError("spatial grid must remain fixed at 1 km")
    return payload


def _all_dates(config: Mapping[str, Any]) -> tuple[str, ...]:
    dates = config["dates"]
    return tuple(str(value) for split in ("train", "validation", "test") for value in dates[split])


def _split_for_date(config: Mapping[str, Any], date: str) -> str:
    for split, values in config["dates"].items():
        if date in {str(value) for value in values}:
            return str(split)
    raise Stage2V52ContractError(f"date is outside frozen audit protocol: {date}")


def _require_columns(columns: Iterable[str], required: Iterable[str], product: str) -> None:
    missing = set(required) - set(columns)
    if missing:
        raise Stage2V52ContractError(f"{product} lacks columns: {sorted(missing)}")


def _partition_value(path: Path, prefix: str) -> str:
    for part in path.parts:
        if part.startswith(prefix + "="):
            return part.split("=", 1)[1]
    raise Stage2V52ContractError(f"{path} lacks {prefix}= partition")


def _discover_inputs(root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    dates = _all_dates(config)
    candidate_root = root / paths["candidate_manifests"]
    candidate = {
        date: candidate_root / f"date={date}.parquet" for date in dates
    }
    missing_candidates = [date for date, path in candidate.items() if not path.is_file()]
    input_root = root / paths["stage1_input"]
    output_root = root / paths["stage1_output"]
    rejection = sorted((input_root / "rejections").glob("split=*/date=*/bucket=*.parquet"))
    accepted = sorted(input_root.glob("split=*/date=*/bucket=*/order_base.parquet"))
    input_manifests = sorted(input_root.glob("split=*/date=*/bucket=*/manifest.json"))
    labels = sorted(output_root.glob("split=*/date=*/bucket=*/traversal_labels.parquet"))
    output_manifests = sorted(output_root.glob("split=*/date=*/bucket=*/manifest.json"))
    required_files = [
        root / paths["stage0_config"],
        root / paths["stage1_release_manifest"],
        root / paths["stage1_scientific_review_json"],
        root / paths["stage1_scientific_review_markdown"],
        root / paths["stage2_sparsity_evidence"],
    ]
    missing = [str(path) for path in required_files if not path.is_file()]
    if missing_candidates or not rejection or not accepted:
        detail = {
            "missing_candidate_dates": missing_candidates,
            "rejection_manifest_count": len(rejection),
            "accepted_order_product_count": len(accepted),
        }
        raise Stage2V52ContractError(
            "UPSTREAM_AUDIT_BLOCKED_BY_MISSING_ARTIFACT: "
            + json.dumps(detail, sort_keys=True)
        )
    if missing:
        raise Stage2V52ContractError(f"frozen evidence inputs are missing: {missing}")
    if not (len(rejection) == len(accepted) == len(input_manifests) == len(labels) == len(output_manifests)):
        raise Stage2V52ContractError(
            "Stage 0/1 bucket inventories do not reconcile: "
            f"rejection={len(rejection)} accepted={len(accepted)} "
            f"input_manifest={len(input_manifests)} labels={len(labels)} "
            f"output_manifest={len(output_manifests)}"
        )
    observed_dates = {
        _partition_value(path, "date") for path in accepted
    }
    if observed_dates != set(dates):
        raise Stage2V52ContractError(
            f"accepted-order dates disagree with protocol: {sorted(observed_dates)}"
        )
    return {
        "candidate": candidate,
        "rejection": rejection,
        "accepted": accepted,
        "input_manifests": input_manifests,
        "labels": labels,
        "output_manifests": output_manifests,
        "required_files": required_files,
        "input_root": input_root,
        "output_root": output_root,
    }


def _production_snapshot(paths: Sequence[Path], root: Path) -> str:
    digest = hashlib.sha256()
    files: list[Path] = []
    for path in paths:
        files.extend(item for item in path.rglob("*") if item.is_file())
    for path in sorted(set(files)):
        stat = path.stat()
        digest.update(path.resolve().relative_to(root.resolve()).as_posix().encode("utf-8"))
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def _grid_ids(
    lon: pd.Series, lat: pd.Series, *, transformer: Transformer,
    cell_size_m: float, origin_m: Sequence[float], prefix: str,
) -> pd.Series:
    longitude = pd.to_numeric(lon, errors="coerce")
    latitude = pd.to_numeric(lat, errors="coerce")
    valid = longitude.between(-180, 180) & latitude.between(-90, 90)
    result = pd.Series("MISSING", index=lon.index, dtype="string")
    if valid.any():
        x, y = transformer.transform(
            longitude.loc[valid].to_numpy(float), latitude.loc[valid].to_numpy(float)
        )
        east = np.floor((np.asarray(x) - float(origin_m[0])) / cell_size_m).astype(np.int64)
        north = np.floor((np.asarray(y) - float(origin_m[1])) / cell_size_m).astype(np.int64)
        result.loc[valid] = pd.Series(
            [f"{prefix}E{e}_N{n}" for e, n in zip(east, north)],
            index=result.index[valid], dtype="string",
        )
    return result


def _raw_context(
    path: Path, *, date: str, config: Mapping[str, Any],
    transformer: Transformer,
) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    _require_columns(frame.columns, RAW_REQUIRED, "raw candidate manifest")
    frame["date"] = frame["date"].astype(str)
    if not frame["date"].eq(date).all():
        raise Stage2V52ContractError(f"candidate rows disagree with partition date {date}")
    if frame["order_id"].astype(str).duplicated().any():
        raise Stage2V52ContractError(f"candidate identities are duplicated on {date}")
    assert_selection_hash_contract(
        frame, date=date, seed=int(config["selection"]["seed"])
    )
    frame["order_id"] = frame["order_id"].astype(str)
    frame["selection_hash"] = frame["selection_hash"].astype(str)
    frame["normalized_selection_rank"] = normalized_selection_rank(frame)
    frame["time_bin"] = local_time_bin(
        frame["start_time"], timezone=config["time_bin"]["timezone"]
    ).fillna(-1).astype("int16")
    grid = config["spatial_grid"]
    frame["origin_grid"] = _grid_ids(
        frame["start_lon"], frame["start_lat"], transformer=transformer,
        cell_size_m=float(grid["cell_size_m"]), origin_m=grid["origin_m"],
        prefix="O",
    )
    frame["destination_grid"] = _grid_ids(
        frame["end_lon"], frame["end_lat"], transformer=transformer,
        cell_size_m=float(grid["cell_size_m"]), origin_m=grid["origin_m"],
        prefix="D",
    )
    frame["od_grid_pair"] = (
        frame["origin_grid"].astype(str) + "->" + frame["destination_grid"].astype(str)
    )
    frame["origin_time_key"] = list(zip(
        frame["origin_grid"].astype(str), frame["time_bin"].astype(int)
    ))
    frame["od_time_key"] = list(zip(
        frame["od_grid_pair"].astype(str), frame["time_bin"].astype(int)
    ))
    return frame


def _counter_update(counter: Counter[Any], values: pd.Series) -> None:
    counts = values.value_counts(dropna=False)
    counter.update({key: int(value) for key, value in counts.items()})


def _fit_raw_support(
    inputs: Mapping[str, Any], config: Mapping[str, Any],
    transformer: Transformer,
) -> tuple[dict[str, Counter[Any]], dict[str, dict[str, int]], int]:
    counters: dict[str, Counter[Any]] = {
        "temporal": Counter(), "origin_grid": Counter(),
        "od_grid_pair": Counter(), "origin_grid_x_time_bin": Counter(),
        "od_grid_pair_x_time_bin": Counter(),
    }
    rows = 0
    for date in config["dates"]["train"]:
        date = str(date)
        frame = _raw_context(
            inputs["candidate"][date], date=date, config=config,
            transformer=transformer,
        )
        rows += len(frame)
        for name, column in (
            ("temporal", "time_bin"), ("origin_grid", "origin_grid"),
            ("od_grid_pair", "od_grid_pair"),
            ("origin_grid_x_time_bin", "origin_time_key"),
            ("od_grid_pair_x_time_bin", "od_time_key"),
        ):
            _counter_update(counters[name], frame[column])
    quantiles = {
        name: positive_support_quantiles(values, config["support"]["quantiles"])
        for name, values in counters.items()
    }
    return counters, quantiles, rows


def _load_stage0_outcomes(
    root: Path, inputs: Mapping[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    accepted_parts: list[pd.DataFrame] = []
    rejection_parts: list[pd.DataFrame] = []
    manifest_accepted = 0
    manifest_candidate = 0
    manifest_prefilter = 0
    manifest_matched = 0
    manifest_rejected = 0
    manifest_exceptions = 0
    for accepted_path, rejection_path, manifest_path in zip(
        inputs["accepted"], inputs["rejection"], inputs["input_manifests"]
    ):
        accepted = pd.read_parquet(accepted_path)
        rejection = pd.read_parquet(rejection_path)
        _require_columns(accepted.columns, ACCEPTED_REQUIRED, "order_base")
        _require_columns(rejection.columns, REJECTION_REQUIRED, "rejection manifest")
        if not accepted["stage1_core_eligible"].fillna(False).astype(bool).all():
            raise Stage2V52ContractError(f"non-core order found in {accepted_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "PASS":
            raise Stage2V52ContractError(f"Stage 0 bucket did not pass: {manifest_path}")
        if int(manifest["accepted_core_count"]) != len(accepted):
            raise Stage2V52ContractError(
                f"accepted identity count disagrees with {manifest_path}"
            )
        expected_date = _partition_value(accepted_path, "date")
        if not accepted["date"].astype(str).eq(expected_date).all():
            raise Stage2V52ContractError(f"accepted partition mismatch: {accepted_path}")
        if len(rejection) and not rejection["date"].astype(str).eq(expected_date).all():
            raise Stage2V52ContractError(f"rejection partition mismatch: {rejection_path}")
        accepted_parts.append(accepted.loc[:, sorted(ACCEPTED_REQUIRED)].copy())
        rejection_parts.append(rejection.loc[:, sorted(REJECTION_REQUIRED)].copy())
        manifest_accepted += int(manifest["accepted_core_count"])
        manifest_candidate += int(manifest["candidate_count"])
        manifest_prefilter += int(manifest["pre_match_eligible_count"])
        manifest_matched += int(manifest["matched_count"])
        manifest_rejected += int(manifest["rejected_count"])
        manifest_exceptions += int(manifest["processing_exception_count"])
    accepted = pd.concat(accepted_parts, ignore_index=True)
    rejected = pd.concat(rejection_parts, ignore_index=True)
    for frame in (accepted, rejected):
        frame["date"] = frame["date"].astype(str)
        frame["order_id"] = frame["order_id"].astype(str)
        frame["selection_hash"] = frame["selection_hash"].astype(str)
    if accepted.duplicated(list(IDENTITY)).any():
        raise Stage2V52ContractError("accepted order identities are duplicated")
    if rejected.duplicated(list(IDENTITY)).any():
        raise Stage2V52ContractError("rejection identities are duplicated")
    overlap = accepted.loc[:, list(IDENTITY)].merge(
        rejected.loc[:, list(IDENTITY)], on=list(IDENTITY), how="inner"
    )
    accepted_identities = set(map(tuple, accepted.loc[:, list(IDENTITY)].to_numpy()))
    rejected_identities = set(map(tuple, rejected.loc[:, list(IDENTITY)].to_numpy()))
    assert_disjoint_identity_sets(
        accepted_identities | rejected_identities,
        accepted_identities,
        rejected_identities,
    )
    audit = {
        "bucket_count": len(inputs["input_manifests"]),
        "manifest_candidate_count": manifest_candidate,
        "manifest_prefilter_eligible_count": manifest_prefilter,
        "manifest_matched_count": manifest_matched,
        "manifest_accepted_count": manifest_accepted,
        "manifest_rejected_count": manifest_rejected,
        "manifest_processing_exception_count": manifest_exceptions,
        "accepted_identity_count": len(accepted),
        "rejection_identity_count": len(rejected),
        "accepted_rejected_overlap_count": len(overlap),
    }
    return accepted, rejected, audit


def _rank_summary(frame: pd.DataFrame, dimension: str, date: str) -> pd.DataFrame:
    values = frame.loc[:, [dimension, "normalized_selection_rank"]].copy()
    minimum = 1 if dimension in {"time_bin", "raw_sparsity_group"} else 30
    frequencies = values[dimension].value_counts(dropna=False)
    retained = frequencies.loc[frequencies.ge(minimum)].index
    values = values.loc[values[dimension].isin(retained)]
    if values.empty:
        return pd.DataFrame()
    grouped = values.groupby(dimension, observed=True, dropna=False)[
        "normalized_selection_rank"
    ]
    result = grouped.agg(count="size", mean_rank="mean", median_rank="median")
    result["p10_rank"] = grouped.quantile(0.10)
    result["p90_rank"] = grouped.quantile(0.90)
    result = result.reset_index().rename(columns={dimension: "stratum"})
    result.insert(0, "dimension", dimension)
    result.insert(0, "date", date)
    result["stratum"] = result["stratum"].astype(str)
    return result


def _increment(counter: Counter[Any], values: pd.Series) -> None:
    grouped = values.value_counts(dropna=False)
    counter.update({key: int(value) for key, value in grouped.items()})


def _audit_raw_funnel(
    inputs: Mapping[str, Any], config: Mapping[str, Any],
    transformer: Transformer, raw_counts: Mapping[str, Counter[Any]],
    raw_quantiles: Mapping[str, Mapping[str, int]], accepted: pd.DataFrame,
    rejected: pd.DataFrame,
) -> dict[str, Any]:
    accepted_by_date = {
        date: group.set_index("order_id", drop=False)
        for date, group in accepted.groupby("date", sort=False)
    }
    rejected_by_date = {
        date: group.set_index("order_id", drop=False)
        for date, group in rejected.groupby("date", sort=False)
    }
    stage_time: dict[str, Counter[Any]] = {
        stage: Counter() for stage in ("raw", "processed", "prefilter", "matched", "accepted")
    }
    stage_origin: dict[str, Counter[Any]] = {
        stage: Counter() for stage in ("raw", "processed", "accepted")
    }
    stage_group: dict[str, Counter[Any]] = {
        stage: Counter() for stage in ("raw", "processed", "prefilter", "matched", "accepted")
    }
    daily_rows: list[dict[str, Any]] = []
    comparison_rows: list[dict[str, Any]] = []
    rank_rows: list[pd.DataFrame] = []
    processed_context: list[pd.DataFrame] = []
    accepted_context: list[pd.DataFrame] = []
    missing_time_count = 0
    missing_spatial_count = 0
    seen_accepted = seen_rejected = 0
    primary_counts = raw_counts[config["support"]["primary_raw_context"]]
    primary_quantiles = raw_quantiles[config["support"]["primary_raw_context"]]
    rare_groups = set(config["support"]["rare_groups"])

    for date in _all_dates(config):
        frame = _raw_context(
            inputs["candidate"][date], date=date, config=config,
            transformer=transformer,
        )
        frame["raw_sparsity_group"] = assign_support_groups(
            frame["origin_time_key"], primary_counts, primary_quantiles
        )
        accepted_day = accepted_by_date.get(date, pd.DataFrame()).copy()
        rejected_day = rejected_by_date.get(date, pd.DataFrame()).copy()
        accepted_ids = set(accepted_day.index.astype(str)) if len(accepted_day) else set()
        rejected_ids = set(rejected_day.index.astype(str)) if len(rejected_day) else set()
        frame["accepted"] = frame["order_id"].isin(accepted_ids)
        frame["rejected"] = frame["order_id"].isin(rejected_ids)
        if (frame["accepted"] & frame["rejected"]).any():
            raise Stage2V52ContractError(f"accepted/rejected overlap on {date}")
        frame["processed"] = frame["accepted"] | frame["rejected"]
        rejection_status = (
            rejected_day["pre_match_status"].astype(str).to_dict()
            if len(rejected_day) else {}
        )
        rejection_reason = (
            rejected_day["rejection_reason"].astype(str).to_dict()
            if len(rejected_day) else {}
        )
        frame["rejection_pre_match_status"] = frame["order_id"].map(rejection_status)
        frame["rejection_reason"] = frame["order_id"].map(rejection_reason)
        frame["prefilter_pass"] = frame["accepted"] | frame[
            "rejection_pre_match_status"
        ].eq("eligible")
        exception = frame["rejection_reason"].fillna("").str.startswith(
            "PROCESSING_EXCEPTION:"
        )
        frame["matched"] = frame["prefilter_pass"] & ~exception
        for outcome, outcome_frame in (
            ("accepted", accepted_day), ("rejected", rejected_day)
        ):
            if not len(outcome_frame):
                continue
            actual = frame.set_index("order_id")["selection_hash"]
            aligned = outcome_frame["selection_hash"].astype(str)
            missing_ids = aligned.index.difference(actual.index)
            if len(missing_ids):
                raise Stage2V52ContractError(
                    f"{len(missing_ids)} {outcome} identities missing from raw manifest on {date}"
                )
            mismatch = int(actual.loc[aligned.index].astype(str).ne(aligned).sum())
            if mismatch:
                raise Stage2V52ContractError(
                    f"{mismatch} {outcome} selection hashes disagree on {date}"
                )
        seen_accepted += int(frame["accepted"].sum())
        seen_rejected += int(frame["rejected"].sum())
        missing_time_count += int(frame["time_bin"].eq(-1).sum())
        missing_spatial_count += int(frame["origin_grid"].eq("MISSING").sum())

        masks = {
            "raw": np.ones(len(frame), dtype=bool),
            "processed": frame["processed"].to_numpy(bool),
            "prefilter": frame["prefilter_pass"].to_numpy(bool),
            "matched": frame["matched"].to_numpy(bool),
            "accepted": frame["accepted"].to_numpy(bool),
        }
        for stage, mask in masks.items():
            _increment(stage_time[stage], frame.loc[mask, "time_bin"])
            _increment(stage_group[stage], frame.loc[mask, "raw_sparsity_group"])
            if stage in stage_origin:
                _increment(stage_origin[stage], frame.loc[mask, "origin_grid"])
        raw_n = len(frame)
        processed_n = int(frame["processed"].sum())
        prefilter_n = int(frame["prefilter_pass"].sum())
        matched_n = int(frame["matched"].sum())
        accepted_n = int(frame["accepted"].sum())
        rejected_n = int(frame["rejected"].sum())
        daily_rows.append({
            "date": date,
            "split": _split_for_date(config, date),
            "raw_candidates": raw_n,
            "processed_candidates": processed_n,
            "prefilter_pass": prefilter_n,
            "valhalla_processed_or_matched": matched_n,
            "core_accepted": accepted_n,
            "rejected": rejected_n,
            "unprocessed_quota": raw_n - processed_n,
            "processing_opportunity_rate": processed_n / raw_n,
            "prefilter_pass_given_processed": prefilter_n / processed_n,
            "matched_given_prefilter": matched_n / prefilter_n,
            "core_acceptance_given_processed": accepted_n / processed_n,
            "core_acceptance_given_raw": accepted_n / raw_n,
        })
        comparison = np.select(
            [frame["raw_sparsity_group"].isin(rare_groups), frame["raw_sparsity_group"].eq("high")],
            ["rare", "common"], default="excluded_medium",
        )
        frame["comparison_group"] = comparison
        for label in ("rare", "common"):
            mask = frame["comparison_group"].eq(label)
            comparison_rows.append({
                "date": date,
                "comparison_group": label,
                "raw": int(mask.sum()),
                "processed": int((mask & frame["processed"]).sum()),
                "prefilter": int((mask & frame["prefilter_pass"]).sum()),
                "matched": int((mask & frame["matched"]).sum()),
                "accepted": int((mask & frame["accepted"]).sum()),
            })
        for dimension in ("time_bin", "origin_grid", "od_grid_pair", "raw_sparsity_group"):
            summary = _rank_summary(frame, dimension, date)
            if len(summary):
                rank_rows.append(summary)
        context_columns = [
            "date", "order_id", "time_bin", "origin_grid", "destination_grid",
            "od_grid_pair", "raw_sparsity_group", "normalized_selection_rank",
        ]
        processed_context.append(frame.loc[frame["processed"], context_columns].copy())
        accepted_context.append(frame.loc[frame["accepted"], context_columns].copy())

    if seen_accepted != len(accepted) or seen_rejected != len(rejected):
        raise Stage2V52ContractError(
            "raw/outcome identity reconciliation failed: "
            f"accepted={seen_accepted}/{len(accepted)} rejected={seen_rejected}/{len(rejected)}"
        )
    processed_frame = pd.concat(processed_context, ignore_index=True)
    accepted_frame = pd.concat(accepted_context, ignore_index=True)
    if processed_frame.duplicated(list(IDENTITY)).any() or accepted_frame.duplicated(list(IDENTITY)).any():
        raise Stage2V52ContractError("context identity products contain duplicates")
    rank_frame = pd.concat(rank_rows, ignore_index=True)
    broad = rank_frame.loc[
        rank_frame["dimension"].isin(["time_bin", "raw_sparsity_group"])
    ].copy()
    broad_aggregate, maximum_broad_rank_gap = aggregate_rank_decision(
        broad, minimum_count=30
    )
    rank_limit = float(config["classification_thresholds"]["maximum_sampling_rank_mean_gap"])
    sampling_hash_status = "PASS" if maximum_broad_rank_gap <= rank_limit else "FAIL"
    return {
        "daily": pd.DataFrame(daily_rows),
        "comparison_daily": pd.DataFrame(comparison_rows),
        "selection_rank": rank_frame,
        "selection_rank_decision_summary": broad_aggregate,
        "processed_context": processed_frame,
        "accepted_context": accepted_frame,
        "stage_time": stage_time,
        "stage_origin": stage_origin,
        "stage_group": stage_group,
        "missing_time_count": missing_time_count,
        "missing_spatial_count": missing_spatial_count,
        "sampling_hash_status": sampling_hash_status,
        "maximum_broad_rank_mean_gap": maximum_broad_rank_gap,
    }


def _audit_rejections(
    rejected: pd.DataFrame, processed_context: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    context = processed_context.loc[:, ["date", "order_id", "raw_sparsity_group"]]
    merged = rejected.merge(
        context, on=["date", "order_id"], how="left", validate="one_to_one"
    )
    if merged["raw_sparsity_group"].isna().any():
        raise Stage2V52ContractError("rejection rows lack raw context")
    mechanism_rows: list[dict[str, Any]] = []
    token_counter: Counter[tuple[str, str]] = Counter()
    mechanism_counter: Counter[tuple[str, str]] = Counter()
    rejection_groups = merged["raw_sparsity_group"].astype(str).to_numpy(object)
    rejection_reasons = merged["rejection_reason"].to_numpy(object)
    for group_value, reason in zip(rejection_groups, rejection_reasons):
        group = str(group_value)
        tokens = split_rejection_tokens(reason)
        mechanisms = rejection_mechanisms(reason)
        for token in tokens:
            canonical_token = (
                "MATCH_REJECTION:*" if token.startswith("MATCH_REJECTION:")
                else "PROCESSING_EXCEPTION:*" if token.startswith("PROCESSING_EXCEPTION:")
                else token
            )
            token_counter[(group, canonical_token)] += 1
        for mechanism in mechanisms:
            mechanism_counter[(group, mechanism)] += 1
    for support_group in SUPPORT_GROUPS:
        group_rows = merged["raw_sparsity_group"].astype(str).eq(support_group)
        row: dict[str, Any] = {
            "support_group": support_group,
            "rejection_count": int(group_rows.sum()),
        }
        for mechanism in REJECTION_GROUPS:
            row[mechanism] = mechanism_counter[(support_group, mechanism)]
            row[f"{mechanism}_share_of_rejections"] = (
                row[mechanism] / row["rejection_count"] if row["rejection_count"] else None
            )
        mechanism_rows.append(row)
    token_rows = [
        {"support_group": group, "rejection_token": token, "count": count}
        for (group, token), count in sorted(token_counter.items())
    ]
    audit = {
        "rejection_count": len(merged),
        "mapped_rejection_count": len(merged),
        "unmapped_rejection_count": 0,
        "multi_mechanism_rejection_count": int(sum(
            len(rejection_mechanisms(reason)) > 1
            for reason in merged["rejection_reason"]
        )),
        "attribution_policy": "multi-label; one rejection may contribute to multiple mechanisms",
    }
    return pd.DataFrame(mechanism_rows), pd.DataFrame(token_rows), audit


def _fit_edge_support(
    label_paths: Sequence[Path], train_dates: set[str],
    quantiles: Sequence[float],
) -> tuple[Counter[str], Counter[tuple[str, int]], dict[str, dict[str, int]], int]:
    edge: Counter[str] = Counter()
    edge_time: Counter[tuple[str, int]] = Counter()
    train_rows = 0
    for path in label_paths:
        date = _partition_value(path, "date")
        if date not in train_dates:
            continue
        frame = pd.read_parquet(
            path, columns=["observed_directed_edge_uid", "time_bin_30m"]
        )
        uid = frame["observed_directed_edge_uid"].astype(str)
        bins = pd.to_numeric(frame["time_bin_30m"], errors="raise").astype(int)
        if not bins.between(0, 47).all():
            raise Stage2V52ContractError(f"invalid frozen time bin in {path}")
        _counter_update(edge, uid)
        pairs = pd.Series(list(zip(uid, bins)), index=frame.index)
        _counter_update(edge_time, pairs)
        train_rows += len(frame)
    support_quantiles = {
        "edge_spatial_support": positive_support_quantiles(edge, quantiles),
        "edge_time_support": positive_support_quantiles(edge_time, quantiles),
    }
    return edge, edge_time, support_quantiles, train_rows


def _audit_stage1_labels(
    inputs: Mapping[str, Any], config: Mapping[str, Any],
    accepted_context: pd.DataFrame,
) -> dict[str, Any]:
    train_dates = {str(value) for value in config["dates"]["train"]}
    edge_counts, edge_time_counts, edge_quantiles, train_label_rows = _fit_edge_support(
        inputs["labels"], train_dates, config["support"]["quantiles"]
    )
    context_by_date = {
        date: group.set_index("order_id").loc[:, [
            "time_bin", "origin_grid", "od_grid_pair", "raw_sparsity_group"
        ]]
        for date, group in accepted_context.groupby("date", sort=False)
    }
    denominator: Counter[tuple[str, str]] = Counter()
    valid_counts: Counter[tuple[str, str, str]] = Counter()
    target_time: dict[str, Counter[int]] = {target: Counter() for target in TARGETS}
    target_origin: dict[str, Counter[str]] = {target: Counter() for target in TARGETS}
    target_raw_group: dict[str, Counter[str]] = {target: Counter() for target in TARGETS}
    direct_raw_group: Counter[str] = Counter()
    daily_target_rows: list[dict[str, Any]] = []
    daily_label_denominator: Counter[tuple[str, str]] = Counter()
    daily_label_valid: Counter[tuple[str, str, str]] = Counter()
    identity_count = 0
    invalid_context = 0
    for path in inputs["labels"]:
        date = _partition_value(path, "date")
        frame = pd.read_parquet(path, columns=list(LABEL_COLUMNS))
        frame["date"] = frame["date"].astype(str)
        frame["order_id"] = frame["order_id"].astype(str)
        if not frame["date"].eq(date).all():
            raise Stage2V52ContractError(f"label partition mismatch: {path}")
        if frame.duplicated(["date", "order_id", "traversal_id"]).any():
            raise Stage2V52ContractError(f"duplicate physical label identity: {path}")
        frame = add_frozen_stage1_target_masks(frame)
        context = context_by_date.get(date)
        if context is None:
            raise Stage2V52ContractError(f"no accepted-order context for {date}")
        frame = frame.join(context, on="order_id", how="left", validate="many_to_one")
        missing = frame["raw_sparsity_group"].isna()
        invalid_context += int(missing.sum())
        if missing.any():
            raise Stage2V52ContractError(
                f"{int(missing.sum())} Stage 1 traversals lack accepted-order identity in {path}"
            )
        edge_uid = frame["observed_directed_edge_uid"].astype(str)
        time_bin = pd.to_numeric(frame["time_bin_30m"], errors="raise").astype(int)
        if not time_bin.between(0, 47).all():
            raise Stage2V52ContractError(f"invalid Stage 1 time bin: {path}")
        edge_group = assign_support_groups(
            edge_uid, edge_counts, edge_quantiles["edge_spatial_support"]
        )
        pairs = pd.Series(list(zip(edge_uid, time_bin)), index=frame.index)
        edge_time_group = assign_support_groups(
            pairs, edge_time_counts, edge_quantiles["edge_time_support"]
        )
        dimension_groups = {
            "origin_time_support": frame["raw_sparsity_group"].astype(str),
            "edge_spatial_support": edge_group.astype(str),
            "edge_time_support": edge_time_group.astype(str),
        }
        rare_groups = set(config["support"]["rare_groups"])
        comparison = np.select(
            [
                frame["raw_sparsity_group"].astype(str).isin(rare_groups),
                frame["raw_sparsity_group"].astype(str).eq("high"),
            ],
            ["rare", "common"], default="excluded_medium",
        )
        comparison = pd.Series(comparison, index=frame.index, dtype="string")
        for label in ("rare", "common"):
            mask = comparison.eq(label)
            daily_label_denominator[(date, label)] += int(mask.sum())
            for target in TARGETS:
                valid = frame[TARGET_VALID_COLUMNS[target]].to_numpy(bool)
                daily_label_valid[(date, target, label)] += int(
                    (mask.to_numpy(bool) & valid).sum()
                )
        for dimension, groups in dimension_groups.items():
            counts = groups.value_counts()
            for group, count in counts.items():
                denominator[(dimension, str(group))] += int(count)
            for target in TARGETS:
                valid = frame[TARGET_VALID_COLUMNS[target]].to_numpy(bool)
                target_counts = groups.loc[valid].value_counts()
                for group, count in target_counts.items():
                    valid_counts[(target, dimension, str(group))] += int(count)
        _counter_update(direct_raw_group, frame["raw_sparsity_group"].astype(str))
        day_row: dict[str, Any] = {
            "date": date, "direct_observed_traversals": len(frame)
        }
        for target in TARGETS:
            valid = frame[TARGET_VALID_COLUMNS[target]].to_numpy(bool)
            day_row[f"{target}_valid"] = int(valid.sum())
            _counter_update(target_time[target], time_bin.loc[valid])
            _counter_update(target_origin[target], frame.loc[valid, "origin_grid"].astype(str))
            _counter_update(
                target_raw_group[target], frame.loc[valid, "raw_sparsity_group"].astype(str)
            )
        daily_target_rows.append(day_row)
        identity_count += len(frame)

    label_rows: list[dict[str, Any]] = []
    for dimension in ("origin_time_support", "edge_spatial_support", "edge_time_support"):
        for group in SUPPORT_GROUPS:
            den = denominator[(dimension, group)]
            for target in TARGETS:
                count = valid_counts[(target, dimension, group)]
                label_rows.append({
                    "target": target,
                    "context_dimension": dimension,
                    "context_group": group,
                    "direct_observed_traversals": den,
                    "target_valid_traversals": count,
                    "availability_rate": count / den if den else None,
                })
    daily_comparison_rows: list[dict[str, Any]] = []
    for date in sorted({_partition_value(path, "date") for path in inputs["labels"]}):
        for target in TARGETS:
            for label in ("rare", "common"):
                daily_comparison_rows.append({
                    "date": date,
                    "target": target,
                    "comparison_group": label,
                    "direct_observed": daily_label_denominator[(date, label)],
                    "target_valid": daily_label_valid[(date, target, label)],
                })
    link_traversal_count = int(sum(
        pq.ParquetFile(path.parent / "link_traversals.parquet").metadata.num_rows
        for path in inputs["accepted"]
    ))
    output_manifest_label_rows = 0
    for path in inputs["output_manifests"]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("engineering_status") != "PASS":
            raise Stage2V52ContractError(f"Stage 1 output bucket did not pass: {path}")
        product_rows = payload.get("product_rows", payload.get("product_row_counts", {}))
        if "traversal_labels" in product_rows:
            output_manifest_label_rows += int(product_rows["traversal_labels"])
    if output_manifest_label_rows and output_manifest_label_rows != identity_count:
        raise Stage2V52ContractError(
            "Stage 1 traversal-label manifest counts do not reconcile: "
            f"{output_manifest_label_rows} != {identity_count}"
        )
    return {
        "label_attrition": pd.DataFrame(label_rows),
        "daily_target": pd.DataFrame(daily_target_rows).groupby("date", as_index=False).sum(),
        "daily_label_comparison": pd.DataFrame(daily_comparison_rows),
        "target_time": target_time,
        "target_origin": target_origin,
        "target_raw_group": target_raw_group,
        "direct_raw_group": direct_raw_group,
        "edge_quantiles": edge_quantiles,
        "train_direct_observed_traversals": train_label_rows,
        "link_traversal_count": link_traversal_count,
        "direct_observed_traversal_count": identity_count,
        "output_manifest_label_rows": output_manifest_label_rows,
        "missing_accepted_order_context_count": invalid_context,
    }


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    return float(numerator / denominator) if denominator else None


def _distribution_distance(
    left: Mapping[Any, int], right: Mapping[Any, int]
) -> tuple[float, float]:
    keys = sorted(set(left) | set(right), key=str)
    p = probability_distribution(left, keys)
    q = probability_distribution(right, keys)
    return jensen_shannon_divergence(p, q), total_variation(p, q)


def _build_tables(
    config: Mapping[str, Any], raw: Mapping[str, Any],
    stage1: Mapping[str, Any], rejection_table: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    stage_time = raw["stage_time"]
    stage_origin = raw["stage_origin"]
    stage_group = raw["stage_group"]
    target_time = stage1["target_time"]
    target_origin = stage1["target_origin"]
    target_group = stage1["target_raw_group"]
    tables: dict[str, pd.DataFrame] = {}
    tables["table_1_daily_funnel"] = raw["daily"].copy()

    time_rows: list[dict[str, Any]] = []
    time_totals = {stage: sum(values.values()) for stage, values in stage_time.items()}
    target_time_totals = {target: sum(values.values()) for target, values in target_time.items()}
    for time_bin in range(48):
        for target in TARGETS:
            time_rows.append({
                "time_bin": time_bin,
                "target": target,
                "raw_count": stage_time["raw"][time_bin],
                "raw_share": _safe_rate(stage_time["raw"][time_bin], time_totals["raw"]),
                "processed_count": stage_time["processed"][time_bin],
                "processed_share": _safe_rate(stage_time["processed"][time_bin], time_totals["processed"]),
                "accepted_count": stage_time["accepted"][time_bin],
                "accepted_share": _safe_rate(stage_time["accepted"][time_bin], time_totals["accepted"]),
                "target_valid_traversals": target_time[target][time_bin],
                "target_valid_share": _safe_rate(
                    target_time[target][time_bin], target_time_totals[target]
                ),
                "acceptance_rate": _safe_rate(
                    stage_time["accepted"][time_bin], stage_time["raw"][time_bin]
                ),
            })
    tables["table_2_time_bin_representativeness"] = pd.DataFrame(time_rows)

    minimum = int(config["spatial_grid"]["minimum_raw_count_for_cell_rate"])
    origin_rows: list[dict[str, Any]] = []
    raw_origin_total = sum(stage_origin["raw"].values())
    accepted_origin_total = sum(stage_origin["accepted"].values())
    for grid in sorted(stage_origin["raw"]):
        raw_count = stage_origin["raw"][grid]
        if raw_count < minimum:
            continue
        accepted_count = stage_origin["accepted"][grid]
        origin_rows.append({
            "origin_grid": grid,
            "raw_count": raw_count,
            "raw_share": raw_count / raw_origin_total,
            "processed_count": stage_origin["processed"][grid],
            "accepted_count": accepted_count,
            "accepted_share": accepted_count / accepted_origin_total,
            "acceptance_rate": accepted_count / raw_count,
        })
    tables["table_3_spatial_representativeness"] = pd.DataFrame(origin_rows)

    retention_rows: list[dict[str, Any]] = []
    for group in SUPPORT_GROUPS:
        raw_count = stage_group["raw"][group]
        processed_count = stage_group["processed"][group]
        accepted_count = stage_group["accepted"][group]
        for target in TARGETS:
            valid_count = target_group[target][group]
            retention_rows.append({
                "support_group": group,
                "target": target,
                "raw_orders": raw_count,
                "processed_orders": processed_count,
                "accepted_orders": accepted_count,
                "target_valid_traversals": valid_count,
                "R1_processed_per_raw": _safe_rate(processed_count, raw_count),
                "R2_accepted_per_processed": _safe_rate(accepted_count, processed_count),
                "R3_target_valid_traversals_per_accepted_order": _safe_rate(
                    valid_count, accepted_count
                ),
            })
    tables["table_4_rare_context_retention"] = pd.DataFrame(retention_rows)
    tables["table_5_rejection_reasons_by_sparse_group"] = rejection_table.copy()
    tables["table_6_target_specific_label_attrition"] = stage1["label_attrition"].copy()

    distance_rows: list[dict[str, Any]] = []
    for dimension, left, right in (
        ("time_bin", stage_time["raw"], stage_time["accepted"]),
        ("origin_grid", stage_origin["raw"], stage_origin["accepted"]),
    ):
        jsd, tv = _distribution_distance(left, right)
        distance_rows.append({
            "dimension": dimension, "target": "orders",
            "comparison": "raw_vs_accepted", "jensen_shannon_divergence": jsd,
            "total_variation_distance": tv,
        })
    for target in TARGETS:
        for dimension, left, right in (
            ("time_bin", stage_time["accepted"], target_time[target]),
            ("origin_grid", stage_origin["accepted"], target_origin[target]),
        ):
            jsd, tv = _distribution_distance(left, right)
            distance_rows.append({
                "dimension": dimension, "target": target,
                "comparison": "accepted_orders_vs_target_valid_traversals",
                "jensen_shannon_divergence": jsd,
                "total_variation_distance": tv,
            })
    tables["table_7_distribution_distance"] = pd.DataFrame(distance_rows)

    bootstrap = config["bootstrap"]
    effects: list[dict[str, Any]] = []
    for comparison, numerator, denominator in (
        ("processing_opportunity", "processed", "raw"),
        ("prefilter_pass_given_processed", "prefilter", "processed"),
        ("core_acceptance_given_raw", "accepted", "raw"),
        ("core_acceptance_given_processed", "accepted", "processed"),
    ):
        effect = cluster_bootstrap_rate_effect(
            raw["comparison_daily"], numerator=numerator,
            denominator=denominator, replicates=int(bootstrap["replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]),
        )
        effects.append({"comparison": comparison, **effect.__dict__})
    tables["table_8_rare_context_acceptance_effects"] = pd.DataFrame(effects)
    tables["selection_rank_by_context"] = raw["selection_rank"].copy()
    tables["daily_rare_common_funnel"] = raw["comparison_daily"].copy()
    tables["daily_target_valid_counts"] = stage1["daily_target"].copy()
    summary = {
        "raw_time_total": time_totals["raw"],
        "processed_time_total": time_totals["processed"],
        "accepted_time_total": time_totals["accepted"],
        "target_valid_totals": target_time_totals,
        "distribution_distance": distance_rows,
        "rare_context_effects": effects,
    }
    return tables, summary


def _classification(
    config: Mapping[str, Any], raw: Mapping[str, Any],
    stage1: Mapping[str, Any], table_summary: Mapping[str, Any],
) -> dict[str, Any]:
    rare = set(config["support"]["rare_groups"])
    stage_group = raw["stage_group"]
    target_group = stage1["target_raw_group"]
    raw_total = sum(stage_group["raw"].values())
    accepted_total = sum(stage_group["accepted"].values())
    raw_rare = sum(stage_group["raw"][group] for group in rare)
    accepted_rare = sum(stage_group["accepted"][group] for group in rare)
    raw_rare_share = raw_rare / raw_total
    accepted_rare_share = accepted_rare / accepted_total
    effects = {
        item["comparison"]: item for item in table_summary["rare_context_effects"]
    }
    quality = effects["core_acceptance_given_processed"]
    thresholds = config["classification_thresholds"]
    material_ratio = float(thresholds["minimum_material_rate_ratio"])
    material_gap = float(thresholds["minimum_material_percentage_point_gap"])
    stage0_effect = material_negative_rate_effect(
        quality, maximum_rate_ratio=material_ratio,
        minimum_absolute_gap=material_gap,
    )
    label_effects: dict[str, Any] = {}
    supervision_targets = 0
    label_effect_rows: list[dict[str, Any]] = []
    bootstrap = config["bootstrap"]
    for target in TARGETS:
        target_daily = stage1["daily_label_comparison"].loc[
            stage1["daily_label_comparison"]["target"].eq(target)
        ]
        effect = cluster_bootstrap_rate_effect(
            target_daily, numerator="target_valid", denominator="direct_observed",
            replicates=int(bootstrap["replicates"]),
            confidence_level=float(bootstrap["confidence_level"]),
            seed=int(bootstrap["seed"]),
        )
        effect_payload = effect.__dict__
        material = material_negative_rate_effect(
            effect, maximum_rate_ratio=material_ratio,
            minimum_absolute_gap=material_gap,
        )
        supervision_targets += int(material)
        label_effects[target] = {
            "rare_direct_observed_traversals": effect.rare_denominator,
            "high_direct_observed_traversals": effect.common_denominator,
            "rare_target_valid_traversals": effect.rare_count,
            "high_target_valid_traversals": effect.common_count,
            "rare_availability_rate": effect.rare_rate,
            "high_availability_rate": effect.common_rate,
            "percentage_point_difference": effect.percentage_point_difference,
            "rate_ratio": effect.rate_ratio,
            "difference_ci_low": effect.difference_ci_low,
            "difference_ci_high": effect.difference_ci_high,
            "ratio_ci_low": effect.ratio_ci_low,
            "ratio_ci_high": effect.ratio_ci_high,
            "material_attrition": material,
        }
        label_effect_rows.append({
            "target": target, **effect_payload, "material_attrition": material,
        })
    stage1_effect = supervision_targets >= int(
        thresholds["minimum_targets_for_supervision_attrition"]
    )
    demand_concentrated = raw_rare_share <= float(
        thresholds["raw_rare_share_max_for_concentrated_demand"]
    )
    classification, conclusion = classify_upstream(
        demand_concentrated=demand_concentrated,
        stage0_quality_effect=stage0_effect,
        stage1_supervision_effect=stage1_effect,
    )
    target_rare_share = {
        target: _safe_rate(
            sum(target_group[target][group] for group in rare),
            sum(target_group[target].values()),
        )
        for target in TARGETS
    }
    return {
        "classification": classification,
        "conclusion": conclusion,
        "demand_concentrated": demand_concentrated,
        "stage0_quality_selection_material": stage0_effect,
        "stage1_supervision_attrition_material": stage1_effect,
        "material_supervision_target_count": supervision_targets,
        "raw_rare_order_count": raw_rare,
        "raw_rare_order_share": raw_rare_share,
        "accepted_rare_order_count": accepted_rare,
        "accepted_rare_order_share": accepted_rare_share,
        "target_valid_rare_traversal_share": target_rare_share,
        "stage0_quality_effect": quality,
        "stage1_label_effects": label_effects,
        "stage1_label_effect_rows": label_effect_rows,
        "sampling_hash_representativeness": raw["sampling_hash_status"],
        "maximum_broad_selection_rank_mean_gap": raw["maximum_broad_rank_mean_gap"],
    }


def _grid_coordinates(value: str) -> tuple[int, int] | None:
    if value == "MISSING":
        return None
    try:
        east_text, north_text = value.split("_N", 1)
        east = int(east_text.split("E", 1)[1])
        north = int(north_text)
        return east, north
    except (IndexError, ValueError):
        return None


def _save_figure(path: Path, figure: plt.Figure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.png")
    figure.savefig(temporary, dpi=160, bbox_inches="tight")
    plt.close(figure)
    temporary.replace(path)


def _make_figures(
    output: Path, tables: Mapping[str, pd.DataFrame]
) -> dict[str, Path]:
    figures = output / "figures"
    result: dict[str, Path] = {}
    time = tables["table_2_time_bin_representativeness"]
    base = time.loc[time["target"].eq(TARGETS[0])]

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.plot(base["time_bin"], base["raw_share"], label="Raw", linewidth=2)
    axis.plot(base["time_bin"], base["accepted_share"], label="Stage 0 accepted", linewidth=2)
    axis.set(xlabel="Xi'an local 30-minute bin", ylabel="Share", title="Raw vs accepted temporal distribution")
    axis.grid(alpha=0.25)
    axis.legend()
    path = figures / "raw_vs_accepted_48bin_temporal_distribution.png"
    _save_figure(path, fig)
    result["raw_vs_accepted_48bin_temporal_distribution"] = path

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(base["time_bin"], base["acceptance_rate"], width=0.85, color="#4C78A8")
    axis.set(xlabel="Xi'an local 30-minute bin", ylabel="Accepted / raw", title="Stage 0 acceptance rate by time bin")
    axis.grid(axis="y", alpha=0.25)
    path = figures / "acceptance_rate_by_time_bin.png"
    _save_figure(path, fig)
    result["acceptance_rate_by_time_bin"] = path

    spatial = tables["table_3_spatial_representativeness"].copy()
    coordinates = spatial["origin_grid"].astype(str).map(_grid_coordinates)
    spatial = spatial.loc[coordinates.notna()].copy()
    pairs = np.asarray(coordinates.loc[coordinates.notna()].tolist(), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for axis, column, title in zip(
        axes, ("raw_count", "accepted_count"), ("Raw candidates", "Stage 0 accepted")
    ):
        color = np.log1p(spatial[column].to_numpy(float))
        scatter = axis.scatter(pairs[:, 0], pairs[:, 1], c=color, s=18, cmap="viridis", marker="s")
        axis.set(title=title, xlabel="UTM 1 km easting cell", ylabel="UTM 1 km northing cell")
        fig.colorbar(scatter, ax=axis, label="log(1 + count)")
    fig.suptitle("Origin-grid density: raw versus accepted")
    path = figures / "raw_vs_accepted_origin_grid_density.png"
    _save_figure(path, fig)
    result["raw_vs_accepted_origin_grid_density"] = path

    retention = tables["table_4_rare_context_retention"]
    rare = retention.loc[
        retention["support_group"].isin(["unseen", "low"])
    ].groupby("target", as_index=False).agg(
        raw_orders=("raw_orders", "sum"),
        processed_orders=("processed_orders", "sum"),
        accepted_orders=("accepted_orders", "sum"),
        target_valid_traversals=("target_valid_traversals", "sum"),
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(TARGETS), dtype=float)
    width = 0.2
    for offset, column, label in (
        (-1.5, "raw_orders", "Raw orders"),
        (-0.5, "processed_orders", "Processed orders"),
        (0.5, "accepted_orders", "Accepted orders"),
        (1.5, "target_valid_traversals", "Target-valid traversals"),
    ):
        values = rare.set_index("target").reindex(TARGETS)[column].to_numpy(float)
        axis.bar(x + offset * width, values, width, label=label)
    axis.set_yscale("log")
    axis.set_xticks(x, TARGETS, rotation=15)
    axis.set(ylabel="Count (log scale)", title="Rare-context funnel")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    path = figures / "rare_context_funnel.png"
    _save_figure(path, fig)
    result["rare_context_funnel"] = path

    rejection = tables["table_5_rejection_reasons_by_sparse_group"].set_index("support_group").reindex(SUPPORT_GROUPS)
    fig, axis = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(SUPPORT_GROUPS), dtype=float)
    for mechanism in REJECTION_GROUPS:
        values = rejection[mechanism].fillna(0).to_numpy(float)
        axis.bar(SUPPORT_GROUPS, values, bottom=bottom, label=mechanism)
        bottom += values
    axis.set(ylabel="Attributed rejection count", title="Rejection mechanisms by raw support group")
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    path = figures / "rejection_reason_composition_by_sparse_group.png"
    _save_figure(path, fig)
    result["rejection_reason_composition_by_sparse_group"] = path

    labels = tables["table_6_target_specific_label_attrition"]
    labels = labels.loc[labels["context_dimension"].eq("origin_time_support")]
    fig, axis = plt.subplots(figsize=(10, 5))
    x = np.arange(len(SUPPORT_GROUPS), dtype=float)
    width = 0.19
    for index, target in enumerate(TARGETS):
        values = labels.loc[labels["target"].eq(target)].set_index("context_group").reindex(SUPPORT_GROUPS)["availability_rate"].to_numpy(float)
        axis.bar(x + (index - 1.5) * width, values, width, label=target)
    axis.set_xticks(x, SUPPORT_GROUPS)
    axis.set(ylabel="Target-valid / direct-observed", title="Stage 1 target-valid rate by raw support group")
    axis.set_ylim(0, 1.05)
    axis.legend(fontsize=8)
    axis.grid(axis="y", alpha=0.2)
    path = figures / "target_valid_rate_by_sparse_group.png"
    _save_figure(path, fig)
    result["target_valid_rate_by_sparse_group"] = path
    return result


def _report_markdown(report: Mapping[str, Any]) -> str:
    classification = report["final_classification"]
    counts = report["counts"]
    effect = classification["stage0_quality_effect"]
    lines = [
        "# Stage 0/1 upstream sampling representativeness audit",
        "",
        f"- Audit status: `{report['audit_status']}`",
        f"- Raw candidate coverage: `{report['raw_candidate_coverage']}`",
        f"- Final classification: `{classification['classification']}`",
        f"- Conclusion: `{classification['conclusion']}`",
        f"- Sampling-hash representativeness: `{classification['sampling_hash_representativeness']}`",
        "- Stage 0/1 rerun, model training and inference: `NO / NO / NO`",
        "",
        "## Frozen scope",
        "",
        "- Raw support fit: 20161009–20161024 only.",
        "- Evaluation coverage: 20161025–20161027 and 20161031.",
        "- Time: Asia/Shanghai, 48 half-hour bins.",
        "- Space: fixed EPSG:32649 1 km × 1 km grid.",
        "- Primary raw sparse context: origin-grid × departure-time-bin.",
        "",
        "## Funnel",
        "",
        f"- Raw candidates: `{counts['raw_candidates']:,}`",
        f"- Processed opportunities: `{counts['processed_candidates']:,}`",
        f"- Stage 0 accepted orders: `{counts['accepted_orders']:,}`",
        f"- Stage 0 rejected orders: `{counts['rejected_orders']:,}`",
        f"- Quota-unprocessed candidates: `{counts['unprocessed_quota']:,}`",
        f"- Link traversals: `{counts['link_traversals']:,}`",
        f"- Direct-observed traversals: `{counts['direct_observed_traversals']:,}`",
        "",
        "## Mechanism separation",
        "",
        f"- Raw rare-context share: `{classification['raw_rare_order_share']:.4%}`",
        f"- Accepted rare-context share: `{classification['accepted_rare_order_share']:.4%}`",
        f"- Rare/common conditional Stage 0 acceptance: `{effect['rare_rate']:.4%}` vs `{effect['common_rate']:.4%}` "
        f"(difference `{effect['percentage_point_difference']:.4%}`, ratio `{effect['rate_ratio']:.4f}`).",
        f"- Stage 0 quality-selection effect material: `{classification['stage0_quality_selection_material']}`",
        f"- Stage 1 supervision attrition material: `{classification['stage1_supervision_attrition_material']}`",
        "",
        "### Stage 1 target-valid availability in rare versus high raw contexts",
        "",
        "| target | rare rate | high rate | pp difference | rate ratio | material |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for target in TARGETS:
        item = classification["stage1_label_effects"][target]
        lines.append(
            f"| {target} | {item['rare_availability_rate']:.4%} | "
            f"{item['high_availability_rate']:.4%} | "
            f"{item['percentage_point_difference']:.4%} | "
            f"{item['rate_ratio']:.4f} | {item['material_attrition']} |"
        )
    lines.extend([
        "",
        "## Distribution preservation",
        "",
        "| dimension | target | comparison | JSD | TV |",
        "|---|---|---|---:|---:|",
    ])
    for item in report["distribution_distance"]:
        lines.append(
            f"| {item['dimension']} | {item['target']} | {item['comparison']} | "
            f"{item['jensen_shannon_divergence']:.6f} | {item['total_variation_distance']:.6f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "The audit is descriptive and does not estimate a causal selection effect. "
        "Raw demand concentration, hash/quota opportunity, Stage 0 quality selection, "
        "and Stage 1 supervision attrition are reported separately.",
        "",
        "## Stop state",
        "",
        "`SPARSITY_STRESS_TEST_AUTHORIZED=NO`",
        "",
        "`TRANSFER_V2_AUTHORIZED=NO`",
        "",
        "`PHASE_D_AUTHORIZED=NO`",
        "",
        "`STAGE3_AUTHORIZED=NO`",
        "",
    ])
    return "\n".join(lines)


def _build_evidence(
    root: Path, config_path: Path, config: Mapping[str, Any],
    inputs: Mapping[str, Any], report_paths: Mapping[str, Path],
    table_paths: Mapping[str, Path], figure_paths: Mapping[str, Path],
    execution_head: str,
) -> dict[str, Any]:
    stage0_code = sorted((root / config["paths"]["stage0_code"]).glob("*.py"))
    accepted_products = list(inputs["accepted"])
    payload = {
        "schema_version": EVIDENCE_SCHEMA,
        "status": "PASS",
        "execution_head": execution_head,
        "frozen_base_commit": config["frozen_base_commit"],
        "audit_scope": config["audit_scope"],
        "inputs": {
            "stage0_frozen_config": _descriptor(
                root / config["paths"]["stage0_config"], root
            ),
            "stage0_production_code": _descriptor_set(stage0_code, root),
            "candidate_manifests": _descriptor_set(
                list(inputs["candidate"].values()), root
            ),
            "rejection_manifests": _descriptor_set(inputs["rejection"], root),
            "accepted_order_products": _descriptor_set(accepted_products, root),
            "accepted_order_bucket_manifests": _descriptor_set(
                inputs["input_manifests"], root
            ),
            "stage1_release_manifest": _descriptor(
                root / config["paths"]["stage1_release_manifest"], root
            ),
            "stage1_scientific_review": {
                "json": _descriptor(
                    root / config["paths"]["stage1_scientific_review_json"], root
                ),
                "markdown": _descriptor(
                    root / config["paths"]["stage1_scientific_review_markdown"], root
                ),
            },
            "stage2_sparsity_diagnostic_evidence": _descriptor(
                root / config["paths"]["stage2_sparsity_evidence"], root
            ),
        },
        "audit_implementation": {
            "config": _descriptor(config_path, root),
            "runner": _descriptor(Path(__file__), root),
            "support": _descriptor(Path(__file__).with_name("upstream_sampling_support.py"), root),
        },
        "outputs": {
            "reports": {
                name: _descriptor(path, root) for name, path in report_paths.items()
            },
            "tables": {
                name: _descriptor(path, root) for name, path in sorted(table_paths.items())
            },
            "figures": {
                name: _descriptor(path, root) for name, path in sorted(figure_paths.items())
            },
        },
        "bootstrap": dict(config["bootstrap"]),
        "grid": dict(config["spatial_grid"]),
        "time_bin": dict(config["time_bin"]),
        "authorizations": dict(config["authorizations"]),
    }
    if any(bool(value) for value in payload["authorizations"].values()):
        raise Stage2V52ContractError("evidence cannot authorize downstream execution")
    return payload


def _verify_descriptor(item: Mapping[str, Any], root: Path) -> None:
    path = root / str(item["path"])
    if not path.is_file():
        raise Stage2V52ContractError(f"bound artifact is missing: {path}")
    if path.stat().st_size != int(item["bytes"]):
        raise Stage2V52ContractError(f"bound artifact size changed: {path}")
    if _sha256_file(path) != item["sha256"]:
        raise Stage2V52ContractError(f"bound artifact hash changed: {path}")


def _verify_descriptor_set(item: Mapping[str, Any], root: Path) -> None:
    files = list(item["files"])
    for descriptor in files:
        _verify_descriptor(descriptor, root)
    digest = hashlib.sha256()
    for descriptor in files:
        digest.update(descriptor["path"].encode("utf-8"))
        digest.update(bytes.fromhex(descriptor["sha256"]))
    if len(files) != int(item["file_count"]) or digest.hexdigest() != item["set_sha256"]:
        raise Stage2V52ContractError("bound artifact-set digest changed")


def verify_evidence_bundle(
    payload: Mapping[str, Any], *, repo_root: str | Path
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    if payload.get("schema_version") != EVIDENCE_SCHEMA or payload.get("status") != "PASS":
        raise Stage2V52ContractError("invalid upstream audit evidence schema/status")
    inputs = payload["inputs"]
    _verify_descriptor(inputs["stage0_frozen_config"], root)
    for key in (
        "stage0_production_code", "candidate_manifests", "rejection_manifests",
        "accepted_order_products", "accepted_order_bucket_manifests",
    ):
        _verify_descriptor_set(inputs[key], root)
    _verify_descriptor(inputs["stage1_release_manifest"], root)
    _verify_descriptor(inputs["stage1_scientific_review"]["json"], root)
    _verify_descriptor(inputs["stage1_scientific_review"]["markdown"], root)
    _verify_descriptor(inputs["stage2_sparsity_diagnostic_evidence"], root)
    for descriptor in payload["audit_implementation"].values():
        _verify_descriptor(descriptor, root)
    for section in payload["outputs"].values():
        for descriptor in section.values():
            _verify_descriptor(descriptor, root)
    if any(bool(value) for value in payload["authorizations"].values()):
        raise Stage2V52ContractError("evidence authorizations are not all false")
    return {
        "status": "PASS",
        "execution_head": payload["execution_head"],
        "candidate_manifest_count": inputs["candidate_manifests"]["file_count"],
        "rejection_manifest_count": inputs["rejection_manifests"]["file_count"],
        "accepted_product_count": inputs["accepted_order_products"]["file_count"],
        "table_count": len(payload["outputs"]["tables"]),
        "figure_count": len(payload["outputs"]["figures"]),
        "sparsity_stress_test_authorized": False,
        "transfer_v2_authorized": False,
        "phase_d_authorized": False,
        "stage3_authorized": False,
    }


def run_audit(
    *, repo_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(repo_root).resolve()
    config_source = (root / config_path).resolve() if not Path(config_path).is_absolute() else Path(config_path).resolve()
    config = _load_config(config_source)
    execution_head = _git_head(root)
    if execution_head != config["frozen_base_commit"]:
        raise Stage2V52ContractError(
            f"audit must start from frozen base {config['frozen_base_commit']}, got {execution_head}"
        )
    inputs = _discover_inputs(root, config)
    output = (root / config["paths"]["output"]).resolve()
    allowed_output = (root / "stage2/docs/v5_2/upstream_sampling_audit").resolve()
    if output != allowed_output:
        raise Stage2V52ContractError(f"refusing output outside frozen audit directory: {output}")
    protected = [
        (root / config["paths"]["candidate_manifests"]).resolve(),
        inputs["input_root"].resolve(), inputs["output_root"].resolve(),
    ]
    production_before = _production_snapshot(protected, root)
    transformer = Transformer.from_crs(
        "EPSG:4326", config["spatial_grid"]["crs"], always_xy=True
    )

    raw_counts, raw_quantiles, train_raw_rows = _fit_raw_support(
        inputs, config, transformer
    )
    accepted, rejected, stage0_identity_audit = _load_stage0_outcomes(root, inputs)
    raw = _audit_raw_funnel(
        inputs, config, transformer, raw_counts, raw_quantiles,
        accepted, rejected,
    )
    rejection_table, rejection_tokens, rejection_audit = _audit_rejections(
        rejected, raw["processed_context"]
    )
    stage1 = _audit_stage1_labels(inputs, config, raw["accepted_context"])
    tables, table_summary = _build_tables(config, raw, stage1, rejection_table)
    tables["rejection_token_decomposition"] = rejection_tokens
    tables["raw_support_quantiles"] = pd.DataFrame([
        {"support_dimension": name, "train_cell_count": len(raw_counts[name]), **values}
        for name, values in raw_quantiles.items()
    ])
    tables["stage1_edge_support_quantiles"] = pd.DataFrame([
        {"support_dimension": name, **values}
        for name, values in stage1["edge_quantiles"].items()
    ])
    classification = _classification(config, raw, stage1, table_summary)
    tables["stage1_target_valid_rare_effects"] = pd.DataFrame(
        classification["stage1_label_effect_rows"]
    )
    tables["selection_rank_decision_summary"] = raw[
        "selection_rank_decision_summary"
    ].copy()

    raw_total = int(raw["daily"]["raw_candidates"].sum())
    processed_total = int(raw["daily"]["processed_candidates"].sum())
    accepted_total = int(raw["daily"]["core_accepted"].sum())
    rejected_total = int(raw["daily"]["rejected"].sum())
    validate_funnel_identity(
        raw=raw_total, processed=processed_total, accepted=accepted_total,
        rejected=rejected_total,
        unprocessed_quota=int(raw["daily"]["unprocessed_quota"].sum()),
    )
    if accepted_total != len(accepted) or rejected_total != len(rejected):
        raise Stage2V52ContractError("Stage 0 outcome totals do not reconcile identities")

    table_paths: dict[str, Path] = {}
    for name, frame in tables.items():
        path = output / "tables" / f"{name}.csv"
        _write_csv(path, frame)
        table_paths[name] = path
    figure_paths = _make_figures(output, tables)
    distance = table_summary["distribution_distance"]
    interpretations = {
        "UP-A": "Raw ride-hailing demand is already concentrated in common origin-time contexts; the frozen Stage 0/1 pipeline does not show a material additional rare-context loss under the preregistered thresholds.",
        "UP-B": "Hash/quota opportunity is approximately context-neutral, but Stage 0 quality selection materially lowers rare-context acceptance. Preserve the demand-weighted 220k benchmark; any future rare-context set must be an independent stress test.",
        "UP-C": "Raw-to-accepted context is broadly preserved, while Stage 1 target-valid availability materially drops in rare contexts. Do not rebuild Stage 0; a future stress test should prioritize label-valid rare traversals.",
        "UP-D": "Raw demand concentration, Stage 0 quality selection, and Stage 1 label availability jointly shape sparse-context representation; no single stage explains the observed sparsity.",
    }
    report: dict[str, Any] = {
        "schema_version": REPORT_SCHEMA,
        "audit_status": "PASS",
        "raw_candidate_coverage": "FULL",
        "execution_head": execution_head,
        "protocol": {
            "train_dates": list(config["dates"]["train"]),
            "validation_dates": list(config["dates"]["validation"]),
            "test_dates": list(config["dates"]["test"]),
            "train_only_raw_support_row_count": train_raw_rows,
            "time_bin": dict(config["time_bin"]),
            "spatial_grid": dict(config["spatial_grid"]),
            "primary_raw_context": config["support"]["primary_raw_context"],
        },
        "counts": {
            "raw_candidates": raw_total,
            "processed_candidates": processed_total,
            "prefilter_pass": int(raw["daily"]["prefilter_pass"].sum()),
            "valhalla_processed_or_matched": int(raw["daily"]["valhalla_processed_or_matched"].sum()),
            "accepted_orders": accepted_total,
            "rejected_orders": rejected_total,
            "unprocessed_quota": raw_total - processed_total,
            "link_traversals": stage1["link_traversal_count"],
            "direct_observed_traversals": stage1["direct_observed_traversal_count"],
            "target_valid_traversals": table_summary["target_valid_totals"],
            "missing_raw_time": raw["missing_time_count"],
            "missing_raw_origin_grid": raw["missing_spatial_count"],
        },
        "identity_audit": stage0_identity_audit,
        "rejection_audit": rejection_audit,
        "raw_support_quantiles": raw_quantiles,
        "stage1_edge_support_quantiles": stage1["edge_quantiles"],
        "selection_opportunity": {
            "selection_hash_inputs": list(config["selection"]["hash_inputs"]),
            "selection_hash_recomputed_for_all_candidates": True,
            "sampling_hash_representativeness": raw["sampling_hash_status"],
            "maximum_broad_selection_rank_mean_gap": raw["maximum_broad_rank_mean_gap"],
            "quota_unprocessed_is_rejected": False,
            "effects": table_summary["rare_context_effects"],
        },
        "stage1_supervision": {
            "frozen_mask_source": "stage2.v4.stage1_adapter._add_component_masks equivalent component rules",
            "train_direct_observed_traversals": stage1["train_direct_observed_traversals"],
            "direct_observed_traversals": stage1["direct_observed_traversal_count"],
            "target_valid_totals": table_summary["target_valid_totals"],
            "missing_accepted_order_context_count": stage1["missing_accepted_order_context_count"],
        },
        "distribution_distance": distance,
        "optional_context_reweighting": {
            "status": "SKIP",
            "reason": "Raw order-grid contexts and Stage 2 edge-time traversal contexts have different units; a defensible direct mapping is not available without expanding engineering scope.",
        },
        "final_classification": classification,
        "interpretation": interpretations[classification["classification"]],
        "authorizations": dict(config["authorizations"]),
        "runtime_s": time.perf_counter() - started,
    }
    report_json = output / "stage0_stage1_upstream_representativeness_report.json"
    report_md = output / "stage0_stage1_upstream_representativeness_report.md"
    _write_json(report_json, report)
    _atomic_text(report_md, _report_markdown(report))

    report_paths = {"json": report_json, "markdown": report_md}
    evidence = _build_evidence(
        root, config_source, config, inputs, report_paths, table_paths,
        figure_paths, execution_head,
    )
    evidence_path = output / "stage0_stage1_upstream_representativeness_evidence_bundle.json"
    _write_json(evidence_path, evidence)
    verification = verify_evidence_bundle(evidence, repo_root=root)
    production_after = _production_snapshot(protected, root)
    if production_before != production_after:
        raise Stage2V52ContractError(
            "Stage 0/1 production roots changed during read-only audit"
        )
    return {
        "status": "PASS",
        "classification": classification["classification"],
        "conclusion": classification["conclusion"],
        "raw_candidate_coverage": "FULL",
        "raw_candidates": raw_total,
        "processed_candidates": processed_total,
        "accepted_orders": accepted_total,
        "evidence_sha256": _sha256_file(evidence_path),
        "verification": verification,
        "authorizations": dict(config["authorizations"]),
    }


def refresh_evidence(
    *, repo_root: str | Path, config_path: str | Path
) -> dict[str, Any]:
    """Rebind completed deterministic outputs after provenance-only code review."""

    root = Path(repo_root).resolve()
    config_source = (
        (root / config_path).resolve()
        if not Path(config_path).is_absolute()
        else Path(config_path).resolve()
    )
    config = _load_config(config_source)
    execution_head = _git_head(root)
    if execution_head != config["frozen_base_commit"]:
        raise Stage2V52ContractError(
            f"evidence refresh requires frozen base {config['frozen_base_commit']}"
        )
    inputs = _discover_inputs(root, config)
    output = (root / config["paths"]["output"]).resolve()
    report_paths = {
        "json": output / "stage0_stage1_upstream_representativeness_report.json",
        "markdown": output / "stage0_stage1_upstream_representativeness_report.md",
    }
    table_paths = {
        path.stem: path for path in sorted((output / "tables").glob("*.csv"))
    }
    figure_paths = {
        path.stem: path for path in sorted((output / "figures").glob("*.png"))
    }
    if len(table_paths) < 8 or len(figure_paths) < 6:
        raise Stage2V52ContractError(
            "evidence-only refresh requires a completed audit output set"
        )
    evidence = _build_evidence(
        root, config_source, config, inputs, report_paths, table_paths,
        figure_paths, execution_head,
    )
    evidence_path = output / "stage0_stage1_upstream_representativeness_evidence_bundle.json"
    _write_json(evidence_path, evidence)
    verification = verify_evidence_bundle(evidence, repo_root=root)
    return {
        "status": "PASS", "mode": "EVIDENCE_ONLY",
        "evidence_sha256": _sha256_file(evidence_path),
        "verification": verification,
        "authorizations": dict(config["authorizations"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only Stage 0/1 sampling representativeness audit"
    )
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--config",
        default="stage2/config/stage0_stage1_upstream_representativeness_audit.json",
    )
    parser.add_argument(
        "--evidence-only", action="store_true",
        help="Rebind existing deterministic outputs after provenance-only review",
    )
    args = parser.parse_args(argv)
    runner = refresh_evidence if args.evidence_only else run_audit
    result = runner(repo_root=args.repo_root, config_path=args.config)
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
