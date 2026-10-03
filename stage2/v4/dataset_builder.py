"""Build the Stage 2 v4 route-conditioned daily products."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .config import Stage2V4Config
from .contracts import (
    COMPONENT_MASKS,
    FORBIDDEN_FORMAL_TARGETS,
    ROUTE_PRIMARY_KEY,
    Stage2V4ContractError,
)
from .entry_time import _local_time_fields, estimate_entry_times
from .history_index import TemporalHistoryIndex
from .history_store import HISTORY_VALUE_COLUMNS, validate_history_store
from .io import (
    atomic_write_json,
    sha256_file,
    stage2_v4_code_identity,
)
from .stage1_adapter import (
    Stage1BucketRef,
    build_route_alignment,
    discover_stage1_buckets,
)


DATASET_SCHEMA_VERSION = "stage2_v4_route_conditioned_dataset.1"
DATASET_MANIFEST_SCHEMA_VERSION = "stage2_v4_dataset_manifest.1"
PROFILE_METRICS = tuple(HISTORY_VALUE_COLUMNS)
TRACK_LABELS = {
    "revealed_route_proxy": "revealed_route_proxy_predispatch",
    "oracle_timing": "oracle_timing_upper_bound",
}
NEIGHBOR_METRICS = (
    "crawl_time_share",
    "stop_time_share",
    "speed_cv_bounded",
    "acceleration_rms_bounded",
    "lcs_raw",
    "rts_raw",
    "observed_sec_per_m",
)
BASE_COLUMNS = (
    "split",
    "date",
    "order_id",
    "route_sequence",
    "traversal_id",
    "canonical_edge_uid",
    "observed_directed_edge_uid",
    "observed_from_node",
    "observed_to_node",
    "observed_direction",
    "decision_time",
    "decision_time_source",
    "start_node",
    "end_node",
    "route_part_length_m",
    "canonical_length_m",
    "canonical_highway",
    "road_class",
    "bridge",
    "tunnel",
    "synthetic_reverse_edge",
    "osm_direction_disagreement",
    "sequence_feature_mask",
    "directed_edge_model_scope",
    "label_available",
    "crawl_time_share",
    "stop_time_share",
    "speed_cv_bounded",
    "acceleration_rms_bounded",
    "crawl_target_valid",
    "stop_target_valid",
    "speed_cv_target_valid",
    "acceleration_rms_target_valid",
    "lcs_raw",
    "lcs_pct",
    "lcs_tail_event",
    "lcs_available",
    "lcs_target_valid",
    "rts_raw",
    "rts_pct",
    "rts_tail_event",
    "rts_available",
    "rts_measurement_available",
    "rts_target_valid",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Stage2V4ContractError(f"cannot read dataset manifest: {path}") from exc
    if not isinstance(value, dict):
        raise Stage2V4ContractError(f"dataset manifest is not an object: {path}")
    return value


def _route_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    groups = result.groupby(["split", "date", "order_id"], sort=False, observed=True)
    result["route_token_count"] = groups["route_sequence"].transform("size").astype("int32")
    denominator = np.maximum(result["route_token_count"].to_numpy(dtype=float) - 1.0, 1.0)
    result["route_position_ratio"] = (
        pd.to_numeric(result["route_sequence"], errors="coerce").to_numpy(dtype=float)
        / denominator
    )
    distance = pd.to_numeric(result["route_part_length_m"], errors="coerce").fillna(0.0)
    total = distance.groupby(
        [result["split"], result["date"], result["order_id"]],
        sort=False,
    ).transform("sum")
    cumulative_end = distance.groupby(
        [result["split"], result["date"], result["order_id"]],
        sort=False,
    ).cumsum()
    result["distance_to_destination_ratio"] = np.divide(
        (total - cumulative_end).to_numpy(dtype=float),
        total.to_numpy(dtype=float),
        out=np.zeros(len(result), dtype=float),
        where=total.to_numpy(dtype=float) > 0,
    )
    return result


def _query_frame(
    frame: pd.DataFrame,
    *,
    timing_column: str,
    timezone: str,
) -> pd.DataFrame:
    timing = pd.to_numeric(frame[timing_column], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    fallback = pd.to_numeric(frame["estimated_entry_time"], errors="coerce").to_numpy(
        dtype=np.float64,
        na_value=np.nan,
    )
    usable_timing = np.where(np.isfinite(timing), timing, fallback)
    time_bin, _hour, weekday = _local_time_fields(usable_timing, timezone)
    return pd.DataFrame(
        {
            "decision_time": frame["decision_time"].to_numpy(dtype=float),
            "observed_directed_edge_uid": frame[
                "observed_directed_edge_uid"
            ].astype(str).to_numpy(),
            "canonical_highway": frame["canonical_highway"].astype(str).to_numpy(),
            "profile_time_bin": time_bin,
            "profile_weekday_type": weekday,
        },
        index=frame.index,
    )


def _select_window_columns(
    values: pd.DataFrame,
    *,
    level: str,
    windows: Iterable[int],
) -> pd.DataFrame:
    columns: list[str] = []
    for minutes in windows:
        prefix = f"{level}_{minutes}m"
        columns.extend(
            [
                f"{prefix}_completed_traversal_count",
                f"{prefix}_maximum_event_time",
                f"{prefix}_feature_age_s",
                f"{prefix}_feature_available",
            ]
        )
        for metric in PROFILE_METRICS:
            columns.extend(
                [
                    f"{prefix}_{metric}_mean",
                    f"{prefix}_{metric}_available_label_count",
                ]
            )
    return values.loc[:, columns]


def _attach_history_features(
    frame: pd.DataFrame,
    history: TemporalHistoryIndex,
    config: Stage2V4Config,
    *,
    timing_column: str,
) -> pd.DataFrame:
    result = frame.copy()
    history_config = config.section("history")
    windows = tuple(int(value) for value in history_config["windows_minutes"])
    minimum = int(history_config["minimum_observations"])
    timezone = str(config.section("causality")["timezone"])
    queries = _query_frame(result, timing_column=timing_column, timezone=timezone)

    profiles = history.query_fallback(
        queries,
        metrics=PROFILE_METRICS,
        minimum_observations=minimum,
    )
    result = pd.concat(
        [result.reset_index(drop=True), profiles.reset_index(drop=True)],
        axis=1,
    )
    queries = queries.reset_index(drop=True)

    edge_full = history.query_window_features(
        queries,
        level="edge",
        metrics=PROFILE_METRICS,
        windows_minutes=windows,
    )
    edge = _select_window_columns(edge_full, level="edge", windows=windows)
    result = pd.concat([result, edge.reset_index(drop=True)], axis=1)
    del edge_full, edge

    anchor_window = 15
    for level in ("highway", "global"):
        values = history.query_window_features(
            queries,
            level=level,
            metrics=PROFILE_METRICS,
            windows_minutes=(anchor_window,),
        )
        selected = _select_window_columns(
            values,
            level=level,
            windows=(anchor_window,),
        )
        result = pd.concat([result, selected.reset_index(drop=True)], axis=1)

    order_groups = ["split", "date", "order_id"]
    for metric in NEIGHBOR_METRICS:
        source = f"edge_{anchor_window}m_{metric}_mean"
        grouped = result.groupby(order_groups, sort=False, observed=True)[source]
        result[f"upstream_{anchor_window}m_{metric}_mean"] = grouped.shift(1)
        result[f"downstream_{anchor_window}m_{metric}_mean"] = grouped.shift(-1)

    maximum_columns = [
        column
        for column in result.columns
        if column.endswith("_maximum_event_time")
    ]
    maximum = result.loc[:, maximum_columns].max(axis=1, skipna=True)
    decision = result["decision_time"].to_numpy(dtype=float)
    finite_maximum = pd.to_numeric(maximum, errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    leakage = np.isfinite(finite_maximum) & (finite_maximum >= decision)
    if leakage.any():
        raise Stage2V4ContractError(
            f"history feature leakage detected in {int(leakage.sum())} route tokens"
        )
    result["availability_timestamp"] = maximum
    result["feature_timestamp"] = maximum
    result["feature_age_s"] = decision - finite_maximum
    result["history_count"] = result["edge_60m_completed_traversal_count"].astype("int64")
    result["source_event_count"] = result["history_count"]
    result["dynamic_available_mask"] = result["history_count"].gt(0)
    result["self_order_excluded"] = True
    result["feature_time_check"] = np.where(
        result["availability_timestamp"].notna(),
        "PASS",
        "NO_HISTORY",
    )
    mask = np.zeros(len(result), dtype=np.uint16)
    for bit, metric in enumerate(PROFILE_METRICS):
        missing = result[f"{metric}_profile_mean"].isna().to_numpy()
        mask |= missing.astype(np.uint16) << bit
    result["numeric_missing_mask"] = mask
    return result


def _base_route_tokens(
    ref: Stage1BucketRef,
    history: TemporalHistoryIndex,
    config: Stage2V4Config,
) -> pd.DataFrame:
    alignment = build_route_alignment(ref)
    missing = sorted(set(BASE_COLUMNS) - set(alignment.route_tokens.columns))
    if missing:
        raise Stage2V4ContractError(f"dataset base route columns are missing: {missing}")
    frame = alignment.route_tokens.loc[:, list(BASE_COLUMNS) + ["enter_time"]].copy()
    forbidden = sorted(set(frame.columns) & FORBIDDEN_FORMAL_TARGETS)
    if forbidden:
        raise Stage2V4ContractError(f"forbidden formal targets entered dataset: {forbidden}")
    frame = _route_context(frame)
    frame = estimate_entry_times(frame, history, config)
    return frame


def _finalize_track(
    base: pd.DataFrame,
    history: TemporalHistoryIndex,
    config: Stage2V4Config,
    *,
    track: str,
) -> pd.DataFrame:
    expected = config.section("stage1_release")
    if track == "revealed_route_proxy":
        result = _attach_history_features(
            base,
            history,
            config,
            timing_column="estimated_entry_time",
        )
        result.drop(columns="enter_time", inplace=True)
        result["oracle_timing_available"] = False
    elif track == "oracle_timing":
        result = base.rename(columns={"enter_time": "oracle_entry_time"}).copy()
        decision = result["decision_time"].to_numpy(dtype=float)
        oracle = pd.to_numeric(result["oracle_entry_time"], errors="coerce").to_numpy(
            dtype=float,
            na_value=np.nan,
        )
        available = np.isfinite(oracle) & (oracle >= decision)
        result["oracle_timing_available"] = available
        result = _attach_history_features(
            result,
            history,
            config,
            timing_column="oracle_entry_time",
        )
        profile_columns = [
            column for column in result.columns if "_profile_" in column
        ]
        result.loc[~available, profile_columns] = np.nan
    else:
        raise Stage2V4ContractError(f"unknown dataset track: {track}")

    result["route_proxy_track"] = TRACK_LABELS[track]
    result["fully_deployable"] = False
    result["stage3_eligible_track"] = track == "revealed_route_proxy"
    result["label_alignment_status"] = np.where(
        result["label_available"],
        "one_to_one",
        "unlabelled_route_token",
    )
    result["stage1_model_id"] = expected["model_id"]
    result["stage1_schema_version"] = expected["label_schema_version"]
    result["stage2_dataset_schema_version"] = DATASET_SCHEMA_VERSION
    if set(result.columns) & FORBIDDEN_FORMAL_TARGETS:
        raise Stage2V4ContractError("formal dataset contains a forbidden legacy target")
    if result.duplicated(list(ROUTE_PRIMARY_KEY)).any():
        raise Stage2V4ContractError("formal dataset route token key is duplicated")
    return result


def _oracle_from_revealed(
    revealed: pd.DataFrame,
    base: pd.DataFrame,
    history: TemporalHistoryIndex,
    config: Stage2V4Config,
) -> pd.DataFrame:
    """Reuse decision-time windows and replace only timing-dependent profiles."""

    result = revealed.copy()
    oracle = pd.to_numeric(base["enter_time"], errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    decision = result["decision_time"].to_numpy(dtype=float)
    available = np.isfinite(oracle) & (oracle >= decision)
    result["oracle_entry_time"] = oracle
    result["oracle_timing_available"] = available
    timing_frame = result.copy()
    timing_frame["oracle_entry_time"] = oracle
    queries = _query_frame(
        timing_frame,
        timing_column="oracle_entry_time",
        timezone=str(config.section("causality")["timezone"]),
    )
    profiles = history.query_fallback(
        queries,
        metrics=PROFILE_METRICS,
        minimum_observations=int(config.section("history")["minimum_observations"]),
    ).reset_index(drop=True)
    for column in profiles.columns:
        result[column] = profiles[column].to_numpy()
    profile_columns = [column for column in result.columns if "_profile_" in column]
    result.loc[~available, profile_columns] = np.nan

    maximum_columns = [
        column
        for column in result.columns
        if column.endswith("_maximum_event_time")
    ]
    maximum = result.loc[:, maximum_columns].max(axis=1, skipna=True)
    maximum_values = pd.to_numeric(maximum, errors="coerce").to_numpy(
        dtype=float,
        na_value=np.nan,
    )
    leakage = np.isfinite(maximum_values) & (maximum_values >= decision)
    if leakage.any():
        raise Stage2V4ContractError("oracle track history leaked past decision_time")
    result["availability_timestamp"] = maximum
    result["feature_timestamp"] = maximum
    result["feature_age_s"] = decision - maximum_values
    result["feature_time_check"] = np.where(
        result["availability_timestamp"].notna(),
        "PASS",
        "NO_HISTORY",
    )
    mask = np.zeros(len(result), dtype=np.uint16)
    for bit, metric in enumerate(PROFILE_METRICS):
        mask |= (
            result[f"{metric}_profile_mean"].isna().to_numpy().astype(np.uint16)
            << bit
        )
    result["numeric_missing_mask"] = mask
    result["route_proxy_track"] = TRACK_LABELS["oracle_timing"]
    result["fully_deployable"] = False
    result["stage3_eligible_track"] = False
    return result


class _AtomicDailyWriter:
    def __init__(self, target: Path):
        self.target = target
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=f".{target.name}.tmp-",
            suffix=".parquet",
            dir=target.parent,
        )
        os.close(descriptor)
        self.temporary = Path(name)
        self.writer: pq.ParquetWriter | None = None
        self.row_count = 0

    def write(self, frame: pd.DataFrame) -> None:
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self.writer is None:
            self.writer = pq.ParquetWriter(
                self.temporary,
                table.schema,
                compression="zstd",
            )
        elif table.schema != self.writer.schema:
            table = table.cast(self.writer.schema)
        self.writer.write_table(table)
        self.row_count += len(frame)

    def publish(self) -> None:
        if self.writer is None:
            raise Stage2V4ContractError(f"no rows written for {self.target}")
        self.writer.close()
        self.writer = None
        os.replace(self.temporary, self.target)

    def abort(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
        self.temporary.unlink(missing_ok=True)


def _resume_manifest(
    manifest_path: Path,
    data_path: Path,
    *,
    config_sha: str,
    code_sha: str,
    history_sha: str,
) -> dict[str, Any] | None:
    if not manifest_path.is_file() or not data_path.is_file():
        return None
    manifest = _read_json(manifest_path)
    expected = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "engineering_status": "PASS",
        "stage2_config_sha256": config_sha,
        "stage2_code_sha": code_sha,
        "history_manifest_sha256": history_sha,
        "file_sha256": sha256_file(data_path),
        "row_count": int(pq.ParquetFile(data_path).metadata.num_rows),
    }
    return manifest if all(manifest.get(key) == value for key, value in expected.items()) else None


def build_route_conditioned_dataset(
    stage1_output: str | Path,
    stage1_input: str | Path,
    history_root: str | Path,
    output_root: str | Path,
    config: Stage2V4Config,
    *,
    tracks: tuple[str, ...],
    resume: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    started = time.perf_counter()
    allowed_tracks = {"revealed_route_proxy", "oracle_timing"}
    if not tracks or set(tracks) - allowed_tracks:
        raise Stage2V4ContractError(f"invalid dataset tracks: {tracks}")
    history_summary = validate_history_store(history_root, config)
    history_manifest_path = Path(history_root) / "history_store_manifest.json"
    history_sha = sha256_file(history_manifest_path)
    history = TemporalHistoryIndex.from_store(history_root, config)
    refs = discover_stage1_buckets(stage1_output, stage1_input)
    by_date: dict[str, list[Stage1BucketRef]] = defaultdict(list)
    for ref in refs:
        by_date[ref.date].append(ref)
    output = Path(output_root)
    code_sha = stage2_v4_code_identity(
        (
            "stage2/v4/dataset_builder.py",
            "stage2/v4/stage1_adapter.py",
            "stage2/v4/history_index.py",
            "stage2/v4/entry_time.py",
            "stage2/v4/contracts.py",
            "stage2/v4/io.py",
        )
    )
    all_manifests: list[dict[str, Any]] = []
    transformed = 0
    resumed = 0

    for date in sorted(by_date):
        writers: dict[str, _AtomicDailyWriter] = {}
        manifests: dict[str, Path] = {}
        active_tracks: list[str] = []
        for track in tracks:
            data_path = output / track / f"day={date}.parquet"
            manifest_path = output / "manifests" / track / f"day={date}.json"
            existing = (
                _resume_manifest(
                    manifest_path,
                    data_path,
                    config_sha=config.digest,
                    code_sha=code_sha,
                    history_sha=history_sha,
                )
                if resume
                else None
            )
            if existing is not None:
                all_manifests.append(existing)
                resumed += 1
                continue
            if (data_path.exists() or manifest_path.exists()) and not force:
                raise Stage2V4ContractError(
                    f"dataset output already exists but is not resumable: {track}/{date}"
                )
            writers[track] = _AtomicDailyWriter(data_path)
            manifests[track] = manifest_path
            active_tracks.append(track)
        if not active_tracks:
            continue

        order_ids: dict[str, set[str]] = {track: set() for track in active_tracks}
        leakage_counts = {track: 0 for track in active_tracks}
        try:
            for ref in sorted(by_date[date], key=lambda item: item.bucket):
                base = _base_route_tokens(ref, history, config)
                revealed: pd.DataFrame | None = None
                if (
                    "revealed_route_proxy" in active_tracks
                    or "oracle_timing" in active_tracks
                ):
                    revealed = _finalize_track(
                        base,
                        history,
                        config,
                        track="revealed_route_proxy",
                    )
                for track in active_tracks:
                    if track == "revealed_route_proxy":
                        if revealed is None:
                            raise AssertionError("revealed track was not prepared")
                        frame = revealed
                    else:
                        if revealed is None:
                            raise AssertionError("oracle source track was not prepared")
                        frame = _oracle_from_revealed(
                            revealed,
                            base,
                            history,
                            config,
                        )
                    leakage_counts[track] += int(
                        frame["feature_time_check"].eq("FAIL").sum()
                    )
                    order_ids[track].update(frame["order_id"].astype(str).unique())
                    writers[track].write(frame)
                    if track != "revealed_route_proxy":
                        del frame
                del base, revealed
            for track in active_tracks:
                writers[track].publish()
                data_path = writers[track].target
                manifest = {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "engineering_status": "PASS",
                    "date": date,
                    "track": track,
                    "stage2_config_sha256": config.digest,
                    "stage2_code_sha": code_sha,
                    "stage1_model_id": config.section("stage1_release")["model_id"],
                    "history_manifest_sha256": history_sha,
                    "source_bucket_count": len(by_date[date]),
                    "row_count": writers[track].row_count,
                    "order_count": len(order_ids[track]),
                    "route_key_duplicate_count": 0,
                    "time_leakage_violation_count": leakage_counts[track],
                    "file_sha256": sha256_file(data_path),
                }
                atomic_write_json(manifests[track], manifest)
                all_manifests.append(manifest)
                transformed += 1
        except Exception:
            for writer in writers.values():
                writer.abort()
            raise

    by_track = {
        track: {
            "row_count": sum(
                int(item["row_count"]) for item in all_manifests if item["track"] == track
            ),
            "order_count_by_date": {
                item["date"]: int(item["order_count"])
                for item in all_manifests
                if item["track"] == track
            },
        }
        for track in tracks
    }
    failures: list[str] = []
    expected = config.section("stage1_release")
    for track in tracks:
        if by_track[track]["row_count"] != expected["route_sequence_count"]:
            failures.append(
                f"{track} row reconciliation: expected "
                f"{expected['route_sequence_count']}, got {by_track[track]['row_count']}"
            )
    summary = {
        "schema_version": DATASET_MANIFEST_SCHEMA_VERSION,
        "engineering_status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "stage2_config_sha256": config.digest,
        "stage2_code_sha": code_sha,
        "stage1_model_id": expected["model_id"],
        "history_manifest_sha256": history_sha,
        "history_event_count": history_summary["event_count"],
        "tracks": by_track,
        "transformed_day_track_count": transformed,
        "resumed_day_track_count": resumed,
        "runtime_s": time.perf_counter() - started,
    }
    atomic_write_json(output / "dataset_manifest.json", summary)
    if failures:
        raise Stage2V4ContractError("; ".join(failures))
    return summary
