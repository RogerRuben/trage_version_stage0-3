"""Engineering audit for Stage 1 v3 outputs.

Passing this audit establishes input/output contract correctness only.  It does
not establish scientific validity of LCS or RTS.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .aggregation import aggregate_order_labels
from .input_adapter import iter_stage0_buckets, load_stage0_bucket
from .freeze import load_stage0_freeze_manifest
from .io import (
    parquet_column_names,
    parquet_row_count,
    parquet_schema_sha256,
    sha256_file,
    stage1_v3_code_identity,
)
from .models import load_model_bundle
from .pipeline import OUTPUT_PRODUCTS, _input_identities
from .primitives import build_interval_labels, build_traversal_primitives
from .schema import (
    ContractError,
    OUTPUT_BUCKET_SCHEMA_VERSION,
    OUTPUT_PRIMARY_KEYS,
    OUTPUT_REQUIRED_COLUMNS,
    OUTPUT_SUMMARY_SCHEMA_VERSION,
)
from .support import apply_directed_support

if TYPE_CHECKING:
    from .config import Stage1V3Config


STRICT_BOOLEAN_OUTPUTS = {
    "interval_labels": (
        "label_valid",
        "is_stop",
        "is_crawl",
        "is_low_speed_total",
        "kinematic_sequence_valid",
        "lcs_component_available",
    ),
    "traversal_labels": (
        "direct_distance_exceeds_allocated",
        "discontinuous_direct_window",
        "lcs_available",
        "rts_available",
        "rts_measurement_available",
        "gns_available",
        "iis_available",
        "pmis_available",
    ),
    "movement_context": ("iis_available",),
    "order_labels": (
        "lcs_available",
        "rts_available",
        "gns_available",
        "iis_available",
        "pmis_available",
    ),
    "order_label_quality": (
        "observed_time_exceeds_order_duration",
        "observed_distance_exceeds_route_distance",
        "direct_coverage_pass",
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read Stage1 v3 audit manifest {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"manifest must be a JSON object: {path}")
    return value


def _duplicates(frame: pd.DataFrame, keys: list[str]) -> int:
    if not set(keys).issubset(frame.columns):
        return len(frame)
    return int(frame.duplicated(keys, keep=False).sum())


def _nonmissing(frame: pd.DataFrame, columns: list[str]) -> int:
    existing = [column for column in columns if column in frame]
    if not existing:
        return 0
    return int(frame[existing].notna().any(axis=1).sum())


def _discover_output_bucket_keys(root: Path) -> set[tuple[str, str, int]]:
    keys: set[tuple[str, str, int]] = set()
    if not root.is_dir():
        raise ContractError(f"Stage1 v3 output root does not exist: {root}")
    for split_path in root.iterdir():
        if not split_path.is_dir():
            continue
        if not split_path.name.startswith("split="):
            raise ContractError(f"unexpected Stage1 v3 output directory: {split_path}")
        split = split_path.name.removeprefix("split=")
        if split not in {"train", "validation", "test"}:
            raise ContractError(f"unexpected Stage1 v3 output split: {split_path}")
        for date_path in split_path.iterdir():
            if not date_path.is_dir() or not date_path.name.startswith("date="):
                raise ContractError(f"unexpected output date path: {date_path}")
            date = date_path.name.removeprefix("date=")
            if len(date) != 8 or not date.isdigit():
                raise ContractError(f"invalid output date partition: {date_path}")
            for bucket_path in date_path.iterdir():
                if (
                    not bucket_path.is_dir()
                    or not bucket_path.name.startswith("bucket=")
                ):
                    raise ContractError(f"unexpected output bucket path: {bucket_path}")
                bucket_text = bucket_path.name.removeprefix("bucket=")
                if len(bucket_text) != 5 or not bucket_text.isdigit():
                    raise ContractError(f"invalid output bucket partition: {bucket_path}")
                key = (split, date, int(bucket_text))
                if key in keys:
                    raise ContractError(f"duplicate output bucket partition: {key}")
                keys.add(key)
    return keys


def _strict_boolean(series: pd.Series, name: str) -> pd.Series:
    if not series.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise ContractError(f"{name} must contain only non-null booleans")
    return series.eq(True)


def _nullable_identity_mismatch(
    left: pd.Series,
    right: pd.Series,
    *,
    numeric: bool = False,
) -> pd.Series:
    """Compare identity columns without turning nullable values into strings."""

    both_missing = left.isna() & right.isna()
    if numeric:
        same_value = (
            pd.to_numeric(left, errors="coerce")
            .astype("Int64")
            .eq(pd.to_numeric(right, errors="coerce").astype("Int64"))
            .fillna(False)
        )
    else:
        same_value = (
            left.astype("string")
            .eq(right.astype("string"))
            .fillna(False)
        )
    return ~(both_missing | same_value)


def _frame_value_mismatches(
    actual: pd.DataFrame,
    expected: pd.DataFrame,
    *,
    keys: list[str],
) -> dict[str, int]:
    """Compare a generated product with a frame regenerated from source inputs."""

    if set(actual.columns) != set(expected.columns):
        return {"__columns__": 1}
    joined = expected.merge(
        actual,
        on=keys,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_expected", "_actual"),
    )
    failures: dict[str, int] = {}
    merge_failures = int(joined["_merge"].ne("both").sum())
    if merge_failures:
        failures["__keys__"] = merge_failures
        return failures
    for column in expected.columns:
        if column in keys:
            continue
        expected_values = joined[f"{column}_expected"]
        actual_values = joined[f"{column}_actual"]
        if pd.api.types.is_bool_dtype(expected[column].dtype):
            both_missing = expected_values.isna() & actual_values.isna()
            strict_actual = actual_values.map(
                lambda value: pd.isna(value)
                or isinstance(value, (bool, np.bool_))
            )
            same_boolean = (
                actual_values.astype("boolean")
                .eq(expected_values.astype("boolean"))
                .fillna(False)
            )
            mismatch = ~(both_missing | (strict_actual & same_boolean))
        elif pd.api.types.is_numeric_dtype(expected[column].dtype):
            left = pd.to_numeric(expected_values, errors="coerce")
            right = pd.to_numeric(actual_values, errors="coerce")
            mismatch = ~np.isclose(
                left,
                right,
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            )
        else:
            both_missing = expected_values.isna() & actual_values.isna()
            same_string = expected_values.astype("string").eq(
                actual_values.astype("string")
            ).fillna(False)
            mismatch = ~(
                both_missing
                | same_string
            )
        count = int(mismatch.sum())
        if count:
            failures[column] = count
    return failures


def verify_stage1_v3(
    input_root: str | Path,
    model_root: str | Path,
    output_root: str | Path,
    stage0_freeze_manifest: str | Path,
    config: "Stage1V3Config",
    *,
    stage1_code_sha: str | None = None,
) -> dict[str, Any]:
    """Verify all v3 buckets with global order-identity reconciliation."""

    from .config import validate_config

    validate_config(config)
    started = time.perf_counter()
    try:
        import psutil

        process = psutil.Process()
        peak_rss_mb = process.memory_info().rss / (1024 * 1024)
    except (ImportError, OSError):
        process = None
        peak_rss_mb = float("nan")
    code_sha = stage1_v3_code_identity()
    if (
        stage1_code_sha is not None
        and str(stage1_code_sha).strip() != code_sha
    ):
        raise ContractError(
            "provided Stage1 code identity differs from the executable source tree"
        )
    freeze = load_stage0_freeze_manifest(stage0_freeze_manifest, config)
    models = load_model_bundle(model_root, config)
    if models.support.counts["scope"].eq("road_class_hour").any():
        raise ContractError("legacy road_class_hour support scope is forbidden")
    if models.manifest.get("stage1_code_sha") != code_sha:
        raise ContractError("audit code SHA differs from the fitted model")
    if models.manifest.get("stage0_freeze_identity", {}).get(
        "manifest_sha"
    ) != freeze.manifest_sha:
        raise ContractError("audit model and Stage0 freeze identities differ")
    failures: list[str] = []
    counters: Counter[str] = Counter()
    schema_hashes: dict[str, set[str]] = defaultdict(set)
    seen_orders: dict[str, tuple[str, str, int]] = {}
    dates: dict[str, set[str]] = defaultdict(set)
    output_root_path = Path(output_root)

    refs = list(iter_stage0_buckets(input_root, config))
    current_identities = _input_identities(refs)
    if current_identities != models.manifest.get("input_bucket_identities"):
        raise ContractError("audit inputs differ from the fitted model inputs")
    expected_bucket_keys = {
        (ref.split, ref.date, ref.bucket) for ref in refs
    }
    actual_bucket_keys = _discover_output_bucket_keys(output_root_path)
    if actual_bucket_keys != expected_bucket_keys:
        missing = sorted(expected_bucket_keys - actual_bucket_keys)
        extra = sorted(actual_bucket_keys - expected_bucket_keys)
        raise ContractError(
            "Stage1 v3 output bucket set differs from input: "
            f"missing={missing[:5]}, extra={extra[:5]}"
        )
    for ref in refs:
        if process is not None:
            peak_rss_mb = max(
                peak_rss_mb,
                process.memory_info().rss / (1024 * 1024),
            )
        dates[ref.split].add(ref.date)
        bucket = load_stage0_bucket(ref, config)
        target = (
            output_root_path
            / f"split={ref.split}"
            / f"date={ref.date}"
            / f"bucket={ref.bucket:05d}"
        )
        manifest_path = target / "manifest.json"
        if not manifest_path.is_file():
            failures.append(f"missing output manifest: {target}")
            continue
        manifest = _read_json(manifest_path)
        if manifest.get("engineering_status") != "PASS":
            failures.append(f"non-PASS output bucket: {target}")
        input_identity = current_identities[
            f"{ref.split}/{ref.date}/{ref.bucket:05d}"
        ]
        for name, expected in {
            "schema_version": OUTPUT_BUCKET_SCHEMA_VERSION,
            "label_schema_version": config.schema_version,
            "scientific_status": "NOT_VALIDATED",
            "model_id": models.model_id,
            "config_sha": config.digest,
            "stage1_code_sha": code_sha,
            "stage0_freeze_manifest_sha": freeze.manifest_sha,
            "stage0_release": config.section("stage0_release"),
            "input_bucket_sha": input_identity["bucket_sha"],
            "split": ref.split,
            "date": ref.date,
            "bucket": ref.bucket,
        }.items():
            if manifest.get(name) != expected:
                failures.append(f"{target}: manifest mismatch for {name}")
        if manifest.get("input_file_hashes") != input_identity["files"]:
            failures.append(f"{target}: input file hashes differ")
        if manifest.get("input_schema_hashes") != input_identity["schemas"]:
            failures.append(f"{target}: input schema hashes differ")
        expected_product_set = set(OUTPUT_PRODUCTS)
        manifest_maps: dict[str, dict[str, Any]] = {}
        for name in (
            "output_file_hashes",
            "output_schema_hashes",
            "product_row_counts",
        ):
            raw_value = manifest.get(name)
            value = raw_value if isinstance(raw_value, dict) else {}
            manifest_maps[name] = value
            if not isinstance(value, dict) or set(value) != expected_product_set:
                failures.append(f"{target}: invalid manifest map {name}")

        products: dict[str, pd.DataFrame] = {}
        bucket_schema_valid = True
        for product in OUTPUT_PRODUCTS:
            path = target / f"{product}.parquet"
            if not path.is_file():
                failures.append(f"missing output product: {path}")
                continue
            expected_hash = manifest_maps["output_file_hashes"].get(product)
            if sha256_file(path) != expected_hash:
                failures.append(f"output file hash mismatch: {path}")
            frame = pd.read_parquet(path)
            products[product] = frame
            actual_count = int(len(frame))
            if parquet_row_count(path) != actual_count:
                failures.append(f"Parquet metadata row count mismatch: {path}")
            counters[f"{product}_rows"] += actual_count
            if actual_count != manifest_maps["product_row_counts"].get(product):
                failures.append(f"output row count mismatch: {path}")
            schema_sha = parquet_schema_sha256(path)
            if actual_count:
                schema_hashes[product].add(schema_sha)
            if schema_sha != manifest_maps["output_schema_hashes"].get(product):
                failures.append(f"output schema hash mismatch: {path}")
            columns = set(parquet_column_names(path))
            missing_columns = OUTPUT_REQUIRED_COLUMNS[product] - columns
            unexpected_columns = columns - OUTPUT_REQUIRED_COLUMNS[product]
            if missing_columns or unexpected_columns:
                bucket_schema_valid = False
                failures.append(
                    f"{path}: output columns differ from contract; "
                    f"missing={sorted(missing_columns)}, "
                    f"unexpected={sorted(unexpected_columns)}"
                )

        actual_parquet = {path.stem for path in target.glob("*.parquet")}
        if actual_parquet != expected_product_set:
            failures.append(f"{target}: unexpected output product set")
            bucket_schema_valid = False
        if set(products) != expected_product_set or not bucket_schema_valid:
            continue
        intervals = products["interval_labels"]
        traversals = products["traversal_labels"]
        movements = products["movement_context"]
        orders = products["order_labels"]
        quality = products["order_label_quality"]

        for product, columns in STRICT_BOOLEAN_OUTPUTS.items():
            for column in columns:
                try:
                    _strict_boolean(
                        products[product][column],
                        f"{product}.{column}",
                    )
                except ContractError as exc:
                    failures.append(f"{target}: {exc}")

        duplicate_checks = {
            f"{product}_duplicate": (
                products[product],
                list(OUTPUT_PRIMARY_KEYS[product]),
            )
            for product in OUTPUT_PRODUCTS
        }
        for name, (frame, keys) in duplicate_checks.items():
            count = _duplicates(frame, keys)
            counters[name] += count
            if count:
                failures.append(f"{target}: {name}={count}")

        for product, frame in products.items():
            for name, value in (("split", ref.split), ("date", ref.date)):
                mismatch = int(
                    (~frame[name].astype(str).eq(str(value))).sum()
                )
                if mismatch:
                    failures.append(
                        f"{target}: {product}.{name} partition mismatch={mismatch}"
                    )

        def key_set(frame: pd.DataFrame, columns: list[str]) -> set[tuple[str, ...]]:
            return {
                tuple(str(value) for value in row)
                for row in frame[columns].itertuples(index=False, name=None)
            }

        interval_key_columns = ["order_id", "gps_interval_id"]
        if key_set(intervals, interval_key_columns) != key_set(
            bucket.link_interval_observations, interval_key_columns
        ):
            failures.append(f"{target}: interval label key set mismatch")
        traversal_key_columns = ["order_id", "traversal_id"]
        direct_input_traversals = bucket.link_traversals.loc[
            bucket.link_traversals["measurement_source"].eq("direct_observed")
        ]
        if key_set(traversals, traversal_key_columns) != key_set(
            direct_input_traversals, traversal_key_columns
        ):
            failures.append(f"{target}: traversal label key set mismatch")
        movement_key_columns = ["order_id", "movement_sequence"]
        if key_set(movements, movement_key_columns) != key_set(
            bucket.turn_movements, movement_key_columns
        ):
            failures.append(f"{target}: movement context key set mismatch")

        try:
            regenerated_intervals = build_interval_labels(
                bucket.link_interval_observations,
                bucket.link_traversals,
                bucket.route_parts,
                config,
            )
            interval_mismatches = _frame_value_mismatches(
                intervals.drop(columns=["split", "date"]),
                regenerated_intervals,
                keys=["order_id", "gps_interval_id"],
            )
            regenerated_primitives = build_traversal_primitives(
                bucket.link_interval_observations,
                bucket.link_traversals,
                bucket.route_parts,
                config,
            )
            primitive_columns = [
                column
                for column in regenerated_primitives.columns
                if column not in {
                    "lcs_raw",
                    "lcs_available",
                    "lcs_unavailable_reason",
                }
            ]
            primitive_mismatches = _frame_value_mismatches(
                traversals[primitive_columns],
                regenerated_primitives[primitive_columns],
                keys=["order_id", "traversal_id"],
            )
        except ContractError as exc:
            failures.append(f"{target}: cannot regenerate primitives: {exc}")
        else:
            if interval_mismatches:
                failures.append(
                    f"{target}: interval labels differ from regenerated "
                    f"primitives: {interval_mismatches}"
                )
            if primitive_mismatches:
                failures.append(
                    f"{target}: traversal primitives differ from regenerated "
                    f"values: {primitive_mismatches}"
                )

        if len(intervals):
            stop = intervals["is_stop"].eq(True)
            crawl = intervals["is_crawl"].eq(True)
            low_total = intervals["is_low_speed_total"].eq(True)
            invalid_low_speed = (stop & crawl) | low_total.ne(stop | crawl)
            if invalid_low_speed.any():
                failures.append(
                    f"{target}: crawl/stop partition failures="
                    f"{int(invalid_low_speed.sum())}"
                )
            try:
                valid_flags = _strict_boolean(
                    intervals["label_valid"],
                    "interval_labels.label_valid",
                )
            except ContractError as exc:
                failures.append(f"{target}: {exc}")
                valid_flags = pd.Series(False, index=intervals.index)
            invalid_direct = int(
                (
                    ~intervals["measurement_source"].eq("direct_observed")
                    | ~valid_flags
                    | ~intervals["label_schema_version"].eq(
                        config.schema_version
                    )
                ).sum()
            )
            counters["non_direct_interval_label"] += invalid_direct
            if invalid_direct:
                failures.append(f"{target}: non-direct interval labels={invalid_direct}")
            traversal_keys = set(
                zip(
                    traversals["order_id"].astype(str),
                    traversals["traversal_id"].astype(str),
                )
            )
            orphan_intervals = sum(
                key not in traversal_keys
                for key in zip(
                    intervals["order_id"].astype(str),
                    intervals["traversal_id"].astype(str),
                )
            )
            counters["orphan_interval_label"] += orphan_intervals
            if orphan_intervals:
                failures.append(f"{target}: orphan interval labels={orphan_intervals}")

            interval_identity = intervals[
                [
                    "order_id",
                    "gps_interval_id",
                    "traversal_id",
                    "canonical_edge_uid",
                    "observed_directed_edge_uid",
                    "observed_direction",
                    "interval_start_time",
                    "interval_end_time",
                    "observed_travel_time_s",
                    "observed_distance_m",
                    "observed_speed_mps",
                ]
            ].merge(
                bucket.link_interval_observations[
                    [
                        "order_id",
                        "gps_interval_id",
                        "traversal_id",
                        "canonical_edge_uid",
                        "observed_directed_edge_uid",
                        "observed_direction",
                        "interval_start_time",
                        "interval_end_time",
                        "observed_travel_time_s",
                        "observed_distance_m",
                        "observed_speed_mps",
                    ]
                ],
                on=["order_id", "gps_interval_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("_output", "_input"),
            )
            for column in (
                "traversal_id",
                "canonical_edge_uid",
                "observed_directed_edge_uid",
                "observed_direction",
            ):
                mismatch = ~interval_identity[f"{column}_output"].astype(str).eq(
                    interval_identity[f"{column}_input"].astype(str)
                )
                if mismatch.any():
                    failures.append(
                        f"{target}: interval {column} mismatch={int(mismatch.sum())}"
                    )
            for column in (
                "interval_start_time",
                "interval_end_time",
                "observed_travel_time_s",
                "observed_distance_m",
                "observed_speed_mps",
            ):
                left = pd.to_numeric(
                    interval_identity[f"{column}_output"], errors="coerce"
                )
                right = pd.to_numeric(
                    interval_identity[f"{column}_input"], errors="coerce"
                )
                mismatch = ~np.isclose(
                    left,
                    right,
                    atol=float(config.section("direct")[
                        "speed_tolerance_mps"
                        if column == "observed_speed_mps"
                        else (
                            "distance_identity_tolerance_m"
                            if column == "observed_distance_m"
                            else "duration_tolerance_s"
                        )
                    ]),
                    rtol=1e-12,
                )
                if mismatch.any():
                    failures.append(
                        f"{target}: interval {column} mismatch={int(mismatch.sum())}"
                    )

        if len(traversals):
            expected_support = apply_directed_support(
                traversals,
                models.support,
                config,
            )
            for column in (
                "edge_observation_count",
                "edge_hour_observation_count",
                "edge_time_bin_30m_observation_count",
                "edge_support_level",
                "edge_hour_support_level",
                "directed_edge_model_scope",
            ):
                mismatch = (
                    traversals[column]
                    .astype("string")
                    .fillna("<NULL>")
                    .ne(
                        expected_support[column]
                        .astype("string")
                        .fillna("<NULL>")
                    )
                )
                if mismatch.any():
                    failures.append(
                        f"{target}: support identity failures for {column}="
                        f"{int(mismatch.sum())}"
                    )
            unseen = traversals["directed_edge_model_scope"].eq(
                "evaluation_unseen"
            )
            unseen_nonzero = unseen & (
                pd.to_numeric(
                    traversals["edge_observation_count"], errors="coerce"
                ).ne(0)
                | pd.to_numeric(
                    traversals["edge_hour_observation_count"],
                    errors="coerce",
                ).ne(0)
                | pd.to_numeric(
                    traversals[
                        "edge_time_bin_30m_observation_count"
                    ],
                    errors="coerce",
                ).ne(0)
            )
            if unseen_nonzero.any():
                failures.append(
                    f"{target}: evaluation unseen support leakage="
                    f"{int(unseen_nonzero.sum())}"
                )
            distance_exceeds = traversals[
                "direct_distance_exceeds_allocated"
            ].eq(True)
            invalid_distance_labels = distance_exceeds & (
                traversals["lcs_available"].eq(True)
                | traversals["rts_available"].eq(True)
                | traversals["rts_measurement_available"].eq(True)
                | pd.to_numeric(
                    traversals["observed_sec_per_m"], errors="coerce"
                ).notna()
            )
            if invalid_distance_labels.any():
                failures.append(
                    f"{target}: labels retained after distance exceed="
                    f"{int(invalid_distance_labels.sum())}"
                )
            acceleration_pairs = pd.to_numeric(
                traversals["acceleration_pair_count"], errors="coerce"
            )
            acceleration_weight = pd.to_numeric(
                traversals["acceleration_weight_s"], errors="coerce"
            )
            weighted_rms = pd.to_numeric(
                traversals["acceleration_rms_mps2"], errors="coerce"
            )
            invalid_acceleration_weight = (
                acceleration_pairs.ge(
                    int(
                        config.section("lcs")[
                            "minimum_acceleration_pairs"
                        ]
                    )
                )
                & (
                    ~np.isfinite(acceleration_weight)
                    | acceleration_weight.le(0)
                    | ~np.isfinite(weighted_rms)
                )
            )
            if invalid_acceleration_weight.any():
                failures.append(
                    f"{target}: acceleration weighting failures="
                    f"{int(invalid_acceleration_weight.sum())}"
                )
            invalid_traversal_provenance = int(
                (
                    ~traversals["measurement_source"].eq("direct_observed")
                    | ~traversals["label_schema_version"].eq(
                        config.schema_version
                    )
                ).sum()
            )
            if invalid_traversal_provenance:
                failures.append(
                    f"{target}: invalid traversal provenance="
                    f"{invalid_traversal_provenance}"
                )
            traversal_identity = traversals[
                [
                    "order_id",
                    "traversal_id",
                    "route_sequence",
                    "canonical_edge_uid",
                    "observed_directed_edge_uid",
                    "direct_observed_time_s",
                    "direct_observed_distance_m",
                    "allocated_distance_m",
                ]
            ].merge(
                direct_input_traversals[
                    [
                        "order_id",
                        "traversal_id",
                        "route_sequence",
                        "canonical_edge_uid",
                        "observed_directed_edge_uid",
                        "observed_travel_time_s",
                        "observed_distance_m",
                        "allocated_distance_m",
                    ]
                ].rename(
                    columns={
                        "observed_travel_time_s": "direct_observed_time_s",
                        "observed_distance_m": "direct_observed_distance_m",
                    }
                ),
                on=["order_id", "traversal_id"],
                how="inner",
                validate="one_to_one",
                suffixes=("_output", "_input"),
            )
            for column in (
                "route_sequence",
                "canonical_edge_uid",
                "observed_directed_edge_uid",
            ):
                mismatch = ~traversal_identity[f"{column}_output"].astype(str).eq(
                    traversal_identity[f"{column}_input"].astype(str)
                )
                if mismatch.any():
                    failures.append(
                        f"{target}: traversal {column} mismatch={int(mismatch.sum())}"
                    )
            traversal_numeric_pairs = {
                "direct_observed_time_s": "direct_observed_time_s",
                "direct_observed_distance_m": "direct_observed_distance_m",
                "allocated_distance_m": "allocated_distance_m",
            }
            for output_column, input_column in traversal_numeric_pairs.items():
                left = pd.to_numeric(
                    traversal_identity[f"{output_column}_output"],
                    errors="coerce",
                )
                right = pd.to_numeric(
                    traversal_identity[f"{input_column}_input"],
                    errors="coerce",
                )
                tolerance_name = (
                    "duration_tolerance_s"
                    if output_column == "direct_observed_time_s"
                    else "distance_identity_tolerance_m"
                )
                mismatch = ~np.isclose(
                    left,
                    right,
                    atol=float(config.section("direct")[tolerance_name]),
                    rtol=1e-12,
                )
                if mismatch.any():
                    failures.append(
                        f"{target}: traversal {output_column} mismatch="
                        f"{int(mismatch.sum())}"
                    )

        if len(movements):
            movement_identity = movements[
                [
                    "order_id",
                    "movement_sequence",
                    "from_edge_uid",
                    "via_node",
                    "to_edge_uid",
                    "movement_source",
                    "movement_quality",
                ]
            ].merge(
                bucket.turn_movements[
                    [
                        "order_id",
                        "movement_sequence",
                        "from_edge_uid",
                        "via_node",
                        "to_edge_uid",
                        "movement_source",
                        "movement_quality",
                    ]
                ],
                on=["order_id", "movement_sequence"],
                how="inner",
                validate="one_to_one",
                suffixes=("_output", "_input"),
            )
            for column in (
                "from_edge_uid",
                "via_node",
                "to_edge_uid",
                "movement_source",
                "movement_quality",
            ):
                output_values = movement_identity[f"{column}_output"]
                input_values = movement_identity[f"{column}_input"]
                mismatch = _nullable_identity_mismatch(
                    output_values,
                    input_values,
                    numeric=column == "via_node",
                )
                if mismatch.any():
                    failures.append(
                        f"{target}: movement {column} mismatch={int(mismatch.sum())}"
                    )

        for dimension in ("lcs", "rts"):
            try:
                available = _strict_boolean(
                    traversals[f"{dimension}_available"],
                    f"traversal_labels.{dimension}_available",
                )
            except ContractError as exc:
                failures.append(f"{target}: {exc}")
                available = pd.Series(False, index=traversals.index)
            raw_source = traversals[f"{dimension}_raw"]
            pct_source = traversals[f"{dimension}_pct"]
            raw = pd.to_numeric(raw_source, errors="coerce")
            percentile = pd.to_numeric(pct_source, errors="coerce")
            finite_raw = np.isfinite(raw)
            finite_pct = np.isfinite(percentile)
            violation = (
                (available & (~finite_raw | ~finite_pct))
                | (~available & (raw_source.notna() | pct_source.notna()))
            )
            leak = int(violation.sum())
            counters[f"{dimension}_missingness_violation"] += leak
            if leak:
                failures.append(
                    f"{target}: {dimension} missingness violations={leak}"
                )
            range_failure = int(
                (
                    finite_pct
                    & (percentile.lt(0.0) | percentile.gt(1.0))
                ).sum()
            )
            if range_failure:
                failures.append(
                    f"{target}: {dimension} percentile range failures="
                    f"{range_failure}"
                )
            reasons = traversals[f"{dimension}_unavailable_reason"].fillna(
                ""
            ).astype(str)
            reason_failure = int(
                ((available & reasons.ne("")) | (~available & reasons.eq(""))).sum()
            )
            if reason_failure:
                failures.append(
                    f"{target}: {dimension} availability reason failures="
                    f"{reason_failure}"
                )
            levels = traversals[f"{dimension}_cdf_level_used"].fillna(
                "unresolved"
            ).astype(str)
            support = pd.to_numeric(
                traversals[f"{dimension}_cdf_sample_size"], errors="coerce"
            )
            metadata_failure = int(
                (
                    available
                    & (
                        levels.eq("unresolved")
                        | ~np.isfinite(support)
                        | support.le(0)
                    )
                ).sum()
            )
            if metadata_failure:
                failures.append(
                    f"{target}: {dimension} CDF metadata failures="
                    f"{metadata_failure}"
                )
            expected_pct, expected_level, expected_support = {
                "lcs": models.lcs,
                "rts": models.rts,
            }[dimension].choose_cdf(
                traversals,
                f"{dimension}_raw",
                minimum_support=int(
                    config.section("normalization")[
                        "minimum_cohort_support"
                    ]
                ),
            )
            percentile_match = np.isclose(
                percentile,
                expected_pct,
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            )
            cdf_identity_failure = (
                ~percentile_match
                | levels.ne(pd.Series(expected_level, index=levels.index))
                | ~support.eq(
                    pd.Series(expected_support, index=support.index)
                )
            )
            if cdf_identity_failure.any():
                failures.append(
                    f"{target}: {dimension} frozen-CDF identity failures="
                    f"{int(cdf_identity_failure.sum())}"
                )

        if len(traversals):
            lcs_available = traversals["lcs_available"].eq(True)
            lcs_components = [
                "crawl_time_share",
                "stop_time_share",
                "speed_cv_bounded",
                "acceleration_rms_bounded",
            ]
            weights = np.asarray(
                [
                    float(
                        config.section("lcs")["components"][component][
                            "weight"
                        ]
                    )
                    for component in lcs_components
                ],
                dtype=np.float64,
            )
            component_values = traversals[lcs_components].apply(
                pd.to_numeric, errors="coerce"
            ).to_numpy(dtype=np.float64)
            expected_lcs = component_values @ weights
            actual_lcs = pd.to_numeric(
                traversals["lcs_raw"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            lcs_formula_failure = lcs_available.to_numpy() & ~np.isclose(
                actual_lcs,
                expected_lcs,
                atol=1e-12,
                rtol=1e-12,
            )
            if lcs_formula_failure.any():
                failures.append(
                    f"{target}: LCS formula failures="
                    f"{int(lcs_formula_failure.sum())}"
                )
            discontinuous_available = (
                traversals["discontinuous_direct_window"].eq(True)
                & lcs_available
            )
            if discontinuous_available.any():
                failures.append(
                    f"{target}: LCS available across discontinuous windows="
                    f"{int(discontinuous_available.sum())}"
                )

            rts_available = traversals["rts_available"].eq(True)
            pace_series = pd.to_numeric(
                traversals["observed_sec_per_m"], errors="coerce"
            )
            pace = pace_series.to_numpy(dtype=np.float64)
            direct_time = pd.to_numeric(
                traversals["direct_observed_time_s"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            direct_distance = pd.to_numeric(
                traversals["direct_observed_distance_m"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            expected_pace = np.full(len(traversals), np.nan, dtype=np.float64)
            pace_eligible = (
                (direct_time >= float(
                    config.section("rts")[
                        "minimum_direct_observed_time_s"
                    ]
                ))
                & (direct_distance >= float(
                    config.section("rts")[
                        "minimum_direct_observed_distance_m"
                    ]
                ))
                & ~traversals["direct_distance_exceeds_allocated"]
                .eq(True)
                .to_numpy(dtype=bool)
                & ~traversals["discontinuous_direct_window"].eq(
                    True
                ).to_numpy(dtype=bool)
                & traversals["rts_direct_speed_valid"]
                .fillna(False)
                .astype(bool)
                .to_numpy(dtype=bool)
            )
            measurement_available = traversals[
                "rts_measurement_available"
            ].eq(True).to_numpy(dtype=bool)
            if np.any(measurement_available != pace_eligible):
                failures.append(
                    f"{target}: RTS measurement availability failures="
                    f"{int(np.sum(measurement_available != pace_eligible))}"
                )
            measurement_reasons = traversals[
                "rts_measurement_unavailable_reason"
            ].fillna("").astype(str)
            invalid_measurement_reason = (
                measurement_available & measurement_reasons.ne("")
            ) | (~measurement_available & measurement_reasons.eq(""))
            if invalid_measurement_reason.any():
                failures.append(
                    f"{target}: RTS measurement reason failures="
                    f"{int(invalid_measurement_reason.sum())}"
                )
            expected_pace[pace_eligible] = (
                direct_time[pace_eligible] / direct_distance[pace_eligible]
            )
            pace_failure = ~np.isclose(
                pace,
                expected_pace,
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            )
            if pace_failure.any():
                failures.append(
                    f"{target}: observed_sec_per_m formula failures="
                    f"{int(pace_failure.sum())}"
                )

            (
                expected_reference,
                expected_reference_level,
                expected_reference_support,
            ) = models.reference.choose_quantile(
                traversals,
                probability=float(
                    config.section("reference")["quantile"]
                ),
                minimum_support=int(
                    config.section("rts")[
                        "minimum_reference_sample_size"
                    ]
                ),
                leave_one_out_value_column=(
                    "observed_sec_per_m"
                    if ref.split == "train"
                    else None
                ),
            )
            reference_series = pd.to_numeric(
                traversals["reference_sec_per_m"], errors="coerce"
            )
            reference = reference_series.to_numpy(dtype=np.float64)
            reference_level = traversals["reference_level_used"].fillna(
                "unresolved"
            ).astype(str)
            reference_support = pd.to_numeric(
                traversals["reference_sample_size"], errors="coerce"
            )
            reference_failure = (
                ~np.isclose(
                    reference,
                    expected_reference,
                    atol=1e-12,
                    rtol=1e-12,
                    equal_nan=True,
                )
                | reference_level.ne(
                    pd.Series(
                        expected_reference_level,
                        index=reference_level.index,
                    )
                )
                | ~reference_support.eq(
                    pd.Series(
                        expected_reference_support,
                        index=reference_support.index,
                    )
                )
            )
            if reference_failure.any():
                failures.append(
                    f"{target}: frozen-reference identity failures="
                    f"{int(reference_failure.sum())}"
                )

            expected_excess = np.maximum(
                pace / expected_reference - 1.0,
                0.0,
            )
            actual_excess = pd.to_numeric(
                traversals["excess_time_ratio"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            excess_failure = ~np.isclose(
                actual_excess,
                expected_excess,
                atol=1e-12,
                rtol=1e-12,
                equal_nan=True,
            )
            if excess_failure.any():
                failures.append(
                    f"{target}: excess_time_ratio formula failures="
                    f"{int(excess_failure.sum())}"
                )
            expected_rts = expected_excess / (1.0 + expected_excess)
            actual_rts = pd.to_numeric(
                traversals["rts_raw"], errors="coerce"
            ).to_numpy(dtype=np.float64)
            rts_formula_failure = rts_available.to_numpy() & (
                ~np.isfinite(pace)
                | ~np.isfinite(reference)
                | (reference <= 0)
                | ~np.isclose(
                    actual_rts,
                    expected_rts,
                    atol=1e-12,
                    rtol=1e-12,
                )
            )
            if rts_formula_failure.any():
                failures.append(
                    f"{target}: RTS formula failures="
                    f"{int(rts_formula_failure.sum())}"
                )
            reference_identity_failure = (
                ~traversals["reference_model_id"].astype(str).eq(
                    models.model_id
                )
                | ~traversals["reference_fit_manifest_id"].astype(str).eq(
                    str(models.manifest["source_manifest_id"])
                )
            )
            if reference_identity_failure.any():
                failures.append(
                    f"{target}: RTS reference identity failures="
                    f"{int(reference_identity_failure.sum())}"
                )
            for dimension, dimension_available in (
                ("lcs", lcs_available),
                ("rts", rts_available),
            ):
                tail_event = traversals[f"{dimension}_tail_event"].astype(
                    "boolean"
                )
                expected_tail = pd.Series(
                    pd.NA, index=traversals.index, dtype="boolean"
                )
                expected_tail.loc[dimension_available] = pd.to_numeric(
                    traversals.loc[
                        dimension_available, f"{dimension}_pct"
                    ],
                    errors="coerce",
                ).ge(
                    float(
                        config.section("aggregation")[
                            "tail_percentile_threshold"
                        ]
                    )
                ).astype(bool)
                if not tail_event.equals(expected_tail):
                    failures.append(
                        f"{target}: {dimension.upper()} tail-event mismatch"
                    )

        for unavailable in ("gns", "iis", "pmis"):
            try:
                available = _strict_boolean(
                    traversals[f"{unavailable}_available"],
                    f"traversal_labels.{unavailable}_available",
                )
            except ContractError as exc:
                failures.append(f"{target}: {exc}")
                available = pd.Series(False, index=traversals.index)
            leak = int(available.sum()) + _nonmissing(
                traversals,
                [f"{unavailable}_raw", f"{unavailable}_pct"],
            )
            counters[f"{unavailable}_availability_violation"] += leak
            if leak:
                failures.append(f"{target}: {unavailable} must be unavailable")
            expected_reason = {
                "gns": "EDGE_STATIC_FEATURE_EXTENSION_NOT_FITTED",
                "iis": str(config.section("iis")["unavailable_reason"]),
                "pmis": str(config.section("pmis")["unavailable_reason"]),
            }[unavailable]
            reason_failure = int(
                (
                    ~traversals[
                        f"{unavailable}_unavailable_reason"
                    ].astype(str).eq(expected_reason)
                ).sum()
            )
            if reason_failure:
                failures.append(
                    f"{target}: {unavailable} reason failures="
                    f"{reason_failure}"
                )
        forbidden = sorted(
            column
            for column in orders
            if column.startswith("core_composite_")
            and column != "core_composite_status"
        )
        if forbidden:
            failures.append(f"{target}: forbidden composite columns {forbidden}")
        if not orders["core_composite_status"].astype(str).eq(
            str(config.data["core_composite_status"])
        ).all():
            failures.append(f"{target}: core composite status mismatch")
        try:
            movement_iis = _strict_boolean(
                movements["iis_available"],
                "movement_context.iis_available",
            )
            if movement_iis.any():
                failures.append(f"{target}: movement IIS unexpectedly available")
            movement_reason_failure = int(
                (
                    ~movements["iis_unavailable_reason"].astype(str).eq(
                        str(config.section("iis")["unavailable_reason"])
                    )
                ).sum()
            )
            if movement_reason_failure:
                failures.append(
                    f"{target}: movement IIS reason failures="
                    f"{movement_reason_failure}"
                )
        except ContractError as exc:
            failures.append(f"{target}: {exc}")

        for dimension in ("lcs", "rts"):
            try:
                order_available = _strict_boolean(
                    orders[f"{dimension}_available"],
                    f"order_labels.{dimension}_available",
                )
            except ContractError as exc:
                failures.append(f"{target}: {exc}")
                order_available = pd.Series(False, index=orders.index)
            numeric_columns = [
                f"{dimension}_{name}"
                for name in ("mean", "max", "tail", "persistence")
            ]
            numeric = orders[numeric_columns].apply(
                pd.to_numeric, errors="coerce"
            )
            unavailable_leak = (
                ~order_available & numeric.notna().any(axis=1)
            )
            required_when_available = np.isfinite(numeric[
                [
                    f"{dimension}_mean",
                    f"{dimension}_max",
                    f"{dimension}_persistence",
                ]
            ]).all(axis=1)
            available_missing = order_available & ~required_when_available
            if unavailable_leak.any() or available_missing.any():
                failures.append(
                    f"{target}: {dimension} order missingness failures="
                    f"{int(unavailable_leak.sum() + available_missing.sum())}"
                )
            tail_present = orders[
                f"{dimension}_tail_event_present"
            ].astype("boolean")
            invalid_tail_state = (
                order_available & tail_present.isna()
            ) | (~order_available & tail_present.notna())
            if invalid_tail_state.any():
                failures.append(
                    f"{target}: {dimension} order tail-state failures="
                    f"{int(invalid_tail_state.sum())}"
                )

        for unavailable in ("gns", "iis", "pmis"):
            try:
                order_available = _strict_boolean(
                    orders[f"{unavailable}_available"],
                    f"order_labels.{unavailable}_available",
                )
            except ContractError as exc:
                failures.append(f"{target}: {exc}")
                order_available = pd.Series(False, index=orders.index)
            value_columns = [
                f"{unavailable}_{name}"
                for name in ("mean", "max", "tail", "persistence")
            ]
            leak = int(order_available.sum()) + _nonmissing(
                orders, value_columns
            )
            if leak:
                failures.append(
                    f"{target}: order {unavailable} must be unavailable"
                )

        input_order_ids = set(bucket.order_base["order_id"].astype(str))
        for name, frame in {
            "order_labels": orders,
            "order_label_quality": quality,
        }.items():
            output_ids = set(frame["order_id"].astype(str))
            if output_ids != input_order_ids:
                failures.append(f"{target}: {name} order set mismatch")
        for order_id in input_order_ids:
            previous = seen_orders.get(order_id)
            partition = (ref.split, ref.date, ref.bucket)
            if previous is not None:
                failures.append(
                    f"order_id {order_id} appears in {previous} and {partition}"
                )
            seen_orders[order_id] = partition

        try:
            expected_orders, expected_quality = aggregate_order_labels(
                traversals,
                bucket.route_parts,
                bucket.link_traversals,
                bucket.order_base,
                config,
            )
        except ContractError as exc:
            failures.append(f"{target}: cannot regenerate order aggregation: {exc}")
        else:
            for name, actual, expected in (
                ("order_labels", orders, expected_orders),
                ("order_label_quality", quality, expected_quality),
            ):
                mismatches = _frame_value_mismatches(
                    actual,
                    expected,
                    keys=["split", "date", "order_id"],
                )
                if mismatches:
                    failures.append(
                        f"{target}: {name} differs from regenerated aggregation: "
                        f"{mismatches}"
                    )

    for product, values in schema_hashes.items():
        if len(values) != 1:
            failures.append(
                f"{product} has {len(values)} schemas across output buckets"
            )

    expected_dates = {
        "train": set(config.train_dates),
        "validation": set(config.validation_dates),
        "test": {config.test_date},
    }
    for split, expected in expected_dates.items():
        if dates.get(split, set()) != expected:
            failures.append(
                f"{split} output dates mismatch: "
                f"actual={sorted(dates.get(split, set()))}"
            )

    summary_path = output_root_path / "stage1_v3_summary.json"
    if not summary_path.is_file():
        failures.append("missing stage1_v3_summary.json")
    else:
        summary = _read_json(summary_path)
        for name, expected in {
            "schema_version": OUTPUT_SUMMARY_SCHEMA_VERSION,
            "engineering_status": "PASS",
            "scientific_status": "NOT_VALIDATED",
            "model_id": models.model_id,
            "source_manifest_id": models.manifest["source_manifest_id"],
            "stage0_freeze_manifest_sha": freeze.manifest_sha,
            "config_sha": config.digest,
            "stage1_code_sha": code_sha,
            "bucket_count": len(refs),
        }.items():
            if summary.get(name) != expected:
                failures.append(f"summary mismatch for {name}")
        expected_counts = {
            product: int(counters[f"{product}_rows"])
            for product in OUTPUT_PRODUCTS
        }
        if summary.get("product_row_counts") != expected_counts:
            failures.append("summary product row counts mismatch")
        expected_summary_dates = {
            "train": list(config.train_dates),
            "validation": list(config.validation_dates),
            "test": [config.test_date],
        }
        if summary.get("dates") != expected_summary_dates:
            failures.append("summary dates mismatch")
        execution_counts = [
            summary.get("transformed_bucket_count"),
            summary.get("resumed_bucket_count"),
        ]
        if (
            any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
                for value in execution_counts
            )
            or sum(
                value
                for value in execution_counts
                if isinstance(value, int) and not isinstance(value, bool)
            )
            != len(refs)
        ):
            failures.append(
                "summary transformed/resumed bucket counts are inconsistent"
            )

    return {
        "schema_version": "stage1_v3_audit.1",
        "engineering_status": "PASS" if not failures else "FAIL",
        "scientific_status": "NOT_VALIDATED",
        "model_id": models.model_id,
        "config_sha": config.digest,
        "stage1_code_sha": code_sha,
        "bucket_count": len(refs),
        "runtime_s": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb,
        "unique_order_count": len(seen_orders),
        "counters": dict(sorted(counters.items())),
        "schema_hashes": {
            product: sorted(values)
            for product, values in sorted(schema_hashes.items())
        },
        "failures": failures,
        "limitations": [
            "engineering PASS does not establish construct or predictive validity",
            "LCS and RTS thresholds remain review candidates",
            "IIS and PMIS are unavailable",
            "GNS is not part of the dynamic v3 label pipeline",
        ],
    }
