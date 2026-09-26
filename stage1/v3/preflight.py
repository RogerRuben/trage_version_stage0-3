"""Read-only global preflight for the frozen 220k Stage 1 input."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .input_adapter import (
    derive_movement_direction_context,
    iter_stage0_buckets,
    load_stage0_bucket,
)
from .io import (
    canonical_json_bytes,
    sha256_bytes,
    sha256_file,
    stage1_v3_code_identity,
)
from .schema import ALL_INPUT_PRODUCTS, ContractError
from .support import DirectedCatalogBuilder, catalog_region_statistics

if TYPE_CHECKING:
    from .config import Stage1V3Config


def run_global_preflight(
    input_root: str | Path,
    config: "Stage1V3Config",
) -> dict[str, Any]:
    """Validate all buckets before any Stage 1 label model is fitted."""

    refs = list(iter_stage0_buckets(input_root, config))
    expected = config.section("preflight")
    split_counts: Counter[str] = Counter()
    product_rows: Counter[str] = Counter()
    seen_orders: dict[str, tuple[str, str, int]] = {}
    direction_counts: Counter[str] = Counter()
    unmapped_traversal_count = 0
    direct_distance_exceed_counts: Counter[str] = Counter()
    direct_distance_exceed_orders: set[str] = set()
    support_key_counts: Counter[str] = Counter()
    canonical_highway_missing_counts: Counter[str] = Counter()
    unresolved_start_node_count = 0
    unresolved_end_node_count = 0
    tolerance = float(config.section("direct")["duration_tolerance_s"])
    distance_tolerance = float(
        config.section("direct")["distance_identity_tolerance_m"]
    )
    maximum_direct_speed = float(
        config.section("rts")["maximum_direct_speed_mps"]
    )
    catalog_builders = {
        "global": DirectedCatalogBuilder(),
        "train": DirectedCatalogBuilder(),
        "validation": DirectedCatalogBuilder(),
        "test": DirectedCatalogBuilder(),
    }
    movement_counts: Counter[str] = Counter()
    validated_buckets: list[dict[str, Any]] = []

    for ref in refs:
        try:
            bucket = load_stage0_bucket(ref, config)
        except ContractError as exc:
            raise ContractError(f"{ref.path}: {exc}") from exc
        accepted = int(ref.manifest["accepted_core_count"])
        try:
            catalog_builders["global"].update(bucket.route_parts)
            catalog_builders[ref.split].update(bucket.route_parts)
        except ContractError as exc:
            raise ContractError(f"{ref.path}: catalog conflict: {exc}") from exc
        split_counts[ref.split] += accepted
        for product in ALL_INPUT_PRODUCTS:
            product_rows[product] += int(
                ref.manifest["product_row_counts"][product]
            )

        for order_id in bucket.order_base["order_id"].astype(str):
            partition = (ref.split, ref.date, ref.bucket)
            previous = seen_orders.get(order_id)
            if previous is not None:
                raise ContractError(
                    f"global order_id duplicate: {order_id} in "
                    f"{previous} and {partition}"
                )
            seen_orders[order_id] = partition

        unresolved_start_node_count += int(
            bucket.order_base["start_node"].isna().sum()
        )
        unresolved_end_node_count += int(
            bucket.order_base["end_node"].isna().sum()
        )
        unmapped_traversal_count += int(
            (
                ~bucket.link_traversals[
                    "canonical_mapping_available"
                ].eq(True)
            ).sum()
        )

        observations = bucket.link_interval_observations
        observation_support = observations.merge(
            bucket.link_traversals[
                [
                    "order_id",
                    "traversal_id",
                    "canonical_highway",
                ]
            ],
            on=["order_id", "traversal_id"],
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if observation_support["_merge"].ne("both").any():
            raise ContractError(
                "support preflight found an orphan direct traversal"
            )
        highway_missing = (
            observation_support["canonical_highway"].isna()
            | observation_support["canonical_highway"]
            .astype("string")
            .str.strip()
            .isin(["", "nan", "<NA>"])
        )
        canonical_highway_missing_counts[
            "direct_observation_count"
        ] += int(highway_missing.sum())
        traversal_highway_missing = (
            bucket.link_traversals["canonical_highway"].isna()
            | bucket.link_traversals["canonical_highway"]
            .astype("string")
            .str.strip()
            .isin(["", "nan", "<NA>"])
        )
        canonical_highway_missing_counts[
            "traversal_count"
        ] += int(traversal_highway_missing.sum())
        finite_start = np.isfinite(
            pd.to_numeric(
                observations["interval_start_time"], errors="coerce"
            )
        )
        support_key_available = (
            observation_support["observed_directed_edge_uid"].notna()
            & ~highway_missing
            & observation_support["observed_from_node"].notna()
            & observation_support["observed_to_node"].notna()
            & finite_start
        )
        support_key_counts["constructable_count"] += int(
            support_key_available.sum()
        )
        support_key_counts["failure_count"] += int(
            (~support_key_available).sum()
        )

        direct_by_traversal = (
            observations.assign(
                _distance=pd.to_numeric(
                    observations["observed_distance_m"], errors="coerce"
                )
            )
            .groupby(
                ["order_id", "traversal_id"], sort=False, dropna=False
            )
            .agg(
                direct_distance_m=("_distance", "sum"),
                direct_interval_count=("gps_interval_id", "size"),
            )
            .reset_index()
        )
        traversal_allocated = bucket.link_traversals[
            ["order_id", "traversal_id", "allocated_distance_m"]
        ].copy()
        traversal_allocated["allocated_distance_m"] = pd.to_numeric(
            traversal_allocated["allocated_distance_m"], errors="coerce"
        )
        distance_audit = direct_by_traversal.merge(
            traversal_allocated,
            on=["order_id", "traversal_id"],
            how="left",
            validate="one_to_one",
            indicator=True,
        )
        if distance_audit["_merge"].ne("both").any():
            raise ContractError(
                "distance preflight found an orphan direct traversal"
            )
        exceeds = distance_audit["direct_distance_m"].gt(
            distance_audit["allocated_distance_m"] + distance_tolerance
        )
        direct_distance_exceed_counts["traversal_count"] += int(
            exceeds.sum()
        )
        direct_distance_exceed_counts["interval_count"] += int(
            distance_audit.loc[exceeds, "direct_interval_count"].sum()
        )
        direct_distance_exceed_orders.update(
            distance_audit.loc[exceeds, "order_id"].astype(str)
        )
        direction_movements = derive_movement_direction_context(
            bucket.turn_movements,
            bucket.route_parts,
        )
        movement_counts["movement_count"] += len(direction_movements)
        movement_counts["direction_mapped_count"] += int(
            direction_movements[
                "movement_direction_mapping_available"
            ].sum()
        )
        movement_counts["lineage_only_count"] += int(
            direction_movements["movement_lineage_only"].sum()
        )
        direction = observations["observed_direction"].astype("string")
        direction_counts["direct_observation_count"] += len(observations)
        direction_counts["actual_F_count"] += int(direction.eq("F").sum())
        direction_counts["actual_R_count"] += int(direction.eq("R").sum())
        suffix = observations["canonical_edge_uid"].astype("string").str.extract(
            r":([FR])$", expand=False
        )
        direction_counts["uid_direction_mismatch_count"] += int(
            suffix.ne(direction).sum()
        )
        direction_counts["synthetic_reverse_edge_count"] += int(
            observations["synthetic_reverse_edge"].sum()
        )
        direction_counts["osm_direction_disagreement_count"] += int(
            observations["osm_direction_disagreement"].sum()
        )
        direction_counts["actual_direction_null_count"] += int(
            observations["observed_direction"].isna().sum()
        )
        direction_counts["directed_identity_null_count"] += int(
            observations["observed_directed_edge_uid"].isna().sum()
        )
        direction_counts["maximum_speed_exceeded_count"] += int(
            pd.to_numeric(
                observations["observed_speed_mps"], errors="coerce"
            ).gt(maximum_direct_speed).sum()
        )
        validated_buckets.append(
            {
                "split": ref.split,
                "date": ref.date,
                "bucket": ref.bucket,
                "manifest_sha": sha256_file(ref.path / "manifest.json"),
                "trusted_product_size_bytes": {
                    product: int(
                        (ref.path / f"{product}.parquet").stat().st_size
                    )
                    for product in (
                        "route_parts",
                        "link_traversals",
                        "link_interval_observations",
                    )
                },
                "trusted_product_row_counts": {
                    product: int(
                        ref.manifest["product_row_counts"][product]
                    )
                    for product in (
                        "route_parts",
                        "link_traversals",
                        "link_interval_observations",
                    )
                },
                "trusted_product_sha256": {
                    product: sha256_file(
                        ref.path / f"{product}.parquet"
                    )
                    for product in (
                        "route_parts",
                        "link_traversals",
                        "link_interval_observations",
                    )
                },
            }
        )

        bounds = bucket.order_base[
            ["order_id", "departure_time", "arrival_time"]
        ]
        timed = observations.merge(
            bounds,
            on="order_id",
            how="left",
            validate="many_to_one",
            indicator=True,
        )
        if timed["_merge"].ne("both").any():
            raise ContractError("direct observation has an orphan order_id")
        start = pd.to_numeric(timed["interval_start_time"], errors="coerce")
        end = pd.to_numeric(timed["interval_end_time"], errors="coerce")
        departure = pd.to_numeric(timed["departure_time"], errors="coerce")
        arrival = pd.to_numeric(timed["arrival_time"], errors="coerce")
        outside = start.lt(departure - tolerance) | end.gt(arrival + tolerance)
        if outside.any():
            raise ContractError(
                "direct observations outside order time bounds: "
                f"{int(outside.sum())}"
            )

    actual_order_count = len(seen_orders)
    catalogs = {
        name: builder.finalize()
        for name, builder in catalog_builders.items()
    }
    edge_sets = {
        name: set(catalog["observed_directed_edge_uid"].astype(str))
        for name, catalog in catalogs.items()
    }
    validation_unseen = edge_sets["validation"] - edge_sets["train"]
    test_unseen = edge_sets["test"] - edge_sets["train"]
    validated_input_manifest_id = sha256_bytes(
        canonical_json_bytes(validated_buckets)
    )
    failures: list[str] = []
    if actual_order_count != int(expected["expected_order_count"]):
        failures.append(
            f"orders expected={expected['expected_order_count']} "
            f"actual={actual_order_count}"
        )
    expected_splits = expected["expected_split_counts"]
    for split, expected_count in expected_splits.items():
        actual = int(split_counts[split])
        if actual != int(expected_count):
            failures.append(
                f"{split} expected={expected_count} actual={actual}"
            )
    expected_direction = expected["expected_direction_audit"]
    direction_mapping = {
        "direct_observation_count": "direct_observation_count",
        "actual_F_count": "actual_F_count",
        "actual_R_count": "actual_R_count",
        "uid_direction_mismatch_count": "uid_direction_mismatch_count",
        "legacy_osm_against_count": "osm_direction_disagreement_count",
    }
    for expected_name, actual_name in direction_mapping.items():
        actual = int(direction_counts[actual_name])
        expected_count = int(expected_direction[expected_name])
        if actual != expected_count:
            failures.append(
                f"{actual_name} expected={expected_count} actual={actual}"
            )
    if (
        direction_counts["synthetic_reverse_edge_count"]
        != expected_direction["uid_direction_mismatch_count"]
    ):
        failures.append(
            "synthetic reverse labels do not exactly cover UID/direction "
            "mismatches"
        )
    if direction_counts["actual_direction_null_count"]:
        failures.append("direct observations contain null actual direction")
    if direction_counts["directed_identity_null_count"]:
        failures.append("direct observations contain null directed identity")
    expected_unmapped = int(expected["expected_unmapped_traversal_count"])
    if unmapped_traversal_count != expected_unmapped:
        failures.append(
            f"unmapped traversals expected={expected_unmapped} "
            f"actual={unmapped_traversal_count}"
        )
    if failures:
        raise ContractError("Stage1 global preflight failed: " + "; ".join(failures))

    return {
        "schema_version": "stage1_v3_global_preflight.2",
        "engineering_status": "PASS",
        "input_root": str(Path(input_root).resolve()),
        "config_sha": config.digest,
        "stage1_code_sha": stage1_v3_code_identity(),
        "stage0_release": config.section("stage0_release"),
        "accepted_order_count": actual_order_count,
        "split_order_counts": dict(sorted(split_counts.items())),
        "global_order_id_unique": True,
        "split_overlap_count": 0,
        "orphan_product_order_id_count": 0,
        "product_primary_key_duplicate_count": 0,
        "partition_identity_failure_count": 0,
        "stage1_core_ineligible_count": 0,
        "direct_observation_count": int(
            direction_counts["direct_observation_count"]
        ),
        "actual_F_count": int(direction_counts["actual_F_count"]),
        "actual_R_count": int(direction_counts["actual_R_count"]),
        "uid_direction_mismatch_count": int(
            direction_counts["uid_direction_mismatch_count"]
        ),
        "synthetic_reverse_edge_count": int(
            direction_counts["synthetic_reverse_edge_count"]
        ),
        "osm_direction_disagreement_count": int(
            direction_counts["osm_direction_disagreement_count"]
        ),
        "actual_direction_null_count": 0,
        "directed_identity_null_count": 0,
        "maximum_direct_speed_mps": maximum_direct_speed,
        "maximum_direct_speed_exceeded_count": int(
            direction_counts["maximum_speed_exceeded_count"]
        ),
        "direct_distance_exceeds_allocated": {
            "interval_count": int(
                direct_distance_exceed_counts["interval_count"]
            ),
            "traversal_count": int(
                direct_distance_exceed_counts["traversal_count"]
            ),
            "order_count": len(direct_distance_exceed_orders),
            "distance_identity_tolerance_m": distance_tolerance,
            "input_direct_labels_deleted_count": 0,
        },
        "canonical_highway_missing": {
            "direct_observation_count": int(
                canonical_highway_missing_counts[
                    "direct_observation_count"
                ]
            ),
            "traversal_count": int(
                canonical_highway_missing_counts["traversal_count"]
            ),
        },
        "support_key_constructability": {
            "constructable_count": int(
                support_key_counts["constructable_count"]
            ),
            "failure_count": int(support_key_counts["failure_count"]),
            "key_fields": [
                "observed_directed_edge_uid",
                "canonical_highway",
                "observed_from_node",
                "observed_to_node",
                "interval_start_time",
            ],
        },
        "observation_traversal_route_part_fk_failure_count": 0,
        "invalid_direct_time_distance_speed_count": 0,
        "direct_observation_outside_order_time_count": 0,
        "unmapped_traversal_count": unmapped_traversal_count,
        "direct_observation_on_unmapped_traversal_count": 0,
        "directed_catalog": {
            "catalog_conflict_count": 0,
            "directed_edge_count": int(len(catalogs["global"])),
            "synthetic_reverse_unique_edge_count": int(
                catalogs["global"]["synthetic_reverse_edge"].sum()
            ),
            "train_seen_directed_edge_count": int(len(catalogs["train"])),
            "validation_directed_edge_count": int(
                len(catalogs["validation"])
            ),
            "test_directed_edge_count": int(len(catalogs["test"])),
            "validation_unseen_directed_edge_count": int(
                len(validation_unseen)
            ),
            "test_unseen_directed_edge_count": int(len(test_unseen)),
            "model_catalog_fit_scope": "train_only",
            "upper_region_usage": "audit_only_not_a_model_fallback",
            "train_upper_region_statistics": catalog_region_statistics(
                catalogs["train"]
            ),
            "global_upper_region_statistics": catalog_region_statistics(
                catalogs["global"]
            ),
        },
        "movement_direction": {
            "movement_count": int(movement_counts["movement_count"]),
            "direction_mapped_count": int(
                movement_counts["direction_mapped_count"]
            ),
            "lineage_only_count": int(
                movement_counts["lineage_only_count"]
            ),
        },
        "validated_input_manifest_id": validated_input_manifest_id,
        "validated_buckets": validated_buckets,
        "nullable_od": {
            "unresolved_start_node_count": unresolved_start_node_count,
            "unresolved_end_node_count": unresolved_end_node_count,
        },
        "product_row_counts": dict(sorted(product_rows.items())),
    }


def validate_preflight_for_fit(
    report_path: str | Path,
    input_root: str | Path,
    refs: list[Any],
    config: "Stage1V3Config",
) -> dict[str, dict[str, Any]]:
    """Bind fit to the exact products validated by a PASS preflight."""

    path = Path(report_path)
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read Stage1 preflight report {path}: {exc}") from exc
    if not isinstance(report, dict):
        raise ContractError("Stage1 preflight report must be a JSON object")
    if report.get("engineering_status") != "PASS":
        raise ContractError("Stage1 fit requires a PASS global preflight")
    if report.get("config_sha") != config.digest:
        raise ContractError("Stage1 preflight config SHA differs from fit config")
    if report.get("stage1_code_sha") != stage1_v3_code_identity():
        raise ContractError("Stage1 preflight code SHA differs from fit code")
    if report.get("stage0_release") != config.section("stage0_release"):
        raise ContractError("Stage1 preflight Stage0 release differs from fit")
    if Path(str(report.get("input_root", ""))).resolve() != Path(
        input_root
    ).resolve():
        raise ContractError("Stage1 preflight input root differs from fit input")

    records = report.get("validated_buckets")
    if not isinstance(records, list):
        raise ContractError("Stage1 preflight has no validated bucket manifest")
    indexed: dict[str, dict[str, Any]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ContractError("invalid Stage1 preflight bucket record")
        key = (
            f"{record.get('split')}/{record.get('date')}/"
            f"{int(record.get('bucket')):05d}"
        )
        if key in indexed:
            raise ContractError(f"duplicate preflight bucket record: {key}")
        indexed[key] = record
    expected_keys = {
        f"{ref.split}/{ref.date}/{ref.bucket:05d}" for ref in refs
    }
    if set(indexed) != expected_keys:
        raise ContractError(
            "Stage1 preflight bucket set differs from fit input"
        )
    identity = sha256_bytes(canonical_json_bytes(records))
    if report.get("validated_input_manifest_id") != identity:
        raise ContractError("Stage1 preflight validated manifest ID is corrupt")

    trusted_products = (
        "route_parts",
        "link_traversals",
        "link_interval_observations",
    )
    for ref in refs:
        key = f"{ref.split}/{ref.date}/{ref.bucket:05d}"
        record = indexed[key]
        if record.get("manifest_sha") != sha256_file(
            ref.path / "manifest.json"
        ):
            raise ContractError(f"bucket manifest changed after preflight: {ref.path}")
        for product in trusted_products:
            product_path = ref.path / f"{product}.parquet"
            sizes = record.get("trusted_product_size_bytes", {})
            counts = record.get("trusted_product_row_counts", {})
            hashes = record.get("trusted_product_sha256", {})
            if (
                not product_path.is_file()
                or int(product_path.stat().st_size)
                != int(sizes.get(product, -1))
                or int(ref.manifest["product_row_counts"][product])
                != int(counts.get(product, -1))
                or sha256_file(product_path) != hashes.get(product)
            ):
                raise ContractError(
                    f"trusted fit product changed after preflight: {product_path}"
                )
    return indexed
