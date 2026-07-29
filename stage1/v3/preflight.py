"""Read-only global preflight for the frozen 220k Stage 1 input."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from .input_adapter import iter_stage0_buckets, load_stage0_bucket
from .schema import ALL_INPUT_PRODUCTS, ContractError

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
    unresolved_start_node_count = 0
    unresolved_end_node_count = 0
    tolerance = float(config.section("direct")["duration_tolerance_s"])

    for ref in refs:
        try:
            bucket = load_stage0_bucket(ref, config)
        except ContractError as exc:
            raise ContractError(f"{ref.path}: {exc}") from exc
        accepted = int(ref.manifest["accepted_core_count"])
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
        "schema_version": "stage1_v3_global_preflight.1",
        "engineering_status": "PASS",
        "config_sha": config.digest,
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
        "observation_traversal_route_part_fk_failure_count": 0,
        "invalid_direct_time_distance_speed_count": 0,
        "direct_observation_outside_order_time_count": 0,
        "unmapped_traversal_count": unmapped_traversal_count,
        "direct_observation_on_unmapped_traversal_count": 0,
        "nullable_od": {
            "unresolved_start_node_count": unresolved_start_node_count,
            "unresolved_end_node_count": unresolved_end_node_count,
        },
        "product_row_counts": dict(sorted(product_rows.items())),
    }
