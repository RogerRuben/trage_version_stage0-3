"""Fixed-600 acceptance, route-only image queue, and final report."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .automated_audit import (
    _load_candidate_raw_points,
    _load_geometry_lookup,
    _plot_case,
)
from .config import Stage0V6Config
from .pipeline import PRODUCTS, _write_json, run_fixed_sample


def _read_product(root: Path, product: str) -> pd.DataFrame:
    files = sorted((root / product).glob("day=*/part=*.parquet"))
    return (
        pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)
        if files
        else pd.DataFrame()
    )


def _semantic_digest(frame: pd.DataFrame) -> str:
    if frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    normalized = frame.copy()
    normalized = normalized.reindex(sorted(normalized.columns), axis=1)
    for column in normalized:
        if normalized[column].dtype == "object":
            normalized[column] = normalized[column].map(
                lambda value: (
                    json.dumps(value, sort_keys=True, ensure_ascii=False)
                    if isinstance(value, (dict, list))
                    else str(value)
                )
            )
    sort_columns = [
        column
        for column in (
            "date",
            "order_id",
            "subtrace_id",
            "segment_id",
            "gps_interval_id",
            "route_sequence",
            "traversal_id",
            "original_point_seq",
        )
        if column in normalized
    ]
    if sort_columns:
        normalized.sort_values(sort_columns, kind="stable", inplace=True)
    hashes = pd.util.hash_pandas_object(
        normalized.reset_index(drop=True), index=False
    ).to_numpy(dtype="uint64")
    digest = hashlib.sha256()
    digest.update("|".join(normalized.columns).encode("utf-8"))
    digest.update(hashes.tobytes())
    return digest.hexdigest()


def _case_index(config: Stage0V6Config) -> pd.DataFrame:
    source = (
        config.repo_root
        / "stage0"
        / "output_v6"
        / "audit"
        / "manual_review_pack"
        / "index.csv"
    )
    index = pd.read_csv(source)
    return index[["case_index", "order_id", "date"]].copy()


def _regression_checks(
    config: Stage0V6Config, quality: pd.DataFrame
) -> tuple[list[dict[str, Any]], bool]:
    cases = _case_index(config)
    cases["order_id"] = cases.order_id.astype(str)
    quality = quality.copy()
    quality["order_id"] = quality.order_id.astype(str)
    joined = cases.merge(
        quality,
        on="order_id",
        how="left",
        validate="one_to_one",
    )
    settings = config.section("fixed600_regression")
    rows: list[dict[str, Any]] = []

    def add(name: str, case_numbers: list[int], predicate: Any) -> None:
        selected = joined.loc[joined.case_index.isin(case_numbers)].copy()
        for row in selected.itertuples(index=False):
            passed = bool(predicate(row))
            rows.append(
                {
                    "check": name,
                    "case_index": int(row.case_index),
                    "order_id": str(row.order_id),
                    "gps_status": str(getattr(row, "gps_status", "")),
                    "route_status": str(getattr(row, "route_status", "")),
                    "local_retry_changed_route": bool(
                        getattr(row, "local_retry_changed_route", False)
                    ),
                    "pass": passed,
                }
            )

    add(
        "route_fail",
        settings["route_fail"],
        lambda row: row.route_status == "route_fail",
    )
    add(
        "no_route_review",
        settings["no_route_review"],
        lambda row: row.route_status == "route_pass",
    )
    add(
        "discontinuity_not_fail",
        settings["discontinuity_not_fail"],
        lambda row: row.route_status != "route_fail",
    )
    add(
        "gps_local_outlier",
        settings["gps_local_outlier"],
        lambda row: row.gps_status == "local_outlier",
    )
    add(
        "parallel_fail_or_improved",
        settings["parallel_fail_or_improved"],
        lambda row: (
            row.route_status == "route_fail"
            or bool(row.local_retry_changed_route)
        ),
    )
    add(
        "route_partial",
        settings["route_partial"],
        lambda row: row.route_status == "route_partial",
    )
    add(
        "route_uncertain_if_not_improved",
        settings["route_uncertain_if_not_improved"],
        lambda row: (
            bool(row.local_retry_changed_route)
            or row.route_status == "route_uncertain"
        ),
    )
    return rows, bool(rows and all(row["pass"] for row in rows))


def _write_image_queues(
    config: Stage0V6Config, final_quality: pd.DataFrame
) -> dict[str, int]:
    output = config.path("output")
    hot = output / "hot"
    route_queue = final_quality.loc[
        final_quality.route_status.isin(
            ["route_fail", "route_partial", "route_uncertain"]
        )
    ].sort_values(
        ["route_status", "raw_gps_route_distance_p90_m"],
        ascending=[True, False],
        kind="stable",
    ).copy()
    route_queue.reset_index(drop=True, inplace=True)
    route_queue["case_index"] = np.arange(1, len(route_queue) + 1)
    route_queue["audit_class"] = route_queue.route_status
    route_queue["audit_score"] = pd.to_numeric(
        route_queue.raw_gps_route_distance_p90_m, errors="coerce"
    ).replace([np.inf, -np.inf], np.nan).fillna(9999.0)
    route_queue["primary_reason"] = route_queue.route_status
    route_queue["secondary_reasons"] = route_queue.get(
        "local_retry_reason", pd.Series("", index=route_queue.index)
    ).fillna("")
    route_queue["risk_window_from_seq"] = 0
    route_queue["risk_window_to_seq"] = 10**9
    route_queue["risk_window_reason"] = route_queue.route_status
    route_queue["snap_p90_m"] = route_queue.raw_gps_route_distance_p90_m
    route_queue["snap_p99_m"] = route_queue.raw_gps_route_distance_p99_m
    route_queue["snap_max_m"] = route_queue.raw_gps_route_distance_max_m
    route_queue["route_resolved_gps_ratio"] = 0.0
    route_queue["od_endpoint_error_m"] = np.nan
    route_queue["unresolved_time_share"] = route_queue.uncovered_time_share
    route_queue["direct_observed_time_share"] = 0.0
    route_queue["canonical_mapping_share"] = np.where(
        route_queue.canonical_status.isin(["unique", "chain_resolved"]),
        1.0,
        0.0,
    )
    route_queue["v5_v6_edge_jaccard"] = np.nan
    target = output / "audit" / "route_review_queue"
    images = target / "images"
    images.mkdir(parents=True, exist_ok=True)
    expected_image_names = {
        f"case_{int(row.case_index):03d}_{row.order_id}.svg"
        for row in route_queue.itertuples(index=False)
    }
    for stale in images.glob("case_*.svg"):
        if stale.name not in expected_image_names:
            stale.unlink()
    raw = _load_candidate_raw_points(
        config, set(route_queue.order_id.astype(str))
    )
    routes = _read_product(hot, "route_parts")
    selected_ids = set(route_queue.order_id.astype(str))
    routes = routes.loc[routes.order_id.astype(str).isin(selected_ids)]
    v5_files = sorted(
        config.path("v5_output").glob("route_parts/day=*/part=*.parquet")
    )
    v5_routes = (
        pd.concat([pd.read_parquet(path) for path in v5_files], ignore_index=True)
        if v5_files
        else pd.DataFrame(
            columns=[
                "order_id",
                "route_sequence",
                "edge_uid",
                "entry_position_m",
                "exit_position_m",
            ]
        )
    )
    if len(v5_routes):
        v5_routes = v5_routes.loc[
            v5_routes.order_id.astype(str).isin(selected_ids)
        ]
    edge_ids = set(routes.canonical_edge_uid.dropna().astype(str))
    if "edge_uid" in v5_routes:
        edge_ids |= set(v5_routes.edge_uid.dropna().astype(str))
    geometry = _load_geometry_lookup(config, edge_ids)
    image_paths: list[str] = []
    for row in route_queue.itertuples(index=False):
        relative = (
            f"images/case_{int(row.case_index):03d}_{row.order_id}.svg"
        )
        image_paths.append(relative)
        _plot_case(
            target / relative,
            pd.Series(row._asdict()),
            raw,
            routes,
            v5_routes,
            geometry,
        )
    route_queue["image_path"] = image_paths
    route_queue.to_csv(
        target / "index.csv", index=False, encoding="utf-8-sig"
    )
    gps_queue = final_quality.loc[
        final_quality.gps_status.eq("local_outlier")
        & (
            final_quality.outlier_time_share.gt(0.05)
            | final_quality.outlier_distance_share.gt(0.05)
        )
    ].copy()
    gps_target = output / "audit" / "gps_diagnostic_queue.csv"
    gps_target.parent.mkdir(parents=True, exist_ok=True)
    gps_queue.to_csv(gps_target, index=False, encoding="utf-8-sig")
    return {
        "route_image_count": int(len(route_queue)),
        "gps_diagnostic_count": int(len(gps_queue)),
    }


def _report_markdown(summary: dict[str, Any]) -> str:
    regression_failed = [
        row
        for row in summary["case_regression"]
        if not row["pass"]
    ]
    failed_text = (
        "\n".join(
            f"- Case {row['case_index']}: {row['check']} "
            f"(gps={row['gps_status']}, route={row['route_status']})"
            for row in regression_failed
        )
        or "- None"
    )
    return f"""# Stage 0 v6 Final Fixed-600 Report

## Result

- Overall status: **{summary['status']}**
- Reconciled orders: **{summary['hot']['output_orders']}/600**
- Processing exceptions: **{summary['hot']['processing_exception_count']}**
- Cold/hot semantic equality: **{summary['cold_hot_equal']}**
- Time conservation failures: **{summary['hot']['time_conservation_failure_count']}**
- Distance conservation failures: **{summary['hot']['distance_conservation_failure_count']}**
- Duplicate interval allocations: **{summary['hot']['duplicate_interval_allocation_count']}**
- Non-direct observed-time violations: **{summary['hot']['non_direct_observed_time_violation_count']}**
- Traversal duplicate-distance violations: **{summary['hot']['traversal_duplicate_distance_count']}**

## Four-axis quality

- GPS: `{json.dumps(summary['hot']['gps_status_counts'], ensure_ascii=False)}`
- Route: `{json.dumps(summary['hot']['route_status_counts'], ensure_ascii=False)}`
- Dynamic: `{json.dumps(summary['hot']['dynamic_status_counts'], ensure_ascii=False)}`
- Canonical: `{json.dumps(summary['hot']['canonical_status_counts'], ensure_ascii=False)}`

## Failed specified-case regressions

{failed_text}

## Review queues

- Route images: **{summary['image_queues']['route_image_count']}**; triggered only by `route_fail`, `route_partial`, or `route_uncertain`.
- GPS diagnostics: **{summary['image_queues']['gps_diagnostic_count']}**.
- Dynamic and `chain_resolved` states do not independently trigger route images.
"""


def run_final_600(config: Stage0V6Config) -> dict[str, Any]:
    """Execute cold and hot fixed-600 runs and enforce the final contract."""

    cold, matcher = run_fixed_sample(config, run_label="cold")
    hot, _ = run_fixed_sample(config, run_label="hot", matcher=matcher)
    comparisons: dict[str, bool] = {}
    for product in PRODUCTS:
        if product == "performance":
            continue
        comparisons[product] = _semantic_digest(
            _read_product(config.path("output") / "cold", product)
        ) == _semantic_digest(
            _read_product(config.path("output") / "hot", product)
        )
    final_quality = _read_product(
        config.path("output") / "hot", "final_quality"
    )
    checks, regressions_pass = _regression_checks(config, final_quality)
    image_queues = _write_image_queues(config, final_quality)
    cold_hot_equal = bool(comparisons and all(comparisons.values()))
    engineering_pass = bool(
        cold["status"] == "PASS"
        and hot["status"] == "PASS"
        and cold_hot_equal
    )
    summary = {
        "schema_version": "stage0_v6_final_600.1",
        "status": (
            "PASS" if engineering_pass and regressions_pass else "FAIL"
        ),
        "cold": cold,
        "hot": hot,
        "cold_hot_equal": cold_hot_equal,
        "product_equality": comparisons,
        "specified_case_regression_pass": regressions_pass,
        "case_regression": checks,
        "image_queues": image_queues,
    }
    _write_json(config.path("output") / "summary.json", summary)
    report = (
        config.repo_root / "stage0" / "docs" / "stage0_v6_final_600_report.md"
    )
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(".md.tmp")
    temporary.write_text(_report_markdown(summary), encoding="utf-8")
    temporary.replace(report)
    return summary
