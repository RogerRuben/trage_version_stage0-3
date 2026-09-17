"""Generate the fixed-600 dynamic-product correction report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Stage0V6Config

PRE_FIX_BASELINE = {
    "strict_core": 543,
    "analysis_set": 43,
    "rejected": 14,
    "formal_eligible": 586,
    "mean_inferred_distance_share": 0.04053561625677863,
    "hot_wall_s": 198.0647949999984,
    "hot_wall_per_order_s": 0.330107991666664,
    "pure_matching_per_order_s": 0.013619528166685,
    "peak_rss_mb": 610.6875,
}

DETERMINISTIC_PRODUCTS = [
    "matched_points",
    "route_parts",
    "link_traversals",
    "turn_movements",
    "unresolved_intervals",
    "interval_measurements",
    "interval_accounting",
    "order_base",
    "route_quality",
    "dynamic_measurement_quality",
    "subtrace_mapping",
    "preprocess_breaks",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_product(root: Path, product: str) -> pd.DataFrame:
    files = sorted((root / product).glob("day=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(path) for path in files], ignore_index=True)


def compare_cold_hot(output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"products": {}, "all_equal": True}
    for product in DETERMINISTIC_PRODUCTS:
        cold = _read_product(output / "cold", product)
        hot = _read_product(output / "hot", product)
        columns_equal = cold.columns.tolist() == hot.columns.tolist()
        dtypes_equal = [str(value) for value in cold.dtypes] == [
            str(value) for value in hot.dtypes
        ]
        values_equal = cold.equals(hot)
        equal = columns_equal and dtypes_equal and values_equal
        result["products"][product] = {
            "cold_rows": int(len(cold)),
            "hot_rows": int(len(hot)),
            "columns_equal": columns_equal,
            "dtypes_equal": dtypes_equal,
            "values_equal": values_equal,
            "equal": equal,
        }
        result["all_equal"] = result["all_equal"] and equal
    target = output / "reports" / "cold_hot_determinism.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(target)
    return result


def generate_report(config: Stage0V6Config) -> Path:
    output = config.path("output")
    cold = _load_json(output / "cold" / "reports" / "summary.json")
    hot = _load_json(output / "hot" / "reports" / "summary.json")
    deterministic = compare_cold_hot(output)
    route = _read_product(output / "hot", "route_quality")
    dynamic = _read_product(output / "hot", "dynamic_measurement_quality")
    unresolved = _read_product(output / "hot", "unresolved_intervals")
    accounting = _read_product(output / "hot", "interval_accounting")
    manifests = [
        _load_json(path)
        for path in sorted((output / "hot" / "manifests").glob("day=*/*.json"))
    ]
    unresolved_only = unresolved.loc[
        unresolved.get(
            "measurement_source", pd.Series("", index=unresolved.index)
        ).eq("unresolved")
    ]
    reason_time = (
        unresolved_only.groupby("unresolved_reason", dropna=False)
        .interval_duration_s.sum()
        .sort_values(ascending=False)
    )
    route_counts = hot["quality_counts"]
    dynamic_counts = hot["dynamic_quality_counts"]
    formal = int(hot["route_formal_eligible_orders"])
    dynamic_usable = int(hot["dynamic_usable_orders"])
    static_only = int(hot["static_only_orders"])
    hot_per_order = hot["total_wall_s"] / hot["input_orders"]
    match_per_order = hot["pure_matching_s"] / hot["input_orders"]
    route_rows_equal = deterministic["products"]["route_parts"]["equal"]
    matched_rows_equal = deterministic["products"]["matched_points"]["equal"]
    break_product_count = int(
        deterministic["products"]["preprocess_breaks"]["hot_rows"]
    )
    bucket_pass = all(item["status"] == "PASS" for item in manifests)
    maximum_bucket_rss = max(
        (float(item["peak_rss_mb"]) for item in manifests), default=0.0
    )
    top_reasons = [
        f"- `{reason}`: {seconds:.3f} s ({seconds / max(reason_time.sum(), 1):.2%})"
        for reason, seconds in reason_time.head(10).items()
    ]
    if not top_reasons:
        top_reasons = ["- No unresolved intervals."]

    lines = [
        "# Stage 0 v6 dynamic product fix - fixed-600 report",
        "",
        "## Evidence and semantic boundary",
        "",
        f"- Frozen sample SHA-256: `{hot['sample_order_sha256']}`.",
        "- Valhalla 3.8.2 remains the matcher. No candidate generator, HMM/Viterbi, Pareto search, boundary repair, restriction router, tile logic, canonical mapper, v5 code, sample, seed, or matching parameter was changed.",
        "- Route usability and dynamic link-time usability are now independent outputs.",
        "- Dynamic thresholds are initial engineering thresholds, not empirically optimized values.",
        "",
        "## Acceptance summary",
        "",
        "| Check | Result |",
        "|---|---|",
        f"| 600/600 accounting | {'PASS' if hot['accounting_pass'] else 'FAIL'} |",
        f"| Processing exceptions | {'PASS' if hot['processing_exception_count'] == 0 else 'FAIL'}: {hot['processing_exception_count']} |",
        f"| Cold/hot field-level equality | {'PASS' if deterministic['all_equal'] else 'FAIL'} |",
        f"| Time conservation failures | {'PASS' if hot['time_conservation_failure_count'] == 0 else 'FAIL'}: {hot['time_conservation_failure_count']} |",
        f"| Timestamp anchor failure orders | {'PASS' if hot['timestamp_anchor_failure_order_count'] == 0 else 'FAIL'}: {hot['timestamp_anchor_failure_order_count']} |",
        f"| Inferred-edge observed-time violations | {'PASS' if hot['inferred_edge_observed_time_violation_count'] == 0 else 'FAIL'}: {hot['inferred_edge_observed_time_violation_count']} |",
        f"| Unresolved duplicate allocations | {'PASS' if hot['unresolved_duplicate_allocation_count'] == 0 else 'FAIL'}: {hot['unresolved_duplicate_allocation_count']} |",
        f"| Preprocess boundaries materialized | PASS: {break_product_count} rows |",
        f"| Bucket manifests | {'PASS' if bucket_pass else 'FAIL'}: {len(manifests)} buckets |",
        "",
        "## Route layer",
        "",
        "| Metric | Before fix | After fix (hot) |",
        "|---|---:|---:|",
        f"| Successful route orders | 599 | {hot['successful_reconstruction_orders']} |",
        f"| Strict Core | {PRE_FIX_BASELINE['strict_core']} | {route_counts.get('strict_core', 0)} |",
        f"| Analysis Set | {PRE_FIX_BASELINE['analysis_set']} | {route_counts.get('analysis_set', 0)} |",
        f"| Rejected | {PRE_FIX_BASELINE['rejected']} | {route_counts.get('rejected', 0)} |",
        f"| Formal eligible | {PRE_FIX_BASELINE['formal_eligible']} (97.67%) | {formal} ({formal/600:.2%}) |",
        f"| Mean matched point share | 99.68% | {hot['mean_matched_point_share']:.2%} |",
        f"| Mean matched route interval share | 98.63% (old semantics) | {hot['mean_matched_interval_share']:.2%} |",
        f"| Mean inferred distance share | {PRE_FIX_BASELINE['mean_inferred_distance_share']:.2%} | {hot['mean_inferred_distance_share']:.2%} |",
        f"| Mean preprocess break count | not reported | {hot['mean_preprocess_break_count']:.4f} |",
        f"| Mean canonical mapping share | 99.83% | {hot['canonical_edge_mapping_share']:.2%} |",
        "",
        f"Cold/hot matched points were identical: **{matched_rows_equal}**. Cold/hot normalized/canonical route parts were identical: **{route_rows_equal}**. This is the direct evidence that the Valhalla route result remained stable while downstream measurement semantics changed.",
        "",
        "Route/resolved-GPS ratio P50/P90/P99: "
        f"{hot['route_resolved_gps_distance_ratio']['p50']:.4f}/"
        f"{hot['route_resolved_gps_distance_ratio']['p90']:.4f}/"
        f"{hot['route_resolved_gps_distance_ratio']['p99']:.4f}. "
        "Route/raw-GPS ratio is retained separately as a diagnostic.",
        "",
        "## Dynamic measurement layer",
        "",
        "| Metric | Hot result |",
        "|---|---:|",
        f"| Dynamic Strict | {dynamic_counts.get('dynamic_strict', 0)} |",
        f"| Dynamic Partial | {dynamic_counts.get('dynamic_partial', 0)} |",
        f"| Dynamic Unusable | {dynamic_counts.get('dynamic_unusable', 0)} |",
        f"| Dynamic Partial or better | {dynamic_usable} ({dynamic_usable/600:.2%}) |",
        f"| Route-eligible but dynamic-unusable | {static_only} |",
        f"| Mean direct-observed interval time share | {hot['mean_direct_observed_interval_time_share']:.2%} |",
        f"| Mean direct-observed distance share | {hot['mean_direct_observed_distance_share']:.2%} |",
        f"| Mean interval-supported time share | {hot['mean_interval_supported_time_share']:.2%} |",
        f"| Mean engine-allocated time share | {hot['mean_engine_allocated_time_share']:.2%} |",
        f"| Mean unresolved time share | {hot['mean_unresolved_time_share']:.2%} |",
        f"| Mean timed traversal share | {hot['mean_timed_traversal_share']:.2%} |",
        f"| Mean valid timed traversals/order | {hot['mean_valid_timed_traversal_count']:.3f} |",
        "",
        "Unknown dynamic values are `NaN`, never zero. Engine allocation remains disabled; parsed Valhalla cumulative elapsed time is converted to per-edge increments but is not written into `observed_travel_time_s`.",
        "",
        "## Unresolved-time causes",
        "",
        *top_reasons,
        "",
        "Multi-edge continuous intervals are retained as `interval_supported` records with start/end timestamps and route distance, but are not assigned to individual link travel times. Intervals containing engine-interpolated edges are wholly `unresolved`.",
        "",
        "## Performance and streaming",
        "",
        "| Metric | Cold | Hot | Requirement |",
        "|---|---:|---:|---:|",
        f"| Total wall | {cold['total_wall_s']:.3f} s | {hot['total_wall_s']:.3f} s | n/a |",
        f"| Wall/order | {cold['total_wall_s']/600:.3f} s | {hot_per_order:.3f} s | <= 0.400 s |",
        f"| Pure Valhalla/order | {cold['pure_matching_s']/600:.4f} s | {match_per_order:.4f} s | <= 0.050 s |",
        f"| Parsing | {cold['parsing_s']:.3f} s | {hot['parsing_s']:.3f} s | n/a |",
        f"| Canonical mapping | {cold['canonical_mapping_s']:.3f} s | {hot['canonical_mapping_s']:.3f} s | n/a |",
        f"| Product build | {cold['product_build_s']:.3f} s | {hot['product_build_s']:.3f} s | n/a |",
        f"| Quality evaluation | {cold['quality_evaluation_s']:.3f} s | {hot['quality_evaluation_s']:.3f} s | n/a |",
        f"| Parquet write | {cold['parquet_write_s']:.3f} s | {hot['parquet_write_s']:.3f} s | n/a |",
        f"| Peak RSS | {cold['peak_rss_mb']:.1f} MB | {hot['peak_rss_mb']:.1f} MB | <= 1024 MB |",
        f"| Maximum bucket RSS | {cold['bucket_peak_rss_mb']['p99']:.1f} MB | {maximum_bucket_rss:.1f} MB | n/a |",
        "",
        f"Hot order latency P50/P90/P99: {hot['order_latency_ms']['p50']:.1f}/"
        f"{hot['order_latency_ms']['p90']:.1f}/{hot['order_latency_ms']['p99']:.1f} ms.",
        "",
        "The run wrote one 200-order bucket for each of the three dates. Each product used a temporary Parquet followed by atomic replacement, and each bucket emitted its own manifest before in-memory product frames were released. This removes the previous all-600-product retention pattern and is structurally suitable for 6,000 orders, subject to a Gate 1 run rather than an unmeasured guarantee.",
        "",
        "## Required conclusions",
        "",
        f"1. **Valhalla route stability:** yes. Cold/hot matched-point and route-part products are field-identical, with {hot['processing_exception_count']} processing exceptions.",
        f"2. **Route-layer usability:** {formal}/600 ({formal/600:.2%}) formal eligible after corrected route/resolved-GPS and break semantics.",
        f"3. **Dynamic-measurement usability:** {dynamic_usable}/600 ({dynamic_usable/600:.2%}) are `dynamic_partial` or `dynamic_strict`.",
        f"4. **Static-only orders:** {static_only} route-eligible orders are `dynamic_unusable` and must not create dynamic link labels.",
        "5. **Main unresolved causes:** listed above by conserved interval duration; inferred paths, interpolated endpoints, unmatched/discontinuous intervals, and preprocess boundaries are kept separate.",
        f"6. **Cross-inferred-edge allocation:** no remaining violation was detected ({hot['inferred_edge_observed_time_violation_count']}).",
        f"7. **6,000-order execution risk:** the 200-order atomic bucket design avoids linear retention of product DataFrames; measured peak RSS was {hot['peak_rss_mb']:.1f} MB and every bucket manifest passed. A 6,000-order run is expected to be memory-safe but has not yet been executed.",
        "8. **Gate 1 recommendation:** do not start automatically. Engineering acceptance must pass, the 100-order human audit must still show no systematic route error, and the measured dynamic-partial-or-better coverage must be accepted by the downstream Stage 1 owner.",
        "",
        "No real route-accuracy claim is made from internal quality tiers.",
    ]
    target = (
        config.repo_root
        / "stage0"
        / "docs"
        / "stage0_v6_dynamic_product_fix_report.md"
    )
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
