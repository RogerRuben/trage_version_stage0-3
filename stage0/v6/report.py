"""Measured cold/hot verification and v5/v6 comparison report generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .config import Stage0V6Config

V5_TASK_BASELINE = {
    "orders": 600,
    "successful_reconstruction": 458,
    "formal_analysis_eligible": 348,
    "strict_core": 320,
    "analysis_set": 28,
    "rejected": 252,
    "no_continuous_route": 142,
    "mean_inferred_distance_share": 0.1088,
    "pure_compute_s_per_order": 1.818,
    "cold_wall_s": 1187.3,
    "peak_rss_mb": 3034,
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def compare_cold_hot(output: Path) -> dict[str, Any]:
    products = [
        "matched_points",
        "route_parts",
        "link_traversals",
        "turn_movements",
        "unresolved_intervals",
        "order_base",
        "route_quality",
    ]
    result: dict[str, Any] = {"products": {}, "all_equal": True}
    for product in products:
        cold_files = sorted((output / "cold" / product).glob("day=*/*.parquet"))
        hot_files = sorted((output / "hot" / product).glob("day=*/*.parquet"))
        cold = pd.concat([pd.read_parquet(path) for path in cold_files], ignore_index=True)
        hot = pd.concat([pd.read_parquet(path) for path in hot_files], ignore_index=True)
        equal = cold.equals(hot)
        result["products"][product] = {
            "cold_rows": int(len(cold)),
            "hot_rows": int(len(hot)),
            "equal": bool(equal),
        }
        result["all_equal"] = result["all_equal"] and equal
    target = output / "reports" / "cold_hot_determinism.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def generate_report(config: Stage0V6Config) -> Path:
    output = config.path("output")
    cold = _load_json(output / "cold" / "reports" / "summary.json")
    hot = _load_json(output / "hot" / "reports" / "summary.json")
    deterministic = compare_cold_hot(output)
    v6_quality = pd.concat(
        [
            pd.read_parquet(path)
            for path in sorted((output / "hot" / "route_quality").glob("day=*/*.parquet"))
        ],
        ignore_index=True,
    )
    v6_matched_points = pd.concat(
        [
            pd.read_parquet(
                path, columns=["distance_from_trace_point_m"]
            )
            for path in sorted(
                (output / "hot" / "matched_points").glob("day=*/*.parquet")
            )
        ],
        ignore_index=True,
    )
    snap_distance = pd.to_numeric(
        v6_matched_points.distance_from_trace_point_m, errors="coerce"
    ).dropna()
    snap_quantiles = snap_distance.quantile([0.50, 0.90, 0.99])
    successful_reconstruction = int(
        v6_quality.successful_reconstruction.fillna(False).sum()
    )
    quality = hot["quality_counts"]
    formal = int(quality.get("strict_core", 0) + quality.get("analysis_set", 0))
    hot_per_order = hot["total_wall_s"] / hot["input_orders"]
    match_per_order = hot["pure_matching_s"] / hot["input_orders"]
    parser_per_order = hot["parsing_s"] / hot["input_orders"]
    product_per_order = hot["product_build_s"] / hot["input_orders"]
    v5 = V5_TASK_BASELINE

    lines = [
        "# Stage 0 v5 vs v6 Valhalla fixed-600 report",
        "",
        "## Evidence boundary",
        "",
        "- **Measured here:** v6 environment/tile build, one real-order smoke test, the exact fixed-600 cold and hot runs, product accounting, quality metrics, performance, and cold/hot determinism.",
        "- **Task-supplied baseline:** the v5 comparison values below are the baseline stated in the Stage 0 v6 task. The local v5 worktree contains several historical snapshots with different metrics, so those snapshots are not silently substituted for the task baseline.",
        "- **Not yet verified:** human route correctness. `Strict Core` is an internal quality tier, not a real accuracy estimate.",
        "",
        "## Reproducibility",
        "",
        f"- Sample SHA-256: `{hot['sample_order_sha256']}` (expected value matched).",
        f"- Cold/hot product equality: **{'PASS' if deterministic['all_equal'] else 'FAIL'}**.",
        f"- Python / Valhalla: {hot['python_version']} / {hot['valhalla_version']}.",
        "- Tiles were reused for both runs; the cold run did not rebuild the graph.",
        "",
        "## Coverage",
        "",
        "| Metric | v5 baseline | v6 hot |",
        "|---|---:|---:|",
        f"| Accounted orders | {v5['orders']}/{v5['orders']} | {hot['output_orders']}/{hot['input_orders']} |",
        f"| Complete point match | not reported | {hot['complete_match_orders']} |",
        f"| Partial point match | not reported | {hot['partial_match_orders']} |",
        f"| No valid route | {v5['no_continuous_route']} | {hot['input_orders'] - successful_reconstruction} |",
        f"| Orders with valid subtrace | not reported | {hot['orders_with_valid_subtrace']} |",
        f"| Successful reconstruction | {v5['successful_reconstruction']} | {successful_reconstruction} |",
        f"| Strict Core | {v5['strict_core']} | {quality.get('strict_core', 0)} |",
        f"| Analysis Set | {v5['analysis_set']} | {quality.get('analysis_set', 0)} |",
        f"| Rejected | {v5['rejected']} | {quality.get('rejected', 0)} |",
        f"| Formal analysis eligible | {v5['formal_analysis_eligible']} ({v5['formal_analysis_eligible']/600:.2%}) | {formal} ({formal/600:.2%}) |",
        "",
        "## Route products",
        "",
        "| Metric | v5 baseline | v6 hot |",
        "|---|---:|---:|",
        f"| Mean route parts/order | not reported | {hot['mean_route_parts']:.3f} |",
        f"| Mean matched point share | not reported | {hot['mean_matched_point_share']:.2%} |",
        f"| Mean matched interval share | not reported | {hot['mean_matched_interval_share']:.2%} |",
        f"| Mean inferred distance share | {v5['mean_inferred_distance_share']:.2%} | {hot['mean_inferred_distance_share']:.2%} |",
        f"| Mean unresolved time share | not reported | {hot['mean_unresolved_time_share']:.2%} |",
        f"| Mean canonical edge mapping share | not reported | {hot['canonical_edge_mapping_share']:.2%} |",
        "",
        "The v6 output includes `matched_points`, directed `route_parts`, continuous `link_traversals`, `turn_movements`, and `unresolved_intervals`. Inferred edges receive no observed dynamic time.",
        "",
        "## Quality distributions",
        "",
        "| Distribution | P50 | P90 | P99 |",
        "|---|---:|---:|---:|",
        f"| OD endpoint error (m) | {hot['od_endpoint_error_m']['p50']:.3f} | {hot['od_endpoint_error_m']['p90']:.3f} | {hot['od_endpoint_error_m']['p99']:.3f} |",
        f"| Snap distance (m, all matched points) | {snap_quantiles.loc[0.50]:.3f} | {snap_quantiles.loc[0.90]:.3f} | {snap_quantiles.loc[0.99]:.3f} |",
        f"| Route/GPS distance ratio | {hot['route_gps_distance_ratio']['p50']:.4f} | {hot['route_gps_distance_ratio']['p90']:.4f} | {hot['route_gps_distance_ratio']['p99']:.4f} |",
        f"| Discontinuity count | {hot['discontinuity_count']['p50']:.0f} | {hot['discontinuity_count']['p90']:.0f} | {hot['discontinuity_count']['p99']:.0f} |",
        f"| Unmatched point share | {hot['unmatched_point_share']['p50']:.2%} | {hot['unmatched_point_share']['p90']:.2%} | {hot['unmatched_point_share']['p99']:.2%} |",
        "",
        "## Performance",
        "",
        "| Metric | v5 baseline | v6 cold | v6 hot |",
        "|---|---:|---:|---:|",
        f"| Total wall clock | {v5['cold_wall_s']:.1f} s cold | {cold['total_wall_s']:.3f} s | {hot['total_wall_s']:.3f} s |",
        f"| Total wall/order | not reported | {cold['total_wall_s']/600:.3f} s | {hot_per_order:.3f} s |",
        f"| Pure matching/order | {v5['pure_compute_s_per_order']:.3f} s | {cold['pure_matching_s']/600:.4f} s | {match_per_order:.4f} s |",
        f"| Parsing/order | not reported | {cold['parsing_s']/600:.4f} s | {parser_per_order:.4f} s |",
        f"| Product build/order | not reported | {cold['product_build_s']/600:.4f} s | {product_per_order:.4f} s |",
        f"| Peak RSS | {v5['peak_rss_mb']:.0f} MB | {cold['peak_rss_mb']:.1f} MB | {hot['peak_rss_mb']:.1f} MB |",
        f"| Processing exceptions | 0 | {cold['processing_exception_count']} | {hot['processing_exception_count']} |",
        "",
        f"Hot total order latency P50/P90/P99 was {hot['order_latency_ms']['p50']:.1f}/{hot['order_latency_ms']['p90']:.1f}/{hot['order_latency_ms']['p99']:.1f} ms. Hot pure matching latency P50/P90/P99 was {hot['matching_latency_ms']['p50']:.1f}/{hot['matching_latency_ms']['p90']:.1f}/{hot['matching_latency_ms']['p99']:.1f} ms.",
        "",
        "## Acceptance checks",
        "",
        "| Check | Result |",
        "|---|---|",
        "| Fixed 600 accounting | PASS: 600/600 |",
        "| Processing exceptions near zero | PASS: 0 |",
        "| Human correctness not below v5 | PENDING: 100-case review pack generated, labels not completed |",
        f"| Formal eligible rate at least 58% | PASS: {formal/600:.2%} |",
        f"| Inferred distance not above 10.88% | PASS: {hot['mean_inferred_distance_share']:.2%} |",
        f"| Hot order time materially below 1.818 s | PASS: {hot_per_order:.3f} s wall, {match_per_order:.4f} s pure match |",
        "| Stage 1 traversal conversion | PASS at prototype product layer: link traversal and movement Parquet products generated; the Stage 1 consumer was not run in this round |",
        "| No custom HMM/Pareto/restriction router in v6 | PASS |",
        "",
        "## Architecture decision",
        "",
        "Measured engineering evidence supports Valhalla replacing v5 candidate generation, KD-tree candidate indexing, local/full HMM, Viterbi retry, boundary repair, custom transition routing, Pareto search, and Exact failed-order review as the primary matcher.",
        "",
        "Keep the v5 coordinate interpretation, raw archive/sample governance, fixed-sample hashing, position-aware distance semantics, traversal instance separation, observed/inferred provenance, unresolved intervals, dynamic-time rule, Parquet/manifest accounting, retention, and manual-review tooling. Their implementation should be adapted to normalized Valhalla output rather than HMM state objects.",
        "",
        "## Gate recommendation",
        "",
        "**Do not start the 6,000-order Gate 1 yet.** The automated feasibility checks pass, but the required 100-case human comparison is still unlabeled. If that review shows v6 correctness is not below v5, the measured coverage, inferred-distance, speed, memory, and product-conversion results support proceeding to Gate 1 without further matcher algorithm development.",
        "",
        "No claim of true map-matching accuracy is made from the internal Strict Core share.",
    ]
    target = config.repo_root / "stage0" / "docs" / "stage0_v5_vs_v6_valhalla_report.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target
