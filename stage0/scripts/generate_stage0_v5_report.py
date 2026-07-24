#!/usr/bin/env python
"""Generate a measured Stage 0 v5 correctness/performance report from products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from stage0.v5.config import Stage0Config, config_hash  # noqa: E402
from stage0.v5.archive import sampling_run_id  # noqa: E402
from stage0.v5.manifest import write_manifest  # noqa: E402
from stage0.v5.pipeline import summarize_run  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPO / "stage0/config/stage0_v5.yaml")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--dates", nargs="+", required=True)
    parser.add_argument("--orders-per-day", type=int, required=True)
    parser.add_argument("--label", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = Stage0Config.load(args.config)
    if args.output_root is not None:
        values = {
            **config.values,
            "paths": {**config.values["paths"], "output": str(args.output_root.resolve())},
        }
        config = Stage0Config(config.source, values, config_hash(values))
    summary = summarize_run(
        config, REPO, [str(value) for value in args.dates], args.orders_per_day
    )
    output = config.path("output", REPO) / "reports"
    output.mkdir(parents=True, exist_ok=True)
    run_id = sampling_run_id(
        [str(value) for value in args.dates],
        args.orders_per_day,
        int(config.section("sampling")["seed"]),
    )
    run_summaries = sorted(
        output.glob(f"stage0_v5_run_summary__{run_id}__*.json"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    if run_summaries:
        measured_run = json.loads(
            run_summaries[-1].read_text(encoding="utf-8")
        )
        if measured_run.get("sample_order_sha256") != summary.get(
            "sample_order_sha256"
        ):
            raise RuntimeError(
                "latest run summary sample SHA does not match products"
            )
        summary.update({
            key: measured_run.get(key)
            for key in (
                "runtime_sec",
                "peak_memory_mb",
                "initialization_ms",
                "profiled_ms",
                "unprofiled_ms",
                "candidate_index_cache_hit",
                "routing_cache_pairs",
            )
        })
    json_path = output / f"{args.label}_measured_report.json"
    write_manifest(json_path, summary)
    lines = [
        f"# Stage 0 v5 measured report: {args.label}",
        "",
        f"- Orders: {summary.get('output_orders')}/{summary.get('input_orders')}",
        f"- Sample SHA-256: `{summary.get('sample_order_sha256')}`",
        f"- Modes: `{json.dumps(summary.get('matching_mode_share', {}), sort_keys=True)}`",
        f"- Quality: `{json.dumps(summary.get('quality_counts', {}), sort_keys=True)}`",
        f"- Full-HMM attempt/failure share: {summary.get('full_hmm_attempt_share')} / {summary.get('full_hmm_failure_share')}",
        f"- Local-HMM attempt share: {summary.get('local_hmm_attempt_share')}",
        f"- Local-HMM order/window/retry attempts: "
        f"{summary.get('local_hmm_attempt_count')} / "
        f"{summary.get('local_hmm_window_attempt_count')} / "
        f"{summary.get('local_hmm_retry_window_count')}",
        f"- Boundary-repair Viterbi count: {summary.get('boundary_repair_viterbi_count')}",
        f"- Fallback share: {summary.get('fallback_share')}",
        f"- Topology gaps: {summary.get('topology_gap_count')}",
        f"- Inferred-distance by scope: `{json.dumps(summary.get('inferred_distance_share_by_scope', {}), sort_keys=True)}`",
        f"- Position audit applicable / N-A / actual failures: "
        f"{summary.get('position_audit_applicable_order_count')} / "
        f"{summary.get('position_audit_not_applicable_match_failure_count')} / "
        f"{summary.get('actual_invalid_position_event_count')}",
        f"- Failed-transition reasons: `{json.dumps(summary.get('failed_transition_reason_counts', {}), sort_keys=True)}`",
        f"- Failed-transition direct raw-movement status (absence means no direct record, not a network gap): "
        f"`{json.dumps(summary.get('failed_transition_raw_movement_status_counts', {}), sort_keys=True)}`",
        f"- Failed-transition diagnostic classes: `{json.dumps(summary.get('failed_transition_diagnostic_class_counts', {}), sort_keys=True)}`",
        f"- Pre-validation to final mode: `{json.dumps(summary.get('pre_validation_to_final_mode_cross_tab', {}), sort_keys=True)}`",
        f"- Path searches/order: `{json.dumps(summary.get('path_searches_per_order_distribution', {}), sort_keys=True)}`",
        f"- Expanded states/order: `{json.dumps(summary.get('expanded_states_per_order_distribution', {}), sort_keys=True)}`",
        f"- Exact/approximate path calls and approximate unresolved: "
        f"{summary.get('exact_path_search_calls')} / "
        f"{summary.get('approximate_path_search_calls')} / "
        f"{summary.get('approximate_search_unresolved_count')}",
        f"- Order-local transition evidence cache hits/misses: "
        f"{summary.get('order_transition_evidence_cache_hits')} / "
        f"{summary.get('order_transition_evidence_cache_misses')}",
        f"- Processing exceptions: {summary.get('processing_exception_count')}",
        f"- HMM/output path mismatches: {summary.get('hmm_path_distance_mismatch_count')}",
        f"- Same-edge jitter audit mismatches: {summary.get('same_edge_jitter_mismatch_count')}",
        f"- Total runtime / peak RSS: {summary.get('runtime_sec')} s / "
        f"{summary.get('peak_memory_mb')} MB",
        f"- Initialization/profiled/unprofiled: "
        f"{summary.get('initialization_ms')} / {summary.get('profiled_ms')} / "
        f"{summary.get('unprofiled_ms')} ms",
        "",
        "## Performance percentiles (ms/order)",
        "",
    ]
    for name, values in summary.get("performance_percentiles_ms", {}).items():
        lines.append(f"- `{name}`: `{json.dumps(values, sort_keys=True)}`")
    md_path = output / f"{args.label}_measured_report.md"
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"json": str(json_path), "markdown": str(md_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
