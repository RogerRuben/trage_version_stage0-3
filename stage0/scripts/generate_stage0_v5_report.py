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
        f"- Fallback share: {summary.get('fallback_share')}",
        f"- Topology gaps: {summary.get('topology_gap_count')}",
        f"- Inferred-distance mean: {summary.get('mean_inferred_distance_share')}",
        f"- Processing exceptions: {summary.get('processing_exception_count')}",
        f"- HMM/output path mismatches: {summary.get('hmm_path_distance_mismatch_count')}",
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
