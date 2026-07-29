"""Command-line entry points for the Stage 0 v6 feasibility prototype."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .canonical_mapper import CanonicalEdgeMapper
from .config import load_config
from .eligibility import evaluate_modeling_eligibility
from .parser import parse_trace_attributes
from .pipeline import benchmark_cold_and_hot, load_fixed_sample, run_fixed_sample
from .preprocess import preprocess_order
from .products import build_order_products
from .quality import evaluate_order_quality
from .valhalla_client import ValhallaMatcher


def _single(config_path: Path) -> dict:
    config = load_config(config_path)
    sample = load_fixed_sample(config)
    (_, order_id), raw_order = next(iter(sample.points.groupby(["date", "order_id"], sort=True)))
    prep = preprocess_order(raw_order, **config.section("preprocess"))
    eligibility = evaluate_modeling_eligibility(
        prep.points,
        prep.metrics,
        **config.section("modeling_eligibility"),
    )
    matcher = ValhallaMatcher(
        config.section("valhalla"),
        valhalla_config_path=config.path("valhalla_config"),
    )
    matched_frames, route_frames, results = [], [], []
    for subtrace_id, subtrace in prep.points.groupby("subtrace_id", sort=False):
        result = (
            matcher.match_order(subtrace)
            if bool(
                eligibility["modeling_eligible"]
                and subtrace.usable_subtrace.iloc[0]
            )
            else {"raw_response": {}, "status": "too_short"}
        )
        matched, routes = parse_trace_attributes(
            result.get("raw_response") or {},
            subtrace.reset_index(drop=True),
            order_id=str(order_id),
            subtrace_id=str(subtrace_id),
        )
        matched_frames.append(matched)
        route_frames.append(routes)
        results.append(
            {
                key: value
                for key, value in result.items()
                if key not in {"raw_response", "request"}
            }
        )
    import pandas as pd

    matched = pd.concat(matched_frames, ignore_index=True)
    routes = pd.concat(route_frames, ignore_index=True)
    mapper = CanonicalEdgeMapper.from_parquet(config.path("canonical_edges"))
    mapped, mapping = mapper.map_route_parts(routes)
    products = build_order_products(
        prep.points,
        matched,
        mapped,
        preprocess_breaks=prep.preprocess_breaks,
        **config.section("products"),
    )
    quality = evaluate_order_quality(
        prep.points,
        matched,
        mapped,
        products["unresolved_intervals"],
        config.section("quality"),
        interval_measurements=products["interval_measurements"],
        link_traversals=products["link_traversals"],
        interval_accounting=products["interval_accounting"],
    )
    return {
        "order_id": str(order_id),
        "preprocess": prep.metrics,
        "modeling_eligibility": eligibility,
        "match": results,
        "matched_points": len(matched),
        "route_parts": len(mapped),
        "mapping": mapping.__dict__,
        "link_traversals": len(products["link_traversals"]),
        "quality": quality,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("stage0/config/stage0_v6_valhalla.yaml"),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("single")
    subparsers.add_parser("benchmark")
    subparsers.add_parser("run-hot")
    subparsers.add_parser("report")
    subparsers.add_parser("manual-audit")
    subparsers.add_parser("audit")
    subparsers.add_parser("audit-pack")
    subparsers.add_parser("audit-all")
    subparsers.add_parser("final-600")
    build_parser = subparsers.add_parser("build-stage1-input")
    build_parser.add_argument("--resume", action="store_true")
    verify_parser = subparsers.add_parser("verify-stage1-input")
    verify_parser.add_argument("--input", type=Path)
    args = parser.parse_args(argv)
    if args.command == "single":
        payload = _single(args.config)
    elif args.command == "benchmark":
        payload = benchmark_cold_and_hot(load_config(args.config))
    elif args.command == "run-hot":
        payload, _ = run_fixed_sample(
            load_config(args.config), run_label="hot"
        )
    elif args.command == "report":
        from .report import generate_report

        payload = {"report": str(generate_report(load_config(args.config)))}
    elif args.command == "manual-audit":
        from .manual_audit import generate_manual_audit

        payload = {"manual_audit": str(generate_manual_audit(load_config(args.config)))}
    elif args.command == "audit":
        from .automated_audit import run_automated_audit

        result = run_automated_audit(load_config(args.config))
        payload = {
            "audit": str(result.audit_path),
            "row_count": result.row_count,
            "class_counts": result.class_counts,
            "runtime_s": result.runtime_s,
            "peak_rss_mb": result.peak_rss_mb,
            "valhalla_invoked": False,
        }
    elif args.command == "audit-pack":
        from .automated_audit import generate_manual_review_pack

        result = generate_manual_review_pack(load_config(args.config))
        payload = {
            "audit_pack": str(result.output_dir),
            "index": str(result.index_path),
            "image_count": result.image_count,
            "runtime_s": result.runtime_s,
            "peak_rss_mb": result.peak_rss_mb,
            "valhalla_invoked": False,
        }
    elif args.command == "audit-all":
        from .automated_audit import (
            generate_manual_review_pack,
            run_automated_audit,
        )

        audit_result = run_automated_audit(load_config(args.config))
        pack_result = generate_manual_review_pack(load_config(args.config))
        payload = {
            "audit": str(audit_result.audit_path),
            "row_count": audit_result.row_count,
            "class_counts": audit_result.class_counts,
            "audit_runtime_s": audit_result.runtime_s,
            "audit_pack": str(pack_result.output_dir),
            "image_count": pack_result.image_count,
            "pack_runtime_s": pack_result.runtime_s,
            "valhalla_invoked": False,
        }
    elif args.command == "final-600":
        from .final_report import run_final_600

        payload = run_final_600(load_config(args.config))
    elif args.command == "build-stage1-input":
        from .stage1_production import build_stage1_input

        payload = build_stage1_input(
            load_config(args.config), resume=bool(args.resume)
        )
    else:
        from .stage1_production import verify_stage1_input

        payload = verify_stage1_input(
            load_config(args.config), args.input
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
