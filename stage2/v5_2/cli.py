"""Phase-gated command line interface for Stage 2 v5.2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import RESEARCH_CONTRACT, validate_research_contract
from .micro_products import (
    aggregate_original_route_micro_conditions,
    aggregate_static_route_complexity,
    build_micro_condition_tokens,
    fit_train_cdf_thresholds,
    write_partition_products,
)
from .performance import run_benchmarks, static_complexity_audit
from .support_transfer import fit_train_support


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _fit_support(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_parquet(args.input, columns=[args.edge_column])
    artifact = fit_train_support(frame[args.edge_column], fit_dates=args.fit_dates).to_payload()
    _write_json(args.output, artifact)
    return {"status": "PASS", "edge_count": len(artifact["counts"]), "output": args.output}


def _fit_cdf(args: argparse.Namespace) -> dict[str, Any]:
    frame = pd.read_parquet(args.input)
    artifact = fit_train_cdf_thresholds(frame, quantile=args.quantile)
    artifact["fit_dates"] = list(args.fit_dates)
    _write_json(args.output, artifact)
    return {"status": "PASS", "output": args.output}


def _build_products(args: argparse.Namespace) -> dict[str, Any]:
    predictions = pd.read_parquet(args.predictions)
    context = pd.read_parquet(args.route_context)
    tokens = build_micro_condition_tokens(
        predictions,
        context,
        support_artifact=_json(args.support_artifact),
        prediction_source=args.prediction_source,
        model_id=args.model_id,
        model_hash=args.model_hash,
    )
    routes = aggregate_original_route_micro_conditions(
        tokens, _json(args.train_cdf), minimum_coverage=args.minimum_coverage
    )
    static = aggregate_static_route_complexity(tokens)
    return write_partition_products(tokens, routes, static, output_root=args.output_root)


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    frame, report = run_benchmarks(tuple(args.sizes))
    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    _write_json(args.output_json, report)
    return report


def _verify_contract(args: argparse.Namespace) -> dict[str, Any]:
    validate_research_contract(RESEARCH_CONTRACT)
    audit = static_complexity_audit(args.source_root)
    return {
        "schema_version": "stage2_v5_2_phase_a_verification.1",
        "status": "PASS" if audit["status"] == "PASS" else "FAIL",
        "research_contract": "PASS",
        "static_complexity": audit,
        "experiments_run": False,
        "stage2_status": "NOT_READY_IMPLEMENTATION_ONLY",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    support = commands.add_parser("fit-train-support")
    support.add_argument("--input", required=True)
    support.add_argument("--edge-column", default="observed_directed_edge_uid")
    support.add_argument("--fit-dates", nargs="+", required=True)
    support.add_argument("--output", required=True)
    support.set_defaults(function=_fit_support)
    cdf = commands.add_parser("fit-train-cdf")
    cdf.add_argument("--input", required=True)
    cdf.add_argument("--fit-dates", nargs="+", required=True)
    cdf.add_argument("--quantile", type=float, default=0.90)
    cdf.add_argument("--output", required=True)
    cdf.set_defaults(function=_fit_cdf)
    products = commands.add_parser("build-micro-products")
    products.add_argument("--predictions", required=True)
    products.add_argument("--route-context", required=True)
    products.add_argument("--support-artifact", required=True)
    products.add_argument("--train-cdf", required=True)
    products.add_argument("--prediction-source", required=True)
    products.add_argument("--model-id", required=True)
    products.add_argument("--model-hash", required=True)
    products.add_argument("--minimum-coverage", type=float, default=0.80)
    products.add_argument("--output-root", default="stage2/output_v5_2")
    products.set_defaults(function=_build_products)
    benchmark = commands.add_parser("benchmark")
    benchmark.add_argument("--sizes", nargs="+", type=int, default=[10_000, 50_000, 100_000, 500_000])
    benchmark.add_argument("--output-csv", default="stage2/docs/v5_2/performance_benchmarks.csv")
    benchmark.add_argument("--output-json", default="stage2/docs/v5_2/stage2_v5_2_performance_report.json")
    benchmark.set_defaults(function=_benchmark)
    verify = commands.add_parser("verify-phase-a")
    verify.add_argument("--source-root", default="stage2/v5_2")
    verify.set_defaults(function=_verify_contract)
    return root


def main() -> int:
    args = parser().parse_args()
    report = args.function(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
