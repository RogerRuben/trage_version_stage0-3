"""Protocol-bound, phase-gated command line interface for Stage 2 v5.2."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import RESEARCH_CONTRACT, Stage2V52ContractError, validate_research_contract
from .evaluation import evaluate_pace_guard, evaluate_spatial_adoption, evaluate_temporal_adoption
from .micro_products import (
    aggregate_original_route_micro_conditions,
    aggregate_static_route_complexity,
    build_micro_condition_tokens,
    fit_train_cdf_thresholds,
    write_partition_products,
)
from .performance import run_benchmarks, static_complexity_audit
from .protocols import PROTOCOLS, get_protocol
from .static_schema_audit import audit_static_schema, schema_names
from .structure_features import fit_static_structure_artifact
from .support_transfer import fit_train_support_frame, select_tau_once
from .training import train_micro_tree_baseline_from_npz, train_transfer_from_shards
from .verification import build_release_manifest, preflight, sha256_file, verify_phase_b


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    return _json(args.config)


def _require_execution(config: dict[str, Any], allowed: set[str]) -> None:
    authorization = str(config.get("execution_authorization", "NONE"))
    if authorization not in allowed:
        raise Stage2V52ContractError(
            f"execution authorization is {authorization}; allowed values are {sorted(allowed)}"
        )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    return preflight(
        config_path=args.config,
        protocol_id=args.protocol,
        source_checkpoint_path=args.source_checkpoint,
        feature_artifact_path=args.feature_artifact,
    )


def _fit_support(args: argparse.Namespace) -> dict[str, Any]:
    protocol = get_protocol(args.protocol)
    columns = ["split", "date", "order_id", "traversal_id", "observed_directed_edge_uid"]
    frame = pd.read_parquet(args.input, columns=columns)
    artifact = fit_train_support_frame(
        frame,
        protocol_id=args.protocol,
        protocol_train_dates=protocol.train_dates,
        input_sha256=sha256_file(args.input),
    ).to_payload()
    _write_json(args.output, artifact)
    return {"status": "PASS", "edge_count": len(artifact["counts"]), "output": args.output}


def _fit_static(args: argparse.Namespace) -> dict[str, Any]:
    protocol = get_protocol(args.protocol)
    frame = pd.read_parquet(args.input)
    if not frame["split"].astype(str).eq("train").all():
        raise Stage2V52ContractError("static artifact input contains non-Train rows")
    observed = tuple(sorted(frame["date"].astype(str).unique()))
    if observed != tuple(sorted(protocol.train_dates)):
        raise Stage2V52ContractError("static artifact dates differ from frozen protocol Train dates")
    artifact = fit_static_structure_artifact([frame], fit_dates=protocol.train_dates).to_payload()
    artifact["input_sha256"] = sha256_file(args.input)
    _write_json(args.output, artifact)
    return {"status": "PASS", "output": args.output}


def _fit_cdf(args: argparse.Namespace) -> dict[str, Any]:
    protocol = get_protocol(args.protocol)
    frame = pd.read_parquet(args.input)
    artifact = fit_train_cdf_thresholds(
        frame,
        protocol_id=args.protocol,
        protocol_train_dates=protocol.train_dates,
        input_sha256=sha256_file(args.input),
        quantile=args.quantile,
    )
    _write_json(args.output, artifact)
    return {"status": "PASS", "output": args.output}


def _tune_tau(args: argparse.Namespace) -> dict[str, Any]:
    if args.protocol != "transfer_tuning":
        raise Stage2V52ContractError("tau can only be selected under transfer_tuning")
    metrics = _json(args.metrics)
    candidates = {
        float(tau): {str(target): float(value) for target, value in target_mae.items()}
        for tau, target_mae in metrics["candidate_mae_by_target"].items()
    }
    artifact = select_tau_once(candidates, metrics["v5_1_mae_by_target"], _json(args.support_artifact))
    artifact["metrics_input_sha256"] = sha256_file(args.metrics)
    _write_json(args.output, artifact)
    return {"status": "PASS", "selected_tau": artifact["selected_tau"], "output": args.output}


def _train_model(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    _require_execution(config, {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D"})
    training = config["training"]
    baseline = _json(args.v5_1_metrics)
    if args.model in {"M4", "M5"} and not args.tau_artifact:
        raise Stage2V52ContractError(f"{args.model} requires the frozen transfer-tuning tau artifact")
    tau = (
        float(_json(args.tau_artifact)["selected_tau"])
        if args.model in {"M4", "M5"}
        else float(training["default_nonselected_tau"])
    )
    if args.model == "M5":
        if not args.m4_adoption:
            raise Stage2V52ContractError("M5 requires the frozen M4 adoption decision")
        adoption = _json(args.m4_adoption)
        if adoption.get("adopt") is not True:
            raise Stage2V52ContractError("M5 is blocked until M4 passes its adoption gate")
    return train_transfer_from_shards(
        protocol_id=args.protocol,
        model_id=args.model,
        tensor_root=args.tensor_root,
        feature_artifact_path=args.feature_artifact,
        source_checkpoint_path=args.source_checkpoint,
        source_model_id=args.source_model_id,
        static_feature_count=int(args.static_feature_count),
        support_tau=tau,
        backbone_kwargs=config["backbone"],
        v5_1_core_mae=baseline["core_mae"],
        v5_1_pace_p50_mae=float(baseline["pace_p50_mae"]),
        output_root=args.output,
        new_branch_lr=float(training["new_branch_lr"]),
        backbone_lr_ratio=float(training["backbone_lr_ratio"]),
        shared_freeze_epochs=int(training["shared_freeze_epochs"]),
        maximum_epochs=int(training["maximum_epochs"]),
        batch_size=int(training["batch_size"]),
        temporal_leakage_count=int(args.temporal_leakage_count),
    )


def _train_tree_baseline(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    _require_execution(config, {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D"})
    return train_micro_tree_baseline_from_npz(
        protocol_id=args.protocol,
        input_path=args.input,
        feature_schema_hash=args.feature_schema_hash,
        output_root=args.output,
    )


def _evaluate_model(args: argparse.Namespace) -> dict[str, Any]:
    payload = _json(args.input)
    if args.kind == "spatial":
        result = evaluate_spatial_adoption(payload)
    elif args.kind == "temporal":
        result = evaluate_temporal_adoption(payload)
    else:
        result = evaluate_pace_guard(payload)
    _write_json(args.output, result)
    return {"status": "PASS", **result}


def _build_products(args: argparse.Namespace) -> dict[str, Any]:
    predictions = pd.read_parquet(args.predictions)
    context = pd.read_parquet(args.route_context)
    tokens = build_micro_condition_tokens(
        predictions,
        context,
        support_artifact=_json(args.support_artifact),
        protocol_id=args.protocol,
        prediction_source=args.prediction_source,
        model_id=args.model_id,
        model_hash=args.model_hash,
    )
    routes = aggregate_original_route_micro_conditions(
        tokens, _json(args.train_cdf), minimum_coverage=args.minimum_coverage
    )
    static = aggregate_static_route_complexity(tokens)
    return write_partition_products(tokens, routes, static, output_root=args.output_root)


def _audit_static(args: argparse.Namespace) -> dict[str, Any]:
    report = audit_static_schema(
        route_columns=schema_names(args.route_products),
        movement_columns=schema_names(args.movement_products),
    )
    _write_json(args.output, report)
    return report


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    frame, report = run_benchmarks(tuple(args.sizes))
    destination = Path(args.output_csv)
    destination.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(destination, index=False)
    _write_json(args.output_json, report)
    return report


def _verify_phase_a(args: argparse.Namespace) -> dict[str, Any]:
    validate_research_contract(RESEARCH_CONTRACT)
    audit = static_complexity_audit(args.source_root)
    return {
        "schema_version": "stage2_v5_2_phase_a_1_verification.1",
        "status": "PASS" if audit["status"] == "PASS" else "FAIL",
        "research_contract": "PASS",
        "static_complexity": audit,
        "experiments_run": False,
        "stage2_status": "NOT_READY_IMPLEMENTATION_ONLY",
    }


def _verify_phase_b(args: argparse.Namespace) -> dict[str, Any]:
    report = verify_phase_b(pd.read_parquet(args.tokens))
    _write_json(args.output, report)
    return report


def _run_plan(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    required = "PHASE_C" if args.command == "run-development-day" else "PHASE_D"
    _require_execution(config, {required})
    protocol_ids = ("development",) if required == "PHASE_C" else ("fold_1", "fold_2", "fold_3")
    specification = _json(args.run_spec)
    if specification.get("schema_version") != "stage2_v5_2_protocol_run.1":
        raise Stage2V52ContractError("run spec has an unknown schema version")
    declared_protocols = tuple(str(value) for value in specification.get("protocols", ()))
    if declared_protocols != protocol_ids:
        raise Stage2V52ContractError(
            f"run spec protocols {declared_protocols} differ from frozen {protocol_ids}"
        )
    handlers = {
        "preflight": _preflight,
        "fit-support": _fit_support,
        "fit-static-artifact": _fit_static,
        "fit-train-cdf": _fit_cdf,
        "tune-tau": _tune_tau,
        "train-tree-baseline": _train_tree_baseline,
        "train-model": _train_model,
        "evaluate-model": _evaluate_model,
        "build-products": _build_products,
        "benchmark": _benchmark,
        "verify-phase-b": _verify_phase_b,
    }
    forbidden_date_arguments = {
        "date", "dates", "train_date", "train_dates", "validation_date",
        "validation_dates", "calibration_dates", "evaluation_dates",
    }
    steps = tuple(specification.get("steps", ()))
    if not steps:
        raise Stage2V52ContractError("run spec must contain at least one executable step")
    reports: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        action = str(step.get("command", ""))
        if action not in handlers:
            raise Stage2V52ContractError(f"run spec step {index} has forbidden command {action}")
        arguments = dict(step.get("arguments", {}))
        if "config" in arguments or forbidden_date_arguments & set(arguments):
            raise Stage2V52ContractError("run spec cannot override frozen protocol dates")
        if "protocol" in arguments and str(arguments["protocol"]) not in protocol_ids:
            raise Stage2V52ContractError("run spec step uses a protocol outside this frozen run")
        namespace = argparse.Namespace(config=args.config, command=action, **arguments)
        report = handlers[action](namespace)
        reports.append({"step": index, "command": action, "report": report})
        if report.get("status") not in {"PASS", "MICRO_FIRST_CHECKPOINT_SELECTED"}:
            return {
                "status": "STOPPED_ON_FAILED_GATE",
                "protocols": list(protocol_ids),
                "completed_steps": reports,
                "dates_from_cli": False,
            }
    return {
        "status": "PASS",
        "protocols": [get_protocol(value).to_payload() for value in protocol_ids],
        "completed_steps": reports,
        "dates_from_cli": False,
        "full_rolling": required == "PHASE_D",
    }


def _build_release_manifest(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    _require_execution(config, {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"})
    output_paths = _json(args.outputs_manifest)
    if not isinstance(output_paths, dict) or not output_paths:
        raise Stage2V52ContractError("outputs manifest must be a non-empty name-to-path mapping")
    payload = build_release_manifest(
        repo_root=args.repo_root,
        config_path=args.config,
        protocol_id=args.protocol,
        source_checkpoint_path=args.source_checkpoint,
        feature_artifact_path=args.feature_artifact,
        support_artifact_path=args.support_artifact,
        static_artifact_path=args.static_artifact,
        tau_artifact_path=args.tau_artifact,
        micro_cdf_path=args.micro_cdf,
        stage1_release=_json(args.stage1_release),
        output_paths=output_paths,
    )
    _write_json(args.output, payload)
    return {"status": "PASS", "output": args.output, "git_commit": payload["git_commit"]}


def _verify_final(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    _require_execution(config, {"PHASE_D_COMPLETE"})
    payload = _json(args.input)
    required_passes = payload.get("required_gates", {})
    passed = bool(required_passes) and all(value == "PASS" for value in required_passes.values())
    return {
        "status": "PASS" if passed else "FAIL",
        "stage2_status": "READY_FOR_AV_ROUTE_SUITABILITY_STAGE" if passed else "NOT_READY",
    }


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", default="stage2/config/stage2_v5_2.json")
    commands = root.add_subparsers(dest="command", required=True)
    protocol_choices = tuple(PROTOCOLS)
    command = commands.add_parser("preflight")
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--source-checkpoint", required=True)
    command.add_argument("--feature-artifact", required=True)
    command.set_defaults(function=_preflight)
    command = commands.add_parser("fit-support", aliases=["fit-train-support"])
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_fit_support)
    command = commands.add_parser("fit-static-artifact")
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_fit_static)
    command = commands.add_parser("fit-train-cdf")
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--quantile", type=float, default=0.90)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_fit_cdf)
    command = commands.add_parser("tune-tau")
    command.add_argument("--protocol", choices=("transfer_tuning",), default="transfer_tuning")
    command.add_argument("--metrics", required=True)
    command.add_argument("--support-artifact", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_tune_tau)
    command = commands.add_parser("train-model")
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--model", choices=("M1", "M2", "M3", "M4", "M5"), required=True)
    command.add_argument("--tensor-root", required=True)
    command.add_argument("--feature-artifact", required=True)
    command.add_argument("--source-checkpoint", required=True)
    command.add_argument("--source-model-id", required=True)
    command.add_argument("--static-feature-count", type=int, required=True)
    command.add_argument("--v5-1-metrics", required=True)
    command.add_argument("--tau-artifact")
    command.add_argument("--m4-adoption")
    command.add_argument("--temporal-leakage-count", type=int, required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_train_model)
    command = commands.add_parser("train-tree-baseline")
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--feature-schema-hash", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_train_tree_baseline)
    command = commands.add_parser("evaluate-model")
    command.add_argument("--kind", choices=("spatial", "temporal", "pace"), required=True)
    command.add_argument("--input", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_evaluate_model)
    command = commands.add_parser("build-products", aliases=["build-micro-products"])
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--predictions", required=True)
    command.add_argument("--route-context", required=True)
    command.add_argument("--support-artifact", required=True)
    command.add_argument("--train-cdf", required=True)
    command.add_argument("--prediction-source", required=True)
    command.add_argument("--model-id", required=True)
    command.add_argument("--model-hash", required=True)
    command.add_argument("--minimum-coverage", type=float, default=0.80)
    command.add_argument("--output-root", default="stage2/output_v5_2")
    command.set_defaults(function=_build_products)
    command = commands.add_parser("audit-static-schema")
    command.add_argument("--route-products", nargs="+", required=True)
    command.add_argument("--movement-products", nargs="+", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_audit_static)
    command = commands.add_parser("benchmark")
    command.add_argument("--sizes", nargs="+", type=int, default=[10_000, 50_000, 100_000, 500_000])
    command.add_argument("--output-csv", default="stage2/docs/v5_2/performance_benchmarks.csv")
    command.add_argument("--output-json", default="stage2/docs/v5_2/stage2_v5_2_performance_report.json")
    command.set_defaults(function=_benchmark)
    command = commands.add_parser("verify-phase-a")
    command.add_argument("--source-root", default="stage2/v5_2")
    command.set_defaults(function=_verify_phase_a)
    command = commands.add_parser("verify-phase-b")
    command.add_argument("--tokens", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_verify_phase_b)
    for name in ("run-development-day", "run-rolling"):
        command = commands.add_parser(name)
        command.add_argument("--run-spec", required=True)
        command.set_defaults(function=_run_plan)
    command = commands.add_parser("build-release-manifest")
    command.add_argument("--protocol", choices=protocol_choices, required=True)
    command.add_argument("--repo-root", default=".")
    command.add_argument("--source-checkpoint", required=True)
    command.add_argument("--feature-artifact", required=True)
    command.add_argument("--support-artifact", required=True)
    command.add_argument("--static-artifact", required=True)
    command.add_argument("--tau-artifact", required=True)
    command.add_argument("--micro-cdf", required=True)
    command.add_argument("--stage1-release", required=True)
    command.add_argument("--outputs-manifest", required=True)
    command.add_argument("--output", required=True)
    command.set_defaults(function=_build_release_manifest)
    command = commands.add_parser("verify-final")
    command.add_argument("--input", required=True)
    command.set_defaults(function=_verify_final)
    return root


def main() -> int:
    args = parser().parse_args()
    report = args.function(args)
    print(json.dumps(report, sort_keys=True))
    return 0 if report.get("status") in {"PASS", "READY_TO_EXECUTE"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
