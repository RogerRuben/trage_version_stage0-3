"""Protocol-bound, phase-gated command line interface for Stage 2 v5.2."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd

from .contracts import CONFIG_SCHEMA_VERSION, RESEARCH_CONTRACT, Stage2V52ContractError, validate_research_contract
from .evaluation import (
    build_tau_metrics_manifest, evaluate_checkpoint, evaluate_spatial_adoption,
    evaluate_rolling_spatial_adoption, evaluate_temporal_adoption, validate_evaluation_payload,
)
from .m0_features import (
    build_m0_feature_matrix, evaluate_m0_baseline, transform_m0_feature_matrix,
)
from .metadata_schema_audit import audit_input_metadata, write_metadata_audit
from .micro_products import (
    aggregate_original_route_micro_conditions, aggregate_static_route_complexity,
    build_micro_condition_tokens, fit_train_cdf_thresholds, write_partition_products,
)
from .performance import run_benchmarks, static_complexity_audit
from .phase_b0_smoke import run_phase_b0_smoke
from .protocols import PROTOCOLS, get_protocol
from .static_schema_audit import audit_static_schema, schema_names
from .structure_features import fit_static_structure_artifact
from .support_transfer import (
    fit_train_support_frame, freeze_tau_selection, select_tau_once, validate_embedded_hash,
)
from .training import train_micro_tree_baseline_from_npz, train_transfer_from_shards
from .transfer_data import build_transfer_shards
from .verification import (
    build_release_manifest, preflight, sha256_file, verify_artifact_payload,
    verify_final_gate_bundle, verify_one_train_one_validation_bucket, verify_phase_b,
)


COMMAND_AUTHORIZATIONS = {
    "preflight": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "audit-static-schema": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "audit-input-metadata": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "benchmark": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "verify-phase-b": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "verify-one-bucket-correctness": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "phase-b0-smoke": {"PHASE_B0"},
    "fit-support": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "fit-static-artifact": {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "fit-train-cdf": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "build-transfer-shards": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "build-m0-feature-matrix": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "transform-m0-feature-matrix": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "evaluate-m0": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "build-tau-metrics": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "tune-tau": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "freeze-tau": {"PHASE_B1"},
    "train-tree-baseline": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "train-model": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "evaluate-model": {"PHASE_B1", "PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "decide-spatial-adoption": {"PHASE_C", "PHASE_D", "PHASE_D_COMPLETE"},
    "decide-rolling-spatial-adoption": {"PHASE_D", "PHASE_D_COMPLETE"},
    "decide-temporal-adoption": {"PHASE_D", "PHASE_D_COMPLETE"},
    "build-products": {"PHASE_D", "PHASE_D_COMPLETE"},
    "build-release-manifest": {"PHASE_D_COMPLETE"},
    "verify-final": {"PHASE_D_COMPLETE"},
}


def _json(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise Stage2V52ContractError(f"expected JSON object: {path}")
    return value


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    temporary.write_bytes((json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    os.replace(temporary, destination)


def _config(args: argparse.Namespace) -> dict[str, Any]:
    config = _json(args.config)
    if config.get("schema_version") != CONFIG_SCHEMA_VERSION:
        raise Stage2V52ContractError("unknown Stage 2 v5.2 config schema")
    return config


def _require_execution(config: dict[str, Any], allowed: set[str]) -> None:
    authorization = str(config.get("execution_authorization", "NONE"))
    if authorization not in allowed:
        raise Stage2V52ContractError(f"execution authorization is {authorization}; allowed={sorted(allowed)}")
    if authorization == "PHASE_B1":
        gate = config.get("phase_b0_gate", {})
        path = Path(str(gate.get("report_path", "")))
        expected_hash = gate.get("report_sha256")
        if not path.is_file() or sha256_file(path) != expected_hash:
            raise Stage2V52ContractError("PHASE_B1 requires the hash-bound Phase B0 smoke report")
        report = _json(path)
        if (
            report.get("schema_version") != "stage2_v5_2_phase_b0_smoke.1"
            or report.get("status") != "PASS" or report.get("authorizes_phase_b1") is not True
        ):
            raise Stage2V52ContractError("Phase B0 smoke report does not authorize PHASE_B1")


def _resolve_training_tau(args: argparse.Namespace, config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    model, protocol_id = str(args.model), str(args.protocol)
    tau_candidate = getattr(args, "tau_candidate", None)
    tau_artifact_path = getattr(args, "tau_artifact", None)
    tau_freeze_path = getattr(args, "tau_freeze_artifact", None)
    support_path = getattr(args, "support_artifact", None)
    if model == "M4" and protocol_id == "transfer_tuning":
        if tau_candidate is None or support_path is None or tau_artifact_path is not None or tau_freeze_path is not None:
            raise Stage2V52ContractError(
                "transfer_tuning M4 requires --tau-candidate and --support-artifact, and forbids --tau-artifact"
            )
        support = _json(support_path)
        verify_artifact_payload(support, artifact_type="support")
        if support.get("protocol_id") != "transfer_tuning":
            raise Stage2V52ContractError("tau candidate must come from transfer_tuning Train support")
        key = str(tau_candidate)
        quantiles = support.get("positive_quantiles", {})
        if key not in {"p25", "p50", "p75"} or key not in quantiles:
            raise Stage2V52ContractError("tau candidate is absent from verified support quantiles")
        tau = float(quantiles[key])
        expected = [float(quantiles[name]) for name in ("p25", "p50", "p75")]
        if [float(value) for value in support.get("tau_candidates", ())] != expected:
            raise Stage2V52ContractError("support tau candidates do not equal Train P25/P50/P75")
        return tau, {
            "kind": "train_support_quantile_candidate", "candidate": key,
            "support_tau_candidate": key, "support_tau_value": tau,
            "support_tau_source_support_sha256": sha256_file(support_path),
            "support_artifact_sha256": sha256_file(support_path),
        }
    if model in {"M4", "M5"}:
        if tau_candidate is not None or tau_artifact_path is not None or tau_freeze_path is None or support_path is None:
            raise Stage2V52ContractError(
                "non-tuning M4/M5 requires --tau-freeze-artifact and --support-artifact; selection artifacts are not accepted"
            )
        support = _json(support_path)
        verify_artifact_payload(support, artifact_type="support")
        freeze = _json(tau_freeze_path)
        if (
            freeze.get("schema_version") != "stage2_v5_2_tau_freeze.1"
            or freeze.get("status") != "PASS"
            or freeze.get("selection_protocol") != "transfer_tuning"
            or freeze.get("rolling_reselection_allowed") is not False
        ):
            raise Stage2V52ContractError("M4/M5 tau artifact is not the one-time B1 freeze")
        validate_embedded_hash(freeze, name="tau freeze")
        freeze_sha = sha256_file(tau_freeze_path)
        expected_freeze_sha = config.get("tau_freeze", {}).get("expected_file_sha256")
        if not isinstance(expected_freeze_sha, str) or len(expected_freeze_sha) != 64:
            raise Stage2V52ContractError("Phase C/D config has no frozen tau file hash")
        if freeze_sha != expected_freeze_sha:
            raise Stage2V52ContractError("M4/M5 tau freeze hash differs from the frozen config")
        label = str(freeze.get("selected_candidate", ""))
        table = freeze.get("candidate_table", {})
        tau = float(freeze.get("selected_tau", float("nan")))
        if label not in {"p25", "p50", "p75"} or label not in table:
            raise Stage2V52ContractError("tau freeze has no valid selected label")
        if not math.isfinite(tau) or tau <= 0 or float(table[label].get("support_tau_value", float("nan"))) != tau:
            raise Stage2V52ContractError("frozen tau is not finite or differs from its candidate table")
        return tau, {
            "kind": "frozen_transfer_tuning_selection",
            "support_tau_candidate": label, "support_tau_value": tau,
            "support_tau_source_support_sha256": freeze["transfer_tuning_support_sha256"],
            "tau_freeze_artifact_sha256": freeze_sha,
            "tau_selection_artifact_sha256": freeze["tau_selection_artifact_sha256"],
            "metrics_manifest_sha256": freeze["metrics_manifest_sha256"],
            "current_protocol_support_artifact_sha256": sha256_file(support_path),
        }
    if tau_candidate is not None or tau_artifact_path is not None or tau_freeze_path is not None:
        raise Stage2V52ContractError("tau arguments are only legal for M4/M5")
    return 1.0, {"kind": "neutral_not_used_by_model", "value": 1.0}


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    return preflight(
        config_path=args.config, protocol_id=args.protocol,
        source_checkpoint_path=args.source_checkpoint, feature_artifact_path=args.feature_artifact,
        source_model_manifest_path=args.source_model_manifest, source_config_path=args.source_config,
    )


def _fit_support(args: argparse.Namespace) -> dict[str, Any]:
    protocol = get_protocol(args.protocol)
    frame = pd.read_parquet(args.input, columns=["split", "date", "order_id", "traversal_id", "observed_directed_edge_uid"])
    artifact = fit_train_support_frame(
        frame, protocol_id=args.protocol, protocol_train_dates=protocol.train_dates,
        input_sha256=sha256_file(args.input),
    ).to_payload()
    _write_json(args.output, artifact)
    return {"status": "PASS", "output": args.output}


def _fit_static(args: argparse.Namespace) -> dict[str, Any]:
    protocol = get_protocol(args.protocol)
    artifact = fit_static_structure_artifact(
        [pd.read_parquet(args.input)], protocol_id=args.protocol,
        protocol_train_dates=protocol.train_dates,
    ).to_payload()
    artifact["input_sha256"] = sha256_file(args.input)
    _write_json(args.output, artifact)
    return {"status": "PASS", "output": args.output}


def _fit_cdf(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); protocol = get_protocol(args.protocol)
    artifact = fit_train_cdf_thresholds(
        pd.read_parquet(args.input), protocol_id=args.protocol,
        protocol_train_dates=protocol.train_dates, input_sha256=sha256_file(args.input),
        quantile=float(config["products"]["high_exposure_train_cdf_quantile"]),
    )
    _write_json(args.output, artifact)
    return {"status": "PASS", "output": args.output}


def _build_transfer(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); shard = config["shards"]
    return build_transfer_shards(
        protocol_id=args.protocol, repo_root=args.repo_root,
        route_feature_root=args.route_feature_root, feature_artifact_path=args.feature_artifact,
        support_artifact_path=args.support_artifact, static_artifact_path=args.static_artifact,
        stage1_release_manifest_path=args.stage1_release, output_root=args.output_root,
        max_seq_len=int(shard["max_seq_len"]), overlap=int(shard["overlap"]),
        chunks_per_file=int(shard["chunks_per_file"]),
    )


def _build_m0(args: argparse.Namespace) -> dict[str, Any]:
    return build_m0_feature_matrix(
        protocol_id=args.protocol, repo_root=args.repo_root,
        route_feature_root=args.route_feature_root, output_matrix_path=args.output_matrix,
        output_manifest_path=args.output_manifest,
    )


def _transform_m0(args: argparse.Namespace) -> dict[str, Any]:
    return transform_m0_feature_matrix(
        protocol_id=args.protocol, role=args.role, repo_root=args.repo_root,
        route_feature_root=args.route_feature_root,
        train_matrix_manifest_path=args.train_matrix_manifest,
        support_artifact_path=args.support_artifact,
        output_matrix_path=args.output_matrix, output_manifest_path=args.output_manifest,
    )


def _evaluate_m0(args: argparse.Namespace) -> dict[str, Any]:
    return evaluate_m0_baseline(
        protocol_id=args.protocol, role=args.role, matrix_path=args.matrix,
        matrix_manifest_path=args.matrix_manifest, model_path=args.model,
        training_manifest_path=args.training_manifest, output_path=args.output,
    )


def _build_tau_metrics(args: argparse.Namespace) -> dict[str, Any]:
    report = build_tau_metrics_manifest(
        m1_evaluation_path=args.m1_evaluation, m4_evaluation_paths=args.m4_evaluations,
        support_artifact_path=args.support_artifact, feature_artifact_path=args.feature_artifact,
    )
    _write_json(args.output, report); return report


def _tune_tau(args: argparse.Namespace) -> dict[str, Any]:
    artifact = select_tau_once(
        _json(args.metrics), _json(args.support_artifact),
        metrics_manifest_sha256=sha256_file(args.metrics),
        support_artifact_sha256=sha256_file(args.support_artifact),
    )
    _write_json(args.output, artifact); return {"status": "PASS", "selected_tau": artifact["selected_tau"]}


def _freeze_tau(args: argparse.Namespace) -> dict[str, Any]:
    artifact = freeze_tau_selection(
        _json(args.selection), _json(args.metrics), _json(args.support_artifact),
        selection_artifact_sha256=sha256_file(args.selection),
        metrics_manifest_sha256=sha256_file(args.metrics),
        support_artifact_sha256=sha256_file(args.support_artifact),
    )
    _write_json(args.output, artifact)
    return {"status": "PASS", "selected_candidate": artifact["selected_candidate"], "selected_tau": artifact["selected_tau"]}


def _train_tree(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); _require_execution(config, {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D"})
    return train_micro_tree_baseline_from_npz(
        protocol_id=args.protocol, input_path=args.input, matrix_manifest_path=args.matrix_manifest,
        output_root=args.output, random_seed=int(config["reproducibility"]["base_seed"]),
    )


def _train_model(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); _require_execution(config, {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D"})
    training = config["training"]
    tau, tau_provenance = _resolve_training_tau(args, config)
    return train_transfer_from_shards(
        protocol_id=args.protocol, model_id=args.model, tensor_root=args.tensor_root,
        feature_artifact_path=args.feature_artifact, source_checkpoint_path=args.source_checkpoint,
        source_model_manifest_path=args.source_model_manifest, source_config_path=args.source_config,
        static_artifact_path=args.static_artifact, support_tau=tau,
        backbone_kwargs=config["backbone"],
        v5_1_metric_manifest=_json(args.m1_metrics) if args.m1_metrics else None,
        output_root=args.output, new_branch_lr=float(training["new_branch_lr"]),
        backbone_lr_ratio=float(training["backbone_lr_ratio"]),
        shared_freeze_epochs=int(training["shared_freeze_epochs"]),
        maximum_epochs=int(training["maximum_epochs"]), batch_size=int(training["batch_size"]),
        base_seed=int(config["reproducibility"]["base_seed"]),
        component_weights=training["loss_weights"],
        m4_checkpoint_path=args.m4_checkpoint,
        m4_training_manifest=_json(args.m4_training_manifest) if args.m4_training_manifest else None,
        m4_adoption_manifest=_json(args.m4_adoption) if args.m4_adoption else None,
        m4_adoption_manifest_sha256=sha256_file(args.m4_adoption) if args.m4_adoption else None,
        support_artifact_path=args.support_artifact,
        support_tau_provenance=tau_provenance,
    )


def _evaluate(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); _require_execution(config, {"PHASE_B0", "PHASE_B1", "PHASE_C", "PHASE_D"})
    return evaluate_checkpoint(
        protocol_id=args.protocol, model_id=args.model, role=args.role,
        tensor_root=args.tensor_root, checkpoint_path=args.checkpoint,
        training_manifest_path=args.training_manifest, output_root=args.output,
        batch_size=int(config["training"]["batch_size"]),
    )


def _decide_spatial(args: argparse.Namespace) -> dict[str, Any]:
    result = evaluate_spatial_adoption(m1=_json(args.m1), m2=_json(args.m2), m4=_json(args.m4))
    _write_json(args.output, result); return result


def _decide_temporal(args: argparse.Namespace) -> dict[str, Any]:
    result = evaluate_temporal_adoption(
        m4_evaluations=[_json(path) for path in args.m4_evaluations],
        m5_evaluations=[_json(path) for path in args.m5_evaluations],
        m4_adoption_manifest=_json(args.m4_adoption),
        m4_adoption_manifest_sha256=sha256_file(args.m4_adoption),
    )
    _write_json(args.output, result); return result


def _decide_rolling_spatial(args: argparse.Namespace) -> dict[str, Any]:
    result = evaluate_rolling_spatial_adoption(
        m1_evaluations=[_json(path) for path in args.m1_evaluations],
        m2_evaluations=[_json(path) for path in args.m2_evaluations],
        m4_evaluations=[_json(path) for path in args.m4_evaluations],
    )
    _write_json(args.output, result)
    return result


def _build_products(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); products = config["products"]
    evaluation = _json(args.evaluation_manifest)
    validate_evaluation_payload(evaluation, protocol_id=args.protocol)
    prediction_path = Path(str(evaluation["prediction_path"]))
    if sha256_file(prediction_path) != evaluation["prediction_sha256"]:
        raise Stage2V52ContractError("product predictions differ from formal evaluation manifest")
    tokens = build_micro_condition_tokens(
        pd.read_parquet(prediction_path), pd.read_parquet(args.route_context),
        support_artifact=_json(args.support_artifact), protocol_id=args.protocol,
        prediction_source="formal_checkpoint_evaluation", model_id=str(evaluation["model_id"]),
        model_hash=str(evaluation["checkpoint_sha256"]),
    )
    routes = aggregate_original_route_micro_conditions(
        tokens, _json(args.train_cdf),
        minimum_coverage=float(products["minimum_prediction_coverage"]),
        service_time_complete_threshold=float(products["service_time_complete_threshold"]),
    )
    return write_partition_products(tokens, routes, aggregate_static_route_complexity(tokens), output_root=args.output_root)


def _benchmark(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); perf = config["performance"]
    frame, report = run_benchmarks(
        tuple(int(value) for value in perf["sizes"]), warmup_runs=int(perf["warmup_runs"]),
        repeat_runs=int(perf["repeat_runs"]),
    )
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True); frame.to_csv(args.output_csv, index=False)
    _write_json(args.output_json, report); return report


def _verify_phase_a(args: argparse.Namespace) -> dict[str, Any]:
    validate_research_contract(RESEARCH_CONTRACT); audit = static_complexity_audit(args.source_root)
    return {"schema_version": "stage2_v5_2_phase_a_2_verification.1", "status": audit["status"], "research_contract": "PASS", "static_complexity": audit, "experiments_run": False, "stage2_status": "NOT_READY_IMPLEMENTATION_ONLY"}


def _verify_b(args: argparse.Namespace) -> dict[str, Any]:
    report = verify_phase_b(pd.read_parquet(args.tokens)); _write_json(args.output, report); return report


def _verify_one_bucket(args: argparse.Namespace) -> dict[str, Any]:
    report = verify_one_train_one_validation_bucket(
        train_traversal_path=args.train_traversal, train_label_path=args.train_label,
        validation_traversal_path=args.validation_traversal,
        validation_label_path=args.validation_label,
    )
    _write_json(args.output, report)
    return report


def _phase_b0_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args)
    report = run_phase_b0_smoke(
        config_path=args.config, protocol_id=args.protocol,
        source_checkpoint_path=args.source_checkpoint, feature_artifact_path=args.feature_artifact,
        source_model_manifest_path=args.source_model_manifest, source_config_path=args.source_config,
        support_artifact_path=args.support_artifact, static_artifact_path=args.static_artifact,
        train_route_feature_path=args.train_route_feature,
        train_traversal_path=args.train_traversal, train_label_path=args.train_label,
        validation_route_feature_path=args.validation_route_feature,
        validation_traversal_path=args.validation_traversal,
        validation_label_path=args.validation_label,
        max_seq_len=int(config["shards"]["max_seq_len"]), overlap=int(config["shards"]["overlap"]),
        backbone_kwargs=config["backbone"],
    )
    _write_json(args.output, report)
    return report


def _audit_static(args: argparse.Namespace) -> dict[str, Any]:
    report = audit_static_schema(route_columns=schema_names(args.route_products), movement_columns=schema_names(args.movement_products)); _write_json(args.output, report); return report


def _audit_input_metadata(args: argparse.Namespace) -> dict[str, Any]:
    report = audit_input_metadata(
        protocol_id=args.protocol, route_feature_root=args.route_feature_root,
        stage1_input_root=args.stage1_input_root, stage1_output_root=args.stage1_output_root,
    )
    write_metadata_audit(args.output, report); return report


def _release(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); _require_execution(config, {"PHASE_D_COMPLETE"})
    payload = build_release_manifest(
        repo_root=args.repo_root, config_path=args.config, protocol_id=args.protocol,
        source_checkpoint_path=args.source_checkpoint, feature_artifact_path=args.feature_artifact,
        source_model_manifest_path=args.source_model_manifest, source_config_path=args.source_config,
        support_artifact_path=args.support_artifact, static_artifact_path=args.static_artifact,
        tau_freeze_artifact_path=args.tau_freeze_artifact, micro_cdf_path=args.micro_cdf,
        transfer_manifest_path=args.transfer_manifest, training_manifest_path=args.training_manifest,
        selected_checkpoint_path=args.selected_checkpoint, evaluation_manifest_path=args.evaluation_manifest,
        stage1_release_manifest_path=args.stage1_release, output_paths=_json(args.outputs_manifest),
    )
    _write_json(args.output, payload); return payload


def _verify_final(args: argparse.Namespace) -> dict[str, Any]:
    config = _config(args); _require_execution(config, {"PHASE_D_COMPLETE"})
    result = verify_final_gate_bundle(_json(args.input)); _write_json(args.output, result); return result


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__); root.add_argument("--config", default="stage2/config/stage2_v5_2.json")
    commands = root.add_subparsers(dest="command", required=True); protocols = tuple(PROTOCOLS)
    def add_protocol(name: str) -> argparse.ArgumentParser:
        command = commands.add_parser(name); command.add_argument("--protocol", choices=protocols, required=True); return command
    command = add_protocol("preflight")
    for flag in ("source-checkpoint", "feature-artifact", "source-model-manifest", "source-config"): command.add_argument(f"--{flag}", required=True)
    command.set_defaults(function=_preflight)
    for name, function in (("fit-support", _fit_support), ("fit-static-artifact", _fit_static), ("fit-train-cdf", _fit_cdf)):
        command = add_protocol(name); command.add_argument("--input", required=True); command.add_argument("--output", required=True); command.set_defaults(function=function)
    command = add_protocol("build-transfer-shards")
    for flag in ("feature-artifact", "support-artifact", "static-artifact", "stage1-release", "route-feature-root"): command.add_argument(f"--{flag}", required=True)
    command.add_argument("--repo-root", default="."); command.add_argument("--output-root", required=True); command.set_defaults(function=_build_transfer)
    command = add_protocol("build-m0-feature-matrix"); command.add_argument("--repo-root", default="."); command.add_argument("--route-feature-root", required=True); command.add_argument("--output-matrix", required=True); command.add_argument("--output-manifest", required=True); command.set_defaults(function=_build_m0)
    command = add_protocol("transform-m0-feature-matrix"); command.add_argument("--role", choices=("validation", "evaluation", "calibration", "legacy"), required=True); command.add_argument("--repo-root", default="."); command.add_argument("--route-feature-root", required=True); command.add_argument("--train-matrix-manifest", required=True); command.add_argument("--support-artifact", required=True); command.add_argument("--output-matrix", required=True); command.add_argument("--output-manifest", required=True); command.set_defaults(function=_transform_m0)
    command = add_protocol("evaluate-m0"); command.add_argument("--role", choices=("validation", "evaluation", "calibration", "legacy"), required=True); command.add_argument("--matrix", required=True); command.add_argument("--matrix-manifest", required=True); command.add_argument("--model", required=True); command.add_argument("--training-manifest", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_evaluate_m0)
    command = commands.add_parser("build-tau-metrics"); command.add_argument("--m1-evaluation", required=True); command.add_argument("--m4-evaluations", nargs=3, required=True); command.add_argument("--support-artifact", required=True); command.add_argument("--feature-artifact", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_build_tau_metrics)
    command = commands.add_parser("tune-tau"); command.add_argument("--metrics", required=True); command.add_argument("--support-artifact", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_tune_tau)
    command = commands.add_parser("freeze-tau"); command.add_argument("--selection", required=True); command.add_argument("--metrics", required=True); command.add_argument("--support-artifact", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_freeze_tau)
    command = add_protocol("train-tree-baseline"); command.add_argument("--input", required=True); command.add_argument("--matrix-manifest", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_train_tree)
    command = add_protocol("train-model"); command.add_argument("--model", choices=("M1", "M2", "M3", "M4", "M5"), required=True)
    for flag in ("tensor-root", "feature-artifact", "source-checkpoint", "source-model-manifest", "source-config", "static-artifact", "output"): command.add_argument(f"--{flag}", required=True)
    for flag in ("m1-metrics", "tau-artifact", "tau-freeze-artifact", "support-artifact", "m4-checkpoint", "m4-training-manifest", "m4-adoption"): command.add_argument(f"--{flag}")
    command.add_argument("--tau-candidate", choices=("p25", "p50", "p75"))
    command.set_defaults(function=_train_model)
    command = add_protocol("evaluate-model"); command.add_argument("--model", choices=("M1", "M2", "M3", "M4", "M5"), required=True); command.add_argument("--role", choices=("train", "validation", "calibration", "evaluation", "legacy"), required=True)
    for flag in ("tensor-root", "checkpoint", "training-manifest", "output"): command.add_argument(f"--{flag}", required=True)
    command.set_defaults(function=_evaluate)
    command = commands.add_parser("decide-spatial-adoption"); command.add_argument("--m1", required=True); command.add_argument("--m2", required=True); command.add_argument("--m4", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_decide_spatial)
    command = commands.add_parser("decide-rolling-spatial-adoption"); command.add_argument("--m1-evaluations", nargs=3, required=True); command.add_argument("--m2-evaluations", nargs=3, required=True); command.add_argument("--m4-evaluations", nargs=3, required=True); command.add_argument("--output", required=True); command.set_defaults(function=_decide_rolling_spatial)
    command = commands.add_parser("decide-temporal-adoption"); command.add_argument("--m4-evaluations", nargs=3, required=True); command.add_argument("--m5-evaluations", nargs=3, required=True); command.add_argument("--m4-adoption", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_decide_temporal)
    command = add_protocol("build-products")
    for flag in ("evaluation-manifest", "route-context", "support-artifact", "train-cdf", "output-root"): command.add_argument(f"--{flag}", required=True)
    command.set_defaults(function=_build_products)
    command = commands.add_parser("audit-static-schema"); command.add_argument("--route-products", nargs="+", required=True); command.add_argument("--movement-products", nargs="+", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_audit_static)
    command = add_protocol("audit-input-metadata"); command.add_argument("--route-feature-root", required=True); command.add_argument("--stage1-input-root", required=True); command.add_argument("--stage1-output-root", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_audit_input_metadata)
    command = commands.add_parser("benchmark"); command.add_argument("--output-csv", required=True); command.add_argument("--output-json", required=True); command.set_defaults(function=_benchmark)
    command = commands.add_parser("verify-phase-a"); command.add_argument("--source-root", default="stage2/v5_2"); command.set_defaults(function=_verify_phase_a)
    command = commands.add_parser("verify-phase-b"); command.add_argument("--tokens", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_verify_b)
    command = commands.add_parser("verify-one-bucket-correctness"); command.add_argument("--train-traversal", required=True); command.add_argument("--train-label", required=True); command.add_argument("--validation-traversal", required=True); command.add_argument("--validation-label", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_verify_one_bucket)
    command = add_protocol("phase-b0-smoke")
    for flag in ("source-checkpoint", "feature-artifact", "source-model-manifest", "source-config", "support-artifact", "static-artifact", "train-route-feature", "train-traversal", "train-label", "validation-route-feature", "validation-traversal", "validation-label", "output"): command.add_argument(f"--{flag}", required=True)
    command.set_defaults(function=_phase_b0_smoke)
    command = add_protocol("build-release-manifest"); command.add_argument("--repo-root", default=".")
    for flag in ("source-checkpoint", "feature-artifact", "source-model-manifest", "source-config", "support-artifact", "static-artifact", "tau-freeze-artifact", "micro-cdf", "transfer-manifest", "training-manifest", "selected-checkpoint", "evaluation-manifest", "stage1-release", "outputs-manifest", "output"): command.add_argument(f"--{flag}", required=True)
    command.set_defaults(function=_release)
    command = commands.add_parser("verify-final"); command.add_argument("--input", required=True); command.add_argument("--output", required=True); command.set_defaults(function=_verify_final)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command != "verify-phase-a":
        allowed = COMMAND_AUTHORIZATIONS.get(args.command)
        if allowed is None:
            raise Stage2V52ContractError(f"command has no fail-closed authorization policy: {args.command}")
        _require_execution(_config(args), allowed)
    result = args.function(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status", "PASS") in {"PASS", "MICRO_FIRST_CHECKPOINT_SELECTED"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
