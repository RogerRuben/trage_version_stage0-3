"""Execute the five-run R0.5c deterministic repositioning protocol."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import pandas as pd

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .deterministic_repositioning_runner import (
    ASSIGNMENT_FIELDS,
    REQUEST_FIELDS,
    SUMMARY_FIELDS,
    _atomic_csv,
    _exact_equal,
    _fingerprint,
    _git_head,
    _read_json,
)
from .final_experiment_runner import execute_scenario, load_execution_config, load_registry
from .paper_enhancement_repositioning_runner import write_gate_products
from .repositioning_policy import POLICY_NAME, POLICY_VERSION


CONFIG_REL = Path("stage4/config/efficient_routing_repositioning.json")
REGISTRY_REL = Path("stage4/output/paper_enhancement/experiment_registry.csv")
OUTPUT_REL = Path("stage4/output/paper_enhancement/efficient_repositioning")
CANONICAL_REL = Path("stage4/output/final_experiments")

RUNS = {
    "DET_Q50_CONTROL_A": ("MAIN_Q50_M_P70", False),
    "DET_Q50_CONTROL_B": ("MAIN_Q50_M_P70", False),
    "DET_Q50_REPOS": ("MAIN_Q50_M_P70", True),
    "DET_Q75_CONTROL": ("MAIN_Q75_M_P70", False),
    "DET_Q75_REPOS": ("MAIN_Q75_M_P70", True),
}


def protocol_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_REL)
    evidence = _read_json(root / config["routing_evidence_source"])
    records = {row["routing_mode"]: row for row in evidence["performance"]}
    if config["routing_mode"] != "SCALAR_ROUTE":
        raise FleetPyCompatibilityError("R0.5c must use the faster stable scalar mode")
    if any(records[mode]["failure_count"] for mode in ("SINGLE_SOURCE_MATRIX", "SCALAR_ROUTE")):
        raise FleetPyCompatibilityError("routing evidence contains unexplained failures")
    if records["SCALAR_ROUTE"]["arcs_per_second"] <= records["SINGLE_SOURCE_MATRIX"]["arcs_per_second"]:
        raise FleetPyCompatibilityError("scalar mode is not faster in the frozen spike")
    if config["max_required_full_day_runs"] != len(RUNS):
        raise FleetPyCompatibilityError("efficient protocol must remain exactly five runs")
    if any(config[key] for key in ("q25_authorized", "all_av_authorized", "gamma_frontier_authorized")):
        raise FleetPyCompatibilityError("unauthorized experiment expansion in R0.5c")
    return config


def execution_config(root: Path, enabled: bool) -> dict[str, Any]:
    base = load_execution_config(root)
    frozen = protocol_config(root)
    return {
        **base,
        "output_root": frozen["output_root"],
        "routing_mode": frozen["routing_mode"],
        "assignment_matrix_representation": frozen["assignment_matrix_representation"],
        "gpu_usage": frozen["gpu_usage"],
        "max_parallel_scenarios": 1,
        "full_day_runtime_guard_s": float(frozen["full_day_runtime_guard_s"]),
        "prospective_gate_logging": bool(frozen["prospective_gate_logging"]),
        "gate_diagnostic_bin_minutes": int(frozen["gate_diagnostic_bin_minutes"]),
        "repositioning_enabled": bool(enabled),
        "repositioning_reference_sha256": frozen["repositioning_reference_sha256"],
        "repositioning_policy_name": POLICY_NAME,
        "repositioning_policy_version": POLICY_VERSION,
    }


def _registry_spec(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for run_id, (base, enabled) in RUNS.items():
        rows.append(
            {
                "run_id": run_id,
                "workstream": "EFFICIENT_REPOSITIONING_R0.5C",
                "base_scenario": base,
                "variant": POLICY_NAME if enabled else "DETERMINISTIC_CONTROL",
                "scientific_question": "Does Train-only AV rebalancing explain medium/high-q service deterioration?",
                "changed_component": POLICY_NAME if enabled else "SCALAR_ROUTE",
                "unchanged_components": "fleet|acceptance|Profile_M|Gamma|candidate_pruning|solver|horizon",
                "seed": "20260827",
                "status": "PLANNED",
                "runtime": "",
                "output_path": f"{config['output_root']}/{run_id}",
                "notes": "Frozen SCALAR_ROUTE; sequential CPU-only; Q25/all-AV/Gamma unauthorized.",
                "policy_version": POLICY_VERSION if enabled else "",
                "train_reference_provenance": (
                    "stage1 Train 20161009-20161024|sha256="
                    + config["repositioning_reference_sha256"]
                    if enabled else ""
                ),
            }
        )
    return rows


def prepare_registry(root: Path) -> pd.DataFrame:
    config = protocol_config(root)
    path = root / REGISTRY_REL
    registry = pd.read_csv(path, dtype=str).fillna("")
    old = registry["workstream"].eq("ROUTING_DETERMINISM_REPOSITIONING")
    registry.loc[old, "status"] = "SUPERSEDED_BY_R0.5C"
    registry.loc[old, "notes"] = "Superseded before a valid full-day result by efficiency-first R0.5c."
    for row in _registry_spec(config):
        mask = registry["run_id"].eq(row["run_id"])
        if bool(mask.any()):
            protected = {
                "COMPLETED",
                "RUNNING",
                "STOPPED_NOT_COST_EFFECTIVE",
                "NOT_RUN_COST_STOP",
            }
            for key, value in row.items():
                if (
                    registry.loc[mask, "status"].iloc[0] not in protected
                    or key not in {"status", "runtime", "notes"}
                ):
                    registry.loc[mask, key] = value
        else:
            registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    _atomic_csv(registry, path)
    return registry


def _update_registry(root: Path, run_id: str, **updates: Any) -> None:
    path = root / REGISTRY_REL
    registry = pd.read_csv(path, dtype=str).fillna("")
    mask = registry["run_id"].eq(run_id)
    if int(mask.sum()) != 1:
        raise FleetPyCompatibilityError(f"registry row not unique: {run_id}")
    for key, value in updates.items():
        registry.loc[mask, key] = str(value)
    _atomic_csv(registry, path)


def _scenario_row(root: Path, run_id: str) -> dict[str, Any]:
    base_id, enabled = RUNS[run_id]
    rows = {row["scenario_id"]: row for row in load_registry(root)}
    row = dict(rows[base_id])
    row["scenario_id"] = run_id
    row["experiment_block"] = "EFFICIENT_REPOSITIONING" if enabled else "EFFICIENT_CONTROL"
    return row


def run_one(root: Path, fleetpy_root: Path, run_id: str) -> dict[str, Any]:
    prepare_registry(root)
    base_id, enabled = RUNS[run_id]
    config = execution_config(root, enabled)
    directory = root / config["output_root"] / run_id
    if (directory / "summary.json").is_file():
        raise FleetPyCompatibilityError(f"refusing to overwrite completed run: {run_id}")
    started = time.perf_counter()
    _update_registry(root, run_id, status="RUNNING", runtime="")
    try:
        summary = execute_scenario(
            root, fleetpy_root, _scenario_row(root, run_id), config, _git_head(root)
        )
        gate = write_gate_products(directory, run_id)
    except Exception as exc:
        _update_registry(
            root, run_id, status="FAILED", runtime=f"{time.perf_counter()-started:.6f}",
            notes=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise
    runtime = time.perf_counter() - started
    _update_registry(
        root, run_id, status="COMPLETED", runtime=f"{runtime:.6f}",
        notes="Frozen SCALAR_ROUTE; sequential CPU-only; canonical output unchanged.",
    )
    return {"run_id": run_id, "base_id": base_id, "runtime_s": runtime, "matched": summary["matched"], "service_rate": summary["service_rate"], **gate}


def _run_dir(root: Path, run_id: str) -> Path:
    return root / protocol_config(root)["output_root"] / run_id


def q50_repeatability(root: Path) -> dict[str, Any]:
    directories = [_run_dir(root, run_id) for run_id in ("DET_Q50_CONTROL_A", "DET_Q50_CONTROL_B")]
    summaries = [_read_json(path / "summary.json") for path in directories]
    requests = [pd.read_parquet(path / "request_outcomes.parquet") for path in directories]
    assignments = [pd.read_parquet(path / "assignment_log.parquet") for path in directories]
    gates = [pd.read_csv(path / "gate_totals.csv").iloc[0] for path in directories]
    request_hash = [_fingerprint(frame, REQUEST_FIELDS, ["order_id"]) for frame in requests]
    assignment_hash = [_fingerprint(frame, ASSIGNMENT_FIELDS, ["assignment_time", "order_id"]) for frame in assignments]
    fields = ("matched", "patience_expired", "HV_assignments", "AV_assignments", "service_rate")
    gate_fields = tuple(f"gate_av_n{i}_{name}" for i, name in ((0,"spatial"),(1,"passenger_compatible"),(2,"structurally_ready"),(3,"evidence_complete"),(4,"pickup_within_patience"),(5,"solver_eligible"),(6,"selected")))
    row = {
        "request_outcomes_exact": request_hash[0] == request_hash[1],
        "assignments_exact": assignment_hash[0] == assignment_hash[1],
        "summary_exact": all(_exact_equal(summaries[0][f], summaries[1][f]) for f in fields),
        "gate_n0_n6_exact": all(_exact_equal(gates[0][f], gates[1][f]) for f in gate_fields),
        "request_fingerprint_a": request_hash[0], "request_fingerprint_b": request_hash[1],
        "assignment_fingerprint_a": assignment_hash[0], "assignment_fingerprint_b": assignment_hash[1],
    }
    row["exact_repeatability"] = all(row[key] for key in ("request_outcomes_exact","assignments_exact","summary_exact","gate_n0_n6_exact"))
    for field in fields:
        row[f"{field}_a"] = summaries[0][field]
        row[f"{field}_b"] = summaries[1][field]
    _atomic_csv(pd.DataFrame([row]), root / OUTPUT_REL / "q50_repeatability.csv")
    if not row["exact_repeatability"]:
        raise FleetPyCompatibilityError("STOP_FOR_ROUTING_NONDETERMINISM")
    return row


def canonical_sensitivity(root: Path, run_id: str) -> dict[str, Any]:
    base_id, _ = RUNS[run_id]
    det = _read_json(_run_dir(root, run_id) / "summary.json")
    canonical = _read_json(root / CANONICAL_REL / base_id / "summary.json")
    return {
        "run_id": run_id,
        "canonical_id": base_id,
        "delta_matched": det["matched"] - canonical["matched"],
        "delta_service_rate": det["service_rate"] - canonical["service_rate"],
        "delta_av_assignment_share": det["AV_assignment_share"] - canonical["AV_assignment_share"],
        "delta_p95_pickup_s": det["request_to_pickup_p95"] - canonical["request_to_pickup_p95"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "run", "q50-gate", "sensitivity"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path)
    parser.add_argument("--run-id", choices=tuple(RUNS))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        result: Any = {"registry_rows": len(prepare_registry(root)), "runs": list(RUNS)}
    elif args.command == "q50-gate":
        result = q50_repeatability(root)
    elif args.command == "sensitivity":
        if args.run_id not in {"DET_Q50_CONTROL_A", "DET_Q75_CONTROL"}:
            parser.error("sensitivity is defined only for Q50 control A or Q75 control")
        result = canonical_sensitivity(root, args.run_id)
    else:
        if args.run_id is None or args.fleetpy_root is None:
            parser.error("run requires --run-id and --fleetpy-root")
        result = run_one(root, args.fleetpy_root.resolve(), args.run_id)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
