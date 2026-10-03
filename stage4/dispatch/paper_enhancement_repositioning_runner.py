"""Run the pre-registered Train-only AV repositioning robustness experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .final_experiment_runner import execute_scenario, load_execution_config, load_registry
from .gate_diagnostics import GATE_COLUMNS, LOSS_COLUMNS, aggregate_gate_epochs, validate_gate_counts
from .repositioning_policy import (
    POLICY_NAME,
    POLICY_VERSION,
    REFERENCE_MANIFEST_REL,
    REFERENCE_REL,
    load_train_demand_reference,
)


BASE_TO_R1 = {
    "MAIN_Q25_M_P70": "R1_Q25_M_P70_REPOS",
    "MAIN_Q50_M_P70": "R1_Q50_M_P70_REPOS",
    "MAIN_Q75_M_P70": "R1_Q75_M_P70_REPOS",
    "BENCH_AV_M": "R1_BENCH_AV_M_REPOS",
}
BASE_ANCHORS = tuple(BASE_TO_R1)
R1_SCENARIOS = tuple(BASE_TO_R1.values())
OUTPUT_REL = Path("stage4/output/paper_enhancement/repositioning_robustness")
NO_REPOS_REL = OUTPUT_REL / "no_reposition_reproduction"
ENABLED_REL = OUTPUT_REL / "enabled"
CANONICAL_REL = Path("stage4/output/final_experiments")
REGISTRY_REL = Path("stage4/output/paper_enhancement/experiment_registry.csv")
CONFIG_REL = Path("stage4/config/repositioning_robustness.json")
SUMMARY_FIELDS = (
    "request_count",
    "matched",
    "completed",
    "patience_expired",
    "HV_assignments",
    "AV_assignments",
    "service_rate",
    "AV_assignment_share",
    "request_to_pickup_mean",
    "request_to_pickup_p50",
    "request_to_pickup_p90",
    "request_to_pickup_p95",
    "first_window_match_rate",
    "carry_over_entry_rate",
    "carry_over_recovery_rate",
    "critical_order_count",
    "critical_recovery",
    "expanded_radius_match_share",
    "pickup_ETA_objective_value",
)
REQUEST_FIELDS = (
    "order_id",
    "request_time",
    "pickup_deadline",
    "first_attempt_time",
    "attempt_count",
    "failed_round_count",
    "matched",
    "assignment_time",
    "pickup_time",
    "completed",
    "patience_expired",
    "vehicle_type",
    "final_search_radius_m",
    "assignment_pickup_eta_s",
    "total_request_to_pickup_wait_s",
    "entered_carry_over",
    "entered_critical",
    "passenger_accepts_av",
    "acceptance_source",
)
ASSIGNMENT_FIELDS = (
    "assignment_time",
    "order_id",
    "vehicle_id",
    "vehicle_type",
    "pickup_eta_s",
    "pickup_route_distance_m",
    "predicted_service_time_s",
    "realized_service_time_s",
    "pickup_time",
    "service_end_time",
    "completed",
    "passenger_accepts_av",
    "exposure_static",
    "exposure_dynamic",
    "exposure_speed",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    os.replace(temp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _fingerprint(frame: pd.DataFrame, columns: tuple[str, ...], sort: list[str]) -> str:
    selected = frame.loc[:, list(columns)].sort_values(sort, kind="mergesort")
    return hashlib.sha256(
        pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    ).hexdigest()


def _exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return bool(np.array_equal(np.asarray([left]), np.asarray([right]), equal_nan=True))
    return left == right


def _load_r1_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_REL)
    if config.get("policy_name") != POLICY_NAME or config.get("policy_version") != POLICY_VERSION:
        raise FleetPyCompatibilityError("repositioning configuration identity mismatch")
    return config


def _execution_config(root: Path, *, enabled: bool) -> dict[str, Any]:
    base = load_execution_config(root)
    frozen = _load_r1_config(root)
    reference_manifest = _read_json(root / REFERENCE_MANIFEST_REL)
    return {
        **base,
        "output_root": (ENABLED_REL if enabled else NO_REPOS_REL).as_posix(),
        "prospective_gate_logging": True,
        "gate_diagnostic_bin_minutes": 15,
        "max_parallel_scenarios": 1,
        "full_day_runtime_guard_s": float(frozen["full_day_runtime_guard_s"]),
        "repositioning_enabled": bool(enabled),
        "repositioning_reference_sha256": reference_manifest["reference_sha256"],
        "repositioning_policy_name": POLICY_NAME,
        "repositioning_policy_version": POLICY_VERSION,
    }


def _registry_plan(reference_sha: str) -> pd.DataFrame:
    rows = []
    for base, scenario in BASE_TO_R1.items():
        rows.append(
            {
                "run_id": scenario,
                "workstream": "REPOSITIONING_ROBUSTNESS",
                "base_scenario": base,
                "variant": POLICY_NAME,
                "scientific_question": "Does Train-only AV rebalancing explain high-q_A service deterioration?",
                "changed_component": POLICY_NAME,
                "unchanged_components": "fleet|acceptance|Profile_M|Gamma|routing|candidate_pruning|solver|horizon",
                "seed": "20260827",
                "status": "PLANNED",
                "runtime": "",
                "output_path": (ENABLED_REL / scenario).as_posix(),
                "notes": "Favorable spatial robustness; empty deadhead is not ODD-certified.",
                "policy_version": POLICY_VERSION,
                "train_reference_provenance": f"stage1 Train 20161009-20161024|sha256={reference_sha}",
            }
        )
    return pd.DataFrame(rows)


def ensure_registry(root: Path, reference_sha: str) -> pd.DataFrame:
    path = root / REGISTRY_REL
    registry = pd.read_csv(path, dtype=str).fillna("")
    for column in ("policy_version", "train_reference_provenance"):
        if column not in registry.columns:
            registry[column] = ""
    for row in _registry_plan(reference_sha).to_dict("records"):
        mask = registry["run_id"].eq(row["run_id"])
        if not bool(mask.any()):
            registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    _atomic_csv(registry, path)
    return registry


def update_registry(root: Path, run_id: str, **updates: Any) -> None:
    path = root / REGISTRY_REL
    registry = pd.read_csv(path, dtype=str).fillna("")
    mask = registry["run_id"].eq(run_id)
    if int(mask.sum()) != 1:
        raise FleetPyCompatibilityError(f"registry row not unique: {run_id}")
    for key, value in updates.items():
        registry.loc[mask, key] = str(value)
    _atomic_csv(registry, path)


def prepare(root: Path) -> dict[str, Any]:
    reference, manifest = load_train_demand_reference(root)
    ensure_registry(root, manifest["reference_sha256"])
    return {
        "policy_name": POLICY_NAME,
        "policy_version": POLICY_VERSION,
        "reference_path": REFERENCE_REL.as_posix(),
        "reference_rows": len(reference),
        **manifest,
    }


def verify_no_reposition_reproduction(root: Path, base_scenario: str) -> dict[str, Any]:
    canonical = root / CANONICAL_REL / base_scenario
    rerun = root / NO_REPOS_REL / base_scenario
    left_summary = _read_json(canonical / "summary.json")
    right_summary = _read_json(rerun / "summary.json")
    differences = {
        name: {"canonical": left_summary[name], "rerun": right_summary[name]}
        for name in SUMMARY_FIELDS
        if not _exact_equal(left_summary[name], right_summary[name])
    }
    left_requests = pd.read_parquet(canonical / "request_outcomes.parquet")
    right_requests = pd.read_parquet(rerun / "request_outcomes.parquet")
    left_assignments = pd.read_parquet(canonical / "assignment_log.parquet")
    right_assignments = pd.read_parquet(rerun / "assignment_log.parquet")
    request_left = _fingerprint(left_requests, REQUEST_FIELDS, ["order_id"])
    request_right = _fingerprint(right_requests, REQUEST_FIELDS, ["order_id"])
    assignment_left = _fingerprint(
        left_assignments, ASSIGNMENT_FIELDS, ["assignment_time", "order_id"]
    )
    assignment_right = _fingerprint(
        right_assignments, ASSIGNMENT_FIELDS, ["assignment_time", "order_id"]
    )
    result = {
        "base_scenario": base_scenario,
        "repositioning_enabled": False,
        "summary_difference_count": len(differences),
        "summary_differences": differences,
        "request_fingerprint_canonical": request_left,
        "request_fingerprint_rerun": request_right,
        "assignment_fingerprint_canonical": assignment_left,
        "assignment_fingerprint_rerun": assignment_right,
        "request_outcomes_exact": request_left == request_right,
        "assignments_exact": assignment_left == assignment_right,
    }
    result["exact_reproduction"] = (
        not differences and result["request_outcomes_exact"] and result["assignments_exact"]
    )
    _atomic_json(result, rerun / "canonical_reproduction.json")
    return result


def write_gate_products(directory: Path, scenario_id: str) -> dict[str, Any]:
    epoch = pd.read_parquet(directory / "epoch_stats.parquet")
    totals = {name: int(epoch[name].sum()) for name in (*GATE_COLUMNS, *LOSS_COLUMNS)}
    validate_gate_counts(totals)
    n0 = totals["gate_av_n0_spatial"]
    row = {
        "scenario_id": scenario_id,
        **totals,
        "eligibility_conversion_n5_over_n0": totals["gate_av_n5_solver_eligible"] / n0,
        "assignment_conversion_n6_over_n0": totals["gate_av_n6_selected"] / n0,
        "structural_retention_n2_over_n1": totals["gate_av_n2_structurally_ready"]
        / totals["gate_av_n1_passenger_compatible"],
        "evidence_retention_n3_over_n2": totals["gate_av_n3_evidence_complete"]
        / totals["gate_av_n2_structurally_ready"],
        "patience_retention_n4_over_n3b": totals["gate_av_n4_pickup_within_patience"]
        / totals["gate_av_n3b_route_returned"],
    }
    _atomic_csv(pd.DataFrame([row]), directory / "gate_totals.csv")
    binned = aggregate_gate_epochs(epoch, 15)
    binned.insert(0, "scenario_id", scenario_id)
    _atomic_csv(binned, directory / "gate_15min.csv")
    return row


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def run_no_reposition_anchor(
    root: Path, fleetpy_root: Path, base_scenario: str
) -> dict[str, Any]:
    if base_scenario not in BASE_ANCHORS:
        raise FleetPyCompatibilityError(f"unauthorized baseline anchor: {base_scenario}")
    prepare(root)
    rows = {row["scenario_id"]: row for row in load_registry(root)}
    config = _execution_config(root, enabled=False)
    execute_scenario(root, fleetpy_root, rows[base_scenario], config, _git_head(root))
    directory = root / NO_REPOS_REL / base_scenario
    write_gate_products(directory, base_scenario)
    result = verify_no_reposition_reproduction(root, base_scenario)
    if not result["exact_reproduction"]:
        raise FleetPyCompatibilityError(
            f"STOP: repositioning_enabled=False changed {base_scenario}"
        )
    return result


def run_preflight(root: Path, fleetpy_root: Path, *, resume: bool) -> pd.DataFrame:
    rows = []
    for scenario in BASE_ANCHORS:
        path = root / NO_REPOS_REL / scenario / "canonical_reproduction.json"
        if resume and path.is_file() and _read_json(path).get("exact_reproduction") is True:
            rows.append(_read_json(path))
        else:
            rows.append(run_no_reposition_anchor(root, fleetpy_root, scenario))
    frame = pd.DataFrame(rows)
    _atomic_csv(frame, root / OUTPUT_REL / "canonical_no_reposition_reproduction.csv")
    if len(frame) != 4 or not frame["exact_reproduction"].all():
        raise FleetPyCompatibilityError("STOP: no-reposition reproduction is not 4/4")
    return frame


def _enabled_row(root: Path, r1_scenario: str) -> dict[str, Any]:
    reverse = {value: key for key, value in BASE_TO_R1.items()}
    if r1_scenario not in reverse:
        raise FleetPyCompatibilityError(f"unauthorized repositioning scenario: {r1_scenario}")
    rows = {row["scenario_id"]: row for row in load_registry(root)}
    row = dict(rows[reverse[r1_scenario]])
    row["scenario_id"] = r1_scenario
    row["experiment_block"] = "REPOSITIONING_ROBUSTNESS"
    return row


def run_enabled_anchor(root: Path, fleetpy_root: Path, r1_scenario: str) -> dict[str, Any]:
    preflight = root / OUTPUT_REL / "canonical_no_reposition_reproduction.csv"
    if not preflight.is_file():
        raise FleetPyCompatibilityError("no-reposition preflight has not been completed")
    check = pd.read_csv(preflight)
    if len(check) != 4 or not check["exact_reproduction"].astype(bool).all():
        raise FleetPyCompatibilityError("no-reposition preflight did not pass 4/4")
    config = _execution_config(root, enabled=True)
    row = _enabled_row(root, r1_scenario)
    started = time.perf_counter()
    update_registry(root, r1_scenario, status="RUNNING", runtime="")
    try:
        summary = execute_scenario(root, fleetpy_root, row, config, _git_head(root))
        gate = write_gate_products(root / ENABLED_REL / r1_scenario, r1_scenario)
    except Exception as exc:
        update_registry(
            root,
            r1_scenario,
            status="FAILED",
            runtime=f"{time.perf_counter() - started:.6f}",
            notes=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise
    runtime = time.perf_counter() - started
    update_registry(
        root,
        r1_scenario,
        status="COMPLETED",
        runtime=f"{runtime:.6f}",
        notes="Train-only 15-minute demand balancing; empty routes are an operational abstraction.",
    )
    return {
        "scenario_id": r1_scenario,
        "runtime_s": runtime,
        "matched": summary["matched"],
        "service_rate": summary["service_rate"],
        **gate,
        **summary["repositioning"],
    }


def run_enabled_all(root: Path, fleetpy_root: Path, *, resume: bool) -> list[dict[str, Any]]:
    results = []
    for scenario in R1_SCENARIOS:
        summary_path = root / ENABLED_REL / scenario / "summary.json"
        gate_path = root / ENABLED_REL / scenario / "gate_totals.csv"
        if resume and summary_path.is_file() and gate_path.is_file():
            summary = _read_json(summary_path)
            gate = pd.read_csv(gate_path).iloc[0].to_dict()
            results.append(
                {
                    "scenario_id": scenario,
                    "matched": summary["matched"],
                    "service_rate": summary["service_rate"],
                    **gate,
                    **summary["repositioning"],
                }
            )
        else:
            results.append(run_enabled_anchor(root, fleetpy_root, scenario))
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("prepare", "preflight", "run-enabled", "all"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path)
    parser.add_argument("--scenario-id", choices=(*BASE_ANCHORS, *R1_SCENARIOS))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    if args.mode == "prepare":
        result: Any = prepare(root)
    else:
        if args.fleetpy_root is None:
            parser.error("--fleetpy-root is required for execution")
        fleetpy_root = args.fleetpy_root.resolve()
        if args.mode == "preflight":
            result = (
                run_no_reposition_anchor(root, fleetpy_root, args.scenario_id)
                if args.scenario_id
                else run_preflight(root, fleetpy_root, resume=args.resume)
            )
        elif args.mode == "run-enabled":
            result = (
                run_enabled_anchor(root, fleetpy_root, args.scenario_id)
                if args.scenario_id
                else run_enabled_all(root, fleetpy_root, resume=args.resume)
            )
        else:
            run_preflight(root, fleetpy_root, resume=args.resume)
            result = run_enabled_all(root, fleetpy_root, resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
