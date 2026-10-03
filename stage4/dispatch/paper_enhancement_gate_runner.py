"""Run four isolated shadow-logging anchors and verify canonical equivalence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .final_experiment_runner import (
    execute_scenario,
    load_execution_config,
    load_registry,
)
from .gate_diagnostics import (
    GATE_COLUMNS,
    LOSS_COLUMNS,
    aggregate_gate_epochs,
    validate_gate_counts,
)


ANCHORS = (
    "MAIN_Q25_M_P70",
    "MAIN_Q50_M_P70",
    "MAIN_Q75_M_P70",
    "BENCH_AV_M",
)
ENHANCEMENT_ROOT = Path("stage4/output/paper_enhancement")
RERUN_ROOT = ENHANCEMENT_ROOT / "gate_decomposition" / "reruns"
CANONICAL_ROOT = Path("stage4/output/final_experiments")
REGISTRY_REL = ENHANCEMENT_ROOT / "experiment_registry.csv"
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
REQUEST_OUTCOME_FIELDS = (
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
REGISTRY_COLUMNS = (
    "run_id",
    "workstream",
    "base_scenario",
    "variant",
    "scientific_question",
    "changed_component",
    "unchanged_components",
    "seed",
    "status",
    "runtime",
    "output_path",
    "notes",
)


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(value: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temp, index=False)
    os.replace(temp, path)


def _fingerprint(frame: pd.DataFrame, columns: tuple[str, ...], sort: list[str]) -> str:
    selected = frame.loc[:, list(columns)].sort_values(sort, kind="mergesort")
    hashed = pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    return hashlib.sha256(hashed).hexdigest()


def _registry_rows() -> list[dict[str, str]]:
    common = {
        "workstream": "prospective_gate_logging",
        "variant": "shadow_diagnostic_same_unit_av_opportunity",
        "scientific_question": "Where do nominal nearby AV opportunities cease to become dispatch-eligible?",
        "changed_component": "diagnostic_counters_only",
        "unchanged_components": "routing|candidate_pruning|solver|acceptance|fleet|seed|availability|canonical_outputs",
        "status": "PLANNED",
        "runtime": "",
        "notes": "UNCONSTRAINED anchor; Gamma excluded. Shared Top-K and routing are explicit sub-gates.",
    }
    return [
        {
            **common,
            "run_id": f"GATE_SHADOW_{scenario_id}",
            "base_scenario": scenario_id,
            "seed": "20260827",
            "output_path": str((RERUN_ROOT / scenario_id).as_posix()),
        }
        for scenario_id in ANCHORS
    ]


def ensure_registry(root: Path) -> pd.DataFrame:
    path = root / REGISTRY_REL
    if path.is_file():
        registry = pd.read_csv(path, dtype=str).fillna("")
    else:
        registry = pd.DataFrame(columns=REGISTRY_COLUMNS)
    planned = pd.DataFrame(_registry_rows())
    for row in planned.to_dict("records"):
        mask = registry["run_id"].eq(row["run_id"])
        if not bool(mask.any()):
            registry = pd.concat([registry, pd.DataFrame([row])], ignore_index=True)
    registry = registry.loc[:, list(REGISTRY_COLUMNS)]
    _atomic_csv(registry, path)
    return registry


def update_registry(root: Path, run_id: str, **updates: str) -> None:
    path = root / REGISTRY_REL
    registry = pd.read_csv(path, dtype=str).fillna("")
    mask = registry["run_id"].eq(run_id)
    if int(mask.sum()) != 1:
        raise FleetPyCompatibilityError(f"enhancement registry row not unique: {run_id}")
    for name, value in updates.items():
        registry.loc[mask, name] = str(value)
    _atomic_csv(registry, path)


def _exact_value_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return bool(np.array_equal(np.asarray([left]), np.asarray([right]), equal_nan=True))
    return left == right


def verify_canonical_reproduction(root: Path, scenario_id: str) -> dict[str, Any]:
    canonical_dir = root / CANONICAL_ROOT / scenario_id
    rerun_dir = root / RERUN_ROOT / scenario_id
    canonical_summary = _read_json(canonical_dir / "summary.json")
    rerun_summary = _read_json(rerun_dir / "summary.json")
    summary_differences = {
        name: {"canonical": canonical_summary[name], "rerun": rerun_summary[name]}
        for name in SUMMARY_FIELDS
        if not _exact_value_equal(canonical_summary[name], rerun_summary[name])
    }
    canonical_requests = pd.read_parquet(canonical_dir / "request_outcomes.parquet")
    rerun_requests = pd.read_parquet(rerun_dir / "request_outcomes.parquet")
    canonical_assignments = pd.read_parquet(canonical_dir / "assignment_log.parquet")
    rerun_assignments = pd.read_parquet(rerun_dir / "assignment_log.parquet")
    request_hash_canonical = _fingerprint(
        canonical_requests, REQUEST_OUTCOME_FIELDS, ["order_id"]
    )
    request_hash_rerun = _fingerprint(rerun_requests, REQUEST_OUTCOME_FIELDS, ["order_id"])
    assignment_hash_canonical = _fingerprint(
        canonical_assignments, ASSIGNMENT_FIELDS, ["assignment_time", "order_id"]
    )
    assignment_hash_rerun = _fingerprint(
        rerun_assignments, ASSIGNMENT_FIELDS, ["assignment_time", "order_id"]
    )
    result = {
        "scenario_id": scenario_id,
        "checked_summary_fields": list(SUMMARY_FIELDS),
        "summary_differences": summary_differences,
        "request_outcome_fingerprint_canonical": request_hash_canonical,
        "request_outcome_fingerprint_rerun": request_hash_rerun,
        "assignment_fingerprint_canonical": assignment_hash_canonical,
        "assignment_fingerprint_rerun": assignment_hash_rerun,
        "request_outcomes_exact": request_hash_canonical == request_hash_rerun,
        "assignments_exact": assignment_hash_canonical == assignment_hash_rerun,
    }
    result["canonical_reproduction_pass"] = (
        not summary_differences
        and result["request_outcomes_exact"]
        and result["assignments_exact"]
    )
    _atomic_json(result, rerun_dir / "canonical_reproduction.json")
    return result


def write_gate_products(root: Path, scenario_id: str) -> dict[str, Any]:
    directory = root / RERUN_ROOT / scenario_id
    epoch = pd.read_parquet(directory / "epoch_stats.parquet")
    totals = {name: int(epoch[name].sum()) for name in (*GATE_COLUMNS, *LOSS_COLUMNS)}
    validate_gate_counts(totals)
    config = _read_json(directory / "scenario_config.json")["scientific_configuration"]
    n0 = totals["gate_av_n0_spatial"]
    total_row = {
        "scenario_id": scenario_id,
        "requested_q_A": config["requested_q_A"],
        "target_p_A": config["acceptance_probability"],
        **totals,
        "eligibility_conversion_n5_over_n0": totals["gate_av_n5_solver_eligible"] / n0,
        "assignment_conversion_n6_over_n0": totals["gate_av_n6_selected"] / n0,
    }
    total = pd.DataFrame([total_row])
    _atomic_csv(total, directory / "gate_totals.csv")
    binned = aggregate_gate_epochs(epoch, 15)
    binned.insert(0, "scenario_id", scenario_id)
    _atomic_csv(binned, directory / "gate_15min.csv")
    temp = directory / "gate_15min.parquet.tmp"
    binned.to_parquet(temp, index=False)
    os.replace(temp, directory / "gate_15min.parquet")
    return total_row


def _enhancement_config(root: Path) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    config = load_execution_config(root)
    config = {
        **config,
        "output_root": str(RERUN_ROOT.as_posix()),
        "prospective_gate_logging": True,
        "gate_diagnostic_bin_minutes": 15,
        "max_parallel_scenarios": 1,
    }
    rows = {row["scenario_id"]: row for row in load_registry(root)}
    return config, rows


def run_anchor(root: Path, fleetpy_root: Path, scenario_id: str) -> dict[str, Any]:
    if scenario_id not in ANCHORS:
        raise FleetPyCompatibilityError(f"unauthorized gate anchor: {scenario_id}")
    config, rows = _enhancement_config(root)
    run_id = f"GATE_SHADOW_{scenario_id}"
    update_registry(root, run_id, status="RUNNING", runtime="", notes=f"started={_now()}")
    started = time.perf_counter()
    execution_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    try:
        execute_scenario(root, fleetpy_root, rows[scenario_id], config, execution_commit)
        total = write_gate_products(root, scenario_id)
        reproduction = verify_canonical_reproduction(root, scenario_id)
    except Exception as exc:
        update_registry(
            root,
            run_id,
            status="FAILED",
            runtime=f"{time.perf_counter() - started:.6f}",
            notes=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise
    runtime = time.perf_counter() - started
    status = "COMPLETED_REPRODUCED" if reproduction["canonical_reproduction_pass"] else "STOPPED_OUTCOME_MISMATCH"
    update_registry(
        root,
        run_id,
        status=status,
        runtime=f"{runtime:.6f}",
        notes=(
            "Shadow counters only; canonical request and assignment fingerprints match."
            if reproduction["canonical_reproduction_pass"]
            else "Canonical outcome mismatch; stop before scientific interpretation."
        ),
    )
    if not reproduction["canonical_reproduction_pass"]:
        raise FleetPyCompatibilityError(f"{scenario_id} did not reproduce canonical outcomes")
    return {"scenario_id": scenario_id, "runtime_s": runtime, **total}


def run_all(root: Path, fleetpy_root: Path, *, resume: bool) -> list[dict[str, Any]]:
    ensure_registry(root)
    results: list[dict[str, Any]] = []
    for scenario_id in ANCHORS:
        reproduction_path = root / RERUN_ROOT / scenario_id / "canonical_reproduction.json"
        if resume and reproduction_path.is_file():
            reproduction = _read_json(reproduction_path)
            if reproduction.get("canonical_reproduction_pass") is True:
                results.append(write_gate_products(root, scenario_id))
                continue
        results.append(run_anchor(root, fleetpy_root, scenario_id))
    totals = pd.concat(
        [pd.read_csv(root / RERUN_ROOT / scenario / "gate_totals.csv") for scenario in ANCHORS],
        ignore_index=True,
    )
    _atomic_csv(totals, root / ENHANCEMENT_ROOT / "gate_decomposition" / "gate_totals_all_anchors.csv")
    bins = pd.concat(
        [pd.read_csv(root / RERUN_ROOT / scenario / "gate_15min.csv") for scenario in ANCHORS],
        ignore_index=True,
    )
    _atomic_csv(bins, root / ENHANCEMENT_ROOT / "gate_decomposition" / "gate_15min_all_anchors.csv")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path, required=True)
    parser.add_argument("--scenario-id", choices=ANCHORS)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    ensure_registry(root)
    if args.scenario_id:
        result: Any = run_anchor(root, args.fleetpy_root.resolve(), args.scenario_id)
    else:
        result = run_all(root, args.fleetpy_root.resolve(), resume=args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
