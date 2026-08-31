"""Run the frozen deterministic-routing repositioning robustness protocol."""

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
from .paper_enhancement_repositioning_runner import (
    ASSIGNMENT_FIELDS,
    REQUEST_FIELDS,
    SUMMARY_FIELDS,
    write_gate_products,
)
from .repositioning_policy import POLICY_NAME, POLICY_VERSION


CONFIG_REL = Path("stage4/config/routing_determinism_spatial_closure_v2.json")
REGISTRY_REL = Path("stage4/output/paper_enhancement/experiment_registry.csv")
ROUTING_OUTPUT_REL = Path("stage4/output/paper_enhancement/routing_determinism")
CANONICAL_REL = Path("stage4/output/final_experiments")

BASES = (
    "MAIN_Q25_M_P70",
    "MAIN_Q50_M_P70",
    "MAIN_Q75_M_P70",
    "BENCH_AV_M",
)
CONTROL_IDS = {
    "MAIN_Q25_M_P70": "DET_Q25_M_P70_CONTROL",
    "MAIN_Q50_M_P70": "DET_Q50_M_P70_CONTROL",
    "MAIN_Q75_M_P70": "DET_Q75_M_P70_CONTROL",
    "BENCH_AV_M": "DET_BENCH_AV_M_CONTROL",
}
TREATMENT_IDS = {
    "MAIN_Q25_M_P70": "DET_Q25_M_P70_REPOS",
    "MAIN_Q50_M_P70": "DET_Q50_M_P70_REPOS",
    "MAIN_Q75_M_P70": "DET_Q75_M_P70_REPOS",
    "BENCH_AV_M": "DET_BENCH_AV_M_REPOS",
}
Q50_REPEAT_IDS = ("DET_Q50_M_P70_CONTROL_A", "DET_Q50_M_P70_CONTROL_B")


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


def _git_head(root: Path) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _fingerprint(frame: pd.DataFrame, columns: tuple[str, ...], sort: list[str]) -> str:
    selected = frame.loc[:, list(columns)].sort_values(sort, kind="mergesort")
    return hashlib.sha256(
        pd.util.hash_pandas_object(selected, index=False).values.tobytes()
    ).hexdigest()


def _exact_equal(left: Any, right: Any) -> bool:
    if isinstance(left, float) or isinstance(right, float):
        return bool(np.array_equal(np.asarray([left]), np.asarray([right]), equal_nan=True))
    return left == right


def _protocol_config(root: Path) -> dict[str, Any]:
    config = _read_json(root / CONFIG_REL)
    decision = _read_json(root / config["routing_mode_decision"])
    if config["routing_mode"] != "SINGLE_SOURCE_MATRIX":
        raise FleetPyCompatibilityError("deterministic routing mode is not frozen to M1")
    if decision.get("selected_routing_mode") != config["routing_mode"]:
        raise FleetPyCompatibilityError("routing decision and execution config disagree")
    if config.get("gamma_frontier_authorized") is not False:
        raise FleetPyCompatibilityError("Gamma frontier must remain unauthorized")
    return config


def _execution_config(root: Path, *, enabled: bool, q50_repeat: bool = False) -> dict[str, Any]:
    base = load_execution_config(root)
    frozen = _protocol_config(root)
    output_root = (
        frozen["q50_repeatability_output_root"]
        if q50_repeat
        else frozen["treatment_output_root"] if enabled else frozen["control_output_root"]
    )
    return {
        **base,
        "output_root": output_root,
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


def _scenario_row(root: Path, base_id: str, run_id: str, *, enabled: bool) -> dict[str, Any]:
    if base_id not in BASES:
        raise FleetPyCompatibilityError(f"unauthorized deterministic base: {base_id}")
    rows = {row["scenario_id"]: row for row in load_registry(root)}
    row = dict(rows[base_id])
    row["scenario_id"] = run_id
    row["experiment_block"] = "DET_REPOSITIONING" if enabled else "DET_CONTROL"
    return row


def _registry_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    specs = [("MAIN_Q50_M_P70", item, False) for item in Q50_REPEAT_IDS]
    specs += [(base, CONTROL_IDS[base], False) for base in BASES]
    specs += [(base, TREATMENT_IDS[base], True) for base in BASES]
    for base, run_id, enabled in specs:
        output_root = (
            config["q50_repeatability_output_root"]
            if run_id in Q50_REPEAT_IDS
            else config["treatment_output_root"] if enabled else config["control_output_root"]
        )
        rows.append(
            {
                "run_id": run_id,
                "workstream": "ROUTING_DETERMINISM_REPOSITIONING",
                "base_scenario": base,
                "variant": POLICY_NAME if enabled else "NO_REPOSITION_CONTROL",
                "scientific_question": "Does frozen Train-only AV rebalancing change deterministic-routing outcomes?",
                "changed_component": POLICY_NAME if enabled else "ARC_LEVEL_ROUTING_MODE",
                "unchanged_components": "fleet|acceptance|Profile_M|Gamma|candidate_pruning|solver|horizon",
                "seed": "20260827",
                "status": "PLANNED",
                "runtime": "",
                "output_path": f"{output_root}/{run_id}",
                "notes": "SINGLE_SOURCE_MATRIX; sequential CPU-only execution; canonical outputs unchanged.",
                "policy_version": POLICY_VERSION if enabled else "",
                "train_reference_provenance": (
                    "stage1 Train 20161009-20161024|sha256="
                    + config["repositioning_reference_sha256"]
                    if enabled
                    else ""
                ),
            }
        )
    return rows


def prepare_registry(root: Path) -> pd.DataFrame:
    config = _protocol_config(root)
    path = root / REGISTRY_REL
    registry = pd.read_csv(path, dtype=str).fillna("")
    for row in _registry_rows(config):
        mask = registry["run_id"].eq(row["run_id"])
        if bool(mask.any()):
            for key, value in row.items():
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
        raise FleetPyCompatibilityError(f"enhancement registry row not unique: {run_id}")
    for key, value in updates.items():
        registry.loc[mask, key] = str(value)
    _atomic_csv(registry, path)


def run_scenario(
    root: Path,
    fleetpy_root: Path,
    *,
    base_id: str,
    run_id: str,
    enabled: bool,
    q50_repeat: bool = False,
) -> dict[str, Any]:
    prepare_registry(root)
    config = _execution_config(root, enabled=enabled, q50_repeat=q50_repeat)
    row = _scenario_row(root, base_id, run_id, enabled=enabled)
    directory = root / config["output_root"] / run_id
    if (directory / "summary.json").is_file():
        raise FleetPyCompatibilityError(f"refusing to overwrite completed deterministic run: {run_id}")
    started = time.perf_counter()
    _update_registry(root, run_id, status="RUNNING", runtime="")
    try:
        summary = execute_scenario(root, fleetpy_root, row, config, _git_head(root))
        gate = write_gate_products(directory, run_id)
    except Exception as exc:
        _update_registry(
            root,
            run_id,
            status="FAILED",
            runtime=f"{time.perf_counter() - started:.6f}",
            notes=f"{type(exc).__name__}: {str(exc)[:500]}",
        )
        raise
    runtime = time.perf_counter() - started
    _update_registry(
        root,
        run_id,
        status="COMPLETED",
        runtime=f"{runtime:.6f}",
        notes="Frozen SINGLE_SOURCE_MATRIX; sequential CPU-only execution; canonical outputs unchanged.",
    )
    return {
        "scenario_id": run_id,
        "runtime_s": runtime,
        "matched": summary["matched"],
        "service_rate": summary["service_rate"],
        **gate,
    }


def compare_q50_repeatability(root: Path) -> dict[str, Any]:
    config = _protocol_config(root)
    base = root / config["q50_repeatability_output_root"]
    directories = [base / run_id for run_id in Q50_REPEAT_IDS]
    summaries = [_read_json(path / "summary.json") for path in directories]
    requests = [pd.read_parquet(path / "request_outcomes.parquet") for path in directories]
    assignments = [pd.read_parquet(path / "assignment_log.parquet") for path in directories]
    gates = [pd.read_csv(path / "gate_totals.csv").iloc[0] for path in directories]
    request_hashes = [
        _fingerprint(frame, REQUEST_FIELDS, ["order_id"]) for frame in requests
    ]
    assignment_hashes = [
        _fingerprint(frame, ASSIGNMENT_FIELDS, ["assignment_time", "order_id"])
        for frame in assignments
    ]
    required_summary = (
        "matched",
        "patience_expired",
        "HV_assignments",
        "AV_assignments",
        "service_rate",
    )
    summary_exact = all(
        _exact_equal(summaries[0][field], summaries[1][field]) for field in required_summary
    )
    gate_fields = [f"gate_av_n{index}_{suffix}" for index, suffix in (
        (0, "spatial"),
        (1, "passenger_compatible"),
        (2, "structurally_ready"),
        (3, "evidence_complete"),
        (4, "pickup_within_patience"),
        (5, "solver_eligible"),
        (6, "selected"),
    )]
    gate_exact = all(_exact_equal(gates[0][field], gates[1][field]) for field in gate_fields)
    exact = (
        request_hashes[0] == request_hashes[1]
        and assignment_hashes[0] == assignment_hashes[1]
        and summary_exact
        and gate_exact
    )
    row = {
        "scenario_a": Q50_REPEAT_IDS[0],
        "scenario_b": Q50_REPEAT_IDS[1],
        "request_fingerprint_a": request_hashes[0],
        "request_fingerprint_b": request_hashes[1],
        "assignment_fingerprint_a": assignment_hashes[0],
        "assignment_fingerprint_b": assignment_hashes[1],
        "request_outcomes_exact": request_hashes[0] == request_hashes[1],
        "assignments_exact": assignment_hashes[0] == assignment_hashes[1],
        "summary_exact": summary_exact,
        "gate_n0_n6_exact": gate_exact,
        "exact_repeatability": exact,
        **{f"{field}_a": summaries[0][field] for field in required_summary},
        **{f"{field}_b": summaries[1][field] for field in required_summary},
    }
    _atomic_csv(pd.DataFrame([row]), root / ROUTING_OUTPUT_REL / "q50_deterministic_repeatability.csv")
    if not exact:
        raise FleetPyCompatibilityError("STOP_FOR_ROUTING_NONDETERMINISM: Q50 A/B differ")
    return row


def compare_q50_canonical(root: Path) -> dict[str, Any]:
    repeat = compare_q50_repeatability(root)
    config = _protocol_config(root)
    det_dir = root / config["q50_repeatability_output_root"] / Q50_REPEAT_IDS[0]
    canonical_dir = root / CANONICAL_REL / "MAIN_Q50_M_P70"
    det = _read_json(det_dir / "summary.json")
    canonical = _read_json(canonical_dir / "summary.json")
    row = {
        "deterministic_scenario": Q50_REPEAT_IDS[0],
        "canonical_scenario": "MAIN_Q50_M_P70",
        "q50_ab_exact": repeat["exact_repeatability"],
        "delta_matched": det["matched"] - canonical["matched"],
        "delta_expired": det["patience_expired"] - canonical["patience_expired"],
        "delta_service_rate": det["service_rate"] - canonical["service_rate"],
        "delta_av_assignment_share": det["AV_assignment_share"] - canonical["AV_assignment_share"],
        "delta_p95_pickup_s": det["request_to_pickup_p95"] - canonical["request_to_pickup_p95"],
        "canonical_service_rate": canonical["service_rate"],
        "deterministic_service_rate": det["service_rate"],
    }
    _atomic_csv(pd.DataFrame([row]), root / ROUTING_OUTPUT_REL / "q50_deterministic_vs_canonical.csv")
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", choices=("prepare", "run-q50-a", "run-q50-b", "compare-q50", "run")
    )
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--fleetpy-root", type=Path)
    parser.add_argument("--scenario-id", choices=tuple(CONTROL_IDS.values()) + tuple(TREATMENT_IDS.values()))
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        result: Any = {"registry_rows": len(prepare_registry(root))}
    elif args.command == "compare-q50":
        result = compare_q50_canonical(root)
    else:
        if args.fleetpy_root is None:
            parser.error("--fleetpy-root is required for execution")
        fleetpy_root = args.fleetpy_root.resolve()
        if args.command in {"run-q50-a", "run-q50-b"}:
            run_id = Q50_REPEAT_IDS[0 if args.command.endswith("a") else 1]
            result = run_scenario(
                root,
                fleetpy_root,
                base_id="MAIN_Q50_M_P70",
                run_id=run_id,
                enabled=False,
                q50_repeat=True,
            )
        else:
            if args.scenario_id is None:
                parser.error("--scenario-id is required for run")
            reverse = {**{v: (k, False) for k, v in CONTROL_IDS.items()}, **{v: (k, True) for k, v in TREATMENT_IDS.items()}}
            base_id, enabled = reverse[args.scenario_id]
            result = run_scenario(
                root,
                fleetpy_root,
                base_id=base_id,
                run_id=args.scenario_id,
                enabled=enabled,
            )
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
