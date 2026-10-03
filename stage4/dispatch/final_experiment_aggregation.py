"""Aggregate completed Stage4 final experiments without loading raw products together."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .final_experiment_runner import (
    PHASE_A_IDS,
    _atomic_json,
    _git_head,
    _read_json,
    completed_is_reusable,
    load_execution_config,
    load_registry,
    required_outputs_exist,
    scenario_dir,
)

AGGREGATE_REL = Path("stage4/output/final_experiments/_aggregate")
DOC_REL = Path("stage4/docs/final_experiments")
FAMILIES = ("static", "dynamic", "speed")


def _scalar_summary(summary: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in summary.items()
        if not isinstance(value, (dict, list, tuple))
    }
    row.update({f"runtime_{key}": value for key, value in summary["runtime"].items()})
    return row


def _write_csv(rows: Iterable[dict[str, Any]], path: Path) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def load_completed_unique(
    root: str | Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    root = Path(root).resolve()
    config = load_execution_config(root)
    registry = load_registry(root, config)
    execution_commit = _git_head(root)
    summaries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for row in registry:
        if row["reuse_source_scenario_id"]:
            continue
        directory = scenario_dir(root, row["scenario_id"], config)
        status_path = directory / "run_status.json"
        if not status_path.is_file():
            failures.append({"scenario_id": row["scenario_id"], "status": "MISSING"})
            continue
        status = _read_json(status_path)
        if status.get("status") != "COMPLETED":
            failures.append(
                {
                    "scenario_id": row["scenario_id"],
                    "status": status.get("status"),
                    "error_type": status.get("error_type"),
                    "error_message": status.get("error_message"),
                }
            )
            continue
        if not completed_is_reusable(root, row, execution_commit, config):
            failures.append(
                {"scenario_id": row["scenario_id"], "status": "PROVENANCE_MISMATCH"}
            )
            continue
        if not required_outputs_exist(directory):
            failures.append(
                {"scenario_id": row["scenario_id"], "status": "OUTPUT_MISSING"}
            )
            continue
        summary = _read_json(directory / "summary.json")
        summaries.append(summary)
    return summaries, failures, {"config": config, "registry": registry}


def _resolved_rows(
    unique: list[dict[str, Any]], registry: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {row["scenario_id"]: row for row in unique}
    resolved: list[dict[str, Any]] = []
    for registry_row in registry:
        source_id = (
            registry_row["reuse_source_scenario_id"] or registry_row["scenario_id"]
        )
        if source_id not in by_id:
            raise FleetPyCompatibilityError(
                f"reuse source is not completed: {registry_row['scenario_id']} -> {source_id}"
            )
        row = dict(by_id[source_id])
        row["scenario_id"] = registry_row["scenario_id"]
        row["experiment_block"] = registry_row["experiment_block"]
        row["scientific_role"] = registry_row["scientific_role"]
        row["reuse_source_scenario_id"] = registry_row["reuse_source_scenario_id"]
        resolved.append(row)
    return resolved


def _cost_degradation(rows: list[dict[str, Any]]) -> None:
    references = {
        float(row["eta_cost_av_to_hv"]): float(row["pickup_ETA_objective_value"])
        for row in rows
        if row["experiment_block"] == "COST_ROBUSTNESS"
        and float(row["pickup_cost_epsilon"]) == 0.0
    }
    for row in rows:
        if row["experiment_block"] != "COST_ROBUSTNESS":
            continue
        reference = references[float(row["eta_cost_av_to_hv"])]
        row["relative_pickup_ETA_degradation_vs_epsilon_0_reference"] = (
            float(row["pickup_ETA_objective_value"]) / reference - 1.0
            if reference
            else 0.0
        )


def _family_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        if int(row["AV_assignments"]) == 0:
            continue
        for family in FAMILIES:
            output.append(
                {
                    "scenario_id": row["scenario_id"],
                    "experiment_block": row["experiment_block"],
                    "profile_id": row["profile_id"],
                    "requested_q_A": row["requested_q_A"],
                    "target_p_A": row["target_p_A"],
                    "gamma_policy": row["gamma_policy"],
                    "family": family,
                    "mean_assigned_exposure": row[f"mean_assigned_exposure_{family}"],
                    "positive_assigned_exposure_share": row[
                        f"positive_assigned_exposure_share_{family}"
                    ],
                    "final_cumulative_mean_exposure": row[
                        f"final_cumulative_mean_exposure_{family}"
                    ],
                    "maximum_cumulative_mean_exposure": row[
                        f"maximum_cumulative_mean_exposure_{family}"
                    ],
                    "binding_epoch_count": row[f"binding_epoch_count_{family}"],
                    "near_binding_epoch_count": row[
                        f"near_binding_epoch_count_{family}"
                    ],
                    "minimum_slack": row[f"minimum_slack_{family}"],
                    "mean_slack": row[f"mean_slack_{family}"],
                }
            )
    return output


def _runtime_rows(unique: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "scenario_id": row["scenario_id"],
            "experiment_block": row["experiment_block"],
            **row["runtime"],
        }
        for row in unique
    ]


def _integrity(
    unique: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> None:
    if len(unique) != 41 or len(resolved) != 42:
        raise FleetPyCompatibilityError(
            f"expected 41 unique/42 resolved, got {len(unique)}/{len(resolved)}"
        )
    gamma_by_id = {row["scenario_id"]: row for row in registry}
    for row in unique:
        if not (
            int(row["request_count"]) == 30000
            and int(row["matched"]) <= int(row["request_count"])
            and int(row["completed"]) <= int(row["matched"])
            and int(row["HV_assignments"]) + int(row["AV_assignments"])
            == int(row["matched"])
            and 0.0 <= float(row["realized_accepted_order_share"]) <= 1.0
            and 0.0 <= float(row["service_rate"]) <= 1.0
        ):
            raise FleetPyCompatibilityError(
                f"aggregate integrity failure: {row['scenario_id']}"
            )
        for family in FAMILIES:
            gamma = gamma_by_id[row["scenario_id"]][f"gamma_{family}"]
            if (
                gamma is not None
                and float(row[f"final_cumulative_mean_exposure_{family}"])
                > float(gamma) + 1e-7
            ):
                raise FleetPyCompatibilityError(
                    f"aggregate Gamma failure: {row['scenario_id']} {family}"
                )


def _make_figures(rows: list[dict[str, Any]], output: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    main = pd.DataFrame(
        [row for row in rows if row["experiment_block"] == "MAIN_STRUCTURAL"]
    )
    odd = pd.DataFrame([row for row in rows if row["experiment_block"] == "ODD_POLICY"])
    cost = pd.DataFrame(
        [row for row in rows if row["experiment_block"] == "COST_ROBUSTNESS"]
    )
    paths: list[str] = []

    def save(name: str) -> None:
        path = figures / name
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        paths.append(str(path))

    plt.figure(figsize=(8, 5))
    for (profile, p_a), group in main.groupby(["profile_id", "target_p_A"]):
        group = group.sort_values("requested_q_A")
        plt.plot(
            group["requested_q_A"],
            group["service_rate"],
            marker="o",
            label=f"{profile}, p={p_a:g}",
        )
    plt.xlabel("Requested AV vehicle-hour share")
    plt.ylabel("Service rate")
    plt.legend(ncol=3, fontsize=7)
    save("01_service_rate_vs_qA.png")

    plt.figure(figsize=(8, 5))
    for (profile, p_a), group in main.groupby(["profile_id", "target_p_A"]):
        group = group.sort_values("requested_q_A")
        plt.plot(
            group["requested_q_A"],
            group["request_to_pickup_p95"],
            marker="o",
            label=f"{profile}, p={p_a:g}",
        )
    plt.xlabel("Requested AV vehicle-hour share")
    plt.ylabel("P95 request-to-pickup wait (s)")
    plt.legend(ncol=3, fontsize=7)
    save("02_p95_wait_vs_qA.png")

    plt.figure(figsize=(8, 5))
    for (profile, p_a), group in main.groupby(["profile_id", "target_p_A"]):
        group = group.sort_values("requested_q_A")
        plt.plot(
            group["requested_q_A"],
            group["AV_assignment_share"],
            marker="o",
            label=f"{profile}, p={p_a:g}",
        )
    plt.xlabel("Requested AV vehicle-hour share")
    plt.ylabel("AV assignment share")
    plt.legend(ncol=3, fontsize=7)
    save("03_av_assignment_share_vs_qA.png")

    family_means = []
    for profile, group in main.groupby("profile_id"):
        for family in FAMILIES:
            family_means.append(
                {
                    "profile": profile,
                    "family": family,
                    "positive_share": group[
                        f"positive_assigned_exposure_share_{family}"
                    ].mean(),
                }
            )
    family = pd.DataFrame(family_means).pivot(
        index="profile", columns="family", values="positive_share"
    )
    family.reindex(["C", "M", "A"])[list(FAMILIES)].plot.bar(figsize=(7, 5))
    plt.ylabel("Mean positive assigned-exposure share")
    plt.xlabel("Capability profile")
    save("04_family_activity_by_capability.png")

    odd = odd.sort_values("gamma_policy")
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].bar(odd["gamma_policy"], odd["service_rate"])
    axes[0].set_ylabel("Service rate")
    for family in FAMILIES:
        axes[1].plot(
            odd["gamma_policy"],
            odd[f"final_cumulative_mean_exposure_{family}"],
            marker="o",
            label=family,
        )
    axes[1].set_ylabel("Final cumulative mean exposure")
    axes[1].legend()
    save("05_odd_policy_comparison.png")

    plt.figure(figsize=(7, 5))
    for eta, group in cost.groupby("eta_cost_av_to_hv"):
        group = group.sort_values("pickup_cost_epsilon")
        plt.plot(
            group["relative_pickup_ETA_degradation_vs_epsilon_0_reference"],
            group["normalized_operating_cost_per_matched_order"],
            marker="o",
            label=f"eta={eta:g}",
        )
    plt.xlabel("Relative pickup-ETA degradation vs epsilon=0")
    plt.ylabel("Normalized operating cost per matched order")
    plt.legend()
    save("06_cost_vs_pickup_degradation.png")
    return paths


def _output_size_bytes(root: Path, output_root: Path) -> int:
    return sum(path.stat().st_size for path in output_root.rglob("*") if path.is_file())


def _execution_timing(
    root: Path, registry: list[dict[str, Any]], config: dict[str, Any]
) -> dict[str, Any]:
    starts: list[pd.Timestamp] = []
    ends: list[pd.Timestamp] = []
    for row in registry:
        if row["reuse_source_scenario_id"]:
            continue
        status = _read_json(
            scenario_dir(root, row["scenario_id"], config) / "run_status.json"
        )
        starts.append(pd.Timestamp(status["start_time"]))
        ends.append(pd.Timestamp(status["end_time"]))
    return {
        "first_start": min(starts).isoformat(),
        "last_end": max(ends).isoformat(),
        "total_wall_clock_s": float((max(ends) - min(starts)).total_seconds()),
    }


def _write_report(
    root: Path,
    unique: list[dict[str, Any]],
    resolved: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    execution: dict[str, Any],
) -> None:
    main = [row for row in resolved if row["experiment_block"] == "MAIN_STRUCTURAL"]
    bench = [row for row in resolved if row["experiment_block"] == "BENCHMARK"]
    odd = [row for row in resolved if row["experiment_block"] == "ODD_POLICY"]
    cost = [row for row in resolved if row["experiment_block"] == "COST_ROBUSTNESS"]
    service = [float(row["service_rate"]) for row in main]
    waits = [float(row["request_to_pickup_p95"]) for row in main]
    av_share = [float(row["AV_assignment_share"]) for row in main]
    family_lines = []
    for profile in ("C", "M", "A"):
        group = [row for row in main if row["profile_id"] == profile]
        values = []
        for family in FAMILIES:
            mean = np.mean(
                [
                    float(row[f"positive_assigned_exposure_share_{family}"])
                    for row in group
                ]
            )
            values.append(f"{family}={mean:.4f}")
        family_lines.append(f"- {profile}: " + ", ".join(values) + ".")
    odd_lines = [
        f"- {row['gamma_policy']}: service={row['service_rate']:.4f}, P95={row['request_to_pickup_p95']:.1f}s, AV share={row['AV_assignment_share']:.4f}."
        for row in sorted(odd, key=lambda item: item["gamma_policy"])
    ]
    cost_changes = [
        float(row["relative_pickup_ETA_degradation_vs_epsilon_0_reference"])
        for row in cost
    ]
    cost_values = [
        float(row["normalized_operating_cost_per_matched_order"]) for row in cost
    ]
    lines = [
        "# Stage4 Final Experiment Execution",
        "",
        "Recommendation: `GO_STAGE4_RESULT_ANALYSIS`"
        if not failures and len(unique) == 41
        else "Recommendation: `REVISE_FINAL_EXPERIMENT_EXECUTION`",
        "",
        "## Execution base",
        "",
        f"- Execution commit: `{execution['execution_commit']}`.",
        f"- S5B registry commit: `{execution['registry_commit']}`.",
        f"- FleetPy commit: `{execution['fleetpy_commit']}`.",
        f"- Horizon: `{execution['horizon_start']}` to `{execution['horizon_end']}` (right-open), 30,000 requests per profile.",
        "",
        "## Batch completion",
        "",
        f"- Unique completed/expected: {len(unique)}/41.",
        f"- Reuse rows resolved: {sum(bool(row.get('reuse_source_scenario_id')) for row in resolved)}.",
        f"- MAIN/BENCHMARK/ODD/COST resolved: {len(main)}/{len(bench)}/{len(odd)}/{len(cost)}.",
        f"- Failures: {len(failures)}.",
        "",
        "## Resource usage",
        "",
        f"- Total wall-clock: {execution['total_wall_clock_s'] / 3600.0:.3f} h; sum scenario runtime: {execution['sum_scenario_runtime_s'] / 3600.0:.3f} h.",
        f"- Peak observed RSS: {execution['peak_rss_mb']:.1f} MB; output size: {execution['total_output_size_bytes'] / (1024**3):.3f} GiB.",
        f"- Production parallelism: {execution['max_parallel_scenarios']}; sparse CSR/Top-K/cKDTree, CPU-only.",
        "",
        "## Main structural results",
        "",
        f"- Service-rate range: {min(service):.4f}-{max(service):.4f}.",
        f"- P95 request-to-pickup range: {min(waits):.1f}-{max(waits):.1f} s.",
        f"- AV assignment-share range: {min(av_share):.4f}-{max(av_share):.4f}.",
        "",
        "## Benchmarks",
        "",
        *[
            f"- {row['scenario_id']}: service={row['service_rate']:.4f}, P95={row['request_to_pickup_p95']:.1f}s, AV share={row['AV_assignment_share']:.4f}."
            for row in bench
        ],
        "",
        "## ODD policy results",
        "",
        *odd_lines,
        "",
        "## Cost robustness",
        "",
        f"- Pickup-ETA degradation range versus eta-specific epsilon=0: {min(cost_changes):.6f}-{max(cost_changes):.6f}.",
        f"- Normalized operating cost per matched order range: {min(cost_values):.3f}-{max(cost_values):.3f}.",
        "",
        "## Family activity",
        "",
        *family_lines,
        "",
        "## Failures / limitations",
        "",
        f"- Failed or missing unique scenarios: {failures if failures else 'none'}.",
        "- Results are frozen scenario contrasts, not calibrated safety probabilities, monetary estimates, or post-hoc tuned policies.",
        "- Speed remains in every scenario even when empirically inactive.",
        "",
    ]
    path = root / DOC_REL / "stage4_final_experiment_execution_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def aggregate_final_experiments(root: str | Path) -> dict[str, Any]:
    root = Path(root).resolve()
    unique, failures, context = load_completed_unique(root)
    config = context["config"]
    registry = context["registry"]
    resolved = _resolved_rows(unique, registry) if not failures else []
    if failures:
        return {
            "recommendation": "REVISE_FINAL_EXPERIMENT_EXECUTION",
            "completed_unique": len(unique),
            "failures": failures,
        }
    _cost_degradation(resolved)
    _integrity(unique, resolved, registry)
    output = root / AGGREGATE_REL
    output.mkdir(parents=True, exist_ok=True)
    unique_flat = [_scalar_summary(row) for row in unique]
    resolved_flat = [_scalar_summary(row) for row in resolved]
    _write_csv(unique_flat, output / "scenario_summary.csv")
    _write_csv(
        [row for row in resolved_flat if row["experiment_block"] == "MAIN_STRUCTURAL"],
        output / "main_structural_results.csv",
    )
    _write_csv(
        [row for row in resolved_flat if row["experiment_block"] == "BENCHMARK"],
        output / "benchmark_results.csv",
    )
    _write_csv(
        [row for row in resolved_flat if row["experiment_block"] == "ODD_POLICY"],
        output / "odd_policy_results.csv",
    )
    _write_csv(
        [row for row in resolved_flat if row["experiment_block"] == "COST_ROBUSTNESS"],
        output / "cost_robustness_results.csv",
    )
    _write_csv(_family_rows(resolved), output / "family_activity_results.csv")
    _write_csv(_runtime_rows(unique), output / "runtime_resource_summary.csv")
    figures = _make_figures(resolved, output)
    timing = _execution_timing(root, registry, config)
    execution = {
        **timing,
        "execution_commit": unique[0]["execution_commit"],
        "registry_commit": config["registry_commit"],
        "fleetpy_commit": config["fleetpy_commit"],
        "horizon_start": config["horizon_start"],
        "horizon_end": config["horizon_end"],
        "max_parallel_scenarios": config["max_parallel_scenarios"],
        "sum_scenario_runtime_s": float(
            sum(float(row["runtime"]["wall_clock_runtime_s"]) for row in unique)
        ),
        "peak_rss_mb": float(
            max(float(row["runtime"].get("peak_rss_mb") or 0.0) for row in unique)
        ),
        "total_output_size_bytes": _output_size_bytes(
            root, root / config["output_root"]
        ),
        "phase_A_runtimes_s": {
            row["scenario_id"]: float(row["runtime"]["wall_clock_runtime_s"])
            for row in unique
            if row["scenario_id"] in PHASE_A_IDS
        },
        "figures": figures,
    }
    summary = {
        "recommendation": "GO_STAGE4_RESULT_ANALYSIS",
        "completed_unique": len(unique),
        "failed": 0,
        "reused": 1,
        "main_structural_count": 27,
        "benchmark_count": 4,
        "odd_policy_resolved_count": 3,
        "cost_count": 8,
        **execution,
    }
    _atomic_json(summary, output / "execution_summary.json")
    _write_report(root, unique, resolved, failures, execution)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(aggregate_final_experiments(args.root), indent=2))


if __name__ == "__main__":
    main()
