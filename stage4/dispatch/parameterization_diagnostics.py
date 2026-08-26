"""Cheap Stage4-S5A exposure and vehicle-hour parameterization diagnostics."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from stage4.fleetpy_adapter.upstream import FleetPyCompatibilityError

from .fleet_normalization import FLEET_REL, SCALING_REL, _priority

STAGE3_REL = Path(
    "stage3/output/odd_tod/final/test31_stage3_to_stage4_interface.parquet"
)
ASSIGNMENT_REL = Path(
    "stage4/output/odd_aware_decision_kernel/canonical_assignment_log.parquet"
)
EXPOSURE_STATE_REL = Path(
    "stage4/output/odd_aware_decision_kernel/canonical_exposure_state.parquet"
)
OUTPUT_REL = Path("stage4/output/parameterization_diagnostics")
DOC_REL = Path("stage4/docs/parameterization_diagnostics")
FAMILIES = ("static", "dynamic", "speed")
Q_LEVELS = (0.00, 0.10, 0.25, 0.50, 0.75, 1.00)
FROZEN_FLEET_SEED = 20260824
HV_TOLERANCE_PCT = 2.0


def _require_inputs(root: Path) -> None:
    required = (
        STAGE3_REL,
        SCALING_REL,
        FLEET_REL,
        ASSIGNMENT_REL,
        EXPOSURE_STATE_REL,
    )
    missing = [str(path) for path in required if not (root / path).is_file()]
    if missing:
        raise FileNotFoundError(f"canonical S5A input missing: {missing}")


def exposure_values(rho: pd.Series) -> pd.Series:
    """Preserve valid zero mass while applying max(rho - 1, 0)."""
    numeric = pd.to_numeric(rho, errors="coerce")
    return (numeric - 1.0).clip(lower=0.0)


def load_dispatch_ready_exposures(root: str | Path) -> pd.DataFrame:
    root = Path(root).resolve()
    columns = [
        "order_id",
        "profile_id",
        "hard_state",
        "evidence_complete",
        "rho_static",
        "rho_dynamic",
        "rho_speed",
    ]
    frame = pd.read_parquet(root / STAGE3_REL, columns=columns)
    frame = frame.loc[frame["profile_id"].astype(str).eq("M")].copy()
    if frame["order_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError("profile M order_id is not unique")
    rho_columns = [f"rho_{family}" for family in FAMILIES]
    numeric = frame[rho_columns].apply(pd.to_numeric, errors="coerce")
    ready = (
        frame["hard_state"].astype(str).eq("FEASIBLE")
        & frame["evidence_complete"].astype(bool)
        & np.isfinite(numeric).all(axis=1)
    )
    frame = frame.loc[ready, ["order_id", *rho_columns]].copy()
    for family in FAMILIES:
        frame[f"e_{family}"] = exposure_values(frame[f"rho_{family}"])
    return frame.sort_values("order_id", kind="mergesort").reset_index(drop=True)


def distribution_summaries(
    exposure: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    full_rows: list[dict[str, Any]] = []
    positive_rows: list[dict[str, Any]] = []
    for family in FAMILIES:
        values = pd.to_numeric(exposure[f"e_{family}"], errors="coerce").dropna()
        positive = values.loc[values > 0.0]
        full_rows.append(
            {
                "family": family,
                "N": len(values),
                "zero_share": float(values.eq(0.0).mean()),
                "mean": float(values.mean()),
                "std": float(values.std()),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "p50": float(values.quantile(0.50)),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "max": float(values.max()),
            }
        )
        row: dict[str, Any] = {
            "family": family,
            "N_positive": len(positive),
            "positive_share": float(len(positive) / len(values))
            if len(values)
            else 0.0,
        }
        for label, value in (
            ("positive_mean", positive.mean()),
            ("positive_p25", positive.quantile(0.25)),
            ("positive_p50", positive.quantile(0.50)),
            ("positive_p75", positive.quantile(0.75)),
            ("positive_p90", positive.quantile(0.90)),
            ("positive_p95", positive.quantile(0.95)),
            ("positive_p99", positive.quantile(0.99)),
            ("positive_max", positive.max()),
        ):
            row[label] = float(value) if len(positive) else None
        positive_rows.append(row)
    return pd.DataFrame(full_rows), pd.DataFrame(positive_rows)


def build_neutral_exposure_path(
    root: str | Path, exposure: pd.DataFrame
) -> pd.DataFrame:
    root = Path(root).resolve()
    columns = [
        "order_id",
        "assignment_time",
        "vehicle_type",
        "exposure_static",
        "exposure_dynamic",
        "exposure_speed",
    ]
    assignment = pd.read_parquet(root / ASSIGNMENT_REL, columns=columns)
    assignment = assignment.loc[assignment["vehicle_type"].astype(str).eq("AV")].copy()
    if assignment.empty or assignment["order_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError(
            "canonical AV assignment trajectory is empty or ambiguous"
        )
    lookup = exposure.set_index(exposure["order_id"].astype(str))
    order_ids = assignment["order_id"].astype(str)
    if not order_ids.isin(lookup.index).all():
        raise FleetPyCompatibilityError(
            "canonical AV assignment cannot join to dispatch-ready exposure"
        )
    for family in FAMILIES:
        logged = pd.to_numeric(assignment[f"exposure_{family}"], errors="coerce")
        derived = lookup.loc[order_ids, f"e_{family}"].to_numpy(dtype=float)
        if not np.isfinite(logged).all() or not np.allclose(
            logged.to_numpy(dtype=float), derived, rtol=0.0, atol=1e-12
        ):
            raise FleetPyCompatibilityError(
                f"canonical {family} exposure disagrees with Stage3"
            )
        assignment[f"e_{family}"] = derived
    assignment["assignment_time"] = pd.to_datetime(
        assignment["assignment_time"], utc=True
    ).dt.tz_convert("Asia/Shanghai")
    assignment = assignment.sort_values(
        ["assignment_time", "order_id"], kind="mergesort"
    ).reset_index(drop=True)
    assignment["av_assignment_rank"] = np.arange(1, len(assignment) + 1)
    for family in FAMILIES:
        assignment[f"cumulative_{family}_excess"] = assignment[f"e_{family}"].cumsum()
        assignment[f"cumulative_mean_{family}_excess"] = (
            assignment[f"cumulative_{family}_excess"] / assignment["av_assignment_rank"]
        )
    state = pd.read_parquet(root / EXPOSURE_STATE_REL)
    if state.empty or int(state.iloc[-1]["cumulative_av_assignments"]) != len(
        assignment
    ):
        raise FleetPyCompatibilityError(
            "canonical aggregate exposure state count disagrees with assignments"
        )
    for family in FAMILIES:
        expected = float(assignment.iloc[-1][f"cumulative_{family}_excess"])
        observed = float(state.iloc[-1][f"cumulative_{family}_excess"])
        if not np.isclose(expected, observed, rtol=0.0, atol=1e-10):
            raise FleetPyCompatibilityError(
                f"canonical aggregate {family} exposure disagrees"
            )
    return assignment[
        [
            "order_id",
            "assignment_time",
            "av_assignment_rank",
            *[f"e_{family}" for family in FAMILIES],
            *[f"cumulative_{family}_excess" for family in FAMILIES],
            *[f"cumulative_mean_{family}_excess" for family in FAMILIES],
        ]
    ]


def gamma_reference_regimes(path: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {
        "semantics": "REFERENCE_ENVELOPE_EXPOSURE_BUDGET_NOT_SAFETY_THRESHOLD",
        "av_assignment_count": int(len(path)),
        "families": {},
    }
    for family in FAMILIES:
        values = path[f"cumulative_mean_{family}_excess"].to_numpy(dtype=float)
        maximum_index = int(np.argmax(values))
        mean = float(values[-1])
        maximum = float(values[maximum_index])
        gap = maximum - mean
        result["families"][family] = {
            "ZERO": 0.0,
            "NEUTRAL_FINAL_MEAN": mean,
            "NEUTRAL_PATH_ENVELOPE": maximum,
            "UNCONSTRAINED": None,
            "neutral_path_min_after_first_assignment": float(values.min()),
            "path_max_assignment_rank": int(
                path.iloc[maximum_index]["av_assignment_rank"]
            ),
            "path_max_assignment_time": pd.Timestamp(
                path.iloc[maximum_index]["assignment_time"]
            ).isoformat(),
            "PATH_minus_MEAN": gap,
            "relative_gap": gap / max(mean, 1e-12),
        }
    return result


def fleet_vehicle_hour_scenarios(
    root: str | Path,
    q_levels: Iterable[float] = Q_LEVELS,
    *,
    seed: int = FROZEN_FLEET_SEED,
) -> pd.DataFrame:
    root = Path(root).resolve()
    scaling = pd.read_parquet(root / SCALING_REL, columns=["simulated_active_supply"])
    if len(scaling) != 96:
        raise FleetPyCompatibilityError("frozen fleet scaling must contain 96 bins")
    h_base = 0.25 * float(scaling["simulated_active_supply"].sum())
    template = pd.read_parquet(
        root / FLEET_REL,
        columns=[
            "source_session_id",
            "availability_start_time",
            "availability_end_time",
        ],
    )
    for column in ("availability_start_time", "availability_end_time"):
        template[column] = pd.to_datetime(template[column], utc=True).dt.tz_convert(
            "Asia/Shanghai"
        )
    if template["source_session_id"].astype(str).duplicated().any():
        raise FleetPyCompatibilityError("source_session_id must be unique")
    template["vehicle_hours"] = (
        template["availability_end_time"] - template["availability_start_time"]
    ).dt.total_seconds() / 3600.0
    template["start_bin_15m"] = (
        template["availability_start_time"].dt.hour * 60
        + template["availability_start_time"].dt.minute
    ) // 15
    template["_priority"] = (
        template["source_session_id"]
        .astype(str)
        .map(lambda value: _priority("HV", seed, value))
    )
    total_template_hours = float(template["vehicle_hours"].sum())
    rows: list[dict[str, Any]] = []
    for requested_q in q_levels:
        requested_av_hours = float(requested_q) * h_base
        av_count = int(round(requested_av_hours / 24.0))
        achieved_av_hours = 24.0 * av_count
        achieved_q = achieved_av_hours / h_base
        target_hv_hours = h_base - achieved_av_hours
        selected_indices: list[int] = []
        if target_hv_hours > 0.0:
            fraction = target_hv_hours / total_template_hours
            for _, group in template.groupby("start_bin_15m", sort=True):
                count = int(round(len(group) * fraction))
                ordered = group.sort_values(
                    ["_priority", "source_session_id"], kind="mergesort"
                )
                selected_indices.extend(ordered.head(count).index.tolist())
        selected = template.loc[selected_indices]
        achieved_hv_hours = float(selected["vehicle_hours"].sum())
        hv_error = (
            abs(achieved_hv_hours - target_hv_hours) / target_hv_hours * 100.0
            if target_hv_hours > 0.0
            else None
        )
        rows.append(
            {
                "requested_q_A": float(requested_q),
                "achieved_q_A": achieved_q,
                "H_base": h_base,
                "requested_AV_vehicle_hours": requested_av_hours,
                "achieved_AV_vehicle_hours": achieved_av_hours,
                "AV_vehicle_count": av_count,
                "AV_rounding_residual_hours": achieved_av_hours - requested_av_hours,
                "target_HV_vehicle_hours": target_hv_hours,
                "achieved_HV_vehicle_hours": achieved_hv_hours,
                "HV_vehicle_hour_error_pct": hv_error,
                "HV_template_support_sufficient": (
                    0.0 <= target_hv_hours <= total_template_hours
                ),
                "HV_within_frozen_tolerance": (
                    hv_error is not None and hv_error <= HV_TOLERANCE_PCT
                ),
                "selected_HV_session_count": int(len(selected)),
                "HV_target_nonpositive_due_to_AV_rounding": target_hv_hours <= 0.0,
            }
        )
    return pd.DataFrame(rows)


def _plot_exposure(exposure: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(11, 3.2), constrained_layout=True)
    for axis, family in zip(axes, FAMILIES):
        values = exposure[f"e_{family}"].to_numpy(dtype=float)
        axis.hist(values, bins=30, color="#4472C4", alpha=0.85)
        zero_share = float(np.mean(values == 0.0))
        axis.set_title(f"{family}: zero={zero_share:.1%}")
        axis.set_xlabel("reference-envelope excess")
        axis.set_ylabel("orders")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_path(path: pd.DataFrame, gamma: dict[str, Any], output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    colors = {"static": "#C44E52", "dynamic": "#4C72B0", "speed": "#55A868"}
    rank = path["av_assignment_rank"]
    for family in FAMILIES:
        values = path[f"cumulative_mean_{family}_excess"]
        refs = gamma["families"][family]
        axis.plot(rank, values, label=family, color=colors[family])
        axis.axhline(
            refs["NEUTRAL_FINAL_MEAN"], color=colors[family], linestyle="--", alpha=0.45
        )
        axis.axhline(
            refs["NEUTRAL_PATH_ENVELOPE"],
            color=colors[family],
            linestyle=":",
            alpha=0.45,
        )
    axis.set_xlabel("AV assignment rank")
    axis.set_ylabel("cumulative mean excess")
    axis.legend(ncol=3)
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _plot_fleet(fleet: pd.DataFrame, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(8.5, 3.4), constrained_layout=True)
    axes[0].plot(fleet["requested_q_A"], fleet["achieved_q_A"], marker="o")
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1)
    axes[0].set(xlabel="requested q_A", ylabel="achieved q_A")
    axes[1].bar(
        fleet["requested_q_A"].astype(str), fleet["AV_vehicle_count"], color="#55A868"
    )
    axes[1].set(xlabel="requested q_A", ylabel="physical AV count")
    figure.savefig(output, dpi=170)
    plt.close(figure)


def _format(value: Any) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "N/A"
    return f"{value:.6f}" if isinstance(value, float) else str(value)


def _markdown_table(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append("| " + " | ".join(_format(value) for value in row) + " |")
    return lines


def _write_report(
    root: Path,
    full: pd.DataFrame,
    positive: pd.DataFrame,
    spearman: pd.DataFrame,
    gamma: dict[str, Any],
    fleet: pd.DataFrame,
    runtime_s: float,
) -> None:
    lines = [
        "# Stage4 S5A Parameterization Diagnostics",
        "",
        "Recommendation: `GO_S5B_EXPERIMENTAL_DESIGN`",
        "",
        "## Exposure population",
        "",
        f"Profile-M AV dispatch-ready orders with complete finite evidence: {int(full.iloc[0]['N'])}.",
        "",
        *_markdown_table(
            full, ["family", "N", "zero_share", "mean", "p50", "p90", "p95"]
        ),
        "",
        "## Positive-only exposure",
        "",
        *_markdown_table(
            positive,
            [
                "family",
                "positive_share",
                "positive_mean",
                "positive_p50",
                "positive_p90",
            ],
        ),
        "",
        "## Spearman correlation",
        "",
        *_markdown_table(
            spearman.reset_index().rename(columns={"index": "family"}),
            ["family", *FAMILIES],
        ),
        "",
        "## Gamma reference regimes",
        "",
        "Gamma denotes a cumulative reference-envelope exposure budget, not a safety or risk threshold.",
        "",
        "| family | ZERO | MEAN | PATH | UNCONSTRAINED | PATH-MEAN | relative gap |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for family in FAMILIES:
        row = gamma["families"][family]
        lines.append(
            f"| {family} | 0 | {row['NEUTRAL_FINAL_MEAN']:.6f} | {row['NEUTRAL_PATH_ENVELOPE']:.6f} | null | {row['PATH_minus_MEAN']:.6f} | {row['relative_gap']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Fleet vehicle-hour scenarios",
            "",
            "q_A is active vehicle-hour share. HV session counts are effective service-session units, not a physical HV fleet count.",
            "",
            *_markdown_table(
                fleet,
                [
                    "requested_q_A",
                    "achieved_q_A",
                    "AV_vehicle_count",
                    "target_HV_vehicle_hours",
                    "achieved_HV_vehicle_hours",
                    "HV_vehicle_hour_error_pct",
                    "HV_template_support_sufficient",
                    "selected_HV_session_count",
                ],
            ),
            "",
            "Rows with HV_template_support_sufficient=false at positive targets saturate all 8,435 frozen effective HV sessions; no synthetic supply or optimizer was introduced.",
            "",
            "The q_A=1 target is slightly negative when 24-hour AV count rounding overshoots H_base; no HV sessions are selected and the relative HV error is reported as N/A.",
            "",
            "## Interpretation",
            "",
            "No dispatch scenario, service-rate comparison, Gamma calibration, passenger-preference estimate, or cost-ratio experiment was run. S5A provides diagnostic evidence only.",
            "",
            f"Diagnostic runtime: {runtime_s:.3f}s.",
            "",
        ]
    )
    path = root / DOC_REL / "stage4_s5a_parameterization_summary.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_parameterization_diagnostics(root: str | Path) -> dict[str, Any]:
    started = time.perf_counter()
    root = Path(root).resolve()
    _require_inputs(root)
    output = root / OUTPUT_REL
    output.mkdir(parents=True, exist_ok=True)
    exposure = load_dispatch_ready_exposures(root)
    full, positive = distribution_summaries(exposure)
    spearman = exposure[[f"e_{family}" for family in FAMILIES]].corr(method="spearman")
    spearman.index = FAMILIES
    spearman.columns = FAMILIES
    path = build_neutral_exposure_path(root, exposure)
    gamma = gamma_reference_regimes(path)
    fleet = fleet_vehicle_hour_scenarios(root)
    q25 = fleet.loc[np.isclose(fleet["requested_q_A"], 0.25)].iloc[0]
    if (
        int(q25["AV_vehicle_count"]) != 150
        or float(q25["HV_vehicle_hour_error_pct"]) > HV_TOLERANCE_PCT
    ):
        raise FleetPyCompatibilityError(
            "q_A=0.25 no longer reproduces frozen normalization"
        )
    full.to_csv(output / "exposure_distribution_summary.csv", index=False)
    positive.to_csv(output / "exposure_positive_summary.csv", index=False)
    spearman.to_csv(output / "exposure_spearman.csv", index=True, index_label="family")
    path.to_parquet(output / "neutral_exposure_path.parquet", index=False)
    (output / "gamma_reference_regimes.json").write_text(
        json.dumps(gamma, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fleet.to_csv(output / "fleet_vehicle_hour_scenarios.csv", index=False)
    _plot_exposure(exposure, output / "exposure_distributions.png")
    _plot_path(path, gamma, output / "neutral_cumulative_exposure.png")
    _plot_fleet(fleet, output / "fleet_scenario_accounting.png")
    runtime_s = time.perf_counter() - started
    _write_report(root, full, positive, spearman, gamma, fleet, runtime_s)
    summary = {
        "phase_status": "STAGE4_S5A_PARAMETERIZATION_DIAGNOSTICS_COMPLETE",
        "recommendation": "GO_S5B_EXPERIMENTAL_DESIGN",
        "dispatch_ready_exposure_N": len(exposure),
        "neutral_AV_assignment_N": len(path),
        "H_base": float(fleet.iloc[0]["H_base"]),
        "runtime_s": runtime_s,
        "fleetpy_launched": False,
        "valhalla_launched": False,
        "milp_launched": False,
        "gpu_usage": "NONE_CPU_ONLY",
    }
    (output / "diagnostics_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    print(json.dumps(run_parameterization_diagnostics(args.root), indent=2))


if __name__ == "__main__":
    main()
