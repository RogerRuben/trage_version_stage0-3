"""Audit effective-capacity gates using only frozen Stage4 logs.

The extractor deliberately does not reconstruct vehicle positions or infer
candidate-type splits.  A complete AV-opportunity funnel is emitted only when
the frozen logs contain like-for-like counts at every gate; the current logs do
not, so unidentifiable stages are labelled explicitly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ANCHORS = (
    "MAIN_Q25_M_P70",
    "MAIN_Q50_M_P70",
    "MAIN_Q75_M_P70",
    "BENCH_AV_M",
)


def _scenario_dir(root: Path, scenario_id: str) -> Path:
    return root / "stage4" / "output" / "final_experiments" / scenario_id


def extract_observed_counts(root: Path) -> pd.DataFrame:
    """Return directly logged counts; mixed-unit columns are named as such."""
    rows: list[dict[str, object]] = []
    for scenario_id in ANCHORS:
        directory = _scenario_dir(root, scenario_id)
        epoch = pd.read_parquet(directory / "epoch_stats.parquet")
        requests = pd.read_parquet(directory / "request_outcomes.parquet")
        assignments = pd.read_parquet(directory / "assignment_log.parquet")
        config = pd.read_json(directory / "scenario_config.json", typ="series")
        scientific = config["scientific_configuration"]
        av_assignments = int(assignments["vehicle_type"].eq("AV").sum())
        total_assignments = int(len(assignments))
        rows.append(
            {
                "scenario_id": scenario_id,
                "requested_q_A": float(scientific["requested_q_A"]),
                "target_p_A": float(scientific["acceptance_probability"]),
                "gamma_policy": str(scientific["gamma_policy"]),
                "request_count": int(len(requests)),
                "passenger_accepting_requests": int(
                    requests["passenger_accepts_av"].fillna(False).sum()
                ),
                "acceptance_rejected_nearby_av_opportunities": int(
                    epoch["av_candidates_pruned_by_acceptance"].sum()
                ),
                "missing_exposure_nearby_av_opportunities": int(
                    epoch["av_candidates_pruned_by_missing_exposure"].sum()
                ),
                "eligible_all_fleet_spatial_pairs": int(
                    epoch["candidate_spatial_pairs"].sum()
                ),
                "eligible_all_fleet_topk_pairs": int(
                    epoch["candidate_topk_pairs"].sum()
                ),
                "all_fleet_patience_arc_exclusions": int(
                    epoch["patience_arc_exclusions"].sum()
                ),
                "all_fleet_valid_or_arcs": int(epoch["valid_or_arcs"].sum()),
                "selected_av_assignments": av_assignments,
                "selected_total_assignments": total_assignments,
                "final_assigned_av_share": (
                    av_assignments / total_assignments if total_assignments else 0.0
                ),
                "enabled_gamma_constraint_count_max": int(
                    epoch["enabled_gamma_constraint_count"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def build_observability_table(observed: pd.DataFrame) -> pd.DataFrame:
    """Describe which funnel stages can be scientifically identified."""
    gate_specs = (
        (
            "nominal_nearby_av_opportunity",
            "AV candidate-opportunity",
            "NOT_IDENTIFIED_FROM_FROZEN_LOGS",
            None,
            "AV positions/counts were not logged for every waiting request before eligibility gates.",
        ),
        (
            "passenger_compatible",
            "AV candidate-opportunity",
            "PARTIALLY_OBSERVED",
            "acceptance_rejected_nearby_av_opportunities",
            "Rejected nearby AV opportunities are logged, but the common entering denominator is absent.",
        ),
        (
            "route_ready",
            "AV candidate-opportunity",
            "NOT_IDENTIFIED_FROM_FROZEN_LOGS",
            None,
            "Non-ready requests exclude AVs before the spatial query and no AV-opportunity counter is stored.",
        ),
        (
            "evidence_complete",
            "AV candidate-opportunity",
            "PARTIALLY_OBSERVED",
            "missing_exposure_nearby_av_opportunities",
            "Nearby AV opportunities removed for missing exposure are logged; the entering denominator is absent.",
        ),
        (
            "pickup_feasible_within_patience",
            "candidate arc",
            "NOT_IDENTIFIED_FOR_AV",
            "all_fleet_patience_arc_exclusions",
            "Patience exclusions combine HV and AV arcs, so an AV-only retained share is not identifiable.",
        ),
        (
            "gamma_feasible",
            "candidate arc",
            "NOT_APPLICABLE_UNCONSTRAINED",
            "enabled_gamma_constraint_count_max",
            "All four anchors use UNCONSTRAINED; candidate-level Gamma attrition is not logged.",
        ),
        (
            "selected_assignment",
            "assignment",
            "OBSERVED",
            "selected_av_assignments",
            "Final AV assignments are directly logged, but cannot be divided by an unobserved nominal-opportunity denominator.",
        ),
    )
    rows: list[dict[str, object]] = []
    for scenario in observed.to_dict("records"):
        for gate, unit, status, column, caveat in gate_specs:
            rows.append(
                {
                    "scenario_id": scenario["scenario_id"],
                    "gate": gate,
                    "unit": unit,
                    "observability": status,
                    "logged_value": scenario[column] if column else pd.NA,
                    "retained_share": pd.NA,
                    "lost_share": pd.NA,
                    "caveat": caveat,
                }
            )
    return pd.DataFrame(rows)


def render_report(observed: pd.DataFrame) -> str:
    view = observed[
        [
            "scenario_id",
            "acceptance_rejected_nearby_av_opportunities",
            "missing_exposure_nearby_av_opportunities",
            "all_fleet_patience_arc_exclusions",
            "selected_av_assignments",
            "selected_total_assignments",
            "final_assigned_av_share",
        ]
    ].copy()
    view["final_assigned_av_share"] = view["final_assigned_av_share"].map(
        lambda value: f"{value:.4f}"
    )
    headers = list(view.columns)
    markdown_rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    markdown_rows.extend(
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in view.itertuples(index=False, name=None)
    )
    markdown_table = "\n".join(markdown_rows)
    return "\n".join(
        [
            "# Effective-capacity gate observability audit",
            "",
            "## Status",
            "",
            "`STOPPED_AT_TASKBOOK_GATE_D4`: a complete candidate-opportunity funnel is not identifiable from the frozen logs. No missing gate is inferred, and no new simulation was started.",
            "",
            "## Directly logged evidence",
            "",
            markdown_table,
            "",
            "The acceptance and missing-exposure columns count nearby **AV candidate-opportunities** removed before routing. The patience column counts **all-fleet candidate arcs** after routing. Final assignments use an **assignment** unit. These quantities therefore cannot be chained into retained/lost percentages.",
            "",
            "## Observable gates",
            "",
            "- Passenger rejection: directly logged as nearby AV opportunities removed, but without the common nominal denominator.",
            "- Missing exposure: directly logged as nearby AV opportunities removed, but without the common entering denominator.",
            "- Selected AV assignments and final assigned-AV share: directly logged.",
            "- Request-level passenger acceptance is logged separately; it is not a substitute for candidate-opportunity retention.",
            "",
            "## Unobservable or non-comparable gates",
            "",
            "- Nominal nearby AV opportunities before route/acceptance/evidence gates.",
            "- Route-ready AV opportunities, because non-ready requests suppress AV spatial queries without a counter.",
            "- AV-only pickup-feasible opportunities, because patience exclusions combine HV and AV arcs.",
            "- Candidate-level Gamma attrition. The four required anchors are UNCONSTRAINED, so Gamma is not an active gate there.",
            "",
            "## Scientific finding",
            "",
            "The frozen logs support selected attrition diagnostics, but not the requested end-to-end effective-capacity funnel. A funnel with retained/lost shares would require new prospective logging and rerunning the anchors; reconstructing it from the current aggregates would invent unavailable states.",
            "",
            "## Decision",
            "",
            "Classification: `QUALIFIES CURRENT STORY`. The existing effective-capacity result remains descriptive, while exact attribution across all proposed gates is not identified. Under the taskbook stop condition, repositioning, Gamma-frontier, and prediction-ablation runs are not authorized in this execution.",
            "",
        ]
    )


def run(root: Path) -> None:
    enhancement = root / "stage4" / "output" / "paper_enhancement"
    output = enhancement / "gate_decomposition"
    docs = root / "stage4" / "docs" / "paper_redesign"
    output.mkdir(parents=True, exist_ok=True)
    observed = extract_observed_counts(root)
    observability = build_observability_table(observed)
    observed.to_csv(output / "observed_gate_diagnostics.csv", index=False)
    observability.to_csv(output / "gate_observability.csv", index=False)
    pd.DataFrame(
        [
            {
                "run_id": "GATE_LOG_AUDIT_4_ANCHORS",
                "workstream": "gate_decomposition",
                "base_scenario": "MAIN_Q25_M_P70|MAIN_Q50_M_P70|MAIN_Q75_M_P70|BENCH_AV_M",
                "variant": "existing_frozen_logs_only",
                "scientific_question": "Where are nominal AV opportunities lost?",
                "changed_component": "none_read_only_analysis",
                "unchanged_components": "all_frozen_scenarios_and_outputs",
                "seed": "not_applicable",
                "status": "STOPPED_UNIDENTIFIABLE",
                "runtime": "not_applicable_read_only",
                "output_path": "stage4/output/paper_enhancement/gate_decomposition",
                "notes": "Complete funnel prohibited because like-for-like denominators and AV-only pickup/Gamma gates are absent.",
            }
        ]
    ).to_csv(enhancement / "experiment_registry.csv", index=False)
    (docs / "gate_decomposition_observability.md").write_text(
        render_report(observed), encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
