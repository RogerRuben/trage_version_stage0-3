from pathlib import Path

import numpy as np
import pandas as pd

from stage4.analysis.result_analysis import (
    ANCHOR_SCENARIOS,
    INFERENTIAL_TOKENS,
    build_interaction_contrasts,
    load_analysis_inputs,
    validate_factorial_grid,
)


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "stage4" / "output" / "result_analysis"
AGGREGATE_DIR = ROOT / "stage4" / "output" / "final_experiments" / "_aggregate"


def test_frozen_factorial_grid_and_descriptive_interaction() -> None:
    main = load_analysis_inputs(ROOT)["main"]
    validate_factorial_grid(main)
    interactions = build_interaction_contrasts(main)
    observed = interactions.query("interaction == 'q_A_x_p_A' and stratum == 'M'").iloc[0]

    def service(q_a: float, p_a: float) -> float:
        row = main[
            np.isclose(main["requested_q_A"], q_a)
            & np.isclose(main["target_p_A"], p_a)
            & (main["profile_id"] == "M")
        ]
        assert len(row) == 1
        return float(row.iloc[0]["service_rate"])

    expected = (service(0.75, 1.0) - service(0.75, 0.4)) - (
        service(0.25, 1.0) - service(0.25, 0.4)
    )
    assert observed["contrast"] == expected
    assert observed["interpretation"] == "descriptive_difference_in_differences"


def test_temporal_anchor_reconciliation_and_test31_only() -> None:
    temporal = pd.read_csv(RESULT_DIR / "temporal_mechanism_15min.csv")
    summary = pd.read_csv(AGGREGATE_DIR / "scenario_summary.csv").set_index("scenario_id")
    assert set(temporal["scenario_id"]) == set(ANCHOR_SCENARIOS)
    timestamps = pd.to_datetime(temporal["time_bin_start"])
    assert timestamps.min().isoformat() == "2016-10-31T00:00:00+08:00"
    assert timestamps.max().isoformat() == "2016-11-01T00:00:00+08:00"

    for scenario_id, rows in temporal.groupby("scenario_id"):
        frozen = summary.loc[scenario_id]
        assert int(rows["new_requests"].sum()) == int(frozen["request_count"])
        assert int(rows["matched_request_cohort"].sum()) == int(frozen["matched"])
        assert int(rows["cohort_patience_expired"].sum()) == int(frozen["patience_expired"])
        assert int(rows["AV_assignments"].sum()) == int(frozen["AV_assignments"])
        assert int(rows["HV_assignments"].sum()) == int(frozen["HV_assignments"])
        first_window_rate = rows["first_window_matches"].sum() / frozen["request_count"]
        assert np.isclose(first_window_rate, frozen["first_window_match_rate"])
        assert not rows["availability_split_observed"].any()


def test_odd_policy_tradeoff_uses_unconstrained_reference() -> None:
    odd = pd.read_csv(RESULT_DIR / "odd_policy_tradeoff.csv")
    policies = odd[odd["row_type"] == "policy"].set_index("policy_or_contrast")
    assert set(policies.index) == {"STRICT", "REFERENCE", "UNCONSTRAINED"}
    unconstrained = policies.loc["UNCONSTRAINED"]
    reference = policies.loc["REFERENCE"]
    assert unconstrained["service_loss_vs_unconstrained"] == 0.0
    assert np.isclose(
        reference["service_loss_vs_unconstrained"],
        unconstrained["service_rate"] - reference["service_rate"],
    )
    expected_static = 1.0 - (
        reference["final_exposure_static"] / unconstrained["final_exposure_static"]
    )
    assert np.isclose(reference["exposure_reduction_static"], expected_static)
    assert pd.isna(reference["exposure_reduction_speed"])


def test_cost_pairs_are_within_eta_only() -> None:
    cost = pd.read_csv(RESULT_DIR / "cost_pairwise_robustness.csv")
    assert set(cost["eta_cost_av_to_hv"]) == {0.5, 0.75, 1.0, 1.25}
    assert (cost["epsilon_low"] == 0.0).all()
    assert (cost["epsilon_high"] == 0.05).all()
    expected = cost["cost_per_matched_eps005"] - cost["cost_per_matched_eps0"]
    assert np.allclose(cost["delta_cost_per_matched"], expected)


def test_output_contract_and_no_inferential_fields() -> None:
    csv_names = {
        "factorial_effects.csv",
        "interaction_contrasts.csv",
        "benchmark_comparison.csv",
        "temporal_mechanism_15min.csv",
        "family_activity_summary.csv",
        "odd_policy_tradeoff.csv",
        "cost_pairwise_robustness.csv",
    }
    png_names = {f"fig0{i}_{suffix}.png" for i, suffix in enumerate(
        (
            "service_rate_factorial",
            "av_share_factorial",
            "acceptance_gain",
            "temporal_mechanism",
            "family_activity",
            "odd_tradeoff",
            "cost_robustness",
        ),
        start=1,
    )}
    assert csv_names <= {path.name for path in RESULT_DIR.glob("*.csv")}
    assert png_names <= {path.name for path in RESULT_DIR.glob("*.png")}
    for name in csv_names:
        columns = {column.lower() for column in pd.read_csv(RESULT_DIR / name, nrows=1)}
        assert not any(token in column for token in INFERENTIAL_TOKENS for column in columns)
    report = (
        ROOT
        / "stage4"
        / "docs"
        / "result_analysis"
        / "stage4_result_analysis_summary.md"
    ).read_text(encoding="utf-8")
    assert "Recommendation: `GO_PAPER_RESULTS_DRAFT`" in report
