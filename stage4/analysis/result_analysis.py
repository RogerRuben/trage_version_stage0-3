"""Paper-oriented descriptive analysis of the frozen Stage4 scenario results.

This module never invokes FleetPy, Valhalla, or a solver.  Aggregate CSVs are
the canonical inputs.  Only five frozen anchor scenarios are read at request
and epoch level, one at a time, for the required 15-minute mechanism analysis.
"""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


Q_LEVELS = (0.25, 0.50, 0.75)
P_LEVELS = (0.40, 0.70, 1.00)
PROFILE_LEVELS = ("C", "M", "A")
FAMILIES = ("static", "dynamic", "speed")
ANCHOR_SCENARIOS = (
    "BENCH_HV",
    "MAIN_Q25_M_P70",
    "MAIN_Q50_M_P70",
    "MAIN_Q75_M_P70",
    "BENCH_AV_M",
)

Q_OUTCOMES = (
    "service_rate",
    "request_to_pickup_p95",
    "AV_assignment_share",
    "first_window_match_rate",
)
PRIMARY_OUTCOMES = (
    "service_rate",
    "request_to_pickup_p95",
    "AV_assignment_share",
)
INFERENTIAL_TOKENS = ("p_value", "pvalue", "confidence_interval", "significance")


def _root(root: str | Path) -> Path:
    return Path(root).resolve()


def _aggregate_dir(root: Path) -> Path:
    return root / "stage4" / "output" / "final_experiments" / "_aggregate"


def _result_dir(root: Path) -> Path:
    return root / "stage4" / "output" / "result_analysis"


def _report_path(root: Path) -> Path:
    return (
        root
        / "stage4"
        / "docs"
        / "result_analysis"
        / "stage4_result_analysis_summary.md"
    )


def load_analysis_inputs(root: str | Path) -> dict[str, pd.DataFrame]:
    root = _root(root)
    directory = _aggregate_dir(root)
    names = {
        "scenario": "scenario_summary.csv",
        "main": "main_structural_results.csv",
        "benchmark": "benchmark_results.csv",
        "odd": "odd_policy_results.csv",
        "cost": "cost_robustness_results.csv",
        "family": "family_activity_results.csv",
    }
    tables = {key: pd.read_csv(directory / name) for key, name in names.items()}
    validate_factorial_grid(tables["main"])
    if len(tables["benchmark"]) != 4:
        raise ValueError("benchmark table must contain exactly four scenarios")
    if set(tables["odd"]["gamma_policy"]) != {
        "STRICT",
        "REFERENCE",
        "UNCONSTRAINED",
    }:
        raise ValueError("ODD table does not contain the three frozen policies")
    if len(tables["cost"]) != 8:
        raise ValueError("cost table must contain exactly eight frozen scenarios")
    return tables


def validate_factorial_grid(main: pd.DataFrame) -> None:
    keys = ["requested_q_A", "profile_id", "target_p_A"]
    if len(main) != 27 or main[keys].duplicated().any():
        raise ValueError("main factorial grid must contain 27 unique scenarios")
    observed = {
        (round(float(q), 2), str(k), round(float(p), 2))
        for q, k, p in main[keys].itertuples(index=False, name=None)
    }
    expected = {(q, k, p) for q in Q_LEVELS for k in PROFILE_LEVELS for p in P_LEVELS}
    if observed != expected:
        raise ValueError(f"factorial grid mismatch: missing={expected-observed}")


def classify_monotonicity(values: Iterable[float], tolerance: float = 1e-12) -> str:
    values = np.asarray(list(values), dtype=float)
    differences = np.diff(values)
    nondecreasing = bool(np.all(differences >= -tolerance))
    nonincreasing = bool(np.all(differences <= tolerance))
    if nondecreasing and np.any(differences > tolerance):
        return "monotone_increasing"
    if nonincreasing and np.any(differences < -tolerance):
        return "monotone_decreasing"
    if nondecreasing and nonincreasing:
        return "constant"
    return "non_monotone"


def _relative_change(low: float, high: float) -> float:
    return float((high - low) / abs(low)) if low != 0 else np.nan


def _slice_row(frame: pd.DataFrame, **filters: Any) -> pd.Series:
    selected = frame
    for column, value in filters.items():
        if isinstance(value, float):
            selected = selected[np.isclose(selected[column].astype(float), value)]
        else:
            selected = selected[selected[column] == value]
    if len(selected) != 1:
        raise ValueError(f"expected one scenario for {filters}, found {len(selected)}")
    return selected.iloc[0]


def _contrast_record(
    *,
    axis: str,
    fixed_profile: str | None,
    fixed_q: float | None,
    fixed_p: float | None,
    outcome: str,
    low_level: Any,
    high_level: Any,
    low_value: float,
    high_value: float,
    response_class: str,
) -> dict[str, Any]:
    return {
        "effect_axis": axis,
        "fixed_profile": fixed_profile,
        "fixed_q_A": fixed_q,
        "fixed_p_A": fixed_p,
        "outcome": outcome,
        "low_level": low_level,
        "high_level": high_level,
        "comparison": f"{low_level}_to_{high_level}",
        "low_value": low_value,
        "high_value": high_value,
        "absolute_change": high_value - low_value,
        "relative_change": _relative_change(low_value, high_value),
        "response_class": response_class,
    }


def build_factorial_effects(main: pd.DataFrame) -> pd.DataFrame:
    validate_factorial_grid(main)
    records: list[dict[str, Any]] = []
    pairs_q = ((0.25, 0.50), (0.50, 0.75), (0.25, 0.75))
    pairs_p = ((0.40, 0.70), (0.70, 1.00), (0.40, 1.00))
    pairs_k = (("C", "M"), ("M", "A"), ("C", "A"))

    for profile in PROFILE_LEVELS:
        for p_a in P_LEVELS:
            rows = [_slice_row(main, profile_id=profile, target_p_A=p_a, requested_q_A=q) for q in Q_LEVELS]
            for outcome in Q_OUTCOMES:
                response = classify_monotonicity(row[outcome] for row in rows)
                by_q = {q: float(row[outcome]) for q, row in zip(Q_LEVELS, rows)}
                for low, high in pairs_q:
                    records.append(
                        _contrast_record(
                            axis="q_A",
                            fixed_profile=profile,
                            fixed_q=None,
                            fixed_p=p_a,
                            outcome=outcome,
                            low_level=low,
                            high_level=high,
                            low_value=by_q[low],
                            high_value=by_q[high],
                            response_class=response,
                        )
                    )

    for q_a in Q_LEVELS:
        for profile in PROFILE_LEVELS:
            rows = [_slice_row(main, requested_q_A=q_a, profile_id=profile, target_p_A=p) for p in P_LEVELS]
            for outcome in PRIMARY_OUTCOMES:
                response = classify_monotonicity(row[outcome] for row in rows)
                by_p = {p: float(row[outcome]) for p, row in zip(P_LEVELS, rows)}
                for low, high in pairs_p:
                    records.append(
                        _contrast_record(
                            axis="p_A",
                            fixed_profile=profile,
                            fixed_q=q_a,
                            fixed_p=None,
                            outcome=outcome,
                            low_level=low,
                            high_level=high,
                            low_value=by_p[low],
                            high_value=by_p[high],
                            response_class=response,
                        )
                    )

    for q_a in Q_LEVELS:
        for p_a in P_LEVELS:
            rows = [_slice_row(main, requested_q_A=q_a, target_p_A=p_a, profile_id=k) for k in PROFILE_LEVELS]
            for outcome in PRIMARY_OUTCOMES:
                response = classify_monotonicity(row[outcome] for row in rows)
                by_k = {k: float(row[outcome]) for k, row in zip(PROFILE_LEVELS, rows)}
                for low, high in pairs_k:
                    records.append(
                        _contrast_record(
                            axis="capability",
                            fixed_profile=None,
                            fixed_q=q_a,
                            fixed_p=p_a,
                            outcome=outcome,
                            low_level=low,
                            high_level=high,
                            low_value=by_k[low],
                            high_value=by_k[high],
                            response_class=response,
                        )
                    )

    for axis, column, levels in (
        ("q_A_marginal_mean", "requested_q_A", Q_LEVELS),
        ("p_A_marginal_mean", "target_p_A", P_LEVELS),
        ("capability_marginal_mean", "profile_id", PROFILE_LEVELS),
    ):
        for level in levels:
            group = main[np.isclose(main[column].astype(float), level)] if isinstance(level, float) else main[main[column] == level]
            for outcome in Q_OUTCOMES:
                value = float(group[outcome].mean())
                records.append(
                    {
                        "effect_axis": axis,
                        "fixed_profile": level if column == "profile_id" else None,
                        "fixed_q_A": level if column == "requested_q_A" else None,
                        "fixed_p_A": level if column == "target_p_A" else None,
                        "outcome": outcome,
                        "low_level": level,
                        "high_level": level,
                        "comparison": "marginal_mean",
                        "low_value": value,
                        "high_value": value,
                        "absolute_change": 0.0,
                        "relative_change": 0.0,
                        "response_class": "not_applicable",
                    }
                )
    return pd.DataFrame.from_records(records)


def build_interaction_contrasts(main: pd.DataFrame) -> pd.DataFrame:
    validate_factorial_grid(main)
    records: list[dict[str, Any]] = []

    def sr(q: float, k: str, p: float) -> float:
        return float(_slice_row(main, requested_q_A=q, profile_id=k, target_p_A=p)["service_rate"])

    for profile in PROFILE_LEVELS:
        corners = {
            "high_high": sr(0.75, profile, 1.0),
            "high_low": sr(0.75, profile, 0.4),
            "low_high": sr(0.25, profile, 1.0),
            "low_low": sr(0.25, profile, 0.4),
        }
        records.append(
            {
                "interaction": "q_A_x_p_A",
                "stratum": profile,
                **corners,
                "contrast": (corners["high_high"] - corners["high_low"])
                - (corners["low_high"] - corners["low_low"]),
                "interpretation": "descriptive_difference_in_differences",
            }
        )
    for p_a in P_LEVELS:
        corners = {
            "high_high": sr(0.75, "A", p_a),
            "high_low": sr(0.75, "C", p_a),
            "low_high": sr(0.25, "A", p_a),
            "low_low": sr(0.25, "C", p_a),
        }
        records.append(
            {
                "interaction": "q_A_x_capability",
                "stratum": p_a,
                **corners,
                "contrast": (corners["high_high"] - corners["high_low"])
                - (corners["low_high"] - corners["low_low"]),
                "interpretation": "descriptive_difference_in_differences",
            }
        )
    for q_a in Q_LEVELS:
        corners = {
            "high_high": sr(q_a, "A", 1.0),
            "high_low": sr(q_a, "C", 1.0),
            "low_high": sr(q_a, "A", 0.4),
            "low_low": sr(q_a, "C", 0.4),
        }
        records.append(
            {
                "interaction": "capability_x_p_A",
                "stratum": q_a,
                **corners,
                "contrast": (corners["high_high"] - corners["high_low"])
                - (corners["low_high"] - corners["low_low"]),
                "interpretation": "descriptive_difference_in_differences",
            }
        )
    return pd.DataFrame.from_records(records)


def build_benchmark_comparison(main: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
    ids = ["BENCH_HV", "BENCH_AV_C", "BENCH_AV_M", "BENCH_AV_A"]
    benchmark_by_id = benchmark.set_index("scenario_id")
    rows: list[tuple[str, pd.Series]] = []
    labels: list[str] = []
    for scenario_id in ids:
        rows.append((scenario_id, benchmark_by_id.loc[scenario_id]))
        labels.append("all_HV_benchmark" if scenario_id == "BENCH_HV" else "all_AV_composition_extreme")
    for q_a in Q_LEVELS:
        row = _slice_row(main, requested_q_A=q_a, profile_id="M", target_p_A=0.70)
        rows.append((str(row["scenario_id"]), row))
        labels.append("mixed_fleet_M_p70")
    columns = [
        "scenario_id",
        "requested_q_A",
        "profile_id",
        "target_p_A",
        "AV_vehicle_count",
        "accepted_order_count",
        "realized_accepted_order_share",
        "matched",
        "patience_expired",
        "service_rate",
        "request_to_pickup_p50",
        "request_to_pickup_p90",
        "request_to_pickup_p95",
        "HV_assignments",
        "AV_assignments",
        "AV_assignment_share",
        "mean_assigned_exposure_static",
        "mean_assigned_exposure_dynamic",
        "mean_assigned_exposure_speed",
    ]
    output = pd.DataFrame(
        [
            {
                column: scenario_id if column == "scenario_id" else row.get(column, np.nan)
                for column in columns
            }
            for scenario_id, row in rows
        ]
    )
    output.insert(1, "comparison_role", labels)
    return output


def build_family_activity_summary(family: pd.DataFrame) -> pd.DataFrame:
    main = family[family["experiment_block"] == "MAIN_STRUCTURAL"].copy()
    records: list[dict[str, Any]] = []
    for profile in PROFILE_LEVELS:
        for name in FAMILIES:
            part = main[(main["profile_id"] == profile) & (main["family"] == name)]
            if len(part) != 9:
                raise ValueError(f"expected nine family rows for {profile}/{name}")
            q_means = part.groupby("requested_q_A")["positive_assigned_exposure_share"].mean()
            p_means = part.groupby("target_p_A")["positive_assigned_exposure_share"].mean()
            records.append(
                {
                    "profile_id": profile,
                    "family": name,
                    "scenario_count": len(part),
                    "positive_share_mean": float(part["positive_assigned_exposure_share"].mean()),
                    "positive_share_min": float(part["positive_assigned_exposure_share"].min()),
                    "positive_share_max": float(part["positive_assigned_exposure_share"].max()),
                    "positive_share_q_marginal_range": float(q_means.max() - q_means.min()),
                    "positive_share_p_marginal_range": float(p_means.max() - p_means.min()),
                    "mean_exposure_mean": float(part["mean_assigned_exposure"].mean()),
                    "mean_exposure_min": float(part["mean_assigned_exposure"].min()),
                    "mean_exposure_max": float(part["mean_assigned_exposure"].max()),
                    "final_exposure_mean": float(part["final_cumulative_mean_exposure"].mean()),
                }
            )
    return pd.DataFrame.from_records(records)


def build_odd_policy_tradeoff(odd: pd.DataFrame) -> pd.DataFrame:
    policy = odd.set_index("gamma_policy")
    unconstrained = policy.loc["UNCONSTRAINED"]
    strict = policy.loc["STRICT"]
    records: list[dict[str, Any]] = []
    for name in ("STRICT", "REFERENCE", "UNCONSTRAINED"):
        row = policy.loc[name]
        service_loss = float(unconstrained["service_rate"] - row["service_rate"])
        record: dict[str, Any] = {
            "row_type": "policy",
            "policy_or_contrast": name,
            "service_rate": float(row["service_rate"]),
            "matched": int(row["matched"]),
            "request_to_pickup_p95": float(row["request_to_pickup_p95"]),
            "AV_assignment_share": float(row["AV_assignment_share"]),
            "service_loss_vs_unconstrained": service_loss,
            "relative_service_loss_vs_unconstrained": service_loss
            / float(unconstrained["service_rate"]),
            "AV_share_change_vs_unconstrained": float(row["AV_assignment_share"] - unconstrained["AV_assignment_share"]),
        }
        for family in FAMILIES:
            value = float(row[f"final_cumulative_mean_exposure_{family}"])
            reference = float(unconstrained[f"final_cumulative_mean_exposure_{family}"])
            record[f"final_exposure_{family}"] = value
            record[f"maximum_exposure_{family}"] = float(row[f"maximum_cumulative_mean_exposure_{family}"])
            record[f"minimum_slack_{family}"] = row[f"minimum_slack_{family}"]
            record[f"mean_slack_{family}"] = row[f"mean_slack_{family}"]
            record[f"binding_epochs_{family}"] = row[f"binding_epoch_count_{family}"]
            record[f"near_binding_epochs_{family}"] = row[f"near_binding_epoch_count_{family}"]
            record[f"exposure_reduction_{family}"] = 1.0 - value / reference if reference > 0 else np.nan
        records.append(record)

    for left, right in (
        ("REFERENCE", "UNCONSTRAINED"),
        ("STRICT", "UNCONSTRAINED"),
        ("REFERENCE", "STRICT"),
    ):
        left_row, right_row = policy.loc[left], policy.loc[right]
        record = {
            "row_type": "contrast",
            "policy_or_contrast": f"{left}_minus_{right}",
            "service_rate": float(left_row["service_rate"] - right_row["service_rate"]),
            "matched": int(left_row["matched"] - right_row["matched"]),
            "request_to_pickup_p95": float(left_row["request_to_pickup_p95"] - right_row["request_to_pickup_p95"]),
            "AV_assignment_share": float(left_row["AV_assignment_share"] - right_row["AV_assignment_share"]),
            "service_loss_vs_unconstrained": np.nan,
            "AV_share_change_vs_unconstrained": np.nan,
        }
        for family in FAMILIES:
            record[f"final_exposure_{family}"] = float(
                left_row[f"final_cumulative_mean_exposure_{family}"]
                - right_row[f"final_cumulative_mean_exposure_{family}"]
            )
        records.append(record)
    return pd.DataFrame.from_records(records)


def build_cost_pairwise_robustness(cost: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for eta, group in cost.groupby("eta_cost_av_to_hv", sort=True):
        by_epsilon = group.set_index("pickup_cost_epsilon")
        if not {0.0, 0.05}.issubset(set(float(x) for x in by_epsilon.index)):
            raise ValueError(f"eta={eta} does not contain epsilon 0 and .05")
        base = by_epsilon.loc[0.0]
        relaxed = by_epsilon.loc[0.05]
        cost0 = float(base["normalized_operating_cost_per_matched_order"])
        cost5 = float(relaxed["normalized_operating_cost_per_matched_order"])
        records.append(
            {
                "eta_cost_av_to_hv": float(eta),
                "epsilon_low": 0.0,
                "epsilon_high": 0.05,
                "cost_per_matched_eps0": cost0,
                "cost_per_matched_eps005": cost5,
                "delta_cost_per_matched": cost5 - cost0,
                "percent_cost_change": (cost5 - cost0) / cost0 if cost0 else np.nan,
                "service_rate_eps0": float(base["service_rate"]),
                "service_rate_eps005": float(relaxed["service_rate"]),
                "delta_service_rate": float(relaxed["service_rate"] - base["service_rate"]),
                "p95_wait_eps0": float(base["request_to_pickup_p95"]),
                "p95_wait_eps005": float(relaxed["request_to_pickup_p95"]),
                "delta_p95_wait_s": float(relaxed["request_to_pickup_p95"] - base["request_to_pickup_p95"]),
                "pickup_objective_eps0": float(base["pickup_ETA_objective_value"]),
                "pickup_objective_eps005": float(relaxed["pickup_ETA_objective_value"]),
                "delta_pickup_objective": float(relaxed["pickup_ETA_objective_value"] - base["pickup_ETA_objective_value"]),
                "AV_share_eps0": float(base["AV_assignment_share"]),
                "AV_share_eps005": float(relaxed["AV_assignment_share"]),
                "delta_AV_assignment_share": float(relaxed["AV_assignment_share"] - base["AV_assignment_share"]),
            }
        )
    if len(records) != 4:
        raise ValueError("cost robustness must yield four within-eta pairs")
    return pd.DataFrame.from_records(records)


def _count_by_bin(frame: pd.DataFrame, mask: pd.Series, timestamp: str, bins: pd.DatetimeIndex) -> pd.Series:
    selected = frame.loc[mask & frame[timestamp].notna(), timestamp]
    return selected.dt.floor("15min").value_counts().reindex(bins, fill_value=0).sort_index()


def aggregate_temporal_scenario(
    root: str | Path,
    scenario_id: str,
    summary: pd.Series | dict[str, Any],
) -> pd.DataFrame:
    root = _root(root)
    directory = root / "stage4" / "output" / "final_experiments" / scenario_id
    outcomes = pd.read_parquet(
        directory / "request_outcomes.parquet",
        columns=[
            "request_time",
            "pickup_deadline",
            "attempt_count",
            "matched",
            "assignment_time",
            "patience_expired",
            "vehicle_type",
            "passenger_accepts_av",
        ],
    )
    start = pd.Timestamp(summary["horizon_start"])
    end = pd.Timestamp(summary["horizon_end"])
    bins = pd.date_range(start=start.floor("15min"), end=end, freq="15min", inclusive="left")
    request_mask = pd.Series(True, index=outcomes.index)
    matched_mask = outcomes["matched"].astype(bool)
    expired_mask = outcomes["patience_expired"].astype(bool)
    first_mask = matched_mask & (outcomes["attempt_count"] == 1)
    av_mask = matched_mask & (outcomes["vehicle_type"] == "AV")
    hv_mask = matched_mask & (outcomes["vehicle_type"] == "HV")
    result = pd.DataFrame(index=bins)
    result["new_requests"] = _count_by_bin(outcomes, request_mask, "request_time", bins)
    result["matched_request_cohort"] = _count_by_bin(outcomes, matched_mask, "request_time", bins)
    result["cohort_patience_expired"] = _count_by_bin(outcomes, expired_mask, "request_time", bins)
    result["patience_expirations"] = _count_by_bin(outcomes, expired_mask, "pickup_deadline", bins)
    result["first_window_matches"] = _count_by_bin(outcomes, first_mask, "request_time", bins)
    result["AV_assignments"] = _count_by_bin(outcomes, av_mask, "assignment_time", bins)
    result["HV_assignments"] = _count_by_bin(outcomes, hv_mask, "assignment_time", bins)
    accepted = outcomes.assign(_bin=outcomes["request_time"].dt.floor("15min")).groupby("_bin")["passenger_accepts_av"].agg(["sum", "mean"])
    result["accepted_requests"] = accepted["sum"].reindex(bins, fill_value=0)
    result["accepted_request_share"] = accepted["mean"].reindex(bins)
    del outcomes
    gc.collect()

    epoch = pd.read_parquet(
        directory / "epoch_stats.parquet",
        columns=["timestamp", "waiting_orders", "available_vehicles", "matched"],
    )
    epoch["_bin"] = epoch["timestamp"].dt.floor("15min")
    epoch_agg = epoch.groupby("_bin").agg(
        available_vehicles_mean=("available_vehicles", "mean"),
        waiting_queue_mean=("waiting_orders", "mean"),
        waiting_queue_max=("waiting_orders", "max"),
        matches_executed=("matched", "sum"),
        epoch_count=("timestamp", "size"),
    )
    result = result.join(epoch_agg.reindex(bins))
    del epoch, epoch_agg
    gc.collect()

    assignments = result["AV_assignments"] + result["HV_assignments"]
    result["time_bin_service_rate"] = result["matched_request_cohort"].div(result["new_requests"].replace(0, np.nan))
    result["cohort_expiration_rate"] = result["cohort_patience_expired"].div(result["new_requests"].replace(0, np.nan))
    result["first_window_match_rate"] = result["first_window_matches"].div(result["new_requests"].replace(0, np.nan))
    result["available_vehicles_per_request"] = result["available_vehicles_mean"].div(result["new_requests"].replace(0, np.nan))
    result["queue_pressure"] = result["waiting_queue_mean"].div(result["available_vehicles_mean"].replace(0, np.nan))
    result["AV_assignment_share"] = result["AV_assignments"].div(assignments.replace(0, np.nan))
    result.insert(0, "time_bin_start", result.index)
    result.insert(0, "scenario_id", scenario_id)
    result["nominal_AV_count"] = int(summary["AV_vehicle_count"])
    result["scenario_service_rate"] = float(summary["service_rate"])
    result["scenario_accepted_order_share"] = float(summary["realized_accepted_order_share"])
    result["availability_split_observed"] = False
    result.reset_index(drop=True, inplace=True)

    reconciliations = {
        "requests": (int(result["new_requests"].sum()), int(summary["request_count"])),
        "matched": (int(result["matched_request_cohort"].sum()), int(summary["matched"])),
        "expired": (int(result["cohort_patience_expired"].sum()), int(summary["patience_expired"])),
        "hv": (int(result["HV_assignments"].sum()), int(summary["HV_assignments"])),
        "av": (int(result["AV_assignments"].sum()), int(summary["AV_assignments"])),
        "first_window": (
            int(result["first_window_matches"].sum()),
            int(round(float(summary["first_window_match_rate"]) * int(summary["request_count"]))),
        ),
    }
    failures = {key: pair for key, pair in reconciliations.items() if pair[0] != pair[1]}
    if failures:
        raise ValueError(f"temporal aggregation mismatch for {scenario_id}: {failures}")
    return result


def build_temporal_mechanism(root: str | Path, scenario: pd.DataFrame) -> pd.DataFrame:
    by_id = scenario.set_index("scenario_id")
    frames: list[pd.DataFrame] = []
    for scenario_id in ANCHOR_SCENARIOS:
        if scenario_id not in by_id.index:
            raise ValueError(f"anchor scenario missing: {scenario_id}")
        frames.append(aggregate_temporal_scenario(root, scenario_id, by_id.loc[scenario_id]))
        gc.collect()
    return pd.concat(frames, ignore_index=True)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _main_plot(main: pd.DataFrame, outcome: str, ylabel: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.1), sharey=True)
    colors = {0.4: "#3B82F6", 0.7: "#F59E0B", 1.0: "#10B981"}
    for axis, profile in zip(axes, PROFILE_LEVELS):
        part = main[main["profile_id"] == profile]
        for p_a in P_LEVELS:
            line = part[np.isclose(part["target_p_A"], p_a)].sort_values("requested_q_A")
            axis.plot(line["requested_q_A"], line[outcome], marker="o", linewidth=2, color=colors[p_a], label=f"p_A={p_a:.1f}")
        axis.set_title(f"Capability {profile}")
        axis.set_xlabel("Active AV vehicle-hour share q_A")
        axis.grid(alpha=0.25)
        axis.set_xticks(Q_LEVELS)
    axes[0].set_ylabel(ylabel)
    axes[-1].legend(frameon=False)
    _save_figure(fig, path)


def create_figures(
    main: pd.DataFrame,
    interactions: pd.DataFrame,
    temporal: pd.DataFrame,
    family: pd.DataFrame,
    odd: pd.DataFrame,
    cost: pd.DataFrame,
    output: Path,
) -> list[Path]:
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / f"fig{i:02d}_{name}.png" for i, name in enumerate(
        (
            "service_rate_factorial",
            "av_share_factorial",
            "acceptance_gain",
            "temporal_mechanism",
            "family_activity",
            "odd_tradeoff",
            "cost_robustness",
        ), start=1
    )]
    _main_plot(main, "service_rate", "Service rate", paths[0])
    _main_plot(main, "AV_assignment_share", "AV assignment share", paths[1])

    fig, axis = plt.subplots(figsize=(7.4, 4.8))
    for profile, color in zip(PROFILE_LEVELS, ("#2563EB", "#D97706", "#059669")):
        gains = []
        for q_a in Q_LEVELS:
            high = _slice_row(main, requested_q_A=q_a, profile_id=profile, target_p_A=1.0)["service_rate"]
            low = _slice_row(main, requested_q_A=q_a, profile_id=profile, target_p_A=0.4)["service_rate"]
            gains.append(float(high - low))
        axis.plot(Q_LEVELS, gains, marker="o", linewidth=2, label=profile, color=color)
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set(xlabel="Active AV vehicle-hour share q_A", ylabel="Acceptance gain in service rate", xticks=Q_LEVELS)
    axis.grid(alpha=0.25)
    axis.legend(title="Capability", frameon=False)
    _save_figure(fig, paths[2])

    time = temporal.copy()
    time["hour"] = pd.to_datetime(time["time_bin_start"]).dt.floor("h")
    hourly = time.groupby(["scenario_id", "hour"], as_index=False).agg(
        new_requests=("new_requests", "sum"),
        matched=("matched_request_cohort", "sum"),
        available=("available_vehicles_mean", "mean"),
        queue=("waiting_queue_mean", "mean"),
    )
    hourly["service_rate"] = hourly["matched"].div(hourly["new_requests"].replace(0, np.nan))
    hourly["available_per_request"] = hourly["available"].div(hourly["new_requests"].replace(0, np.nan))
    hourly["queue_pressure"] = hourly["queue"].div(hourly["available"].replace(0, np.nan))
    fig, axes = plt.subplots(3, 1, figsize=(11.5, 9.0), sharex=True)
    for scenario_id in ANCHOR_SCENARIOS:
        part = hourly[hourly["scenario_id"] == scenario_id]
        x = pd.to_datetime(part["hour"]).dt.hour + pd.to_datetime(part["hour"]).dt.day.sub(31).mul(24)
        axes[0].plot(x, part["service_rate"], label=scenario_id, linewidth=1.6)
        axes[1].plot(x, part["queue_pressure"], linewidth=1.6)
        axes[2].plot(x, part["available_per_request"], linewidth=1.6)
    axes[0].set_ylabel("Cohort service rate")
    axes[1].set_ylabel("Queue / available")
    axes[2].set_ylabel("Available / request")
    axes[2].set_xlabel("Hour since Test31 start")
    axes[0].legend(ncol=2, fontsize=8, frameon=False)
    for axis in axes:
        axis.grid(alpha=0.22)
    _save_figure(fig, paths[3])

    pivot = family.pivot(index="profile_id", columns="family", values="positive_share_mean").reindex(PROFILE_LEVELS)
    fig, axis = plt.subplots(figsize=(7.8, 4.8))
    pivot.plot(kind="bar", ax=axis, color=["#4C78A8", "#F58518", "#54A24B"])
    axis.set(xlabel="Assumed capability profile", ylabel="Mean positive assigned exposure share")
    axis.tick_params(axis="x", rotation=0)
    axis.legend(title="Exposure family", frameon=False)
    axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, paths[4])

    policy = odd[odd["row_type"] == "policy"]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0), sharey=True)
    for axis, name in zip(axes, FAMILIES):
        axis.scatter(policy[f"final_exposure_{name}"], policy["service_rate"], s=65)
        for row in policy.itertuples(index=False):
            axis.annotate(row.policy_or_contrast, (getattr(row, f"final_exposure_{name}"), row.service_rate), xytext=(4, 4), textcoords="offset points", fontsize=8)
        axis.set_xlabel(f"Final {name} exposure")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Service rate")
    _save_figure(fig, paths[5])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.0))
    axes[0].bar(cost["eta_cost_av_to_hv"].astype(str), 100 * cost["percent_cost_change"], color="#4C78A8")
    axes[1].bar(cost["eta_cost_av_to_hv"].astype(str), cost["delta_AV_assignment_share"], color="#F58518")
    axes[2].bar(cost["eta_cost_av_to_hv"].astype(str), cost["delta_p95_wait_s"], color="#54A24B")
    axes[0].set_ylabel("Within-eta cost change (%)")
    axes[1].set_ylabel("Change in AV assignment share")
    axes[2].set_ylabel("Change in P95 wait (s)")
    for axis in axes:
        axis.set_xlabel("Normalized AV/HV cost ratio eta")
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
    _save_figure(fig, paths[6])
    return paths


def _fmt(value: float, digits: int = 4) -> str:
    return f"{float(value):.{digits}f}"


def _scenario(main: pd.DataFrame, q: float, k: str, p: float) -> pd.Series:
    return _slice_row(main, requested_q_A=q, profile_id=k, target_p_A=p)


def build_report(
    root: Path,
    tables: dict[str, pd.DataFrame],
    effects: pd.DataFrame,
    interactions: pd.DataFrame,
    benchmarks: pd.DataFrame,
    temporal: pd.DataFrame,
    family: pd.DataFrame,
    odd: pd.DataFrame,
    cost: pd.DataFrame,
) -> str:
    main = tables["main"]
    q_means = main.groupby("requested_q_A")["service_rate"].mean()
    p_gain = {
        (q, k): float(_scenario(main, q, k, 1.0)["service_rate"] - _scenario(main, q, k, 0.4)["service_rate"])
        for q in Q_LEVELS for k in PROFILE_LEVELS
    }
    cap_gain = {
        (q, p): float(_scenario(main, q, "A", p)["service_rate"] - _scenario(main, q, "C", p)["service_rate"])
        for q in Q_LEVELS for p in P_LEVELS
    }
    anchor = benchmarks.set_index("scenario_id")
    hv = anchor.loc["BENCH_HV"]
    avm = anchor.loc["BENCH_AV_M"]
    m25, m50, m75 = (_scenario(main, q, "M", 0.7) for q in Q_LEVELS)
    hourly = temporal.copy()
    peak_mask = pd.to_datetime(hourly["time_bin_start"]).dt.hour.isin(range(7, 10)) | pd.to_datetime(hourly["time_bin_start"]).dt.hour.isin(range(17, 20))
    peak = hourly[peak_mask].groupby("scenario_id").agg(
        requests=("new_requests", "sum"),
        matched=("matched_request_cohort", "sum"),
        queue=("waiting_queue_mean", "mean"),
        available=("available_vehicles_mean", "mean"),
    )
    peak["service"] = peak["matched"] / peak["requests"]
    peak["queue_pressure"] = peak["queue"] / peak["available"]
    fam = family.set_index(["profile_id", "family"])
    odd_policy = odd[odd["row_type"] == "policy"].set_index("policy_or_contrast")
    strict, reference, unconstrained = (odd_policy.loc[x] for x in ("STRICT", "REFERENCE", "UNCONSTRAINED"))
    cost_min, cost_max = float(cost["percent_cost_change"].min()), float(cost["percent_cost_change"].max())
    av_delta_min, av_delta_max = float(cost["delta_AV_assignment_share"].min()), float(cost["delta_AV_assignment_share"].max())
    p95_delta_min, p95_delta_max = float(cost["delta_p95_wait_s"].min()), float(cost["delta_p95_wait_s"].max())
    interaction_qp = interactions[interactions["interaction"] == "q_A_x_p_A"].set_index("stratum")

    lines = [
        "# Stage4 Result Analysis",
        "",
        "This report is a deterministic contrast analysis of the frozen Test31 scenarios. It reports no p-values, inferential confidence intervals, causal effects, or population-level behavioral estimates.",
        "",
        "## 1. Main fleet-transition result",
        "",
        f"Across the 27 main scenarios, marginal mean service rate fell from {_fmt(q_means.loc[0.25])} at q_A=.25 to {_fmt(q_means.loc[0.50])} at .50 and {_fmt(q_means.loc[0.75])} at .75. The .25→.75 change was {_fmt(q_means.loc[0.75]-q_means.loc[0.25])} ({_fmt((q_means.loc[0.75]/q_means.loc[0.25]-1)*100, 1)}%). This is a modeled effective-capacity result, not a claim about AV technology outside Test31.",
        f"Conditional P95 pickup waits remained tightly clustered ({_fmt(main.request_to_pickup_p95.min(),1)}–{_fmt(main.request_to_pickup_p95.max(),1)} s) while service rates ranged {_fmt(main.service_rate.min())}–{_fmt(main.service_rate.max())}; similar waits among served passengers therefore did not imply similar platform performance.",
        "",
        "## 2. Acceptance interaction",
        "",
        f"Raising p_A from .40 to 1.00 changed service at q_A=.25 by C/M/A = {_fmt(p_gain[(0.25,'C')])}/{_fmt(p_gain[(0.25,'M')])}/{_fmt(p_gain[(0.25,'A')])}, versus {_fmt(p_gain[(0.75,'C')])}/{_fmt(p_gain[(0.75,'M')])}/{_fmt(p_gain[(0.75,'A')])} at q_A=.75. The descriptive q_A×p_A contrasts were C/M/A = {_fmt(interaction_qp.loc['C','contrast'])}/{_fmt(interaction_qp.loc['M','contrast'])}/{_fmt(interaction_qp.loc['A','contrast'])}.",
        "The acceptance parameter becomes operationally more consequential when a larger fraction of active vehicle-hours is assigned to AVs, but it remains a frozen scenario-level probability rather than a calibrated Xi'an behavioral estimate.",
        "",
        "## 3. Capability interaction",
        "",
        f"The C→A service gain at q_A=.25 was {_fmt(cap_gain[(0.25,0.4)])}/{_fmt(cap_gain[(0.25,0.7)])}/{_fmt(cap_gain[(0.25,1.0)])} for p_A=.40/.70/1.00; at q_A=.75 it was {_fmt(cap_gain[(0.75,0.4)])}/{_fmt(cap_gain[(0.75,0.7)])}/{_fmt(cap_gain[(0.75,1.0)])}. Capability improvement mitigated reference-envelope restrictions, but did not restore the all-HV service level.",
        "",
        "## 4. Benchmark/extreme interpretation",
        "",
        f"The all-HV benchmark served {_fmt(hv.service_rate)} of requests. The M-profile mixed scenarios at p_A=.70 served {_fmt(m25.service_rate)}, {_fmt(m50.service_rate)}, and {_fmt(m75.service_rate)} at q_A=.25/.50/.75; the all-AV M composition extreme served {_fmt(avm.service_rate)}. Active vehicle-hour substitution therefore did not preserve effective service capacity in this modeled system.",
        "All-AV cases are composition extremes, not performance upper bounds.",
        "",
        "## 5. Temporal mechanism",
        "",
        f"In the defined morning/evening peak windows, cohort service for BENCH_HV versus MAIN_Q25/50/75_M_P70 and BENCH_AV_M was "
        + "/".join(_fmt(peak.loc[s, 'service']) for s in ANCHOR_SCENARIOS)
        + "; corresponding mean queue pressure was "
        + "/".join(_fmt(peak.loc[s, 'queue_pressure']) for s in ANCHOR_SCENARIOS)
        + "; mean total available stock was "
        + "/".join(_fmt(peak.loc[s, 'available'], 1) for s in ANCHOR_SCENARIOS)
        + ". Service fell and queue pressure rose even as more total vehicles remained available, supporting a modeled effective-service-capacity constraint rather than a shortage of logged total stock. The outputs do not record HV/AV available-stock counts separately, so the analysis does not attribute that constraint to a fabricated vehicle-type inventory path.",
        "",
        "## 6. Multi-family ODD result",
        "",
        f"Mean positive assigned-exposure shares for C were static/dynamic/speed = {_fmt(fam.loc[('C','static'),'positive_share_mean'])}/{_fmt(fam.loc[('C','dynamic'),'positive_share_mean'])}/{_fmt(fam.loc[('C','speed'),'positive_share_mean'])}; for M = {_fmt(fam.loc[('M','static'),'positive_share_mean'])}/{_fmt(fam.loc[('M','dynamic'),'positive_share_mean'])}/{_fmt(fam.loc[('M','speed'),'positive_share_mean'])}; for A = {_fmt(fam.loc[('A','static'),'positive_share_mean'])}/{_fmt(fam.loc[('A','dynamic'),'positive_share_mean'])}/{_fmt(fam.loc[('A','speed'),'positive_share_mean'])}. Speed was active for C, nearly inactive for M, and inactive for A: the dominant reference-envelope dimensions changed with the assumed capability profile.",
        "",
        "## 7. ODD policy trade-off",
        "",
        f"STRICT/REFERENCE/UNCONSTRAINED service rates were {_fmt(strict.service_rate)}/{_fmt(reference.service_rate)}/{_fmt(unconstrained.service_rate)}, with AV assignment shares {_fmt(strict.AV_assignment_share)}/{_fmt(reference.AV_assignment_share)}/{_fmt(unconstrained.AV_assignment_share)}. Relative to UNCONSTRAINED, REFERENCE service loss was {_fmt(reference.service_loss_vs_unconstrained)} ({100*reference.relative_service_loss_vs_unconstrained:.1f}%) and STRICT loss was {_fmt(strict.service_loss_vs_unconstrained)} ({100*strict.relative_service_loss_vs_unconstrained:.1f}%).",
        f"REFERENCE exposure reductions versus UNCONSTRAINED were static/dynamic/speed = {_fmt(reference.exposure_reduction_static)}/{_fmt(reference.exposure_reduction_dynamic)}/not-defined (zero unconstrained denominator). STRICT operates at the zero-exposure boundary; this is not evidence of safety.",
        "REFERENCE Gamma was calibrated once from q_A=.25, profile M, p_A=1, UNCONSTRAINED and then held fixed. Positive slack or zero binding epochs does not imply that the policy had no assignment effect.",
        "",
        "## 8. Cost robustness",
        "",
        f"Within eta-matched epsilon=.05 versus 0 comparisons, normalized cost per matched order changed by {100*cost_min:.2f}% to {100*cost_max:.2f}%, AV assignment share by {_fmt(av_delta_min)} to {_fmt(av_delta_max)}, and P95 wait by {_fmt(p95_delta_min,2)} to {_fmt(p95_delta_max,2)} s. These are within-eta comparisons; raw cost levels are not compared as if the objective were invariant across eta.",
        "Each dispatch decision permits a bounded local pickup-objective relaxation; full-day pickup performance emerges endogenously and is not guaranteed to worsen by at most 5%.",
        "",
        "## 9. Managerial implications",
        "",
        f"1. Higher modeled AV penetration reduced marginal service by {_fmt(q_means.loc[0.25]-q_means.loc[0.75])} between q_A=.25 and .75; this applies to the frozen Test31 fleet/session construction.",
        f"2. Acceptance mattered more at high penetration: the M-profile p_A gain increased from {_fmt(p_gain[(0.25,'M')])} at q_A=.25 to {_fmt(p_gain[(0.75,'M')])} at .75; p_A remains a scenario parameter.",
        f"3. Capability C→A improved high-penetration service by {_fmt(cap_gain[(0.75,0.7)])} at q_A=.75,p_A=.70, but the resulting service remained below the all-HV benchmark by {_fmt(hv.service_rate-_scenario(main,0.75,'A',0.7).service_rate)}.",
        f"4. Reference-envelope relevance shifted across profiles: speed positive activity moved from {_fmt(fam.loc[('C','speed'),'positive_share_mean'])} (C) to {_fmt(fam.loc[('M','speed'),'positive_share_mean'])} (M) and {_fmt(fam.loc[('A','speed'),'positive_share_mean'])} (A).",
        f"5. REFERENCE retained {_fmt(reference.service_rate/unconstrained.service_rate*100,1)}% of UNCONSTRAINED service while reducing static and dynamic final exposure by {_fmt(reference.exposure_reduction_static*100,1)}% and {_fmt(reference.exposure_reduction_dynamic*100,1)}%; this is an operational-envelope trade-off, not a safety probability.",
        "",
        "## 10. Limitations / wording constraints",
        "",
        "- Deterministic frozen-scenario contrasts provide no replication-based uncertainty, inferential significance, or causal identification.",
        "- Waiting quantiles are conditional on served passengers; service rate and expiration remain the capacity outcomes.",
        "- Available vehicle stock is logged only in total, not separately for HV and AV.",
        "- Exposure is reference-envelope utilization, not accident risk, failure probability, or certification.",
        "- Cost is normalized and changes definition with eta; no currency conversion or cross-eta raw-cost ranking is made.",
        "- No new simulation, routing, solver run, parameter tuning, or alternative seed was used.",
        "",
        "Recommendation: `GO_PAPER_RESULTS_DRAFT`",
        "",
    ]
    return "\n".join(lines)


def assert_no_inferential_fields(frames: Iterable[pd.DataFrame]) -> None:
    for frame in frames:
        lowered = {str(column).lower() for column in frame.columns}
        for token in INFERENTIAL_TOKENS:
            if any(token in column for column in lowered):
                raise ValueError(f"inferential field forbidden: {token}")


def run_result_analysis(root: str | Path = ".") -> dict[str, Any]:
    root = _root(root)
    tables = load_analysis_inputs(root)
    effects = build_factorial_effects(tables["main"])
    interactions = build_interaction_contrasts(tables["main"])
    benchmarks = build_benchmark_comparison(tables["main"], tables["benchmark"])
    family = build_family_activity_summary(tables["family"])
    odd = build_odd_policy_tradeoff(tables["odd"])
    cost = build_cost_pairwise_robustness(tables["cost"])
    temporal = build_temporal_mechanism(root, tables["scenario"])
    outputs = {
        "factorial_effects.csv": effects,
        "interaction_contrasts.csv": interactions,
        "benchmark_comparison.csv": benchmarks,
        "temporal_mechanism_15min.csv": temporal,
        "family_activity_summary.csv": family,
        "odd_policy_tradeoff.csv": odd,
        "cost_pairwise_robustness.csv": cost,
    }
    assert_no_inferential_fields(outputs.values())
    output_dir = _result_dir(root)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    figures = create_figures(
        tables["main"], interactions, temporal, family, odd, cost, output_dir
    )
    report = build_report(
        root, tables, effects, interactions, benchmarks, temporal, family, odd, cost
    )
    report_path = _report_path(root)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")
    return {
        "recommendation": "GO_PAPER_RESULTS_DRAFT",
        "factorial_scenarios": len(tables["main"]),
        "anchor_scenarios": len(ANCHOR_SCENARIOS),
        "temporal_rows": len(temporal),
        "tables": [str(output_dir / name) for name in outputs],
        "figures": [str(path) for path in figures],
        "report": str(report_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    import json

    print(json.dumps(run_result_analysis(args.root), indent=2))


if __name__ == "__main__":
    main()
