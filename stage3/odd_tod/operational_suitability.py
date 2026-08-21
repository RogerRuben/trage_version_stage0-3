"""Stage 3 S4 v2 operational-suitability interface for frozen Test31 routes.

This module is deliberately a thin derivation over the existing S4 products.
It does not reload raw GPS, rerun route matching, run M3 inference, fit a CDF,
or change the frozen Stage 3 capability profiles.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stage3.odd_tod.network_foundation import (
    Stage3S2AError,
    atomic_json,
    atomic_parquet,
    atomic_text,
    parquet_descriptor,
    payload_hash,
    read_json,
    sha256_file,
    source_descriptor,
)
from stage3.odd_tod.original_route_suitability import (
    DYNAMIC_DIMS,
    EXPECTED_ORDER_COUNT,
    EXPECTED_ORDER_PROFILE_COUNT,
    M3_SHA256,
    PROFILE_REL,
    PROFILES,
    ROUTE_REL,
    TEST_DATE,
    _profile_map,
)


PHASE_STATUS = "STAGE3_S4_V2_COMPLETE"
OUTPUT_REL = Path("stage3/output/odd_tod/s4")
DOCS_REL = Path("stage3/docs/odd_tod/s4_v2")
DESCRIPTOR_REL = OUTPUT_REL / "test31_original_route_descriptors.parquet"
V1_SUITABILITY_REL = OUTPUT_REL / "test31_original_route_suitability.parquet"
PREDICTION_MANIFEST_REL = OUTPUT_REL / "test31_m3_predictions.json"
OUTPUT_PRODUCT_REL = OUTPUT_REL / "test31_av_operational_suitability.parquet"
SUMMARY_REL = OUTPUT_REL / "test31_av_operational_suitability_summary.json"
MANIFEST_REL = OUTPUT_REL / "test31_av_operational_suitability_manifest.json"
REPORT_REL = DOCS_REL / "stage3_s4_v2_operational_suitability_report.md"
AUDIT_REL = DOCS_REL / "stage3_s4_v2_reuse_audit.md"
CONTRACT_REL = DOCS_REL / "stage3_s4_v2_to_stage4_contract.md"

STATIC_DIMENSIONS = {
    "A": ("route_max_A_c", "external_physical_connection_count"),
    "M": ("route_max_M_c", "topological_movement_count"),
    "D": ("route_max_D_c", "road_class_diversity"),
    "L": ("route_max_L_c", "internal_length_m"),
}
DYNAMIC_METRICS = tuple(
    f"{dimension}_{metric}"
    for dimension in DYNAMIC_DIMS
    for metric in ("E", "Q", "C")
)

HARD_REASON_CODES = frozenset(
    {
        "KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE",
        "CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE",
        "UTURN_PROFILE_INCOMPATIBLE",
        "CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE",
        "CERTIFIED_MOVEMENT_PROHIBITION",
    }
)
SOFT_REASON_CODES = frozenset(
    {
        "SPEED_DOMAIN_CAP_EXCEEDED",
        "STATIC_A_CAP_EXCEEDED",
        "STATIC_M_CAP_EXCEEDED",
        "STATIC_D_CAP_EXCEEDED",
        "STATIC_L_CAP_EXCEEDED",
        *{
            f"DYNAMIC_{dimension.upper()}_{metric}_CAP_EXCEEDED"
            for dimension in DYNAMIC_DIMS
            for metric in ("E", "Q", "C")
        },
    }
)

OUTPUT_COLUMNS = (
    "date",
    "order_id",
    "profile_id",
    "hard_state",
    "rho_static",
    "rho_dynamic",
    "rho_speed",
    "rho_overall",
    "static_A_ratio",
    "static_M_ratio",
    "static_D_ratio",
    "static_L_ratio",
    "dynamic_12_ratios",
    "static_vector",
    "dynamic_vector",
    "hard_reason_codes",
    "unknown_reason_codes",
    "soft_exceedance_reason_codes",
    "reason_codes",
    "dominant_utilization_family",
    "direction_hard_constraint",
    "passenger_acceptance_probability",
    "original_route",
)


def _json_list(value: Any) -> list[str]:
    if isinstance(value, str):
        parsed = json.loads(value)
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = []
    if not isinstance(parsed, list):
        raise Stage3S2AError("reason-code field is not a JSON list")
    return sorted({str(item) for item in parsed})


def utilization_ratio(observed: Any, cap: Any) -> float:
    """Return x/B without clipping; invalid evidence stays unknown."""
    if observed is None or cap is None or pd.isna(observed) or pd.isna(cap):
        return float("nan")
    observed_value = float(observed)
    cap_value = float(cap)
    if not np.isfinite(observed_value) or not np.isfinite(cap_value) or cap_value <= 0:
        return float("nan")
    return observed_value / cap_value


def overall_utilization(rho_static: Any, rho_dynamic: Any, rho_speed: Any) -> float:
    """Non-compensatory max; missing required evidence is not silently dropped."""
    values = np.asarray([rho_static, rho_dynamic, rho_speed], dtype=np.float64)
    if not np.isfinite(values).all():
        return float("nan")
    return float(values.max())


def hard_state_from_reasons(
    hard_reasons: Sequence[str], unknown_reasons: Sequence[str]
) -> str:
    """Known structural prohibition precedes critical missing evidence."""
    if hard_reasons:
        return "INFEASIBLE"
    if unknown_reasons:
        return "UNKNOWN"
    return "FEASIBLE"


def _soft_label(code: str) -> str:
    if code == "SPEED_DOMAIN_CAP_EXCEEDED":
        return "SOFT_SPEED_ENVELOPE_EXCEEDED"
    if code.startswith("STATIC_"):
        return code.replace("_CAP_EXCEEDED", "_ENVELOPE_EXCEEDED").replace(
            "STATIC_", "SOFT_STATIC_", 1
        )
    if code.startswith("DYNAMIC_"):
        return code.replace("_CAP_EXCEEDED", "_ENVELOPE_EXCEEDED").replace(
            "DYNAMIC_", "SOFT_DYNAMIC_", 1
        )
    raise Stage3S2AError(f"unclassified soft reason code: {code}")


def _input_audit(root: Path) -> dict[str, Any]:
    required = {
        "frozen_profile": root / PROFILE_REL,
        "frozen_m3_checkpoint": root
        / "stage2/output_v5_2/development/M3/epoch_004.pt",
        "frozen_original_route": root / ROUTE_REL,
        "route_descriptors": root / DESCRIPTOR_REL,
        "v1_profile_assessment": root / V1_SUITABILITY_REL,
        "prediction_manifest": root / PREDICTION_MANIFEST_REL,
    }
    missing = [name for name, path in required.items() if not path.is_file()]
    if missing:
        raise Stage3S2AError(f"S4 v2 missing reusable inputs: {missing}")
    profile = read_json(required["frozen_profile"])
    _profile_map(profile)
    prediction = read_json(required["prediction_manifest"])
    prediction_gates = {
        "checkpoint_unchanged": prediction.get("checkpoint_sha256") == M3_SHA256,
        "decision_time_only": prediction.get("decision_time_only") is True,
        "predicted_progression_only": prediction.get("predicted_progression_only") is True,
        "realized_future_time_used": prediction.get("realized_future_time_used") is False,
        "realized_targets_persisted": prediction.get("realized_target_columns_persisted")
        is False,
        "route_search_performed": prediction.get("test31_route_search_performed") is False,
        "fallback_performed": prediction.get("test31_fallback_performed") is False,
    }
    if not all(prediction_gates.values()):
        raise Stage3S2AError(f"frozen M3 reuse gate failed: {prediction_gates}")
    if sha256_file(required["frozen_m3_checkpoint"]) != M3_SHA256:
        raise Stage3S2AError("frozen M3 checkpoint changed")
    return {
        "schema_version": "stage3_s4_v2_input_audit.1",
        "phase_status": "S4_V2_REUSE_AUDIT_PASS",
        "reused_modules": [
            "Test31 historical-route loading",
            "typed route identity resolution",
            "Stage3 edge mapping",
            "production intersection-complex extraction",
            "frozen M3 decision-time inference",
            "frozen Train mid-CDF transformation",
            "route E/Q/C construction",
            "static A/M/D/L descriptors with boundary D",
            "reason attribution",
        ],
        "rerun_required": {
            "route_loading": False,
            "identity_resolution": False,
            "intersection_parsing": False,
            "m3_inference": False,
            "cdf_transformation": False,
            "eqc_construction": False,
        },
        "inputs": {
            name: (
                parquet_descriptor(path, root)
                if path.suffix == ".parquet"
                else source_descriptor(path, root)
            )
            for name, path in required.items()
        },
        "prediction_gates": prediction_gates,
    }


def _ratio_vectors(
    frame: pd.DataFrame, profile_by_id: Mapping[str, Mapping[str, Any]]
) -> pd.DataFrame:
    result = frame.copy()
    for label in STATIC_DIMENSIONS:
        result[f"static_{label}_ratio"] = np.nan
    for metric in DYNAMIC_METRICS:
        result[f"_ratio_{metric}"] = np.nan
    result["rho_speed"] = np.nan

    for profile_id in PROFILES:
        mask = result["profile_id"].eq(profile_id)
        profile = profile_by_id[profile_id]
        for label, (source, cap_key) in STATIC_DIMENSIONS.items():
            observed = pd.to_numeric(result.loc[mask, source], errors="coerce")
            # A route with no encountered complex has zero exposure. Missing
            # metrics on an encountered complex remain unknown.
            observed = observed.mask(
                observed.isna()
                & result.loc[mask, "resolved_complex_encounter_count"].eq(0),
                0.0,
            )
            result.loc[mask, f"static_{label}_ratio"] = (
                observed / float(profile["static_caps"][cap_key])
            ).to_numpy()
        for dimension in DYNAMIC_DIMS:
            for metric in ("E", "Q", "C"):
                source = f"{dimension}_{metric}"
                result.loc[mask, f"_ratio_{source}"] = (
                    pd.to_numeric(result.loc[mask, source], errors="coerce")
                    / float(profile["dynamic_caps"][dimension][metric])
                ).to_numpy()
        speed = pd.to_numeric(result.loc[mask, "max_route_speed_domain_kmh"], errors="coerce")
        speed = speed.mask(
            speed.isna() & result.loc[mask, "full_network_edge_token_count"].eq(0),
            0.0,
        )
        speed = speed.mask(result.loc[mask, "unknown_speed_edge_count"].gt(0))
        result.loc[mask, "rho_speed"] = (
            speed / float(profile["speed_domain_max_kmh"])
        ).to_numpy()

    static_ratio_columns = [f"static_{label}_ratio" for label in STATIC_DIMENSIONS]
    dynamic_ratio_columns = [f"_ratio_{metric}" for metric in DYNAMIC_METRICS]
    result["rho_static"] = result[static_ratio_columns].max(axis=1, skipna=False)
    result["rho_dynamic"] = result[dynamic_ratio_columns].max(axis=1, skipna=False)
    result["rho_overall"] = result[["rho_static", "rho_dynamic", "rho_speed"]].max(
        axis=1, skipna=False
    )
    result["static_vector"] = [
        json.dumps(
            {label: (None if pd.isna(row[f"static_{label}_ratio"]) else float(row[f"static_{label}_ratio"])) for label in STATIC_DIMENSIONS},
            sort_keys=True,
            separators=(",", ":"),
        )
        for _, row in result.iterrows()
    ]
    result["dynamic_12_ratios"] = [
        json.dumps(
            {metric: (None if pd.isna(row[f"_ratio_{metric}"]) else float(row[f"_ratio_{metric}"])) for metric in DYNAMIC_METRICS},
            sort_keys=True,
            separators=(",", ":"),
        )
        for _, row in result.iterrows()
    ]
    result["dynamic_vector"] = result["dynamic_12_ratios"]
    return result


def _states_and_attribution(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    hard_lists: list[list[str]] = []
    unknown_lists: list[list[str]] = []
    soft_lists: list[list[str]] = []
    states: list[str] = []
    for row in result.itertuples(index=False):
        known = _json_list(row.known_violation_reason_codes)
        unknown = set(_json_list(row.unknown_reason_codes))
        if int(row.unresolved_token_count) > 0:
            unknown.add("UNRESOLVED_ROUTE_IDENTITY")
        hard = sorted(code for code in known if code in HARD_REASON_CODES)
        soft_raw = sorted(code for code in known if code in SOFT_REASON_CODES)
        unexpected = sorted(set(known) - HARD_REASON_CODES - SOFT_REASON_CODES)
        if unexpected:
            raise Stage3S2AError(f"unclassified v1 reason codes: {unexpected}")
        soft = sorted(_soft_label(code) for code in soft_raw)
        unknown_sorted = sorted(unknown)
        hard_lists.append(hard)
        unknown_lists.append(unknown_sorted)
        soft_lists.append(soft)
        states.append(hard_state_from_reasons(hard, unknown_sorted))
    result["hard_state"] = states
    result["hard_reason_codes"] = [json.dumps(v, separators=(",", ":")) for v in hard_lists]
    result["unknown_reason_codes"] = [json.dumps(v, separators=(",", ":")) for v in unknown_lists]
    result["soft_exceedance_reason_codes"] = [
        json.dumps(v, separators=(",", ":")) for v in soft_lists
    ]
    result["reason_codes"] = [
        json.dumps(sorted(set(hard) | set(unknown) | set(soft)), separators=(",", ":"))
        for hard, unknown, soft in zip(hard_lists, unknown_lists, soft_lists)
    ]
    result["direction_hard_constraint"] = result["reverse_overlay_token_count"].gt(0)
    utilization = result[["rho_static", "rho_dynamic", "rho_speed"]]
    complete = utilization.notna().all(axis=1)
    dominant = pd.Series("UNKNOWN_INCOMPLETE", index=result.index, dtype="string")
    dominant.loc[complete] = (
        utilization.loc[complete]
        .idxmax(axis=1)
        .str.replace("rho_", "", regex=False)
        .str.upper()
    )
    dominant.loc[result["direction_hard_constraint"]] = "DIRECTION"
    result["dominant_utilization_family"] = dominant
    result["passenger_acceptance_probability"] = np.nan
    result["original_route"] = "FROZEN_HISTORICAL_ROUTE_20161031"
    return result


def _distribution(values: pd.Series) -> dict[str, Any]:
    finite = pd.to_numeric(values, errors="coerce").dropna().to_numpy(np.float64)
    return {
        "finite_count": int(len(finite)),
        "missing_count": int(len(values) - len(finite)),
        "p50": float(np.quantile(finite, 0.50)) if len(finite) else None,
        "p90": float(np.quantile(finite, 0.90)) if len(finite) else None,
        "p95": float(np.quantile(finite, 0.95)) if len(finite) else None,
        "p99": float(np.quantile(finite, 0.99)) if len(finite) else None,
        "max": float(finite.max()) if len(finite) else None,
        "near_one_0p9_to_1p1_count": int(((finite >= 0.9) & (finite <= 1.1)).sum()),
        "above_one_count": int((finite > 1.0).sum()),
        "above_two_count": int((finite > 2.0).sum()),
    }


def _summary(product: pd.DataFrame) -> dict[str, Any]:
    profiles: dict[str, Any] = {}
    for profile_id in PROFILES:
        frame = product[product["profile_id"].eq(profile_id)]
        counts = frame["hard_state"].value_counts()
        reason_lists = frame["reason_codes"].map(json.loads)
        all_reasons = sorted({code for values in reason_lists for code in values})
        profiles[profile_id] = {
            "hard_state_counts": {
                state: int(counts.get(state, 0))
                for state in ("FEASIBLE", "UNKNOWN", "INFEASIBLE")
            },
            "rho_static": _distribution(frame["rho_static"]),
            "rho_dynamic": _distribution(frame["rho_dynamic"]),
            "rho_speed": _distribution(frame["rho_speed"]),
            "rho_overall": _distribution(frame["rho_overall"]),
            "dominant_utilization_family_counts": {
                str(key): int(value)
                for key, value in frame["dominant_utilization_family"].value_counts().items()
            },
            "direction_hard_constraint_count": int(frame["direction_hard_constraint"].sum()),
            "marginal_reason_counts": {
                code: int(reason_lists.map(lambda values, c=code: c in values).sum())
                for code in all_reasons
            },
        }
    return {
        "schema_version": "stage3_s4_v2_summary.1",
        "phase_status": PHASE_STATUS,
        "date": TEST_DATE,
        "order_count": int(product["order_id"].nunique()),
        "order_profile_row_count": int(len(product)),
        "profiles": profiles,
        "rho_definition": "max family utilization; no weighted average; ratios are not clipped",
        "hard_state_definition": "structural constraints only; known prohibition > critical unknown > feasible",
        "near_one_reporting_band": "0.9 <= rho_overall <= 1.1 (descriptive only)",
        "extreme_reporting_band": "rho_overall > 2.0 (descriptive only)",
        "original_route_only": True,
        "route_replanning": False,
        "passenger_model_fitted": False,
        "s5_authorized": False,
        "next_phase_authorized": False,
    }


def _report(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 3 S4 v2 — AV Operational Suitability Interface",
        "",
        "S4 v2 keeps the frozen Test31 historical route fixed and exposes three separate products: structural hard feasibility, continuous capability-envelope utilization, and diagnostic attribution.",
        "",
        "It does not estimate AV safety, accident probability, legal certification, passenger choice, or an optimized route. No rerouting, fallback, dispatch, profile refit, CDF fit, or M3 retraining was performed.",
        "",
        "## Population",
        "",
        f"- Test date: `{summary['date']}`",
        f"- Orders: {summary['order_count']:,}",
        f"- Order-profile rows: {summary['order_profile_row_count']:,}",
        "- Original route only: YES",
        "- Passenger acceptance probability: reserved nullable Stage4 input; not modeled here",
        "",
        "## Profile comparison",
        "",
        "| Profile | Hard FEASIBLE | Hard UNKNOWN | Hard INFEASIBLE | rho overall p50 | p90 | p99 | near 1 | >1 | >2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for profile_id in PROFILES:
        item = summary["profiles"][profile_id]
        state = item["hard_state_counts"]
        rho = item["rho_overall"]
        lines.append(
            f"| {profile_id} | {state['FEASIBLE']:,} | {state['UNKNOWN']:,} | {state['INFEASIBLE']:,} | "
            f"{rho['p50']:.4f} | {rho['p90']:.4f} | {rho['p99']:.4f} | "
            f"{rho['near_one_0p9_to_1p1_count']:,} | "
            f"{rho['above_one_count']:,} | {rho['above_two_count']:,} |"
        )
    lines.extend(
        [
            "",
            "`rho > 1` means the frozen route descriptor exceeds at least one frozen profile envelope. It is a capability-requirement signal, not an automatic impossibility or safety claim.",
            "",
            "## Capability requirement distribution by family",
            "",
            "| Profile | Static p50 / p90 | Dynamic p50 / p90 | Speed p50 / p90 | Overall p50 / p90 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for profile_id in PROFILES:
        item = summary["profiles"][profile_id]
        lines.append(
            f"| {profile_id} | {item['rho_static']['p50']:.4f} / {item['rho_static']['p90']:.4f} | "
            f"{item['rho_dynamic']['p50']:.4f} / {item['rho_dynamic']['p90']:.4f} | "
            f"{item['rho_speed']['p50']:.4f} / {item['rho_speed']['p90']:.4f} | "
            f"{item['rho_overall']['p50']:.4f} / {item['rho_overall']['p90']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Dominant utilization family",
            "",
        ]
    )
    for profile_id in PROFILES:
        counts = summary["profiles"][profile_id]["dominant_utilization_family_counts"]
        lines.append(
            f"- {profile_id}: " + ", ".join(f"{key}={value:,}" for key, value in counts.items())
        )
    lines.extend(
        [
            "",
            "Direction is reported as a structural hard constraint. Static, dynamic, and speed bottlenecks are determined by the non-weighted maximum utilization family. Missing critical evidence remains UNKNOWN and is never silently dropped from `rho_overall`.",
            "",
            "## Stage4 contract",
            "",
            "Stage4 should consume `hard_state + rho_* + vectors + reason_codes + passenger_acceptance_probability`. The passenger field is intentionally null in S4 v2.",
            "",
            "S5_AUTHORIZED = NO",
            "NEXT_PHASE_AUTHORIZED = NO",
        ]
    )
    return "\n".join(lines) + "\n"


def build_operational_suitability(root: str | Path) -> dict[str, Any]:
    root_path = Path(root).resolve()
    audit = _input_audit(root_path)
    profile_sha_before = sha256_file(root_path / PROFILE_REL)
    checkpoint_sha_before = sha256_file(
        root_path / "stage2/output_v5_2/development/M3/epoch_004.pt"
    )
    profile_by_id = _profile_map(read_json(root_path / PROFILE_REL))
    descriptor = pd.read_parquet(root_path / DESCRIPTOR_REL)
    v1 = pd.read_parquet(root_path / V1_SUITABILITY_REL)
    if len(descriptor) != EXPECTED_ORDER_COUNT or descriptor["order_id"].nunique() != EXPECTED_ORDER_COUNT:
        raise Stage3S2AError("S4 v2 descriptor source is not exactly 30,000 Test31 orders")
    if (
        len(v1) != EXPECTED_ORDER_PROFILE_COUNT
        or v1["order_id"].nunique() != EXPECTED_ORDER_COUNT
        or set(v1["date"].astype(str).unique()) != {TEST_DATE}
        or set(v1["profile_id"].astype(str).unique()) != set(PROFILES)
        or v1.duplicated(["order_id", "profile_id"]).any()
    ):
        raise Stage3S2AError("S4 v2 source is not exact Test31 30,000 x C/M/A")

    ratio_frame = _ratio_vectors(v1, profile_by_id)
    product = _states_and_attribution(ratio_frame)
    product.insert(0, "date", product.pop("date").astype(str))
    product = product[list(OUTPUT_COLUMNS)]
    if len(product) != EXPECTED_ORDER_PROFILE_COUNT or product.duplicated(["order_id", "profile_id"]).any():
        raise Stage3S2AError("S4 v2 output is not exactly 90,000 unique order/profile rows")
    if set(product["hard_state"].unique()) - {"FEASIBLE", "UNKNOWN", "INFEASIBLE"}:
        raise Stage3S2AError("S4 v2 hard_state vocabulary is invalid")
    if not np.allclose(
        product["rho_overall"].dropna().to_numpy(),
        product.loc[product["rho_overall"].notna(), ["rho_static", "rho_dynamic", "rho_speed"]].max(axis=1).to_numpy(),
    ):
        raise Stage3S2AError("rho_overall is not the non-weighted family maximum")

    output_path = root_path / OUTPUT_PRODUCT_REL
    atomic_parquet(output_path, product)
    summary = _summary(product)
    summary["profile_sha256_before"] = profile_sha_before
    summary["profile_sha256_after"] = sha256_file(root_path / PROFILE_REL)
    summary["m3_checkpoint_sha256_before"] = checkpoint_sha_before
    summary["m3_checkpoint_sha256_after"] = sha256_file(
        root_path / "stage2/output_v5_2/development/M3/epoch_004.pt"
    )
    if summary["profile_sha256_before"] != summary["profile_sha256_after"]:
        raise Stage3S2AError("frozen Stage3 profile changed during S4 v2")
    if summary["m3_checkpoint_sha256_before"] != summary["m3_checkpoint_sha256_after"]:
        raise Stage3S2AError("frozen M3 checkpoint changed during S4 v2")
    summary["output"] = parquet_descriptor(output_path, root_path)
    summary["artifact_sha256"] = payload_hash(summary)
    atomic_json(root_path / SUMMARY_REL, summary)

    docs = root_path / DOCS_REL
    docs.mkdir(parents=True, exist_ok=True)
    atomic_text(
        root_path / AUDIT_REL,
        "# Stage 3 S4 v2 Reuse Audit\n\n"
        "The existing S4 route loading, typed identity, Stage3 edge mapping, production complex parser, frozen M3 inference, frozen Train CDF transform, E/Q/C construction, AMDL descriptors, and reason attribution were inspected and reused. No upstream computation was repeated.\n\n"
        "Only the hard/continuous interface is new. Static, dynamic, and speed envelope exceedances are continuous utilization evidence rather than structural hard infeasibility.\n",
    )
    atomic_text(root_path / REPORT_REL, _report(summary))
    atomic_text(
        root_path / CONTRACT_REL,
        "# Stage 3 S4 v2 to Stage 4 Contract\n\n"
        "Canonical input: `stage3/output/odd_tod/s4/test31_av_operational_suitability.parquet`.\n\n"
        "- Key: `(date, order_id, profile_id)`; exactly one row for each Test31 order and C/M/A profile.\n"
        "- `hard_state`: structural dispatch constraint only. `INFEASIBLE` means a known forbidden direction/maneuver/restriction; `UNKNOWN` means no known hard violation but critical evidence is missing.\n"
        "- `rho_static`, `rho_dynamic`, `rho_speed`: nonnegative, unclipped envelope-utilization ratios.\n"
        "- `rho_overall`: `max(rho_static, rho_dynamic, rho_speed)` with no weighted average. It is null when any required family is unevaluable.\n"
        "- `static_vector` and `dynamic_12_ratios`/`dynamic_vector`: compact JSON objects of the component ratios.\n"
        "- `reason_codes`: union of structural hard, critical unknown, and `SOFT_*_ENVELOPE_EXCEEDED` diagnostic codes.\n"
        "- `passenger_acceptance_probability`: nullable Stage4 input placeholder; S4 v2 leaves every value null.\n"
        "- `original_route`: always the frozen historical Test31 route marker.\n\n"
        "Stage4 may combine hard state, continuous utilization, and passenger preference. S4 v2 does not perform dispatch, optimization, rerouting, fallback, or passenger modeling.\n\n"
        "S5_AUTHORIZED = NO\n",
    )

    manifest = {
        **audit,
        "schema_version": "stage3_s4_v2_manifest.1",
        "phase_status": PHASE_STATUS,
        "output": parquet_descriptor(output_path, root_path),
        "summary": source_descriptor(root_path / SUMMARY_REL, root_path),
        "report": source_descriptor(root_path / REPORT_REL, root_path),
        "reuse_audit": source_descriptor(root_path / AUDIT_REL, root_path),
        "stage4_contract": source_descriptor(root_path / CONTRACT_REL, root_path),
        "hard_state_logic": "structural known prohibition > critical unknown > feasible",
        "rho_logic": "rho_static=max(A/Ak,M/Mk,D/Dk,L/Lk); rho_dynamic=max(12 ratios); rho_speed=exposure/cap; rho_overall=max(families)",
        "weighted_average_used": False,
        "route_replanning": False,
        "fallback": False,
        "dispatch": False,
        "passenger_model": False,
        "optimization": False,
        "s5_authorized": False,
        "next_phase_authorized": False,
    }
    manifest["artifact_sha256"] = payload_hash(manifest)
    atomic_json(root_path / MANIFEST_REL, manifest)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    summary = build_operational_suitability(args.root)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
