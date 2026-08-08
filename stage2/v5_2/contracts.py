"""Frozen research and product contracts for Stage 2 v5.2."""

from __future__ import annotations

from collections.abc import Iterable, Mapping


SCHEMA_VERSION = "stage2_v5_2.1"
FINAL_READY_STATUS = "READY_FOR_AV_ROUTE_SUITABILITY_STAGE"
IMPLEMENTATION_ONLY_STATUS = "NOT_READY_IMPLEMENTATION_ONLY"

MICRO_TARGETS = (
    "crawl",
    "stop",
    "speed_cv",
    "acceleration_rms",
    "rts",
)
CORE_TRANSFER_TARGETS = ("crawl", "stop", "speed_cv", "acceleration_rms")
SECONDARY_TRANSFER_TARGETS = ("rts",)
COMMON_OPERATIONAL_TARGETS = ("pace_p50", "travel_time_p50")
TOKEN_IDENTITY_COLUMNS = (
    "order_id",
    "route_sequence",
    "traversal_id",
    "observed_directed_edge_uid",
    "canonical_edge_uid",
)
TOKEN_PREDICTION_COLUMNS = (
    "pred_crawl_share",
    "pred_stop_share",
    "pred_speed_cv_bounded",
    "pred_acceleration_rms_bounded",
    "pred_rts_raw",
    "pred_pace_p50",
    "estimated_travel_time_p50_s",
)
TOKEN_AVAILABILITY_COLUMNS = tuple(f"{name}_target_available" for name in (
    "crawl", "stop", "speed_cv", "acceleration", "rts"
))
TOKEN_SUPPORT_COLUMNS = (
    "history_support",
    "edge_train_support",
    "edge_seen_in_train",
    "support_group",
)
TOKEN_PROVENANCE_COLUMNS = (
    "protocol_id",
    "prediction_source",
    "model_id",
    "model_hash",
    "decision_time",
    "feature_cutoff_time",
    "feature_age_s",
    "route_track",
    "route_source",
    "route_product_version",
)
TOKEN_REQUIRED_COLUMNS = (
    *TOKEN_IDENTITY_COLUMNS,
    *TOKEN_PREDICTION_COLUMNS,
    *TOKEN_AVAILABILITY_COLUMNS,
    *TOKEN_SUPPORT_COLUMNS,
    *TOKEN_PROVENANCE_COLUMNS,
)

ROUTE_DYNAMIC_COLUMNS = (
    "crawl_weighted_mean", "crawl_weighted_p90", "crawl_high_exposure_share",
    "crawl_max_consecutive_high_share",
    "stop_weighted_mean", "stop_weighted_p90", "stop_high_exposure_share",
    "stop_max_consecutive_high_share",
    "speed_cv_weighted_mean", "speed_cv_weighted_p90", "speed_cv_high_exposure_share",
    "acceleration_weighted_mean", "acceleration_weighted_p90", "acceleration_high_exposure_share",
    "rts_weighted_mean", "rts_weighted_p90", "rts_high_exposure_share",
    "rts_distance_weighted_mean", "rts_distance_weighted_p90",
    "route_total_distance_m", "partial_travel_time_p50_s", "travel_time_p50_s",
    "pace_prediction_coverage_distance", "micro_prediction_coverage_distance",
    "crawl_prediction_coverage", "stop_prediction_coverage", "speed_cv_prediction_coverage",
    "acceleration_prediction_coverage", "rts_prediction_coverage",
    "micro_condition_coverage", "low_support_route_share",
    "unseen_edge_route_share", "support_weighted_mean", "unknown_flag",
)

STATIC_COMPLEXITY_COLUMNS = (
    "intersection_exposure_share", "signal_exposure_share", "merge_exposure_share",
    "turn_exposure_share", "ramp_exposure_share", "bridge_exposure_share",
    "tunnel_exposure_share", "road_class_transition_rate",
    "canonical_highway_transition_rate", "canonical_highway_entropy",
    "motorway_trunk_exposure_share", "primary_secondary_exposure_share",
)
STABLE_STATIC_INPUTS = ("canonical_highway", "road_class", "bridge", "tunnel")
UNAVAILABLE_STATIC_INPUTS = (
    "intersection", "signal", "merge", "turn", "ramp", "speed_limit", "lane_information"
)

FORBIDDEN_MODEL_INPUTS = frozenset(
    {
        "order_id", "driver_id", "stage1_ground_truth", "actual_future_travel_time",
        "actual_link_entry_time", "oracle_entry_time", "test_truth", "evaluation_support",
        "validation_support", "future_state", "av_safety_probability",
    }
)
FORBIDDEN_STAGE3_FIELDS = frozenset(
    {
        "stage1_ground_truth", "actual_future_travel_time", "oracle_entry_time",
        "driver_id", "test_truth", "av_safety_probability",
    }
)
FORBIDDEN_STAGE3_CONCEPTS = frozenset(
    {"route_decision_variable", "path_planning", "fallback_route_search", "hv_av_assignment"}
)

RESEARCH_CONTRACT = {
    "schema_version": SCHEMA_VERSION,
    "formal_micro_targets": list(MICRO_TARGETS),
    "common_operational_targets": list(COMMON_OPERATIONAL_TARGETS),
    "travel_time_is_only_formal_target": False,
    "route_identity": "historical_original_service_route",
    "hv_route_policy": "original_route_only",
    "av_route_policy": "evaluate_original_route_first; fallback is deferred to Stage 3",
    "transfer_scope": "high-support/historical knowledge to low-support, unseen edges and later dates",
    "stage2_excludes": [
        "AV safety or failure probability", "path planning", "fallback route search",
        "HV/AV assignment", "Stage 4 optimization",
    ],
    "deprecated_to_appendix": [
        "strict route P90/P95 coverage", "route mean/std/CVaR", "joint random service-time distribution",
        "complex copula", "cross-order deep scenario calibration", "log-normal mean",
    ],
}


class Stage2V52ContractError(ValueError):
    """Raised when v5.2 input or output violates a frozen contract."""


def require_columns(columns: Iterable[str], required: Iterable[str], *, product: str) -> None:
    missing = sorted(set(required) - set(columns))
    if missing:
        raise Stage2V52ContractError(f"{product} is missing required columns: {missing}")


def validate_model_inputs(columns: Iterable[str]) -> None:
    forbidden = sorted(set(columns) & FORBIDDEN_MODEL_INPUTS)
    if forbidden:
        raise Stage2V52ContractError(f"forbidden decision-time model inputs: {forbidden}")


def validate_research_contract(payload: Mapping[str, object]) -> None:
    targets = tuple(payload.get("formal_micro_targets", ()))
    if targets != MICRO_TARGETS:
        raise Stage2V52ContractError("formal micro targets differ from the frozen v5.2 contract")
    if payload.get("travel_time_is_only_formal_target") is not False:
        raise Stage2V52ContractError("travel time cannot be the only formal Stage 2 target")
    if payload.get("hv_route_policy") != "original_route_only":
        raise Stage2V52ContractError("HV must always retain the original route")
