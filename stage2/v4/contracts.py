"""Frozen contracts shared by the Stage 2 v4 pipeline."""

from __future__ import annotations

from collections.abc import Iterable


STAGE2_V4_SCHEMA_VERSION = "stage2_v4.1"
STAGE2_V4_PREFLIGHT_SCHEMA_VERSION = "stage2_v4_preflight.1"
DECISION_TIME_SOURCE = "stage0_order_departure_time"

ROUTE_PRIMARY_KEY = ("split", "date", "order_id", "route_sequence")
TRAVERSAL_PRIMARY_KEY = ("split", "date", "order_id", "traversal_id")
ROUTE_JOIN_KEY = ("split", "date", "order_id", "route_sequence")
ORDER_PRIMARY_KEY = ("split", "date", "order_id")

ROUTE_REQUIRED_COLUMNS = frozenset(
    {
        *ROUTE_PRIMARY_KEY,
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "observed_from_node",
        "observed_to_node",
        "observed_direction",
        "route_part_length_m",
        "canonical_highway",
        "road_class",
        "bridge",
        "tunnel",
        "synthetic_reverse_edge",
        "osm_direction_disagreement",
        "sequence_feature_mask",
        "directed_edge_model_scope",
    }
)

TRAVERSAL_REQUIRED_COLUMNS = frozenset(
    {
        *TRAVERSAL_PRIMARY_KEY,
        "route_sequence",
        "canonical_edge_uid",
        "observed_directed_edge_uid",
        "crawl_time_share",
        "stop_time_share",
        "speed_cv_bounded",
        "acceleration_rms_bounded",
        "acceleration_pair_count",
        "acceleration_weight_s",
        "direct_interval_count",
        "direct_observed_time_s",
        "observation_window_end_time",
        "lcs_raw",
        "lcs_pct",
        "lcs_tail_event",
        "lcs_available",
        "lcs_unavailable_reason",
        "rts_raw",
        "rts_pct",
        "rts_tail_event",
        "rts_available",
        "rts_measurement_available",
        "rts_unavailable_reason",
        "reference_model_id",
        "label_schema_version",
    }
)

ORDER_REQUIRED_COLUMNS = frozenset(
    {
        *ORDER_PRIMARY_KEY,
        "departure_time",
        "start_node",
        "end_node",
        "stage1_core_eligible",
    }
)

PHYSICAL_TRAVERSAL_REQUIRED_COLUMNS = frozenset(
    {
        "order_id",
        "traversal_id",
        "route_sequence",
        "route_sequence_end",
        "enter_time",
        "exit_time",
        "travel_time_s",
        "time_source",
        "time_observation_valid",
        "measurement_source",
        "allocated_distance_m",
    }
)

COMPONENT_TARGETS = (
    "crawl_time_share",
    "stop_time_share",
    "speed_cv_bounded",
    "acceleration_rms_bounded",
)
COMPONENT_MASKS = {
    "crawl_time_share": "crawl_target_valid",
    "stop_time_share": "stop_target_valid",
    "speed_cv_bounded": "speed_cv_target_valid",
    "acceleration_rms_bounded": "acceleration_rms_target_valid",
}
LCS_BASELINE_TARGETS = ("lcs_raw", "lcs_pct", "lcs_tail_event", "lcs_available")
RTS_TARGETS = (
    "rts_raw",
    "rts_pct",
    "rts_tail_event",
    "rts_available",
    "rts_measurement_available",
)
FORBIDDEN_FORMAL_TARGETS = frozenset(
    {"iis_raw", "iis_pct", "pmis_raw", "pmis_pct", "gns_raw", "gns_pct"}
)
FORBIDDEN_LEGACY_INPUT_FRAGMENTS = (
    "stage1/output/prediction_split",
    "stage2/output/link_dataset",
    "stage2/output/route_conditioned_dataset",
    "stage2/output/deep_v3",
)


class Stage2V4ContractError(RuntimeError):
    """Raised when a frozen Stage 2 v4 contract is violated."""


def require_columns(
    actual: Iterable[str],
    required: Iterable[str],
    product: str,
) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        raise Stage2V4ContractError(
            f"{product} schema is missing required columns: {missing}"
        )
