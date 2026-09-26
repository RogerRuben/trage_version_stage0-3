"""Stage 0 v6 input schema frozen for Stage 1 v3."""

from __future__ import annotations

from types import MappingProxyType


INPUT_ROOT_NAME = "input_v1"
INPUT_BUCKET_SCHEMA_VERSION = "stage1_input_bucket.1"
OUTPUT_BUCKET_SCHEMA_VERSION = "stage1_v3_output_bucket.2"
OUTPUT_SUMMARY_SCHEMA_VERSION = "stage1_v3_output_summary.2"

ALL_INPUT_PRODUCTS = (
    "order_base",
    "route_segments",
    "route_parts",
    "link_traversals",
    "link_interval_observations",
    "interval_measurements",
    "turn_movements",
    "gps_quality",
    "route_quality",
    "dynamic_quality",
    "canonical_quality",
)

REQUIRED_PRODUCTS = (
    "order_base",
    "route_parts",
    "link_traversals",
    "link_interval_observations",
    "interval_measurements",
    "turn_movements",
    "gps_quality",
    "route_quality",
    "dynamic_quality",
    "canonical_quality",
)

DIRECT_MEASUREMENT_SOURCE = "direct_observed"
KNOWN_MEASUREMENT_SOURCES = frozenset(
    {
        DIRECT_MEASUREMENT_SOURCE,
        "interval_supported",
        "engine_interpolated",
        "unresolved",
    }
)

DYNAMIC_STATUSES = frozenset({"dynamic_strict", "dynamic_partial"})
KNOWN_MOVEMENT_SOURCES = frozenset(
    {
        "directly_observed_transition",
        "path_supported_transition",
        "engine_inferred_transition",
        "unmapped_transition",
    }
)

REQUIRED_COLUMNS = MappingProxyType(
    {
        "order_base": frozenset(
            {
                "order_id",
                "date",
                "split",
                "departure_time",
                "arrival_time",
                "start_node",
                "end_node",
                "gps_status",
                "route_status",
                "dynamic_status",
                "canonical_status",
                "selection_hash",
                "stage1_core_eligible",
            }
        ),
        "route_parts": frozenset(
            {
                "order_id",
                "route_sequence",
                "canonical_edge_uid",
                "canonical_from_node",
                "canonical_to_node",
                "canonical_highway",
                "canonical_length_m",
                "length_m",
                "begin_osm_node_id",
                "end_osm_node_id",
                "mapping_status",
                "canonical_traversal_direction",
                "osm_oneway",
                "traversed_against_osm_oneway",
                "road_class",
                "bridge",
                "tunnel",
                "measurement_source",
            }
        ),
        "link_traversals": frozenset(
            {
                "order_id",
                "traversal_id",
                "route_sequence",
                "route_sequence_end",
                "canonical_edge_uid",
                "measurement_source",
                "observed_travel_time_s",
                "observed_distance_m",
                "allocated_distance_m",
            }
        ),
        "link_interval_observations": frozenset(
            {
                "order_id",
                "gps_interval_id",
                "traversal_id",
                "canonical_edge_uid",
                "interval_start_time",
                "interval_end_time",
                "observed_travel_time_s",
                "observed_distance_m",
                "observed_speed_mps",
                "measurement_source",
                "label_valid",
            }
        ),
        "interval_measurements": frozenset(
            {
                "order_id",
                "gps_interval_id",
                "interval_start_time",
                "interval_end_time",
                "interval_duration_s",
                "gps_interval_distance_m",
                "measurement_source",
                "direct_observed_travel_time_s",
                "interval_supported_time_s",
                "engine_allocated_only_time_s",
                "unresolved_time_s",
                "direct_observed_distance_m",
            }
        ),
        "turn_movements": frozenset(
            {
                "order_id",
                "movement_sequence",
                "from_edge_uid",
                "via_node",
                "to_edge_uid",
                "movement_source",
                "movement_quality",
                "movement_travel_time_s",
                "movement_delay_s",
                "observed_interval_time_s",
                "dynamic_time_source",
            }
        ),
        "gps_quality": frozenset({"order_id", "gps_status"}),
        "route_quality": frozenset({"order_id", "route_status"}),
        "dynamic_quality": frozenset(
            {
                "order_id",
                "direct_observed_time_s",
                "direct_observed_distance_m",
                "interval_supported_time_s",
                "engine_allocated_only_time_s",
                "unresolved_interval_time_s",
                "total_interval_time_s",
                "time_conservation_error_s",
                "duplicate_interval_allocation_count",
                "non_direct_observed_time_violation_count",
                "unresolved_duplicate_allocation_count",
                "traversal_distance_conservation_error_m",
                "traversal_duplicate_distance_count",
                "time_conservation_valid",
                "distance_conservation_valid",
                "valid_direct_interval_count",
                "unique_timed_edge_count",
                "dynamic_status",
            }
        ),
        "canonical_quality": frozenset({"order_id", "canonical_status"}),
    }
)

OUTPUT_PRIMARY_KEYS = MappingProxyType(
    {
        "interval_labels": ("split", "date", "order_id", "gps_interval_id"),
        "traversal_labels": ("split", "date", "order_id", "traversal_id"),
        "route_sequence_context": (
            "split",
            "date",
            "order_id",
            "route_sequence",
        ),
        "movement_context": (
            "split",
            "date",
            "order_id",
            "movement_sequence",
        ),
        "order_labels": ("split", "date", "order_id"),
        "order_label_quality": ("split", "date", "order_id"),
    }
)

OUTPUT_REQUIRED_COLUMNS = MappingProxyType(
    {
        "interval_labels": frozenset(
            {
                "split",
                "date",
                "order_id",
                "gps_interval_id",
                "traversal_id",
                "route_sequence",
                "canonical_edge_uid",
                "observed_directed_edge_uid",
                "observed_from_node",
                "observed_to_node",
                "observed_direction",
                "synthetic_reverse_edge",
                "osm_direction_disagreement",
                "canonical_mapping_available",
                "mapping_status",
                "osm_oneway",
                "canonical_highway",
                "allocated_distance_m",
                "route_part_distance_m",
                "interval_start_time",
                "interval_end_time",
                "observed_travel_time_s",
                "observed_distance_m",
                "observed_speed_mps",
                "measurement_source",
                "label_valid",
                "interval_duration_s",
                "previous_direct_gps_interval_id",
                "adjacent_gap_s",
                "speed_delta_mps",
                "acceleration_mps2",
                "is_stop",
                "is_crawl",
                "is_low_speed_total",
                "kinematic_sequence_valid",
                "lcs_component_available",
                "lcs_component_unavailable_reason",
                "label_schema_version",
            }
        ),
        "traversal_labels": frozenset(
            {
                "split",
                "date",
                "order_id",
                "traversal_id",
                "route_sequence",
                "canonical_edge_uid",
                "observed_directed_edge_uid",
                "observed_from_node",
                "observed_to_node",
                "observed_direction",
                "synthetic_reverse_edge",
                "osm_direction_disagreement",
                "canonical_mapping_available",
                "mapping_status",
                "osm_oneway",
                "canonical_highway",
                "direct_interval_count",
                "direct_observed_time_s",
                "direct_observed_distance_m",
                "allocated_distance_m",
                "direct_distance_coverage_share",
                "direct_distance_exceeds_allocated",
                "observation_window_start_time",
                "observation_window_end_time",
                "time_weighted_speed_mean_mps",
                "maximum_internal_gap_s",
                "discontinuous_direct_window",
                "crawl_time_share",
                "stop_time_share",
                "time_weighted_speed_cv",
                "speed_cv_bounded",
                "acceleration_pair_count",
                "acceleration_weight_s",
                "acceleration_rms_mps2",
                "acceleration_rms_bounded",
                "maximum_absolute_acceleration_mps2",
                "lcs_available",
                "lcs_unavailable_reason",
                "lcs_raw",
                "lcs_pct",
                "lcs_tail_event",
                "lcs_cdf_level_used",
                "lcs_cdf_sample_size",
                "rts_available",
                "rts_unavailable_reason",
                "rts_measurement_available",
                "rts_measurement_unavailable_reason",
                "rts_raw",
                "rts_pct",
                "rts_tail_event",
                "rts_cdf_level_used",
                "rts_cdf_sample_size",
                "rts_direct_speed_valid",
                "reference_sec_per_m",
                "excess_time_ratio",
                "reference_level_used",
                "reference_sample_size",
                "reference_model_id",
                "reference_fit_manifest_id",
                "edge_observation_count",
                "edge_hour_observation_count",
                "edge_time_bin_30m_observation_count",
                "edge_support_level",
                "edge_hour_support_level",
                "directed_edge_model_scope",
                "observed_sec_per_m",
                "time_bin_30m",
                "weekday_type",
                "peak_offpeak",
                "measurement_source",
                "gns_available",
                "gns_raw",
                "gns_pct",
                "gns_unavailable_reason",
                "iis_available",
                "iis_raw",
                "iis_pct",
                "iis_unavailable_reason",
                "pmis_available",
                "pmis_raw",
                "pmis_pct",
                "pmis_unavailable_reason",
                "label_schema_version",
            }
        ),
        "route_sequence_context": frozenset(
            {
                "split",
                "date",
                "order_id",
                "route_sequence",
                "canonical_edge_uid",
                "observed_directed_edge_uid",
                "observed_from_node",
                "observed_to_node",
                "observed_direction",
                "canonical_mapping_available",
                "route_lineage_status",
                "sequence_feature_mask",
                "directed_edge_model_scope",
                "synthetic_reverse_edge",
                "osm_direction_disagreement",
                "mapping_status",
                "osm_oneway",
                "route_part_length_m",
                "canonical_length_m",
                "canonical_highway",
                "road_class",
                "bridge",
                "tunnel",
            }
        ),
        "movement_context": frozenset(
            {
                "split",
                "date",
                "order_id",
                "movement_sequence",
                "from_edge_uid",
                "observed_from_directed_edge_uid",
                "via_node",
                "to_edge_uid",
                "observed_to_directed_edge_uid",
                "movement_direction_mapping_available",
                "movement_lineage_only",
                "movement_source",
                "movement_quality",
                "iis_available",
                "iis_unavailable_reason",
            }
        ),
        "order_labels": frozenset(
            {
                "split",
                "date",
                "order_id",
                "all_dimension_mask",
                "valid_core_dimension_count",
                "core_composition_signature",
                "composition_signature",
                "core_composite_status",
                "lcs_tail_event_present",
                "rts_tail_event_present",
                *{
                    f"{dimension}_{suffix}"
                    for dimension in ("lcs", "rts", "gns", "iis", "pmis")
                    for suffix in (
                        "mean",
                        "max",
                        "tail",
                        "persistence",
                        "coverage_share",
                        "available",
                        "unavailable_reason",
                    )
                },
            }
        ),
        "order_label_quality": frozenset(
            {
                "split",
                "date",
                "order_id",
                "direct_interval_count",
                "unique_timed_edge_count",
                "direct_observed_time_s",
                "direct_observed_distance_m",
                "order_duration_s",
                "route_distance_m",
                "observed_time_share",
                "observed_distance_share",
                "observed_time_exceeds_order_duration",
                "observed_distance_exceeds_route_distance",
                "direct_coverage_pass",
                "lcs_missing_reason",
                "rts_missing_reason",
                "gns_missing_reason",
                "lcs_coverage_share",
                "rts_coverage_share",
                "gns_coverage_share",
                "iis_coverage_share",
                "pmis_coverage_share",
            }
        ),
    }
)


class ContractError(ValueError):
    """Base class for every fail-closed Stage 1 v3 contract violation."""


class Stage1V3InputError(ContractError):
    """Raised when a Stage 0 v6 bucket violates the Stage 1 v3 contract."""
