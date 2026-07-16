"""Final Stage0 v4 three-tier quality semantics, independent of Core share."""

from __future__ import annotations

from collections.abc import Mapping


HARD_FLAGS = {
    "direction_gap": "core_direction_continuous",
    "unreasonable_detour": "core_no_unreasonable_detour",
    "origin_endpoint": "core_origin_error_ok",
    "destination_endpoint": "core_destination_error_ok",
    "u_turn": "core_u_turn_ok",
    "minimum_route_links": "core_route_link_count_ok",
}

SOFT_FLAGS = {
    "fallback_share": "core_fallback_share_ok",
    "projection_distance": "core_projection_ok",
    "route_length_ratio": "core_route_length_ratio_ok",
    "interpolated_distance_share": "core_interpolation_ok",
    "match_confidence": "core_confidence_ok",
    "repeated_link_share": "core_repeated_link_ok",
}


def _failed(row: Mapping[str, object], definitions: Mapping[str, str]) -> list[str]:
    return [name for name, column in definitions.items() if not bool(row.get(column, False))]


def classify_quality_layer(row: Mapping[str, object]) -> dict[str, object]:
    """Apply frozen hard/soft semantics without tuning to increase Core coverage."""

    hard = _failed(row, HARD_FLAGS)
    soft = _failed(row, SOFT_FLAGS)
    limitations = list(row.get("data_limitation_flags", []) or [])
    if hard:
        quality_class = "rejected"
        weight = 0.0
    elif soft:
        quality_class = "analysis_set"
        weight = max(0.50, 1.0 - 0.08 * len(soft))
    else:
        quality_class = "strict_core"
        weight = 1.0
    return {
        "route_quality_class_v4_final": quality_class,
        "formal_analysis_eligible": not hard,
        "strict_evaluation_eligible": not hard and not soft,
        "quality_weight": weight,
        "hard_error_flags": "|".join(hard),
        "soft_quality_flags": "|".join(soft),
        "data_limitation_flags": "|".join(map(str, limitations)),
        "exclusion_reason": "|".join(hard) if hard else "",
    }


def canonical_promotion_gate(
    *, manual_pass: bool, conservation_pass: bool, connector_pass: bool, full_date_pass: bool
) -> bool:
    """Promotion deliberately has no Core-share threshold."""

    return all((manual_pass, conservation_pass, connector_pass, full_date_pass))
