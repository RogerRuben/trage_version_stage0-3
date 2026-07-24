"""Pure order-level route-quality metrics and threshold application."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import numpy as np


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    if not all(np.isfinite([lon1, lat1, lon2, lat2])):
        return math.nan
    radius = 6_371_008.8
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi, dlambda = phi2 - phi1, math.radians(lon2 - lon1)
    value = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.asin(min(1.0, math.sqrt(value)))


def route_sequence_metrics(
    link_ids: Sequence[str],
    interpolated: Sequence[bool],
    link_lengths_m: Mapping[str, float],
    source_link_ids: Mapping[str, str],
    gps_length_m: float,
) -> dict[str, float | int]:
    links = [str(value) for value in link_ids]
    flags = np.asarray(interpolated, dtype=bool)
    lengths = np.asarray([max(0.0, float(link_lengths_m.get(link, 0.0))) for link in links])
    route_length = float(lengths.sum())
    interpolated_distance = float(lengths[flags].sum()) if len(flags) else 0.0
    source_links = [str(source_link_ids.get(link, link)) for link in links]
    repeated = max(0, len(links) - len(set(links)))
    u_turn_count = sum(
        source_links[index] == source_links[index + 2]
        and source_links[index] != source_links[index + 1]
        for index in range(max(0, len(source_links) - 2))
    )
    return {
        "interpolated_link_count": int(flags.sum()),
        "interpolated_link_share": float(flags.mean()) if len(flags) else 0.0,
        "interpolated_distance_m": interpolated_distance,
        "interpolated_distance_share": interpolated_distance / route_length if route_length > 0 else 0.0,
        "matched_route_length_m": route_length,
        "route_length_ratio": route_length / gps_length_m if gps_length_m > 0 else math.nan,
        "u_turn_count": int(u_turn_count),
        "repeated_link_count": int(repeated),
        "repeated_link_share": repeated / len(links) if links else 0.0,
    }


def projection_metrics(
    projection_distances_m: Sequence[float],
    source_lon: Sequence[float],
    source_lat: Sequence[float],
    projected_lon: Sequence[float],
    projected_lat: Sequence[float],
) -> dict[str, float]:
    distances = np.asarray(projection_distances_m, dtype=float)
    finite = distances[np.isfinite(distances)]
    quantiles = np.quantile(finite, [0.5, 0.9, 0.95]) if len(finite) else [math.nan] * 3
    return {
        "p50_projection_distance_m": float(quantiles[0]),
        "p90_projection_distance_m": float(quantiles[1]),
        "p95_projection_distance_m": float(quantiles[2]),
        "origin_projection_error_m": haversine_m(
            float(source_lon[0]), float(source_lat[0]), float(projected_lon[0]), float(projected_lat[0])
        ),
        "destination_projection_error_m": haversine_m(
            float(source_lon[-1]), float(source_lat[-1]), float(projected_lon[-1]), float(projected_lat[-1])
        ),
    }


def core_threshold_flags(row: Mapping[str, object], config: Mapping[str, object]) -> dict[str, bool]:
    def number(name: str) -> float:
        try:
            return float(row[name])
        except (KeyError, TypeError, ValueError):
            return math.nan

    flags = {
        "core_direction_continuous": number("direction_gap_count") <= float(config["maximum_direction_gaps"]),
        "core_no_unreasonable_detour": number("unreasonable_detour_count") <= float(config["maximum_unreasonable_detour_count"]),
        "core_fallback_share_ok": number("fallback_point_share") <= float(config["maximum_fallback_point_share"]),
        "core_projection_ok": number("p90_projection_distance_m") <= float(config["maximum_p90_projection_distance_m"]),
        "core_route_length_ratio_ok": (
            float(config["minimum_route_length_ratio"])
            <= number("route_length_ratio")
            <= float(config["maximum_route_length_ratio"])
        ),
        "core_interpolation_ok": number("interpolated_distance_share") <= float(config["maximum_interpolated_distance_share"]),
        "core_origin_error_ok": number("origin_projection_error_m") <= float(config["maximum_od_projection_error_m"]),
        "core_destination_error_ok": number("destination_projection_error_m") <= float(config["maximum_od_projection_error_m"]),
        "core_confidence_ok": number("mean_match_confidence") >= float(config["minimum_mean_match_confidence"]),
        "core_u_turn_ok": number("u_turn_count") <= float(config["maximum_u_turn_count"]),
        "core_repeated_link_ok": number("repeated_link_share") <= float(config["maximum_repeated_link_share"]),
    }
    flags["core_all_thresholds_pass"] = all(flags.values())
    return flags
