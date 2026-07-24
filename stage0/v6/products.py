"""Build dynamic products from mutually exclusive raw-GPS intervals."""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd


INTERVAL_COLUMNS = [
    "order_id",
    "gps_interval_id",
    "subtrace_id",
    "from_original_point_seq",
    "to_original_point_seq",
    "interval_start_time",
    "interval_end_time",
    "interval_duration_s",
    "gps_interval_distance_m",
    "from_edge_index",
    "to_edge_index",
    "measurement_source",
    "interval_reason",
    "route_interval_supported",
    "covered_route_sequences_json",
    "direct_observed_travel_time_s",
    "interval_supported_time_s",
    "engine_allocated_only_time_s",
    "unresolved_time_s",
    "interval_route_distance_m",
    "direct_observed_distance_m",
    "time_source",
    "time_observation_valid",
]

UNRESOLVED_COLUMNS = [
    "order_id",
    "unresolved_interval_id",
    "gps_interval_id",
    "subtrace_id",
    "from_original_point_seq",
    "to_original_point_seq",
    "from_timestamp",
    "to_timestamp",
    "from_edge_index",
    "to_edge_index",
    "measurement_source",
    "interval_duration_s",
    "interval_route_distance_m",
    "unresolved_interval_time_s",
    "interval_supported_time_s",
    "interval_time_source",
    "unresolved_reason",
]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=columns)


def _finite(value: Any) -> bool:
    return pd.notna(value) and np.isfinite(float(value))


def _route_sequences(frame: pd.DataFrame) -> list[int]:
    if frame.empty:
        return []
    return [
        int(value)
        for value in pd.to_numeric(frame.route_sequence, errors="coerce")
        .dropna()
        .astype(int)
        .tolist()
    ]


def _interval_row(
    *,
    order_id: str,
    interval_id: int,
    left: dict[str, Any],
    right: dict[str, Any],
    source: str,
    reason: str,
    route_supported: bool,
    covered_sequences: list[int],
    interval_route_distance_m: float,
    direct_sequence: int | None = None,
    direct_distance_m: float = np.nan,
) -> dict[str, Any]:
    duration = max(float(right["timestamp"] - left["timestamp"]), 0.0)
    sequences = covered_sequences
    if direct_sequence is not None:
        sequences = [direct_sequence]
    return {
        "order_id": order_id,
        "gps_interval_id": interval_id,
        "subtrace_id": (
            str(left["subtrace_id"])
            if str(left["subtrace_id"]) == str(right["subtrace_id"])
            else f"{left['subtrace_id']}->{right['subtrace_id']}"
        ),
        "from_original_point_seq": int(left["original_point_seq"]),
        "to_original_point_seq": int(right["original_point_seq"]),
        "interval_start_time": float(left["timestamp"]),
        "interval_end_time": float(right["timestamp"]),
        "interval_duration_s": duration,
        "gps_interval_distance_m": float(right.get("step_distance_m", 0.0)),
        "from_edge_index": left.get("edge_index", pd.NA),
        "to_edge_index": right.get("edge_index", pd.NA),
        "measurement_source": source,
        "interval_reason": reason,
        "route_interval_supported": bool(route_supported),
        "covered_route_sequences_json": json.dumps(sequences),
        "direct_observed_travel_time_s": (
            duration if source == "direct_observed" else np.nan
        ),
        "interval_supported_time_s": (
            duration if source == "interval_supported" else np.nan
        ),
        "engine_allocated_only_time_s": (
            duration if source == "engine_allocated" else np.nan
        ),
        "unresolved_time_s": duration if source == "unresolved" else np.nan,
        "interval_route_distance_m": interval_route_distance_m,
        "direct_observed_distance_m": (
            float(direct_distance_m) if source == "direct_observed" else np.nan
        ),
        "time_source": (
            "raw_gps_interval"
            if source in {"direct_observed", "interval_supported", "unresolved"}
            else "valhalla_engine_allocation"
        ),
        "time_observation_valid": source == "direct_observed",
    }


def _direct_distance(
    left: dict[str, Any], right: dict[str, Any], route: pd.Series
) -> float:
    left_position = float(left["percent_along"])
    right_position = float(right["percent_along"])
    span = float(route.get("target_percent_along", 1.0)) - float(
        route.get("source_percent_along", 0.0)
    )
    if span > 0:
        return max(
            float(route.get("length_m", 0.0))
            * max(right_position - left_position, 0.0)
            / span,
            0.0,
        )
    return max(float(right.get("step_distance_m", 0.0)), 0.0)


def build_order_products(
    source_points: pd.DataFrame,
    matched_points: pd.DataFrame,
    route_parts: pd.DataFrame,
    *,
    preprocess_breaks: pd.DataFrame | None = None,
    position_backtrack_tolerance: float = 0.01,
    enable_engine_allocation: bool = False,
) -> dict[str, pd.DataFrame]:
    """Build products without assigning multi-edge GPS time to individual links."""

    if source_points.empty:
        raise ValueError("source_points cannot be empty")
    order_id = str(source_points.order_id.iloc[0])
    source = source_points.sort_values(
        ["timestamp", "original_point_seq"], kind="stable"
    ).reset_index(drop=True)
    primary_routes = route_parts.loc[
        route_parts.get("path_id", pd.Series(index=route_parts.index, dtype=float))
        .fillna(0)
        .eq(0)
    ].copy()
    primary_routes.sort_values(
        ["subtrace_id", "route_sequence"], kind="stable", inplace=True
    )
    routes_by_subtrace: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for route in primary_routes.to_dict("records"):
        numeric_edge_index = pd.to_numeric(
            pd.Series([route.get("valhalla_edge_index")]), errors="coerce"
        ).iloc[0]
        route["_edge_index_numeric"] = (
            int(numeric_edge_index) if pd.notna(numeric_edge_index) else None
        )
        routes_by_subtrace[str(route.get("subtrace_id"))].append(route)
    route_range_cache: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    route_range_meta: dict[tuple[str, int, int], tuple[list[int], float, bool]] = {}

    match_lookup: dict[tuple[str, int], dict[str, Any]] = {}
    for row in matched_points.to_dict("records"):
        match_lookup[
            (str(row["subtrace_id"]), int(row["original_point_seq"]))
        ] = row
    break_lookup: dict[tuple[int, int], str] = {}
    if preprocess_breaks is not None and len(preprocess_breaks):
        for row in preprocess_breaks.itertuples(index=False):
            break_lookup[
                (int(row.from_original_point_seq), int(row.to_original_point_seq))
            ] = str(row.break_reason)

    intervals: list[dict[str, Any]] = []
    direct_intervals_by_sequence: dict[int, list[dict[str, Any]]] = defaultdict(list)
    supported_sequences: set[int] = set()
    interval_sequences: dict[int, list[int]] = {}

    source_records = source.to_dict("records")
    for interval_id in range(max(len(source_records) - 1, 0)):
        left_source = source_records[interval_id]
        right_source = source_records[interval_id + 1]
        pair = (
            int(left_source["original_point_seq"]),
            int(right_source["original_point_seq"]),
        )
        left = match_lookup.get(
            (
                str(left_source["subtrace_id"]),
                int(left_source["original_point_seq"]),
            )
        )
        right = match_lookup.get(
            (
                str(right_source["subtrace_id"]),
                int(right_source["original_point_seq"]),
            )
        )
        if left is None:
            left = left_source.copy()
            left["matched_point_status"] = "unmatched"
            left["edge_index"] = pd.NA
            left["route_discontinuity"] = False
            left["percent_along"] = np.nan
        else:
            left = {
                **left_source,
                **left,
                "step_distance_m": left_source.get("step_distance_m", 0.0),
                "usable_subtrace": left_source.get("usable_subtrace", True),
            }
        if right is None:
            right = right_source.copy()
            right["matched_point_status"] = "unmatched"
            right["edge_index"] = pd.NA
            right["route_discontinuity"] = False
            right["percent_along"] = np.nan
        else:
            right = {
                **right_source,
                **right,
                "step_distance_m": right_source.get("step_distance_m", 0.0),
                "usable_subtrace": right_source.get("usable_subtrace", True),
            }

        candidates: list[dict[str, Any]] = []
        route_supported = False
        source_type = "unresolved"
        reason = ""
        direct_sequence: int | None = None
        direct_distance = np.nan
        duration = float(right["timestamp"] - left["timestamp"])
        same_subtrace = str(left["subtrace_id"]) == str(right["subtrace_id"])

        if pair in break_lookup or not same_subtrace:
            reason = break_lookup.get(pair, "preprocess_split")
        elif not bool(left_source.get("usable_subtrace", True)):
            reason = "unusable_short_subtrace"
        elif duration <= 0:
            reason = "nonpositive_gps_interval"
        elif (
            str(left["matched_point_status"]) == "unmatched"
            or str(right["matched_point_status"]) == "unmatched"
        ):
            reason = "unmatched_endpoint"
        elif bool(left.get("route_discontinuity", False)) or bool(
            right.get("route_discontinuity", False)
        ):
            reason = "valhalla_route_discontinuity"
        elif pd.isna(left.get("edge_index")) or pd.isna(right.get("edge_index")):
            reason = "missing_edge_index"
        elif int(right["edge_index"]) < int(left["edge_index"]):
            reason = "nonmonotonic_edge_index"
        else:
            from_edge, to_edge = int(left["edge_index"]), int(right["edge_index"])
            cache_key = (str(left["subtrace_id"]), from_edge, to_edge)
            if cache_key not in route_range_cache:
                cached = [
                    route
                    for route in routes_by_subtrace.get(
                        str(left["subtrace_id"]), []
                    )
                    if route["_edge_index_numeric"] is not None
                    and from_edge <= route["_edge_index_numeric"] <= to_edge
                ]
                route_range_cache[cache_key] = cached
                route_range_meta[cache_key] = (
                    [int(route["route_sequence"]) for route in cached],
                    float(sum(float(route.get("length_m", 0.0)) for route in cached)),
                    bool(
                        any(
                            str(route.get("route_source")) == "inferred"
                            or bool(route.get("is_interpolated", False))
                            for route in cached
                        )
                    ),
                )
            candidates = route_range_cache[cache_key]
            candidate_sequences, candidate_distance, candidate_inferred = (
                route_range_meta[cache_key]
            )
            route_supported = bool(candidates)
            if not candidates:
                reason = "missing_route_parts"
            elif from_edge == to_edge:
                exact = candidates
                canonical_count = len(
                    {
                        str(route.get("canonical_edge_uid"))
                        for route in exact
                        if pd.notna(route.get("canonical_edge_uid"))
                    }
                )
                positions_valid = _finite(left.get("percent_along")) and _finite(
                    right.get("percent_along")
                )
                direction_valid = positions_valid and (
                    float(right["percent_along"]) + position_backtrack_tolerance
                    >= float(left["percent_along"])
                )
                endpoints_direct = (
                    str(left["matched_point_status"]) == "matched"
                    and str(right["matched_point_status"]) == "matched"
                )
                if (
                    len(exact) == 1
                    and canonical_count == 1
                    and endpoints_direct
                    and direction_valid
                ):
                    route = exact[0]
                    direct_sequence = int(route["route_sequence"])
                    direct_distance = _direct_distance(left, right, route)
                    source_type = "direct_observed"
                    reason = "same_canonical_edge_gps_pair"
                elif not endpoints_direct:
                    reason = "engine_interpolated_endpoint"
                elif not direction_valid:
                    reason = "same_edge_position_backtrack"
                else:
                    reason = "same_valhalla_edge_not_unique_canonical_edge"
            else:
                if candidate_inferred:
                    reason = "inferred_path_between_gps_anchors"
                else:
                    source_type = "interval_supported"
                    reason = "multi_edge_interval_without_direct_timing"

        row = _interval_row(
            order_id=order_id,
            interval_id=interval_id,
            left=left,
            right=right,
            source=source_type,
            reason=reason,
            route_supported=route_supported,
            covered_sequences=(candidate_sequences if candidates else []),
            interval_route_distance_m=(
                candidate_distance if candidates else 0.0
            ),
            direct_sequence=direct_sequence,
            direct_distance_m=direct_distance,
        )
        intervals.append(row)
        sequences = json.loads(row["covered_route_sequences_json"])
        interval_sequences[interval_id] = sequences
        if source_type == "direct_observed" and direct_sequence is not None:
            direct_intervals_by_sequence[direct_sequence].append(row)
        elif source_type == "interval_supported":
            supported_sequences.update(sequences)

    interval_frame = pd.DataFrame(intervals, columns=INTERVAL_COLUMNS)

    route_output = primary_routes.copy().reset_index(drop=True)
    if route_output.empty:
        route_output = route_parts.copy().reset_index(drop=True)
    route_sources: list[str] = []
    direct_time: list[float] = []
    direct_distance: list[float] = []
    enter_times: list[float] = []
    exit_times: list[float] = []
    anchor_valid: list[bool] = []
    for route in route_output.itertuples(index=False):
        sequence = int(route.route_sequence)
        assigned = direct_intervals_by_sequence.get(sequence, [])
        if assigned:
            measurement_source = "direct_observed"
            observed_time = float(
                sum(item["direct_observed_travel_time_s"] for item in assigned)
            )
            observed_distance = float(
                sum(item["direct_observed_distance_m"] for item in assigned)
            )
            enter_time = float(min(item["interval_start_time"] for item in assigned))
            exit_time = float(max(item["interval_end_time"] for item in assigned))
            valid = abs((exit_time - enter_time) - observed_time) <= 1e-6
        elif str(getattr(route, "route_source", "")) == "inferred" or bool(
            getattr(route, "is_interpolated", False)
        ):
            measurement_source = "engine_interpolated"
            observed_time = observed_distance = enter_time = exit_time = np.nan
            valid = True
        elif sequence in supported_sequences:
            measurement_source = "interval_supported"
            observed_time = observed_distance = enter_time = exit_time = np.nan
            valid = True
        else:
            measurement_source = "unresolved"
            observed_time = observed_distance = enter_time = exit_time = np.nan
            valid = True
        route_sources.append(measurement_source)
        direct_time.append(observed_time)
        direct_distance.append(observed_distance)
        enter_times.append(enter_time)
        exit_times.append(exit_time)
        anchor_valid.append(valid)

    route_output["measurement_source"] = route_sources
    route_output["observed_travel_time_s"] = direct_time
    route_output["observed_distance_m"] = direct_distance
    route_output["observed_speed_mps"] = np.where(
        pd.Series(direct_time).gt(0),
        pd.Series(direct_distance) / pd.Series(direct_time),
        np.nan,
    )
    if "engine_allocated_travel_time_s" not in route_output:
        route_output["engine_allocated_travel_time_s"] = np.nan
    if not enable_engine_allocation:
        route_output["engine_allocated_travel_time_s"] = np.nan
    route_output["travel_time_s"] = route_output["observed_travel_time_s"].where(
        route_output.measurement_source.eq("direct_observed"),
        route_output["engine_allocated_travel_time_s"].where(
            route_output.measurement_source.eq("engine_allocated")
        ),
    )
    route_output["time_source"] = np.where(
        route_output.measurement_source.eq("direct_observed"),
        "raw_gps_interval",
        np.where(
            route_output.measurement_source.eq("engine_allocated"),
            "valhalla_engine_allocation",
            pd.NA,
        ),
    )
    route_output["time_observation_valid"] = anchor_valid
    route_output["enter_time"] = enter_times
    route_output["exit_time"] = exit_times

    observed_point_counts = (
        matched_points.loc[matched_points.matched_point_status.eq("matched")]
        .groupby(["subtrace_id", "edge_index"], dropna=True)
        .size()
        .to_dict()
        if len(matched_points)
        else {}
    )
    traversal_rows: list[dict[str, Any]] = []
    for route in route_output.itertuples(index=False):
        link_id = (
            route.canonical_edge_uid
            if pd.notna(route.canonical_edge_uid)
            else f"valhalla:{route.valhalla_edge_id}"
        )
        assigned_intervals = direct_intervals_by_sequence.get(
            int(route.route_sequence), []
        )
        observations = assigned_intervals or [None]
        for observation_sequence, interval in enumerate(observations):
            if interval is not None:
                enter_time = interval["interval_start_time"]
                exit_time = interval["interval_end_time"]
                observed_time = interval["direct_observed_travel_time_s"]
                observed_distance = interval["direct_observed_distance_m"]
                observed_speed = (
                    observed_distance / observed_time
                    if observed_time > 0
                    else np.nan
                )
                measurement_source = "direct_observed"
                time_source = "raw_gps_interval"
                travel_time = observed_time
                observation_valid = True
            else:
                enter_time = exit_time = observed_time = observed_distance = np.nan
                observed_speed = np.nan
                measurement_source = route.measurement_source
                time_source = route.time_source
                travel_time = route.travel_time_s
                observation_valid = bool(route.time_observation_valid)
            traversal_rows.append(
                {
                    "order_id": order_id,
                    "subtrace_id": str(route.subtrace_id),
                    "traversal_id": len(traversal_rows),
                    "traversal_observation_sequence": observation_sequence,
                    "route_sequence": int(route.route_sequence),
                    "edge_uid": link_id,
                    "canonical_edge_uid": route.canonical_edge_uid,
                    "valhalla_edge_id": route.valhalla_edge_id,
                    "enter_time": enter_time,
                    "exit_time": exit_time,
                    "observed_travel_time_s": observed_time,
                    "engine_allocated_travel_time_s": (
                        route.engine_allocated_travel_time_s
                    ),
                    "travel_time_s": travel_time,
                    "time_source": time_source,
                    "time_observation_valid": observation_valid,
                    "measurement_source": measurement_source,
                    "entry_position_m": getattr(route, "entry_position_m", np.nan),
                    "exit_position_m": getattr(route, "exit_position_m", np.nan),
                    "observed_distance_m": observed_distance,
                    "allocated_distance_m": float(route.length_m),
                    "observed_speed_mps": observed_speed,
                    "observed_point_count": int(
                        observed_point_counts.get(
                            (str(route.subtrace_id), route.valhalla_edge_index), 0
                        )
                    ),
                    "traversal_source": measurement_source,
                    "traversal_quality": route.mapping_status,
                    "is_interpolated": bool(route.is_interpolated),
                    "interpolated_distance_share": (
                        1.0 if measurement_source == "engine_interpolated" else 0.0
                    ),
                    "observed_interval_time_s": observed_time,
                    "valhalla_edge_elapsed_time_s": getattr(
                        route, "valhalla_edge_elapsed_time_s", np.nan
                    ),
                }
            )
    traversals = pd.DataFrame(traversal_rows)

    movement_rows: list[dict[str, Any]] = []
    directly_observed_transitions: set[tuple[int, int]] = set()
    path_supported_transitions: set[tuple[int, int]] = set()
    for row in intervals:
        if row["measurement_source"] != "interval_supported":
            continue
        sequences = interval_sequences[row["gps_interval_id"]]
        pairs = set(zip(sequences, sequences[1:]))
        path_supported_transitions.update(pairs)
        if len(sequences) == 2:
            directly_observed_transitions.add((sequences[0], sequences[1]))
    for (_, _), group in route_output.groupby(
        ["subtrace_id", "path_id"], sort=False, dropna=False
    ):
        rows = list(group.sort_values("route_sequence", kind="stable").itertuples(index=False))
        for left, right in zip(rows, rows[1:]):
            left_sequence, right_sequence = int(left.route_sequence), int(right.route_sequence)
            mapped = pd.notna(left.canonical_edge_uid) and pd.notna(
                right.canonical_edge_uid
            )
            pair = (left_sequence, right_sequence)
            direct_transition = pair in directly_observed_transitions
            path_supported = pair in path_supported_transitions
            if not mapped:
                movement_source = "unmapped_transition"
            elif direct_transition:
                movement_source = "directly_observed_transition"
            elif path_supported:
                movement_source = "path_supported_transition"
            elif (
                left.measurement_source == "engine_interpolated"
                or right.measurement_source == "engine_interpolated"
            ):
                movement_source = "engine_inferred_transition"
            else:
                movement_source = "path_supported_transition"
            continuous = (
                mapped
                and pd.notna(left.canonical_to_node)
                and pd.notna(right.canonical_from_node)
                and int(left.canonical_to_node) == int(right.canonical_from_node)
            )
            movement_rows.append(
                {
                    "order_id": order_id,
                    "subtrace_id": left.subtrace_id,
                    "movement_sequence": len(movement_rows),
                    "from_edge_uid": (
                        left.canonical_edge_uid
                        if pd.notna(left.canonical_edge_uid)
                        else f"valhalla:{left.valhalla_edge_id}"
                    ),
                    "to_edge_uid": (
                        right.canonical_edge_uid
                        if pd.notna(right.canonical_edge_uid)
                        else f"valhalla:{right.valhalla_edge_id}"
                    ),
                    "movement_source": movement_source,
                    "movement_observed": movement_source
                    == "directly_observed_transition",
                    "movement_travel_time_s": np.nan,
                    "movement_delay_s": np.nan,
                    "observed_interval_time_s": np.nan,
                    "dynamic_time_source": pd.NA,
                    "via_node": left.canonical_to_node,
                    "movement_type": (
                        "continuous" if continuous else "valhalla_transition"
                    ),
                    "restriction_status": "valhalla_legal",
                    "movement_quality": (
                        "mapped" if continuous else "engine_only"
                    ),
                }
            )
    movements = pd.DataFrame(movement_rows)

    unresolved_rows = []
    for row in intervals:
        if row["measurement_source"] == "direct_observed":
            continue
        unresolved_rows.append(
            {
                "order_id": order_id,
                "unresolved_interval_id": len(unresolved_rows),
                "gps_interval_id": row["gps_interval_id"],
                "subtrace_id": row["subtrace_id"],
                "from_original_point_seq": row["from_original_point_seq"],
                "to_original_point_seq": row["to_original_point_seq"],
                "from_timestamp": row["interval_start_time"],
                "to_timestamp": row["interval_end_time"],
                "from_edge_index": row["from_edge_index"],
                "to_edge_index": row["to_edge_index"],
                "measurement_source": row["measurement_source"],
                "interval_duration_s": row["interval_duration_s"],
                "interval_route_distance_m": row["interval_route_distance_m"],
                "unresolved_interval_time_s": row["unresolved_time_s"],
                "interval_supported_time_s": row["interval_supported_time_s"],
                "interval_time_source": row["time_source"],
                "unresolved_reason": row["interval_reason"],
            }
        )
    unresolved_frame = pd.DataFrame(unresolved_rows, columns=UNRESOLVED_COLUMNS)

    direct_time_total = float(
        pd.to_numeric(
            interval_frame.direct_observed_travel_time_s, errors="coerce"
        ).fillna(0).sum()
    )
    supported_time_total = float(
        pd.to_numeric(
            interval_frame.interval_supported_time_s, errors="coerce"
        ).fillna(0).sum()
    )
    engine_time_total = float(
        pd.to_numeric(
            interval_frame.engine_allocated_only_time_s, errors="coerce"
        ).fillna(0).sum()
    )
    unresolved_time_total = float(
        pd.to_numeric(interval_frame.unresolved_time_s, errors="coerce")
        .fillna(0)
        .sum()
    )
    total_time = float(
        pd.to_numeric(interval_frame.interval_duration_s, errors="coerce")
        .fillna(0)
        .sum()
    )
    conservation_error = (
        direct_time_total
        + supported_time_total
        + engine_time_total
        + unresolved_time_total
        - total_time
    )
    timed = traversals.loc[
        pd.to_numeric(
            traversals.get("observed_travel_time_s", pd.Series(dtype=float)),
            errors="coerce",
        ).notna()
    ]
    anchor_failures = int(
        (
            timed.enter_time.isna()
            | timed.exit_time.isna()
            | (
                (
                    pd.to_numeric(timed.exit_time, errors="coerce")
                    - pd.to_numeric(timed.enter_time, errors="coerce")
                    - pd.to_numeric(timed.observed_travel_time_s, errors="coerce")
                ).abs()
                > 1e-6
            )
        ).sum()
    ) if len(timed) else 0
    inferred_time_violations = int(
        (
            traversals.measurement_source.eq("engine_interpolated")
            & pd.to_numeric(
                traversals.observed_travel_time_s, errors="coerce"
            ).notna()
        ).sum()
    ) if len(traversals) else 0
    raw_distance = float(
        pd.to_numeric(source.step_distance_m, errors="coerce").fillna(0).sum()
    )
    resolved_subtrace_distance = float(
        pd.to_numeric(
            source.loc[
                ~source.get(
                    "preprocess_break_before",
                    pd.Series(False, index=source.index),
                ).fillna(False),
                "step_distance_m",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    unresolved_gap_distance = float(
        pd.to_numeric(
            interval_frame.loc[
                interval_frame.measurement_source.eq("unresolved"),
                "gps_interval_distance_m",
            ],
            errors="coerce",
        ).fillna(0).sum()
    )
    direct_distance_total = float(
        pd.to_numeric(
            interval_frame.direct_observed_distance_m, errors="coerce"
        ).fillna(0).sum()
    )
    accounting = pd.DataFrame(
        [
            {
                "order_id": order_id,
                "direct_observed_time_s": direct_time_total,
                "interval_supported_time_s": supported_time_total,
                "engine_allocated_only_time_s": engine_time_total,
                "unresolved_interval_time_s": unresolved_time_total,
                "total_interval_time_s": total_time,
                "time_conservation_error_s": conservation_error,
                "time_conservation_valid": abs(conservation_error) <= 1e-6,
                "timestamp_anchor_failure_count": anchor_failures,
                "timestamp_anchor_valid": anchor_failures == 0,
                "unresolved_duplicate_allocation_count": 0,
                "inferred_edge_observed_time_violation_count": inferred_time_violations,
                "valid_timed_traversal_count": int(len(timed)),
                "timed_traversal_share": (
                    float(len(timed) / len(traversals)) if len(traversals) else 0.0
                ),
                "direct_observed_distance_m": direct_distance_total,
                "raw_order_gps_distance_m": raw_distance,
                "resolved_subtrace_gps_distance_m": resolved_subtrace_distance,
                "unresolved_gap_distance_m": unresolved_gap_distance,
            }
        ]
    )
    return {
        "route_parts": route_output.reset_index(drop=True),
        "link_traversals": traversals.reset_index(drop=True),
        "turn_movements": movements.reset_index(drop=True),
        "unresolved_intervals": unresolved_frame.reset_index(drop=True),
        "interval_measurements": interval_frame.reset_index(drop=True),
        "interval_accounting": accounting,
    }
