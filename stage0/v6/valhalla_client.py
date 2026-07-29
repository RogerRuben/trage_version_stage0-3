"""Persistent Python/HTTP clients for Valhalla ``trace_attributes``."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests


TRACE_ATTRIBUTES = [
    "edge.id",
    "edge.way_id",
    "edge.length",
    "edge.road_class",
    "edge.bridge",
    "edge.tunnel",
    "edge.speed_limit",
    "edge.forward",
    "edge.begin_osm_node_id",
    "edge.end_osm_node_id",
    "node.elapsed_time",
    "matched.point",
    "matched.type",
    "matched.edge_index",
    "matched.begin_route_discontinuity",
    "matched.end_route_discontinuity",
    "matched.distance_along_edge",
    "matched.distance_from_trace_point",
]


@dataclass(frozen=True)
class MatchResult:
    status: str
    raw_response: dict[str, Any] | None
    request: dict[str, Any]
    backend: str
    response_ms: float
    request_point_count: int
    matched_edge_count: int
    matched_point_count: int
    unmatched_point_count: int
    interpolated_point_count: int
    discontinuity_count: int
    retry_count: int
    error_code: str | None = None
    error_message: str | None = None
    soft_timeout_exceeded: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


class ValhallaMatcher:
    """Create one Actor or Session and reuse it for every order."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        valhalla_config_path: str | Path | None = None,
        actor: Any | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.config = dict(config)
        self.backend = str(self.config.get("backend", "python")).lower()
        self.timeout_s = float(self.config.get("request_timeout_s", 120))
        self.actor = None
        self.session = None
        if self.backend == "python":
            if actor is not None:
                self.actor = actor
            else:
                from valhalla import Actor

                if valhalla_config_path is None:
                    raise ValueError("valhalla_config_path is required for the python backend")
                self.actor = Actor(str(Path(valhalla_config_path).resolve()))
        elif self.backend == "http":
            self.session = session or requests.Session()
        else:
            raise ValueError(f"unsupported Valhalla backend: {self.backend}")

    def _request(
        self,
        points: pd.DataFrame,
        search_radius_m: float,
        *,
        ignore_oneways: bool | None = None,
    ) -> dict[str, Any]:
        lon_column = "matching_lon" if "matching_lon" in points else "lon"
        lat_column = "matching_lat" if "matching_lat" in points else "lat"
        shape = [
            {"lat": float(lat), "lon": float(lon), "time": int(timestamp)}
            for lon, lat, timestamp in points[
                [lon_column, lat_column, "timestamp"]
            ].itertuples(index=False, name=None)
        ]
        costing = str(self.config.get("costing", "auto"))
        request = {
            "shape": shape,
            "costing": costing,
            "shape_match": str(self.config.get("shape_match", "map_snap")),
            "use_timestamps": True,
            "trace_options": {
                "search_radius": float(search_radius_m),
                "gps_accuracy": float(self.config.get("gps_accuracy_m", 20)),
                "breakage_distance": float(self.config.get("breakage_distance_m", 1000)),
                "interpolation_distance": float(
                    self.config.get("interpolation_distance_m", 10)
                ),
            },
            "filters": {"action": "include", "attributes": TRACE_ATTRIBUTES},
        }
        if ignore_oneways is None:
            ignore_oneways = bool(self.config.get("ignore_oneways", False))
        if ignore_oneways:
            request["costing_options"] = {costing: {"ignore_oneways": True}}
        return request

    def _call(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.backend == "python":
            assert self.actor is not None
            response = self.actor.trace_attributes(request)
        else:
            assert self.session is not None
            url = str(self.config.get("service_url", "http://127.0.0.1:8002")).rstrip("/")
            http_response = self.session.post(
                f"{url}/trace_attributes", json=request, timeout=self.timeout_s
            )
            http_response.raise_for_status()
            response = http_response.json()
        if not isinstance(response, dict):
            raise TypeError(f"Valhalla returned {type(response).__name__}, expected object")
        return response

    @staticmethod
    def _response_counts(response: dict[str, Any]) -> dict[str, int]:
        paths = [response, *list(response.get("alternate_paths") or [])]
        edges = sum(len(path.get("edges") or []) for path in paths if isinstance(path, dict))
        matched_points = list(response.get("matched_points") or [])
        statuses = [str(point.get("type", "unmatched")) for point in matched_points]
        discontinuities = sum(
            bool(point.get("begin_route_discontinuity"))
            + bool(point.get("end_route_discontinuity"))
            for point in matched_points
        )
        return {
            "matched_edge_count": int(edges),
            "matched_point_count": int(sum(status == "matched" for status in statuses)),
            "unmatched_point_count": int(sum(status == "unmatched" for status in statuses)),
            "interpolated_point_count": int(
                sum(status == "interpolated" for status in statuses)
            ),
            "discontinuity_count": int(discontinuities),
        }

    def _match_candidate(
        self, points: pd.DataFrame, *, ignore_oneways: bool
    ) -> dict[str, Any]:
        radii = [float(self.config.get("search_radius_m", 80))]
        retry_radius = float(self.config.get("retry_search_radius_m", radii[0]))
        if bool(self.config.get("controlled_retry", True)) and retry_radius > radii[0]:
            radii.append(retry_radius)
        started = time.perf_counter()
        final_request: dict[str, Any] = {}
        last_error: Exception | None = None
        for attempt, radius in enumerate(radii):
            final_request = self._request(
                points, radius, ignore_oneways=ignore_oneways
            )
            try:
                response = self._call(final_request)
                counts = self._response_counts(response)
                if counts["matched_edge_count"] == 0 and attempt + 1 < len(radii):
                    continue
                elapsed_ms = (time.perf_counter() - started) * 1000
                return MatchResult(
                    status="success" if counts["matched_edge_count"] else "unmatched",
                    raw_response=response,
                    request=final_request,
                    backend=self.backend,
                    response_ms=elapsed_ms,
                    request_point_count=len(points),
                    retry_count=attempt,
                    soft_timeout_exceeded=elapsed_ms > self.timeout_s * 1000,
                    **counts,
                ).as_dict()
            except Exception as exc:  # backend errors are normalized at this boundary
                last_error = exc
                if attempt + 1 < len(radii):
                    continue
        elapsed_ms = (time.perf_counter() - started) * 1000
        error_code = getattr(last_error, "status_code", None)
        return MatchResult(
            status="error",
            raw_response=None,
            request=final_request,
            backend=self.backend,
            response_ms=elapsed_ms,
            request_point_count=len(points),
            matched_edge_count=0,
            matched_point_count=0,
            unmatched_point_count=len(points),
            interpolated_point_count=0,
            discontinuity_count=0,
            retry_count=max(0, len(radii) - 1),
            error_code=str(error_code) if error_code is not None else type(last_error).__name__,
            error_message=str(last_error),
            soft_timeout_exceeded=elapsed_ms > self.timeout_s * 1000,
        ).as_dict()

    @staticmethod
    def _candidate_quality(result: dict[str, Any]) -> dict[str, float | int]:
        response = result.get("raw_response") or {}
        edges = list(response.get("edges") or [])
        topology_gaps = 0
        for left, right in zip(edges, edges[1:]):
            left_end = (left.get("end_node") or {}).get("node_id")
            right_begin = right.get("node_id")
            if (
                left_end is None
                or right_begin is None
                or int(left_end) != int(right_begin)
            ):
                topology_gaps += 1
        distances = pd.to_numeric(
            pd.Series(
                [
                    point.get("distance_from_trace_point")
                    for point in response.get("matched_points") or []
                    if str(point.get("type", "unmatched")) != "unmatched"
                ]
            ),
            errors="coerce",
        ).dropna()
        return {
            "topology_gaps": topology_gaps,
            "discontinuities": int(result.get("discontinuity_count", 0) or 0),
            "snap_p90_m": (
                float(distances.quantile(0.90)) if len(distances) else float("inf")
            ),
            "snap_max_m": float(distances.max()) if len(distances) else float("inf"),
        }

    def _candidate_is_risky(self, quality: dict[str, float | int]) -> bool:
        return bool(
            int(quality["topology_gaps"]) > 0
            or int(quality["discontinuities"]) > 0
            or float(quality["snap_p90_m"])
            > float(self.config.get("oneway_compare_snap_p90_m", 12.0))
            or float(quality["snap_max_m"])
            > float(self.config.get("oneway_compare_snap_max_m", 40.0))
        )

    def _select_candidate(
        self,
        preferred: dict[str, Any],
        alternative: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        preferred_quality = self._candidate_quality(preferred)
        alternative_quality = self._candidate_quality(alternative)
        preferred_ok = preferred.get("status") == "success"
        alternative_ok = alternative.get("status") == "success"
        if preferred_ok != alternative_ok:
            return (
                (preferred, "preferred") if preferred_ok else (alternative, "alternative")
            )
        for key in ("topology_gaps", "discontinuities"):
            left = int(preferred_quality[key])
            right = int(alternative_quality[key])
            if left != right:
                return (
                    (preferred, "preferred")
                    if left < right
                    else (alternative, "alternative")
                )
        minimum_improvement = float(
            self.config.get("oneway_selection_minimum_snap_improvement_m", 2.0)
        )
        preferred_snap = float(preferred_quality["snap_p90_m"])
        alternative_snap = float(alternative_quality["snap_p90_m"])
        if (
            np.isfinite(alternative_snap)
            and alternative_snap + minimum_improvement < preferred_snap
        ):
            return alternative, "alternative"
        return preferred, "preferred"

    def match_order(self, points: pd.DataFrame) -> dict[str, Any]:
        if len(points) < 2:
            raise ValueError("Valhalla requires at least two trace points")
        configured_ignore = bool(self.config.get("ignore_oneways", False))
        preferred = self._match_candidate(
            points, ignore_oneways=configured_ignore
        )
        compared = False
        selected_label = "preferred"
        chosen = preferred
        strategy = str(self.config.get("oneway_candidate_selection", "disabled"))
        preferred_quality = self._candidate_quality(preferred)
        if strategy == "compare_on_risk" and self._candidate_is_risky(
            preferred_quality
        ):
            compared = True
            alternative = self._match_candidate(
                points, ignore_oneways=not configured_ignore
            )
            chosen, selected_label = self._select_candidate(preferred, alternative)
        chosen = dict(chosen)
        chosen["oneway_candidate_compared"] = compared
        chosen["selected_ignore_oneways"] = bool(
            configured_ignore
            if selected_label == "preferred"
            else not configured_ignore
        )
        chosen["configured_ignore_oneways"] = configured_ignore
        return chosen

    def match_batch(
        self, orders: list[pd.DataFrame] | tuple[pd.DataFrame, ...]
    ) -> list[dict[str, Any]]:
        return [self.match_order(points) for points in orders]
