"""Small deterministic matrix-failure versus single-route equivalence audit."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import pandas as pd


class MatrixFailureRouteAuditor:
    """Audit adapter-recorded matrix failures with identical single routes."""

    def __init__(self, actor: Any) -> None:
        self.actor = actor
        self.failed_arcs: list[dict[str, Any]] = []

    def audit_with_single_route(
        self, *, sample_size: int, seed: int
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        rows: list[dict[str, Any]] = []
        for event_index, record in enumerate(self.failed_arcs):
            identity = "|".join(
                [
                    str(int(seed)),
                    f"{record['origin_lon_wgs84']:.7f}",
                    f"{record['origin_lat_wgs84']:.7f}",
                    f"{record['pickup_lon_wgs84']:.7f}",
                    f"{record['pickup_lat_wgs84']:.7f}",
                    record["timestamp_value"],
                    str(event_index),
                ]
            )
            rows.append(
                {
                    **record,
                    "event_index": event_index,
                    "sample_priority": hashlib.sha256(
                        identity.encode("utf-8")
                    ).hexdigest(),
                }
            )
        if not rows:
            return {
                "matrix_failed_arc_events": 0,
                "sample_seed": int(seed),
                "requested_sample_size": int(sample_size),
                "sampled_matrix_failures": 0,
                "single_route_success": 0,
                "single_route_failure": 0,
                "single_route_success_rate": None,
                "audit_runtime_s": 0.0,
            }, pd.DataFrame()
        sample = (
            pd.DataFrame(rows)
            .sort_values(["sample_priority"], kind="mergesort")
            .head(int(sample_size))
        )
        results: list[dict[str, Any]] = []
        started = time.perf_counter()
        for row in sample.itertuples(index=False):
            route_request = {
                "locations": [
                    {
                        "lon": float(row.origin_lon_wgs84),
                        "lat": float(row.origin_lat_wgs84),
                        "type": "break",
                    },
                    {
                        "lon": float(row.pickup_lon_wgs84),
                        "lat": float(row.pickup_lat_wgs84),
                        "type": "break",
                    },
                ],
                "costing": "auto",
                "units": "kilometers",
                "directions_type": "none",
                "date_time": {"type": 1, "value": row.timestamp_value},
            }
            success = False
            failure_reason = None
            try:
                trip = self.actor.route(route_request)["trip"]
                success = (
                    int(trip.get("status", 0)) == 0 and len(trip.get("legs", [])) == 1
                )
                if not success:
                    failure_reason = "NON_SUCCESS_TRIP"
            except Exception as exc:
                failure_reason = f"ROUTE_EXCEPTION:{type(exc).__name__}"
            results.append(
                {
                    **row._asdict(),
                    "route_success": success,
                    "route_failure_reason": failure_reason,
                }
            )
        detail = pd.DataFrame(results)
        sampled = len(detail)
        successes = int(detail.get("route_success", pd.Series(dtype=bool)).sum())
        summary = {
            "matrix_failed_arc_events": len(self.failed_arcs),
            "sample_seed": int(seed),
            "requested_sample_size": int(sample_size),
            "sampled_matrix_failures": sampled,
            "single_route_success": successes,
            "single_route_failure": sampled - successes,
            "single_route_success_rate": successes / sampled if sampled else None,
            "audit_runtime_s": time.perf_counter() - started,
        }
        return summary, detail
