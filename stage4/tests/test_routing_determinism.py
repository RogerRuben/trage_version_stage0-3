from __future__ import annotations

import pandas as pd
from pathlib import Path

from stage4.dispatch.candidate_graph import SpatialVehicle
from stage4.dispatch.deterministic_routing import (
    SCALAR_ROUTE,
    SINGLE_SOURCE_MATRIX,
    ArcDeterministicValhallaAdapter,
)
from stage4.dispatch.routing_determinism_runner import _matrix_call

ROOT = Path(__file__).resolve().parents[2]


class BatchSensitiveActor:
    def matrix(self, request):
        size = len(request["sources"])
        return {
            "sources_to_targets": [
                [{"time": 10.0 + size + index, "distance": 1.0}]
                for index in range(size)
            ]
        }

    def route(self, request):
        return {
            "trip": {
                "status": 0,
                "legs": [{}],
                "summary": {"time": 12.0, "length": 1.0},
            }
        }


def _vehicles():
    return [
        SpatialVehicle("v1", 1, "AV", 108.90, 34.20),
        SpatialVehicle("v2", 2, "AV", 108.91, 34.21),
    ]


def test_single_source_mode_is_independent_of_candidate_batch() -> None:
    actor = BatchSensitiveActor()
    adapter = ArcDeterministicValhallaAdapter(
        ROOT, actor=actor, routing_mode=SINGLE_SOURCE_MATRIX
    )
    result = adapter.estimate_many(
        _vehicles(), 108.92, 34.22, pd.Timestamp("2016-10-31T08:00:00+08:00")
    )
    assert [result[index].valhalla_time_s for index in (1, 2)] == [11.0, 11.0]
    assert adapter.routing_queries == 2
    assert adapter.routing_arc_evaluations == 2


def test_scalar_mode_routes_each_arc_independently() -> None:
    actor = BatchSensitiveActor()
    adapter = ArcDeterministicValhallaAdapter(
        ROOT, actor=actor, routing_mode=SCALAR_ROUTE
    )
    result = adapter.estimate_many(
        _vehicles(), 108.92, 34.22, pd.Timestamp("2016-10-31T08:00:00+08:00")
    )
    assert [result[index].valhalla_time_s for index in (1, 2)] == [12.0, 12.0]
    assert adapter.routing_queries == 2


def test_source_order_diagnostic_reads_focal_index() -> None:
    actor = BatchSensitiveActor()
    focal = type(
        "Focal",
        (),
        {
            "pickup_lon_wgs84": 108.92,
            "pickup_lat_wgs84": 34.22,
            "timestamp": pd.Timestamp("2016-10-31T08:00:00+08:00"),
        },
    )()
    sources = [(108.90, 34.20), (108.91, 34.21), (108.89, 34.19)]
    first = _matrix_call(actor, focal, sources, 0)
    last = _matrix_call(actor, focal, list(reversed(sources)), 2)
    assert first["raw_time_s"] == 13.0
    assert last["raw_time_s"] == 15.0
    assert first["returned_source_count"] == 3
