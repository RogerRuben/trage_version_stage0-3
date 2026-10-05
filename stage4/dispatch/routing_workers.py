"""Independent 1x1 routing primitives; no fleet/demand state in workers."""
from __future__ import annotations

from math import isfinite
import time

_ACTOR = None


def initialize_worker(config_path):
    from valhalla import Actor
    global _ACTOR
    _ACTOR = Actor(config_path)


def route_query(payload):
    mode, origin_lon, origin_lat, target_lon, target_lat, minute = payload
    started = time.perf_counter()
    request = dict(costing="auto", units="kilometers", date_time={"type": 1, "value": minute})
    try:
        if mode == "SINGLE_SOURCE_MATRIX":
            request.update(sources=[dict(lon=origin_lon, lat=origin_lat)],
                           targets=[dict(lon=target_lon, lat=target_lat)])
            cell = _ACTOR.matrix(request)["sources_to_targets"][0][0]
            raw, distance = float(cell["time"]), float(cell["distance"]) * 1000
            if not isfinite(raw) or raw < 0:
                raise ValueError("invalid matrix time")
        elif mode == "SCALAR_ROUTE":
            request.update(locations=[dict(lon=origin_lon, lat=origin_lat, type="break"),
                                      dict(lon=target_lon, lat=target_lat, type="break")],
                           directions_type="none")
            trip = _ACTOR.route(request)["trip"]
            if int(trip.get("status", 0)) != 0 or len(trip.get("legs", [])) != 1:
                raise ValueError("Valhalla did not return one successful leg")
            raw, distance = float(trip["summary"]["time"]), float(trip["summary"]["length"]) * 1000
        else:
            raise ValueError("unsupported worker routing mode")
        return raw, distance, None, time.perf_counter() - started
    except Exception as error:
        return None, None, f"{mode}:{type(error).__name__}", time.perf_counter() - started
