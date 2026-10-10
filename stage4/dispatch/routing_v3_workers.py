"""Bounded single-source forward groups; experimental until compared."""
from __future__ import annotations

from math import isfinite
import time

from . import routing_workers as base


def query_group(payload):
    queries, grouped = payload
    if not grouped or len(queries) == 1:
        return dict(results=[base.route_query(q) for q in queries], queries=len(queries), fallbacks=0)
    if not 1 < len(queries) <= 32:
        raise ValueError("single-source group exceeds 32 targets")
    first = queries[0]
    if first[0] != "SINGLE_SOURCE_MATRIX" or any((q[0], q[1], q[2], q[5]) != (first[0], first[1], first[2], first[5]) for q in queries):
        raise ValueError("group contains different sources/modes/departure minutes")
    request = dict(sources=[dict(lon=first[1], lat=first[2])],
        targets=[dict(lon=q[3], lat=q[4]) for q in queries], costing="auto", units="kilometers",
        date_time=dict(type=1, value=first[5]))
    start = time.perf_counter()
    try:
        cells = base._ACTOR.matrix(request)["sources_to_targets"][0]
    except Exception:
        return dict(results=[base.route_query(q) for q in queries], queries=1 + len(queries), fallbacks=len(queries))
    elapsed = time.perf_counter() - start
    result = []
    fallbacks = 0
    for i, q in enumerate(queries):
        try:
            raw, distance = float(cells[i]["time"]), float(cells[i]["distance"]) * 1000.
            if not isfinite(raw) or raw < 0 or not isfinite(distance) or distance < 0:
                raise ValueError("invalid grouped cell")
            result.append((raw, distance, None, elapsed / len(queries)))
        except (IndexError, KeyError, TypeError, ValueError):
            result.append(base.route_query(q))
            fallbacks += 1
    return dict(results=result, queries=1 + fallbacks, fallbacks=fallbacks)
