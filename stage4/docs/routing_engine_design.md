# RoutingEngine design

Simulator v3 requires all final travel-time queries to pass through `RoutingEngine`.

Current implementation:

- BallTree is used only for coarse candidate filtering.
- Final pickup ETA is queried through `RoutingEngine.query`.
- Route cache key includes origin zone, destination zone, time bin, vehicle type, and rounded coordinates.
- The fixed `Haversine / 8 m/s` final ETA shortcut is removed.

Phase 1 smoke:

```text
routing_query_count = 264,215
routing_cache_hit_rate = 0.8702
mean_candidate_truncation_rate = 0.0874
```

