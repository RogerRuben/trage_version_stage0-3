# Stage 3 S2A Field Provenance

| Product / field | Authority | S2A rule |
|---|---|---|
| directed topology, edge id, auto_routable | Frozen Valhalla 3.8.2 tiles | GraphId identity; include direction iff `forwardaccess & 1` |
| geometry, length, Valhalla class/use/layer | Frozen Valhalla MVT | Non-shortcut edge layer, zoom filtering disabled for all classes |
| OSM way/node, maxspeed, highway, bridge/tunnel/layer/roundabout | Frozen OSM PBF | pyosmium exact ids/tags; raw and effective fields separate |
| observed canonical identity and Stage0 static fields | Frozen Stage1 route_parts | Exact Valhalla id first, exact OSM endpoint direction second; no nearest-only mapping |
| control evidence | Frozen OSM PBF | Positive signal/stop/yield/roundabout evidence only; missing tag is unknown |
| turn restrictions | Frozen OSM PBF relations | pyosmium preserves member type/ref/role; uncertified direction stays uncertain |
| historical free-flow proxy | Frozen Stage1 link_traversals, Train 09–24 | direct_observed, valid time, physical positive bounds; edge-level P85/P90/P95 |
| speed-domain anchor | Stage0 positive speed_limit | Exact anchor set regenerated and SHA-bound |
| inferred speed domain | Frozen validation choice | Discrete grid, simple B0/B1/B2 selection on grouped anchor CV only |

POI is excluded. Inferred speed-domain values are not posted-speed observations. Intersection complexes are absent by design.
