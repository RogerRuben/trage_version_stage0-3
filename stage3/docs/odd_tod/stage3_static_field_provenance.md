# Stage 3 S1 static field provenance

Status: `STAGE3_S1_STATIC_INVENTORY_COMPLETE`

S2 authorized: `NO`

This is a read-only inventory. No intersection clustering, tolerance selection,
static enrichment, capability calibration, Stage 2 inference, route assessment,
or Test31 fitting was performed.

## Population boundary

- Frozen Stage0/Stage1 route parts: 15,649,455 rows.
- Observed canonical segment IDs: 18,992.
- Observed directed identities (`canonical_edge_uid` + direction): 25,382.
- This is the accepted-order observed subnetwork, not the complete routable graph.
- Legacy full `canonical_edges.parquet` present: `false`.

## Priority fields

| Field | Source actually present | Coverage/result | Provenance limit |
|---|---|---|---|
| `speed_limit` | Stage0 route parts / Valhalla edge field | edge positive coverage 2.643%; null 18,490; zero 0 | no per-edge source field; cannot call every value DIRECT posted speed |
| OSM `maxspeed` | frozen PBF highway ways | base-tag coverage 0.963%; parseable among tagged 100.000% | legal defaults are not inferred in S1 |
| signalization | OSM `highway=traffic_signals` nodes | 4,005 raw; endpoint-ID mapping 10.112% | mapping denominator is the observed accepted-order subnetwork |
| turn restriction | OSM `type=restriction` relations | 20 relations; all member ways observed 35.000% | GDAL does not expose member roles/refs; exact directed enforcement not certified |

## Stage0 speed-limit facts

The field is numeric and assumed km/h by the Valhalla contract. Positive
coverage is reported by route-part row, route length, unique canonical edge, and
road class in the JSON inventory. Values below
5 km/h or above
160 km/h are descriptive
diagnostic flags only, not Stage3 thresholds. Stage0 has no `speed_limit_source`,
so OSM agreement does not prove DIRECT provenance and disagreement does not by
itself prove an error.

## Frozen OSM maxspeed

- Highway ways: 53,181.
- Base `maxspeed`: 512.
- Directional-only ways: 0.
- Conditional ways: 0.
- Parseable base values: 512.
- Motorway base coverage: 4.418%;
  trunk: 15.831%;
  primary: 4.233%;
  secondary: 2.107%.
- Observed canonical edges with parseable OSM base speed but missing positive
  Stage0 speed: 0.

OSM `maxspeed`, `maxspeed:forward/backward`, and `maxspeed:conditional` therefore
remain distinct raw evidence. S2 must not replace missing posted limits with road
class or design speed.

## Signals and control

OSM is the primary signal source. Missing `highway=traffic_signals` is not evidence
of `UNSIGNALIZED_CONTROLLED`. POI has 0
signal-term rows and 1,220
junction-term rows, of which
1,168 are
bus-stop records whose names merely contain “路口”. POI has no OSM identity
columns and is corroboration-only; it is not a signal/control inventory.

## Roundabouts

- `junction=roundabout` ways: 69.
- `junction=roundabout` nodes: 0.
- `highway=mini_roundabout` nodes: 7.
- Roundabout ways represented in the observed graph: 36.232%.

The dominant representation is way-level. S1 does not consolidate or construct
intersection complexes.

## Turn restrictions

Node-via/way-via counts are inferred only from relation geometry composition:
`{"ambiguous_geometry": 5, "node_via": 14, "way_via": 1}`.
Geometry members can often be associated with OSM ways, but exact `from`/`via`/`to`
roles and refs are unavailable through the current GDAL reader. A role-preserving
PBF reader is a prerequisite before S2 can enforce directed restrictions.

## Layer, bridge, tunnel, grade separation

- Nonzero OSM layer: 5,387 highway ways.
- Bridge: 4,376.
- Tunnel: 1,115.
- Any grade-separation tag: 5,771.

Stage0 route parts retain bridge/tunnel booleans but not `layer`. Spatial
proximity alone is therefore insufficient for future intersection merging. On
the observed canonical segments, OSM identifies
687 bridge segments,
while Stage0 marks 418;
269 need
OSM-side enrichment review. Tunnel tags agree on all
86 observed segments.

## Graph identity

Observed route parts retain canonical base-segment UID, explicit traversal
direction, OSM way ID, begin/end OSM node IDs, and canonical from/to nodes. Raw tagged signal nodes can
be joined exactly by OSM node ID. The GDAL PBF point layer does not expose all
untagged way nodes, and the cleaned legacy full canonical-edge table is absent.

## S2 decisions required after review

1. Choose a role-preserving OSM relation reader before turn-restriction enforcement.
2. Decide whether to reconstruct/export a complete frozen-network edge table from
   the frozen PBF/Valhalla tiles; do not treat the
   18,992 observed route segments
   (or the 11,754
   direct-observed subset) as the complete graph.
3. Define a defensible speed-source provenance adapter separating Valhalla,
   OSM base/directional/conditional values, legal defaults, and UNKNOWN.

`NEXT_PHASE_AUTHORIZED = NO`
