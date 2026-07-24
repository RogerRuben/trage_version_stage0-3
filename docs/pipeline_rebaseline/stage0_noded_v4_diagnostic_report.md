# Stage 0 noded-network v4 diagnostic and promotion status

## Outcome

`xian_2017_core_noded_v4` fixes boolean parsing and prevents cross-layer endpoint
clustering. It is an **exploratory promotion candidate**, not canonical. The
fixed-sample comparison triggered the required manual-review stop gate, so no
full-date run or downstream stage was started.

## Network construction

- `T/F`, `TRUE/FALSE`, `1/0`, `Y/N`, `YES/NO`, Python booleans and missing values
  are parsed explicitly.
- Endpoint clusters are partitioned by normalized `(layer, bridge, tunnel)`.
- 4,689 incompatible-level intersection pairs were left un-noded.
- 2,267 zero-length terminal transition connectors were added only when
  endpoint proximity, heading and road semantics agreed. These connectors are
  graph-only and cannot be selected as GPS candidates. Connector direction is
  derived from the incident links: 652 are bidirectional and 1,615 unidirectional.
- The resulting graph has 23,658 nodes, 31,639 links, 65 weak components and 865
  strong components. The largest WCC contains 98.63% of nodes, 99.11% of links,
  and 98.82% of road length.

## Fixed 1,000-order comparison (2016-10-23)

| Metric | Old network | Noded v3 | Noded v4 |
|---|---:|---:|---:|
| Successful reconstruction | 1,000 | 984 | 984 |
| Explicit failures | 0 | 16 | 16 |
| v4-quality Core | 66 | 311 | 139 |
| Orders with direction gaps | 827 | 84 | 169 |
| Mean direction gaps | 3.832 | 0.0976 | 0.2236 |
| Geometric fallback orders | 94 | 0 | 0 |
| Directed OD reachability | 61.40% | 99.70% | 99.80% |
| Mean matched-route length | 9,159.6 m | 5,360.1 m | 5,891.1 m |
| P95 matched-route length | 21,890.9 m | 11,907.2 m | 13,508.1 m |

The v4 failure count is unchanged, but the direction-corrected route length is
9.9% above v3 and v4-quality Core decreases by 17.2 percentage points. Directed OD reachability
is suspiciously close to 100%. This does not prove an error by itself, but it
meets the pre-registered stop rule and requires route-level and grade-transition
manual review before full-date processing.

## Train/validation diagnostic

The same deterministic 1,000-order scheme was run for 2016-10-19, 20 and 22.
Successful/failed counts are 982/18, 990/10 and 982/18. Core counts under the
pre-registered v4 rules are 160, 159 and 131; no geometric fallback was used.

## Manual truth pack

A deterministic 150-order v2 pack is available in
`docs/pipeline_rebaseline/manual_truth/`. It contains 78 Core candidates and 72
boundary/Rejected routes, covers the three dates, and includes bridge, tunnel,
elevated, complex-intersection, high/low confidence, detour, U-turn, gap,
interpolation and distance strata. The GeoJSON overlays GPS traces and matched
routes. A separate 40-order sheet is provided for independent double review,
plus a 50-connector direction/level review pack.

No human judgment has yet been entered. The audit therefore reports `HOLD`, not
`PASS`; Core precision, false-positive rate and reviewer agreement are unknown.

## Promotion status

Full-date matching, time/distance conservation on those full outputs, and the
canonical manifest are deliberately not generated until the human review gate
passes. Stage 1--4 remain `HOLD`.
