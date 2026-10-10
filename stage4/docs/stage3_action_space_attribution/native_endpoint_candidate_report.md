# Frozen native endpoint candidates — stop before interface mutation

## Decision

The requested independent candidate table is complete. **Interface application is
NOT approved by the verification result.** No Stage3 production code, frozen data,
movement table, complex membership, profile, route readiness or assignment has
been changed.

| Check | Result |
|---|---:|
| Audited directed edges / missing endpoint sides | 88 / 88 |
| Native endpoints read and independently cross-checked | 88 / 88 |
| Consistent with every existing non-null endpoint | 69 / 88 |
| Existing non-null endpoint identity conflicts | 19 / 88 |
| Of those, explicit native hierarchy transition to correct node | 18 |
| Of those, same-level different node without that transition | 1 |
| Distinct native nodes on the missing side | 55 |
| Missing-side native IDs already in frozen Stage3 node table | 0 / 88 |
| Original 391 audited encounters involving a conflict edge | 65 / 391 |

“Native identity verified” is distinct from “safe to apply to the existing Stage3
graph.” The 69 consistent rows are **not 69 repaired movements**. Their recovered
nodes are still absent from the frozen node product; complex/boundary semantics
have not been established. All 391 original unresolved encounters remain unchanged,
including the two previously identified discontinuous route chains.

## Exact reading, not coordinate inference

Reader: `stage4/tools/read_native_endpoint_candidates.py`.

The installed native binding parses GraphTileHeader. The small read-only decoder
is explicitly locked to Valhalla 3.8.2, little-endian x64 and the official record
layouts. It checks native header size, tile byte count and section boundaries.
It decodes the DirectedEdge endnode GraphId and finds its origin from the unique
NodeInfo outgoing-edge index range. All 207,911 directed-edge slots across the four
loaded tiles have unique owners. It validates both opposing-edge links for every
target, the OSM way ID, forward access mask, road class and length. Node coordinates
are checked against the installed native GraphUtils edge shape. The oriented frozen
MVT geometry is checked only for consistency: coordinates never choose an ID.

Official implementation references:

- [GraphTile initialization](https://github.com/valhalla/valhalla/blob/3.8.2/src/baldr/graphtile.cc)
- [NodeInfo](https://github.com/valhalla/valhalla/blob/3.8.2/valhalla/baldr/nodeinfo.h)
- [DirectedEdge](https://github.com/valhalla/valhalla/blob/3.8.2/valhalla/baldr/directededge.h)
- [NodeTransition](https://github.com/valhalla/valhalla/blob/3.8.2/valhalla/baldr/nodetransition.h)
- [GraphTileHeader](https://github.com/valhalla/valhalla/blob/3.8.2/valhalla/baldr/graphtileheader.h)
- [EdgeInfo](https://github.com/valhalla/valhalla/blob/3.8.2/valhalla/baldr/edgeinfo.h)

The first reader draft incorrectly equated the export's MVT-relative F/R marker
with native EdgeInfo `forward`. The exporter source shows these are different
reference frames. The final v2 reader records the 33 marker differences but does
not call them direction failures; all 88 oriented geometries agree with native
source/target coordinates within the documented MVT quantization consistency
check. The initial draft output is retained locally only and is not final evidence.

## What the conflict means

The 18 cross-level cases have explicit NodeTransition links, not merely close
coordinates. They may describe the same physical location at different graph
levels, but they are **not the same directed-topology identity**. Substituting or
collapsing them requires a deliberate interface contract. This audit does not
automatically transfer old complex membership across them.

The remaining static-network example is:

| Field | Value |
|---|---|
| stage3 edge | `s3e_03f8008f1965f994be0b1dae` |
| directed GraphId | `759773026672` |
| old non-null to-node | `309438994800` |
| native to-node | `309606766960` |

Both IDs are in the same hierarchy level and are different native nodes. There
is no explicit transition from the old node to the correct one. It cannot be
treated as a cross-level alias or silently overwritten.

The existing S2A exporter assigns endpoint IDs using rounded MVT coordinates and
`min(candidates)` without native hierarchy/edge ownership validation. That code
path can conflate graph identities. The native result now demonstrates actual
identity inconsistency on 19 of the audited edges; it does **not** establish a
full-network error prevalence or invalidate the independently supported reverse
direction restrictions. No service or capacity gain is claimed.

## Artifacts and reproducibility

- [Independent candidate table (CSV)](native_endpoint_candidates_v2/endpoint_candidates.csv)
- [Summary, frozen source/tile hashes and resources](native_endpoint_candidates_v2/summary.json)
- [QA results](native_endpoint_candidates_v2/qa.json): 3/3 tests, compileall and diff check PASS; application gate STOP.
- Local Parquet: `stage4/output/paper_enhancement/native_endpoint_candidates_v2/endpoint_candidates.parquet`
- Necessary tests: `stage4/tests/test_native_endpoint_reader.py` (3 read-only checks).

The candidate table contains static network identities only, not order/vehicle
IDs or trajectory coordinates. A row with `STOP_CONFLICT_OR_UNRESOLVED` must not
be applied. The complete batch status is `STOP_VERIFICATION_FAILED`, despite
successful extraction of all native identities.

Execution: CPU single process, about 0.52 s for the final extraction, peak process
working set 478.68 MiB; no GPU, routing, matrix, rerouting, inference, optimizer,
or full-day simulation. Source edge/node/config and loaded tile hashes were
identical before/after. The tool refuses to overwrite completed evidence.

## Narrow next decision, not an automatic expansion

Before interface repair, decide how to represent exact native node identities
alongside the frozen physical intersection membership. A shadow reconciliation
must separately handle the 18 explicit cross-level transitions, the one genuine
same-level identity conflict, and the 55 missing native nodes. It must reconcile
the original 117 movement keys / 391 encounters without deleting UNKNOWN evidence.
If that requires changing complex membership or clustering, stop for explicit
authorization. Do not extend this into a full-network rebuild or change reverse,
fallback selection, C/M/A, M3, the manuscript, or the 41 frozen scenarios.
