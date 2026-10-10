# Stage 3 S2A Network Foundation Report

Phase status: `STAGE3_S2A_NETWORK_FOUNDATION_FROZEN`. S2A exports a Stage3-owned complete directed auto-access graph; it does not build intersection complexes.

## Full network

- Nodes: `89,607`
- Directed auto-routable edges: `209,454`
- Unique OSM ways: `44,112`
- Authority: frozen Valhalla 3.8.2 directed edges with `forwardaccess & kAutoAccess`.
- Identity: `stage3_edge_uid = s3e_ + sha256('valhalla-3.8.2|' + uint64 GraphId)[:24]`.
- Road-class distribution: `{"motorway": 2884, "primary": 11755, "residential": 65404, "secondary": 19842, "service_other": 29907, "tertiary": 44774, "trunk": 5134, "unclassified": 29754}`
- Effective bridge / tunnel edges: `4,945` / `1,532`
- Non-zero OSM layer edges: `6,813`

## Observed mapping

- Observed segments: `18,878` / `18,992` (99.400%)
- Observed directed identities: `18,878` / `25,382` (74.376%)
- Ambiguous / unmatched: `0` / `6,504`
- Unmatched by observed traversal direction: `{"F": 2, "R": 6502}`. Reverse identities are not silently projected onto an auto-forbidden direction.
- Geometry-only remapping: `false`.

## Static provenance

- S1 bridge discrepancy reconciliation: `{"s1_observed_canonical_segment_total": 18992, "s1_stage0_bridge_true_segments": 418, "s1_osm_bridge_true_segments": 687, "s1_osm_true_stage0_false_segments": 269, "mapped_observed_segment_count_in_s2a": 18878, "full_network_directed_bridge_effective_count": 4945, "interpretation": "S1 discrepancy is preserved exactly in its canonical-segment unit. S2A effective bridge enriches the full directed network while retaining raw Stage0/Valhalla/OSM fields and conflicts."}`
- Raw Stage0, Valhalla, and OSM bridge/tunnel fields remain separate; effective values are OR-enrichment with explicit conflicts.
- OSM highway names are context, not claims of Chinese statutory functional class.

## Controls and restrictions

- Positive control evidence: `4,081` rows; `{"ROUNDABOUT": 76, "SIGNALIZED": 4005}`
- Traffic signals mapped to at least one full-network edge: `3,898`
- Restriction relations: `20`; role-preserving parse: `PASS`.
- Directed enforcement certified: `0`. Uncertified restrictions remain uncertain; no geometry-guessed enforcement is fabricated.
- Missing OSM control tags are not negative evidence. POI was not used.

## Scope and limitations

- No intersection clustering or tolerance selection occurred.
- No Stage2 training/inference, profile calibration, Test31 fitting, fallback routing, or Stage4 execution occurred.
- MVT geometries are an export representation of the frozen routing graph. Topology identity remains Valhalla GraphId; OSM endpoint ids are best-effort exact endpoint enrichment and never override GraphId.
