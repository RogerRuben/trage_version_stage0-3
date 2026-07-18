# Hybrid Selective-HMM Stage 0 v5

## Canonical network

The PBF is expanded to a motor-vehicle directed multigraph. Stable edge identity is
`osm_way_id:segment_seq:direction`; reverse-oneway, ordinary bidirectional roads, roundabouts,
motorways, ramps, service/access rules, bridge, tunnel, and layer semantics are handled before
matching. Parallel edges are retained except where the audit explicitly classifies an exact or
semantic duplicate. Parsed and unresolved OSM restrictions are both retained.

Three network products have deliberately different roles:

1. canonical edge store: authoritative edge identity and attributes;
2. compact metric graph: distance-only bounded routing;
3. edge movement graph: legal edge-to-edge movements and final reconstruction.

The 2026 PBF snapshot post-dates the 2016 trajectories. Every post-2016 edge is marked
`network_snapshot_mismatch`; the mismatch is a limitation, not silently treated as historical
truth.

## Matching

Curvature- and complexity-aware densification feeds a `cKDTree`. KD distance is only a coarse
filter; final distance is an exact point-to-LineString projection. Each candidate preserves its
canonical `edge_uid`. Emission evidence includes projection distance, heading, direction,
grade context, road class, access, and previous topology.

Only genuinely close candidates, explicit parallel/grade ambiguity, or topology evidence
trigger local HMM windows. Disjoint windows remain local. Full-order HMM is reserved for
pervasive ambiguity or a merged window covering most of an order. Metric transitions use:

- direct along-edge distance for a repeated edge;
- direct movement distance for adjacent legal edges;
- batched, bounded SciPy Dijkstra with a source/cutoff LRU otherwise.

## Reconstruction and dynamic evidence

After state selection, edge-aware shortest path traverses the movement graph. Inserted edges are
`inferred_path` and `is_interpolated=true`. They can support static topology and exposure only;
they never manufacture realized time, acceleration, stop, DTP, ORU, or dynamic movement labels.
Observed GPS intervals are partitioned once between supported traversals and observed edge
transitions, providing an auditable time-conservation identity.

## Quality layers

Hard structural failures produce `rejected`. Passing every hard condition produces the
`analysis_set`; passing every hard and soft condition produces `strict_core`. Soft failures remain
explicit in the output. Thresholds are frozen in `stage0/config/stage0_v5.yaml` before Test.

## POI boundary

POI is never an input to candidate generation or HMM. It is cleaned and mapped to raw standard
categories. Ground POI is assigned at most once, only to a compatible ground-level road edge;
bridge, tunnel, and non-zero-layer edges do not inherit ground exposure.
