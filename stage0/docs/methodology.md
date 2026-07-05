# Methodology and reproducibility notes

## Road extraction

The validated road source is the Geofabrik China free Shapefile snapshot dated `2017-01-01T20:28:02Z`, close to the trajectory date (`2016-10-01`). The source documentation states that all coordinates are unprojected WGS84 (EPSG:4326).

Because that historical free archive has no administrative-boundary layer, two rectangular extracts are produced:

- a conservative Xi'an municipality envelope for reuse;
- a smaller core network that fully covers the observed trajectories.

Non-drivable classes (`footway`, `path`, `steps`, `cycleway`, `bridleway`, and `pedestrian`) are excluded. One-way codes follow the Geofabrik documentation: `F` follows LineString direction, `T` is the reverse direction, and `B` permits both.

## Full-day processing

The 28.3 million source rows are interleaved rather than contiguous by order. Loading them into memory or treating contiguous runs as orders would be incorrect. The pipeline therefore:

1. hashes each `order_id` into one of 128 Parquet buckets;
2. sorts complete orders by timestamp inside each bucket;
3. removes invalid coordinates, timestamp duplicates, and isolated extreme-speed spikes;
4. converts source coordinates when requested, while preserving the originals;
5. matches points to road candidates and projects them exactly onto road geometry;
6. audits route continuity using shared nodes and direct geometry adjacency;
7. aggregates quality, matching, semantic, and behavior features per order.

## Matching diagnostics

Order-level outputs include:

- matched-point ratio;
- mean and P90 GPS-link distance;
- matched/GPS route-length ratio;
- topology-gap and parallel-road-switch counts;
- composite matching confidence.

Matched route sequences are stored as a scalable long table rather than large JSON arrays in the order table.

## Primitive behavior features

The full-day order table contains speed statistics, low-speed ratio, stop count and duration, acceleration variability, heading changes, turn count, intersection-delay proxy, curvature, dominant road class, and semantic coverage.

`intersection_delay_s` is a primitive indicator: time spent below 10 km/h within 30 m of a topological or signalized intersection. It is not a causal estimate of signal delay.

## Known limitations

- The empirical coordinate evidence strongly favors GCJ-02 input despite the WGS84 declaration.
- The free 2017 shape has no lane-count field.
- Only about 1% of core road features carry a non-zero `maxspeed` value.
- The Xi'an envelope is rectangular, not an exact administrative boundary.
- The original Stage0 baseline matcher is topology-audited nearest-road projection; the monthly extension adds HMM/Viterbi while retaining that baseline for comparison and fallback.

## Monthly HMM/Stage1 extension

The monthly extension retains the geometric/topology matcher as a baseline and fallback, then applies an HMM/Viterbi matcher with exact road projections. Emission likelihood is distance-based. Transition likelihood compares observed displacement with a locally routed network distance; directed routing is primary and a penalized undirected relaxation handles clipped-network and incomplete-oneway edge cases. A non-degradation guard falls back to the geometric result when the HMM topology audit is worse.

Consecutive HMM states are expanded to link paths. Interpolated links are explicitly flagged and are used for route topology, turn movements, and intrinsic GNS only; traversal-specific realized stress is not imputed where no GPS behavior was observed.

Stage1 produces LCS, IIS, GNS, RTS, and PMIS at traversal level. RTS uses a hierarchical historical seconds-per-meter reference. Empirical percentiles use bounded streaming histograms and fall back through link/time/weekday, link/peak, road-class/grid/time, road-class/time, road-class, and global cohorts. Topology gaps, route-length ratio, and matching distance remain quality controls rather than stress inputs.
