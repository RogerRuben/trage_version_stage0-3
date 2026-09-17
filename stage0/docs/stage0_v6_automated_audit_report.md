# Stage 0 v6 Automated Audit Report

## Outcome

- Fixed sample audited: **600 orders**
- `auto_pass`: **400**
- `auto_fail`: **7**
- `manual_review`: **101**
- `excluded_low_information`: **92**
- Processing exceptions: **0**

## Main manual-review triggers

- `VALHALLA_DISCONTINUITY`: 34
- `LOW_DYNAMIC_COVERAGE`: 30
- `DYNAMIC_UNUSABLE`: 25
- `EXTREME_UNRESOLVED_TIME`: 20
- `V5_V6_ROUTE_DIVERGENCE`: 12
- `LOW_ROUTE_BUFFER_COVERAGE`: 7
- `HIGH_SNAP_P90`: 6
- `POSSIBLE_UTURNS`: 3
- `ROUTE_QUALITY_REJECTED`: 3
- `TOPOLOGY_VIOLATIONS`: 1

## Review images

- Images generated: **37**
- Case index range: **1-37**
- Manual-review images: **30**
- Auto-fail images: **7**
- Queue: `stage0/output_v6/audit/manual_review_pack/index.csv`

## Reuse and efficiency

- Audit source: existing `stage0/output_v6/hot/` Parquet products.
- Valhalla reruns: **0**.
- Automated audit runtime: **1.536 s**.
- Automated audit peak RSS: **429.2 MB**.
- Image-pack runtime: **1.681 s**.
- Image-pack peak RSS: **411.2 MB**.
- Raw GPS loading was restricted to selected image candidates.
- Stability rematching was intentionally not run: it is optional and would invoke
  the matcher; this audit instead localizes risk windows from existing products.

## Most discriminating indicators

The conservative rules prioritize route reconstruction success, route/GPS distance
ratio, snap p90/p99 and buffer coverage, OD endpoint validity, canonical topology,
Valhalla discontinuities, extreme implied speeds, unresolved-time share, and large
v5/v6 edge-set divergence. Reverse traversal against an OSM one-way tag is
reported as an informational conflict, not a failure. `audit_score` combines
the risk indicators only for ranking; information-poor orders are separated
before the pass/review/fail rules.

Dynamic evidence is reported separately as `high_dynamic_coverage`,
`direct_time_observations`, `low_dynamic_coverage`, or `static_route_only`.
The existence of one directly observed interval is not treated as sufficient
dynamic coverage.

## How to respond

Open `manual_review_pack/index.md` or the numbered SVG image files, then reply with
case numbers only, for example: **`1, 3, 7, 12`**. The numbering is deterministic
for unchanged audit inputs.
