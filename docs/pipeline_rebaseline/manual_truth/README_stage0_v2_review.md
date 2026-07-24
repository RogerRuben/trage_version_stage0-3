# Stage 0 v4 targeted review instructions

The registered primary workload is `stage0_route_truth_v2_review_pack.csv` (150
routes: 78 Strict Core and 72 boundary/rejected). A second independent reviewer uses
`stage0_route_truth_v2_review_pack_double_review.csv` (40 routes). Route/GPS
geometries are in `stage0_route_truth_v2_review_geometries.geojson`.

For every completed row, set `reviewer_id`, `review_status=completed`, exactly one
`review_class`, the corresponding Boolean issue fields, and optional comments.
Allowed classes are:

- `Correct`
- `Minor error`
- `Major error`
- `Uncertain / data limitation`

Do not change matcher outputs or quality fields. A data limitation is not silently
treated as a correct route or an algorithmic major error. At least 120 primary and
30 paired secondary reviews are required. The automated gate checks Core major
errors (<=15%), wrong direction (<=5%), wrong road level/bridge/tunnel (<=5%),
unreasonable detour (<=10%), and exact-class double-review agreement (>=80%).

The separate `stage0_connector_review_v1.csv` covers 50 connectors. Review whether
each connection is directionally and physically plausible; uncertain complex
interchanges should be marked as a data limitation rather than guessed.
