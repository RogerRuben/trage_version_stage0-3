# Stage 0 route-truth v1 review protocol

This directory is a review instrument, not completed ground truth.

## Files

- `stage0_route_truth_v1_review_pack.csv`: 500 unique orders and primary review fields.
- `stage0_route_truth_v1_review_pack_double_review.csv`: 100 independently selected
  orders for a second reviewer.
- `stage0_route_truth_v1_review_geometries.geojson`: two features per sampled
  order (`gps_trace` and `matched_route`) for visual inspection in QGIS or an
  equivalent GIS.
- `stage0_route_truth_v1_manifest.json`: frozen sample and acceptance criteria.
- `stage0_route_truth_v1_audit.json`: computed review gate; currently `HOLD`.

## Review rules

Reviewers must inspect the GPS trace and matched route without consulting Stage
1 labels or later-stage predictions. Set `review_status=completed`, provide a
non-empty `reviewer_id`, and complete `route_correct` plus every error flag. Use
`needs_adjudication` when road level, parallel-road choice, direction or endpoint
correctness cannot be resolved confidently.

The second reviewer must not see the first review before submitting the double
review sheet. Disagreements are adjudicated and retained rather than overwritten.
Run:

```powershell
python stage0/scripts/audit_manual_route_truth.py `
  --primary docs/pipeline_rebaseline/manual_truth/stage0_route_truth_v1_review_pack.csv `
  --secondary docs/pipeline_rebaseline/manual_truth/stage0_route_truth_v1_review_pack_double_review.csv `
  --output docs/pipeline_rebaseline/manual_truth/stage0_route_truth_v1_audit.json
```

Promotion requires at least 300 completed primary reviews, a non-empty independent
double-review set, and Core precision of at least 90%. The preferred precision is
95%. Threshold changes after review require a new config version and review audit;
2016-10-23 cannot be used to tune them.
