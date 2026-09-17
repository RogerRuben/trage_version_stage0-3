# Stage 0 v6 fixed-600 manual audit pack

The deterministic review pack is generated under:

```text
stage0/output_v6/manual_audit/
```

It contains:

- `manual_review_100.csv` and `.parquet`: 100 selected orders, multi-label
  strata, v5/v6 internal quality, route-edge overlap, map path, and blank
  human-review fields;
- `maps/`: 100 self-contained HTML/SVG maps showing WGS84-converted raw GPS,
  the v5 route, and the v6 Valhalla/canonical route;
- `manifest.json`: sample SHA, selection method, tag counts, and the explicit
  `review_complete=false` / `accuracy_claim_allowed=false` gate.

Allowed human labels are:

```text
v5正确
v6正确
两者都正确
两者都错误
无法判断
```

The selection covers ordinary roads, complex routes, elevated/ground
structures, main/auxiliary road patterns, ramps, bridge/tunnel edges, sparse
GPS, low-speed/stopped traces, v5/v6 success differences, and route
differences when those cases exist in the fixed sample.

This artifact is a review input, not a completed human audit. Until reviewers
fill `human_label`, `review_notes`, and `review_status`, neither the Strict
Core share nor this pack may be presented as real map-matching accuracy.
