# Stage 2 v5.2 micro product report

Status: **NOT RUN — Phase A implementation only**.

Implemented schemas and writers:

- `stage2/output_v5_2/micro_condition_tokens/`
- `stage2/output_v5_2/original_route_micro_conditions/`
- `stage2/output_v5_2/static_route_complexity/`

Row count, order count, coverage, support shares, hashes, and partition manifests
remain pending Phase B. No placeholder number is treated as an observed result.

Static schema audit: `canonical_highway`, `road_class`, `bridge`, and `tunnel`
are available upstream. Intersection, signal, merge, turn, ramp, speed-limit,
and lane fields are explicitly NA.
