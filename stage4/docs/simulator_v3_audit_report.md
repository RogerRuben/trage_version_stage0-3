# Simulator v3 audit report

Audit outputs are stored under:

```text
stage4/docs/results/simulator_v3/
```

Phase 1 smoke configuration:

```text
replication = 1
strategy = Safe GlobalMatch-MinPickup
operation = O0
request-time scenario = RT-Base
orders = 1,000 regression window after 2016-10-23 00:00 UTC
vehicles = 2,000 regression subset
```

Key results:

```text
completed_orders = 657
cancelled_orders = 343
match_rate = 0.657
vehicle_leg_count = 1,314
plan_revision_count = 657
routing_cache_hit_rate = 0.8702
mean_candidate_truncation_rate = 0.0874
overall_phase1_status = PASS
```

Not yet claimed:

- full-day v3 replication 1;
- O0–O3;
- Preassignment;
- formal Balanced constraints;
- FleetPy cross-validation.

