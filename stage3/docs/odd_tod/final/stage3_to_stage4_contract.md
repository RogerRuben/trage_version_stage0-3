# Frozen Stage3 → Stage4 Contract

The canonical input is `stage3/output/odd_tod/final/test31_stage3_to_stage4_interface.parquet`, exactly 30,000 Test31 orders × three C/M/A profiles.

- `hard_state == INFEASIBLE`: Stage4 must forbid the AV assignment.
- `hard_state == UNKNOWN`: Stage4 chooses whether baseline policy excludes or allows it.
- `rho_static`, `rho_dynamic`, and `rho_speed` remain separate continuous capability-utilization families. They are not a safety score and must not be collapsed into a Stage3 binary label.
- `selected_service_time_p50_s == null`: Stage4 may exclude that AV arc under its baseline policy; Stage3 does not impute it.
- Passenger acceptance is supplied separately by Stage4. Stage3 contains no passenger model or dispatch solver.
- `selected_route_reference` resolves either to the frozen historical route (`ORIGINAL:<order_id>`) or `test31_fallback_route_edges.parquet` (`FALLBACK:<order_id>:<digest>`).

A fallback means only that a bounded route exists on the frozen AV-routable network under the hypothetical capability profile. It is not AV safety, legal, or commercial certification.

## Limited-search and Stage4 eligibility semantics

`selected_route_type=NONE` with `fallback_search_state=NOT_ESTABLISHED_UNDER_LIMITED_K1_SEARCH` means the frozen bounded K=1 procedure did not establish a hard-feasible AV route. It is `hard_state=UNKNOWN`, not proof that no AV route exists for the OD.

Stage4 should distinguish structural route availability from evidence completeness. Under the conservative baseline, an AV arc is dispatch-ready only when a hard-feasible route is selected and static, dynamic, speed, and service-time evidence are complete; passenger acceptance is then applied separately.
