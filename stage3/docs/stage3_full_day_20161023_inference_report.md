# Full-day 2016-10-23 Stage3 inference audit

| stage                      |   orders |   share_of_raw |
|:---------------------------|---------:|---------------:|
| raw_stage0_order_base      |   114356 |      1         |
| stage0_od_valid            |   114356 |      1         |
| route_conditioned_ready    |   112165 |      0.980841  |
| stage2_rc_mstnet_predicted |   112165 |      0.980841  |
| stage3_stage4_exported     |   112165 |      0.980841  |
| not_route_conditioned      |     2191 |      0.0191595 |

Audit status: **PASS**

- Missing condition vectors were not imputed.
- Orders outside the exported universe failed before route-conditioned inference and are reported as `not_route_conditioned`.
- IIS full-day movement predictions are unavailable in this run; IIS is represented by `iis_availability=false`, not by zero stress.