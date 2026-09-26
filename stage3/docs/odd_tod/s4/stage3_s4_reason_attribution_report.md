# Stage 3 S4 Reason Attribution Report

Marginal reasons and co-occurrences preserve simultaneous causes; they are descriptive and not causal.

## C

- Multi-cause known-violation routes: 29,841 (99.470000%)
- Mean distinct known reasons per infeasible route: 7.773164
- Known violation plus unknown evidence: 17,297
- Routes carrying unknown evidence with exactly one reason (all final states): 12,910
- Routes carrying unknown evidence with multiple reasons (all final states): 4,389
- Final UNKNOWN routes with exactly one reason: 2
- Final UNKNOWN routes with multiple reasons: 0

### Final UNKNOWN route decomposition

| Unknown reason | Final UNKNOWN routes |
|---|---:|
| `CONSERVATIVE_LEFT_UNKNOWN_CONTROL` | 2 |
| `DYNAMIC_ROUTE_INCOMPLETE` | 0 |
| `MOVEMENT_LOOKUP_UNRESOLVED` | 0 |
| `SPEED_DOMAIN_UNKNOWN` | 0 |
| `STATIC_METRIC_UNKNOWN` | 0 |
| `TURN_GEOMETRY_UNKNOWN` | 0 |
| `UNRESOLVED_ROUTE_IDENTITY` | 0 |

### All-state marginal attribution

| Reason code | Marginal routes | Only-known-cause routes |
|---|---:|---:|
| `CERTIFIED_MOVEMENT_PROHIBITION` | 0 | 0 |
| `CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE` | 0 | 0 |
| `CONSERVATIVE_LEFT_UNKNOWN_CONTROL` | 11,035 | 0 |
| `CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE` | 4,480 | 0 |
| `DYNAMIC_ACCELERATION_RMS_C_CAP_EXCEEDED` | 7,589 | 0 |
| `DYNAMIC_ACCELERATION_RMS_E_CAP_EXCEEDED` | 7,482 | 13 |
| `DYNAMIC_ACCELERATION_RMS_Q_CAP_EXCEEDED` | 7,553 | 4 |
| `DYNAMIC_CRAWL_C_CAP_EXCEEDED` | 8,640 | 3 |
| `DYNAMIC_CRAWL_E_CAP_EXCEEDED` | 6,515 | 3 |
| `DYNAMIC_CRAWL_Q_CAP_EXCEEDED` | 8,796 | 7 |
| `DYNAMIC_ROUTE_INCOMPLETE` | 136 | 0 |
| `DYNAMIC_SPEED_CV_C_CAP_EXCEEDED` | 4,907 | 0 |
| `DYNAMIC_SPEED_CV_E_CAP_EXCEEDED` | 5,825 | 1 |
| `DYNAMIC_SPEED_CV_Q_CAP_EXCEEDED` | 4,246 | 0 |
| `DYNAMIC_STOP_C_CAP_EXCEEDED` | 5,784 | 0 |
| `DYNAMIC_STOP_E_CAP_EXCEEDED` | 5,661 | 0 |
| `DYNAMIC_STOP_Q_CAP_EXCEEDED` | 5,205 | 0 |
| `KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE` | 14,837 | 21 |
| `MOVEMENT_LOOKUP_UNRESOLVED` | 10,199 | 0 |
| `SPEED_DOMAIN_CAP_EXCEEDED` | 24,538 | 88 |
| `SPEED_DOMAIN_UNKNOWN` | 0 | 0 |
| `STATIC_A_CAP_EXCEEDED` | 28,619 | 0 |
| `STATIC_D_CAP_EXCEEDED` | 25,169 | 0 |
| `STATIC_L_CAP_EXCEEDED` | 28,979 | 1 |
| `STATIC_METRIC_UNKNOWN` | 0 | 0 |
| `STATIC_M_CAP_EXCEEDED` | 26,180 | 0 |
| `TURN_GEOMETRY_UNKNOWN` | 0 | 0 |
| `UNRESOLVED_ROUTE_IDENTITY` | 546 | 0 |
| `UTURN_PROFILE_INCOMPATIBLE` | 2,050 | 0 |

Pairwise major-family counts are in `test31_reason_cooccurrence.parquet`.

## M

- Multi-cause known-violation routes: 27,767 (92.556667%)
- Mean distinct known reasons per infeasible route: 4.478553
- Known violation plus unknown evidence: 10,517
- Routes carrying unknown evidence with exactly one reason (all final states): 10,347
- Routes carrying unknown evidence with multiple reasons (all final states): 255
- Final UNKNOWN routes with exactly one reason: 82
- Final UNKNOWN routes with multiple reasons: 3

### Final UNKNOWN route decomposition

| Unknown reason | Final UNKNOWN routes |
|---|---:|
| `CONSERVATIVE_LEFT_UNKNOWN_CONTROL` | 0 |
| `DYNAMIC_ROUTE_INCOMPLETE` | 3 |
| `MOVEMENT_LOOKUP_UNRESOLVED` | 82 |
| `SPEED_DOMAIN_UNKNOWN` | 0 |
| `STATIC_METRIC_UNKNOWN` | 0 |
| `TURN_GEOMETRY_UNKNOWN` | 0 |
| `UNRESOLVED_ROUTE_IDENTITY` | 3 |

### All-state marginal attribution

| Reason code | Marginal routes | Only-known-cause routes |
|---|---:|---:|
| `CERTIFIED_MOVEMENT_PROHIBITION` | 0 | 0 |
| `CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE` | 0 | 0 |
| `CONSERVATIVE_LEFT_UNKNOWN_CONTROL` | 0 | 0 |
| `CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE` | 0 | 0 |
| `DYNAMIC_ACCELERATION_RMS_C_CAP_EXCEEDED` | 2,998 | 29 |
| `DYNAMIC_ACCELERATION_RMS_E_CAP_EXCEEDED` | 2,989 | 75 |
| `DYNAMIC_ACCELERATION_RMS_Q_CAP_EXCEEDED` | 2,884 | 18 |
| `DYNAMIC_CRAWL_C_CAP_EXCEEDED` | 3,584 | 4 |
| `DYNAMIC_CRAWL_E_CAP_EXCEEDED` | 2,723 | 44 |
| `DYNAMIC_CRAWL_Q_CAP_EXCEEDED` | 3,986 | 115 |
| `DYNAMIC_ROUTE_INCOMPLETE` | 136 | 0 |
| `DYNAMIC_SPEED_CV_C_CAP_EXCEEDED` | 1,426 | 0 |
| `DYNAMIC_SPEED_CV_E_CAP_EXCEEDED` | 2,183 | 7 |
| `DYNAMIC_SPEED_CV_Q_CAP_EXCEEDED` | 1,462 | 2 |
| `DYNAMIC_STOP_C_CAP_EXCEEDED` | 1,615 | 0 |
| `DYNAMIC_STOP_E_CAP_EXCEEDED` | 2,131 | 15 |
| `DYNAMIC_STOP_Q_CAP_EXCEEDED` | 1,828 | 11 |
| `KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE` | 14,837 | 535 |
| `MOVEMENT_LOOKUP_UNRESOLVED` | 10,199 | 0 |
| `SPEED_DOMAIN_CAP_EXCEEDED` | 44 | 1 |
| `SPEED_DOMAIN_UNKNOWN` | 0 | 0 |
| `STATIC_A_CAP_EXCEEDED` | 27,181 | 496 |
| `STATIC_D_CAP_EXCEEDED` | 11,703 | 58 |
| `STATIC_L_CAP_EXCEEDED` | 25,903 | 170 |
| `STATIC_METRIC_UNKNOWN` | 0 | 0 |
| `STATIC_M_CAP_EXCEEDED` | 19,923 | 1 |
| `TURN_GEOMETRY_UNKNOWN` | 0 | 0 |
| `UNRESOLVED_ROUTE_IDENTITY` | 546 | 0 |
| `UTURN_PROFILE_INCOMPATIBLE` | 2,050 | 3 |

Pairwise major-family counts are in `test31_reason_cooccurrence.parquet`.

## A

- Multi-cause known-violation routes: 20,451 (68.170000%)
- Mean distinct known reasons per infeasible route: 3.253742
- Known violation plus unknown evidence: 9,313
- Routes carrying unknown evidence with exactly one reason (all final states): 10,347
- Routes carrying unknown evidence with multiple reasons (all final states): 255
- Final UNKNOWN routes with exactly one reason: 1,271
- Final UNKNOWN routes with multiple reasons: 18

### Final UNKNOWN route decomposition

| Unknown reason | Final UNKNOWN routes |
|---|---:|
| `CONSERVATIVE_LEFT_UNKNOWN_CONTROL` | 0 |
| `DYNAMIC_ROUTE_INCOMPLETE` | 20 |
| `MOVEMENT_LOOKUP_UNRESOLVED` | 1,266 |
| `SPEED_DOMAIN_UNKNOWN` | 0 |
| `STATIC_METRIC_UNKNOWN` | 0 |
| `TURN_GEOMETRY_UNKNOWN` | 0 |
| `UNRESOLVED_ROUTE_IDENTITY` | 25 |

### All-state marginal attribution

| Reason code | Marginal routes | Only-known-cause routes |
|---|---:|---:|
| `CERTIFIED_MOVEMENT_PROHIBITION` | 0 | 0 |
| `CONSERVATIVE_LEFT_STOP_YIELD_INCOMPATIBLE` | 0 | 0 |
| `CONSERVATIVE_LEFT_UNKNOWN_CONTROL` | 0 | 0 |
| `CONSERVATIVE_ROUNDABOUT_INCOMPATIBLE` | 0 | 0 |
| `DYNAMIC_ACCELERATION_RMS_C_CAP_EXCEEDED` | 739 | 83 |
| `DYNAMIC_ACCELERATION_RMS_E_CAP_EXCEEDED` | 749 | 58 |
| `DYNAMIC_ACCELERATION_RMS_Q_CAP_EXCEEDED` | 707 | 35 |
| `DYNAMIC_CRAWL_C_CAP_EXCEEDED` | 935 | 53 |
| `DYNAMIC_CRAWL_E_CAP_EXCEEDED` | 748 | 86 |
| `DYNAMIC_CRAWL_Q_CAP_EXCEEDED` | 1,154 | 68 |
| `DYNAMIC_ROUTE_INCOMPLETE` | 136 | 0 |
| `DYNAMIC_SPEED_CV_C_CAP_EXCEEDED` | 185 | 0 |
| `DYNAMIC_SPEED_CV_E_CAP_EXCEEDED` | 505 | 54 |
| `DYNAMIC_SPEED_CV_Q_CAP_EXCEEDED` | 320 | 3 |
| `DYNAMIC_STOP_C_CAP_EXCEEDED` | 234 | 0 |
| `DYNAMIC_STOP_E_CAP_EXCEEDED` | 515 | 15 |
| `DYNAMIC_STOP_Q_CAP_EXCEEDED` | 380 | 4 |
| `KNOWN_REVERSE_DIRECTION_AV_UNROUTABLE` | 14,837 | 3,391 |
| `MOVEMENT_LOOKUP_UNRESOLVED` | 10,199 | 0 |
| `SPEED_DOMAIN_CAP_EXCEEDED` | 0 | 0 |
| `SPEED_DOMAIN_UNKNOWN` | 0 | 0 |
| `STATIC_A_CAP_EXCEEDED` | 16,565 | 141 |
| `STATIC_D_CAP_EXCEEDED` | 11,703 | 577 |
| `STATIC_L_CAP_EXCEEDED` | 18,257 | 692 |
| `STATIC_METRIC_UNKNOWN` | 0 | 0 |
| `STATIC_M_CAP_EXCEEDED` | 15,381 | 79 |
| `TURN_GEOMETRY_UNKNOWN` | 0 | 0 |
| `UNRESOLVED_ROUTE_IDENTITY` | 546 | 0 |
| `UTURN_PROFILE_INCOMPATIBLE` | 0 | 0 |

Pairwise major-family counts are in `test31_reason_cooccurrence.parquet`.
