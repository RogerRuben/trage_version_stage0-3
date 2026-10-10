# Stage 3 S4 Original-Route Suitability Report

## Overall

- Test31 orders: 30,000
- Order-profile evaluations: 90,000

| Profile | FEASIBLE | UNKNOWN | INFEASIBLE |
|---|---:|---:|---:|
| C | 16 (0.0533%) | 2 (0.0067%) | 29,982 (99.9400%) |
| M | 564 (1.8800%) | 85 (0.2833%) | 29,351 (97.8367%) |
| A | 2,921 (9.7367%) | 1,289 (4.2967%) | 25,790 (85.9667%) |

## Major known and unknown evidence families

| Profile | Family | Route count | Share |
|---|---|---:|---:|
| C | direction | 14,837 | 49.4567% |
| C | speed | 24,538 | 81.7933% |
| C | static_A | 28,619 | 95.3967% |
| C | static_M | 26,180 | 87.2667% |
| C | static_D | 25,169 | 83.8967% |
| C | static_L | 28,979 | 96.5967% |
| C | movement/control | 2,050 | 6.8333% |
| C | roundabout | 4,480 | 14.9333% |
| C | restriction | 0 | 0.0000% |
| C | dynamic | 24,031 | 80.1033% |
| C | unknown:identity | 546 | 1.8200% |
| C | unknown:speed | 0 | 0.0000% |
| C | unknown:static | 0 | 0.0000% |
| C | unknown:movement | 10,199 | 33.9967% |
| C | unknown:control | 11,035 | 36.7833% |
| C | unknown:dynamic | 136 | 0.4533% |
| M | direction | 14,837 | 49.4567% |
| M | speed | 44 | 0.1467% |
| M | static_A | 27,181 | 90.6033% |
| M | static_M | 19,923 | 66.4100% |
| M | static_D | 11,703 | 39.0100% |
| M | static_L | 25,903 | 86.3433% |
| M | movement/control | 2,050 | 6.8333% |
| M | roundabout | 0 | 0.0000% |
| M | restriction | 0 | 0.0000% |
| M | dynamic | 14,253 | 47.5100% |
| M | unknown:identity | 546 | 1.8200% |
| M | unknown:speed | 0 | 0.0000% |
| M | unknown:static | 0 | 0.0000% |
| M | unknown:movement | 10,199 | 33.9967% |
| M | unknown:control | 0 | 0.0000% |
| M | unknown:dynamic | 136 | 0.4533% |
| A | direction | 14,837 | 49.4567% |
| A | speed | 0 | 0.0000% |
| A | static_A | 16,565 | 55.2167% |
| A | static_M | 15,381 | 51.2700% |
| A | static_D | 11,703 | 39.0100% |
| A | static_L | 18,257 | 60.8567% |
| A | movement/control | 0 | 0.0000% |
| A | roundabout | 0 | 0.0000% |
| A | restriction | 0 | 0.0000% |
| A | dynamic | 4,784 | 15.9467% |
| A | unknown:identity | 546 | 1.8200% |
| A | unknown:speed | 0 | 0.0000% |
| A | unknown:static | 0 | 0.0000% |
| A | unknown:movement | 10,199 | 33.9967% |
| A | unknown:control | 0 | 0.0000% |
| A | unknown:dynamic | 136 | 0.4533% |

Counts are marginal route-level prevalence and overlap across families; they must not be summed as mutually exclusive causes.

## Gates

- Nestedness: **PASS**
- Known-violation precedence: **PASS**
- Frozen profile unchanged: **TRUE**
- Frozen CDF unchanged: **TRUE**
- Fallback attempted: **NO**
- Route search performed: **NO**
- Profile retuned: **NO**
- Test31 calibration: **NO**

These are exact historical/original-route compatibility outcomes. They are not an AV safety, legality, failure, accident, disengagement, or product-approval result.
