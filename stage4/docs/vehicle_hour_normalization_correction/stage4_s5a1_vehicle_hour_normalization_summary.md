# Stage4 S5A.1 Vehicle-Hour Normalization Correction

Recommendation: `GO_S5B_EXPERIMENTAL_DESIGN`

## Root cause

S0 15-minute overlap-based active supply was integrated as if it were exact vehicle-hours, while HV availability was measured using exact continuous session duration.

## Corrected semantics

- `H_base_exact = 12279.336389` vehicle-hours: total exact continuous duration of the frozen effective HV sessions; this is the q_A denominator.
- `H_base_15min_equivalent = 14369.500000` vehicle-hours: temporal supply-profile bin equivalent; it is not the q_A denominator.

## Corrected fleet accounting

| requested q_A | achieved q_A | AV count | raw HV residual h | target HV h | achieved HV h | HV error % | HV sessions |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.00 | 0.000000 | 0 | 12279.336389 | 12279.336389 | 12279.336389 | 0.000000 | 8435 |
| 0.10 | 0.099680 | 51 | 11055.336389 | 11055.336389 | 11124.673333 | 0.627181 | 7594 |
| 0.25 | 0.250176 | 128 | 9207.336389 | 9207.336389 | 9265.387500 | 0.630488 | 6314 |
| 0.50 | 0.500353 | 256 | 6135.336389 | 6135.336389 | 6154.377778 | 0.310356 | 4188 |
| 0.75 | 0.750529 | 384 | 3063.336389 | 3063.336389 | 3080.836111 | 0.571263 | 2101 |
| 1.00 | 1.000706 | 512 | -8.663611 | 0.000000 | 0.000000 | 0.000000 | 0 |

## Corrected q_A=0.25 neutral replay

- S3 requests/matched/completed/expired: 1458/1144/1144/314.
- S3 HV/AV assignments: 1007/137; runtime 109.097s.
- S4 requests/matched/completed/expired: 1458/1144/1144/314.
- S4 first-window/carry-recovered/critical-matched: 1027/117/2; runtime 119.548s.
- Corrected S3/S4 aggregate equality: `True`.
- Corrected S4 fingerprint: `af6dc8997dbe234e1a3c51e37fe74453cb518c9924ffebc2129ce40d3fc12555`.

## Refreshed Gamma references

Gamma is a cumulative reference-envelope exposure budget, not a safety threshold.

| family | ZERO | MEAN | PATH | UNCONSTRAINED | PATH-MEAN |
| --- | ---: | ---: | ---: | --- | ---: |
| static | 0 | 2.145068 | 2.200874 | null | 0.055807 |
| dynamic | 0 | 0.149343 | 0.401175 | null | 0.251833 |
| speed | 0 | 0.000000 | 0.000000 | null | 0.000000 |

Dynamic PATH maximum occurs at AV assignment rank 1 (2016-10-31T08:01:00+08:00).

## Canonical status

The rolling matcher and ODD-aware kernel remain valid. The previous q_A=0.25 canonical fleet composition used an inconsistent vehicle-hour denominator and is superseded for scientific penetration comparisons.

The previous fingerprint `a90f1285813cfe5fc9fedeeb6514ed5b204ad5de7a5e230316639d8e1ff2c961` is engineering lineage only. This correction is the canonical base for S5B.
