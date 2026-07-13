# 2016-10-23 single-day ABM simulation report

## Full-day prediction coverage

| stage | orders |
| --- | ---: |
| raw Stage0 order base | 114,356 |
| Stage0 OD valid | 114,356 |
| route-conditioned ready | 112,165 |
| Stage2 RC-MSTNet predicted | 112,165 |
| Stage3/Stage4 condition vector exported | 112,165 |
| not route-conditioned | 2,191 |

No median, mean, random, rule-based, or realized-stress fallback was used to create condition vectors.  IIS full-day movement prediction is not yet available, so IIS is exported as unavailable rather than as zero stress.

## Agent population

| item | value |
| --- | ---: |
| HV agents | 18,301 |
| HV sessions | 31,627 |
| AV count | 915 |
| AV/HV ratio | 4.9997% |
| depots | 8 |

Historical replay succeeded for 95.47% of full-day historical orders.  Missing driver rate is 0.

## Strategy results

| strategy | match rate | cancel rate | mean wait sec | mean pickup sec | platform profit | AV share | AV ODD violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| GlobalMatch-MinPickup | 0.9233 | 0.0767 | 66.42 | 52.97 | 1,733,573 | 0.2270 | 23,506 |
| ODD-Gated Price-Aware | 0.6491 | 0.3509 | 126.67 | 67.67 | 1,805,284 | 0.0000 | 0 |
| Three-Stakeholder Balanced | 0.6485 | 0.3515 | 126.28 | 68.04 | 1,804,582 | 0.0000 | 0 |

The current `moderate_av` capability profile has 0% AV feasibility on the full-day condition vector because IIS is unavailable for all orders and the existing uncertainty/missing-modality policy is conservative.  This is an important methodological finding, not a parameter to hide: Stage4 now needs either full-day IIS movement prediction or a revised, validation-set capability prior before AVs can participate under the hard ODD gate.

## Audits

All core audits pass:

- full-day inference audit: PASS
- HV session audit: PASS
- AV depot audit: PASS
- dynamic-radius audit: PASS
- ABM output/state audit: PASS

