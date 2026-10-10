# Prospective effective-capacity gate decomposition

## Execution verdict

`PASS — GO_REPOSITIONING_ROBUSTNESS`.

All four enhancement reruns used the frozen fleet, acceptance realization, routing, candidate pruning, solver, and vehicle evolution. Shadow logging changed observation only. Canonical products were never overwritten.

| scenario_id | summary_difference_count | request_outcomes_exact | assignments_exact | canonical_reproduction_pass |
| --- | --- | --- | --- | --- |
| MAIN_Q25_M_P70 | 0 | True | True | True |
| MAIN_Q50_M_P70 | 0 | True | True | True |
| MAIN_Q75_M_P70 | 0 | True | True | True |
| BENCH_AV_M | 0 | True | True | True |

## Same-unit conversion totals

| scenario_id | gate_av_n0_spatial | gate_av_n5_solver_eligible | gate_av_n6_selected | eligibility_conversion_n5_over_n0 | assignment_conversion_n6_over_n0 |
| --- | --- | --- | --- | --- | --- |
| MAIN_Q25_M_P70 | 7069575 | 6638 | 1187 | 0.0939% | 0.0168% |
| MAIN_Q50_M_P70 | 19545797 | 14074 | 2207 | 0.0720% | 0.0113% |
| MAIN_Q75_M_P70 | 42559860 | 18933 | 3134 | 0.0445% | 0.0074% |
| BENCH_AV_M | 74470227 | 27425 | 4545 | 0.0368% | 0.0061% |

An opportunity is one `(waiting order, available AV, decision epoch)` tuple. Repeated opportunities are intentional rolling-decision observations, not unique vehicles or orders.

## Conditional retention

| scenario_label | evidence | other arc | passenger | patience | routing | selection | shared Top-K | structure |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| all AV, p=1.00 | 41.73% | 100.00% | 100.00% | 2.71% | 100.00% | 16.57% | 7.53% | 43.30% |
| q=.25, p=.70 | 52.21% | 100.00% | 68.32% | 6.44% | 100.00% | 17.88% | 8.67% | 47.15% |
| q=.50, p=.70 | 47.27% | 100.00% | 67.49% | 5.33% | 100.00% | 15.68% | 9.38% | 45.15% |
| q=.75, p=.70 | 42.59% | 100.00% | 66.82% | 4.20% | 100.00% | 16.55% | 8.63% | 43.10% |

## Main findings

- Among the directly comparable p=.70 central anchors, eligibility conversion N5/N0 falls from 0.0939% at q=.25 to 0.0445% at q=.75, a relative change of -52.6%.
- Passenger compatibility is not the main changing bottleneck: its opportunity-weighted retention is 68.32%, 67.49%, and 66.82% for q=.25/.50/.75. The large absolute rejection counts mainly reflect the rapidly expanding N0 denominator.
- Structural retention declines from 47.15% to 43.10%, evidence retention from 52.21% to 42.59%, and routed-Top-K patience retention from 6.44% to 4.20%. These are the clearest penetration-related conversion changes.
- Shared Top-K retains 8.67%, 9.38%, and 8.63% of N3 across the central anchors. This is explicit algorithmic candidate compression, not route incompatibility or a safety statement.
- Routing-return retention and remaining arc-condition retention are 100% in all four anchors. Matrix routing failure and post-patience arc conditions do not explain the observed conversion loss.
- N5→N6 selection retention remains between 15.68% and 17.88% in the central anchors. This final difference is dispatch competition, not eligibility attrition.
- The all-AV anchor uses p=1.00 and is a composition extreme; it must not be used as a like-for-like acceptance comparison with the p=.70 central anchors.
- Hourly aggregation identifies 17:00–18:59 local time as the weakest eligibility-conversion period across all four anchors, coinciding with the largest or near-largest nominal opportunity volumes.

## Important qualification

The former `missing_exposure` diagnostic was zero because it was conditioned on the old dispatch-ready state. The prospective same-unit evidence gate is broader: it tests all structurally ready opportunities against the complete static/dynamic/speed plus AV-service-time contract. Its nonzero loss is therefore a newly identified mechanism, not a contradiction or implementation defect.

N0 grows both because more AVs are present and because unserved/carry-over orders reappear at later epochs. It is an endogenous rolling opportunity stock and must not be described as a unique-vehicle supply count.

## Scientific story decision

`QUALIFIES CURRENT STORY`. The effective-capacity conversion result is now directly quantified and supports the paper mechanism, but the loss is not primarily an acceptance story. It is jointly associated with structural readiness, complete decision evidence, shared sparse candidate construction, and pickup feasibility under patience.

Gamma remains excluded from this funnel because the four anchors are UNCONSTRAINED. Its causal policy role belongs in the separately authorized service–exposure frontier.
