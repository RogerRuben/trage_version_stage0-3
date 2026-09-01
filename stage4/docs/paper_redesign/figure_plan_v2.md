# Figure and visual-evidence plan v2

No figure is drawn in this planning phase. Every item below answers a specific scientific question.

| figure_id | scientific_question | data_source | required_fields | visual_form | placement | expected_message | new_simulation_required |
|---|---|---|---|---|---|---|---|
| Figure 1 | How does information move from empirical observation to rolling mixed-fleet outcomes? | Frozen data contracts, method artifacts, and dispatch model | layer names, inputs, outputs, information regime, forbidden future information | Left-to-right framework flow with three horizontal information-regime bands | Main text | The contribution is an auditable observation-to-decision chain, not disconnected modules | No |
| Figure 2A | What empirical road and demand system underlies the evaluation? | Frozen Xi'an road network, evaluation orders, observed route/link summaries | network geometry, OD coordinates, link-flow count, request hour | Four coordinated map/chart panels | Main text | The model is grounded in a real directed network and temporally heterogeneous demand | No |
| Figure 2B | How is one noisy trajectory converted into directed operational identity? | Existing audited route examples and directed route products | GPS points, matched directed links, traversals, nodes, movements, unresolved intervals | Annotated map sequence and compact provenance strip | Appendix or main method inset | Network grounding preserves direction and separates observed from unresolved evidence | No |
| Figure 3 | How do prospective conditions become profile-specific AV suitability? | Frozen route descriptors, decision-time predictions, C/M/A profiles | predicted crawl/stop/speed-CV/acceleration, static descriptors, rho families, hard state | One real route map plus three capability radar/bar panels | Main text | The same route can activate different envelope families under different assumed capability | No |
| Figure 4 | Where are nominal nearby AV opportunities lost? | Existing candidate/epoch/request logs where observable | nearby AV count, acceptance pruning, evidence pruning, routed cells, patience pruning, Gamma feasibility, selected AV arcs | Funnel or aligned attrition bars for four anchors | Main text | Effective capacity is the retained portion after explicit compatibility and feasibility gates | No unless missing gates require new logging; never infer missing states |
| Figure 5 | Does simple Train-only repositioning explain the high-penetration service decline? | Separate repositioning robustness outputs | service, expiration, queue pressure, P95, AV share, reposition time/distance by anchor | Paired baseline/reposition bars plus service-vs-q lines | Main text | Classify whether spatial rebalancing supports, qualifies, or changes the fleet-transition story | Yes |
| Figure 6 | Is REFERENCE on a stable service–exposure frontier? | Separate Gamma-frontier outputs | lambda, service, AV share, static/dynamic exposure, P95, expiration | Two linked frontier plots with STRICT/REFERENCE/UNCONSTRAINED markers | Main text | Show the shape and local position of the calibrated REFERENCE point without declaring an optimum | Yes |
| Figure 7 | Does deep prediction change downstream decisions? | Prediction-ablation outputs and existing prediction evaluation | prediction errors, ODD-ready share, service, AV share, expiration, family exposure | Two-panel prediction-level and decision-level contrast | Main text conditional on measured value | Link prediction improvement to decision consequence or reduce prediction prominence | Yes |
| Figure A | How are demand and service distributed across Xi'an? | Frozen network/orders and baseline outcomes | OD density, spatial service/expiration, hourly demand | Network map plus hourly profile | Appendix | Document empirical coverage and spatial heterogeneity | No |
| Figure B | How does effective capacity vary spatially across fleet compositions? | Frozen all-HV and q=.25/.50/.75 outcomes | order coordinates, matched/expired state, spatial aggregation key | Small-multiple service/expiration maps | Main text or appendix | Identify whether loss is concentrated or system-wide | No if coordinates retained; otherwise observable fields must be documented |
| Figure C | Where do ODD policies change AV-served flows? | Frozen STRICT/REFERENCE/UNCONSTRAINED assignment logs | AV-served OD or route geometry, family exposure, policy | Small-multiple flow or OD-density maps | Appendix | Show spatial assignment consequences of family budgets | No if assignment geometry retained |
| Figure D | Is the central transition curve linear or threshold-like? | Optional transition-curve extension | q_A, service, expiration, queue, AV share | Four aligned response curves | Appendix or main text if nonlinear region is important | Characterize the transition shape without fitting an unjustified nonlinear model | Optional P1 |
| Figure E | Are findings stable to acceptance realization or profile thresholds? | Optional seed/profile robustness | seed or perturbation, primary outcomes, exposure | Compact range/dot plot | Appendix | Test ordering robustness without p-value inference | Optional P1 |

## Figure 1 framework specification

### Blocks

```text
Real GPS + orders + road network
→ directed map matching
→ operational indicators
→ decision-time deep prediction
→ hard + static/dynamic/speed suitability
→ patience/acceptance/Gamma rolling assignment
→ counterfactual vehicle evolution
→ service/queue/composition/exposure outcomes
```

### Information-regime bands

- **Offline/historical:** grounding, labels, training, Train-only calibration.
- **Decision-time:** prediction, suitability, passenger/fleet state, routing, assignment.
- **Counterfactual evaluation:** scenario fleet, vehicle evolution, outcomes.

### Required boundary annotations

- No future realized evaluation-day traffic at dispatch time.
- Historical operational indicators are not AV safety outcomes.
- Hard feasibility and three continuous families remain separate.
- Canonical and enhancement experiments use separate registries.

### Design constraints

- No software logos as primary visual elements.
- No decorative arrows without a data product or decision object.
- No implication that AV follows historical HV trajectories.
- No composite weighted “risk” meter.

Framework figure specification completed: **YES**
Framework figure drawn: **NO**
# V3 priority update

Main figures are now: (1) data-to-decision framework; (2) real network, map matching, and empirical system; (3) prediction-to-suitability route example; (4) fleet transition plus effective-capacity mechanism; and (5) ODD policy/family activity. Fleet representativeness, routing reproducibility, fixed-state prediction ablation, and cost robustness remain appendix candidates. The incomplete repositioning experiment must not appear as a main result figure.
