# Paper storyline v2

## Central narrative

The paper is a **data-to-decision framework for ODD-aware mixed-fleet operations**. It explains how observed ride-hailing trajectories become prospective operational knowledge and how that knowledge changes rolling HV/AV assignment and full-day service outcomes.

The scientific chain is:

```text
Observed GPS, orders, and road network
→ directed network grounding
→ operational indicator families
→ decision-time deep prediction
→ hard-plus-continuous AV suitability
→ rolling mixed-fleet optimization
→ counterfactual fleet evolution
→ service, queue, assignment, and exposure outcomes
```

The broader framing is conditional rather than predetermined. It is retained only if the prediction-to-decision ablation shows that the prospective prediction layer has measurable downstream value. Otherwise, the headline returns to ODD-aware mixed-fleet dispatch and treats prediction as supporting infrastructure.

## The scientific object at each transition

1. **Observation → network grounding.** Noisy GPS traces and trip records are attached to a physically directed road representation so that route order, direction, traversals, nodes, and movements have stable identities.
2. **Network grounding → operational representation.** Directly observed traversal intervals produce measurable crawl, stop, speed-variability, acceleration, reliability, and tail-delay quantities without manufacturing dynamic labels for unresolved intervals.
3. **Operational representation → prediction.** A frozen deep route-conditioned model converts historical, static, and decision-time features into prospective travel-time and operational-indicator estimates. Completed evaluation-day trajectories and future realized traffic states are excluded.
4. **Prediction → AV decision interface.** Route structure and predicted conditions become a hard state plus separate static, dynamic, and speed suitability ratios. This is an optimization interface, not a safety-probability model.
5. **Interface → mixed-fleet decision.** Passenger acceptance, patience, current fleet state, route eligibility, cumulative family budgets, and sparse routed pickup opportunities determine feasible assignments.
6. **Decision → system evaluation.** Event-driven vehicle evolution propagates each local assignment into later supply, queues, expirations, and cumulative exposure.

## Information boundary

### Offline / historical

- Map matching and directed network identity construction.
- Link-, traversal-, node-, and movement-level operational labels.
- Model training, normalization, vocabulary construction, and checkpoint selection.
- Train-only historical reference profiles, distribution transforms, and capability-envelope calibration.
- Train-only demand information used by any repositioning robustness policy.

### Decision-time

- Current request, passenger eligibility, waiting time, and deadline.
- Current vehicle position and availability.
- Original-route hard feasibility and evidence completeness.
- Prospective travel-time and operational-indicator predictions.
- Static, dynamic, and speed suitability and cumulative exposure state.
- Routed pickup ETA and sparse rolling assignment.

The dispatch process does **not** consume completed evaluation-day trajectories, realized future route conditions, or future demand.

### Counterfactual evaluation

- Scenario fleet composition and AV availability assumptions.
- Event-driven vehicle-state evolution.
- Optional robustness policies declared before execution.
- Service, expiration, pickup, queue, assignment-composition, repositioning, and exposure outcomes.

## Narrative moves

1. **Motivation:** AV and HV vehicle-hours are not operationally interchangeable when serviceability is order-, route-, capability-, and time-dependent.
2. **Gap:** Dynamic dispatch, passenger acceptance, prospective traffic knowledge, and multi-family ODD suitability are rarely integrated in one empirical rolling system.
3. **Framework:** Construct a leakage-safe chain from real trajectories to sparse mixed-fleet decisions.
4. **Identification:** Use the frozen factorial to establish interactions, then isolate reviewer-level alternatives through gate attrition, repositioning, Gamma-frontier, and prediction-to-decision extensions.
5. **Mechanism:** Explain effective capacity as the conversion of nominal nearby supply through sequential compatibility and feasibility gates.
6. **Control:** Show why family-specific cumulative budgets are transparent and why they prevent hidden cross-family compensation.
7. **Implication:** Coordinate penetration, temporal fleet placement, passenger acceptance, capability, and ODD-aware dispatch rather than optimizing any one in isolation.

## Primary and supporting contributions

### Primary

- A data-to-decision chain that links real trajectory evidence to prospective mixed-fleet decisions.
- A hard-plus-continuous, family-specific AV suitability interface.
- A patience-aware sparse rolling assignment with cumulative family exposure controls.
- Empirical evidence about effective capacity and its sensitivity to acceptance, capability, repositioning, Gamma, and prediction information.

### Supporting

- Directed map matching, label conservation, and route/network identity.
- Deep prediction accuracy and leakage audits.
- Fleet simulation and routing infrastructure.
- Normalized cost and computational sensitivity.

Supporting modules enter the main text only to the extent needed to explain scientific validity or measured downstream value.

## Decision rule for the final headline

- **Use “data-to-decision ODD-aware mixed-fleet framework”** if the proposed prediction improves at least one material decision outcome or exposure-control outcome relative to the Train-only historical baseline without causing a material service loss.
- **Use “ODD-aware mixed-fleet dispatch”** if prediction differences are negligible at the decision layer, even if predictive accuracy differs.
- **Change the fleet-transition story** if simple Train-only repositioning largely restores high-penetration service.
- **Stop for implementation review** if any extension reveals future leakage, cumulative-budget violation, or inconsistent frozen semantics.

## Measured effective-capacity decision

The prospective four-anchor shadow-logging rerun closes the former observability gap. Every counter uses the same `(waiting order, available AV, decision epoch)` unit, and canonical outcomes reproduce exactly. For the p=.70 anchors, N5/N0 falls from 0.0939% at q=.25 to 0.0445% at q=.75. Passenger compatibility retention changes only slightly, whereas structural readiness, evidence completeness, and pickup feasibility under patience weaken more clearly.

This result qualifies rather than overturns the storyline. Effective capacity is a multi-gate conversion mechanism, not a passenger-acceptance-only mechanism. N0 is an endogenous rolling opportunity stock, shared Top-K is algorithmic compression rather than safety attrition, and N5→N6 is dispatch competition. The next decision point is repositioning robustness.
