# Manuscript architecture v2

This is an architecture and emphasis plan, not rewritten manuscript prose.

## Title and abstract

- Keep the current ODD-aware mixed-fleet title provisionally.
- Decide whether to add “data-to-decision” only after the prediction-to-decision ablation.
- Rewrite the abstract last using the final P0 findings.

## 1. Introduction

1. Mixed HV/AV transition as an effective-capacity problem.
2. Why prospective operational knowledge is required for AV assignment.
3. Gap: limited integration of empirical trajectories, prediction, suitability, and rolling dispatch.
4. Data-to-decision framework and explicit offline/decision-time/evaluation boundary.
5. Three research questions, four contributions, and paper organization.

## 2. Literature review

- Dynamic ride-hailing matching, patience, and fleet repositioning.
- Mixed HV/AV operations and effective-capacity accounting.
- Passenger acceptance and realized service eligibility.
- ODD, route suitability, and capability-aware operation.
- Prediction-to-decision integration and constrained/lexicographic dispatch.
- Conclude with the integrated research gap; use verified references only after literature verification.

## 3. Empirical system and problem setting

### 3.1 Xi'an ride-hailing observations and road network

- Data dates, coordinate/network grounding, directed identity, and evaluation day.
- Briefly state quality and conservation gates; move engineering audits to supplement.

### 3.2 Dynamic mixed-fleet service process

- Orders, finite patience, carry-over, HV sessions, full-horizon AVs, passenger acceptance.

### 3.3 Information timeline

- One diagram and one concise table separating offline, decision-time, and counterfactual information.
- Explicitly prohibit future realized evaluation-day traffic in dispatch.

## 4. Data-to-decision methodology

### 4.1 Directed network grounding and map matching — brief but explicit

- Directed route parts, continuous traversals, nodes/movements, and unresolved evidence.
- Scientific role: physical identity and route ordering.

### 4.2 Operational indicator-family construction — moderate

- Direct-observed crawl, stop, bounded speed variability, bounded acceleration RMS, reliability/tail delay.
- Conservation and missingness semantics.

### 4.3 Decision-time deep prediction — moderate and conditional

- Route-conditioned structured model, predicted targets, Train-only artifacts, leakage boundary.
- Report predictive evidence compactly; emphasize measured downstream value after ablation.

### 4.4 AV route-suitability / ODD interface — strong

- Hard FEASIBLE/INFEASIBLE/UNKNOWN state.
- Evidence completeness and separate static/dynamic/speed ratios.
- Capability profiles and positive exceedance semantics.

### 4.5 Mixed-fleet reconstruction and passenger acceptance — moderate

- Exact vehicle-hour denominator and whole-session selection.
- Full-horizon AV assumption.
- `p_A` versus realized `a_o^A`.

### 4.6 Rolling mixed HV/AV assignment — strong

- Sparse feasible arcs, patience, carry-over, critical priority, one-to-one assignment.
- Family-specific cumulative budgets and lexicographic objectives.
- Add four concise propositions; proofs in supplement.

### 4.7 Counterfactual simulation framework — concise

- Event-driven fleet evolution, pickup routing, candidate sparsification, and reproducibility.
- Repositioning is a declared robustness policy, not part of the canonical baseline.

## 5. Experimental design

- Frozen 27-scenario factorial, benchmarks, central policy comparison, and cost sensitivity.
- Separate enhancement registry for gate, repositioning, Gamma, and prediction experiments.
- Predeclare variant grids and decision gates.

## 6. Results

- Preserve frozen results as the canonical baseline.
- Integrate gate attrition directly after the temporal effective-capacity result.
- Keep acceptance/capability interactions and family activity.
- Replace the single REFERENCE-point emphasis with a frontier interpretation.

## 7. Mechanism and robustness analysis

1. Effective-capacity gate decomposition.
2. Train-only demand-based AV repositioning robustness.
3. Gamma service–exposure frontier.
4. Prediction-to-decision ablation.
5. Optional central transition curve only if still scientifically necessary.

Every subsection ends with SUPPORTS, QUALIFIES, CHANGES, or IMPLEMENTATION DEFECT FOUND.

## 8. Discussion and managerial implications

- Interpret nominal versus effective capacity using observed gates.
- State the role of repositioning honestly.
- Scale prediction prominence to downstream value.
- Explain family-specific budgets using the propositions and frontier.
- Keep implications conditional on the empirical system and frozen assumptions.

## 9. Limitations and future work

- Preserve current limitations.
- Add robustness-policy simplicity, Train-only hotspot assumptions, and lack of endogenous demand/traffic response.

## 10. Conclusion

- Restate the complete trajectory-to-decision chain.
- Report canonical and P0 robustness findings separately.
- Avoid declaring a universally optimal penetration, Gamma, or AV policy.

## Appendix / supplement

- Detailed map matching and conservation audits.
- Indicator formulas and prediction architecture/evaluation.
- Capability-envelope calibration.
- Theory proofs and counterexample.
- Full registries, 27-scenario matrix, robustness results, and computational diagnostics.
