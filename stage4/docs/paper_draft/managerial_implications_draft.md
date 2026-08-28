# Managerial implications draft

The following implications are scoped to the frozen Test31 mixed-fleet design. Each implication states the observed contrast, the operational reading, and the principal boundary on its use.

## 1. Manage effective capacity, not nominal AV share

**Finding.** Increasing (q_A) from 0.25 to 0.75 reduced marginal mean service rate from 0.7258 to 0.3924, an absolute change of -0.3334 and a relative decline of 45.9%.

**Implication.** Fleet-transition dashboards should track how nominal AV hours convert into feasible assignments, completed service, and avoided expiration. Penetration targets based only on vehicle counts or active hours can overlook passenger compatibility and route eligibility.

**Boundary.** This is a modeled Test31 result under a frozen fleet construction, not a universal estimate of the effect of AV adoption.

## 2. Coordinate passenger acceptance with the pace of fleet transition

**Finding.** Under Profile M, increasing (p_A) from 0.40 to 1.00 improved service by 0.0087 at (q_A=0.25), compared with 0.0477 at (q_A=0.75).

**Implication.** Passenger eligibility has greater operational leverage when the platform depends more heavily on AV supply. A staged rollout can therefore coordinate fleet substitution with opt-in design, passenger communication, and retention of flexible HV capacity for ineligible orders.

**Boundary.** The experiment does not estimate passenger preferences or the effect of a real acceptance intervention; (p_A) is an exogenous scenario probability.

## 3. Treat capability improvement as mitigation, not automatic capacity replacement

**Finding.** At (q_A=0.75,p_A=0.70), moving from Profile C to A increased service by 0.0330, but the resulting performance remained below the all-HV benchmark of 0.7889.

**Implication.** Broader operational capability can recover assignment opportunities, yet fleet-replacement decisions should still be validated against service and expiration outcomes. Capability and penetration should be planned jointly rather than assuming a broader envelope guarantees one-for-one HV replacement.

**Boundary.** C/M/A are analytical reference profiles, not real certification or product categories.

## 4. Monitor operational-envelope families separately

**Finding.** Positive speed-exposure share changed from 0.7073 under C to 0.0006 under M and 0.0000 under A. Static and dynamic activity also declined but remained more relevant under the broader profiles.

**Implication.** The operational bottleneck can migrate as capability changes. Family-specific monitoring helps identify whether static network structure, predicted traffic dynamics, or speed-domain compatibility is limiting AV assignments. A single weighted score would conceal this shift.

**Boundary.** Exposure is reference-envelope utilization and must not be interpreted as a safety, accident, or failure probability.

## 5. Use finite exposure budgets as testable dispatch controls

**Finding.** REFERENCE retained 99.9% of UNCONSTRAINED service while reducing final static exposure by 9.6% and dynamic exposure by 5.6%. STRICT reduced service by 8.5% relative to UNCONSTRAINED and lowered AV assignment share by 0.1104.

**Implication.** In the tested central scenario, a finite family-specific budget provided a more balanced service–envelope outcome than either a zero-exposure boundary or unconstrained accumulation. Platforms can treat such budgets as transparent scenario-tested control parameters rather than hidden terms in one weighted objective.

**Boundary.** REFERENCE was calibrated once from a different frozen trajectory and is neither a transferable optimum nor a safety threshold.

## 6. Evaluate cost-aware dispatch within a fixed cost definition

**Finding.** Within each fixed \(\eta\), setting \(\epsilon_W=0.05\) instead of zero reduced normalized cost per matched order by 0.54%–1.60%, while P95 pickup changed by only -0.19 to +0.29 s. AV assignment-share changes ranged from -0.0174 to +0.0145.

**Implication.** A small local pickup-objective relaxation can create useful cost flexibility without a large matched-passenger P95 change in these scenarios. The direction of fleet-composition adjustment, however, depends on the assumed AV/HV cost ratio and should be examined pairwise.

**Boundary.** Normalized cost is not money, raw values are not comparable across different \(\eta\), and a 5% local relaxation is not a full-day pickup-time guarantee.
