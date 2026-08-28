# 6. Discussion

## 6.1 Why higher AV penetration does not automatically improve service

**Result synthesis.** Under the frozen Test31 design, increasing the AV share of active vehicle-hours from 0.25 to 0.75 reduced marginal mean service rate from 0.7258 to 0.3924. The temporal anchors further showed that service could fall while both queue pressure and logged total available stock increased. Nominal vehicle-hours and nominal idle stock were therefore insufficient descriptions of the supply that the platform could actually deploy.

**Mechanism interpretation.** The distinction between nominal and effective service capacity is central to the mixed-fleet setting. An available vehicle contributes to effective capacity only if it is compatible with the order, temporally and spatially positioned to serve it within patience, and eligible under the relevant operational constraints. In the implemented framework, full-horizon AV availability expands nominal supply but does not remove passenger acceptance or AV route-eligibility conditions. Conversely, reconstructed HV sessions encode an empirical time pattern of service availability. These modeled asymmetries are consistent with the observed contrast, but the frozen logs do not identify their separate causal shares.

**Managerial implication.** A platform evaluating staged AV deployment should monitor the conversion of nominal AV hours into feasible assignments, rather than treating AV and HV hours as automatically interchangeable. Service rate, patience expiration, feasible-candidate availability, and queue pressure provide more informative operational indicators than fleet share alone.

**Limitation.** The conclusion concerns one modeled day and one frozen supply construction. It should not be generalized into a claim that AV penetration inherently reduces service. Alternative fleet scheduling, charging, repositioning, vehicle capability, or passenger response could change the effective-capacity relationship.

## 6.2 Why acceptance matters more in AV-heavy fleets

**Result synthesis.** For Profile M, raising the scenario-level acceptance probability from 0.40 to 1.00 increased service by 0.0087 at (q_A=0.25) and by 0.0477 at (q_A=0.75). The positive 0.0390 descriptive interaction contrast indicates that the operational association of acceptance strengthened as the AV share increased.

**Mechanism interpretation.** At low AV penetration, the larger HV component can absorb more orders that are ineligible for AV assignment. At high penetration, the same order-level eligibility restriction excludes a larger share of nominal supply from serving a rejected order. Acceptance and penetration are therefore complements in an operational sense within this design: the value of wider acceptance rises when the system relies more heavily on AV hours.

**Managerial implication.** Fleet transition and passenger adoption should not be planned as independent workstreams. A platform increasing AV penetration may need to coordinate passenger communication, service differentiation, and opt-in design with fleet deployment, while retaining enough flexible capacity to accommodate non-accepting orders. The experiment supports coordination as a design principle, not a particular marketing intervention.

**Limitation.** Acceptance is exogenous and scenario-based. The realized order indicator is binary, but (p_A) is a probability and was not estimated from observed passenger choices. The results consequently do not quantify welfare, preference heterogeneity, or the effect of a real acceptance intervention.

## 6.3 Capability improvement versus fleet-composition effects

**Result synthesis.** More permissive assumed capability profiles improved service, especially at high penetration. At (q_A=0.75,p_A=0.70), the C-to-A improvement was 0.0330. This gain was operationally meaningful but much smaller than the 0.3334 marginal service decline between (q_A=0.25) and 0.75, and it did not close the gap to the 0.7889 all-HV benchmark.

**Mechanism interpretation.** Capability improvement expands the set of operationally compatible routes and assignments, thereby recovering opportunities that narrower profiles exclude. It cannot by itself correct all temporal supply mismatch, passenger ineligibility, or patience-related losses. This explains why capability can mitigate restrictions without neutralizing aggressive HV-to-AV active-hour substitution.

**Managerial implication.** Capability investment and fleet composition should be evaluated jointly. A broader operational envelope can improve the productivity of AV hours, but a platform should not infer from that improvement that any target penetration can preserve service. Deployment targets can instead be conditioned on demonstrated effective capacity under the relevant demand, acceptance, and dispatch regime.

**Limitation.** Profiles C, M, and A are assumed nested analytical envelopes rather than observed commercial systems. The results measure consequences of those frozen definitions and do not establish how much a specific engineering upgrade would cost or which real vehicle would satisfy a profile.

## 6.4 Multi-dimensional operational-envelope relevance and adaptive bottlenecks

**Result synthesis.** The positive-exposure pattern shifted strongly across profiles. Speed exposure was common under C (0.7073), nearly absent under M (0.0006), and absent under A (0.0000). Static exposure remained active but declined from 0.9248 to 0.6394, while dynamic exposure declined from 0.7739 to 0.1822. The restrictive dimension was therefore not invariant to assumed capability.

**Mechanism interpretation.** Retaining separate static, dynamic, and speed families reveals an adaptive bottleneck structure. When one envelope expands enough that a family rarely becomes active, another family can remain relevant for assignment decisions. A single weighted exposure index would obscure this shift and would require arbitrary trade-off weights across quantities with different operational meanings.

**Managerial implication.** Operational monitoring and capability development should remain family-specific. The appropriate emphasis may shift from speed-domain compatibility toward intersection/static or traffic-dynamic compatibility as capability changes. This allows the platform to diagnose why AV utilization remains constrained instead of attributing all restrictions to a single composite score.

**Limitation.** Exposure indicates reference-envelope utilization, not physical danger. A positive value does not represent an accident, a failure, or a certification violation, and the profile comparison does not validate real-world safety performance.

## 6.5 Operational value of reference exposure budgets

**Result synthesis.** REFERENCE preserved 0.6038 service compared with 0.6044 under UNCONSTRAINED, an absolute loss of 0.0007 (approximately 0.1%), while reducing final static and dynamic exposure by 9.6% and 5.6%. STRICT produced lower service (0.5532) and almost eliminated AV assignments, whereas REFERENCE retained an AV assignment share of 0.1244.

**Mechanism interpretation.** Family-specific cumulative budgets can redirect a small set of assignments before the final budget becomes numerically binding. This explains why REFERENCE can change cumulative exposure despite positive slack and zero binding-epoch counts. STRICT illustrates the opposite boundary: requiring zero exposure removes most AV assignment opportunities and imposes a larger service cost.

**Managerial implication.** Within the tested central scenario, a finite reference budget offered a practical operational compromise between unconstrained utilization and a zero-exposure boundary. This supports using explicit family budgets as dispatch-control parameters that can be stress-tested against service objectives, rather than folding all compatibility dimensions into one opaque weight.

**Limitation.** The REFERENCE budget was calibrated once from the (q_A=0.25), M, (p_A=1.0), UNCONSTRAINED trajectory and then held fixed. Its favorable trade-off on the central policy scenario is not evidence that the same budget is optimal, safe, or transferable to another city, day, demand level, or capability definition.

## 6.6 Platform implications for staged AV deployment

**Result synthesis.** Taken together, the factorial, temporal, family, policy, and cost contrasts show that penetration, acceptance, capability, and operational-envelope policy interact. No single dimension determined system performance. Cost-aware tie-breaking further changed AV share in both directions across \(\eta\), while keeping P95 pickup changes within -0.19 to +0.29 s in the tested pairs.

**Mechanism interpretation.** A mixed-fleet platform is a coupled rolling system: eligibility affects the candidate set, capability affects route compatibility, exposure budgets shape assignment choices, and each assignment changes later vehicle states. Static one-factor reasoning therefore misses the feedback through which local decisions determine full-day outcomes.

**Framework contribution.** The evidence is enabled by the implemented combination of reconstructed empirical HV supply and full-horizon AV supply; hard route feasibility plus continuous static, dynamic, and speed suitability; scenario-level probabilistic acceptance with order-level realized eligibility; and family-specific cumulative exposure budgets. The factorial design then exposes how penetration, capability, acceptance, and policy interact within that framework. This statement describes the implemented contribution and does not assert novelty beyond it.

**Managerial implication.** A staged deployment process can coordinate four decisions: the share and temporal placement of AV hours, the capability profile supported by the operational domain, passenger acceptance arrangements, and ODD-aware dispatch controls. The tested evidence favors joint scenario evaluation over maximizing nominal AV share or minimizing a single local objective. The result is a design implication for this framework, not a universal deployment prescription.

**Limitation.** The analysis does not estimate monetary benefits, social welfare, safety outcomes, induced demand, congestion feedback, or long-run behavioral adaptation. Those questions require evidence and model components outside the frozen Stage4 experiment.

## 6.7 Limitations and external validity

The findings should be read within eight principal boundaries. First, the evaluation uses a single empirical Test31 case in Xi'an, so day-to-day and cross-city external validity remains untested. Second, (p_A) is an exogenous scenario probability rather than an estimated individual choice model. Third, C/M/A are assumed nested reference profiles rather than commercial certification classes. Fourth, operational-envelope exposure is neither a safety nor failure probability.

Fifth, HV supply is reconstructed from empirical service sessions and represents an effective operating fleet rather than registered fleet size. Sixth, idle AVs are available across the full horizon; charging, state of charge, maintenance, depot constraints, and duty scheduling are absent. Seventh, the simulator does not include endogenous network congestion or traffic feedback from dispatch decisions. Eighth, the operating-cost terms are normalized analytical quantities rather than calibrated monetary economics. These limitations constrain interpretation but do not alter the internal descriptive comparisons among the frozen scenarios.
