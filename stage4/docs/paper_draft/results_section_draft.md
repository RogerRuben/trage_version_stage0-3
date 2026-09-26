# 5. Results

All results reported in this section are deterministic contrasts from the frozen Test31 scenario design. The 27 main scenarios form a complete 3 × 3 × 3 factorial grid over the AV active vehicle-hour share `q_A ∈ {0.25, 0.50, 0.75}`, the assumed capability profile `k ∈ {C, M, A}`, and the scenario-level AV acceptance probability `p_A ∈ {0.40, 0.70, 1.00}`. The benchmark, operational-envelope-policy, and cost-robustness scenarios are analyzed separately. The reported contrasts are descriptive and do not represent replication-based statistical or causal estimates.

## 5.1 Overall effect of AV fleet penetration

**Result.** The marginal mean service rate across the nine profile–acceptance combinations decreased monotonically as AV penetration increased: from 0.7258 at `q_A=0.25`, to 0.5984 at `q_A=0.50`, and to 0.3924 at `q_A=0.75` (Table 1; Figure 1). The 0.25 → 0.75 change was -0.3334, equivalent to a 45.9% relative reduction from the `q_A=0.25` level. The decline was present in every fixed profile–acceptance slice, although its magnitude varied. Thus, increasing the AV share of active vehicle-hours did not translate into proportional effective service capacity in the modeled Test31 mixed-fleet system.

**Result and interpretation of waiting outcomes.** Across the main factorial scenarios, matched-passenger P95 pickup times occupied a narrow 289.3–293.0 s range even though service rates ranged from 0.3544 to 0.7309. These conditional waiting-time quantiles therefore cannot be interpreted independently of the unserved fraction. Similar pickup times among passengers who were served coexisted with materially different expiration rates, making service rate and patience expiration the primary measures of system capacity in this experiment.

**Mechanism interpretation.** The penetration result is consistent with a gap between nominal active vehicle-hours and vehicle-hours that can be converted into feasible assignments at the required time and location. In the implemented system, assignment opportunities are conditioned jointly on passenger eligibility, AV readiness and operational-envelope compatibility. Consequently, a unit of nominal AV availability need not be operationally interchangeable with an empirically reconstructed HV service hour.

**Scope limitation.** This result applies to the frozen Test31 demand, supply reconstruction, patience rule, acceptance realization, and dispatch design. It does not imply that AV adoption is generally harmful or that AV technology is intrinsically less capable than human-driven service.

## 5.2 Passenger acceptance × AV penetration

**Result.** Passenger acceptance had a larger operational association at higher AV penetration (Table 2; Figure 2). For Profile M, the acceptance gain

\[
\mathrm{AcceptanceGain}(q)=SR(p_A=1.0,q)-SR(p_A=0.4,q)
\]

was 0.0087 at (q_A=0.25), but 0.0477 at (q_A=0.75). The corresponding descriptive difference-in-differences contrast was 0.0390. The same directional pattern occurred for Profiles C and A: their low-to-high-penetration interaction contrasts were 0.0311 and 0.0350, respectively.

**Mechanism interpretation.** When AVs account for a modest share of active vehicle-hours, rejected AV eligibility can more often be absorbed by the larger HV component. As the AV share rises, the same acceptance restriction applies to a larger portion of the nominal supply, so the scenario-level acceptance probability becomes more consequential for realized service. This interpretation is consistent with the frozen design; it does not decompose the service change into a causal acceptance contribution.

**Definition and limitation.** The parameter (p_A) is an exogenous scenario-level acceptance probability. Each order receives a realized binary eligibility indicator (a_o^A) under the frozen common-random-number realization. The probability itself is not binary and was not estimated from individual Xi'an passenger-choice data.

## 5.3 Capability-dependent mitigation

**Result.** A more permissive assumed capability profile improved service within otherwise fixed high-penetration scenarios, but the gain was smaller than the fleet-composition loss. At (q_A=0.75) and (p_A=0.70), moving from Profile C to Profile A increased service rate by 0.0330 (Table 2). Capability gains were generally larger at high than at low penetration; nevertheless, the A-profile service rate in this slice remained well below the all-HV benchmark.

**Mechanism interpretation.** Moving from C to M or A expands the set of assignments that lie within the assumed static, dynamic, and speed envelopes. This mitigates capability-dependent assignment restrictions. It does not, however, remove passenger acceptance constraints, the temporal structure of supply, or all other feasibility conditions. Capability improvement therefore has operational value without guaranteeing parity with a flexible all-HV benchmark.

**Scope limitation.** C, M, and A are nested analytical capability/reference profiles in the implemented framework. They are not commercial certification classes, and their contrasts should not be interpreted as measured performance differences among real AV products.

## 5.4 Benchmark comparison and temporal mechanism

**Benchmark result.** The all-HV benchmark served 0.7889 of requests (Table 3). For Profile M with (p_A=0.70), service fell from 0.7297 at (q_A=0.25), to 0.6044 at (q_A=0.50), and to 0.4013 at (q_A=0.75). The all-AV M composition extreme served 0.1515. Equalized active vehicle-hours therefore did not produce equal effective service capacity once availability timing, passenger compatibility, and AV route eligibility were considered jointly. The all-AV case is a composition extreme, not a technological performance upper bound.

**Temporal result.** The five temporal anchors displayed the same ordering during the defined morning and evening peak windows (Table 3; Figure 3). Peak-window service rates were 0.7705 for BENCH_HV, 0.6959 for MAIN_Q25_M_P70, 0.5661 for MAIN_Q50_M_P70, 0.3743 for MAIN_Q75_M_P70, and 0.1381 for BENCH_AV_M. Mean queue pressure increased along this sequence from 0.1222 to 0.1512, 0.1921, 0.2298, and 0.2566. Meanwhile, mean logged total available stock was 378.5, 370.6, 377.6, 413.5, and 467.0, respectively.

**Mechanism interpretation.** Service declined and queue pressure rose even when more vehicles were recorded as available. The temporal evidence is therefore more consistent with an effective-service-capacity constraint than with a simple shortage of total logged vehicle stock. Within the modeled design, passenger acceptance eligibility, AV readiness or operational-envelope eligibility, and the time alignment between reconstructed HV sessions and full-horizon AV availability are plausible channels. Their separate contributions cannot be identified from the frozen outputs.

**Evidence boundary.** Available stock was logged only as a total and not separately for HVs and AVs. The results consequently do not support a vehicle-type-specific inventory trajectory or a causal decomposition of the service gap. No such decomposition is inferred here.

## 5.5 Capability-dependent operational-envelope dimensions

**Result.** The relevance of the three operational-envelope families changed substantially across assumed capability profiles (Table 4; Figure 4). For Profile C, the mean shares of assignments with positive static, dynamic, and speed exposure were 0.9248, 0.7739, and 0.7073. For Profile M, the corresponding shares were 0.8698, 0.4764, and 0.0006. For Profile A, they were 0.6394, 0.1822, and 0.0000. Speed exposure was therefore active under C, nearly inactive under M, and inactive under A, while static and dynamic activity also declined as the assumed envelope expanded.

Within each profile, variation over penetration and acceptance was smaller than the cross-profile shifts. For example, the range of q-marginal positive activity was 0.0000–0.0240 across the nine profile–family combinations, whereas the C-to-A differences in mean positive activity were 0.2854 for static, 0.5917 for dynamic, and 0.7073 for speed. The dominant envelope dimensions were therefore primarily capability-dependent under the frozen scenario grid.

**Mechanism interpretation.** The family-specific representation avoids treating operational compatibility as a single weighted score. A dimension that is restrictive under one assumed profile can become nearly irrelevant under another, leaving different dimensions to govern the set of eligible AV assignments. The speed-family transition provides the clearest example of this adaptive bottleneck pattern.

**Scope limitation.** Positive exposure denotes utilization of a reference-envelope dimension. It is not an accident probability, failure probability, safety certificate, or observed hazardous event.

## 5.6 Operational-envelope policy trade-off

**Result.** The STRICT, REFERENCE, and UNCONSTRAINED policies produced service rates of 0.5532, 0.6038, and 0.6044, respectively, with AV assignment shares of 0.0113, 0.1244, and 0.1217 (Table 5; Figure 5). Relative to UNCONSTRAINED, REFERENCE reduced service by 0.0007, or approximately 0.1%, while lowering final static exposure by 9.6% and dynamic exposure by 5.6%. Its AV assignment share was 0.0027 higher. STRICT reduced service by 0.0512 (8.5%) and lowered the AV assignment share by 0.1104 relative to UNCONSTRAINED.

**Mechanism interpretation.** A finite family-specific cumulative budget altered assignment composition and reduced exposure while retaining almost all of the unconstrained service level in the tested central scenario. By contrast, STRICT imposed a zero-exposure boundary and left very few AV assignments available. REFERENCE thus represented a tested middle position between zero-exposure operation and unconstrained exposure accumulation.

**Calibration provenance.** The REFERENCE \(\Gamma\) values were calibrated once from the (q_A=0.25), Profile M, (p_A=1.0), UNCONSTRAINED trajectory and then held fixed. They were not recalibrated on the (q_A=0.50,p_A=0.70) policy-comparison scenario. Positive remaining slack and zero numerically binding epochs do not imply that the budget had no assignment effect because discrete assignment choices can change before the cumulative boundary is reached.

**Scope limitation.** STRICT is a zero-exposure policy boundary, not a safe or certified operating policy. Similarly, REFERENCE is an operational-envelope budget evaluated in this scenario, not a validated safety threshold.

## 5.7 Cost robustness

**Result.** Within each fixed normalized AV-to-HV cost ratio \(\eta\), allowing a local pickup-objective relaxation of \(\epsilon_W=0.05\) instead of zero reduced normalized operating cost per matched order by 0.54%–1.60% (Table 6). Across the four within-\(\eta\) pairs, service-rate changes ranged from -0.0051 to +0.0041, AV assignment-share changes ranged from -0.0174 to +0.0145, and P95 pickup-time changes ranged from -0.19 to +0.29 s.

The assignment response was not uniform across cost ratios. AV share increased by 0.0145 and 0.0129 at \(\eta=0.50\) and 0.75, but decreased by 0.0041 and 0.0174 at \(\eta=1.00\) and 1.25. The direction of the composition adjustment therefore depended on the assumed normalized relative operating cost.

**Mechanism interpretation.** The local lexicographic relaxation gives the dispatch kernel a small feasible region in which it can exchange pickup optimality for lower normalized operating cost. Because vehicle states propagate through the rolling process, the full-day pickup objective and service response emerge endogenously and need not move in a common direction across cost ratios.

**Scope limitation.** Raw normalized cost levels are not ranked across \(\eta\), because changing \(\eta\) changes the objective definition itself. The 5% tolerance applies to the local epoch-level pickup objective; it does not guarantee that full-day pickup time worsens by no more than 5%. Costs are normalized analytical quantities rather than calibrated monetary estimates.
