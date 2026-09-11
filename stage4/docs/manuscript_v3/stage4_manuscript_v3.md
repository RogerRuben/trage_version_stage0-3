# Effective Service Capacity in Mixed-Autonomy Ride-Hailing: State-Dependent Serviceability and Rolling Assignment

## Abstract

Fleet transitions are often described through the share of autonomous vehicles, yet heterogeneous serviceability can prevent nominal supply from replacing an equivalent amount of service. We formulate effective capacity through a state-dependent request–vehicle compatibility graph and its maximum matching, separating nominal availability, instantaneous matchable demand, and realized service over a rolling horizon. Passenger acceptance, route-level operational compatibility, information availability and pickup deadlines determine eligible assignments, while family-specific exposure controls preserve distinct operating constraints. In an empirical ride-hailing replay, mean service rate decreases from 0.726 to 0.392 as the baseline-normalized autonomous active-hour share increases from one quarter to three quarters under zero request lead and five-minute pickup patience. Matching-capacity analysis shows that substantive passenger and operational gates remove maximum-matchable demand, rather than only redundant candidate edges. Broader passenger acceptance and operational capability partially recover service. The magnitude of capacity loss, however, varies materially with reconstructed request timing and remaining patience. Fleet planning should therefore compare effective substitutable service capacity rather than assume equivalence from vehicle counts or available hours alone.

**Keywords:** effective service capacity; mixed-autonomy ride-hailing; state-dependent serviceability; bipartite matching; rolling assignment; fleet substitution

## 1. Introduction

When a ride-hailing platform replaces human-driven vehicles (HVs) with autonomous vehicles (AVs), the strategic question is usually expressed as a fleet percentage. The operational question is more demanding: how much service can those vehicles substitute for? An available vehicle contributes little to a waiting passenger if the passenger will not accept it, the route is incompatible with its operating envelope, evidence needed to assess the route is absent, or pickup cannot occur before the deadline. Even individually compatible vehicles can compete for the same small set of requests. Nominal fleet supply is therefore not equivalent to effective service capacity when serviceability is state dependent.

Existing research already models important forms of mixed-fleet heterogeneity. Mo, Chen and Zhang examine mixed on-demand services with congestion and market interactions; Ao, Lai and Li incorporate human-driver scheduling, strategic relocation and platform AV control. Dynamic matching and rebalancing studies also show that spatial allocation and assignment structure matter for service. These contributions should not be characterized as uniformly assuming interchangeable vehicles. Our narrower question concerns how request-level operating restrictions translate into an explicit instantaneous capacity object, complementing market-equilibrium and fleet-control perspectives. [Mo et al. (2022)](https://doi.org/10.1016/j.trb.2022.01.003); [Ao et al. (2024)](https://doi.org/10.1016/j.tre.2024.103680); [Alonso-Mora et al. (2017)](https://doi.org/10.1073/pnas.1611675114); [Pavone et al. (2012)](https://doi.org/10.1177/0278364912444766).

A second distinction concerns vehicle serviceability. Operational-design-domain descriptions recognize that automated operation is conditional on the operating environment, while recent network-design work explicitly considers where robotaxis can operate and which paths satisfy service requirements. We study a complementary operational link: route conditions and evidence determine a request–vehicle compatibility graph, and the connectivity of that graph determines maximum-matchable demand. The analytical capability envelopes used here are not manufacturer-certified domains. The purpose is to represent serviceability for fleet operations, not to infer autonomous-driving safety. [SAE International (2021)](https://doi.org/10.4271/J3016_202104); [AVSC (2020)](https://saemobilus.sae.org/reports/avsc-best-practice-describing-operational-design-domain-conceptual-framework-lexicon-avsc00002202004); [Li and Zardini (2026, preprint)](https://arxiv.org/abs/2602.19341).

The central framework distinguishes nominal available supply, a state-dependent compatibility graph, instantaneous effective matching capacity, and rolling realized service:
\[
S_t^{nom}\ \longrightarrow\ G_t\ \longrightarrow\ C_t^{eff}=\nu(G_t)\ \longrightarrow\ Y_{1:T}.
\]
The arrows represent dependencies, not proportional conversions. Information can alter the graph as well as assignment costs: transportation research on demand information and decision-focused optimization motivates examining downstream decisions rather than forecast error alone. Here prediction supports graph construction; it is not a separate contribution or a new decision-focused learning algorithm. [Wen et al. (2019)](https://doi.org/10.1016/j.tra.2019.01.018); [Elmachtoub and Grigas (2022)](https://doi.org/10.1287/mnsc.2020.3922); [Wilder et al. (2019)](https://doi.org/10.1609/aaai.v33i01.33011658).

This paper asks when nominal AV supply fails to translate into service and which operating mechanisms determine or mitigate that conversion. It makes three contributions. Conceptually, it operationalizes effective matching capacity while distinguishing it from available hours and realized service. Methodologically, it constructs a state-dependent compatibility graph and embeds it in rolling assignment with hard readiness and separate continuous exposure controls. Empirically, it uses a real ride-hailing system to show genuine matching-capacity attrition, partial recovery through acceptance and capability, and substantial timing dependence. Maximum matching itself is classical; the contribution is its use as an explicit intermediate operational quantity connecting heterogeneous supply to service. Sections 2–3 formulate the model, Sections 4–5 describe implementation and design, and Sections 6–8 present findings and implications.

## 2. Effective capacity in heterogeneous ride-hailing fleets

### 2.1 Nominal supply

Let \(\widetilde{\mathcal V}_t\) be vehicles active in the operating schedule at epoch \(t\), including vehicles currently serving a request. Nominal active supply is
\[
S_t^{nom}=|\widetilde{\mathcal V}_t|,\qquad
H^b=\int_0^T|\widetilde{\mathcal V}_t^b|\,dt,\quad b\in\{H,A\}.
\]
The active-hour quantities have units of vehicle-hours. The dispatchable set \(\mathcal V_t\subseteq\widetilde{\mathcal V}_t\) contains idle vehicles available for a new assignment; it is not the full active fleet. To compare transition scenarios, define \(q_A=H^A/H^{base}\), where \(H^{base}\) is the all-HV baseline's active-hour denominator. This is not necessarily the within-scenario fraction \(H^A/(H^A+H^H)\).

Equal counts or active hours need not imply equal serviceability. Substitution changes the types, locations and availability schedules of the vehicles offered to the waiting demand. We do not treat an AV label change and an availability-policy change as the same intervention.

### 2.2 State-dependent compatibility graph

Let \(\mathcal O_t\) contain released, unassigned, unexpired requests. Order \(o\) has release \(r_o\), pickup deadline \(D_o\), origin, destination and a route representation. Define
\[
G_t=(\mathcal O_t,\mathcal V_t,\mathcal A_t),\qquad
\mathcal A_t=\{(o,v):a_{ovt}=1\},
\]
where \(a_{ovt}\) indicates pairwise eligibility under the current information and operating rules. These rules include passenger acceptance for AVs, route readiness, required evidence, spatial consideration, routing success, pickup feasibility and applicable session-end admission. A continuous envelope exceedance is retained as an exposure quantity unless an explicit policy excludes it.

A gate sequence produces \(\mathcal A_t^{g+1}\subseteq\mathcal A_t^g\). At each gate,
\[
E_t^g=|\mathcal A_t^g|,\qquad
U_t^g=|\{o:\deg_{G_t^g}(o)>0\}|,\qquad
M_t^g=\nu(G_t^g).
\]
These have distinct meanings: feasible pairs, orders with any option, and simultaneously matchable orders. Attribution follows the implemented gate order; it is not a decomposition of independent causal effects. Algorithmic candidate compression is identified separately from substantive serviceability.

### 2.3 Effective matching capacity

The instantaneous graph-based capacity is
\[
C_t^{eff}=\nu(G_t)=
\max_{x\in\{0,1\}^{|\mathcal A_t|}}
\left\{\sum_{(o,v)\in\mathcal A_t}x_{ov}:
\sum_vx_{ov}\le1,\ \sum_ox_{ov}\le1\right\}.
\]
It is bounded by both demand with an option and available vehicles. For example, ten requests all connected to the same vehicle have ten edges and ten eligible requests but capacity one. Removing nine edges in a highly redundant graph may instead leave capacity unchanged.

For AV-specific diagnosis, \(G_t^A\) restricts the vehicle side to AVs and \(C_t^{eff,A}=\nu(G_t^A)\). The matching-capacity results below concern this AV subgraph, not total mixed-fleet capacity. In general \(C_t^{eff}\) is not the sum of the HV and AV subgraph capacities because the same requests may appear in both.

Graph capacity represents pairwise serviceability before coupled policy constraints. For the cumulative exposure constraints introduced below, define the policy-feasible set \(\mathcal F_t(\Gamma)\) and, if needed,
\[
C_t^{eff,\Gamma}=\max_{x\in\mathcal F_t(\Gamma)}\sum x_{ov}
\le C_t^{eff}.
\]
A system-wide exposure budget couples assignments and cannot generally be represented by independently deleting edges. The measured graph capacity is therefore not silently equated with the budget-constrained dispatch solution.

### 2.4 Rolling realized service

Let \(y_o=1\) if request \(o\) is counted as served in the rolling evaluation and zero otherwise. Then
\[
Y_{1:T}=\sum_{o\in\mathcal O_{1:T}}y_o,\qquad
R_{1:T}=Y_{1:T}/|\mathcal O_{1:T}|.
\]
Requests are counted once, whereas a waiting request can appear in several graphs. Thus \(\sum_t C_t^{eff}\) is not daily service or a count of distinct orders. Assignments change subsequent vehicle positions, remaining availability and future competition; locally similar capacities can generate different rolling outcomes. A fixed-state change in \(C_t^{eff,A}\) does not itself identify a daily service effect.

The experiments address different links of the framework. Fleet-transition scenarios compare nominal supply with \(Y\). Gate diagnostics measure how graph restrictions reduce \(C^{eff,A}\). Acceptance and capability alter graph eligibility or exposure values, and exposure budgets alter jointly feasible assignments. Timing and information sensitivity examine how the same physical setting supports different compatibility and matching outcomes.

## 3. State-dependent mixed-fleet dispatch

### 3.1 Passenger and operational eligibility

Each request receives a common uniform draw \(u_o\). At acceptance level \(p_A\), its AV pairs pass the passenger gate if \(u_o\le p_A\); HV pairs do not require AV acceptance. This is an exogenous scenario parameter, not an estimated choice model.

At epoch \(t\), routed pickup estimate \(\widehat{\tau}_{ovt}\) must satisfy
\[
\widehat{\tau}_{ovt}\le D_o-t.
\]
Spatial neighborhoods expand as requests wait, and at most K nearby candidates per request proceed to the routed sparse graph. The search rule and K are held fixed across each paired comparison. Requests with \(0<D_o-t\le d^{crit}\) are critical; previously unsuccessful requests receive carry-over priority.

Availability policy is explicit. For an empirical-session vehicle ending at \(b_v\), predicted pickup and service must finish within its admission window:
\[
t+\widehat{\tau}_{ovt}+\widehat{s}_{ovt}\le b_v.
\]
Full-horizon AV availability does not inherit this empirical-session admission rule. Neutral label comparisons hold policy fixed; fleet-transition scenarios retain the intended policy difference.

### 3.2 Hard and continuous compatibility

For a route and profile \(k\), hard readiness is \(h_{ok}\in\{\text{feasible},\text{unknown},\text{infeasible}\}\). Known incompatible directions or prohibited movements can produce hard infeasibility. Missing critical identity or evidence remains unknown. The conservative AV admission rule requires hard readiness and complete required evidence. These are operational classifications under a specified representation, not safety certification.

For descriptor \(d_{oj}\) and positive profile cap \(B_{kj}\),
\[
r_{ojk}=d_{oj}/B_{kj},\qquad
\rho^f_{ok}=\max_{j\in\mathcal J_f}r_{ojk},\qquad
e^f_{ov}=[\rho^f_{ok(v)}-1]_+,
\]
for \(f\in\{\mathrm{static},\mathrm{dynamic},\mathrm{speed}\}\). The implemented route-level exposure is identical across vehicles sharing the route/profile, although an arc index permits heterogeneous profiles. Overall utilization is the maximum of the three family utilizations; it is not a weighted risk score. Exceeding a cap is continuous exposure, not automatically hard infeasibility. The separate treatment retains both operational meaning and attribution.

### 3.3 Rolling assignment

For \((o,v)\in\mathcal A_t\), binary \(x_{ov}\) obeys
\[
\sum_{v:(o,v)\in\mathcal A_t}x_{ov}\le1,\qquad
\sum_{o:(o,v)\in\mathcal A_t}x_{ov}\le1.
\]
Within these and the enabled policy constraints, the objective hierarchy is
\[
\operatorname{lexmax}
\left(
\sum c_o x_{ov},\
\sum x_{ov},\
\sum b_o x_{ov},\
-\sum\widehat{\tau}_{ovt}x_{ov}
\right),
\]
where \(c_o\) and \(b_o\) indicate critical and carry-over status. Each later level retains preceding optima. An optional operating-cost level follows the pickup objective and allows only the explicitly bounded pickup-quality relaxation in Appendix D; it does not relax the earlier service-count optima.

The assignment is unpooled: each available vehicle receives at most one current request. This differs from trip–vehicle formulations supporting pooled multi-request trips, such as Alonso-Mora et al. (2017). The use of a compatibility graph here is not a claim to introduce dynamic assignment. Its role is to expose how heterogeneous serviceability restricts the service set before selection. [Alonso-Mora et al. (2017)](https://doi.org/10.1073/pnas.1611675114).

### 3.4 Cumulative exposure control

Let \(Z_t^f\) be system-wide cumulative exposure in family \(f\) before epoch \(t\), and \(N_t^A\) the cumulative number of AV assignments. Let \(\mathcal A_t^A\) be AV arcs. For every enabled family, dispatch imposes
\[
Z_t^f+\sum_{(o,v)\in\mathcal A_t^A}e^f_{ov}x_{ov}
\le
\Gamma_f\left(N_t^A+\sum_{(o,v)\in\mathcal A_t^A}x_{ov}\right). \tag{1}
\]
The same AV-assignment count normalizes each family. The states update additively after selection. Equation (1) bounds day-to-date **mean exposure per AV assignment**, not exposure per vehicle, per hour, or per distance. It allows a high-exposure route to be balanced by lower exposure in the same family, while preventing compensation across families.

Strict control sets all \(\Gamma_f=0\); from zero initial exposure and nonnegative excess, only zero-exposure AV selections satisfy it. Reference control uses pre-specified positive family limits transferred from a designated calibration run. Unconstrained control disables these continuous rows while retaining passenger, hard-readiness and evidence gates. These are operating postures, not optimized manufacturer capability limits.

### 3.5 Structural properties

**Proposition 1 — Cumulative family-exposure guarantee.** Under additive updates, any selection satisfying (1) gives \(Z_{t+1}^f\le\Gamma_fN_{t+1}^A\). If \(N_{t+1}^A>0\), cumulative mean family exposure is at most \(\Gamma_f\). The guarantee concerns accumulated assignment exposure, not each trip or a safety probability.

**Proposition 2 — Same-epoch feasible-set monotonicity.** Fix vehicles, waiting requests, candidates, acceptance, pickup estimates, exposure values and all non-budget constraints. For componentwise \(\Gamma'\ge\Gamma\), \(\mathcal F_t(\Gamma)\subseteq\mathcal F_t(\Gamma')\). Relaxing limits cannot worsen the highest-priority lexicographic optimum in that state. This does not guarantee monotone lower-priority objectives or daily service: a different selection changes subsequent states.

**Proposition 3 — Separate budgets versus a weighted scalar.** Family means \(\bar e_f\le\Gamma_f\) imply \(\sum_fw_f\bar e_f\le\sum_fw_f\Gamma_f\) for nonnegative weights, but the converse fails. Two unit family caps, equal weights and \((\bar e_1,\bar e_2)=(2,0)\) provide a counterexample. Separate limits preserve family meaning without claiming globally optimal caps.

Full proofs and Proposition 4, the local pickup-quality guarantee for the optional cost level, are in Appendix D. These elementary structural statements explain the implemented control, rather than constitute a new matching theorem.

## 4. Empirical implementation

### 4.1 Xi'an ride-hailing data

The empirical application uses ride-hailing GPS trajectories from Xi'an, China, in October 2016. The evaluation set contains 30,000 quality-screened orders from 31 October. All scenarios use the same set. Trajectory endpoints provide an observed departure/boarding proxy and completion location, not independently measured platform request times.

Training precedes evaluation, with later dates used for validation and calibration. The evaluation day is a common temporal benchmark across system versions, not an untouched holdout for every upstream development choice. This study evaluates pre-specified operational contrasts on that day, rather than estimating a population-wide causal response. The retained subset prioritizes supported routes and direct timing observations; it is not a census of all raw requests. Appendix A describes selection and route quality.

### 4.2 Directed route representation

GPS trajectories are map matched to a directed OpenStreetMap road network. Historical reverse directions are preserved rather than projected onto current forward edges because direction identity affects AV compatibility. Detailed map-matching validation is in Appendix A.

Static descriptors summarize intersection approaches, movements, boundary road-class diversity and extent. Boundary diversity uses incoming/outgoing edges, so a single-node intersection can have nonzero diversity without internal edges. Dynamic descriptors characterize crawl, stops, speed variability and acceleration variability. Route summaries retain family distinctions instead of collapsing them into one score. Routes follow the existing pre-specified selection/fallback rule and are not replanned in response to dispatch outcomes.

The three analytical capability profiles—Conservative, Moderate and Advanced—have nested descriptor caps. Quantile anchors are marginal descriptor thresholds, not promised joint route-acceptance rates. Their operational interpretation follows the distinction between an environment description and a vehicle's actual validated operating domain; the profiles are not certifications. [AVSC (2020)](https://saemobilus.sae.org/reports/avsc-best-practice-describing-operational-design-domain-conceptual-framework-lexicon-avsc00002202004).

### 4.3 Decision-time operational prediction

A multivariate prediction model supplies travel-time/pace and raw operating-condition estimates. Training-only normalization, temporal calibration and distribution mappings are retained for evaluation. The decision representation uses predicted route progression rather than completed-route outcomes. Realized service duration is used for post-assignment progression, not substituted for a missing dispatch prediction.

Predictions enter graph evidence and exposure calculations. We therefore examine whether changing information changes candidates and selected assignments; we do not claim to train the model through dispatch loss. This distinction separates our validation from decision-focused learning methods and from independent evidence of superior realized decisions. [Elmachtoub and Grigas (2022)](https://doi.org/10.1287/mnsc.2020.3922); [Wilder et al. (2019)](https://doi.org/10.1609/aaai.v33i01.33011658). Descriptor construction and prediction validation details are in Appendix B.

### 4.4 Fleet reconstruction

HVs inherit empirical driver-session availability; AVs are available over the modeled full horizon. The all-HV denominator is \(H^{base}=12{,}279.336389\) vehicle-hours. Each transition targets \(q_A=H^A/H^{base}\), with realized availability checked against this denominator rather than inferred from vehicle counts.

Changing composition therefore changes the mix of availability schedules and spatial states, not only labels. All scenarios retain their pre-specified fleet construction. The controlled neutral-label comparison instead preserves availability policy and positions to isolate label-only behavior. Supply construction and representativeness details are in Appendix A.

## 5. Experimental design

### 5.1 Main fleet-transition scenarios

The main replay treats the observed trajectory departure/boarding proxy as request release: **zero request lead and 300-second pickup patience**, with decisions every 30 seconds. This is a modeling assumption, not an observed request-time fact. Main outcomes are conditional on it.

A factorial combines \(q_A\in\{.25,.50,.75\}\), three analytical profiles and acceptance \(p_A\in\{.40,.70,1.00\}\), giving 27 mixed-fleet settings within 41 pre-specified full-day scenarios. All-HV and all-AV profile cases are composition benchmarks. Acceptance uses common random draws. The main factorial disables continuous budgets and the additional cost objective; route policy and candidate rules are unchanged. Service rate is served requests divided by the same 30,000 orders.

At \(q_A=.50\), Moderate capability and \(p_A=.70\), strict, reference and unconstrained exposure policies compare alternative operating limits. Reference limits are calibrated once using the designated q_A=.25, Moderate, universal-acceptance run and transferred without tuning to central outcomes. All-AV cases are composition extremes, not performance ceilings.

### 5.2 Mechanism validation

Ten existing decision states span low-load, midday and evening periods. At each, AV gate graphs provide E, U and maximum matching M on the same arc set. Gate-level differences in M measure lost contemporaneous capacity rather than infer it from edge percentages. State sums are not daily unique orders.

A neutral identity comparison changes labels but preserves availability policy and physical inputs. Candidate and selected identities agree in all ten states. A separate candidate-compression check compares K=10/20/40/80 on the same states. Both are controls supporting mechanism interpretation, not independent fleet-transition experiments. Appendix C contains detailed outputs and definitions.

### 5.3 Sensitivity analyses

Because historical trajectories record a boarding proxy rather than request time, three pre-specified lead scenarios are reconstructed from training-day driver chains and applied to the same evaluation orders. Four physical states—q_A=.50/.75 at noon and 17:30, Moderate profile and acceptance .70—are compared under zero lead and the three alternatives.

Vehicles, baseline prior-assignment history, route descriptors, routing clock and operating rules are held fixed within each comparison. Release changes waiting membership and remaining patience. This conditional-state design does not reconstruct alternate fleet histories; its fixed descriptors are not asserted to be available at an earlier reconstructed release. It measures timing dependence, not earlier-release forecast performance or full-day service effects. Appendix E records the reconstruction and its limitations.

A ten-state information comparison substitutes historical estimates for model predictions while retaining physical and policy inputs. It measures graph and decision relevance, not superiority under a common independent outcome evaluator. Detailed target errors, numerical controls and cost robustness are supplementary material, not additional primary research questions.

## 6. Results

### 6.1 Nominal supply versus realized service

Under zero lead and 300-second patience, mean service rates across the factorial are .7258, .5984 and .3924 at q_A=.25, .50 and .75. The decrease from .25 to .75 is .3334, approximately 45.9% of the lower-share mean. This is an operational contrast on common demand, not a timing-invariant technological effect.

| Fleet condition | Service rate |
| --- | ---: |
| All HV | .7889 |
| q_A=.25, Moderate, acceptance .70 | .7297 |
| q_A=.50, Moderate, acceptance .70 | .6044 |
| q_A=.75, Moderate, acceptance .70 | .4013 |
| All AV, Moderate | .1515 |

The all-AV result illustrates a composition extreme without HV alternatives; it is not an upper bound or a prediction for mature autonomous technology. These daily results establish the supply-to-service pattern. The next analysis evaluates whether pairwise restrictions remove genuine matchable demand, rather than assuming this from service rates alone.

### 6.2 Matching-capacity conversion

The AV matching-capacity sequence over ten states is
\[
718\ \xrightarrow{\mathrm{passenger}}\ 493\
\xrightarrow{\mathrm{structural}}\ 238\
\xrightarrow{\mathrm{evidence}}\ 124\
\xrightarrow{\mathrm{patience}}\ 12.
\]
The associated order counts establish that losses concern potential service as well as candidate volume.

| Gate | E: AV edges | U: orders with an AV option | M: maximum AV matching |
| --- | ---: | ---: | ---: |
| Spatial opportunity | 95180 | 720 | 718 |
| Passenger compatible | 65444 | 493 | 493 |
| Structural ready | 31153 | 238 | 238 |
| Evidence complete | 15775 | 124 | 124 |
| Candidate compression | 1371 | 124 | 124 |
| Route returned | 1371 | 124 | 124 |
| Pickup/patience feasible | 48 | 12 | 12 |
| Solver eligible | 48 | 12 | 12 |

These are sums over states, not unique daily requests or selected assignments. Passenger, structural, evidence and patience gates remove 225, 255, 114 and 112 matching units in the sample. Candidate compression removes approximately 91% of pre-compression edges without changing final maximum matching at any of the tested K values in these states.

The evidence distinguishes redundant connectivity from usable capacity. Spatial U=720 but M=718 already shows competition for vehicles; a request having an option does not ensure all such requests can be served simultaneously. The graph sequence validates the serviceability mechanism locally, but it does not assign the entire daily service decline causally among gates. Repeated-epoch edge-survival percentages provide supplementary opportunity-volume evidence only (Appendix C).

### 6.3 Capacity recovery levers

Acceptance and capability affect different parts of serviceability. Under Moderate capability, increasing acceptance from .40 to 1.00 raises service rate by .0087 at q_A=.25 and .0477 at q_A=.75. At q_A=.75 and acceptance .70, moving from Conservative to Advanced capability raises service by .0330. These levers partly recover service, but do not eliminate the larger composition contrast under the main timing assumption.

A willing passenger does not create a compatible route, and a broader capability envelope does not ensure timely pickup. These changes alter graph eligibility or exposure severity; new edges may remain redundant for simultaneous matching. We report the measured service gains without claiming that every eligibility expansion produces a proportional capacity increase.

Exposure control addresses a different trade-off: the jointly admissible operating conditions of selected AV service. At the central comparison:

| Exposure policy | Service rate | AV service share |
| --- | ---: | ---: |
| Strict zero excess | .5532 | .0113 |
| Reference family limits | .6038 | .1244 |
| Unconstrained continuous exposure | .6044 | .1217 |

Relative to unconstrained operation, reference control reduces selected static and dynamic exposure by 9.6% and 5.6%, while nearly retaining service. The difference computed from rounded service rates need not equal the unrounded comparison. These are tested policy outcomes, not evidence that the reference limits are optimal or safe. Because the budgets couple assignments, the policy comparison concerns \(\mathcal F_t(\Gamma)\) and realized service, not simply removal of individually infeasible edges.

### 6.4 Timing and information dependence

Matching capacity remains constrained under alternative request leads, but its magnitude varies substantially. Across four physical states:

| Request timing | Pre-patience M | Final M | Aggregate patience retention |
| --- | ---: | ---: | ---: |
| Zero lead | 53 | 8 | 15.09% |
| Low lead scenario | 130 | 26 | 20.00% |
| Base lead scenario | 128 | 25 | 19.53% |
| High lead scenario | 156 | 40 | 25.64% |

Retention divides summed final M by summed pre-patience M. These four-state totals must not be pooled with the ten-state totals above. At noon, q_A=.50/.75 gives final M=4/3 under zero lead but 10/11 under high lead; at 17:30 the values are 1/0 and 8/11. Local ordering can change, but differing waiting cohorts and fixed fleet histories prevent inference of an order-level treatment effect or a reversed daily ordering. The evidence qualifies the mechanism's magnitude, not its existence.

Information also changes compatibility. Prediction-based and historical-information decisions differ in nine of ten compared states, with mean selected-AV-arc Jaccard .10. This establishes operational relevance of graph information. It does not establish decision superiority under independently evaluated outcomes; no such common evaluator was run. Prediction thus remains a component of serviceability construction, rather than a second central paper claim.

## 7. Discussion

### 7.1 Fleet substitutability

The appropriate unit of fleet transition is substitutable service capacity, not vehicle count. One hundred AVs need not replace one hundred HVs; even equal active hours can support different request sets and matching structures. The meaningful operational comparisons are changes in instantaneous capacity, \(\Delta C^{eff}\), and in realized service, \(\Delta Y\), under specified demand and operating conditions.

The results support a conditional interpretation. Greater AV active-hour share is associated with lower service under the main zero-lead, five-minute-patience replay. Separate graph analysis verifies genuine matchable-demand losses, while timing sensitivity changes their magnitude. This is not a universal AV productivity penalty. Nor does graph capacity alone determine daily performance: competition, dispatch priorities and vehicle evolution connect instantaneous opportunities to service.

This view complements research on strategic mixed-fleet control and spatial rebalancing rather than replacing it. Those decisions shape the state in which compatibility is evaluated; our capacity object measures what that state can support. [Ao et al. (2024)](https://doi.org/10.1016/j.tre.2024.103680); [Pavone et al. (2012)](https://doi.org/10.1177/0278364912444766).

### 7.2 Operational and managerial implications

Four managerial levers can be understood through the compatibility graph.

| Lever | Immediate modeling effect | What must be evaluated |
| --- | --- | --- |
| Fleet expansion or substitution | Adds/removes vehicle nodes and changes future availability | Whether new supply connects to underserved requests, not only existing competitors |
| Capability expansion | Changes hard compatibility or exposure values | Whether newly usable routes create additional matching and acceptable exposure |
| Passenger acceptance | Restores otherwise excluded AV pairs | Whether accepted requests remain route- and pickup-feasible |
| Information/evidence improvement | Changes known compatibility and exposure estimates | Whether reduced uncertainty changes admissible matching or decisions |

These are graph-centered interpretations, not four separately estimated intervention effects. More information can also reveal incompatibility and remove previously presumed options. It need not monotonically expand the graph or improve service. Similarly, broader profiles do not restore missing evidence, and fleet expansion can add little instantaneous capacity if every new vehicle competes for the same orders. No evidence-investment model is estimated here.

Pickup patience and request timing cut across all four levers. An operationally compatible vehicle contributes no current option if it cannot reach the passenger in time. Capacity planning should therefore report timing assumptions alongside fleet hours, graph-based capacity and realized service. The measured sensitivity is consistent with prior work emphasizing demand information and advance requests, without supplying a transferable optimal lead-time policy. [Wen et al. (2019)](https://doi.org/10.1016/j.tra.2019.01.018).

### 7.3 Generalizability and limitations

Five limitations define the scope. First, the application concerns one city and one evaluation day with quality-screened orders, so effect sizes are not population-wide estimates. Second, true request times are unobserved; the main boarding proxy and reconstructed alternatives produce conditional timing results. Third, capability profiles are analytical operating envelopes, not certified manufacturer specifications. Fourth, the main comparison retains a fixed route-construction policy and no active repositioning, so it does not estimate gains from joint route and rebalance optimization. Fifth, passenger acceptance is exogenous and omits price-, wait- and passenger-specific behavior.

## 8. Conclusion

Nominal AV availability and effective service capacity are distinct: counts and active hours alone do not establish substitutability for human-driven service.

State-dependent passenger and operational compatibility genuinely reduces maximum-matchable demand in the sampled system, rather than merely removing redundant candidate edges.

The magnitude depends on timing, passenger acceptance and operating capability. Fleet substitution should therefore be evaluated through compatibility-aware matching and rolling service outcomes, not fleet share alone.

## References

1. Alonso-Mora, J., Samaranayake, S., Wallar, A., Frazzoli, E., and Rus, D. (2017). On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment. *Proceedings of the National Academy of Sciences*, 114(3), 462–467. [DOI](https://doi.org/10.1073/pnas.1611675114).
2. Ao, D., Lai, Z., and Li, S. (2024). Control of dynamic ride-hailing networks with a mixed fleet of autonomous vehicles and for-hire human drivers. *Transportation Research Part E*, 189, 103680. [DOI](https://doi.org/10.1016/j.tre.2024.103680).
3. Automated Vehicle Safety Consortium (AVSC) (2020). *AVSC Best Practice for Describing an Operational Design Domain: Conceptual Framework and Lexicon*. AVSC00002202004. SAE International. [Official record](https://saemobilus.sae.org/reports/avsc-best-practice-describing-operational-design-domain-conceptual-framework-lexicon-avsc00002202004).
4. Elmachtoub, A. N., and Grigas, P. (2022). Smart “Predict, then Optimize”. *Management Science*, 68(1), 9–26. [DOI](https://doi.org/10.1287/mnsc.2020.3922).
5. Li, X., and Zardini, G. (2026). Where Should Robotaxis Operate? Strategic Network Design for Autonomous Mobility-on-Demand. *arXiv preprint*, arXiv:2602.19341. [Preprint](https://arxiv.org/abs/2602.19341).
6. Mo, D., Chen, X. M., and Zhang, J. (2022). Modeling and Managing Mixed On-Demand Ride Services of Human-Driven Vehicles and Autonomous Vehicles. *Transportation Research Part B*, 157, 80–119. [DOI](https://doi.org/10.1016/j.trb.2022.01.003).
7. Pavone, M., Smith, S. L., Frazzoli, E., and Rus, D. (2012). Robotic load balancing for mobility-on-demand systems. *The International Journal of Robotics Research*, 31(7), 839–854. [DOI](https://doi.org/10.1177/0278364912444766).
8. SAE International (2021). *Taxonomy and Definitions for Terms Related to Driving Automation Systems for On-Road Motor Vehicles*. J3016_202104. [DOI](https://doi.org/10.4271/J3016_202104).
9. Wen, J., Nassir, N., and Zhao, J. (2019). Value of demand information in autonomous mobility-on-demand systems. *Transportation Research Part A*, 121, 346–359. [DOI](https://doi.org/10.1016/j.tra.2019.01.018).
10. Wilder, B., Dilkina, B., and Tambe, M. (2019). Melding the Data-Decisions Pipeline: Decision-Focused Learning for Combinatorial Optimization. *Proceedings of the AAAI Conference on Artificial Intelligence*, 33(1), 1658–1665. [DOI](https://doi.org/10.1609/aaai.v33i01.33011658).
