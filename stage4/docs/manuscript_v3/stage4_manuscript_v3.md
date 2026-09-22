# Instantaneous Effective Matching Capacity in Mixed-Autonomy Ride-Hailing

## Abstract

Introducing autonomous vehicles into a ride-hailing fleet changes not only available supply but also the requests that each vehicle can serve. This paper examines how heterogeneous serviceability mediates the relationship between nominal supply and realized service. We represent passenger acceptance, route compatibility, information requirements and pickup deadlines through a state-dependent request–vehicle graph. Its maximum matching defines instantaneous effective matching capacity, distinct from vehicle-hours and daily throughput. A replay of 30,000 Xi'an trips compares 41 operating scenarios. Under zero request lead and five-minute pickup patience, the mean service rate decreases from 0.7258 to 0.3924 as the baseline-normalized autonomous active-hour level increases from 0.25 to 0.75. Across ten decision states, successive acceptance, route and information requirements reduce autonomous-vehicle matching capacity from 718 to 124; pickup deadlines reduce it further to 12. These are losses of simultaneously matchable demand, not merely reductions in redundant candidate pairs. Higher passenger acceptance and broader assumed capability partly recover service, while reconstructed request times materially alter the magnitude of capacity loss. Incomplete movement representation also excludes some opportunities without establishing physical infeasibility. The findings therefore identify a conditional loss of service substitutability, rather than an intrinsic productivity disadvantage of autonomous vehicles. Fleet-transition assessments should distinguish nominal availability, admissible matching opportunities and realized service, and state the operational and information assumptions connecting them.

**Keywords:** instantaneous effective matching capacity; mixed-autonomy ride-hailing; service substitutability; bipartite matching; rolling assignment

## 1. Introduction

An autonomous vehicle added to a ride-hailing fleet does not necessarily replace an equivalent amount of human-driven service. A waiting passenger may decline an autonomous ride, the requested route may fall outside the assumed operating capability, or the vehicle may be unable to arrive before the pickup deadline. Moreover, vehicles that can each serve several requests may compete for the same small subset of demand. The operational value of supply therefore depends on both the availability of vehicles and the structure of their service opportunities.

Mixed-fleet research provides several explanations for differences between human-driven and autonomous service. Mo et al. (2022) examine mixed on-demand services with congestion and market interactions. Ao et al. (2024) model human-driver scheduling and strategic relocation alongside platform control of autonomous vehicles. These studies establish that fleet composition interacts with incentives, demand and spatial allocation. A complementary question arises at the point of dispatch: given the vehicles and requests currently present, how much demand can the available fleet simultaneously serve under heterogeneous service requirements?

Compatibility is an established concern in operations research. Akçay et al. (2010) study dynamic assignment when service resources can accommodate overlapping subsets of jobs, and Tsitsiklis and Xu (2017) investigate flexibility in queueing architectures represented by bipartite graphs. In transportation, dynamic trip–vehicle assignment makes simultaneous service opportunities explicit (Alonso-Mora et al., 2017), while rebalancing changes the spatial distribution from which those opportunities arise (Pavone et al., 2012). Building on these perspectives, this paper treats maximum matching as an intermediate operational measure between nominal fleet supply and realized ride-hailing service. The contribution lies in the application and empirical interpretation of that measure, rather than a new matching algorithm.

Autonomous service introduces an additional distinction between physical capability and the information available to establish compatibility. Operational-design-domain descriptions characterize the conditions under which automated operation is intended (SAE International, 2021; Automated Vehicle Safety Consortium [AVSC], 2020). Operation-network design also addresses where robotaxis should operate and which paths support service requirements (Li & Zardini, 2026, preprint). At dispatch, however, an admissible route requires both compatibility with the adopted operating assumptions and sufficient information to evaluate it. A route with missing movement information may be withheld from autonomous service even when no physical prohibition has been established. This distinction is central to interpreting observed capacity loss.

We organize the analysis around four quantities: nominal active supply, the request–vehicle compatibility graph, instantaneous effective matching capacity, and rolling realized service. The graph records which individual assignments are admissible; its maximum matching records how many can be made simultaneously. Rolling service additionally depends on the assignments chosen and the vehicle states they create. This separation allows us to test whether restrictions remove meaningful service opportunities or merely redundant candidate pairs, without equating instantaneous matching counts with daily throughput.

The empirical application replays 30,000 quality-screened trips from Xi'an under alternative autonomous active-hour levels, passenger acceptance assumptions and analytical capability profiles. Full-day scenarios measure realized service, while fixed-state comparisons examine matching capacity, candidate compression and sensitivity to request timing. Operational predictions supply route information rather than constitute a separate methodological contribution. Demand-information research motivates examining their operational role (Wen et al., 2019), while decision-focused optimization distinguishes predictive accuracy from downstream decision value (Elmachtoub & Grigas, 2022; Wilder et al., 2019).

The paper contributes a graph-based account of service substitutability, a rolling assignment formulation that separates categorical eligibility from continuous exposure control, and empirical evidence on the conditions under which nominal autonomous supply translates poorly into service. The results are conditional on the adopted capability assumptions, network information and request timing. In particular, the autonomous-subgraph analysis identifies a mechanism of lost substitutability, not a decomposition of total mixed-fleet service. Section 2 defines the capacity measures, Section 3 presents the dispatch model, Sections 4–5 describe the data and experimental design, and Sections 6–8 discuss the results and their implications.

## 2. From nominal supply to matching capacity

### 2.1 Nominal availability

Let \(\widetilde{\mathcal V}_t\) denote vehicles active in the operating schedule at time \(t\), including those currently serving a request. Nominal active supply and type-specific active hours are
\[
S_t^{nom}=|\widetilde{\mathcal V}_t|,\qquad
H^b=\int_0^T|\widetilde{\mathcal V}_t^b|\,dt,\quad b\in\{H,A\},
\]
where \(H\) and \(A\) indicate human-driven vehicles (HVs) and autonomous vehicles (AVs). Time is expressed in hours when computing \(H^b\). The dispatchable set \(\mathcal V_t\subseteq\widetilde{\mathcal V}_t\) contains idle vehicles available for a new assignment.

We describe autonomous supply by the baseline-normalized AV active-hour level
\[
q_A=\frac{H^A}{H^{base}},
\]
where \(H^{base}\) is active availability in the all-HV baseline. This normalization compares AV hours with a common reference. It is not necessarily the within-scenario fraction \(H^A/(H^A+H^H)\), and it should not be interpreted as a vehicle-count share.

Active hours measure the amount of supply offered, but not its substitutability. Two fleets with equal active hours can differ in location, availability schedules and compatibility with waiting requests. The distinction between active and dispatchable vehicles is also important: a vehicle already carrying a passenger contributes to active supply but cannot receive another request in the unpooled service considered here.

### 2.2 State-dependent serviceability

Let \(\mathcal O_t\) contain released requests that remain unassigned and have not expired. Request \(o\) has release time \(r_o\), pickup deadline \(D_o\), an origin and destination, and a route representation. The compatibility graph is
\[
G_t=(\mathcal O_t,\mathcal V_t,\mathcal A_t),\qquad
\mathcal A_t=\{(o,v):a_{ovt}=1\},
\]
where \(a_{ovt}\) indicates that vehicle \(v\) is eligible to serve request \(o\) under the information and operating rules at time \(t\).

Eligibility combines passenger acceptance, route compatibility, required route information and pickup feasibility. Vehicle availability can impose an additional service-completion constraint. Candidate search and routing determine which pairs are evaluated. We distinguish these computational restrictions from the substantive service requirements, since removing candidate pairs need not reduce the number of requests that can be matched.

For a sequence of nested admissible graphs, indexed by \(g\), define
\[
E_t^g=|\mathcal A_t^g|,\qquad
U_t^g=|\{o:\deg_{G_t^g}(o)>0\}|,\qquad
M_t^g=\nu(G_t^g).
\]
Here \(E_t^g\) counts eligible pairs, \(U_t^g\) counts requests with at least one option, and \(M_t^g\) is maximum matching cardinality. Differences between successive graphs measure attrition in the stated sequence. Because the requirements interact, these differences are not independent causal effects.

### 2.3 Instantaneous effective matching capacity

We define instantaneous effective matching capacity as
\[
C_t^{eff}=\nu(G_t)=
\max_{x\in\{0,1\}^{|\mathcal A_t|}}
\left\{\sum_{(o,v)\in\mathcal A_t}x_{ov}:
\sum_vx_{ov}\le1,\ \sum_ox_{ov}\le1\right\}.
\]
This quantity is the largest number of simultaneous unpooled assignments admitted by the graph. We use *matching capacity* as shorthand below. It is a count at a particular state, not a service rate per unit time.

The distinction from candidate volume is immediate. Ten requests connected only to one vehicle produce ten eligible pairs and ten requests with an option, but matching capacity is one. Conversely, removing many pairs from a graph with alternative assignments can leave its maximum matching unchanged. A capacity analysis must therefore examine the assignment structure rather than infer service losses from edge counts alone.

For AV-specific analysis, let \(G_t^A\) restrict the vehicle side to AVs and define \(C_t^{eff,A}=\nu(G_t^A)\). The empirical mechanism analysis uses this subgraph. Its capacity cannot generally be added to HV-subgraph capacity, because both vehicle types may serve the same requests. Accordingly, AV-subgraph losses identify reduced autonomous service opportunities, not the fraction of total mixed-fleet service lost.

Matching capacity also differs from capacity under coupled operating constraints. If \(\mathcal F_t(\Gamma)\) contains assignments satisfying the cumulative exposure limits introduced in Section 3, then
\[
C_t^{eff,\Gamma}=\max_{x\in\mathcal F_t(\Gamma)}\sum x_{ov}
\le C_t^{eff}.
\]
A shared exposure limit can make a combination of individually eligible assignments inadmissible. Such a constraint cannot generally be represented by deleting individual pairs independently.

### 2.4 Realized service over time

Let \(y_o=1\) when request \(o\) is counted as served in the rolling evaluation and zero otherwise. Total service and service rate are
\[
Y_{1:T}=\sum_{o\in\mathcal O_{1:T}}y_o,\qquad
R_{1:T}=\frac{Y_{1:T}}{|\mathcal O_{1:T}|}.
\]
A request is counted once in \(Y_{1:T}\), although it can remain in several consecutive compatibility graphs. Thus, summing instantaneous capacities does not yield daily service.

The conceptual relationship is
\[
S_t^{nom}\longrightarrow G_t
\longrightarrow C_t^{eff}\longrightarrow Y_{1:T}.
\]
These dependencies are not proportional conversions. Compatibility determines the available choices, dispatch selects among them, and selected trips change subsequent vehicle positions and availability. Full-day comparisons and fixed-state diagnostics consequently address different questions: the former measure realized performance, whereas the latter reveal how a particular state limits simultaneous service.

## 3. Rolling assignment with heterogeneous service requirements

### 3.1 Passenger acceptance and pickup feasibility

Passenger acceptance is represented by a scenario parameter \(p_A\). Each request receives a common uniform draw \(u_o\), and AV assignments are allowed when \(u_o\le p_A\). The same draws are used across compared scenarios, so higher acceptance admits nested sets of requests. HV service does not require AV acceptance. This specification represents willingness to use an AV exogenously; it is not an estimated passenger-choice model.

At epoch \(t\), the routed pickup-time estimate \(\widehat{\tau}_{ovt}\) must satisfy
\[
\widehat{\tau}_{ovt}\le D_o-t.
\]
Search neighborhoods expand as requests wait, with at most \(K\) nearby vehicles per request retained for pickup routing. Requests with \(0<D_o-t\le d^{crit}\) are classified as critical; those remaining after an unsuccessful decision epoch receive carry-over priority.

Vehicle availability is treated separately from vehicle type. An HV whose empirical session ends at \(b_v\) is admitted only if predicted pickup and service can finish within that session:
\[
t+\widehat{\tau}_{ovt}+\widehat{s}_{ovt}\le b_v.
\]
AVs available throughout the modeled horizon are not subject to an empirical-session endpoint. The fleet-transition comparison retains this scheduling distinction, while controlled vehicle-label comparisons hold availability policy fixed.

### 3.2 Route compatibility and information requirements

Route compatibility has three states: feasible, unknown and infeasible under the adopted operating assumptions. Incompatible travel directions, prohibited movements and profile-specific maneuver restrictions can establish infeasibility. Missing route or movement information instead produces an unknown classification. AV admission requires a feasible classification and complete information for the prescribed route assessment.

The distinction between unknown and infeasible is substantive even though both prevent admission. In the road representation used here, some boundary edges lack endpoint identities needed to construct movements. Their routes can therefore be excluded because the movement cannot be evaluated, not because it has been shown to be physically impossible. The selection of a primary or available alternative route also determines which information is assessed. Section 6.5 quantifies these limitations.

Continuous operating requirements are represented separately. For route descriptor \(d_{oj}\) and positive capability cap \(B_{kj}\) for profile \(k\), define
\[
r_{ojk}=\frac{d_{oj}}{B_{kj}},\qquad
\rho^f_{ok}=\max_{j\in\mathcal J_f}r_{ojk},\qquad
e^f_{ov}=[\rho^f_{ok(v)}-1]_+,
\]
where \(f\in\{\mathrm{static},\mathrm{dynamic},\mathrm{speed}\}\), \(\mathcal J_f\) is the corresponding descriptor set, and \([z]_+=\max(z,0)\). Vehicles sharing a route and profile have the same route exposure. Overall utilization is the maximum of the three family utilizations.

These ratios describe operating-envelope utilization, not accident risk. A ratio above one indicates that a descriptor exceeds its analytical reference cap. It does not by itself establish categorical infeasibility. Keeping the three families separate also prevents a low value in one family from concealing a high value in another.

### 3.3 Assignment priorities

For each eligible pair, binary assignment variable \(x_{ov}\) satisfies
\[
\sum_{v:(o,v)\in\mathcal A_t}x_{ov}\le1,\qquad
\sum_{o:(o,v)\in\mathcal A_t}x_{ov}\le1.
\]
The model assigns one request per available vehicle and does not pool passengers. This differs from pooled trip–vehicle formulations such as Alonso-Mora et al. (2017).

Subject to these constraints and any enabled exposure limits, the objective is lexicographic:
\[
\operatorname{lexmax}
\left(
\sum_{(o,v)\in\mathcal A_t}c_ox_{ov},\
\sum_{(o,v)\in\mathcal A_t}x_{ov},\
\sum_{(o,v)\in\mathcal A_t}b_ox_{ov},\
-\sum_{(o,v)\in\mathcal A_t}\widehat{\tau}_{ovt}x_{ov}
\right),
\]
where \(c_o\) and \(b_o\) indicate critical and carry-over requests. Each objective is optimized while preserving the optima of the preceding objectives. This prioritizes imminent expiry, then total service, then requests carried over from earlier epochs, and finally pickup time.

An optional operating-cost objective follows pickup-time minimization. It preserves the three service-count optima and permits only a bounded relaxation of the aggregate pickup objective. The main factorial comparison excludes this additional cost objective; its formulation and supplementary results are provided in Appendix D.

### 3.4 Cumulative exposure limits

Let \(Z_t^f\) be cumulative system-wide exposure in family \(f\) before epoch \(t\), and let \(N_t^A\) be the cumulative number of AV assignments. For each enabled family, impose
\[
Z_t^f+\sum_{(o,v)\in\mathcal A_t^A}e^f_{ov}x_{ov}
\le
\Gamma_f\left(N_t^A+\sum_{(o,v)\in\mathcal A_t^A}x_{ov}\right). \tag{1}
\]
The state is updated additively after each assignment. Equation (1) limits cumulative mean excess per AV assignment. It is neither a per-vehicle budget nor a time- or distance-weighted limit.

Three operating policies are compared. Strict control sets all \(\Gamma_f=0\), admitting only zero-excess AV assignments from an initially zero exposure state. Reference control uses positive family limits calibrated once in a designated scenario. Unconstrained exposure control removes (1) but retains categorical compatibility and information requirements. A high-exposure assignment can be offset by lower exposure within the same family under a positive mean limit; exposure in one family cannot be offset by another.

### 3.5 Properties of the exposure policy

The following properties clarify what the operating limits guarantee.

**Proposition 1.** With additive updates, any assignment satisfying (1) gives \(Z_{t+1}^f\le\Gamma_fN_{t+1}^A\). When the assignment count is positive, cumulative mean exposure is at most \(\Gamma_f\).

**Proposition 2.** Hold the current graph, cumulative exposure state, objectives and all other constraints fixed. If \(\Gamma'\ge\Gamma\) componentwise, then \(\mathcal F_t(\Gamma)\subseteq\mathcal F_t(\Gamma')\). Relaxing the limits cannot reduce the highest-priority lexicographic optimum in that state.

**Proposition 3.** Separate mean-exposure bounds \(\bar e_f\le\Gamma_f\) imply a weighted bound \(\sum_fw_f\bar e_f\le\sum_fw_f\Gamma_f\) for nonnegative weights, but the converse need not hold.

The first property follows by substitution, and the second by feasible-set inclusion. For the third, two unit caps with equal weights admit the vector \((2,0)\) under the aggregate bound while violating the first family cap. Full arguments appear in Appendix D. These properties characterize the exposure policy; they do not imply a safety guarantee, monotone daily service or a new matching theorem.

## 4. Data and operational representation

### 4.1 Trajectories and evaluation sample

The application uses ride-hailing GPS trajectories from Xi'an, China, in October 2016. The evaluation sample contains 30,000 quality-screened orders from 31 October, shared by all scenarios. Screening requires a supported route, adequate GPS quality, consistent road identity and usable direct timing observations. The sample is therefore a selected subset of recorded trips rather than a census of demand.

Trajectory endpoints provide an observed departure or boarding proxy and a completion location. They do not independently identify platform request times. Training precedes the evaluation day, with separate later dates used for validation and calibration. Because 31 October was also used in earlier system evaluations, it is a common temporal benchmark rather than an untouched holdout for every development choice. The present analysis reports operating contrasts on that sample, not population-level causal effects.

### 4.2 Route and capability descriptors

Trajectories are map matched to a directed OpenStreetMap network. Historical travel in the reverse direction of a represented road remains distinct rather than being projected onto the forward direction. Intersections are represented as complexes of connected road elements, with movements defined by an incoming edge, any internal edges and an outgoing edge.

Four static descriptors characterize each intersection complex: external physical connections, topological movements, boundary road-class diversity and internal road length. Boundary diversity counts distinct road classes among incoming and outgoing edges, so a single-node intersection can have nonzero diversity even without internal edges. Route-level static descriptors take the corresponding maxima over encountered complexes.

Dynamic descriptors summarize crawl, stops, speed variability and acceleration variability. For each component, segment predictions are mapped to a training-derived distribution and combined using predicted median traversal times. The route summaries describe mean percentile exposure, the proportion of predicted time above the 90th-percentile threshold, and the longest consecutive duration above that threshold. Their definitions and units are given in Appendix B. Speed utilization is assessed separately against the profile's speed cap.

Conservative, Moderate and Advanced profiles provide nested analytical capability assumptions. They combine maneuver rules with descriptor caps; they are not specifications of deployed vehicles. Caps based on marginal quantiles do not imply that the same fraction of routes satisfies all dimensions jointly. This interpretation follows the distinction between describing operating conditions and validating an actual automated-driving domain (AVSC, 2020).

Route selection is determined before evaluating dispatch outcomes. The analysis uses the established primary-route and limited alternative-route procedure, rather than jointly optimizing service routes and vehicle assignment. Missing information for an alternative route can prevent its use even when its represented geometry is compatible.

### 4.3 Prediction and fleet availability

A multivariate model provides travel-time, pace and operating-condition predictions using decision-time inputs and predicted route progression. Preprocessing and distribution mappings are estimated from training data and retained for evaluation. Realized trip duration is used to advance the simulation after assignment, not to replace missing predictions at dispatch.

Prediction can affect both route assessment and continuous exposure. We examine whether replacing model estimates with historical estimates changes the available or selected assignments. This is a test of decision relevance, distinct from training through an optimization loss or demonstrating superior realized outcomes (Elmachtoub & Grigas, 2022; Wilder et al., 2019).

HVs inherit empirical driver-session availability, while AVs are available throughout the modeled horizon. The all-HV reference contains \(H^{base}=12{,}279.336389\) vehicle-hours. Transition scenarios target \(q_A=H^A/H^{base}\). Consequently, a change in autonomous supply also changes the mix of availability schedules and evolving spatial states; it is not a vehicle-label-only intervention.

## 5. Experimental design

### 5.1 Full-day comparisons

The reference replay treats the observed departure or boarding proxy as request release. Requests therefore have zero lead relative to that proxy and expire if pickup cannot occur within 300 seconds. Dispatch decisions occur every 30 seconds, with a 30-second critical window. These timing assumptions are explicit model inputs rather than observed passenger behavior.

The main factorial combines three AV active-hour levels, \(q_A\in\{0.25,0.50,0.75\}\), three capability profiles, and three acceptance levels, \(p_A\in\{0.40,0.70,1.00\}\). These 27 settings form the main mixed-fleet comparison within 41 full-day scenarios. All-HV and all-AV cases provide composition benchmarks. All use the same 30,000 orders and common acceptance draws.

The main factorial excludes continuous exposure budgets and the optional cost objective. Thus, a continuous cap exceedance alone does not remove an AV assignment. Categorical route compatibility, information requirements and pickup feasibility remain active. The same road representation and route-selection procedure are used throughout, including the movement-information limitation quantified in Section 6.5. No full-day comparison evaluates a corrected representation.

A separate policy comparison holds \(q_A=0.50\), Moderate capability and \(p_A=0.70\) fixed. It contrasts strict, reference and unconstrained exposure control. Reference limits are calibrated in the designated \(q_A=0.25\), Moderate, universal-acceptance scenario and transferred without retuning to the central comparison. Idle-vehicle repositioning is not active in these main comparisons.

### 5.2 Fixed-state mechanism analysis

Ten decision states spanning low-load, midday and evening periods are used to examine the AV compatibility graph. For each successive requirement, we calculate eligible pairs \(E\), requests with an AV option \(U\), and maximum matching \(M\) on the same graph. This directly tests whether a reduction in candidate pairs also removes simultaneously matchable demand. Summed counts across these states are descriptive aggregates, not unique daily requests.

Two controls distinguish serviceability restrictions from computational artifacts. First, vehicle-label comparisons preserve physical inputs and availability policy; candidate and selected identities agree in all ten states. Second, the candidate limit is varied over \(K=10,20,40,80\), with \(K=20\) used in the reference specification. This checks whether the observed capacity loss primarily reflects candidate compression.

The analysis is sequential: passenger acceptance is applied before route compatibility, followed by information completeness, candidate compression, pickup routing and the pickup deadline. Attribution is conditional on this order. It neither estimates independent effects of the requirements nor provides a causal decomposition of full-day service.

### 5.3 Request timing and information sensitivity

Three alternative request-lead scenarios are reconstructed from driver-chain statistics on 19–22 October. Low, Base and High scenarios use different positions in the observed inter-trip-gap distribution rather than fixed lead times. Request release is the boarding proxy minus the reconstructed lead.

Timing comparisons use four physical states: \(q_A=0.50\) and \(0.75\), each at noon and 17:30, with Moderate capability and acceptance 0.70. Vehicle positions, prior assignment history, route descriptors, routing time and operating rules remain fixed. Changing release times changes the waiting cohort and remaining pickup patience. This design isolates conditional-state sensitivity; it does not simulate alternate daily histories or validate forecasts at earlier reconstructed request times.

A separate ten-state comparison replaces model predictions with historical estimates while retaining physical and policy inputs. Its outcomes are changes in compatibility and selected assignments. An independent common outcome evaluator is not available, so these comparisons do not establish which information source yields better realized decisions. Additional prediction errors, timing-reconstruction checks and cost results are reported in the supplementary material.

## 6. Results

### 6.1 Autonomous availability and realized service

A higher autonomous active-hour level is associated with lower service in the reference replay. Averaged across the capability and acceptance settings, service rates are 0.7258, 0.5984 and 0.3924 at \(q_A=0.25,0.50,0.75\), respectively. The decline from the lowest to the highest level is 0.3334, or approximately 45.9% of the lower-level mean.

Table 1 shows the composition benchmarks and the Moderate-profile comparisons at acceptance 0.70. The all-AV result represents service without HV alternatives under the specified assumptions; it is not a performance ceiling for autonomous technology.

**Table 1. Realized service under selected fleet conditions.** All scenarios share 30,000 orders, zero request lead and 300-second pickup patience. Service rate is the fraction of requests served.

| Fleet condition | Service rate |
| --- | ---: |
| All HV | 0.7889 |
| \(q_A=0.25\), Moderate, acceptance 0.70 | 0.7297 |
| \(q_A=0.50\), Moderate, acceptance 0.70 | 0.6044 |
| \(q_A=0.75\), Moderate, acceptance 0.70 | 0.4013 |
| All AV, Moderate | 0.1515 |

These comparisons establish a supply-to-service pattern in the modeled system. They do not identify an intrinsic AV productivity penalty: fleet composition changes availability schedules and spatial evolution, while serviceability depends on route information and the timing assumptions. The fixed-state analysis examines one mechanism underlying this pattern.

### 6.2 From candidate pairs to matchable demand

The service requirements reduce AV matching capacity, not merely candidate volume. Across the ten states, maximum matching decreases from 718 spatially available matches to 493 after passenger acceptance, 238 after categorical route assessment, and 124 after information requirements. Pickup deadlines reduce the remaining capacity to 12 (Table 2).

**Table 2. AV opportunities and matching capacity across ten decision states.** Values are sums over states. \(E\) counts eligible request–vehicle pairs, \(U\) counts requests with at least one AV option, and \(M\) is maximum AV matching cardinality.

| Admissible set | \(E\): AV pairs | \(U\): requests with an AV option | \(M\): maximum AV matching |
| --- | ---: | ---: | ---: |
| Spatial opportunity | 95,180 | 720 | 718 |
| Passenger compatible | 65,444 | 493 | 493 |
| Categorically route compatible | 31,153 | 238 | 238 |
| Required route information available | 15,775 | 124 | 124 |
| After candidate compression | 1,371 | 124 | 124 |
| Pickup route available | 1,371 | 124 | 124 |
| Pickup deadline satisfied | 48 | 12 | 12 |
| Available to assignment | 48 | 12 | 12 |

In the stated sequence, acceptance, categorical route assessment, information requirements and pickup deadlines remove 225, 255, 114 and 112 matching units, respectively. By contrast, candidate compression removes approximately 91% of the preceding pairs without reducing matching capacity. Final maximum matching is identical at each state for all four tested values of \(K\).

The difference between the first row's \(U=720\) and \(M=718\) illustrates vehicle competition: providing every request with an individual option need not make all requests simultaneously serviceable. The subsequent decreases in \(M\) establish that the eligibility requirements remove potential simultaneous service in these AV subgraphs. They do not partition the decline in mixed-fleet daily service, and requests appearing in multiple states must not be counted as distinct daily losses.

### 6.3 Passenger acceptance, capability and exposure control

Passenger acceptance has a larger service effect at the higher AV active-hour level in the reported comparisons. Under Moderate capability, increasing acceptance from 0.40 to 1.00 raises service rate by 0.0087 at \(q_A=0.25\) and by 0.0477 at \(q_A=0.75\). At \(q_A=0.75\) and acceptance 0.70, broadening the capability assumption from Conservative to Advanced raises service rate by 0.0330.

These gains partly recover service but do not remove the larger composition contrast. Acceptance supplies permission to use an AV, whereas capability determines which routes are admissible. Neither ensures that a vehicle can arrive before the pickup deadline. Their effects therefore depend on the other constraints in the compatibility graph.

Exposure control addresses the operating conditions of selected AV trips rather than passenger willingness or categorical compatibility. In the central scenario, reference family limits retain a service rate close to unconstrained exposure control, whereas strict zero-excess limits reduce both service and the AV share of assignments (Table 3).

**Table 3. Exposure policy comparison at \(q_A=0.50\), Moderate capability and acceptance 0.70.** AV assignment share is the fraction of selected assignments made to AVs, distinct from the active-hour normalization \(q_A\).

| Exposure policy | Service rate | AV assignment share |
| --- | ---: | ---: |
| Strict zero excess | 0.5532 | 0.0113 |
| Reference family limits | 0.6038 | 0.1244 |
| Unconstrained continuous exposure | 0.6044 | 0.1217 |

Relative to unconstrained exposure control, reference limits reduce selected static and dynamic exposure by 9.6% and 5.6%. This comparison illustrates a service–exposure trade-off for the tested limits, not their optimality or a safety benefit. Because cumulative limits couple assignments and alter later vehicle states, their daily effects cannot be inferred from individual route eligibility alone.

### 6.4 Sensitivity to request timing and operational information

Request timing materially changes the capacity available before and after pickup deadlines. Across the four physical states, final AV maximum matching is 8 under zero lead, compared with 26, 25 and 40 under Low, Base and High lead scenarios (Table 4). The fraction retained after the pickup constraint also increases, although Low and Base are not strictly ordered.

**Table 4. Request-timing sensitivity across four physical states.** Retention is summed final \(M\) divided by summed pre-deadline \(M\). These states are distinct from the ten-state aggregation in Table 2.

| Request timing | Pre-deadline \(M\) | Final \(M\) | Aggregate retention |
| --- | ---: | ---: | ---: |
| Zero lead | 53 | 8 | 15.09% |
| Low lead | 130 | 26 | 20.00% |
| Base lead | 128 | 25 | 19.53% |
| High lead | 156 | 40 | 25.64% |

Local ordering can also change. At noon, final matching capacity for \(q_A=0.50/0.75\) is \(4/3\) under zero lead but \(10/11\) under High lead. At 17:30, the corresponding pairs are \(1/0\) and \(8/11\). These comparisons retain the same physical vehicle states but change waiting cohorts; they are not order-level treatment effects or evidence of reversed full-day performance.

Operational information also affects decisions. Model-based and historical-information assignments differ in nine of ten compared states, with mean selected-AV-pair Jaccard similarity of 0.10. This demonstrates that prediction is relevant to the decision representation. It does not establish superior realized outcomes, since the alternatives were not evaluated with an independent common outcome measure.

Together, the timing and information comparisons locate the empirical finding more precisely. The observed capacity loss depends on how requests become available, how much pickup time remains, and what route information supports admission. The full-day numerical contrast should therefore not be interpreted as invariant to those choices.

### 6.5 Incomplete route information and excluded opportunities

Some excluded service opportunities arise from incomplete network representation rather than established physical incompatibility. Examination of the ten-state cohort covers 720 unique orders and 13,378 intersection-complex encounters. All 391 unresolved movement encounters, involving 276 orders and 117 distinct movement keys, include boundary edges whose incomplete endpoint identities prevent movement construction. Boundary recognition and movement construction therefore do not operate on the same fully represented set of roads.

The underlying routing graph supplies endpoints for the 88 implicated edges, but inconsistencies with existing node identities and intersection membership prevent those endpoints from directly establishing admissible movements. The analysis identifies a specific information limitation; it neither measures a full-network error rate nor estimates the service gained by correcting it. Detailed topology findings are reported in Appendix G.

Direction exclusions have a different basis. All 14,837 evaluation-day orders flagged for historical reverse travel are supported by one-way tags and opposed node ordering in the source OSM network. This supports the adopted direction rule, but it does not certify legal restrictions as they existed in 2016 or demonstrate that a lawful alternative route was unavailable. Missing movement information and source-supported direction restrictions should therefore not be treated as a single category of physical AV incapability.

These findings qualify the interpretation of Table 2. Its capacity reductions are real within the assessed compatibility graphs, but the graphs reflect capability assumptions, available network information and route-selection rules together. They do not provide a direct measure of the physical service limits of autonomous vehicles.

## 7. Discussion

### 7.1 Service substitutability as a fleet-transition criterion

Fleet transition requires evaluating the service that new supply can replace, rather than the number of vehicles or hours introduced. Matching capacity makes this distinction operational: it records whether available vehicles connect to enough distinct requests to support simultaneous assignments. Realized service then depends on how dispatch uses those opportunities over time.

The Xi'an replay shows lower service at higher normalized AV active-hour levels under the reference timing and serviceability assumptions. Fixed-state analysis supplies a more specific explanation than candidate attrition alone: acceptance, route compatibility and information requirements remove maximum-matchable AV demand. Yet this remains a mechanism of lost substitutability, not an additive explanation of the entire mixed-system outcome. HV alternatives, vehicle competition and subsequent spatial evolution remain relevant.

This perspective complements mixed-fleet market and control models (Mo et al., 2022; Ao et al., 2024), as well as spatial rebalancing (Pavone et al., 2012). Those decisions influence the states in which dispatch operates. The matching-capacity measure describes what a state can support under specified service requirements. It connects fleet planning to operational opportunity without replacing an equilibrium model or a dynamic control policy.

### 7.2 Implications for operations and information provision

Additional supply is most useful when it connects to requests that existing vehicles cannot collectively serve. Adding vehicles with highly overlapping service opportunities can create many eligible pairs while adding little matching capacity. Conversely, a modest expansion of compatibility can matter if it connects otherwise unserviceable demand to available vehicles. This is why pair counts, matching counts and realized service should be reported separately.

Passenger acceptance, assumed maneuver capability and route information act at different points in that relationship. The reported gains from broader acceptance and capability show that both can matter, but neither overcomes every remaining constraint. Information improvement is also distinct from capability expansion: resolving an unknown movement may establish compatibility, or it may reveal a genuine incompatibility. Better information need not expand the admissible graph monotonically.

Pickup timing cuts across these distinctions. Even a willing passenger and an admissible route provide no current match if the vehicle cannot arrive before expiry. The sensitivity results are consistent with the broader importance of demand information in autonomous mobility operations (Wen et al., 2019), while providing no transferable optimal advance-request policy. Capacity assessments should state the assumed release and patience process alongside fleet availability.

Continuous exposure limits offer a separate means of controlling the operating conditions of selected service. The reference policy nearly preserves service in the central comparison while reducing two exposure families. Its managerial interpretation is a tested operating trade-off, not a recommendation that the same numerical limits be transferred to another fleet or treated as safety thresholds.

### 7.3 Scope and limitations

The empirical findings concern one city, one evaluation day and a quality-selected trajectory sample. Observed effect sizes are descriptive scenario contrasts, without population-level uncertainty estimates. True request times are unavailable, and the request-lead sensitivity changes waiting cohorts while retaining physical states from the reference history. It cannot establish the daily service effect of a different release process.

The capability profiles are analytical scenarios rather than certified vehicle specifications. Passenger acceptance is exogenous, and the model omits price- and passenger-specific choice behavior. The main comparisons also retain a predetermined service-route procedure and no active idle-vehicle repositioning. They do not quantify gains from joint routing, rebalancing or endogenous demand management.

A further limitation is directly observed: incomplete endpoint and movement representation can exclude routes without demonstrating physical infeasibility. Route-selection rules can also leave an uncertain primary route in place despite an available alternative representation. No corrected-network daily comparison measures the effect of these behaviors. Accordingly, the service results characterize the adopted information and admission policy, not an error-free map of AV capability.

Finally, prediction comparisons establish changes in decisions rather than superiority under common outcome evaluation. Neither the route descriptors nor the exposure constraints estimate accident probabilities. The framework concerns operational serviceability and its consequences for assignment.

## 8. Conclusion

Nominal autonomous supply, instantaneous effective matching capacity and realized ride-hailing service measure different aspects of fleet performance. Their distinction becomes consequential when vehicles serve heterogeneous sets of requests.

In the Xi'an application, service decreases as the baseline-normalized AV active-hour level rises under zero request lead and five-minute pickup patience. Fixed-state analysis shows that passenger, route and information requirements remove maximum-matchable AV demand, while candidate compression does not explain the sampled capacity loss. Higher acceptance and broader capability assumptions partly recover service, and alternative request timing changes the magnitude of the remaining constraint.

These results support a conditional account of service substitutability. The observed opportunities depend on capability assumptions, network information and pickup deadlines, including a documented movement-representation limitation. Evaluating a fleet transition therefore requires more than counting autonomous vehicles or hours: it requires assessing which requests the fleet can collectively serve and how dispatch converts those opportunities into service over time.

## Declarations

**Data availability.** This study uses individual trip trajectories and derived road and assignment data. Access conditions for the underlying data and the final code-and-materials availability statement must be confirmed by the authors before submission.

**Ethics.** The study is a retrospective computational analysis of mobility records. The applicable data-use authorization, privacy safeguards and institutional ethics determination must be supplied by the authors; no ethics approval or exemption is asserted here.

**Author contributions.** Author names and CRediT roles remain to be provided.

**Competing interests and funding.** The authors must provide their competing-interest declaration and funding information.

**AI-assisted writing.** AI assistance was used for manuscript restructuring, language revision and consistency checks. The authors must review the resulting text and confirm the disclosure required by the selected journal.

## References

Akçay, Y., Balakrishnan, A., & Xu, S. H. (2010). Dynamic assignment of flexible service resources. *Production and Operations Management, 19*(3), 279–304. [DOI](https://doi.org/10.1111/j.1937-5956.2009.01095.x).

Alonso-Mora, J., Samaranayake, S., Wallar, A., Frazzoli, E., & Rus, D. (2017). On-demand high-capacity ride-sharing via dynamic trip-vehicle assignment. *Proceedings of the National Academy of Sciences, 114*(3), 462–467. [DOI](https://doi.org/10.1073/pnas.1611675114).

Ao, D., Lai, Z., & Li, S. (2024). Control of dynamic ride-hailing networks with a mixed fleet of autonomous vehicles and for-hire human drivers. *Transportation Research Part E, 189*, 103680. [DOI](https://doi.org/10.1016/j.tre.2024.103680).

Automated Vehicle Safety Consortium. (2020). *AVSC best practice for describing an operational design domain: Conceptual framework and lexicon*. AVSC00002202004. SAE International. [Official record](https://saemobilus.sae.org/reports/avsc-best-practice-describing-operational-design-domain-conceptual-framework-lexicon-avsc00002202004).

Elmachtoub, A. N., & Grigas, P. (2022). Smart “Predict, then Optimize”. *Management Science, 68*(1), 9–26. [DOI](https://doi.org/10.1287/mnsc.2020.3922).

Li, X., & Zardini, G. (2026). *Where should robotaxis operate? Strategic network design for autonomous mobility-on-demand* [Preprint]. arXiv:2602.19341. [Preprint](https://arxiv.org/abs/2602.19341).

Mo, D., Chen, X. M., & Zhang, J. (2022). Modeling and managing mixed on-demand ride services of human-driven vehicles and autonomous vehicles. *Transportation Research Part B, 157*, 80–119. [DOI](https://doi.org/10.1016/j.trb.2022.01.003).

Pavone, M., Smith, S. L., Frazzoli, E., & Rus, D. (2012). Robotic load balancing for mobility-on-demand systems. *The International Journal of Robotics Research, 31*(7), 839–854. [DOI](https://doi.org/10.1177/0278364912444766).

SAE International. (2021). *Taxonomy and definitions for terms related to driving automation systems for on-road motor vehicles*. J3016_202104. [DOI](https://doi.org/10.4271/J3016_202104).

Tsitsiklis, J. N., & Xu, K. (2017). Flexible queueing architectures. *Operations Research, 65*(5), 1398–1413. [DOI](https://doi.org/10.1287/opre.2017.1620).

Wen, J., Nassir, N., & Zhao, J. (2019). Value of demand information in autonomous mobility-on-demand systems. *Transportation Research Part A, 121*, 346–359. [DOI](https://doi.org/10.1016/j.tra.2019.01.018).

Wilder, B., Dilkina, B., & Tambe, M. (2019). Melding the data-decisions pipeline: Decision-focused learning for combinatorial optimization. *Proceedings of the AAAI Conference on Artificial Intelligence, 33*(1), 1658–1665. [DOI](https://doi.org/10.1609/aaai.v33i01.33011658).
