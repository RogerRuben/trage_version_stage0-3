# From Ride-Hailing Trajectories to ODD-Aware Dispatch: Effective Service Capacity in Mixed Human-Driven and Autonomous Fleets

## Abstract

Mixed human-driven vehicle (HV) and autonomous vehicle (AV) fleets need not convert nominal availability into equivalent service. We connect empirical ride-hailing trajectories, directed road identity, leakage-safe prediction, hard route readiness, continuous capability utilization, and sparse rolling assignment. The 41 frozen Test31 scenarios use 30,000 common requests, observed trajectory departure/boarding proxies as request release (zero lead), and 300-second pickup patience. Within the 27-scenario factorial, mean service rate decreases from 0.7258 to 0.3924 as baseline-normalized realized AV active-hour share increases from 0.25 to 0.75. Separate mechanism validation in ten fixed states shows that passenger, structural, evidence, and patience gates reduce state-summed maximum AV matching from 718 to 493, 238, 124, and 12. Neutral relabeling preserves candidate and selected identities in all ten states when availability policy is held fixed; K=10/20/40/80 preserves final matching capacity in every sampled state. However, fingerprint-verified request-time reconstruction materially changes matching magnitudes in four paired physical states: zero/Low/Base/High lead scenarios yield summed matching capacities of 8/26/25/40. These are instantaneous capacities, not daily service counts. The evidence supports a conditional mechanism: genuine matchable-capacity losses interact with request timing and remaining pickup patience. Prediction is decision-relevant, but superiority under a common independent evaluator is not established. The framework represents operational compatibility without claiming AV safety certification.

**Keywords:** autonomous ride-hailing; mixed-fleet dispatch; operational design domain; decision-time prediction; effective service capacity; lexicographic assignment

## 1. Introduction

### 1.1 Operational problem and gap

Ride-hailing platforms are likely to operate mixed fleets before fully autonomous service becomes universal. In such a fleet, an available HV and an available AV are not necessarily interchangeable. A passenger may decline an AV; a directed route may contain a known incompatible movement; evidence needed to assess a route may be missing; predicted operating conditions may exceed a vehicle capability envelope; or the AV may be too far from the pickup before the request expires. Even when an AV-order pair survives every individual filter, it can lose to another pair in the rolling assignment. The platform therefore faces a conversion problem: how much effective service can a nominal AV active hour actually deliver?

Much of the fleet-transition discussion abstracts from this conversion by varying an AV percentage while preserving a homogeneous service technology [CITATION NEEDED — mixed-fleet autonomous ride-hailing transition models]. Other work studies ODD constraints, passenger adoption, routing, prediction, or assignment as separate modules [CITATION NEEDED — ODD-aware fleet operations and passenger AV acceptance]. These abstractions are useful, but they leave three linked gaps. First, vehicle share is commonly measured by counts even though availability is temporal and heterogeneous. Second, AV compatibility is often compressed into a binary route flag, obscuring the difference between structural infeasibility, missing evidence, and continuous capability utilization. Third, predictive models are usually evaluated by forecast error without showing whether their information changes the sparse candidate graph and actual assignments.

Three research streams are particularly relevant but rarely joined in one empirical design. Dynamic fleet-management studies provide rolling assignment, relocation, and routing formulations, yet frequently assume that all vehicles can serve all requests after travel-time feasibility is established [CITATION NEEDED — dynamic ride-hailing assignment and rebalancing]. Automated-mobility studies relax this assumption through service zones or ODD constraints, but often encode them as fixed geographic masks or binary compatibility [CITATION NEEDED — operational design domains in mobility-on-demand]. Prediction-and-optimization research demonstrates that a forecast should be valued through downstream decisions, although applications commonly optimize a scalar cost rather than construct a multistage eligibility graph [CITATION NEEDED — decision-focused learning and predict-then-optimize]. Our setting intersects these streams: heterogeneity appears before optimization, is route- and time-specific, and depends on predicted multivariate conditions.

The distinction between installed, available, and effective capacity is familiar in other infrastructure domains [CITATION NEEDED — effective capacity under operational derating], but mixed-fleet ride-hailing adds endogenous matching. A unit of supply is not simply derated by a constant factor. Its usefulness depends on which passengers are waiting, which routes connect them, where other vehicles are located, and which assignments compete at the same epoch. Effective capacity is therefore a system outcome of compatibility and matching rather than a static AV productivity coefficient.

An empirical answer requires a chain that begins before the optimizer. Raw GPS observations must be mapped to directed network identities without inventing continuity. Intersection and dynamic attributes must be derived under frozen, training-only rules. Prediction must be leakage-safe at decision time. Candidate accounting must retain a common unit through passenger, route, evidence, patience, routing, sparsification, and optimization stages. Finally, fleet supply must be normalized by realized active hours, not simply by the number of vehicle identities. Any break in this chain can create an apparently precise fleet result whose operational meaning is unclear.

We address this problem with an integrated replay framework built from Xi'an ride-hailing trajectories. The framework does not seek to certify AV safety, learn passenger preferences, or optimize AV technology. Instead, it provides an operational interface between empirical route conditions and mixed-fleet dispatch. It combines a three-state hard route status with continuous static, dynamic, and speed-family utilization; uses exogenous passenger acceptance; and solves a patience-aware sparse rolling assignment. This structure makes the conversion from nominal supply to effective capacity observable and attributable.

### 1.2 Research questions

The study asks three questions.

**RQ1: How does increasing the baseline-normalized realized AV active-hour share affect effective dispatchable service capacity?** We examine full-day service and trace the same opportunity unit through prospective gates to determine why nominal substitution may fail to preserve service.

**RQ2: How do passenger acceptance, capability profiles, and family-specific ODD constraints shape the conversion from AV availability to realized assignments?** We compare nested acceptance and capability settings, then contrast strict, frozen-reference, and unconstrained continuous-exposure policies.

**RQ3: Does leakage-safe multivariate prediction provide decision-relevant information beyond target-wise forecast accuracy?** We use paired fixed states to test whether predicted operational conditions change candidate identity, selected route exposure, and pickup objectives.

These questions are deliberately ordered. RQ1 establishes the system-level phenomenon; RQ2 decomposes operational and behavioral mechanisms; RQ3 tests the information bridge that allows those mechanisms to enter a real-time decision.

### 1.3 Contributions and organization

The paper makes four contributions. First, it constructs a traceable empirical pipeline from high-quality ride-hailing GPS trajectories through directed-network route identities to operational dispatch descriptors. The representation preserves historical reverse directions, separates hard unknowns from continuous exceedances, and maintains distinct static, dynamic, and speed families.

Second, it defines AV supply as a baseline-normalized realized active-hour share and introduces same-unit prospective accounting for candidate survival. This yields a measurable distinction between nominal fleet composition and effective dispatchable service capacity.

Third, it embeds hard route readiness, continuous family utilization, passenger acceptance, and pickup deadlines in a scalable sparse rolling assignment. The implementation avoids dense order-by-vehicle matrices and uses lexicographic objectives to protect service priorities.

Fourth, it evaluates predictions in decision space as well as target space. The paired ablation shows how leakage-safe information changes sparse candidate arcs and assignments even when prediction accuracy is heterogeneous across targets.

Section 2 describes the empirical data and network grounding. Section 3 develops the operational representation and prediction interface. Section 4 presents the rolling assignment and structural propositions. Section 5 specifies the preregistered experiments. Section 6 reports the fleet, mechanism, policy, prediction, and cost results. Section 7 discusses implications and limitations, and Section 8 concludes.

## 2. Data and empirical grounding

### 2.1 Empirical demand and trajectory source

We construct the study from a large ride-hailing trajectory archive for Xi'an, China, rather than from synthetic origins, destinations, or travel times. The upstream pipeline retains trip identifiers, request and completion times, ordered GPS observations, and the provenance needed to connect each operational record to its reconstructed route. Quality control is performed before any dispatch experiment. Trips with insufficient temporal support, unresolved long gaps, material route disagreement, or ambiguous network identity are excluded from the core modeling set. Local GPS outliers may be retained only when their temporal and distance shares are small and the principal corridor remains supported. This distinction is important: the pipeline does not claim to recover every sparse trace, but targets a reproducible high-quality subset appropriate for link-level supervision and route-level operational analysis.

The final dispatch day contains 30,000 requests. Every experimental scenario replays exactly this same request set, in the same chronological order, with common random numbers for passenger acceptance. Scenario differences therefore arise from the fleet composition or decision policy rather than demand resampling. Request patience is represented explicitly. An order released at time \(r_o\) remains eligible until deadline \(D_o=r_o+W_o^{\max}\), and a vehicle-order pair is pickup-feasible only if elapsed waiting plus routed pickup time does not exceed \(W_o^{\max}\). In the baseline design, rolling decisions occur every 30 seconds and unmatched orders may carry over for up to five minutes. This replay framing preserves the temporal competition among requests while keeping cross-scenario comparisons paired.

The data are observational and geographically specific. They provide realistic temporal demand, routed movement, and empirical operating conditions, but do not constitute an autonomous-driving safety dataset. Accordingly, the quantities developed below are described as operational compatibility, capability-envelope utilization, and dispatch eligibility—not as safety probabilities or certification decisions.

Upstream model development follows temporal ordering. Early dates support training, later dates support validation and calibration, and 31 October is retained as a frozen external temporal benchmark consistent with the prior system version. Because that benchmark informed subsequent model development at the project level, we do not call it a never-observed final test. Scientific stability of the predictive subsystem is instead supported by rolling-origin evaluation in the upstream stage, while the present paper uses Test31 for a frozen cross-version operational replay. This terminology prevents a legacy benchmark from being overstated as a pristine holdout.

Only the products required by downstream decisions are persisted. Accepted orders retain route sequences, traversals, interval observations, segment quality, and canonical identities. Rejected orders retain a lightweight manifest rather than complete routes and images. Processing is partitioned by date and bucket with atomic writes and resume support. These choices reduce memory pressure and also limit accidental selection on post hoc visual inspection.

### 2.2 Directed network reconstruction and map matching

Raw trajectories are matched to a frozen OpenStreetMap-derived directed network through a Valhalla-based pipeline. Directional edge identity is preserved, including historical reverse-direction observations that are distinguishable from the current forward edge. A reverse historical observation is not silently projected onto the opposite directed edge. Instead, the route resolver labels each record as a full-network edge, a historical-direction overlay, or unresolved. This treatment prevents artificial continuity and allows known direction incompatibilities to enter the hard operational state explicitly.

Map matching is audited against all raw GPS points over their corresponding temporal subtraces and route components. Route quality is evaluated at continuous segment level rather than by a single order-wide snap statistic. The pipeline separates GPS condition, route condition, dynamic label usability, and canonical network identity. Directly timed GPS intervals are also separated from physical edge traversals: one continuous visit to an edge is a traversal, while multiple GPS intervals may contribute nonduplicated timing evidence to that traversal. Conservation and allocation checks prevent the same physical distance or observation interval from being counted more than once.

For the retained orders, the operational representation achieves near-complete directed-network identity coverage under the frozen upstream audit. We omit a rounded percentage because the manuscript audit did not recover a single authoritative denominator for the earlier approximate figure. The remaining unresolved identities are not imputed as convenient forward movements. At intersections, the network is consolidated into calibrated complexes using frozen upstream rules. Incoming, internal, and outgoing directed edges define movement sequences; boundary incoming and outgoing edges, rather than only internal edges, determine road-class diversity. This matters for single-node intersections, where an internal-edge-only definition would incorrectly assign zero diversity to a crossing of two different road classes.

### 2.3 Operational indicator construction

The directed route is translated into static and dynamic operational descriptors. Static descriptors summarize four distinct sources of route complexity: intersection approach structure \(A\), movement structure \(M\), boundary road-class diversity \(D\), and route or intersection extent \(L\). These are calibrated from unique training-exposed intersection complexes so that frequently traversed locations do not dominate the capability thresholds simply because they appear in more orders.

Dynamic descriptors characterize four observed phenomena—crawl, stop, speed variability, and acceleration variability—each through three route-level summaries denoted \(E\), \(Q\), and \(C\). The resulting 12-dimensional vector preserves the different ways that difficult operating conditions may be distributed along a route. It is deliberately not compressed into a weighted risk score. Profile envelopes are estimated from frozen training predictions using marginal quantile anchors. Consequently, a nominal profile percentile is a marginal cap for each descriptor, not a promise that the same percentage of complete routes will jointly satisfy all dimensions. Joint route acceptance can be substantially lower because all required dimensions must be met simultaneously.

Speed-domain compatibility is kept as a separate family. Static intersection structure, predicted dynamic conditions, and speed exposure represent different operational mechanisms and remain separately attributable throughout dispatch. The profiles labeled Conservative (C), Moderate (M), and Advanced (A) are nested capability envelopes. The labels indicate increasingly broad modeled operating envelopes; they do not certify vehicle safety or establish a universal taxonomy of automated-driving systems.

### 2.4 Fleet reconstruction

Vehicle availability is reconstructed from empirical driver activity rather than represented as a fixed number of perpetually active vehicles. Each driver contributes one or more observed online sessions, and scenario fleets inherit these temporal sessions. The normalization target \(q_A\) is therefore defined as the **baseline-normalized realized AV active-hour share**. Let \(H^{\mathrm{base}}\) be the exact active vehicle-hours in the all-human baseline and \(H_A(q)\) the active hours assigned to AV sessions under a scenario. Then

\[
q_A=\frac{H_A(q)}{H^{\mathrm{base}}},
\qquad H^{\mathrm{base}}=12{,}279.336389\ \text{vehicle-hours}.
\]

This denominator is held fixed across fleet compositions. It avoids conflating a count of vehicles with the service time they actually contribute. The realized shares for the three nominal design levels are close to 0.25, 0.50, and 0.75, with exact achieved values recorded in the audit. Because sessions differ in length and timing, replacing a fraction of driver identities would not generally produce the intended active-hour exposure; the session-based construction is thus necessary for interpretable mixed-fleet comparisons.

[Figure 2 about here]

## 3. Decision-time operational representation

### 3.1 Leakage-safe prediction requirement

Dispatch decisions can use only information available at the decision epoch. Realized end-of-trip congestion, completed-route travel time, and future trajectory progression are therefore prohibited from candidate evaluation. We freeze the Stage 2 prediction model and its preprocessing artifacts before the Test31 replay. For each candidate route, the model receives decision-time context and predicted route progression only. Training-only normalization, vocabulary, calibration, and empirical cumulative distribution functions are frozen upstream. Prediction manifests verify the model checkpoint, input schema, decision-time flag, and absence of realized target columns.

This leakage restriction is more than a machine-learning convention. In rolling dispatch, predicted conditions affect which vehicle-order arcs reach the optimizer. Using realized future conditions would alter not merely a reported forecast score but the feasible decision set itself. We therefore evaluate prediction quality in two complementary ways: target-space error on held-out observations and decision-space changes under paired fixed-state replays.

### 3.2 Multivariate model and targets

The frozen multivariate model predicts physical route outcomes and raw operational components, including travel time or pace, crawl share, stop share, bounded speed variation, and bounded acceleration variation. Percentile mappings are derived only through frozen training references and are used as operational normalizations rather than as independent ground truth. This design avoids allowing future-period percentile information to select the model.

Dynamic route descriptors are built from predicted—not realized—conditions. Weighted mid-distribution transforms map continuous predictions into the frozen reference domain; empirical exact-value tie mass is negligible for the selected model, but the mid-distribution definition remains well posed when ties occur. Route-level \(E\), \(Q\), and \(C\) summaries are then computed for each of the four dynamic families. The model is not assumed uniformly superior for every target. Its role is to provide a coherent, multivariate decision representation whose value must ultimately be assessed through changes in candidate construction and dispatch outcomes.

### 3.3 Hard route readiness and selected AV service route

For order \(o\), profile \(k\), and a candidate AV service route, the interface first emits a hard state

\[
h_{ok}\in\{\text{FEASIBLE},\text{UNKNOWN},\text{INFEASIBLE}\}.
\]

Hard infeasibility is reserved for structural evidence such as a known AV-unroutable historical direction, an explicitly prohibited movement, or another certified hard violation in the frozen interface. UNKNOWN denotes missing critical identity, movement, or dynamic evidence. Exceeding a continuous profile envelope is not automatically described as physical impossibility. This separation prevents a soft capability threshold from being misreported as a universal safety boundary.

The evaluated route is the selected AV service route used by the rolling decision process. When the canonical route identity is available, the interface uses its static and dynamic descriptors. A minimal fallback is allowed only under the frozen rules and remains identifiable in provenance. The experiment does not perform route replanning to search for an easier ODD-compatible path. Thus, comparisons concern dispatch over a fixed route construction policy, not joint assignment-and-routing optimization.

FEASIBLE should be read as “no hard violation was identified under the available frozen evidence,” not as a guarantee of successful autonomous operation. UNKNOWN is likewise a substantive output rather than a missing-value nuisance. Treating unknown evidence as feasible would inflate capacity; treating every unknown as physically infeasible would confound data coverage with vehicle capability. The dispatch policy can therefore choose a conservative handling of UNKNOWN while the reporting layer preserves its cause.

### 3.4 Continuous operational families

For every descriptor \(x_{oj}\) and profile cap \(B_{kj}>0\), define utilization ratio

\[
r_{ojk}=\frac{x_{oj}}{B_{kj}}.
\]

Static, dynamic, and speed utilization are kept as separate maxima:

\[
\rho^{\mathrm{static}}_{ok}=\max_{j\in\mathcal J_s}r_{ojk},\qquad
\rho^{\mathrm{dynamic}}_{ok}=\max_{j\in\mathcal J_d}r_{ojk},\qquad
\rho^{\mathrm{speed}}_{ok}=\max_{j\in\mathcal J_v}r_{ojk}.
\]

Overall utilization is

\[
\rho^{\mathrm{overall}}_{ok}=\max\{\rho^{\mathrm{static}}_{ok},
\rho^{\mathrm{dynamic}}_{ok},\rho^{\mathrm{speed}}_{ok}\}.
\]

No weighted average is used. A weighted sum could conceal a severe exceedance in one operational family by averaging it with easy conditions elsewhere. The maximum preserves the bottleneck interpretation. For cumulative dispatch control we define nonnegative family exposure

\[
e^f_{ok}=[\rho^f_{ok}-1]_+,\qquad f\in\{s,d,v\},
\]

where \([z]_+=\max(z,0)\). A route within a profile cap has zero exposure for that family; an exceedance remains continuous and attributable. The interface exports the hard state, all family utilizations, descriptor-level ratios, and reason codes. It deliberately does not emit a single claim that an AV is “safe” or “unsafe.”

### 3.5 Passenger acceptance and prediction-to-decision evaluation

Passenger willingness to accept an AV is represented as an exogenous Bernoulli gate. Each order receives a frozen common-random-number draw \(u_o\); under acceptance level \(p\), an AV candidate is eligible only if \(u_o\le p\). This construction yields nested acceptance sets as \(p\) rises and prevents Monte Carlo noise from obscuring paired policy comparisons. It is an interface parameter, not an estimated choice model. No demographic inference or behavioral utility model is claimed.

Prediction-to-decision value is assessed through a fixed-state ablation. The same vehicles, orders, time, locations, route candidates, profiles, and optimizer configuration are replayed while only the prediction information state changes. We compare candidate survival, AV-arc overlap, selected exposure, and pickup objective, alongside target-space errors. This design tests whether predictions are decision-relevant without extrapolating the short fixed-state result to a full operating day.

[Figure 3 about here]

## 4. Rolling mixed-fleet assignment

### 4.1 Epoch state and sparse candidates

At epoch \(t\), let \(\mathcal O_t\) be released, unserved, unexpired orders and \(\mathcal V_t\) be active, idle vehicles. The full Cartesian product is neither operationally meaningful nor computationally necessary. Candidate generation first uses a spatial index to retrieve vehicles near each pickup, with a search radius expanding from 2 km to 8 km as patience is consumed. At most 20 nearby vehicles per order are retained before routing. Routed pickup time is then computed with Valhalla; invalid matrix cells receive a narrowly scoped single-route fallback under the frozen adapter. Candidates that cannot reach the pickup before the remaining deadline are discarded.

Vehicle-order candidates are filtered in a fixed order. Human-driven vehicle (HV) candidates require spatial, temporal, and routing feasibility. AV candidates additionally require passenger acceptance, route identity and hard-state readiness, and the chosen operational policy. This produces a sparse bipartite graph \(\mathcal A_t\subseteq\mathcal O_t\times\mathcal V_t\). Importantly, counts at successive gates are opportunity counts \((o,v,t)\), not unique passengers or trips. An order may appear at multiple epochs while waiting, so the initial rolling stock cannot be interpreted as 30,000 independent one-time opportunities.

The fixed ordering is essential for attribution. If two filters would reject the same arc, the ledger credits the first reached filter rather than double counting the loss. The reported retention therefore describes the implemented prospective pipeline, not a Shapley decomposition of independent causal effects. Changing gate order could redistribute attributed counts even if the final candidate set remained identical. We consequently interpret each stage as an operational bottleneck within the frozen construction and avoid adding its percentages as if they were mutually independent treatment effects.

Unmatched orders remain in the rolling pool. Carry-over priority and expanding spatial neighborhoods operate before the final critical window, while an order is called critical only when no more than 30 seconds of patience remains. Because a candidate must still satisfy pickup time within remaining patience, critical recovery is inherently difficult and is not used as a standalone measure of policy quality.

### 4.2 Assignment constraints and lexicographic objectives

For each sparse arc \((o,v)\in\mathcal A_t\), binary variable \(x_{ovt}\) indicates assignment. Feasibility requires

\[
\sum_{v:(o,v)\in\mathcal A_t}x_{ovt}\le1\quad\forall o\in\mathcal O_t,
\qquad
\sum_{o:(o,v)\in\mathcal A_t}x_{ovt}\le1\quad\forall v\in\mathcal V_t.
\]

The optimizer uses a lexicographic hierarchy rather than a single arbitrary weighted objective. It first maximizes service to critical orders, then total assignments, then carried-over orders, and finally minimizes routed pickup time. In cost experiments, a fleet operating-cost term is appended only after higher-priority service objectives and is evaluated within a frozen tolerance \(\epsilon\). This ordering ensures that economic trade-offs are not allowed to silently sacrifice the principal service objective.

The use of binary assignment is operationally simple but analytically useful. Dispatch competition occurs only after candidates survive all upstream filters. A passenger-compatible, ODD-compatible AV arc may still lose to an HV or another AV because a vehicle can serve only one order and the optimizer must resolve the sparse graph jointly. Thus, pre-optimizer eligibility and realized AV service are distinct. This distinction is central to interpreting the effective-capacity mechanism.

### 4.3 Family-specific cumulative exposure control

For each AV \(v\), family \(f\in\{s,d,v\}\), and current horizon, let \(E^f_{v,t}\) denote accumulated exposure from previously selected service routes. A candidate assignment adds \(e^f_{ok}\). The strict policy admits only zero-exposure routes. More generally, the dispatch kernel can enforce

\[
E^f_{v,t}+\sum_{o:(o,v)\in\mathcal A_t}e^f_{ok}x_{ovt}
\le \Gamma^f_{v,t},
\]

where \(\Gamma^f\) is a family-specific cumulative allowance. Separate constraints prevent static, dynamic, and speed exceedances from canceling each other. The reference policy freezes \(\Gamma^f\) values calibrated once from the corrected \(q_A=0.25\), Moderate-profile, \(p=1\) unconstrained canonical trajectory. These values are then transferred without retuning to the central comparison.

The unconstrained policy retains hard-state and passenger gates but does not impose cumulative continuous-exposure bounds. The strict, reference, and unconstrained policies therefore isolate three operational postures: zero modeled exceedance, a frozen tested allowance, and no continuous allowance. “Reference” means a preregistered compromise used for comparison; it is not labeled optimal, safe, or a discovered knee point.

### 4.4 Structural propositions and computation

Three structural statements organize the analysis. **Proposition 1 (capacity non-equivalence):** increasing \(q_A\) does not guarantee higher service because an AV active hour is filtered by acceptance, route readiness, profile compatibility, pickup feasibility, and assignment competition. A nominal supply substitution can therefore reduce effective dispatchable capacity. **Proposition 2 (acceptance-set nesting):** higher passenger acceptance weakly enlarges the AV candidate set under common random numbers, but service need not increase one-for-one because newly eligible arcs may be redundant or lose in assignment competition. **Proposition 3 (capability-envelope nesting):** a broader nested profile weakly lowers descriptor utilization and relaxes profile-based candidate restrictions, yet the realized gain depends on whether the previously excluded arcs are temporally and spatially useful. Proposition 4, placed in the appendix, establishes why separate family constraints cannot generally be replaced by a weighted aggregate. Formal proofs and boundary conditions are given in the appendix.

The implementation is designed around sparsity. A cKDTree retrieves local candidates; only Top-K pairs are routed; repeated route lookups use a cache; and the assignment matrix is built in sparse form for the SciPy/HiGHS mixed-integer solver. No \(|\mathcal O_t|\times|\mathcal V_t|\) dense matrix is constructed. Earlier benchmarks reduced hundreds of thousands of spatial pairs to a few thousand valid arcs, and solver time was negligible relative to routing. The engineering implication is that routing and caching, not GPU acceleration or a denser optimization formulation, are the relevant computational bottlenecks.

[Figure 1 about here]

## 5. Experimental design

### 5.1 Demand, fleet normalization, and factorial design

**Canonical timing assumption.** All 41 formal scenarios treat the observed trajectory departure/boarding proxy as simulated request release: lead time is zero and pickup patience is 300 seconds. These are not observed platform request timestamps. An order released at \(r_o\) has deadline \(D_o=r_o+300\), with decisions every 30 seconds. Recovered RT sensitivity shows that this assumption affects mechanism magnitudes; the full-day numbers must not be interpreted as timing-invariant effects.

The main experiment is a full-day deterministic replay of 30,000 Test31 requests. Demand, request patience, vehicle sessions, route interface, M3 prediction checkpoint, random acceptance uniforms, candidate construction, and optimizer settings are frozen before scenario execution. The main factorial combines three baseline-normalized realized AV active-hour shares \(q_A\in\{0.25,0.50,0.75\}\), three capability profiles \(k\in\{C,M,A\}\), and three passenger acceptance levels \(p\in\{0.40,0.70,1.00\}\), yielding 27 mixed-fleet scenarios.

The primary outcome is service rate: served requests divided by 30,000. Secondary outcomes include AV service share, patience expiration, candidate survival through same-unit gates, selected family exposure, pickup objective, and computation diagnostics. The main factorial disables cumulative \(\Gamma\) controls and additional cost penalties so that the effects of fleet composition, acceptance, and capability are not confounded by a second policy layer.

The factorial estimands are paired finite-population contrasts on this demand day. For example, the acceptance contrast compares two nested passenger gates holding the fleet session realization, profile, and request stream fixed. The capability contrast changes the frozen envelope while holding acceptance and composition fixed. The composition contrast changes the mix of empirical HV sessions and full-horizon AV availability at a fixed baseline-normalized target; it is not a label-only intervention and includes the corresponding spatial-temporal consequences. We report these as operational scenario contrasts rather than estimates of a population-wide behavioral causal effect.

No hyperparameter, profile cap, policy allowance, or routing fallback is changed after inspecting the 27 outcomes. This freeze is especially important for the reference exposure policy: it is transferred from one designated calibration trajectory and is not selected because it happens to look favorable at the central scenario.

The exact all-HV active-hour denominator is \(12{,}279.336389\) hours. We use “baseline-normalized realized AV active-hour share” throughout because \(q_A\) is neither a simple vehicle-count fraction nor the within-scenario share of only those hours that happened to remain active after assignments. Scenario manifests bind the realized sessions to the same baseline denominator.

The design uses one frozen acceptance draw per order across all scenarios. Consequently, \(p=0.40\) is nested within \(p=0.70\), which is nested within \(p=1.00\). Capability profiles are nested in the same direction by their frozen caps. These design choices eliminate independent resampling as an explanation for pairwise changes and make unexpected nonmonotonicity auditable at the assignment layer. They do not remove the possibility that service changes less than eligibility, because multiple feasible arcs may refer to the same order or vehicle and only one can be selected.

### 5.2 Benchmarks, ODD policies, and cost robustness

Four composition benchmarks supplement the factorial: an all-HV replay and all-AV replays under C, M, and A profiles. The all-HV result is a conventional service anchor. The all-AV cases are composition extremes, not upper bounds: removing HV flexibility can make an all-AV scenario perform worse than a mixed fleet, especially when passenger acceptance or operational compatibility restricts AV assignments.

ODD-policy analysis uses the central condition \(q_A=0.50\), \(p=0.70\), and Moderate capability. Strict, reference, and unconstrained continuous-exposure policies share the same hard route state, acceptance draws, demand, and fleet. Only their family-exposure constraints differ. This paired construction identifies the service and AV-use consequences of the continuous policy layer without changing nominal supply.

Cost robustness is evaluated with tolerance \(\epsilon=0.05\) and operating-cost coefficients \(\eta\in\{0.50,0.75,1.00,1.25\}\), compared with \(\eta=0\). The tolerance protects the lexicographically prior service objective; cost is allowed to distinguish solutions only within the frozen acceptable band. Conclusions are consequently stated within the tested \(\eta\) range and do not imply a globally optimal fleet-cost schedule.

### 5.3 Frozen-state prediction ablation and rebalancing baseline

The prediction ablation contains ten preregistered fixed decision states. In each state, a prediction-informed condition (P) and a history-based condition (H) receive identical order and vehicle snapshots. The replay records target errors, candidate eligibility, sparse AV-arc identity, selected dynamic exposure, and pickup-time objective. It is intentionally a mechanism experiment, not a second full-day dispatch evaluation.

The main full-day scenarios use no active repositioning. This baseline preserves the interpretation of fleet substitution without allowing a learned or tuned rebalance policy to compensate differently across treatments. A later deterministic spatial-closure exercise tests frozen, nonadaptive repositioning variants for robustness, but it is not used to redefine the main estimand and is reported as a bounded extension rather than a new optimized policy.

Computational diagnostics are treated as part of experimental validity. All scenario runners process one day and one sparse epoch state at a time, release intermediate data, and avoid GPU dependence. Route caches are bounded by the candidate process rather than precomputing a citywide matrix. Scenario manifests record completion and routing failures. This architecture supports exact replay on the available workstation while preventing a change in memory strategy from becoming an unreported difference between treatments.

### 5.4 Fixed-state mechanism and timing diagnostics

Mechanism validation is separate from the 41 full-day outcomes. Ten existing zero-lead states are used for neutral identity and AV gate graphs. Relabeling holds inherited availability policy fixed; canonical HV empirical sessions and AV full-horizon availability are unchanged. For each sparse gate graph, \(E_t^g=|A_t^g|\) counts distinct order–vehicle edges, \(U_t^g\) counts waiting orders with an AV option, and \(M_t^g=\nu(G_t^g)\) is maximum bipartite matching cardinality. This M denotes matching capacity, not the Moderate profile or static movement descriptor. K=10/20/40/80 changes only sparsification; selected assignments are not an additional feasibility gate.

Timing sensitivity uses q_A=.50/.75, Moderate capability, acceptance=.70, and 12:00/17:30 local time: four physical states, each paired across zero lead and RT-Low/Base/High. The unchanged original transform uses Train 20161019–22 gap P25/P50/P75/P90 = 420/748/1733/3693 seconds, OD-dependent lower bounds, stable order/scenario jitter, clipping and the original UTC business-day convention. Fifteen surviving summary fingerprints on 114,356 legacy orders reproduce with maximum error approximately \(2.27\times10^{-13}\) seconds. This is a **fingerprint-verified reconstruction**, not recovery of the original missing manifest or per-order hashes.

The transform covers all 30,000 Test31 orders. Within each state, vehicles, availability, canonical prior assignments, route descriptors, routing clock and gate rules stay fixed; release and its derived waiting membership, failed-round proxy and remaining patience change. This is conditional-state sensitivity, not a new RT vehicle history or a full-day replay. Frozen descriptors are not asserted available at earlier release, and no earlier-decision prediction validation is implied. Aggregate patience retention divides summed post-patience M by summed pre-patience M, rather than averaging state percentages.

## 6. Results

### 6.1 Empirical grounding

The upstream products provide a near-complete operational bridge from high-quality observed trajectories to the directed road representation. Coverage is not treated as evidence that every original trace is usable: low-quality or unresolved orders are rejected before the Stage 4 core set, and historical reverse identities remain explicit overlays rather than being converted into convenient forward traversals. The resulting interface therefore prioritizes identity correctness and conservation over maximal retention.

Static and dynamic descriptors are available on the selected service routes used in Test31. The corrected boundary-road-class descriptor ensures that ordinary single-node intersections can exhibit diversity even when they contain no internal consolidation edge. The 12 dynamic descriptors retain crawl, stop, speed-variation, and acceleration-variation information separately. These products make it possible to attribute candidate removal and selected exposure by family instead of reporting a single opaque AV-eligibility label.

### 6.2 Fleet transition and benchmark anchors

Under the canonical zero-lead, 300-second-patience replay, mean service rate across the 27 mixed-fleet factorial scenarios falls monotonically with the baseline-normalized realized AV active-hour share. At \(q_A=0.25\), the mean over profiles and acceptance settings is 0.7258. It decreases to 0.5984 at \(q_A=0.50\) and 0.3924 at \(q_A=0.75\). The change from 0.25 to 0.75 is \(-0.3334\), a relative decline of approximately 45.9%. Because demand and total baseline-normalized active-hour exposure are paired, this pattern is not caused by sampling different request days. It indicates that substituting nominal AV active hours for HV active hours does not preserve the same dispatchable service opportunity.

Benchmark cases sharpen the interpretation. The all-HV replay serves 0.7889 of requests. Under the Moderate profile and \(p=0.70\), service rates are 0.7297, 0.6044, and 0.4013 for \(q_A=0.25,0.50,0.75\), respectively. The all-AV Moderate case serves only 0.1515. This all-AV result is not an AV performance ceiling; it is a composition extreme in which no HV capacity remains to serve passengers or routes filtered out of the AV candidate set. The comparison demonstrates why nominal fleet share and effective service capacity must be separated.

[Table 1 about here]

### 6.3 Matching-capacity mechanism and timing qualification

#### 6.3.1 Neutral identity and genuine capacity loss

Neutral identity passes in all ten states: candidate graphs, selected assignments and pickup objectives agree when only labels change while availability policy is inherited. The previously coupled session rule favors AV admission rather than suppressing it. This diagnostic neither modifies canonical HV=empirical-session/AV=full-horizon definitions nor establishes equality between those different policies.

**Table MC1. Gate-level AV capacity summed over ten canonical zero-lead fixed states.** Counts are not unique daily orders or selected assignments.

| Gate | E: edges | U: orders with AV option | M: maximum matching | Loss of M |
| --- | ---: | ---: | ---: | ---: |
| Spatial opportunity | 95180 | 720 | 718 | — |
| Passenger compatible | 65444 | 493 | 493 | 225 |
| Structural ready | 31153 | 238 | 238 | 255 |
| Evidence complete | 15775 | 124 | 124 | 114 |
| Top-K compression | 1371 | 124 | 124 | 0 |
| Route returned | 1371 | 124 | 124 | 0 |
| Pickup/patience feasible | 48 | 12 | 12 | 112 |
| Solver eligible | 48 | 12 | 12 | 0 |

Passenger, structural, evidence and patience gates remove genuine maximum-matchable orders. Top-K instead removes 14,404 of 15,775 pre-compression edges (91.31%) without M loss. Final M is identical at K=10/20/40/80 in every sampled state, although final E totals are 38/48/50/55. Thus K20 is not driving the sampled loss; this is not a guarantee over all states. U and M differ when orders compete for vehicles, as spatial U=720 versus M=718 illustrates.

#### 6.3.2 Supporting full-day opportunity accounting

The prospective \((o,v,t)\) ledger reports solver-input/spatial-opportunity ratios of 0.0939%, 0.0720% and 0.0445% at q_A=.25/.50/.75. These repeated-epoch edge ratios are neither passenger acceptance rates nor fractions of matching capacity retained. They supply opportunity-volume context; Table MC1 is the primary matching-capacity evidence. The ten-state result does not causally allocate the entire full-day service decline among gates. Final assignment competition must also remain separate from eligibility.

#### 6.3.3 Sensitivity qualification

**Table RT1. Conditional timing sensitivity summed over four physical states.** This population differs from Table MC1 and must not be pooled with it.

| Request-time variant | Pre-patience M | Final M | Aggregate patience retention |
| --- | ---: | ---: | ---: |
| Canonical zero lead | 53 | 8 | 15.09% |
| RT-Low | 130 | 26 | 20.00% |
| RT-Base | 128 | 25 | 19.53% |
| RT-High | 156 | 40 | 25.64% |

Operational/passenger gates still reduce M, but timing materially changes its magnitude. Local q ordering also changes: at noon final M for q_A=.50/.75 is 4/3 under zero lead and 10/11 under RT-High; at 17:30 it is 1/0 and 8/11. Because waiting cohorts differ and canonical vehicle histories are held fixed, these are not order-level treatment effects or evidence of a reversed full-day service ordering.

The classification is **RT_SENSITIVE**, qualifying rather than overturning the mechanism. Genuine matching-capacity attrition remains, but the magnitude of conversion loss interacts with request reconstruction and remaining pickup patience:

\[
M_t^g=M_t^g(\text{operational gates},\text{request timing},\text{pickup patience}\mid\text{fixed physical state}).
\]

[Figure 4A–C about here: canonical full-day service; ten-state E/U/M mechanism; four-state RT qualification. Separate populations and denominators.]

### 6.4 Acceptance, capability, and family activity

Passenger acceptance and capability expansion both improve service, but their effect depends on how binding the corresponding gate is. Raising acceptance from \(p=0.40\) to \(p=1.00\) under the Moderate profile increases service by 0.0087 at \(q_A=0.25\) and by 0.0477 at \(q_A=0.75\). The larger high-\(q_A\) gain is consistent with a fleet in which more potential service depends on AV eligibility. Even then, acceptance alone cannot restore all-HV performance because accepted candidates remain subject to route, evidence, patience, and competition constraints.

Capability expansion is also more consequential when AV dependence is high. At \(q_A=0.75\) and \(p=0.70\), moving from the Conservative to the Advanced profile increases service by 0.0330. The gain is real but smaller than the service loss associated with the broad fleet-composition shift. This indicates that a broader envelope recovers some routes without resolving every source of effective-capacity loss.

Acceptance and capability are complementary but not interchangeable. Acceptance determines whether a passenger permits an AV option; capability determines whether the route falls within the modeled operational envelope. Raising one parameter cannot recover arcs removed exclusively by the other. Moreover, both operate before pickup feasibility. A universally accepting passenger with an Advanced-compatible route still receives no AV candidate if the reachable vehicles cannot arrive before the deadline. The factorial therefore should be read as a map of interacting bottlenecks rather than three separable elasticities.

Family activity clarifies what the profiles actually change. Under Conservative capability, the shares of evaluated opportunities with active static, dynamic, and speed constraints are 0.9248, 0.7739, and 0.7073. Under Moderate capability, they are 0.8698, 0.4764, and 0.0006. Under Advanced capability, they are 0.6394, 0.1822, and 0.0000. Static intersection and movement structure remains active even for the broadest profile, while speed nearly disappears as a binding family beyond the Conservative boundary. The main differentiation in this setting therefore comes from static structure and dynamic operating conditions, not from speed alone.

[Table 2 about here]

### 6.5 ODD policy comparison

At the central condition \(q_A=0.50\), \(p=0.70\), and Moderate profile, the strict zero-exposure policy achieves service rate 0.5532 and AV service share 0.0113. The frozen reference allowance increases these values to 0.6038 and 0.1244. The unconstrained continuous policy yields 0.6044 and 0.1217. Thus, strict zero-exposure control sharply reduces AV use and total service, whereas the transferred reference allowance nearly matches unconstrained service.

The near-equality in service does not mean the two policies select equivalent routes. Relative to unconstrained operation, the reference policy lowers service by about 0.0007 while reducing selected static exposure by 9.6% and dynamic exposure by 5.6%. The AV service share is slightly higher under reference in this run, illustrating that cumulative controls can alter which AV assignments are preserved rather than simply suppressing all AV use. This is a tested trade-off under one central condition. We do not characterize the reference allowance as universally optimal or as a safety boundary.

The result supports a two-layer interface. Hard states exclude known structural violations or unresolved critical evidence; continuous family exposures allow the dispatch policy to manage how often and how strongly selected routes exceed a profile envelope. Collapsing both layers into a binary suitability flag would conceal the difference between impossible, unknown, marginal, and heavily utilized operating conditions.

[Figure 5 about here]

### 6.6 Prediction-to-decision evidence

The paired fixed-state ablation shows that prediction information changes operational decisions even when mean absolute error is not uniformly lower for every target. Across ten preregistered states, the prediction-informed (P) and history-based (H) conditions differ in 9 of 10 states. Mean selected-AV-arc Jaccard is 0.10. H supplies on average 9.1 more solver-input AV arcs, yet its selected routes have 2.402 greater dynamic exposure and its pickup-time objective is 64.02 seconds worse in the audited comparison. Across 14,374 sparse arcs, routing records zero failure.

Target-space errors explain why a single accuracy ranking would be incomplete. P versus H mean absolute errors are 0.0636 versus 0.0669 for crawl and 0.0224 versus 0.0252 for speed variation, favoring P. For stop share, they are 0.0179 versus 0.0023, and for acceleration RMS, 0.0739 versus 0.0572, favoring H. The prediction-informed state is therefore not universally more accurate. Nevertheless, it changes which candidate arcs are admitted and selects routes with lower audited dynamic exposure and pickup objective in this fixed-state experiment.

These findings establish decision relevance, not a full-day causal service gain from prediction. The ablation isolates ten states and completes in 91.19 CPU seconds. A full-day paired prediction-policy experiment would be needed to estimate system-wide service, waiting, and exposure effects. The present result instead answers a narrower question: whether leakage-safe multivariate forecasts convey information that materially changes the sparse dispatch decision. They do.

[Table 3 about here]

### 6.7 Cost robustness

Within the tested tolerance \(\epsilon=0.05\), adding the operating-cost term reduces the audited cost measure relative to \(\eta=0\) by 1.60%, 1.01%, 0.54%, and 1.14% for \(\eta=0.50,0.75,1.00,1.25\), respectively. The response is not monotone in \(\eta\), which is unsurprising under a discrete lexicographic assignment with a protected service band. The appropriate conclusion is limited: modest cost reductions are available within this coefficient range without overriding the higher-priority service criteria. The experiment does not identify a globally optimal \(\eta\), nor does it support extrapolation beyond the frozen tolerance and tested values.

[Figure 4 about here]

[Figure 5 about here]

## 7. Discussion and limitations

### 7.1 Conditional effective-capacity conversion

Under the canonical zero-lead, 300-s-patience replay, increasing nominal AV active-hour share is associated with lower realized service. Fixed-state mechanism analysis shows that passenger and operational gates remove genuine maximum-matchable capacity rather than merely redundant candidate edges; however, the magnitude of this conversion loss interacts materially with request-time reconstruction and remaining pickup patience.

Three evidence levels remain separate. The 41 full-day scenarios establish operational contrasts on common demand. Ten-state E/U/M validates substantive gate attrition and distinguishes redundant Top-K edges from lost matching capacity. Four-state RT sensitivity qualifies magnitudes and local cross-fleet ordering. Neither fixed-state analysis causally decomposes the whole daily service curve.

Active-hour normalization is necessary but insufficient to measure dispatch capacity. Availability becomes useful when passengers, routes, evidence and pickup reachability permit simultaneous matching. Neutral identity excludes label-only asymmetry in the controlled sample; it does not erase the intended distinction between empirical-session HVs and full-horizon AVs.

HV flexibility is a plausible interpretation of the canonical substitution pattern, not a timing-invariant productivity theorem. The all-AV Moderate result is a composition extreme under the tested assumptions, not a technology ceiling. RT sensitivity warns against generalizing either effect sizes or fleet ordering. The scientific classification is QUALIFIES_CURRENT_STORY: the mechanism survives with a substantive timing/patience qualification.

Opportunity counts, maximum matching and realized service measure different objects. M describes simultaneous feasibility before selection; service additionally reflects assignments and vehicle evolution. Explicit timing assumptions and E/U/M reporting prevent these quantities from being treated as interchangeable.

### 7.2 Operational envelopes and prediction as decision information

The hard-plus-continuous interface provides a more useful dispatch abstraction than a single “AV suitable” indicator. Known direction or movement violations can remain hard constraints; unresolved evidence can remain unknown; and routes near or beyond a capability cap can be ranked and controlled continuously. Keeping static, dynamic, and speed utilization separate preserves diagnostic meaning. In the present network, speed is nearly inactive for Moderate and Advanced profiles, while static and dynamic families remain consequential. A single weighted risk index would hide this empirical structure.

The central policy comparison shows why continuous exposure matters. Strict zero exposure excludes most AV service and lowers total service. The frozen reference allowance nearly reproduces unconstrained service while reducing selected static and dynamic exposure. This does not identify an optimal ODD policy. It demonstrates that a platform can represent and enforce a transparent family-level operating posture rather than relying on a binary label or an unreported weighted score. Such an interface can later support regulator-, operator-, or manufacturer-specified limits without changing the underlying dispatch formulation.

Prediction remains decision information, not proven decision superiority. The fixed-state ablation changes candidate and selected-assignment identities despite mixed target-wise accuracy. Its reported exposure and pickup-objective differences are diagnostics under the existing comparison; they are not evaluation under a common independent outcome layer. That evaluator has not been run. The supported classification remains DECISION-RELEVANT, not decision-superior, and neither the prediction ablation nor RT sensitivity establishes full-day prediction benefit [CITATION NEEDED — predict-then-optimize and decision-focused evaluation].

### 7.3 Passenger acceptance and managerial interpretation

Passenger acceptance is a first-order part of effective capacity, especially as the fleet becomes AV-heavy. The larger gain from increasing \(p\) at \(q_A=0.75\) shows that behavioral compatibility becomes more binding when fewer HV substitutes remain. Yet even universal modeled acceptance does not recover all lost service, because route evidence, operational envelopes, pickup feasibility, and competition remain. A deployment strategy that treats adoption as the sole bottleneck would therefore overstate the service benefit of improving acceptance.

For platform managers, the results suggest three operational metrics beyond nominal AV count: realized active-hour exposure, opportunity survival by gate, and the distribution of selected family utilization. These metrics answer different questions. Active hours measure supplied availability; the gate ledger measures conversion; selected utilization measures the operating conditions actually assigned. Monitoring only completed AV trips would conflate them.

Capability expansion also has targeted value. The Advanced profile recovers more service than Conservative capability in the high-\(q_A\) condition, but static structure remains active even under the broad profile. Investment priorities should therefore be informed by family attribution rather than by a generic “broader ODD” objective. In this case, improving intersection and dynamic-condition handling is more relevant than expanding speed capability beyond the Moderate boundary. This is a contextual diagnostic, not an engineering prescription for all cities.

### 7.4 Limitations and future research

Timing is a substantive design limitation, not a secondary implementation detail. Trajectory departure/boarding is not an observed platform request time. All 41 formal scenarios use zero lead and 300-second patience. Fingerprint-verified RT reconstruction reproduces aggregate summaries but does not recover the original manifest or per-order hashes, and true request times remain unidentified. The original UTC boundary convention is retained.

The four-state timing diagnostic holds canonical history and descriptors fixed while release changes waiting cohorts and remaining patience. It cannot establish RT-specific daily service, full-day q ordering or prediction validity at earlier release. State-summed M is not a daily unique-order or selected-assignment count. Neutral identity and Top-K stability also have sampled-state scope.

Several limitations bound the conclusions. First, the empirical setting is one city and one historical period. Demand structure, road geometry, signalization quality, and driver sessions may differ elsewhere. Multi-city and multi-season validation is needed before estimating transferable effect sizes.

Second, passenger acceptance is an exogenous nested parameter, not an estimated behavioral model. It does not capture heterogeneity by trip purpose, wait time, price, familiarity, or passenger characteristics. This simplification is intentional for mechanism identification but limits welfare interpretation [CITATION NEEDED — empirical passenger acceptance of autonomous ride-hailing].

Third, capability profiles are operational envelopes derived from observed and predicted descriptors. They are not manufacturer specifications, failure probabilities, or safety certification. Static OSM attributes may be incomplete, and dynamic predictions carry estimation error. UNKNOWN states mitigate but do not eliminate evidence limitations.

Fourth, the main experiments use a fixed route construction and no active repositioning. Joint route choice could find alternative compatible paths, while a well-designed repositioning policy could improve spatial availability. A bounded deterministic closure has been conducted separately, but the study does not claim to optimize rebalancing. Future work should evaluate preregistered repositioning policies without tuning them on treatment outcomes [CITATION NEEDED — ride-hailing rebalancing under heterogeneous fleets].

Fifth, the prediction-to-decision experiment contains ten fixed states rather than paired full-day simulations. It demonstrates candidate and assignment sensitivity but does not estimate a full-day causal treatment effect. A future evaluation should freeze competing information policies and replay all scenarios end to end.

Sixth, travel demand is replayed rather than behaviorally endogenous. Prices, cancellation decisions, induced demand, fleet learning, charging, maintenance, and operator strategic response are outside scope. AV operating cost is represented only through a limited lexicographic robustness range. These omissions make the framework a dispatch-capacity study, not a complete market equilibrium or lifecycle assessment.

Finally, the results are deterministic conditional on frozen inputs and common random numbers. This strengthens paired comparison but does not quantify uncertainty from alternative demand days, map reconstructions, prediction checkpoints, or acceptance draws. Rolling-origin replications and bootstrap designs at the day level are appropriate next steps where sufficient independent days are available.

## 8. Conclusion

This study connects empirical ride-hailing trajectories to an ODD-aware mixed-fleet dispatch problem without assuming that an AV is an interchangeable replacement for an HV. The framework preserves directed route identity, uses leakage-safe multivariate predictions, separates hard route readiness from continuous static, dynamic, and speed utilization, and solves a patience-aware sparse rolling assignment over reconstructed vehicle sessions.

In the full-day replay, increasing the baseline-normalized realized AV active-hour share substantially reduces service under the tested conditions. Same-unit gate accounting explains the result: only a small and declining fraction of AV opportunities survives passenger, route, evidence, temporal, and competitive constraints. Broader acceptance and capability recover some service; a frozen continuous-exposure policy nearly matches unconstrained service while reducing selected exposure; and prediction changes decisions even when forecast accuracy is mixed across targets.

The general lesson is methodological. Fleet composition should be translated into effective dispatchable service capacity through observable operational gates, not treated as a nominal percentage. A hard-plus-continuous interface makes that translation compatible with sparse real-time optimization while preserving the limits of the evidence. This provides a foundation for future work on routing, rebalancing, passenger choice, and fleet economics without conflating operational compatibility with AV safety certification.

## References

_Verified references will be inserted in the authorized literature-verification phase. Semantic citation placeholders are registered separately._
