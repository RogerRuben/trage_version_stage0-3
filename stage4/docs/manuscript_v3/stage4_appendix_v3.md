# Supplementary material: Instantaneous Effective Matching Capacity in Mixed-Autonomy Ride-Hailing

## Appendix A. Data selection and route representation

Orders are screened separately for GPS quality, agreement between the observed trajectory and the represented route, usable dynamic timing observations, and consistent road identity. Limited local outliers need not invalidate an otherwise supported corridor. Substantial unresolved gaps and corridor disagreement exclude orders from the core sample. GPS-to-route distance is evaluated against the relevant temporal subtrace and route component, rather than against any nearby part of the trip.

Physical traversals and directly timed GPS intervals are represented separately. A continuous road-edge visit constitutes one traversal; several GPS intervals can provide timing observations for that traversal without duplicating its physical distance. Time and distance conservation and duplicate-allocation checks support the retained observations. Historical reverse travel remains distinct from forward road identity, and unresolved route segments interrupt movement interpretation.

Intersection complexes use a 10 m consolidation distance selected through targeted comparison with a 5 m alternative. Incoming, internal and outgoing directed edges define movements. For complex \(c\), boundary road-class diversity is
\[
D_c=|\{\operatorname{roadclass}(e):e\in\delta^-(c)\cup\delta^+(c)\}|.
\]
Internal edges alone do not define diversity. Layer, bridge and tunnel information describe grade separation.

The 30,000 evaluation orders are a quality-selected subset of recorded demand. Two orders have timestamps in the first minute after local midnight; the specified replay horizon includes that minute. All scenario comparisons retain the same demand set.

HVs use empirical driver-session availability, whereas AVs remain available throughout the modeled horizon. The all-HV reference contains 12,279.336389 active vehicle-hours. Composition changes therefore alter availability-policy mixtures as well as vehicle types. Fixed-state comparisons preserve the relevant scenario's positions, session policies and prior assignment history.

## Appendix B. Route descriptors and operational prediction

### B.1 Static and dynamic descriptors

Static descriptors are the number of external physical connections, number of topological movements, boundary road-class diversity, and internal road length of each intersection complex. Route descriptors take the maximum encountered value in each dimension. Static caps are calibrated using one observation per distinct training-exposed complex rather than weighting frequently traversed complexes more heavily.

Dynamic predictions cover crawl, stop, speed variability and acceleration variability. Let \(\widehat{x}_{o\ell d}\) be the prediction for component \(d\) on ordered route segment \(\ell\), and \(w_{o\ell}>0\) its predicted median traversal time in seconds. A component-specific, training-time-weighted distribution maps each prediction to \(z_{o\ell d}\). At an observed training support value \(x\), this mid-distribution is
\[
F_d^{mid}(x)=
\frac{\sum_i w_i\mathbf1(x_i<x)+\tfrac12\sum_i w_i\mathbf1(x_i=x)}
{\sum_iw_i}.
\]
Between support values, the mapping uses the cumulative weight below the new value. Route summaries are
\[
E_{od}=\frac{\sum_\ell w_{o\ell}z_{o\ell d}}{\sum_\ell w_{o\ell}},\qquad
Q_{od}=\frac{\sum_\ell w_{o\ell}\mathbf1(z_{o\ell d}>0.90)}
{\sum_\ell w_{o\ell}},
\]
and
\[
C_{od}=\max_{\mathcal R}
\sum_{\ell\in\mathcal R}w_{o\ell},
\]
where \(\mathcal R\) ranges over consecutive runs above the threshold. If no segment exceeds the threshold, \(C_{od}=0\). \(E\) is mean percentile exposure, \(Q\) is a time share, and \(C\) is a duration in seconds. These descriptor symbols are distinct from graph edge count and matching capacity.

Each complete training route contributes one observation when calibrating dynamic caps. Conservative, Moderate and Advanced caps use marginal quantile anchors of 0.75, 0.90 and 0.975, respectively. Meeting each marginal cap does not imply a corresponding joint route-acceptance rate. Speed caps are 60, 80 and 120 km/h. The profiles remain analytical scenarios, not validated operating domains.

Categorical assumptions distinguish maneuver capability from these continuous caps. The Conservative profile excludes roundabouts and U-turns; it permits signalized left turns, excludes stop- or yield-controlled left turns, and leaves left turns with unknown control unclassified. The Moderate profile permits left turns but excludes U-turns, while the Advanced profile permits both. Unresolved movements remain unknown, and independently established movement prohibitions take precedence. These assumptions are not empirical estimates of what all vehicles at a given automation level can perform.

### B.2 Prediction comparison

The retained multivariate predictor uses decision-time inputs and predicted route progression, with training-only preprocessing and separate calibration. Evaluation-day realized travel times are used for subsequent simulation progression and diagnostic targets, not substituted for missing dispatch predictions.

In ten fixed states, prediction-based (P) and historical-information (H) assignments differ in nine states. Mean selected-AV-pair Jaccard similarity is 0.10. Historical information supplies 9.1 more AV pairs to the assignment model on average; this is a count, not a percentage. Within that comparison, H-minus-P selected dynamic exposure is 2.402 and the pickup objective difference is 64.02 seconds. These are decision diagnostics, not independently evaluated gains.

Target mean absolute errors for P/H are 0.0636/0.0669 for crawl, 0.0224/0.0252 for speed variability, 0.0179/0.0023 for stop, and 0.0739/0.0572 for acceleration RMS, in their respective target scales. Neither source uniformly dominates. Since no common independent outcome evaluation was conducted, the supported conclusion is that information changes decisions, not that the prediction-based decisions are superior.

## Appendix C. Matching diagnostics

Duplicate request–vehicle pairs are removed before calculating \(E\), \(U\) and maximum bipartite matching. AV-subgraph \(M\) measures simultaneous AV service opportunities, not total mixed-fleet matching or the AV assignments actually selected by the lexicographic objective.

Controlled relabeling preserves availability policy, physical candidates, selected identities and pickup objective in all ten states. This distinguishes a vehicle-type effect from a change in session rules. The control supports invariance in the observed states, not a proof covering every possible state.

At \(K=10,20,40,80\), final AV-pair totals are 38, 48, 50 and 55, respectively. Final maximum matching sums to 12 at every value, with equality within each state. At \(K=20\), candidate compression removes \(14{,}404/15{,}775=91.31\%\) of preceding pairs without a matching-capacity loss. Routing availability and assignment admission are recorded separately to avoid attributing routing or policy effects to compression.

Full-day opportunity counts use repeated-epoch tuples \((o,v,t)\). The ratios of assignment-input pairs to spatially available pairs are 0.0939%, 0.0720% and 0.0445% for \(q_A=0.25,0.50,0.75\). A request can contribute to several epochs, so these ratios do not estimate unique-passenger coverage or retained maximum matching capacity. The available aggregate records do not establish how many requests had an eligible option at any time before expiry.

## Appendix D. Exposure-policy properties and operating cost

### D.1 Cumulative mean-exposure bound

For every enabled family, additive updates give
\[
Z_{t+1}^f=Z_t^f+\sum_{\mathcal A_t^A}e_{ov}^fx_{ov},\qquad
N_{t+1}^A=N_t^A+\sum_{\mathcal A_t^A}x_{ov}.
\]
Substitution into main-text Equation (1) yields \(Z_{t+1}^f\le\Gamma_fN_{t+1}^A\). Dividing by a positive assignment count establishes Proposition 1. With no AV assignments, the mean is undefined; from zero initial conditions, no exposure has been incurred. The bound is system-wide and assignment-weighted.

### D.2 Same-state feasible-set inclusion

Fix the current graph, cumulative exposure state, objectives and all non-exposure constraints. For an assignment feasible under \(\Gamma\), the quantity \(N_t^A+\sum_{\mathcal A_t^A}x_{ov}\) is nonnegative. Increasing \(\Gamma\) componentwise therefore weakly increases the right-hand side of each exposure constraint. Every previously feasible assignment remains feasible, proving Proposition 2.

The highest-priority maximum over this enlarged feasible set cannot decrease. Separate lower-priority objective values need not be monotone if the higher-priority optimum changes. Nor does the argument compare different daily histories, since assignments change future states.

### D.3 Separate and aggregated limits

For positive cumulative assignment count, let \(\bar e_f=Z^f/N^A\le\Gamma_f\). Multiplication by nonnegative weights and summation yield
\[
\sum_fw_f\bar e_f\le\sum_fw_f\Gamma_f.
\]
The converse fails. With two unit caps and equal weights, \((\bar e_1,\bar e_2)=(2,0)\) satisfies the aggregate limit but violates the first family cap. Separate limits prevent compensation across exposure families, while still allowing variation among trips within a family.

### D.4 Bounded pickup relaxation for cost minimization

Conditional on the optimal critical-, total- and carry-over-match counts, define
\[
P_t^*=\min_x\sum_{\mathcal A_t}\widehat{\tau}_{ovt}x_{ov}.
\]
An optional cost-minimization step retains these count equalities and requires
\[
\sum_{\mathcal A_t}\widehat{\tau}_{ovt}x_{ov}
\le(1+\epsilon_W)P_t^*+\delta,
\]
where \(\delta\) is numerical tolerance. Any resulting assignment satisfies this local aggregate pickup-time bound by construction. It is not a relaxation of service count, a per-passenger guarantee or a bound on daily mean waiting time.

The exposure constraint can equivalently be written with coefficient \(e_{ov}^f-\Gamma_f\) on each AV assignment and right-hand side \(\Gamma_fN_t^A-Z_t^f\). HV assignments have zero exposure coefficient. This form clarifies that the cumulative state is shared across AV assignments, not attached to individual vehicles.

At \(\epsilon_W=0.05\), the supplementary cost comparisons report reductions of 1.60%, 1.01%, 0.54% and 1.14% relative to the zero-weight cost case for cost weights \(\eta=0.50,0.75,1.00,1.25\), respectively. These results neither establish a monotone response nor identify an optimal weight.

## Appendix E. Request-time reconstruction

The alternative timing scenarios use endpoint records for all available orders on 19–22 October, rather than only the quality-selected modeling sample. Inter-trip chains produce gap percentiles P25/P50/P75/P90 of 420/748/1733/3693 seconds. There are 440,049 chain rows, of which 302,772 meet the 60–7200-second feasible-gap criterion.

Lead construction combines a 120-second response-and-pickup lower bound, an origin–destination-distance-dependent lower bound, training-gap quantiles capped at 1800 seconds, and stable request/scenario variation. Low, Base and High use positions 0.25, 0.50 and 0.75, respectively, rather than constant lead times. The reference procedure uses a UTC business boundary and a 60-minute warmup. Request time is the boarding proxy minus the generated lead.

The original parameter manifest and per-order hashes are unavailable. Reconstruction reproduces all 15 retained summary statistics covering count, mean, median, 90th percentile and clipped proportion for 114,356 historical target-day orders; maximum discrepancy is approximately \(2.27\times10^{-13}\) seconds. This is a reconstruction verified against surviving aggregate outputs, not recovery of the original per-order records. Reconstructed chain counts and empty-speed statistics are not independently verified archival values.

For the 30,000 evaluation orders, mean Low/Base/High leads are 311.013, 498.148 and 1308.781 seconds. The four-state comparison changes release times and resulting patience while retaining reference assignment history, vehicles, descriptors and routing time. It does not establish that those descriptors would have been available at the earlier reconstructed releases.

Pre-/post-deadline maximum matching pairs are \((53,8)\), \((130,26)\), \((128,25)\) and \((156,40)\) for zero/Low/Base/High, respectively. Retention is calculated as a ratio of sums. These counts are separate from the ten-state totals and do not estimate full-day service, expiry or assignment treatment effects.

## Appendix F. Computational considerations

Neighborhood search and bounded routing caches produce sparse candidate graphs rather than dense request-by-vehicle matrices. The identity and matching-capacity checks evaluate 30,779 routed pairs without routing failures; the timing comparison evaluates 5,979, also without failures. These workloads are distinct and should not be combined into a daily routing-performance estimate.

## Appendix G. Network representation and route-information limitations

### G.1 Conditional route selection

A separate action-space diagnostic examines changes in eligibility semantics while retaining fixed physical states. Current admission, removal of reverse-direction exclusion, relaxation of information completeness, their conditional combination, and neutral AV route eligibility yield final state-summed AV matching counts of 12, 39, 44, 41 and 113, respectively. These comparisons are distinct from the sequential requirements in main-text Table 2.

The combined condition uses the route selection obtained after removing reverse exclusion and then relaxes information requirements only for an established feasible route. It is not the union of the two single changes. Among the 720 orders, 26 admitted under information relaxation are not admitted under the combination: 23 have unresolved movements and three have unresolved route identity. Removing reverse exclusion changes the primary route from infeasible to unknown, which prevents selection of the previously available alternative under the existing route-selection rule.

These counts describe a diagnostic change to admission semantics. They are not estimates of deployable repair gains or of service on a corrected road network.

### G.2 Endpoint and movement coverage

The 720-order cohort contains 13,378 complex encounters. All 391 unresolved movement encounters, involving 276 orders and 117 unique keys, use endpoint-incomplete boundary edges excluded from movement construction. Internal connectivity alone does not establish a complete movement, and two recorded chains are discontinuous.

Inspection of the native routing graph recovers exact endpoints for 88 implicated edges. Nineteen conflict with existing non-null endpoint identities: 18 are associated with explicit transitions between network hierarchy levels, while one involves a different node at the same level. The missing sides include 55 native node identities in 51 transition groups that are absent from the node table used to construct complexes.

Relative to existing complex members, 17 of these nodes lie within 10 m and 14 within 10–20 m. For nodes farther than 20 m, adjacency excluding shortcut edges reaches one complex in 16 cases, multiple complexes in seven, and no directly anchored complex in one. These are contextual relationships, not established membership assignments. Forty-five of the 51 groups have branching or traffic-control evidence and cannot uniformly be interpreted as ordinary shape points.

The findings identify a mismatch between the road elements recognized along routes and those represented in the movement set. They do not justify automatically accepting the missing movements. The reported daily scenarios retain the original representation, so the service effect of reconciling these identities remains unmeasured.

### G.3 Direction evidence

One-way tags and opposed node ordering in the source OSM network support the direction exclusions for all 14,837 flagged evaluation-day orders, comprising 3,322 road identities and 85,538 route tokens. Five unresolved identities among the full 6,502-identity reverse representation do not occur in the evaluation sample.

This evidence supports the direction rule relative to the source network. It is not independent confirmation of restrictions in 2016, nor proof that no lawful alternative route existed. Direction evidence and unresolved movement coverage therefore carry different interpretations, even when both prevent AV admission.
