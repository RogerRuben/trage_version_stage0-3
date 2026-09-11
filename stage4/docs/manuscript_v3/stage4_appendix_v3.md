# Supplementary material for Manuscript V3

## Appendix A. Empirical data and directed-route support

Core orders are screened separately for GPS quality, route agreement, dynamic timing support and canonical identity. A local outlier need not invalidate a supported main corridor; unresolved substantial gaps or corridor mismatch exclude an order from the core set. Raw GPS agreement is evaluated against the corresponding temporal subtrace and route component, not the nearest route anywhere in an order.

Physical traversals and direct timing intervals are separate. A continuous edge visit is one traversal; several GPS intervals can contribute timing observations without repeating that physical distance. Conservation and duplicate-allocation checks support the retained products. Historical reverse directions remain explicit overlays rather than being mapped onto forward identities; unresolved segments interrupt movement parsing.

Intersection consolidation uses the existing 10 m construction, selected after targeted 5 m/10 m visual review. Incoming, internal and outgoing directed edges define movements. Boundary road-class diversity is
\[
D_c=|\{\operatorname{roadclass}(e):e\in\delta^-(c)\cup\delta^+(c)\}|,
\]
not the diversity of internal edges alone. Layer, bridge and tunnel evidence inform representation of separated structures.

The evaluation set is a quality-selected 30,000-order subset, not all raw demand. Two orders have timestamps in the first minute after local midnight; the pre-specified replay horizon includes that minute. No new demand filtering is applied in the mechanism study.

HVs retain empirical-session admission; AVs retain full-horizon availability. The denominator is 12,279.336389 all-HV vehicle-hours. Composition changes the mix of those policies, not merely labels attached to otherwise identical sessions. Exact fixtures and spatial state are restored from existing scenario configurations and logs for the fixed-state comparisons.

Implementation persists partitioned products with atomic writes and resume support. Rejected orders have lightweight records, not complete route archives. These are engineering choices, not claims of methodological novelty.

## Appendix B. Operational prediction and information comparison

Four dynamic components—crawl, stop, speed variation and acceleration variation—are represented through the existing E/Q/C route descriptors. Training prediction values are transformed with the held-fixed weighted mid-CDF; route descriptors and analytical profile caps are not refit on evaluation outcomes. Marginal quantile caps do not imply equal joint route acceptance. Descriptor identities E/Q/C should not be confused with graph edge count E or effective matching capacity C.

The retained model identifier is M3. Its inputs use decision-time information and predicted progression, with training-only preprocessing and calibration. Cache/source records identify checkpoint and schema provenance; they do not turn operational predictions into safety outcomes. Evaluation-day realized travel time is reserved for post-assignment progression and diagnostic targets.

In the ten-state prediction comparison, predicted (P) and historical (H) information produce different decisions in nine states. Mean selected-AV-arc Jaccard is .10. H supplies 9.1 more solver-input AV arcs on average; this is an arc count, not 9.1%. The earlier diagnostic reports H-minus-P selected dynamic exposure of 2.402 and pickup objective of 64.02 seconds. These are retained within-comparison diagnostics, not independent outcome-value estimates.

Target MAEs (P/H) are crawl .0636/.0669, speed variation .0224/.0252, stop .0179/.0023, and acceleration RMS .0739/.0572. No uniform accuracy dominance is claimed. A common independent outcome evaluator was not run. The supported interpretation is DECISION-RELEVANT, not decision-superior or a daily causal service gain.

## Appendix C. Matching diagnostics and opportunity-volume evidence

For each sparse graph, duplicate order–vehicle pairs are removed before calculating E, U and maximum bipartite matching. AV subgraph M is a diagnostic of simultaneous AV serviceability; it is not mixed-fleet capacity or the selected AV count.

Ten physical-state reconstructions reproduce the existing registry. Controlled relabeling preserves availability policy, candidates, selected identities and pickup objective in all ten states. Earlier label/session coupling was AV-favoring and does not explain the observed decline in AV-heavy service. The comparison establishes invariance in the sampled states, not a proof about all possible branches or states.

The gate totals in main-text Section 6.2 come from matching_capacity_summary.csv. At K=10/20/40/80, final E sums are 38/48/50/55 and final M sums are 12 at all K values, with equality at each state. Top-K removes 14,404/15,775 = 91.31% of pre-compression edges at K20 without M loss. Route-return and solver-eligibility rows are retained separately so that compression, routing failure and joint policy constraints are not conflated.

The full-day prospective ledger counts repeated-epoch opportunities \((o,v,t)\). Solver-input/spatial-opportunity ratios .0939%/.0720%/.0445% for q_A=.25/.50/.75 are opportunity-volume evidence only. An order can occur in several epochs. These percentages do not estimate the fraction of unique passengers or maximum capacity retained. Selection is assignment competition, not another individually applied feasibility gate. Deadline-ever-eligible order coverage is not available from the retained aggregate logs and is not reconstructed.

## Appendix D. Structural proofs and local cost control

### Proposition 1: cumulative family-exposure guarantee

For every enabled family, assume additive updates
\[
Z_{t+1}^f=Z_t^f+\sum_{\mathcal A_t^A}e_{ov}^fx_{ov},\qquad
N_{t+1}^A=N_t^A+\sum_{\mathcal A_t^A}x_{ov}.
\]
Substitution into main-text (1) gives \(Z_{t+1}^f\le\Gamma_fN_{t+1}^A\). Division by a positive assignment count yields the mean-exposure bound. If no AV assignment exists, division is undefined; with zero initial state there is no incurred exposure. This is an assignment-weighted, system-level guarantee.

### Proposition 2: same-epoch feasible-set monotonicity

Hold the graph, current exposure state, objectives and all non-Gamma constraints fixed. For any old feasible x, \(N_t^A+\sum_{\mathcal A_t^A}x_{ov}\ge0\). Replacing each \(\Gamma_f\) by a weakly larger value increases or preserves the right side of (1). Thus every old feasible x remains feasible and \(\mathcal F_t(\Gamma)\subseteq\mathcal F_t(\Gamma')\).

Maximization of the highest-priority objective over a superset cannot yield a smaller optimum. No statement is made about the separate lower-priority optima if the higher-priority optimum changes, or about future states and daily service. The claim assumes a feasible original problem and holds the same current state fixed; it does not compare differently evolved policy histories.

### Proposition 3: separate budgets versus a weighted scalar

For positive cumulative assignment count, suppose \(\bar e_f=Z^f/N^A\le\Gamma_f\). Multiplying by \(w_f\ge0\) and summing proves
\[
\sum_fw_f\bar e_f\le\sum_fw_f\Gamma_f.
\]
The converse fails: two unit family caps and weights (1,1) permit scalar total 2, so (2,0) passes the scalar inequality but violates the first family limit. The example concerns compensation across families; the cumulative mean may still compensate high and low exposures within the same family.

### Proposition 4: local epsilon pickup-quality guarantee

Conditional on the critical-, total- and carry-over-match optima, let
\[
P_t^*=\min_x\sum_{\mathcal A_t}\widehat{\tau}_{ovt}x_{ov}.
\]
The optional cost level retains those count equalities and adds
\[
\sum_{\mathcal A_t}\widehat{\tau}_{ovt}x_{ov}
\le(1+\epsilon_W)P_t^*+\delta,
\]
where \(\delta\) is the numerical tolerance. Every feasible cost-stage selection obeys the inequality by construction. This is a local aggregate pickup-ETA guarantee, not a service-count relaxation, a per-passenger bound, or a daily mean/P95 waiting guarantee.

### Implementation correspondence and retained cost results

The solver row has coefficients \(e_{ov}^f-\Gamma_f\) on AV arcs, zero on HV arcs, and right side \(\Gamma_fN_t^A-Z_t^f\). This is algebraically identical to (1). State accumulates across AV assignments system-wide. No vehicle-indexed budget is implemented.

The actual hierarchy is critical, total, carry-over, pickup ETA, then optional operating cost. At \(\epsilon_W=.05\), the retained cost differences versus eta=0 are reductions of 1.60%/1.01%/.54%/1.14% for eta=.50/.75/1.00/1.25. They do not imply a monotone response or globally optimal coefficient. The earlier V2 description of an epsilon service band is superseded by the pickup-quality statement above.

Local verification sources: stage4/dispatch/solver.py, stage4/dispatch/exposure.py, and stage4/docs/paper_redesign/theory_notes_v2.md. No production mathematics or code was changed for V3.

## Appendix E. Request-time reconstruction and conditional-state scope

The original request-time generator is stage4/scripts/build_decoupled_abm_environment.py. Its original manifest and per-order output hashes are unavailable. Reconstruction uses retained all-order endpoint prescans, not only accepted modeling orders, for 20161019–22. Original coordinate conversion and chain processing yield gap P25/P50/P75/P90 = 420/748/1733/3693 seconds, 440,049 chain rows and 302,772 feasible rows. Feasible gaps are 60–7200 seconds.

The unchanged transform uses a response/pickup lower bound of 120 seconds, an OD-distance-dependent lower bound, the three chain quantiles capped by 1800 seconds, stable order/scenario jitter, and the original UTC business boundary with 60-minute warmup. Scenario positions are .25/.50/.75, not fixed lead durations. Request time is boarding proxy minus generated lead.

All 15 retained count/mean/P50/P90/clipped-share fingerprints on 114,356 legacy target-day orders agree, with maximum error approximately 2.27e-13 seconds. The correct provenance is **fingerprint-verified reconstruction**. Aggregate agreement does not recover the missing manifest or certify unavailable per-order hashes; reconstructed chain counts and empty-speed statistics are not independently verified archived numbers.

For the 30,000 evaluation orders, mean Low/Base/High leads are 311.013/498.148/1308.781 seconds. Four physical states hold canonical history, vehicles, descriptors and routing time fixed while release changes the waiting set and derived patience. This is not an earlier-decision forecast validation or an RT-specific rolling history. Original microsecond request times are retained.

The four-state pre/post-patience M pairs are (53,8), (130,26), (128,25), (156,40), aggregated by zero/Low/Base/High. Retentions are ratios of sums. Counts cannot be pooled with the ten-state mechanism totals. No selected-assignment or expired-order treatment effect is fabricated from these snapshots. The scope is RT-sensitive magnitude and local ordering, not full-day reversal.

Detailed sources: stage4/docs/paper_redesign/recovered_rt_environment_parameters.json and request_time_patience_sensitivity_report.md; output CSVs under stage4/output/paper_enhancement/mechanism_validity. Their development identifiers, including Test31, are retained here solely for traceability.

## Appendix F. Computational scope

Sparse neighborhood candidates and bounded routing caches avoid a dense order-by-vehicle matrix. Actual routing modes and failure handling belong to the corresponding pre-specified diagnostic configurations, not a new routing architecture. The ten-state identity/capacity checks use 30,779 routed arcs with zero failure; the RT comparison uses 5,979 with zero failure. These are different workloads and are not pooled as daily routing performance.

No new mechanism, rebalancing, common-evaluator or full-day experiment was run for the manuscript reconstruction. The current claim remains QUALIFIES_CURRENT_STORY. Engineering logs, partition/resume design and source hashes are implementation support, not central paper results.
