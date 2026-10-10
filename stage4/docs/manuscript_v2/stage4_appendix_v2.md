# Appendix and supplementary material v2

## A. Data filtering and map-matching QA

The upstream production process separates four quality axes. The GPS status records clean, local-outlier, sparse-or-ineligible, or unresolved-gap conditions. Route status records pass, partial, fail, or uncertain reconstruction. Dynamic status describes whether direct timing labels are strict, partial, or unusable. Canonical status records unique, chain-resolved, ambiguous, or unmapped network identity. Core Stage 1 orders require a passing route, usable GPS, unique or chain-resolved identity, and sufficient nonduplicated direct timing observations.

Raw-GPS route distance is evaluated against the temporally corresponding subtrace and final route component, not the nearest edge anywhere in an order. Continuous segment boundaries include preprocessing breaks, topology gaps, outlier windows, unmatched runs, and route-component changes. This prevents a distant but geometrically nearby component from masking a missed corridor. The audit also distinguishes local GPS anomalies from route failure; isolated outliers can lower only the affected segment, while a sustained parallel-route mismatch or missing main corridor fails the route.

Physical traversals and directly timed intervals are separate products. One uninterrupted visit to a canonical directed edge produces one traversal. A traversal may contain multiple GPS timing intervals, but each interval identifier belongs to exactly one classification and direct time is allocated at most once. Time and distance conservation, duplicate interval allocation, and non-direct observed-time violations are checked before downstream use.

## B. Network reconstruction and movement representation

The frozen network uses directed canonical edge identities. Historical observations in a direction not routable in the frozen graph are represented as historical-direction overlays and may trigger a hard AV incompatibility; they are not counted as missing and are not projected to the opposite edge. Unresolved identity interrupts the movement parser.

Intersection complexes are constructed upstream with a frozen 10 m consolidation tolerance selected through targeted 5 m versus 10 m adjudication. The Stage 3 parser recognizes incoming edge, zero or more internal edges, and outgoing edge. Movement identity therefore spans a complex rather than treating every adjacent edge pair as an independent turn. Grade-separated evidence, bridge, tunnel, and layer attributes are retained in construction diagnostics.

Static route descriptors use approach structure \(A\), parsed movement structure \(M\), boundary road-class diversity \(D\), and length or extent \(L\). The corrected diversity definition is

\[
D_c=\left|\{\operatorname{roadclass}(e):e\in\delta^-(c)\cup\delta^+(c)\}\right|,
\]

where \(\delta^-(c)\) and \(\delta^+(c)\) are incoming and outgoing boundary edges. Internal edges do not define road-class diversity. This correction changes the physical meaning for single-node intersections while leaving clustering and movement identity unchanged.

## C. Indicator definitions and prediction provenance

The dynamic interface contains four phenomena \(g\in\{\text{crawl},\text{stop},\text{speed CV},\text{acceleration RMS}\}\) and three summaries \(m\in\{E,Q,C\}\), producing 12 descriptors \(x_{g,m}\). Each is derived from the frozen M3 prediction under decision-time-only and predicted-progression-only inputs. Realized target columns are absent from the cache used by Stage 3 and Stage 4.

Training prediction caches are bound by date, path, SHA-256, row count, schema, model identifier, checkpoint hash, and leakage flags. Validation transforms use the same frozen training CDF and capability profiles. The Test31 dispatch does not refit normalization, vocabulary, CDF, calibration, or profile caps.

The profile quantiles are marginal. If \(B_{k,j}=F_j^{-1}(\pi_k)\), then \(P(X_j\le B_{k,j})\) is anchored near \(\pi_k\) for descriptor \(j\) under the calibration distribution. It does not follow that

\[
P(X_1\le B_{k,1},\ldots,X_{12}\le B_{k,12})=\pi_k.
\]

Dependence and simultaneous enforcement generally make joint route acceptance smaller. The profile labels therefore describe nested marginal envelopes, not route-level coverage guarantees.

## D. Analytical capability-envelope construction

For descriptor \(j\), route \(o\), and profile \(k\), utilization is \(r_{ojk}=x_{oj}/B_{kj}\). Family maximum \(\rho^f_{ok}\) is the largest ratio among descriptors in family \(f\). Overall utilization is the maximum over families. The construction has three useful properties:

1. **Monotonicity:** if every cap in profile \(k'\) is at least its cap in \(k\), then \(\rho^f_{ok'}\le \rho^f_{ok}\).
2. **Bottleneck preservation:** \(\rho^{overall}>1\) if and only if at least one family cap is exceeded.
3. **Attribution:** the maximizing family and descriptor identify the modeled bottleneck without an arbitrary cross-family weight.

Exposure \(e^f=[\rho^f-1]_+\) is zero inside the envelope and continuous above it. It is an operational distance from the frozen cap, not a calibrated probability of failure.

## E. Fleet reconstruction and spatial representativeness

Fleet sessions are reconstructed from observed driver activity intervals. The all-HV baseline contains exactly \(H^{base}=12{,}279.336389\) active vehicle-hours. AV labels are assigned to frozen sessions to target \(q_A\in\{0.25,0.50,0.75\}\) against that same denominator. The achieved shares are checked from realized sessions rather than inferred from vehicle counts.

Spatial representativeness audits compare fleet start locations, activity by time, and demand exposure across the transition samples. These diagnostics are designed to detect a sample that attains the active-hour target by selecting geographically atypical sessions. They do not force identical vehicle locations across compositions, because such duplication would remove the empirical spatial consequence of the selected sessions.

## F. Same-unit effective-capacity gate definitions

The prospective ledger uses opportunity arc \((o,v,t)\) throughout. Its conceptual stages are:

- \(N_0\): rolling spatial opportunity stock before AV-specific filtering;
- \(N_1\): passenger-acceptance survivors;
- \(N_2\): structural and hard-route survivors;
- \(N_3\): evidence-complete survivors;
- \(N_4\): pickup-patience survivors;
- \(N_5\): candidates reaching the frozen pre-optimizer interface after the remaining common gates;
- \(N_6\): selected assignments.

Exact implementation labels are bound in the source audit. A ratio \(N_{i+1}/N_i\) is meaningful only when both counts have the same opportunity unit. The Top-K step is reported separately as computational compression. \(N_5-N_6\) represents joint assignment competition, not a new statement that the discarded arcs were individually ineligible.

## G. Proofs of Propositions 1–4

**Proposition 1 (nominal share is not effective capacity).** Holding baseline active hours fixed, an increase in \(q_A\) need not weakly increase service.

*Proof.* Consider an HV session capable of serving at least one waiting order. Replace it with an AV session of equal active duration. If every AV arc from that session is removed by passenger acceptance, hard readiness, evidence, operational policy, or pickup deadline, the replacement adds zero feasible assignments while removing at least one HV assignment. The maximum cardinality of the epoch assignment can decrease. Repeating this substitution proves that greater nominal AV active-hour share does not imply greater effective service. \(\square\)

**Proposition 2 (nested acceptance enlarges eligibility).** With common random number \(u_o\), if \(p'\ge p\), the passenger-accepted AV candidate set under \(p\) is a subset of that under \(p'\).

*Proof.* Any order accepted under \(p\) satisfies \(u_o\le p\). Since \(p'\ge p\), it also satisfies \(u_o\le p'\). All other gates held fixed, no previously accepted arc is removed by increasing \(p\). \(\square\)

This proposition concerns eligibility, not realized service. Degeneracy, assignment competition, or altered lexicographic tie resolution can prevent a one-for-one service gain.

**Proposition 3 (nested profiles reduce utilization).** Suppose profile \(k'\) has \(B_{k'j}\ge B_{kj}>0\) for every descriptor. Then \(\rho^f_{ok'}\le\rho^f_{ok}\) and \(e^f_{ok'}\le e^f_{ok}\) for every family.

*Proof.* Division by a weakly larger positive denominator weakly reduces each ratio. Taking the maximum preserves the inequality. The positive-part operator is monotone, so exposure also weakly decreases. \(\square\)

**Proposition 4 (family constraints prevent cross-family compensation).** Separate cumulative inequalities for static, dynamic, and speed exposure cannot be replaced by one weighted sum without potentially admitting a candidate that violates one family allowance.

*Proof by construction.* Let a candidate have static exposure above its static allowance and zero dynamic and speed exposure. For any finite positive static weight, sufficiently large unused allowance or favorable weights in the other families can make a single weighted aggregate satisfy its bound even though the static bound is violated. Separate inequalities reject the candidate. Therefore the aggregate is not logically equivalent unless it reproduces all family constraints, in which case it no longer provides compression. \(\square\)

## H. Frozen-state prediction-ablation details

Ten states were preregistered before result comparison. P and H share orders, vehicles, positions, clock time, route calls, profile, acceptance, optimizer, and sparse candidate limits. Only the information used to construct dynamic route descriptors changes. Reported errors are computed on the same supported target rows. Decision outcomes include solver-input AV arc count, arc-identity overlap, selected dynamic exposure, and pickup objective.

The ablation evaluates a local information mechanism. It does not permit the following extrapolation: “P improves full-day service by the fixed-state difference.” A full-day estimate would require identical paired scenario replays under frozen P and H policies. The present inference is limited to candidate and assignment sensitivity at the audited states.

## I. Cost robustness

The cost term is lexicographically subordinate to service priorities. Under tolerance \(\epsilon=0.05\), a solution may trade within the specified service band before minimizing the operating-cost expression with coefficient \(\eta\). The tested set is \(\{0,0.50,0.75,1.00,1.25\}\). Because the feasible assignment set is discrete, increasing \(\eta\) need not yield a monotonic percentage difference relative to \(\eta=0\). No interpolation or claim outside this finite set is made.

## J. Routing reproducibility and repositioning-computation boundary

Routing uses a Valhalla matrix call for batches of vehicle sources to a pickup target. Failed cells receive a single-route fallback under the frozen adapter. Determinism checks bind route inputs, costing, timestamp handling, and cache keys. Routing statistics distinguish matrix batch queries, uncached arc evaluations, cache hits, and failed routing arcs; these quantities have different units and are not combined into one denominator.

The main manuscript estimand excludes active repositioning. Robustness work uses frozen deterministic policies and does not tune policy parameters on service outcomes. It also preserves sparse neighborhood queries and bounded candidate lists; no citywide dense order-by-vehicle or origin-by-destination matrix is constructed. Repositioning is therefore a bounded computational extension rather than a hidden component of the baseline fleet transition.
