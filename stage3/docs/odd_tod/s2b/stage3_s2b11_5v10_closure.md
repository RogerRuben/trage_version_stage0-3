# Stage 3 S2B-1.1: 5m vs 10m Closure Pack

Status: `STAGE3_S2B11_FINAL_REVIEW_PAIR_READY`. `S2B1_ENGINEERING = PASS`; tolerance selection remains open. The prior 5m recommendation is withdrawn as scientific evidence because its rule structurally preferred smaller radii.

## Endpoint incompleteness

There are `13,663` endpoint-incomplete edges. Within 20m of a junction candidate: `5,638` (41.26%); signal: `779` (5.70%); a 5-to-10m changed area: `1,503` (11.00%). Relative to endpoint-complete edges, the three near shares are lower (ratios 0.44, 0.78, and 0.46), so there is no evidence of systematic enrichment around intersection evidence. The 1,503 changed-area-near edges remain a localized risk. These edges are excluded from topology rather than guessed.

By road class:
- motorway: 159/2,884 (5.51%)
- primary: 591/11,755 (5.03%)
- residential: 4,504/65,404 (6.89%)
- secondary: 1,047/19,842 (5.28%)
- service_other: 1,986/29,907 (6.64%)
- tertiary: 2,843/44,774 (6.35%)
- trunk: 354/5,134 (6.90%)
- unclassified: 2,179/29,754 (7.32%)

## Signal fragmentation

- 5m: `{"signal_complex_count": 2715, "signal_nodes_per_complex_distribution": {"1": 1943, "2": 602, "3": 37, "4+": 133}, "singleton_signal_node_share": 0.7156537753222836, "total_signal_node_assignments": 3792}`
- 10m: `{"signal_complex_count": 1629, "signal_nodes_per_complex_distribution": {"1": 438, "2": 670, "3": 80, "4+": 441}, "singleton_signal_node_share": 0.26887661141804786, "total_signal_node_assignments": 3792}`

## Degree sanity

`{"definition": "unique unordered endpoint pairs over endpoint-complete frozen auto-routable directed edges", "degree_distribution": {"0": 403, "1": 11765, "2": 19328, "3": 38073, "4": 19711, "5": 318, "6": 9}, "degree_ge3_count": 58111, "degree_ge4_count": 20038, "degree_max": 6, "node_count": 89607, "physical_undirected_edge_pair_count": 122564}`

## Targeted human adjudication

The pack contains `70` deterministic changed areas: `{"grade_separated": 10, "high_degree": 10, "multi_node_divided_road": 20, "random_changed": 10, "signalized": 20}`; `23` came from areas already represented in the prior QA pack. Each PNG shows 5m on the left and 10m on the right. The Parquet/CSV adjudication sheet is intentionally blank and accepts only `["5_CORRECT", "10_CORRECT", "BOTH_ACCEPTABLE", "NEITHER", "UNCERTAIN"]`.

`15m` and `20m` remain rejected baselines. `5m` versus `10m` is the final review pair. No tolerance is frozen and S2B-2 remains unauthorized.
