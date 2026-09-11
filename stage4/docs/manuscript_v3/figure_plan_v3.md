# Five-figure plan after V3 story reconstruction

This plan follows the completed V3 model/section reconstruction. It replaces pipeline-centered figure priorities. No final image is drawn and no additional experiment is authorized.

| Figure | Scientific role | Existing source | Proposed panels and units | Placement |
| --- | --- | --- | --- | --- |
| 1 | Define the capacity object | V3 Sections 2–3 | Nominal active supply → compatibility graph → maximum matching → rolling service; small graph illustrating E/U/M; note Gamma-constrained capacity is bounded by graph capacity | Section 2 |
| 2 | Does nominal supply translate into service? | Existing factorial and benchmark results | Service rate versus baseline-normalized AV active-hour share; benchmark markers; zero lead/300-s patience annotation | Section 6.1 |
| 3 | Do serviceability restrictions remove capacity? | mechanism_validity/matching_capacity_summary.csv | Primary U/M gate plot, secondary E strip; distinct Top-K compression marker | Section 6.2 |
| 4 | Which levers recover service and manage exposure? | Existing acceptance/capability contrasts and ODD policy results | Acceptance/capability service contrasts; strict/reference/unconstrained service and family exposure, separately labeled | Section 6.3 |
| 5 | How timing changes instantaneous capacity | mechanism_validity/request_time_sensitivity.csv | Four physical-state small multiples for zero/Low/Base/High; final M and pre/post-patience retention | Section 6.4 |

## Figure 1

Use the main-text S_nom, G, C_eff and Y notation. S_nom includes active vehicles, while the graph uses idle dispatchable vehicles. Distinguish available hours from instantaneous counts. Draw no implication that Y is the sum of M across epochs or that separate HV/AV matching capacities add. A policy box below graph capacity explains coupled Gamma constraints; it is not another independent edge filter. A simple conceptual graph is a definition illustration, not new empirical data.

## Figure 2

Retain factorial means .7258/.5984/.3924. Moderate/.70 values .7297/.6044/.4013 and all-HV .7889/all-AV Moderate .1515 can appear as labeled benchmark markers, not another factorial mean. Do not fit a smooth response or invented confidence interval to three deterministic means. Caption: evaluation-day service contrast conditional on zero request lead and 300-second patience.

## Figure 3

Gate order and M: 718/493/238/124/124/124/12/12 for spatial/passenger/structural/evidence/compression/route/patience/solver. U begins at 720; all later U equal M in these ten-state totals. E=95180/65444/31153/15775/1371/1371/48/48. Use separate scales for E and U/M. Counts sum ten states and are not unique daily requests. Compression annotation: approximately 91% edge removal, no M loss in the sampled K comparison. Old .0939%/.0720%/.0445% repeated-epoch ratios stay in the supplement.

## Figure 4

Acceptance contrast .40→1.00 under Moderate: service gains .0087 at q=.25 and .0477 at q=.75. Capability Conservative→Advanced at q=.75/.70: gain .0330. Do not compare these as identically sized or priced interventions. Policy service .5532/.6038/.6044 and AV share .0113/.1244/.1217; reference static/dynamic exposure reductions 9.6%/5.6% versus unconstrained. No common weighted risk or new return-on-investment estimate. Label Gamma as system cumulative mean exposure.

## Figure 5

q50 noon M=4/9/7/10; q50 evening=1/3/4/8; q75 noon=3/9/8/11; q75 evening=0/5/6/11. Four-state aggregate pre-patience M=53/130/128/156 and final M=8/26/25/40. Do not pool these with Figure 3. Retention is ratio of summed counts. Caption must state fixed canonical history/descriptors and changing waiting cohorts. No full-day RT outcome, treatment-effect interpretation or earlier-release forecast-validity claim.

Prediction diagnostics, target MAEs, reconstruction fingerprints, cost results, map examples and engineering details belong in the supplement. None becomes a sixth main figure. Final artwork awaits review of this V3 story.
