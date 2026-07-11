# Stress label semantics

Stage1 labels are retrospective measurements derived from completed
human-driven trips. They are trajectory-informed operational-stress proxies,
not direct AV safety outcomes and not causal estimates of driver burden.

## Native units and interpretation

| Construct | Native unit | Interpretation |
|---|---|---|
| LCS | order-link traversal | realized stop-go and longitudinal-control burden |
| IIS | turn movement | realized intersection/turn burden conditional on an applicable movement |
| GNS | static link/route context | geometry/navigation context rather than a primary dynamic target |
| RTS | order-link traversal | excess/tail travel-time burden relative to a historical reference |
| PMIS | order-link traversal | POI-exposure and behavior interaction, not an independent fifth physical outcome |

The current PMIS construction is approximately
`PMIS_raw = activity_intensity * (LCS_raw + RTS_raw) / 2`. Reports must show
POI exposure, behavioral stress and their interaction separately. PMIS must not
be presented as independent of LCS and RTS without an incremental-validity test.

## Required target forms

For LCS, RTS and PMIS retain raw realized stress, cohort percentile, tail event,
and repeatability/uncertainty. Raw expected stress is a baseline-exposure target;
cohort percentile is a relative anomaly target; tail probability is a candidate
ODD-style gate; uncertainty must be preserved for Stage3.

IIS uses a hurdle definition:

```text
IIS_applicable = planned movement is intersection/turn relevant
IIS_severity   = stress severity | applicable and label observed
```

Missing IIS must never be filled with zero. Applicability and observation
quality are separate concepts.

Monotonicity against construction indicators is an internal consistency check,
not independent construct validity. Stage1 also requires matcher sensitivity,
independent spatial/temporal cases, dimension discriminant validity,
repeated-traversal reliability and label-definition sensitivity.

