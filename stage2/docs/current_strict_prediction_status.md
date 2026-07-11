# Strict prediction round status

## Current decision

Stage0 remains passed and Stage1 remains an accepted retrospective measurement
pipeline. Stage2 is now split into an oracle-route upper-bound track and a future
deployable pre-dispatch track. Stage3 and Stage4 remain paused.

## Completed in this round

- decision-time contract and field reclassification;
- raw/percentile variance decomposition, ICC/repeatability and profile oracles;
- compact leakage-safe rolling profile builder;
- strictly lagged completed-bin traffic-state builder with timestamp audit;
- explicit actual-route oracle and shortest-path route prototypes;
- target semantics/reorganization documents;
- rolling-fold readiness audit.

## Main empirical result

The predictability analysis used up to 500k percentile rows per split and a 3%
order sample of raw primitives. Within `link_id x time_bin`, residual variance
shares were approximately:

| Target | Percentile residual share | Raw residual share |
|---|---:|---:|
| LCS | 94.2% | 67.4% |
| IIS | 90.9% | 46.4% |
| RTS | 92.6% | 88.0% |
| PMIS | 93.3% | 76.7% |

Historical link means were strong for raw IIS/LCS/PMIS but weak for percentile
targets. This supports separate modeling of expected raw stress, conditional
percentile anomaly and calibrated tail probability. RTS remains the least
repeatable target and requires dynamic state and uncertainty.

A 1% order-sample lagged-state pilot produced 188,876 train, 25,908 validation
and 20,020 test feature rows. All rows passed
`availability_timestamp < target_prediction_timestamp`. Recent same-link state
showed useful raw LCS/PMIS signal, but it remains an oracle-route result because
the prediction timestamp is actual link entry time.

## Blocking evidence

- Only one consecutive 7+1+1 fold is retained; at least three are required for
  rolling/OOF claims.
- True dispatch-time OD and platform planned routes are not stored separately.
  Shortest-path routes therefore use first/last matched links as OD proxies.
- In the 500-order-per-split routing pilot, directed shortest-path success was
  56.2%-57.6% and planned-link overlap with actual traversed links was only
  36.2%-38.8%. Realized supervision is therefore missing for many planned links.
- Physical directed-topology upstream/downstream lagged propagation is complete;
  route-conditioned propagation still awaits a deployable planned route.

## Next gate

Before model scaling:

1. retain enough additional compact days for at least three rolling folds;
2. freeze the Stage2 main targets as raw expectation plus tail probability;
3. decide whether endpoint proxies are acceptable or recover dispatch-time OD;
4. build planned-route estimated entry times and join lagged state causally;
5. run static -> rolling profile -> dynamic state -> route/topology residual
   ablations on common prediction keys.
