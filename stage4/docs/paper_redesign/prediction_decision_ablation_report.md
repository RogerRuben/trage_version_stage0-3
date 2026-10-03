# Frozen-state prediction-to-decision ablation report

## Result

The bounded analysis completed 10 preregistered snapshots and P/H/D0 in 91.19 seconds on CPU. It evaluated 14,374 sparse routing arcs with zero failures, used no GPU, and did not invoke a full-day simulator.

Relative to P, H changed the exact assignment set in 9/10 snapshots. Mean selected-AV-arc Jaccard was 0.10. H supplied on average 9.1 more solver-input AV arcs, raised selected dynamic exposure excess by 2.402, and raised the aggregate pickup objective by 64.02 seconds. D0 diagnoses removal of the dynamic control family; it is not a full-day policy estimate.

On held-out orders in these snapshots, P did not dominate H on every raw metric. Mean snapshot MAE for P versus H was 0.0636 versus 0.0669 for crawl, 0.0224 versus 0.0252 for speed-CV, 0.0179 versus 0.0023 for stop, and 0.0739 versus 0.0572 for acceleration-RMS. A blanket accuracy-superiority claim is therefore unsupported.

## Interpretation

Classification: **DECISION-RELEVANT**. Multivariate decision-time information materially changes dynamic-family exposure and the exact sparse assignment problem, although a historical median is competitive on some individual metrics. The defensible contribution is the measured connection from prediction to family-specific control, not universal predictive dominance.

These are one-epoch counterfactuals under identical physical states. They do not estimate full-day service, future redistribution, or a long-run treatment effect. The 23:00 snapshot produced no selected assignment under any variant and remains in the denominator as a valid zero-decision state.
