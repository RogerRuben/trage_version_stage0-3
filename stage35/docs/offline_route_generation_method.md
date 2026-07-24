# Offline HV/AV route generation method

## Inputs

- order OD and the frozen planned-route departure time;
- the canonical directed road network;
- Stage 2 dispatch-time link predictions;
- Stage 3 calibrated route-risk and ODD products;
- the pre-registered route-selection configuration.

Historical/revealed trajectories may define the primary HV replay baseline, but
they are labelled as an oracle-style fixed replay product. The planned shortest
expected-time HV route is retained as the formal robustness alternative.

## Candidate generation

For each order, generate shortest-expected-time, shortest-distance, K-shortest,
and risk-averse candidates. Candidate identity, link sequence, generation method,
and network version must be persisted before scoring.

## AV selection

Among candidates with no hard ODD violation and within registered distance/time
detour caps, minimize expected travel time plus registered risk and uncertainty
penalties. No coefficient or threshold may be tuned on the test-day result.

If no route passes, retain the order and report the minimum achievable risk,
binding dimension, violation count and `av_route_available=false`.

## Reproducibility gate

The output manifest must name the Stage 0 network, Stage 2 model, Stage 3
calibrator, ODD scenario, configuration hash, dates and code commit. Stage 4
must reject an input whose schema or manifest lineage differs from the declared
pipeline configuration.
