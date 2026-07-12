# Stage2 Deep v3 final contract

## Frozen prediction setting

```text
route condition: completed matched route as assigned-route proxy
prediction clock: estimated link entry time
state availability: every lagged feature timestamp < estimated link entry time
evaluation: three-fold rolling, fixed by rolling_threefold_config.json
data scale: 15,000 orders/day, 20161009-20161019
prediction key: date + order_id + route_link_id + route_link_seq
```

The completed matched route is a route-conditioned service-route proxy. It is
not evidence that the route itself is known before dispatch and is not a
route-choice model.

## Stage2-L production candidate

RC-MSTNet is frozen as the main candidate for:

```text
LCS expected raw stress and tail probability
PMIS expected raw stress and tail probability
RTS expected raw stress and tail probability
validation-normalized predictive uncertainty
```

The frozen architecture contains link semantic encoding, four-window lagged
state encoding, local route convolution, a route Transformer, shared
multi-task heads, and route-level auxiliary supervision. Structural ablations
may switch these components off but may not change the data or target contract.

## Stage2 baseline

The causal-route LightGBM model is retained as:

```text
strong benchmark
interpretability reference
optional shallow stacking input
```

It is not retained automatically in the final pipeline; stacking must improve
RC-MSTNet consistently across rolling folds.

## Stage2-I

IIS remains movement-level:

```text
applicability probability
severity prediction conditional on applicability/validity
tail probability conditional on applicability/validity
```

Missing IIS severity is structural missingness and must never be filled with
zero. IIS is not forced into the RC-MSTNet link heads.

## Frozen items

The following may not change in this round:

```text
fold definitions
prediction keys
Stage1 target definitions
estimated-entry-time contract
rolling-profile construction
strict backward lagged-state rule
test-day evaluation protocol
```

Test labels may be used only for final evaluation. Validation labels may fit
calibrators, conformal residual quantiles, and model selection. Stage3 inputs
must be rolling/OOF predictions and must exclude Stage1 realized labels,
post-trip realized features, and future profiles.
