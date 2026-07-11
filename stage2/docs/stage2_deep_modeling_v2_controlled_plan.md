# Stage2 Deep Modeling v2 controlled experiment plan

## Correct interpretation of the current v2 probe

The current RouteLocalTransformer and DualGraphRouteTransformer runs are small structural probes:

```text
train budget: about 30k orders
eval budget:  about 15k orders
epochs:       2 supervised epochs
hidden dim:   96
```

They must not be treated as a final head-to-head comparison against `full_tabular_lgbm_3m_tail`, which uses a much larger training budget and a richer engineered tabular feature set.

The defensible conclusion is:

> Under the current small-probe setting, the v2 deep models have not yet shown a stable overall advantage. This does not prove that route-aware or graph-aware deep structure is weaker than the strong tabular baseline.

## Why the comparison is currently not final

The current tables mix at least three differences:

1. data budget: 30k-order deep probe vs large / 3M-row tabular training;
2. input strength: route/link sequence features vs strong engineered historical and route-context tabular features;
3. optimization budget: two-epoch probe vs tuned tail-weighted LightGBM.

So the next question is not “does deep beat tabular?” but:

> Does route-local context, dual-graph propagation, or intersection gating add incremental information beyond strong tabular features?

## Controlled experiment ladder

### A. Fair small-sample comparison

Train all methods with the same order budget, starting with 30k train orders:

```text
full_tabular_lgbm_30k
BiGRU_30k
RouteLocalTransformer_30k
DualGraphRouteTransformer_30k
plain_GNN_30k
```

This answers the budget-matched question:

> Given the same training order budget, which architecture learns the strongest signal?

### B. Scaling curve

Repeat at increasing budgets:

```text
30k
100k
300k
1M
3M or all feasible rows/orders
```

Interpretation:

- tabular always leads: engineered features already capture most available pre-dispatch signal;
- deep catches up with scale: route/graph models are data-hungry and the 30k probe was underpowered;
- hybrid wins: deep route/graph representations contain useful incremental structure but are best used with tabular calibration.

### C. Hybrid fusion

The preferred v2 thesis path is likely:

```text
route/graph model -> embedding or stress score
strong tabular features + deep representation -> LGBM / MLP / calibrated model
```

Hybrid fusion should be evaluated as incremental value over the full tabular model, not as a replacement by default.

### D. RTS-specific branch

RTS appears more route-propagation-like than LCS/IIS/PMIS. It should have a separate branch:

```text
RTS full tabular
RTS route-local
RTS dual-graph
RTS peak-aware
RTS hybrid fusion
```

RTS must be reported separately because it is also the most drift-sensitive dimension.

### E. Temporal-shift verification

Temporal shift is plausible but not proven. To test it, compare:

```text
random split AUC
time split AUC
rolling split AUC
day-wise AUC drift
calibration drift
train-day expansion curve
peak/off-peak stability
rare-link stability
```

### F. Slice confidence intervals

Potential slice gains, such as DualGraph on peak RTS, require bootstrap confidence intervals:

```text
peak/off-peak
seen/rare links
high endpoint_degree links
IIS-valid subset
short/long routes
high-stress tail subset
order-level tail separation
```

## Stage3 implication

Do not use current v2 deep predictions as the Stage3 calibrated ODD-stress vector input yet.

Stage3 should wait for either:

1. a controlled deep model that beats the strong tabular baseline under comparable budget and input conditions; or
2. a hybrid model that demonstrates stable incremental value over full tabular predictions with rolling / OOF validation.
