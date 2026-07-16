# Stage 3.5 offline route-generation contract

## Research task

Stage 3.5 generates and freezes vehicle-type-specific route alternatives before
Stage 4 starts. It converts an order OD, a frozen network, and Stage 2/3 route
predictions into an HV reference route and an AV risk-adapted route. It is an
offline route-selection product, not an online dispatch heuristic.

## Inputs

- A canonical Stage 0 OD and frozen directed network manifest.
- Stage 2 dispatch-time link predictions for every candidate path.
- Stage 3 calibrated route-risk and uncertainty outputs.
- Pre-registered ODD scenario parameters.
- A versioned candidate-generation and detour-constraint configuration.

Historical/revealed routes may be read only for the explicitly named
`historical_route_replay_baseline`; they are not represented as routes known at
dispatch time.

## Outputs

- Candidate-route table with route/link sequence and generation method.
- One frozen HV reference route per eligible order.
- Zero or one AV risk-adapted route per order and capability scenario.
- Route time, distance, risk, uncertainty, detour, ODD violations, binding
  dimension, and remote-assistance requirement.
- Failure rows for orders with no valid path or no ODD-feasible AV route.
- A manifest linking every output to the Stage 0, Stage 2, Stage 3, network,
  candidate-generation, and ODD scenario versions.

## Allowed information

- OD and request/decision time available before assignment.
- Static network attributes and dispatch-time Stage 2/3 predictions.
- ODD profiles frozen using external assumptions or train/validation evidence.
- Historical routes only in the separately labelled replay/oracle product.

## Forbidden information

- Test outcomes used to tune route weights, detour caps, or ODD thresholds.
- Actual future link entry state or completed service duration.
- Online route re-planning inside the Stage 4 comparison.
- Silently substituting an HV route when the AV candidate set is infeasible.
- Selecting inputs by directory discovery or file modification time.

## Route-selection rule

For candidates `r` of order `o`, AV selection minimizes

`E[T_r] + lambda_risk * Risk_r + lambda_uncertainty * Uncertainty_r`

subject to a hard route ODD gate and pre-registered distance/time detour caps.
If no candidate satisfies every hard constraint, `av_route_available=false` and
the minimum-achievable risk and binding dimension remain explicit.

## Acceptance rules

- Candidate paths are directed, connected, unique, and share the declared OD.
- All candidate risk predictions use a single order-level dispatch cutoff.
- HV and AV route semantics are distinct and their source fields are explicit.
- Test-set tuning count is zero.
- Route metrics recompute from the stored link sequence within tolerance.
- Every Stage 4 request resolves through one explicit Stage 3.5 manifest row.

## Version and downstream consumer

Contract version: `stage35_offline_routes_contract_v1`. Stage 4 is the sole
downstream consumer. Once frozen, Stage 4 algorithm changes cannot mutate or
regenerate Stage 3.5 routes.

