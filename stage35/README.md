# Stage 3.5 offline route products

Stage 3.5 converts a frozen Stage 3 route-risk model into fixed, versioned route
choices for Stage 4. It does not run inside the dispatch simulator.

Each order receives an HV replay-baseline route, an HV planned-route robustness
alternative, and an AV route selected from pre-generated candidates under the
registered ODD and detour constraints. If no candidate is feasible, the product
must retain the order with `av_route_available=false` and the binding reason.

Formal Stage 4 runs may read only one explicit Stage 3.5 manifest. Directory
globbing and online replacement of the service route are prohibited.

Current status: definition frozen; implementation and canonical production are
blocked by formal Stage 0--3 artifacts.
