# Stage3 Final Methodology

Stage3 finalization preserves the frozen M3 checkpoint, Train weighted mid-CDF, C/M/A profiles, A/M/D/L definitions, movement rules, restrictions, roundabout semantics, speed caps, and all 36 dynamic caps.

Only an original-route structural `INFEASIBLE` state triggers fallback. A single deterministic Valhalla `auto` route is requested from the frozen raw-GPS OD at the original decision time. The routed shape is passed back to the same Valhalla engine with `edge_walk`; every returned directed edge ID must exist in the frozen Stage3 full-network table. No reverse overlay, synthetic reverse, geometry-nearest repair, or forward substitution is permitted. Candidates beyond the fixed 1.25 distance ratio are rejected.

Candidate movement evidence reuses the production intersection-complex parser and frozen movement/control/restriction rules. Static, dynamic, and speed envelope exceedance never changes hard feasibility. Because a new route lacks the exact frozen M3 historical feature rows, fallback dynamic evidence and service time remain null; no imputation or historical-route prediction copying is performed.

Candidate selection is frozen as minimum M3 P50 followed by distance. The baseline produces one candidate, so a hard-feasible candidate with unavailable soft dynamic evidence may still be selected, with `evidence_complete=false`.
