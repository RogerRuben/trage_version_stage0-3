# Stage 3 S2A Operational Speed-Domain Report

The output is an operational speed-domain proxy. Inferred classes are **not verified posted speed limits**.

## Frozen validation choice

- Exact known-anchor directed identities: `502` in `68` physical OSM-way groups
- Anchor set SHA-256: `20f5aabab08541101dcf6c0db283c41224fdb6472ba3318d6f51f0d4066fcbce`
- Train-only history: `20161009–20161024`; Test31 used: `false`
- Selected quantile: `0.85`
- Selected method: `MAP_SPEED_AND_ROAD_CLASS`
- Compatibility accuracy at 60 / 80 / 120: `0.806773` / `1.000000` / `1.000000`
- Macro / within-10 / exact / MAE: `0.935591` / `0.852590` / `0.792829` / `7.928` km/h
- Candidate methods: B0 road-class mode, B1 `v_ff / 1.10` nearest class, B2 simple MAP of robust speed ratio × empirical class prior.
- Known-anchor speed-class distribution: `{"20.0": 43, "30.0": 9, "50.0": 27, "60.0": 38, "70.0": 63}`
- Train historical support: `6,716` full-network edges linked to `3,002,425` valid direct observations.
- Frozen P85/P90/P95 × B0/B1/B2 comparison: `[{"exact": 0.8007968127490039, "macro": 0.9349269588313414, "mae_kmh": 7.390438247011952, "method": "ROAD_CLASS_MODE_ONLY", "q": 0.85, "within10": 0.8605577689243028}, {"exact": 0.10358565737051793, "macro": 0.7848605577689244, "mae_kmh": 30.926294820717132, "method": "V_FF_DIV_1P10_NEAREST_CLASS", "q": 0.85, "within10": 0.21314741035856574}, {"exact": 0.7928286852589641, "macro": 0.9355909694555112, "mae_kmh": 7.9282868525896415, "method": "MAP_SPEED_AND_ROAD_CLASS", "q": 0.85, "within10": 0.852589641434263}, {"exact": 0.8007968127490039, "macro": 0.9349269588313414, "mae_kmh": 7.390438247011952, "method": "ROAD_CLASS_MODE_ONLY", "q": 0.9, "within10": 0.8605577689243028}, {"exact": 0.10358565737051793, "macro": 0.7881806108897743, "mae_kmh": 29.830677290836654, "method": "V_FF_DIV_1P10_NEAREST_CLASS", "q": 0.9, "within10": 0.23306772908366533}, {"exact": 0.7908366533864541, "macro": 0.9355909694555112, "mae_kmh": 7.95816733067729, "method": "MAP_SPEED_AND_ROAD_CLASS", "q": 0.9, "within10": 0.852589641434263}, {"exact": 0.8007968127490039, "macro": 0.9349269588313414, "mae_kmh": 7.390438247011952, "method": "ROAD_CLASS_MODE_ONLY", "q": 0.95, "within10": 0.8605577689243028}, {"exact": 0.11354581673306773, "macro": 0.7901726427622843, "mae_kmh": 28.09760956175299, "method": "V_FF_DIV_1P10_NEAREST_CLASS", "q": 0.95, "within10": 0.29282868525896416}, {"exact": 0.7868525896414342, "macro": 0.9349269588313414, "mae_kmh": 8.087649402390438, "method": "MAP_SPEED_AND_ROAD_CLASS", "q": 0.95, "within10": 0.850597609561753}]`

## Full-network coverage

- Provenance: `{"INFERRED_CLASS_DOMINANT": 1679, "INFERRED_SPEED_AND_CLASS": 4898, "KNOWN_STAGE0_OSM": 180, "ROAD_CLASS_PRIOR_ONLY": 202697}`
- Confidence: `{"HIGH": 180, "LOW": 204376, "MEDIUM": 4898}`
- Unobserved edges: `199,033`; non-UNKNOWN domain coverage: `100.000%`
- Observed-edge non-UNKNOWN coverage: `100.000%`
- Full-network road-class distribution: `{"motorway": 2884, "primary": 11755, "residential": 65404, "secondary": 19842, "service_other": 29907, "tertiary": 44774, "trunk": 5134, "unclassified": 29754}`
- C/M/A compatible shares among known-domain edges: `74.802%` / `99.999%` / `100.000%`
- C/M/A compatibility by confidence: `{"HIGH": {"A": 1.0, "C": 0.65, "M": 1.0}, "LOW": {"A": 1.0, "C": 0.7509981602536502, "M": 1.0}, "MEDIUM": {"A": 1.0, "C": 0.6271947733768886, "M": 0.9993875051041241}}`

## Limitations

- The 10% divisor is an empirical behavioral/regulatory inference anchor, not a statement of legal permission.
- Road-class priors are contextual empirical priors; OSM class is not Chinese statutory class.
- Sparse and unobserved edges receive low-confidence class priors, never fabricated historical speeds.
- No route feasibility, route-level F/U/I propagation, intersection complex, or profile envelope was constructed.
