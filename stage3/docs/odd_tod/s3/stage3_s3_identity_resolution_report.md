# Stage 3 S3 Identity Resolution Report

Status: `STAGE3_S3_CAPABILITY_ENVELOPE_FROZEN`. Train covers `20161009` through `20161024`; Test31 was not read.

- Orders: 160,000
- Route tokens: 11,432,534
- `FULL_NETWORK_EDGE`: 10,953,993 (95.814218%)
- `HISTORICAL_REVERSE_OVERLAY`: 472,969 (4.137044%)
- `UNRESOLVED`: 5,572 (0.048738%)
- Fully full-network-resolved orders: 77,088
- Orders with reverse overlay: 79,979
- Orders with unresolved token: 5,536

The resolver is typed. A historical reverse traversal remains `HISTORICAL_REVERSE_OVERLAY` with `AV_ROUTABILITY_VIOLATION`; its forward physical reference is provenance only and is never substituted as the traversed edge. Unresolved and reverse tokens break complex-parser continuity. No nearest-edge repair is used.
