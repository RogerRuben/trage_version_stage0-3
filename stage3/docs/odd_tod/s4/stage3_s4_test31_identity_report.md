# Stage 3 S4 Test31 Identity Report

- Orders: 30,000
- Route tokens: 2,116,712
- Full-network tokens: 2,030,182 (95.912056%)
- Historical reverse overlays: 85,538 (4.041079%)
- Unresolved tokens: 992 (0.046865%)
- Fully full-network-resolved orders: 14,617 (48.723333%)
- Orders with reverse overlay: 14,837
- Orders with unresolved identity: 981

## Descriptive temporal comparison

| Split | FULL_NETWORK_EDGE | Reverse overlay | Unresolved | Fully resolved orders |
|---|---:|---:|---:|---:|
| Train 09-24 | 95.814218% | 4.137044% | 0.048738% | 48.180000% |
| Validation 25-27 | 95.852556% | 4.102496% | 0.044948% | 48.290000% |
| Test31 | 95.912056% | 4.041079% | 0.046865% | 48.723333% |

This comparison is descriptive only and triggers no calibration action.

Reverse overlays are known AV-routability violations and are never projected onto the physical forward edge. Unresolved identity is unknown evidence and is never geometry-imputed.
