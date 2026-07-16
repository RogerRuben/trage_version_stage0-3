# Stage 0 v4 promotion audit

## Decision

**HOLD — not canonical.** Stage 1--4 remain blocked.

| Gate | Status | Evidence |
|---|---|---|
| Network topology diagnostic | Pass | `xian_2017_core_noded_v4/network_audit.json` |
| Grade separation | Diagnostic pass | 4,689 incompatible intersections left un-noded |
| Fixed 1,000-order comparison | Stop for review | `stage0_network_comparison_v4_audit.json` |
| Expanded route-quality schema | Implemented | `stage0_route_quality_v4.json` and train/validation summaries |
| Manual truth | **HOLD** | 0/500 reviews completed; 100-order second-review sheet pending |
| Diagnostic conservation | Pass | 0 time failures, 0 distance failures on 19/20/22 samples |
| Full-date chain | **Not authorized** | Must follow the manual-truth gate |
| Canonical manifest | **Not generated** | Canonical registration guard requires all audits to pass |

## Why processing stops here

The v4 graph preserves failure rate and route length relative to v3, but its
strict route-quality Core share falls from 31.1% to 16.6% in the fixed diagnostic
and graph-level directed OD reachability is 99.90%. The pre-registered task says
such behavior must be located and manually reviewed before full-date execution.
Running the complete date chain now would violate that sequence and would not
convert missing human judgments into evidence.

## Reproduction

The compact comparison, review pack, review audit, conservation audit and
promotion audit are committed. Large network/rematch payloads remain local under
`artifacts/exploratory/` and are excluded from Git. Once independent reviews pass,
rerun the promotion audit; only then start the fit/train/validation/development
date chain and its full conservation and lineage audits.
