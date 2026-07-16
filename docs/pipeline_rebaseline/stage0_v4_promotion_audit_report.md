# Stage 0 v4 promotion audit

## Decision

**HOLD — not canonical.** Stage 1 computation remains blocked; only its startup
configuration is prepared.

| Gate | Status | Evidence |
|---|---|---|
| Direction-aware connector implementation | Pass | 652 bidirectional and 1,615 unidirectional connectors |
| Fixed 1,000-order directional rematch | Diagnostic pass | 984/1,000 reconstructed; no geometric fallback |
| Connector usage and disabled ablation | Diagnostic pass | `stage0_v4_connector_audit_report.md` |
| Connector human review | **HOLD** | 0/50 connector judgments completed |
| Core failure decomposition | Complete | 3,938 reconstructed diagnostic orders |
| Route human review | **HOLD** | 0/150 primary reviews and 0/40 double reviews completed |
| Diagnostic conservation | Pass from the prior v4 milestone | zero registered time/distance failures |
| Final quality rule | Candidate only | `stage0/config/route_quality_v4_final_candidate.json` |
| Full-date chain | **Not authorized** | starts only after both human gates pass |
| Canonical manifest | **Not generated** | requires manual and full-date audits |

## Why processing stops before the full date chain

The previous all-bidirectional connector implementation was a systematic direction
error. It has been corrected without creating a new road-network version. The
corrected fixed sample retains 98.4% reconstruction coverage, but Strict Core falls
to 13.9%, directed reachability remains 99.80%, and automated U-turn/detour flags
dominate exclusions. These facts require the targeted independent review specified
by the task; they do not authorize either threshold tuning or canonical promotion.

The v2 review package now implements the revised, finite workload: 150 primary
routes (78 Strict Core and 72 boundary/rejected), 40 double-review routes, and 50
used connectors. Its acceptance rules are executable and have no Core-share gate.

## Next authorized action

Independent reviewers fill the registered CSVs without changing the matcher. Once
at least 120 primary reviews, 30 paired reviews, and the error/agreement gates pass,
the final quality configuration may be frozen exactly once. Only then may the full
date chain run and `artifacts/canonical/stage0_v4/stage0_v4.manifest.json` be created.
