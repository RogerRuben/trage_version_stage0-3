# Neutral AV identity: availability policy isolated

**PASS_IDENTITY — 10/10 frozen states.** Candidate identities, selected assignment identities, maximum matching, selected counts, and pickup objectives agree in every pair. All ten source-state hashes match the prior frozen registry.

## Local change and canonical behavior

VehicleFixture resolves availability_policy once at creation: canonical HV -> EMPIRICAL_SESSION, canonical AV -> FULL_HORIZON. The candidate admission branch reads that policy. Relabeling preserves the inherited policy, session endpoint, IDs and positions. Canonical defaults are unchanged; the frozen 41 scenarios were not rerun or overwritten.

The diagnostic relabels original HV vehicles as AV while retaining EMPIRICAL_SESSION; already-AV vehicles retain FULL_HORIZON. Both variants disable explicit AV restrictions and share request state, service duration, routing, Top-K, optimizer and tie inputs. It calls the actual production time_trigger and captures solver inputs before progression or assignment execution. This proves the requested one-epoch graph/selection invariant, not every simulator lifecycle branch.

| State | E in each variant | U | Maximum M | Selected in each variant |
|---|---:|---:|---:|---:|
| 07:30 | 109 | 25 | 24 | 24 |
| 08:30 | 90 | 22 | 21 | 21 |
| 12:00 | 97 | 30 | 24 | 24 |
| 13:00 | 71 | 23 | 21 | 21 |
| 17:00 | 88 | 15 | 14 | 14 |
| 17:30 | 117 | 22 | 20 | 20 |
| 18:00 | 59 | 15 | 14 | 14 |
| 18:30 | 45 | 10 | 10 | 10 |
| 21:00 | 120 | 25 | 19 | 19 |
| 23:00 | 23 | 7 | 6 | 6 |

Exact identities and pickup objectives are in mechanism_validity/neutral_av_identity.csv. Neutral counts need not equal the original restricted P/H experiment.

## Calibration of the earlier finding

Commit e467a3c captured a label-plus-policy change: 109 -> 110 edges at 07:30, with U=25, M=24, selected identities and pickup objective unchanged. HV_S3_00361 had 199 seconds left in its session while order 855699a09cd4dd73d3829ba3829e1c54 required 730.384937 seconds of predicted service. Removing the session check admitted one redundant edge.

That was **vehicle-type / availability-policy coupling**, in the AV-favoring direction. It did not demonstrate canonical anti-AV filtering or invalidate the 41 scenarios. The earlier IMPLEMENTATION_DEFECT label is superseded by this narrower interpretation and the passing controlled test.

The focused test checks inherited EMPIRICAL_SESSION under relabeling and unchanged FULL_HORIZON admission. This neutral identity issue is closed for the requested ten-state candidate/assignment comparison.
