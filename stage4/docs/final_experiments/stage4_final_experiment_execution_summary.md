# Stage4 Final Experiment Execution

Recommendation: `GO_STAGE4_RESULT_ANALYSIS`

## Execution base

- Execution commit: `6b6d3410fadc80a25d60a4affc9a439c71c3f174`.
- S5B registry commit: `b82cd3fda60a39c7474de49bdf4f205850a6725d`.
- FleetPy commit: `0379f9725a147ff33c674de4884cdf89fd787fa9`.
- Horizon: `2016-10-31T00:00:00+08:00` to `2016-11-01T00:01:00+08:00` (right-open), 30,000 requests per profile.

## Batch completion

- Unique completed/expected: 41/41.
- Reuse rows resolved: 1.
- MAIN/BENCHMARK/ODD/COST resolved: 27/4/3/8.
- Failures: 0.

## Resource usage

- Total wall-clock: 16.647 h; sum scenario runtime: 24.607 h.
- Peak observed RSS: 1674.7 MB; output size: 0.271 GiB.
- Production parallelism: 2; sparse CSR/Top-K/cKDTree, CPU-only.

## Main structural results

- Service-rate range: 0.3544-0.7309.
- P95 request-to-pickup range: 289.3-293.0 s.
- AV assignment-share range: 0.0310-0.3082.

## Benchmarks

- BENCH_HV: service=0.7889, P95=287.6s, AV share=0.0000.
- BENCH_AV_C: service=0.1092, P95=289.1s, AV share=1.0000.
- BENCH_AV_M: service=0.1515, P95=291.9s, AV share=1.0000.
- BENCH_AV_A: service=0.1600, P95=290.9s, AV share=1.0000.

## ODD policy results

- REFERENCE: service=0.6038, P95=291.6s, AV share=0.1244.
- STRICT: service=0.5532, P95=292.2s, AV share=0.0113.
- UNCONSTRAINED: service=0.6044, P95=291.8s, AV share=0.1217.

## Cost robustness

- Pickup-ETA degradation range versus eta-specific epsilon=0: -0.004284-0.014808.
- Normalized operating cost per matched order range: 597.318-656.014.

## Family activity

- C: static=0.9248, dynamic=0.7739, speed=0.7073.
- M: static=0.8698, dynamic=0.4764, speed=0.0006.
- A: static=0.6394, dynamic=0.1822, speed=0.0000.

## Failures / limitations

- Failed or missing unique scenarios: none.
- Results are frozen scenario contrasts, not calibrated safety probabilities, monetary estimates, or post-hoc tuned policies.
- Speed remains in every scenario even when empirically inactive.
