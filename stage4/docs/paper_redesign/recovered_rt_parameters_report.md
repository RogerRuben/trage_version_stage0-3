# Recovery of legacy RT parameters

**RECOVERED_FINGERPRINT_VERIFIED.** This is a reconstruction, not recovery of the missing original manifest.

The unchanged generator was called on retained all-order prescan manifests for 20161019–22: 100348 / 104047 / 115187 / 120467 orders. No Stage1 quality filtering was applied. All raw points passed the prescan validity checks. Endpoints were sorted by order ID and converted using the original GCJ-02-to-WGS84 function. No archive extraction, map matching, training or simulation was needed.

| Recomputed chain statistic | Value |
| --- | ---: |
| gap P25 (s) | 420 |
| gap P50 (s) | 748 |
| gap P75 (s) | 1733 |
| gap P90 (s) | 3693 |
| empty-speed P50 (m/s) | 1.0836658495076241 |
| chain rows | 440049 |
| feasible chain rows | 302772 |

The original 60–7200-second feasible-gap filter and quantile functions are unchanged. RT caps are 420 / 748 / 1733 seconds, not realized mean leads.

## Surviving fingerprint

All 114356 retained 20161023 orders were transformed. Reference: stage4/docs/results/request_time_reconstruction_summary.csv.

| Scenario | Mean (s) | P50 (s) | P90 (s) | Clipped share |
| --- | ---: | ---: | ---: | ---: |
| RT-Low | 318.50330106916647 | 289.5808632833151 | 506.1923605348263 | 0.2012574766518591 |
| RT-Base | 501.9096013928667 | 497.97856176370203 | 626.9027962306016 | 0 |
| RT-High | 1308.7568309649687 | 1364.176402753013 | 1486.3696770491251 | 0 |

All 15 count/mean/P50/P90/clipped checks pass. Tolerances were fixed before comparison: exact count, 1e-6 seconds, 1e-12 share, zero relative tolerance. Maximum error: 2.274e-13 seconds. No parameter fitting against the fingerprint. Runtime: 7.89 seconds.

Frozen reconstruction: [recovered_rt_environment_parameters.json](recovered_rt_environment_parameters.json). Comparison: [recovered_rt_fingerprint_comparison.csv](recovered_rt_fingerprint_comparison.csv).

**reconstructed from the original generator and verified against surviving frozen RT summary outputs.**

## Claim boundaries

Missing original OD files prevent independent per-order hash equivalence; raw endpoint tie selection is not independently recoverable from prescans. The surviving RT summaries do not independently validate empty-speed P50 or chain counts. Those are recomputed statistics, not certified archived values. The full aggregate RT fingerprint does reproduce to floating-point precision.

The original UTC business-day boundary is retained, including for Test31. These are latent scenarios, not observed passenger request times. Only RT construction is recovered; unrelated legacy fleet/depot defaults are not certified as realized settings.

Run from repository root: python -m stage4.analysis.recover_rt_parameters. Source and generator SHA256 values are recorded in JSON; no large intermediate products are retained.
