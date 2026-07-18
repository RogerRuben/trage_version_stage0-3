# Stage 0 v5 Gate 0/1 execution record

> Legacy pre-P0 result. This run predates the structural movement-routing,
> level-transition, selective-HMM, and compact-retention fixes documented in
> `stage0_v5_p0_structural_fix_report.md`. It remains a baseline audit record and is not Gate 2
> readiness evidence. Corrected Gate 1 has not yet been run.

## Scope and provenance

- Branch: `codex/stage0-v5`
- Base commit: `fd1e7c0b1e27871c25c759c9b077fdb9e85ea56a`
- Gate 1 dates: `20161010`, `20161014`, `20161016`
- Sampling: 2,000 complete orders/day, seed `20261009`, 128 stable buckets
- Sampling run: `orders=2000__dates=b76d57164111`
- Execution: four mutually exclusive bucket shards/date, one worker/shard

The raw RAR, PBF and POI were read only. No Test date was materialized or matched.

## Gate 0

Gate 0 is `PASS`.

- Canonical directed edges: 623,839
- Canonical nodes: 335,315
- Legal movement rows: 1,389,094
- Audited parallel pairs: 18,200
- OSM restrictions retained: 20; parsed: 13 (65%)
- PBF snapshot: `2026-07-15T23:00:00Z`
- Edges flagged for 2016/2026 temporal mismatch: 595,313
- POI source/valid rows: 399,566 / 399,566
- POI assigned/unassigned rows: 365,168 / 34,398
- Network build runtime: 321.87 s
- POI runtime: 14.33 s
- Free disk after Gate 1: 108.78 GiB

The canonical 23-day, 10,000-order/day sampling manifest is namespaced as
`orders=10000__dates=590a5f0f6387`; Gate 1 cannot overwrite it.

## Gate 1 measured result

Gate 1 was executed over all 6,000 sampled orders. Execution accounting passed, but research
readiness is `FAIL`.

| Check | Measured result |
|---|---:|
| Input/output orders | 6,000 / 6,000 |
| Time-conservation failures | 0 |
| Distance-conservation failures | 0 |
| Strict core | 1,902 (31.70%) |
| Analysis set | 382 (6.37%) |
| Rejected | 3,716 (61.93%) |
| Fast deterministic | 0.37% |
| Local HMM | 38.63% |
| Full-order HMM | 33.28% |
| Geometric fallback | 27.38% |
| Explicit matcher rejection | 0.33% |
| Orders with parallel ambiguity | 94 |
| Topology/direction gaps | 11,014 |
| Mean inferred-distance share | 25.18% |

Per-date quality counts were:

- `20161010`: strict 644, analysis 111, rejected 1,245;
- `20161014`: strict 623, analysis 135, rejected 1,242;
- `20161016`: strict 635, analysis 136, rejected 1,229.

Four-shard wall time was approximately 34.8 minutes for `20161014` and 33.8 minutes for
`20161016`; `20161010` was of the same order. Per-shard peak RSS was approximately 2.0–2.4
GiB. The largest observed four-process aggregate working set was approximately 8.15 GiB,
below the 12 GiB gate. Formal outputs occupy 0.347 GiB and work/cache data 4.421 GiB.

## Readiness decision

The following measured checks failed:

- geometric fallback must be at most 25%; measured 27.38%;
- rejected share must be at most 50%; measured 61.93%.

Therefore `gate2_allowed=false`. Gate 2, Train, Validation, Freeze and Test commands are
provided for recovery, but must not be run until a methodologically justified fix is applied
and this exact Gate 1 is rerun. Threshold relaxation alone is not an acceptable fix.

Evidence files are `stage0/output_v5/reports/gate0_report.json`,
`stage0/output_v5/reports/gate1_readiness.json`, and
`stage0/output_v5/reports/gate1_readiness_report.md`.
