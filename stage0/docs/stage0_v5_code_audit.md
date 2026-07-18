# Stage 0 v5 code audit

## Baseline inspected

The v5 branch starts from `origin/codex/pipeline-rebaseline` at
`fd1e7c0b1e27871c25c759c9b077fdb9e85ea56a`. The audit covered:

- `stage0/canonical/` and its route-quality contracts;
- `hmm_viterbi_matcher.py`, `reconstruct_hmm_routes.py`, full-day/monthly/split runners;
- compact-retention, interval-conservation, topology, manual-review, and promotion tests;
- Stage 0 documentation, split configuration, manifests, and quality rules.

`reconstruct_route_links.py` named in the task was not present. Its effective predecessor is
`reconstruct_hmm_routes.py`.

## Retained decisions

V5 retains complete-order sampling, stable order hashing, Parquet streaming, scale-aware
bucket counts (32 for Gate 1, 64 for 10,000-order days), completed-partition skipping,
manifests, link traversals, turn movements, compact
retention, observed/inferred separation, and strict-core/analysis-set/rejected governance.

## Replaced decisions

The previous HMM ran over every point and reconstructed gaps through a node shortest path
followed by the minimum parallel edge. V5 uses fast concrete-edge candidates, selective local
HMM, bounded full-order fallback, and movement-graph reconstruction that returns concrete
`edge_uid` values. A metric graph may collapse node-pair costs, but is never used to recover
edge identity.

## Baseline test record

The machine-wide Anaconda environment could collect only 36 tests and raised 12 collection
errors due to a NumPy/Shapely ABI mismatch and missing project dependencies. Tests were not
deleted. A repository-local ignored environment at `stage0/work_v5/.venv` was created from the
pinned `stage0/requirements.txt`; the combined legacy and v5 suite then passed.

## Data safety

The RAR, PBF, and POI sources are opened read-only. All extracted daily archives, candidate
index arrays, and sampled point fragments remain under `stage0/work_v5/`; all formal products
remain under `stage0/output_v5/`.
