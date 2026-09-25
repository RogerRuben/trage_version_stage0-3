# Lead × patience execution status

Update: user-authorized retry1 completed 48/48 cells with 16/16 baseline reproduction. See `lead_patience_report.md`, `lead_patience_cells.csv`, and `lead_patience_summary.json`. The first-attempt record below is retained as history, not current completion status.

## Material Passport

- Origin: academic-research-suite / experiment-agent
- Date: 2026-09-25
- Status: FIRST_ATTEMPT_FAILED; FIX_TESTED; RETRY_NOT_RUN
- Scope: 4 fixed states × 4 lead variants × 180/300/600 seconds, no full-day simulation.

The first attempt ran `python -m stage4.analysis.lead_patience_factorial`. Two 300-second cells were written before the mixed-graph capture failed in q50/noon/RT-Base. They are partial diagnostics, not completed experiment results. The 180/600-second phase never started. The full sixteen-cell baseline pass is NOT established.

Cause: production `patience_expired` casts deadlines to integer seconds; the legacy fixed-state waiting filter retains microseconds. A subsecond residual can be waiting in the legacy analysis but expired in production. The analysis capture fixture lacked `expired_rids`, `cancelled_rids`, `rq_dict`, and `demand.rq_db` and raised AttributeError while executing the legitimate expiry branch. No production policy defect is inferred.

Fix: initialize capture-only expiry state. For new mixed-graph metrics, explicitly use the production integer deadline convention. Preserve microsecond timing for the legacy AV E/U/M baseline. Count subsecond expiries and AV arcs lost to the integer boundary, and retain separate production AV capacity. Verify production AV arcs are a subset of the legacy continuous-time AV arcs. This is not a new timing policy or threshold change.

Four focused tests pass, including deadline boundary/critical behavior, nonadditive mixed capacity, unchanged default300 behavior and integer-vs-continuous boundary semantics. Compilation and diff check passed before the first execution; targeted tests and diff check passed after the fix. No experiment was automatically retried.

Partial data remains at `stage4/output/paper_enhancement/lead_patience_factorial/cells.csv`. New attempts refuse to overwrite prior cells. Proposed retry after user confirmation:

```powershell
D:/anaconda/envs/stage0-valhalla/python.exe -u -m stage4.analysis.lead_patience_factorial --attempt retry1
```

Retry outputs would go to `stage4/output/paper_enhancement/lead_patience_factorial/retry1/`. First reproduce all16 original300-second AV results, then run remaining32 cells. Single process, per-state sparse routing cache; no GPU, no full-day fleet progression, no model inference, no cumulative-Gamma optimization. Retained180-second cohorts and additional cohorts are reported separately; their capacities cannot be summed when vehicles overlap. The900-second budget is checked between cells; it is not a guarantee against one native routing call blocking.

Standing automatic commit/push authorization applies to this code and status record; it does not label the unfinished experiment complete.
