# Monthly Stage0/Stage1 workflow

## Output contract

The monthly pipeline writes Hive-style daily partitions under `stage0_output/`:

```text
matched_points/day=YYYYMMDD/bucket=*.parquet
route_parts/day=YYYYMMDD/part=*.parquet
order_base/day=YYYYMMDD.parquet
quality_reports/day=YYYYMMDD.md
hmm_matched_points/day=YYYYMMDD/bucket=*.parquet
hmm_route_parts/day=YYYYMMDD/part=*.parquet
hmm_link_traversals/day=YYYYMMDD/part=*.parquet
hmm_turn_movements/day=YYYYMMDD/part=*.parquet
stage0_order_link_poi_behavior/day=YYYYMMDD/part=*.parquet
stage1_link_labels/day=YYYYMMDD/part=*.parquet
stage1_order_labels/day=YYYYMMDD.parquet
```

Every long-running phase writes manifests and logs. Existing complete partitions are reused.

## 1. POI cleaning and link exposure

The source CSV is explicitly read as UTF-8. Processing stops if replacement characters are detected. Raw coordinates are retained, while an empirical road-alignment diagnostic chooses the matching coordinate interpretation.

```powershell
python .\stage0\scripts\process_poi.py `
  --poi ".\map_data\西安市POI数据.csv" `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes .\map_data\xian_2017\xian_2017_core_nodes.parquet `
  --output-root .\stage0_output `
  --input-crs auto
```

## 2. Resume-safe monthly geometric Stage0

Only one daily nested archive is extracted at a time. The GPS member inside the daily `tar.gz` is streamed directly into Parquet order buckets; the uncompressed GPS file is never materialized.

```powershell
python .\stage0\scripts\run_monthly_stage0.py `
  --archive .\22-4-8.rar `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes .\map_data\xian_2017\xian_2017_core_nodes.parquet `
  --output-root .\stage0_output
```

Use `--start-date` and `--end-date` to run a subset. Completed days are skipped.

## 3. HMM/Viterbi and Stage1 preconditions

```powershell
python .\stage0\scripts\run_stage01_day.py `
  --date 20161001 `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes .\map_data\xian_2017\xian_2017_core_nodes.parquet `
  --poi-exposure .\stage0_output\stage0_link_poi_exposure.parquet `
  --output-root .\stage0_output `
  --workers 4
```

The geometric matcher remains available. HMM failures fall back to it and retain `fallback_used` and `fallback_reason`. Matcher comparison tables and reports are written per day.

HMM state transitions are expanded to network link paths. Links inserted between two observed HMM states are retained with `traversal_quality=inferred_path`. They support route topology, movement reconstruction, and GNS, but do not receive realized LCS/IIS/RTS/PMIS values because no GPS behavior was directly observed on those inserted links.

## 4. Monthly cohort labels

After the desired HMM traversal days exist, fit reference travel times and cohort histograms, then generate link and order labels:

```powershell
python .\stage0\scripts\build_stage1_labels.py `
  --traversal-root .\stage0_output\hmm_link_traversals `
  --movement-root .\stage0_output\hmm_turn_movements `
  --poi-exposure .\stage0_output\stage0_link_poi_exposure.parquet `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --order-base-root .\stage0_output\order_base `
  --output-root .\stage0_output `
  --fit-dates all `
  --target-dates all `
  --min-cohort-size 100
```

The empirical CDF is estimated with bounded histograms and the requested six-level fallback hierarchy. Reference travel time uses weighted part-level medians and the same hierarchy, keeping the fit streaming and reproducible.

## 5. Validity audit

```powershell
python .\stage0\scripts\audit_stage1_labels.py `
  --output-root .\stage0_output `
  --date 20161001 `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --matched-dir .\stage0_output\hmm_matched_points\day=20161001 `
  --poi-exposure .\stage0_output\stage0_link_poi_exposure.parquet
```

## Storage note

The geometric matched points for 2016-10-01 occupy about 1.43 GB. HMM point output is wider. Keeping both matcher versions plus traversal, primitive, and label partitions for all 31 days requires substantially more than 66 GB of free space; provision a larger output volume before the full retained monthly run.

Once a sufficiently large output volume is available, the complete daily loop plus monthly label fit can be launched with `run_monthly_stage01.py`. It performs a free-space preflight and refuses a multi-day retained run below the configured safety threshold.
