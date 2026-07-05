# DiDi Xi'an Trajectory Stage0

A reproducible Stage0 pipeline for validating whether large-scale ride-hailing GPS trajectories can support:

> GPS cleaning → order reconstruction → map matching → road-semantic fusion → primitive behavior features

The project was validated on one full day of Xi'an trajectories (28.3 million GPS points, 119,018 orders) against a temporally aligned 2017 Geofabrik/OpenStreetMap road snapshot.

## Validation result

| Metric | Full-day result |
|---|---:|
| Input GPS points | 28,334,617 |
| Orders / drivers | 119,018 / 17,855 |
| High-quality orders | 96.70% |
| Matching-success orders | 98.99% |
| High-confidence orders | 94.79% |
| Median order-level P90 GPS-link distance | 8.67 m |
| P90 of order-level P90 GPS-link distance | 15.60 m |
| Route-length ratio in 0.8–1.3 | 99.40% |

Overall verdict: **CONDITIONAL GO** for Stage1 label construction.

## Important coordinate finding

The source schema describes the coordinates as WGS84. Empirically, however, the points align with the 2017 WGS84 OSM roads only after a GCJ-02 → WGS84 conversion:

| Interpretation | P90 GPS-link distance |
|---|---:|
| Source values used directly as WGS84 | 120.17 m |
| GCJ-02 converted to WGS84 | 10.78 m |

The pipeline therefore preserves `source_lon/source_lat` and writes converted matching coordinates separately as `lon/lat`. This inference should be confirmed with the original data provider before publication.

## Repository layout

```text
scripts/
  extract_xian_2017_network.py     # Extract Xi'an roads from Geofabrik China shapes
  prepare_sample.py                # Stream and sample complete orders
  run_stage0.py                    # Small-sample Stage0 experiment
  run_full_day_2017.py             # 128-bucket scalable full-day pipeline
  recalculate_full_day_topology.py # Geometry-aware topology audit correction
  generate_full_day_cases.py       # Representative visual cases
docs/
  methodology.md
  results/                         # Aggregate, non-identifying results only
```

Raw trajectories, road data, matched point partitions, and order-level outputs are intentionally excluded from Git.

## Data expectations

The headerless trajectory input is interpreted as:

```text
driver_id, order_id, timestamp, lon, lat
```

- `timestamp`: Unix epoch seconds
- source coordinates: retained unchanged
- matching coordinates: converted to WGS84 when `--input-crs gcj02` is selected

The Geofabrik free road shape uses EPSG:4326 and provides `osm_id`, `fclass`, `name`, `ref`, `oneway`, `maxspeed`, `layer`, `bridge`, and `tunnel`. It does not provide lane counts, and speed-limit coverage is sparse.

## Quick start

```powershell
python -m pip install -r requirements.txt

python .\scripts\extract_xian_2017_network.py `
  --source-dir .\map_data\china-170101-free.shp `
  --output-dir .\map_data\xian_2017

python .\scripts\run_full_day_2017.py `
  --input .\10.1\gps_20161001 `
  --roads .\map_data\xian_2017\xian_2017_core_roads.parquet `
  --nodes .\map_data\xian_2017\xian_2017_core_nodes.parquet `
  --output-dir .\full_day_output `
  --input-crs gcj02
```

The full-day pipeline hashes complete orders into 128 Parquet buckets, keeping memory bounded and allowing completed buckets to be reused.

## Method boundary

The matcher uses a 10 m densified-road candidate index, exact point-to-LineString projection, and geometry-aware topology diagnostics. It is designed for feasibility validation, not as a replacement for a production HMM/Viterbi map matcher.

See [docs/methodology.md](docs/methodology.md) for the full workflow and limitations.

