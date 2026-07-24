# Stage 0 v6 Valhalla environment and tile build

## Measured environment

- Operating system: Windows 11 x86-64.
- Environment: isolated Conda environment `stage0-valhalla`.
- Python: 3.12.13.
- Installation: `python -m pip install pyvalhalla`.
- `pyvalhalla` / Valhalla: 3.8.2.
- Native import check: `from valhalla import Actor, get_config` passed.
- Docker was not installed and WSL had no Linux distribution, but neither fallback
  was needed because the official Windows wheel included `Actor`,
  `valhalla_build_tiles`, and `valhalla_service`.

## Measured tile build

- PBF: `D:\pycodes\didi_xian_raw\map_data\xian_roadmap_update.osm.pbf`.
- PBF size: 21,753,051 bytes.
- PBF SHA-256:
  `4D918B7ED2201A8F7A75FA7FB2974679343C89A10BF83638E7BCF21AC07C1526`.
- Tile directory: `D:\pycodes\didi_xian_raw\valhalla_data\tiles`.
- The build retained all OSM node IDs so `trace_attributes` can emit directed
  start/end node IDs needed by the canonical-edge mapper.
- The one-time build completed successfully. The build found 52,789 routable
  ways, 104,909 graph nodes, and 301,238 directed edges.
- Missing admin, timezone, traffic, and elevation databases generated warnings.
  These optional datasets are not required for the current auto map-matching
  feasibility test.

The machine-readable evidence is
`D:\pycodes\didi_xian_raw\valhalla_data\build_manifest.json`. Tiles, logs,
runtime config, Parquet outputs, and raw response samples are ignored by Git.

## Reproduction

```powershell
conda create -n stage0-valhalla python=3.12 -y
conda run -n stage0-valhalla python -m pip install -r stage0/v6/requirements.txt
conda run -n stage0-valhalla python -m stage0.v6.build_tiles `
  --pbf D:\pycodes\didi_xian_raw\map_data\xian_roadmap_update.osm.pbf `
  --config D:\pycodes\didi_xian_raw\valhalla_data\valhalla.json `
  --tiles D:\pycodes\didi_xian_raw\valhalla_data\tiles
```

The builder refuses to overwrite existing `.gph` tiles unless `--force` is
explicitly supplied. A normal cold benchmark loads existing tiles and never
rebuilds them.
