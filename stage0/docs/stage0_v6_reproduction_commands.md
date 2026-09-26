# Stage 0 v6 Valhalla reproduction commands

Run all commands from the repository root. Runtime data is written to the
git-ignored `valhalla_data/`, `stage0/work_v6/`, and `stage0/output_v6/`
directories.

## 1. Create the environment

```powershell
conda create -n stage0-valhalla python=3.12 -y
conda run -n stage0-valhalla python -m pip install -r stage0/v6/requirements.txt
```

## 2. Build Valhalla tiles once

```powershell
conda run -n stage0-valhalla python -m stage0.v6.build_tiles `
  --pbf map_data/xian_roadmap_update.osm.pbf `
  --config valhalla_data/valhalla.json `
  --tiles valhalla_data/tiles
```

When valid tiles and a matching build manifest already exist, do not rebuild
them for cold/hot experiments.

## 3. Run a one-order smoke test

```powershell
conda run -n stage0-valhalla python -m stage0.v6.cli `
  --config stage0/config/stage0_v6_valhalla.yaml single
```

## 4. Run the fixed 600-order cold/hot experiment

```powershell
conda run -n stage0-valhalla python -m stage0.v6.cli `
  --config stage0/config/stage0_v6_valhalla.yaml benchmark
```

Both runs reuse a persistent Valhalla Actor in one Python process. The cold
run does not rebuild tiles.

## 5. Generate the comparison report and manual-audit package

```powershell
conda run -n stage0-valhalla python -m stage0.v6.cli `
  --config stage0/config/stage0_v6_valhalla.yaml report

conda run -n stage0-valhalla python -m stage0.v6.cli `
  --config stage0/config/stage0_v6_valhalla.yaml manual-audit
```

## 6. Validate

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
conda run -n stage0-valhalla python -m pytest -q
conda run -n stage0-valhalla python -m compileall -q stage0/v6 stage0/tests
git diff --check
```
