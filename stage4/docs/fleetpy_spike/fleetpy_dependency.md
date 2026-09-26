# FleetPy Dependency Record

- Upstream repository: `https://github.com/TUM-VT/FleetPy.git`
- Pinned commit: `0379f9725a147ff33c674de4884cdf89fd787fa9`
- License: MIT (`Copyright (c) 2021 TUM-VT`)
- Installation method: external shallow Git checkout pinned by full SHA; FleetPy source is not copied into this repository.
- Upstream environment: Python 3.10, NumPy 2.2, pandas 2.2.3, SciPy 1.15 and the geospatial packages listed in FleetPy `environment.yml`.
- Spike runtime: existing `stage0-valhalla` environment. The pinned pure-Python request, external-vehicle, route-leg and state modules import without a separate FleetPy environment.
- FleetPy C++ router required: no.
- Gurobi required: no.
- OR-Tools required: no.
- Core FleetPy patch: none.

Reproduction checkout:

```powershell
git clone --depth 1 https://github.com/TUM-VT/FleetPy.git <external-path>/FleetPy
git -C <external-path>/FleetPy checkout 0379f9725a147ff33c674de4884cdf89fd787fa9
```

Spike command:

```powershell
conda run -n stage0-valhalla python -m stage4.fleetpy_adapter.spike_runner `
  --root . `
  --fleetpy-root <external-path>/FleetPy
```
