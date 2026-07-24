# FleetPy cross-validation protocol

The cross-validation uses a frozen one-hour input containing 500–2,000
requests. Stage4 and FleetPy must receive the same request IDs, request times,
vehicle IDs, initial locations, online windows, passenger patience, pickup
travel input, and service durations. The controlled setting is Stay,
preassignment off, and Safe MinPickup/closest vehicle.

Prepare the immutable input package with:

```powershell
python stage4/scripts/prepare_fleetpy_cross_validation.py --overwrite
```

FleetPy is an external source repository rather than a package installed by
this project. Its official installation is:

```powershell
git clone https://github.com/TUM-VT/FleetPy.git
cd FleetPy
conda env create -f environment.yml
conda activate fleetpy
```

After both engines have produced real output files, compare them with:

```powershell
python stage4/scripts/compare_fleetpy_cross_validation.py `
  --stage4-result <stage4-result-directory> `
  --fleetpy-result <fleetpy-result-directory> `
  --input-manifest stage4/output/fleetpy_cross_validation/input/manifest.json
```

The comparison command raises an error if FleetPy's native `*user_stats.csv`
or `*op-stats.csv` is absent. It therefore cannot manufacture a `NOT_RUN` or
synthetic `PASS` result.

Acceptance thresholds are 5% for completed/cancelled orders and 10% for mean
pickup time, mean waiting time, vehicle busy time, pickup distance, and service
distance. A final result report is generated only from actual outputs.
