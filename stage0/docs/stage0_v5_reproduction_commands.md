# Stage 0 v5 PowerShell reproduction commands

Run from the repository root on `codex/stage0-v5`. The environment path below keeps all package
state inside the repository work directory.

```powershell
$py = "stage0/work_v5/.venv/Scripts/python.exe"
$runner = "stage0/scripts/run_stage0_v5.py"

function Assert-Gate1Pass {
  $gate = Get-Content "stage0/output_v5/reports/gate1_readiness.json" -Raw | ConvertFrom-Json
  if ($gate.status -ne "PASS" -or -not $gate.gate2_allowed) {
    throw "Gate 1 is not PASS; Gate 2 and all later data runs are locked."
  }
}
```

## Gate 0 and Gate 1

```powershell
& $py $runner --phase gate0
& $py $runner --phase materialize --dates 20161010 20161014 20161016 --orders-per-day 2000 --buckets 32
& $py $runner --phase gate1 --orders-per-day 2000 --buckets 32 --resume
```

For RAM-controlled partition sharding, materialize once and run mutually exclusive shards. Do
not run two different shard counts concurrently. The canonical default is one worker because the
compact road and movement indexes consume about 2.8 GB RSS per process in the fixed benchmark.
Only use the four-process example after confirming at least 12 GB of free RAM beyond the OS and
other applications; otherwise run the single-process Gate 1 command above.

```powershell
0..3 | ForEach-Object -Parallel {
  & $using:py $using:runner --phase match `
    --dates 20161010 20161014 20161016 `
    --orders-per-day 2000 --buckets 32 --workers 1 --no-resume `
    --bucket-shard-index $_ --bucket-shard-count 4
} -ThrottleLimit 4
& $py $runner --phase gate1 --orders-per-day 2000 --buckets 32 --resume
```

## Gate 2: three days, 10,000 complete orders/day

```powershell
Assert-Gate1Pass
& $py $runner --phase inventory --dates 20161010 20161014 20161016 --orders-per-day 10000
& $py $runner --phase materialize --dates 20161010 20161014 20161016 --orders-per-day 10000 --buckets 64 --force
& $py $runner --phase match --dates 20161010 20161014 20161016 --orders-per-day 10000 --buckets 64 --workers 1 --resume
```

Gate 2 must not start unless `stage0/output_v5/reports/gate1_readiness.json` is `PASS`.

## Gate 3: Train

```powershell
Assert-Gate1Pass
& $py $runner --phase inventory --split train --orders-per-day 10000
& $py $runner --phase materialize --split train --orders-per-day 10000 --buckets 64 --force
& $py $runner --phase match --split train --orders-per-day 10000 --buckets 64 --workers 1 --resume
```

## Gate 4: Validation

```powershell
Assert-Gate1Pass
& $py $runner --phase inventory --split validation --orders-per-day 10000
& $py $runner --phase materialize --split validation --orders-per-day 10000 --buckets 64 --force
& $py $runner --phase match --split validation --orders-per-day 10000 --buckets 64 --workers 1 --resume
& $py $runner --phase manual-review --review-pack development
```

## Freeze and Gate 5: Test

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD = "1"
& $py -m pytest -q stage0/tests
& $py $runner --phase freeze
& $py $runner --phase inventory --split test --orders-per-day 10000
& $py $runner --phase materialize --split test --orders-per-day 10000 --buckets 64 --force
& $py $runner --phase match --split test --orders-per-day 10000 --buckets 64 --workers 1 --resume
& $py $runner --phase manual-review --review-pack test
```

The Test command is methodologically valid only after
`stage0/output_v5/manifests/stage0_freeze_manifest.json` exists. Test results must not be used to
rewrite matcher or quality thresholds.

## Compact retention (dry-run by default)

```powershell
& $py $runner --phase prune
& $py $runner --phase prune --execute
```
