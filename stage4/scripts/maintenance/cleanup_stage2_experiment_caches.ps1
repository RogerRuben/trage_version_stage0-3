param([switch]$Apply)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../../..')).TrimEnd('\')
$prefix = $root + '\'

# Explicit files only: never delete a workspace, output tree, or model directory.
function Assert-File([string]$Path) {
    $full = (Resolve-Path -LiteralPath $Path).ProviderPath
    if (-not $full.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Outside task worktree: $full"
    }
    $ancestor = $full
    while ($ancestor -ne $root) {
        $item = Get-Item -LiteralPath $ancestor -Force
        if ($item.Attributes -band [IO.FileAttributes]::ReparsePoint) {
            throw "Reparse path refused: $ancestor"
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    if (-not (Test-Path -LiteralPath $full -PathType Leaf)) { throw "Not a file: $full" }
    return $full
}

$tracked = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
git -c "safe.directory=$root" -C $root ls-files | ForEach-Object { [void]$tracked.Add($_) }
if ($LASTEXITCODE -ne 0) { throw 'Cannot establish tracked-file protection' }
$plans = [Collections.Generic.List[object]]::new()
function Add-Plan([string]$Relative, [string]$Category) {
    $path = Join-Path $root $Relative
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return }
    $full = Assert-File $path
    $relative = $full.Substring($prefix.Length).Replace('\', '/')
    if ($tracked.Contains($relative)) { throw "Tracked file refused: $relative" }
    $file = Get-Item -LiteralPath $full
    $plans.Add([pscustomobject]@{
        path = $full; relative = $relative; category = $Category
        bytes = $file.Length; modified_utc = $file.LastWriteTimeUtc.ToString('o')
    })
}

# Obsolete derived oracle-timing inputs, not the revealed-route production data.
# Keep Test31 for the surviving historical diagnostic; retain all manifests.
for ($day = 9; $day -le 27; $day++) {
    Add-Plan ('stage2/output_v4/route_conditioned_dataset/oracle_timing/day=201610{0:00}.parquet' -f $day) 'legacy_oracle_features'
}
# Keep M0 matrix manifests/evaluations, every checkpoint, and all transfer shards.
foreach ($relative in @(
    'stage2/output_v5_2/development/M0/train_matrix.npz',
    'stage2/output_v5_2/development/M0/validation_matrix.npz',
    'stage2/output_v5_2/development/M0/evaluation_matrix.npz',
    'stage2/output_v5_2/transfer_tuning/M0/train_matrix.npz',
    'stage2/output_v5_2/transfer_tuning/M0/validation_matrix.npz'
)) { Add-Plan $relative 'm0_work_matrix' }

$groups = [Collections.Generic.List[string]]::new()
foreach ($group in @('ALL_HOT_COMMON_CONTROL', 'COMPATIBLE_HOT_COMMON_CONTROL', 'OPPORTUNITY_JOINT_COMMON_CONTROL')) {
    $groups.Add("stage4/output/capability_chain_planning/layout_comparison_300_600_v1/$group")
}
foreach ($condition in @('Q05_REFERENCE_LABELS', 'Q10_LABEL_SEED2', 'Q20_REFERENCE_LABELS')) {
    foreach ($group in @('COMPATIBLE_HOT_COMMON_CONTROL', 'OPPORTUNITY_JOINT_COMMON_CONTROL')) {
        $groups.Add("stage4/output/capability_chain_planning/layout_confirmation_v1/$condition/native/$group")
    }
}
foreach ($group in $groups) {
    $summaryPath = Join-Path $root "$group/summary.json"
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    if ($summary.status -ne 'TWO_SCALE_CITY_GROUP_COMPLETE' -or -not $summary.full_day) {
        throw "Incomplete simulation refused: $group"
    }
    foreach ($suffix in @('', '-wal', '-shm', '-journal')) {
        Add-Plan "$group/routing/cache.sqlite3$suffix" 'completed_group_raw_route_cache'
    }
}

# Small preservation check, not a simulation or an experiment audit.
$protected = [Collections.Generic.List[string]]::new()
foreach ($relative in @(
    'stage2/output_v5_2/development/M3/epoch_004.pt',
    'stage2/output_v5_2/development/M3/model_manifest.json',
    'stage2/output_v5/protocols/development/tensor_shards/feature_artifacts.json',
    'stage2/output_v5_2/transfer_shards/protocol=development/transfer_manifest.json',
    'stage2/output_v4/route_conditioned_dataset/oracle_timing/day=20161031.parquet',
    'stage3/output/odd_tod/s2a/stage3_full_network_edges.parquet',
    'stage3/output/odd_tod/s4/test31_m3_predictions.parquet',
    'stage4/output/symmetric_flexibility_v1/input/test31_research_routes.parquet',
    'stage4/output/symmetric_flexibility_v1/input/train_request_templates.parquet'
)) { $protected.Add((Assert-File (Join-Path $root $relative))) }
foreach ($group in $groups) {
    foreach ($name in @('assignments.parquet', 'empty_movements.parquet', 'summary.json', 'cohort_outcomes.parquet')) {
        $protected.Add((Assert-File (Join-Path $root "$group/$name")))
    }
}
$beforeHashes = @{}
foreach ($path in $protected) { $beforeHashes[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
$beforeFree = (Get-PSDrive -Name D).Free
$deleted = [Collections.Generic.List[object]]::new()
$errors = [Collections.Generic.List[object]]::new()
$activePython = @(Get-Process -Name python, pythonw -ErrorAction SilentlyContinue)
if ($Apply -and $activePython.Count -gt 0) {
    throw 'Python process present: no cleanup while an experiment may be active'
}
if ($Apply) {
    foreach ($plan in $plans) {
        try {
            $path = Assert-File $plan.path
            $current = Get-Item -LiteralPath $path
            if ($current.Length -ne $plan.bytes -or $current.LastWriteTimeUtc.ToString('o') -ne $plan.modified_utc) {
                throw 'File changed after plan; skipped'
            }
            # A live cache/memmap must not be removed. No forced unlock or ACL edit.
            $handle = [IO.File]::Open($path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::None)
            $handle.Dispose()
            Remove-Item -LiteralPath $path -ErrorAction Stop
            if (Test-Path -LiteralPath $path) { throw 'Removal was not confirmed' }
            $deleted.Add($plan)
        } catch {
            $errors.Add([pscustomobject]@{path=$plan.relative;reason=$_.Exception.Message})
        }
    }
}
$changed = [Collections.Generic.List[string]]::new()
foreach ($path in $protected) {
    if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $beforeHashes[$path]) {
        $changed.Add($path.Substring($prefix.Length).Replace('\', '/'))
    }
}
$categories = @($deleted | Group-Object category | ForEach-Object {
    [pscustomobject]@{category=$_.Name;files=$_.Count;bytes=($_.Group | Measure-Object bytes -Sum).Sum}
})
[pscustomobject]@{
    client_date = '2026-10-14'; worktree = $root; apply = [bool]$Apply
    planned_file_count = $plans.Count; planned_bytes = ($plans | Measure-Object bytes -Sum).Sum
    deleted_file_count = $deleted.Count; deleted_bytes = ($deleted | Measure-Object bytes -Sum).Sum
    free_before_bytes = $beforeFree; free_after_bytes = (Get-PSDrive -Name D).Free
    protected_file_count = $protected.Count; protected_changed = $changed.ToArray()
    category_totals = $categories; deletions = $deleted.ToArray(); errors = $errors.ToArray()
    planned_files = $(if (-not $Apply) { $plans.ToArray() } else { @() })
    recovery = 'Regenerate derived features/matrices/routes from retained source data and frozen code; no local copy of deleted binary caches.'
    experiments_run = 0; models_changed = 0; acl_changed = $false
} | ConvertTo-Json -Depth 7
if ($changed.Count -gt 0) { throw 'A protected file changed' }
