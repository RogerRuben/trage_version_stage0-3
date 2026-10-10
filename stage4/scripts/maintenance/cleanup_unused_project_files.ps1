param(
    [string]$WorkspaceRoot = 'D:/pycodes/didi_xian_raw',
    [switch]$Apply,
    [switch]$ResumeVerifiedArchive,
    [long]$InitialDFreeBytes = -1
)
$ErrorActionPreference = 'Stop'
$workspace = (Resolve-Path -LiteralPath $WorkspaceRoot).ProviderPath.TrimEnd('\')
$prefix = $workspace + '\'
$reportDir = Join-Path $workspace '.worktrees/stage0-v6-valhalla/stage4/docs/maintenance'

function Assert-Child([string]$Path, [switch]$MayNotExist) {
    $target = if ($MayNotExist) { [IO.Path]::GetFullPath($Path) } else { (Resolve-Path -LiteralPath $Path).ProviderPath }
    if (-not $target.StartsWith($prefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Target outside named workspace: $target" }
    $ancestor = if (Test-Path -LiteralPath $target) { $target } else { Split-Path -Parent $target }
    while ($ancestor -ne $workspace -and $ancestor.Length -ge $workspace.Length) {
        if (Test-Path -LiteralPath $ancestor) {
            if ((Get-Item -LiteralPath $ancestor -Force).Attributes -band [IO.FileAttributes]::ReparsePoint) { throw "Reparse target refused: $ancestor" }
        }
        $ancestor = Split-Path -Parent $ancestor
    }
    return $target
}

$protected = [Collections.Generic.Dictionary[string,string]]::new([StringComparer]::OrdinalIgnoreCase)
$signalRoot = Join-Path $workspace 'map_data/signal_gapfill_20261004'
$manifest = Get-Content -LiteralPath (Join-Path $signalRoot 'manifest.json') -Raw | ConvertFrom-Json
$seal = Get-Content -LiteralPath (Join-Path $signalRoot 'SEAL.json') -Raw | ConvertFrom-Json
if ((Get-FileHash -LiteralPath (Join-Path $signalRoot 'manifest.json') -Algorithm SHA256).Hash.ToLower() -ne $seal.manifest_sha256) { throw 'Signal manifest differs from seal' }
foreach ($item in $manifest.artifacts) { $protected[(Assert-Child (Join-Path $signalRoot $item.path))] = $item.sha256 }
$external = Get-Content -LiteralPath (Join-Path $signalRoot 'inputs/protected_hashes.json') -Raw | ConvertFrom-Json
foreach ($item in $external.PSObject.Properties) { $protected[(Assert-Child $item.Name)] = [string]$item.Value }
function Check-Protected {
    foreach ($entry in $protected.GetEnumerator()) {
        if ((Get-FileHash -LiteralPath $entry.Key -Algorithm SHA256).Hash.ToLower() -ne $entry.Value) { throw "Protected file differs: $($entry.Key)" }
    }
}
Check-Protected

$snapshotDirs = [Collections.Generic.List[string]]::new()
$state = Join-Path $workspace 'map_data/amap_signal_reconstruction_v2/state'
Get-ChildItem -LiteralPath $state -Directory | Where-Object { $_.Name -match '^(before_rematch_|before_bypass_|before_location_fences_|pre_member_geometry_|post_member_geometry_|after_bypass_rematch_)' } | ForEach-Object { $snapshotDirs.Add($_.FullName) }
$batches = Join-Path $workspace 'map_data/amap_local_intersection_v1/state/batches'
Get-ChildItem -LiteralPath $batches -Directory | ForEach-Object { $snapshotDirs.Add($_.FullName) }
$snapshotDirs.Add((Join-Path $workspace 'map_data/amap_geometry_reconstruction_v1/state/batches/2026-10-02_native_004350/before'))
$files = [Collections.Generic.List[object]]::new()
foreach ($directory in $snapshotDirs) {
    $checked = Assert-Child $directory
    foreach ($file in Get-ChildItem -LiteralPath $checked -File -Recurse -Force) {
        if ($file.Length -lt 5MB -or $protected.ContainsKey($file.FullName)) { continue }
        $full = Assert-Child $file.FullName
        $files.Add([pscustomobject]@{path=$full;relative=$full.Substring($prefix.Length).Replace('\','/');bytes=$file.Length;sha256=(Get-FileHash -LiteralPath $full -Algorithm SHA256).Hash.ToLower()})
    }
}
$archivePath = Assert-Child (Join-Path $workspace 'map_data/_cleanup_archives/snapshots_20261005.zip') -MayNotExist
$archiveBytes = 0L
$deletedBytes = 0L
$deletions = [Collections.Generic.List[object]]::new()
$skipped = [Collections.Generic.List[object]]::new()
$before = if ($InitialDFreeBytes -ge 0) { $InitialDFreeBytes } else { (Get-PSDrive -Name D).Free }
Write-Output ("Snapshot plan: {0} files, {1:N3} GiB; protected {2} files" -f $files.Count,(($files | Measure-Object bytes -Sum).Sum/1GB),$protected.Count)
if (-not $Apply) { $files | Select-Object relative,bytes | ConvertTo-Json -Depth 4; return }
if ((Test-Path -LiteralPath $archivePath) -and -not $ResumeVerifiedArchive) { throw 'Archive exists; never overwrite a previous recovery archive' }

Add-Type -AssemblyName System.IO.Compression.FileSystem
New-Item -ItemType Directory -Path (Split-Path -Parent $archivePath) -Force | Out-Null
if ($ResumeVerifiedArchive) {
    $files.Clear()
    $archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
    try {
        foreach ($entry in $archive.Entries) {
            $full = Assert-Child (Join-Path $workspace $entry.FullName) -MayNotExist
            if (-not @($snapshotDirs | Where-Object { $full.StartsWith($_.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase) }).Count -or $protected.ContainsKey($full)) { throw 'Resume entry is not a declared snapshot' }
            $stream = $entry.Open(); $hasher = [Security.Cryptography.SHA256]::Create()
            try { $hash = ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLower() }
            finally { $stream.Dispose(); $hasher.Dispose() }
            $files.Add([pscustomobject]@{path=$full;relative=$entry.FullName;bytes=$entry.Length;sha256=$hash})
        }
    } finally { $archive.Dispose() }
} else {
    $archive = [IO.Compression.ZipFile]::Open($archivePath,[IO.Compression.ZipArchiveMode]::Create)
    try {
        foreach ($file in $files) {
            [IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive,$file.path,$file.relative,[IO.Compression.CompressionLevel]::Optimal) | Out-Null
            Write-Output ("Archived: " + $file.relative)
        }
    } finally { $archive.Dispose() }
}
$archive = [IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    foreach ($file in $files) {
        $entry = $archive.GetEntry($file.relative)
        if ($null -eq $entry -or $entry.Length -ne $file.bytes) { throw "Archive length mismatch: $($file.relative)" }
        $stream = $entry.Open(); $hasher = [Security.Cryptography.SHA256]::Create()
        try { $hash = ([BitConverter]::ToString($hasher.ComputeHash($stream))).Replace('-','').ToLower() }
        finally { $stream.Dispose(); $hasher.Dispose() }
        if ($hash -ne $file.sha256) { throw "Archive hash mismatch: $($file.relative)" }
        if ((Test-Path -LiteralPath $file.path) -and (Get-FileHash -LiteralPath $file.path -Algorithm SHA256).Hash.ToLower() -ne $hash) { throw "Live source hash mismatch: $($file.relative)" }
        if (-not (Test-Path -LiteralPath $file.path) -and -not $ResumeVerifiedArchive) { throw "Source disappeared during archive: $($file.relative)" }
    }
} finally { $archive.Dispose() }
Write-Output 'All archive entries and live source hashes verified; removing only these file copies.'
foreach ($file in $files) {
    if (Test-Path -LiteralPath $file.path) {
        $full = Assert-Child $file.path
        Remove-Item -LiteralPath $full -Force
    }
    $deletedBytes += $file.bytes
    $deletions.Add([pscustomobject]@{path=$file.relative;bytes=$file.bytes;category='archived_snapshot';recoverable=$true;sha256=$file.sha256})
}
$archiveBytes = (Get-Item -LiteralPath $archivePath).Length

# Rebuildable bytecode/cache folders only; never remove a workspace, worktree,
# source/results folder, raw archive, model, or inference input by name guessing.
$cacheScanErrors = @()
$caches = @(Get-ChildItem -LiteralPath $workspace -Directory -Recurse -Force -ErrorAction SilentlyContinue -ErrorVariable +cacheScanErrors | Where-Object { $_.Name -in @('__pycache__','.pytest_cache') })
foreach ($scanError in $cacheScanErrors) { $skipped.Add([pscustomobject]@{path=[string]$scanError.TargetObject;reason='cache scan permission denied; ACL not changed'}) }
foreach ($directory in $caches) {
    try {
    if (-not (Test-Path -LiteralPath $directory.FullName)) { continue }
    $full = Assert-Child $directory.FullName
    $dirPrefix = $full.TrimEnd('\') + '\'
    if (@($protected.Keys | Where-Object { $_.StartsWith($dirPrefix,[StringComparison]::OrdinalIgnoreCase) }).Count) { continue }
    if (@(Get-ChildItem -LiteralPath $full -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { continue }
    $bytes = (Get-ChildItem -LiteralPath $full -File -Recurse -Force | Measure-Object Length -Sum).Sum
        Remove-Item -LiteralPath $full -Recurse -Force
        $deletedBytes += [long]$bytes
        $deletions.Add([pscustomobject]@{path=$full.Substring($prefix.Length).Replace('\','/');bytes=[long]$bytes;category='rebuildable_python_cache';recoverable='regenerates_on_import'})
    } catch { $skipped.Add([pscustomobject]@{path=$directory.FullName;reason=$_.Exception.Message}) }
}

$sourcePath = Join-Path $workspace 'tmp/valhalla-src'
if (Test-Path -LiteralPath $sourcePath) {
    $sourcePath = Assert-Child $sourcePath
    $dirty = @(& git -c "safe.directory=$sourcePath" -C $sourcePath status --porcelain)
    $stashes = @(& git -c "safe.directory=$sourcePath" -C $sourcePath stash list)
    $branches = @(& git -c "safe.directory=$sourcePath" -C $sourcePath for-each-ref refs/heads --format='%(refname)')
    $head = (& git -c "safe.directory=$sourcePath" -C $sourcePath rev-parse HEAD).Trim()
    if ($dirty.Count -eq 0 -and $stashes.Count -eq 0 -and $branches.Count -eq 0 -and $head -eq '17af0d031babb05d24304a16e89239a074c7bc8c') {
        $bytes = (Get-ChildItem -LiteralPath $sourcePath -File -Recurse -Force | Measure-Object Length -Sum).Sum
        if (@(Get-ChildItem -LiteralPath $sourcePath -Recurse -Force | Where-Object { $_.Attributes -band [IO.FileAttributes]::ReparsePoint }).Count) { throw 'Source clone contains reparse points' }
        Remove-Item -LiteralPath $sourcePath -Recurse -Force
        $deletedBytes += [long]$bytes
        $deletions.Add([pscustomobject]@{path='tmp/valhalla-src';bytes=[long]$bytes;category='clean_upstream_build_source';recoverable='git clone https://github.com/valhalla/valhalla.git; checkout 17af0d031babb05d24304a16e89239a074c7bc8c'})
    } else { $skipped.Add([pscustomobject]@{path=$sourcePath;reason='local changes, branches/stashes, or unexpected commit'}) }
}
Check-Protected
$rowsAll = @(Import-Csv -LiteralPath (Join-Path $signalRoot 'results/xian_signalized_intersections_final.csv')).Count
$rowsMain = @(Import-Csv -LiteralPath (Join-Path $signalRoot 'results/map_or_amap_positive_final.csv')).Count
if ($rowsAll -ne 699 -or $rowsMain -ne 693) { throw 'Final signal set row counts changed' }
New-Item -ItemType Directory -Path $reportDir -Force | Out-Null
$report = [ordered]@{date='2026-10-05';workspace=$workspace;archive=$archivePath;archive_bytes=$archiveBytes;archive_sha256=(Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLower();snapshot_file_count=$files.Count;deleted_bytes=$deletedBytes;net_released_bytes=$deletedBytes-$archiveBytes;d_free_before_bytes=$before;d_free_after_bytes=(Get-PSDrive -Name D).Free;protected_files_verified=$protected.Count;signal_rows=699;map_or_amap_rows=693;deletions=$deletions;skipped=$skipped}
$output = Join-Path $reportDir 'project_disk_cleanup_20261005.json'
[IO.File]::WriteAllText($output,($report | ConvertTo-Json -Depth 8),[Text.UTF8Encoding]::new($false))
Write-Output ("Cleanup complete: net {0:N3} GiB; D free {1:N3} GiB; signals 699/693 unchanged" -f (($deletedBytes-$archiveBytes)/1GB),((Get-PSDrive -Name D).Free/1GB))
