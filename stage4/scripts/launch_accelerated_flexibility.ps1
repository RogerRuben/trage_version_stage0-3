param(
    [string]$Workspace = 'D:/pycodes/didi_xian_raw/.worktrees/stage0-v6-valhalla',
    [string]$FleetPy = 'D:/pycodes/didi_xian_raw/.external/FleetPy',
    [string]$AccelerationConfig = 'stage4/config/symmetric_flexibility_acceleration_v1.json',
    [string]$RunLogDirectory,
    [switch]$Worker
)
$ErrorActionPreference = 'Stop'
$Workspace = (Resolve-Path -LiteralPath $Workspace).ProviderPath
$configPath = (Resolve-Path -LiteralPath (Join-Path $Workspace $AccelerationConfig)).ProviderPath
$version = (Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json).version
if ($version -notin @('symmetric_flexibility_acceleration_v1','symmetric_flexibility_acceleration_v2')) { throw 'Unknown acceleration version' }
$outputName = if ($version -eq 'symmetric_flexibility_acceleration_v2') {'accelerated_v2_full_day'} else {'accelerated_full_day'}
$policyDirectory = Join-Path $Workspace ("stage4/output/symmetric_flexibility_v1/$outputName/SERVICE_PRESERVING_LOOKAHEAD")
if (-not $Worker) {
    if (Test-Path -LiteralPath $policyDirectory) { throw 'Existing accelerated output is never automatically restarted or overwritten' }
    $live = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like '*stage4.analysis.symmetric_flexibility_full_day*' })
    if ($live.Count) { throw 'A native experiment is already running' }
    $stamp = [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ')
    $RunLogDirectory = Join-Path $Workspace ("stage4/output/symmetric_flexibility_v1/runner_logs/" + $stamp)
    New-Item -ItemType Directory -Path $RunLogDirectory -Force | Out-Null
    $codeSha = (& git -C $Workspace -c "safe.directory=$Workspace" rev-parse HEAD).Trim()
    if ($LASTEXITCODE -ne 0) { throw 'Cannot record the execution commit' }
    $accelerationSha = (Get-FileHash -LiteralPath $configPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($version -eq 'symmetric_flexibility_acceleration_v2') {
        $checks = Get-Content -LiteralPath (Join-Path $Workspace 'stage4/docs/flexibility_dispatch/acceleration_v2/checks.json') -Raw | ConvertFrom-Json
        if ($checks.status -ne 'PASS' -or $checks.acceleration_config_sha256 -ne $accelerationSha) { throw 'The finite v2 construction checks are not complete for this configuration' }
    }
    $shell = (Get-Process -Id $PID).Path
    $script = $PSCommandPath
    $arguments = @('-NoProfile','-ExecutionPolicy','Bypass','-File',$script,
                   '-Workspace',$Workspace,'-FleetPy',$FleetPy,'-AccelerationConfig',$AccelerationConfig,'-RunLogDirectory',$RunLogDirectory,'-Worker')
    $process = Start-Process -FilePath $shell -ArgumentList $arguments -WorkingDirectory $Workspace -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $RunLogDirectory 'stdout.log') `
        -RedirectStandardError (Join-Path $RunLogDirectory 'stderr.log') -PassThru
    $record = [ordered]@{status='LAUNCHED';runner_pid=$process.Id;started_at_utc=[DateTime]::UtcNow.ToString('o');logs=$RunLogDirectory;policy='SERVICE_PRESERVING_LOOKAHEAD';myopic_reused=$true;code_sha=$codeSha;acceleration_config_sha256=$accelerationSha}
    [IO.File]::WriteAllText((Join-Path $RunLogDirectory 'launch.json'),($record | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
    $record | ConvertTo-Json
    return
}
$logRoot = [IO.Path]::GetFullPath($RunLogDirectory)
if (-not $logRoot.StartsWith($Workspace.TrimEnd('\')+'\',[StringComparison]::OrdinalIgnoreCase)) { throw 'Runner logs must stay in the workspace' }
$env:OMP_NUM_THREADS='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
$env:CUDA_VISIBLE_DEVICES='-1'
Set-Location -LiteralPath $Workspace
$started = [DateTime]::UtcNow
$exitCode = 1
$failure = $null
try {
    & 'D:/anaconda/envs/stage0-valhalla/python.exe' -X faulthandler -m stage4.analysis.symmetric_flexibility_full_day `
        --fleetpy-root $FleetPy --policy SERVICE_PRESERVING_LOOKAHEAD `
        --acceleration-config $AccelerationConfig --administrative-timeout-s 21600
    $exitCode = $LASTEXITCODE
} catch { $failure = $_.Exception.ToString() }
finally {
    $record = [ordered]@{status=if ($exitCode -eq 0) {'EXITED_SUCCESS'} else {'EXITED_FAILURE'};exit_code=$exitCode;runner_pid=$PID;started_at_utc=$started.ToString('o');ended_at_utc=[DateTime]::UtcNow.ToString('o');error=$failure}
    [IO.File]::WriteAllText((Join-Path $logRoot 'exit.json'),($record | ConvertTo-Json),[Text.UTF8Encoding]::new($false))
}
exit $exitCode
