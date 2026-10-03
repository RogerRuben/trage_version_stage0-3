$ErrorActionPreference = 'Stop'
$taskRoot = (Resolve-Path (Join-Path $PSScriptRoot '../..')).Path
$taskOutput = Join-Path $taskRoot 'stage4/output/paper_enhancement/control_freedom_structure'
if (Test-Path (Join-Path $taskOutput 'summary.json')) { throw 'Existing result: do not overwrite' }
New-Item -ItemType Directory -Path $taskOutput -Force | Out-Null
$env:OMP_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:NUMEXPR_NUM_THREADS = '1'
$taskProcess = Start-Process -FilePath 'D:/anaconda/envs/stage0-valhalla/python.exe' -ArgumentList '-u','-m','stage4.tools.control_freedom_diagnostic' -WorkingDirectory $taskRoot -WindowStyle Hidden -PassThru -RedirectStandardOutput (Join-Path $taskOutput 'run.log') -RedirectStandardError (Join-Path $taskOutput 'stderr.log')
$taskTimer = [System.Diagnostics.Stopwatch]::StartNew()
$taskPeak = 0L
while (-not $taskProcess.HasExited) {
    $taskProcess.Refresh()
    $taskPeak = [Math]::Max($taskPeak, $taskProcess.WorkingSet64)
    if ($taskProcess.WorkingSet64 -gt 2GB -or $taskTimer.Elapsed.TotalSeconds -gt 600) {
        $taskProcess.Kill()
        $taskProcess.WaitForExit()
        throw ('RESOURCE_LIMIT terminated only diagnostic PID ' + $taskProcess.Id)
    }
    Start-Sleep -Milliseconds 250
}
$taskProcess.WaitForExit()
Write-Output ('supervisor_sampled_peak_rss_mb=' + ($taskPeak / 1MB))
Write-Output ('supervisor_elapsed_s=' + $taskTimer.Elapsed.TotalSeconds)
Write-Output ('child_exit_code=' + $taskProcess.ExitCode)
Get-Content (Join-Path $taskOutput 'run.log') -Tail 12
if ($taskProcess.ExitCode -ne 0) {
    Get-Content (Join-Path $taskOutput 'stderr.log') -Tail 30
    exit 2
}
