param([int]$WaitSeconds = 0)
# Phase 3 progress watcher: prints status of ALL parallel queues + their runs.
# Usage: powershell -File scripts\phase3_status.ps1 [-WaitSeconds 300]
if ($WaitSeconds -gt 0) { Start-Sleep -Seconds $WaitSeconds }
$root = Split-Path $PSScriptRoot -Parent
$statusDir = Join-Path $root "results\phase3"
$statusFiles = Get-ChildItem $statusDir -Filter "queue_status*.json" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ne "queue_status.json.tmp" }
if (-not $statusFiles) { Write-Output "no queue_status*.json found (queues not started)"; exit 0 }
foreach ($sf in $statusFiles) {
    $s = Get-Content $sf.FullName -Raw | ConvertFrom-Json
    $failed = if ($s.failed) { $s.failed.name } else { "none" }
    Write-Output ("QUEUE [{0}]: current={1}  completed={2}/{3}  failed={4}  updated={5}" -f `
        $sf.BaseName, $s.current, @($s.completed).Count, $s.total_entries, $failed, $s.updated_at)
    if ($s.current) {
        $runDir = Join-Path $root ("results\phase3\runs\" + $s.current)
        $progPath = Join-Path $runDir "progress.json"
        if (Test-Path $progPath) {
            $p = Get-Content $progPath -Raw | ConvertFrom-Json
            Write-Output ("  RUN   : {0}  epoch {1}/{2}  status={3}" -f $s.current, $p.epoch, $p.epochs_total, $p.status)
            if ($null -ne $p.train_loss) {
                Write-Output ("  METRIC: train_loss={0:F4}  val_f1={1:F4}  val_acc={2:F4}  best={3:F4}@ep{4}" -f `
                    $p.train_loss, $p.val_f1, $p.val_accuracy, $p.best_val_f1, $p.best_epoch)
            }
            Write-Output ("  TIME  : epoch={0}min  elapsed={1}min  eta={2}min  updated={3}" -f `
                [math]::Round($p.epoch_time_s/60,2), [math]::Round($p.elapsed_s/60,1), `
                [math]::Round($p.eta_s/60,1), $p.updated_at)
        } else {
            Write-Output ("  RUN   : {0}  (epoch 1 in progress)" -f $s.current)
        }
    }
}
$gpu = nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
Write-Output ("GPU    : " + $gpu)
