# ---------------------------------------------------------------------------
# TAMNet audit — full reproduction pipeline (PowerShell, the native entry
# point on the machine this audit was run on).
#
#   .\run_all.ps1                 # every stage, in order
#   .\run_all.ps1 -Stages check   # QA only
#   .\run_all.ps1 -Stages figures,tables,paper
#
# Every stage is skip-if-done (training keys on metrics.json), so re-running
# never re-trains and never evaluates the TEST set twice for one config.
# ---------------------------------------------------------------------------
[CmdletBinding()]
param(
    [string[]]$Stages = @('split','train','evaluate','aggregate','bootstrap',
                          'external','uncertainty','robustness','seg',
                          'brats3d','attention','efficiency','qual',
                          'figures','tables','paper','check'),
    [string]$Python = 'python'
)

$ErrorActionPreference = 'Stop'
Set-Location -Path $PSScriptRoot

function Invoke-Stage([string]$name, [string[]]$cmd) {
    Write-Host ""
    Write-Host "=== $name ===" -ForegroundColor Cyan
    & $cmd[0] @($cmd[1..($cmd.Count-1)])
    if ($LASTEXITCODE -ne 0) { throw "stage '$name' failed (exit $LASTEXITCODE)" }
}

foreach ($s in $Stages) {
    switch ($s) {
        'split'       { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_split.py') }
        'train'       { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_train.py') }
        'evaluate'    { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_evaluate.py') }
        'aggregate'   { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_aggregate.py') }
        'bootstrap'   { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_bootstrap.py') }
        'external'    { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_external_eval.py') }
        'uncertainty' { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_uncertainty.py') }
        'robustness'  { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_robustness.py') }
        'seg'         { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_seg_eval.py') }
        'brats3d'     { Invoke-Stage $s @($Python,'-X','utf8','-m','scripts.evaluate_segmentation','--split','test')
                        Invoke-Stage $s @($Python,'-X','utf8','scripts/phase5_ingest_unet3d.py') }
        'attention'   { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_attention_alignment.py') }
        'efficiency'  { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase3_efficiency.py') }
        # qualitative + per-sample predictions + supplementary S1-S10
        'qual'        { Invoke-Stage $s @($Python,'-X','utf8','scripts/phase4_seg_qualitative.py')
                        Invoke-Stage $s @($Python,'-X','utf8','scripts/phase4_save_predictions.py')
                        Invoke-Stage $s @($Python,'-X','utf8','scripts/make_supplementary.py') }
        'figures'     { Invoke-Stage $s @($Python,'-X','utf8','scripts/make_figures.py')
                        Invoke-Stage $s @($Python,'-X','utf8','scripts/make_figures_static.py') }
        'tables'      { Invoke-Stage $s @($Python,'-X','utf8','scripts/make_tables.py') }
        'paper'       { Invoke-Stage $s @($Python,'-X','utf8','paper/rewrite_paper.py') }
        'check'       { Invoke-Stage $s @($Python,'-X','utf8','scripts/check_paper_numbers.py')
                        Invoke-Stage $s @($Python,'-X','utf8','scripts/check_figures.py')
                        Invoke-Stage $s @($Python,'-X','utf8','-m','pytest','tests','-q') }
        'test'        { Invoke-Stage $s @($Python,'-X','utf8','-m','pytest','tests','-q') }
        'install'     { Invoke-Stage $s @($Python,'-m','pip','install','-r','requirements.txt') }
        default       { throw "unknown stage: $s" }
    }
}

Write-Host ""
Write-Host "run_all.ps1: done ($($Stages -join ', '))" -ForegroundColor Green
