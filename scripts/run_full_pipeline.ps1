<#
.SYNOPSIS
    Self-Audit: Canonical Unified Training Pipeline Runner (PowerShell)
.DESCRIPTION
    Runs the single-process unified training pipeline (Schema Version 1)
    via configs/self_audit_full.yaml across the 130-epoch curriculum.
    For historical multi-phase (A -> B -> C) execution, see scripts\run_full_pipeline_legacy.ps1.
.EXAMPLE
    .\scripts\run_full_pipeline.ps1 -Device cuda
    .\scripts\run_full_pipeline.ps1 -Smoke -Device cpu
    .\scripts\run_full_pipeline.ps1 -Wandb -WandbMode offline
#>

[CmdletBinding()]
param (
    [string]$Config = "configs/self_audit_full.yaml",
    [switch]$Smoke,
    [string]$DataRoot,
    [string]$SplitManifest,
    [string]$Device,
    [Nullable[int]]$NumWorkers,
    [Nullable[int]]$MaxSteps,
    [Nullable[int]]$MaxValBatches,
    [string]$OutputDir,
    [string]$ReportDir,
    [string]$Resume,
    [Nullable[float]]$TauAccept,
    [switch]$SkipCalibration,
    [switch]$Wandb,
    [switch]$NoWandb,
    [string]$WandbMode,
    [string]$WandbProject,
    [string]$WandbEntity,
    [string]$WandbRunName,
    [switch]$NoTqdm,
    # Explicit rejection of legacy parameters
    [string]$StartPhase,
    [string]$ConfigA,
    [string]$ConfigB,
    [string]$ConfigC,
    [string]$ConfigAnnotation,
    [string]$ConfigAuditor,
    [string]$ConfigJoint,
    [Nullable[int]]$EpochsA,
    [Nullable[int]]$EpochsB,
    [Nullable[int]]$EpochsC
)

$ErrorActionPreference = "Stop"

if ($StartPhase -or $ConfigA -or $ConfigB -or $ConfigC -or $ConfigAnnotation -or $ConfigAuditor -or $ConfigJoint -or $EpochsA -or $EpochsB -or $EpochsC) {
    Write-Error "Legacy multi-phase parameters are not supported by canonical run_full_pipeline.ps1. Use scripts\run_full_pipeline_legacy.ps1 for historical multi-stage execution."
    exit 2
}

if ($OutputDir -and -not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }
if ($ReportDir -and -not (Test-Path $ReportDir)) { New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null }

$displayDevice = if ($PSBoundParameters.ContainsKey('Device')) { $Device } else { "<from config>" }
$displayOut = if ($PSBoundParameters.ContainsKey('OutputDir')) { $OutputDir } else { "<from config>" }
$displayRep = if ($PSBoundParameters.ContainsKey('ReportDir')) { $ReportDir } else { "<from config>" }
$displayWandb = if ($Wandb) { "true" } elseif ($NoWandb) { "false" } else { "<from config>" }

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "          Self-Audit: Canonical Unified Training Pipeline" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " Config:          $Config"
Write-Host " Device:          $displayDevice"
Write-Host " Smoke Mode:      $Smoke"
Write-Host " Output Dir:      $displayOut"
Write-Host " Report Dir:      $displayRep"
Write-Host " WandB:           $displayWandb"
Write-Host "==============================================================================" -ForegroundColor Cyan

$cmdArgs = @("scripts/train_self_audit.py", "--config", $Config)

if ($PSBoundParameters.ContainsKey('Device')) { $cmdArgs += @("--device", $Device) }
if ($PSBoundParameters.ContainsKey('OutputDir')) { $cmdArgs += @("--output_dir", $OutputDir) }
if ($PSBoundParameters.ContainsKey('ReportDir')) { $cmdArgs += @("--report_dir", $ReportDir) }
if ($DataRoot) { $cmdArgs += @("--data_root", $DataRoot) }
if ($SplitManifest) { $cmdArgs += @("--split_manifest", $SplitManifest) }
if ($PSBoundParameters.ContainsKey('NumWorkers')) { $cmdArgs += @("--num_workers", $NumWorkers) }
if ($Resume) { $cmdArgs += @("--resume", $Resume) }
if ($PSBoundParameters.ContainsKey('TauAccept')) { $cmdArgs += @("--tau_accept", $TauAccept) }
if ($SkipCalibration) { $cmdArgs += @("--skip_calibration") }

if ($Smoke) {
    $cmdArgs += @("--max_steps", "2", "--max_val_batches", "2")
    if (-not $PSBoundParameters.ContainsKey('Device')) { $cmdArgs += @("--device", "cpu") }
} else {
    if ($PSBoundParameters.ContainsKey('MaxSteps')) { $cmdArgs += @("--max_steps", $MaxSteps) }
    if ($PSBoundParameters.ContainsKey('MaxValBatches')) { $cmdArgs += @("--max_val_batches", $MaxValBatches) }
}

if ($Wandb) {
    $cmdArgs += @("--wandb")
    if ($WandbMode) { $cmdArgs += @("--wandb_mode", $WandbMode) }
    if ($WandbProject) { $cmdArgs += @("--wandb_project", $WandbProject) }
    if ($WandbEntity) { $cmdArgs += @("--wandb_entity", $WandbEntity) }
    if ($WandbRunName) { $cmdArgs += @("--wandb_run_name", $WandbRunName) }
} elseif ($NoWandb) {
    $cmdArgs += @("--no_wandb")
}

if ($NoTqdm) { $cmdArgs += @("--no_tqdm") }

& python $cmdArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
