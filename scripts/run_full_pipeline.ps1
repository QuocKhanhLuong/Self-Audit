<#
.SYNOPSIS
    Self-Audit: Full End-to-End Pipeline Execution Script (PowerShell)
.DESCRIPTION
    Runs the full cardiac segmentation & self-audit pipeline locally on Windows:
      Phase 0: Runtime Preflight Sanity Check
      Phase 1: Phase A - Supervised Annotation Network Training
      Phase 2: Phase B - Counterfactual Transition Auditor Training
      Phase 3: Phase C - Threshold-Controlled Joint Fine-tuning
      Phase 4: Validation Transition Cache Collection
      Phase 5: Decision Threshold (tau_accept) Calibration
.EXAMPLE
    .\scripts\run_full_pipeline.ps1 -Device cuda
    .\scripts\run_full_pipeline.ps1 -Smoke -Device cpu
    .\scripts\run_full_pipeline.ps1 -Wandb -WandbMode offline
#>

[CmdletBinding()]
param (
    [string]$StartPhase = "A",
    [switch]$Smoke,
    [switch]$SkipPreflight,
    [string]$DataRoot,
    [string]$Device = "cuda",
    [Nullable[int]]$BatchSize,
    [Nullable[int]]$NumWorkers,
    [Nullable[int]]$ImageSize,
    [Nullable[int]]$EpochsA,
    [Nullable[int]]$EpochsB,
    [Nullable[int]]$EpochsC,
    [switch]$Wandb,
    [switch]$NoWandb,
    [string]$WandbMode = "offline",
    [string]$WandbProject = "self-audit",
    [string]$WandbEntity,
    [string]$SplitManifest,
    [string]$OutputDir = "weights/self_audit",
    [string]$ReportDir = "reports",
    [Alias("ConfigAnnotation")]
    [string]$ConfigA = "configs/self_audit_annotation.yaml",
    [Alias("ConfigAuditor")]
    [string]$ConfigB = "configs/self_audit_auditor.yaml",
    [Alias("ConfigJoint")]
    [string]$ConfigC = "configs/self_audit_joint.yaml"
)

$ErrorActionPreference = "Stop"

$StartPhase = $StartPhase.ToUpper()
if (-not (Test-Path $OutputDir)) { New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null }
if (-not (Test-Path $ReportDir)) { New-Item -ItemType Directory -Force -Path $ReportDir | Out-Null }

Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host "          Self-Audit: Cardiac Segmentation & Audit Pipeline" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " Device:          $Device"
Write-Host " Start Phase:     $StartPhase"
Write-Host " Smoke Mode:      $Smoke"
Write-Host " Output Dir:      $OutputDir"
Write-Host " Report Dir:      $ReportDir"
Write-Host " WandB:           $($Wandb.IsPresent) (mode: $WandbMode, project: $WandbProject)"
Write-Host " TQDM Bars:       $(if ($NoTqdm) { 'disabled' } else { 'enabled' })"
Write-Host "==============================================================================" -ForegroundColor Cyan

# Common arguments
$commonArgs = @()
if ($DataRoot) { $commonArgs += @("--data_root", $DataRoot) }
if ($SplitManifest) { $commonArgs += @("--split_manifest", $SplitManifest) }
if ($Device) { $commonArgs += @("--device", $Device) }
if ($BatchSize) { $commonArgs += @("--batch_size", "$BatchSize") }
if ($NumWorkers) { $commonArgs += @("--num_workers", "$NumWorkers") }
if ($ImageSize) { $commonArgs += @("--image_size", "$ImageSize") }
if ($Wandb.IsPresent -and -not $NoWandb.IsPresent) {
    $commonArgs += @("--wandb", "--wandb_mode", $WandbMode, "--wandb_project", $WandbProject)
    if ($WandbEntity) { $commonArgs += @("--wandb_entity", $WandbEntity) }
}
if ($NoTqdm.IsPresent) { $commonArgs += "--no_tqdm" }

function Invoke-PythonStep {
    param([string[]]$Arguments, [string]$StepName)
    & python $Arguments
    if ($LASTEXITCODE -ne 0) {
        Write-Error "$StepName failed with exit code $LASTEXITCODE"
        exit $LASTEXITCODE
    }
}

# Phase 0: Preflight
if (-not $SkipPreflight -and ($StartPhase -eq "A" -or $StartPhase -eq "PREFLIGHT")) {
    Write-Host "`n>>> [Phase 0] Running Preflight Sanity Check..." -ForegroundColor Yellow
    $preArgs = @("scripts/self_audit_preflight.py", "--config", $ConfigA)
    if ($DataRoot) { $preArgs += @("--data_root", $DataRoot) }
    if ($SplitManifest) { $preArgs += @("--split_manifest", $SplitManifest) }
    Invoke-PythonStep -Arguments $preArgs -StepName "Phase 0 Preflight"
    Write-Host ">>> [Phase 0] Preflight Check Completed Successfully." -ForegroundColor Green
}

# Phase 1: Phase A
$checkpointA = Join-Path $OutputDir "phase_a_annotation.pt"
if ($StartPhase -eq "A") {
    Write-Host "`n>>> [Phase 1/5] Phase A: Training Annotation Network..." -ForegroundColor Yellow
    $phaseAArgs = @("src/self_audit/training/train_annotation.py", "--config", $ConfigA, "--output", $checkpointA) + $commonArgs
    if ($EpochsA) { $phaseAArgs += @("--epochs", "$EpochsA") }
    if ($Smoke) { $phaseAArgs += @("--epochs", "1", "--max_steps", "2", "--max_val_batches", "2", "--no_pretrained") }
    Invoke-PythonStep -Arguments $phaseAArgs -StepName "Phase 1 Annotation Training"
    Write-Host ">>> [Phase 1/5] Phase A Completed. Checkpoint saved: $checkpointA" -ForegroundColor Green
}

# Phase 2: Phase B
$checkpointB = Join-Path $OutputDir "phase_b_auditor.pt"
if ($StartPhase -eq "A" -or $StartPhase -eq "B") {
    Write-Host "`n>>> [Phase 2/5] Phase B: Training Transition Auditor..." -ForegroundColor Yellow
    $annotCkpt = $checkpointA
    if (-not (Test-Path $annotCkpt) -and (Test-Path (Join-Path $OutputDir "best.pt"))) {
        $annotCkpt = Join-Path $OutputDir "best.pt"
    }
    if (-not (Test-Path $annotCkpt)) {
        Write-Error "Phase A checkpoint not found at $checkpointA. Run Phase A first."
        exit 1
    }

    $phaseBArgs = @("src/self_audit/training/train_auditor.py", "--config", $ConfigB, "--annotation_checkpoint", $annotCkpt, "--output", $checkpointB) + $commonArgs
    if ($EpochsB) { $phaseBArgs += @("--epochs", "$EpochsB") }
    if ($Smoke) { $phaseBArgs += @("--epochs", "1", "--max_steps", "2", "--max_val_batches", "2") }
    Invoke-PythonStep -Arguments $phaseBArgs -StepName "Phase 2 Auditor Training"
    Write-Host ">>> [Phase 2/5] Phase B Completed. Checkpoint saved: $checkpointB" -ForegroundColor Green
}

# Phase 3: Phase C
$checkpointC = Join-Path $OutputDir "phase_c_joint.pt"
if ($StartPhase -eq "A" -or $StartPhase -eq "B" -or $StartPhase -eq "C") {
    Write-Host "`n>>> [Phase 3/5] Phase C: Joint Fine-Tuning..." -ForegroundColor Yellow
    $auditCkpt = $checkpointB
    if (-not (Test-Path $auditCkpt) -and (Test-Path (Join-Path $OutputDir "best.pt"))) {
        $auditCkpt = Join-Path $OutputDir "best.pt"
    }
    if (-not (Test-Path $auditCkpt)) {
        Write-Error "Phase B checkpoint not found at $checkpointB. Run Phase B first."
        exit 1
    }

    $phaseCArgs = @("src/self_audit/training/finetune_joint.py", "--config", $ConfigC, "--checkpoint", $auditCkpt, "--output", $checkpointC) + $commonArgs
    if ($EpochsC) { $phaseCArgs += @("--epochs", "$EpochsC") }
    if ($Smoke) { $phaseCArgs += @("--epochs", "1", "--max_steps", "2", "--max_val_batches", "2") }
    Invoke-PythonStep -Arguments $phaseCArgs -StepName "Phase 3 Joint Fine-Tuning"
    Write-Host ">>> [Phase 3/5] Phase C Completed. Checkpoint saved: $checkpointC" -ForegroundColor Green
}

# Phase 4: Cache Transitions
$transitionsCache = Join-Path $ReportDir "validation_transitions.pt"
if ($StartPhase -eq "A" -or $StartPhase -eq "B" -or $StartPhase -eq "C" -or $StartPhase -eq "CACHE") {
    Write-Host "`n>>> [Phase 4/5] Caching Validation Transitions..." -ForegroundColor Yellow
    $jointCkpt = $checkpointC
    if (-not (Test-Path $jointCkpt) -and (Test-Path (Join-Path $OutputDir "best.pt"))) {
        $jointCkpt = Join-Path $OutputDir "best.pt"
    }
    if (-not (Test-Path $jointCkpt)) {
        Write-Error "Joint checkpoint not found at $checkpointC."
        exit 1
    }

    $cacheArgs = @("scripts/cache_validation_transitions.py", "--config", $ConfigC, "--checkpoint", $jointCkpt, "--output", $transitionsCache)
    if ($DataRoot) { $cacheArgs += @("--data_root", $DataRoot) }
    if ($SplitManifest) { $cacheArgs += @("--split_manifest", $SplitManifest) }
    if ($ImageSize) { $cacheArgs += @("--image_size", "$ImageSize") }
    if ($Device) { $cacheArgs += @("--device", $Device) }
    if ($BatchSize) { $cacheArgs += @("--batch_size", "$BatchSize") }
    if ($NoTqdm.IsPresent) { $cacheArgs += "--no_tqdm" }
    Invoke-PythonStep -Arguments $cacheArgs -StepName "Phase 4 Caching Transitions"
    Write-Host ">>> [Phase 4/5] Transitions Cached: $transitionsCache" -ForegroundColor Green
}

# Phase 5: Calibrate Threshold
$calibrationJson = Join-Path $ReportDir "calibration_result.json"
if ($StartPhase -eq "A" -or $StartPhase -eq "B" -or $StartPhase -eq "C" -or $StartPhase -eq "CACHE" -or $StartPhase -eq "CALIBRATE") {
    Write-Host "`n>>> [Phase 5/5] Calibrating Threshold (tau_accept)..." -ForegroundColor Yellow
    if (-not (Test-Path $transitionsCache)) {
        Write-Error "Transitions cache not found at $transitionsCache."
        exit 1
    }

    $calibArgs = @("scripts/calibrate_threshold.py", "--transitions", $transitionsCache, "--output", $calibrationJson)
    Invoke-PythonStep -Arguments $calibArgs -StepName "Phase 5 Threshold Calibration"
    Write-Host ">>> [Phase 5/5] Calibration Complete. Result saved: $calibrationJson" -ForegroundColor Green
}

Write-Host "`n==============================================================================" -ForegroundColor Cyan
Write-Host "                 Self-Audit Pipeline Execution Complete!" -ForegroundColor Cyan
Write-Host "==============================================================================" -ForegroundColor Cyan
Write-Host " Phase A Checkpoint:  $checkpointA"
Write-Host " Phase B Checkpoint:  $checkpointB"
Write-Host " Phase C Checkpoint:  $checkpointC"
Write-Host " Transitions Cache:   $transitionsCache"
Write-Host " Calibration Report:  $calibrationJson"
Write-Host "==============================================================================" -ForegroundColor Cyan
