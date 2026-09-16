[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot
)

$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
$sourceRoot = Join-Path $repoRoot 'src'
$env:PYTHONPATH = $sourceRoot
$annotationsRoot = Join-Path $DatasetRoot 'compressed\annotations'
$compressedRoot = Join-Path $DatasetRoot 'compressed'
$partitionRoot = Join-Path $repoRoot 'artifacts\partitions\seed20260803'
$completeRoot = Join-Path $repoRoot 'artifacts\metadata_complete'
$yoloRoot = Join-Path $DatasetRoot 'yolo'

& python (Join-Path $PSScriptRoot 'validate_coco.py') `
    --annotations (Join-Path $annotationsRoot 'instances_train.json')
if ($LASTEXITCODE -ne 0) { throw 'Training annotation validation failed' }

& python (Join-Path $PSScriptRoot 'validate_coco.py') `
    --annotations (Join-Path $annotationsRoot 'instances_val.json')
if ($LASTEXITCODE -ne 0) { throw 'Validation annotation validation failed' }

& python (Join-Path $PSScriptRoot 'partition_training_coco.py') `
    --annotations (Join-Path $annotationsRoot 'instances_train.json') `
    --output-dir $partitionRoot
if ($LASTEXITCODE -ne 0) { throw 'Training partitioning failed' }

& python (Join-Path $PSScriptRoot 'filter_metadata_complete_coco.py') `
    --annotations (Join-Path $partitionRoot 'instances_calibration_fit.json') `
    --output (Join-Path $completeRoot 'instances_calibration_fit.json')
if ($LASTEXITCODE -ne 0) { throw 'Calibration metadata filtering failed' }

& python (Join-Path $PSScriptRoot 'filter_metadata_complete_coco.py') `
    --annotations (Join-Path $partitionRoot 'instances_policy_tune.json') `
    --output (Join-Path $completeRoot 'instances_policy_tune.json')
if ($LASTEXITCODE -ne 0) { throw 'Policy metadata filtering failed' }

& python (Join-Path $PSScriptRoot 'filter_metadata_complete_coco.py') `
    --annotations (Join-Path $annotationsRoot 'instances_val.json') `
    --output (Join-Path $completeRoot 'instances_official_val.json')
if ($LASTEXITCODE -ne 0) { throw 'Official validation metadata filtering failed' }

& python (Join-Path $PSScriptRoot 'prepare_yolo_dataset.py') `
    --train-annotations (Join-Path $annotationsRoot 'instances_train.json') `
    --val-annotations (Join-Path $annotationsRoot 'instances_val.json') `
    --dataset-root $compressedRoot `
    --output-yaml (Join-Path $yoloRoot 'seadronessee_odv2.yaml') `
    --partition-dir $partitionRoot
if ($LASTEXITCODE -ne 0) { throw 'YOLO data preparation failed' }
