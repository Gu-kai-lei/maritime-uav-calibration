param(
    [string]$DatasetRoot = "D:\dataset\SeaDronesSee_ODv2",
    [string]$PythonExe = "python",
    [switch]$Resume
)

$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONPATH = Join-Path $RepoRoot "src"
$Checkpoint = Join-Path $RepoRoot "runs\detect\sds_v2_yolov8n_640_seed20260803_noamp_launch_replay\weights\best.pt"
$RunDir = Join-Path $RepoRoot "runs\detect\sds_v2_yolov8n_640_seed20260803_noamp_launch_replay"
$TrainingLog = Join-Path $RepoRoot "runs\detect\noamp_launch_replay.stdout.log"
$ArtifactRoot = Join-Path $RepoRoot "artifacts\sensitivities\stable_640_noamp"
$ResultsRoot = Join-Path $RepoRoot "results\sensitivities\stable_640_noamp"
$Predictions = Join-Path $ArtifactRoot "predictions"
$Tables = Join-Path $ArtifactRoot "tables"
$CompleteCase = Join-Path $ArtifactRoot "calibration\complete_case"
$AllImages = Join-Path $ArtifactRoot "calibration\all_images"
$Analysis = Join-Path $ArtifactRoot "analysis\complete_case"
$TrainImages = Join-Path $DatasetRoot "compressed\images\train"
$ValImages = Join-Path $DatasetRoot "compressed\images\val"
$ValAll = Join-Path $DatasetRoot "compressed\annotations\instances_val.json"
$CalibrationAnnotations = Join-Path $RepoRoot "artifacts\metadata_complete\instances_calibration_fit.json"
$PolicyAnnotations = Join-Path $RepoRoot "artifacts\metadata_complete\instances_policy_tune.json"
$ValComplete = Join-Path $RepoRoot "artifacts\metadata_complete\instances_official_val.json"

if ((Test-Path -LiteralPath $ArtifactRoot) -and -not $Resume) {
    throw "$ArtifactRoot already exists. Use -Resume only after inspecting the retained outputs."
}

$Required = @(
    $Checkpoint,
    $TrainingLog,
    $TrainImages,
    $ValImages,
    $ValAll,
    $CalibrationAnnotations,
    $PolicyAnnotations,
    $ValComplete
)
foreach ($Path in $Required) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required input not found: $Path"
    }
}

$ExpectedCheckpoint = "cccc90e9303a37ddc8131288f59b68547c0e05c696d9472fee4f650d16ae5321"
$ActualCheckpoint = (Get-FileHash -Algorithm SHA256 -LiteralPath $Checkpoint).Hash.ToLowerInvariant()
if ($ActualCheckpoint -ne $ExpectedCheckpoint) {
    throw "Checkpoint SHA-256 mismatch: expected $ExpectedCheckpoint, got $ActualCheckpoint"
}

function Invoke-PythonStep {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$Arguments)
    & $PythonExe @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python step failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

New-Item -ItemType Directory -Force -Path $Predictions, $Tables, $CompleteCase, $AllImages, $Analysis, $ResultsRoot | Out-Null

Invoke-PythonStep scripts\export_predictions.py --model $Checkpoint --annotations $CalibrationAnnotations --image-root $TrainImages --output (Join-Path $Predictions "calibration_fit.json") --imgsz 640 --conf 0.001 --iou 0.70 --device 0 --batch 8
Invoke-PythonStep scripts\export_predictions.py --model $Checkpoint --annotations $PolicyAnnotations --image-root $TrainImages --output (Join-Path $Predictions "policy_tune.json") --imgsz 640 --conf 0.001 --iou 0.70 --device 0 --batch 8
Invoke-PythonStep scripts\export_predictions.py --model $Checkpoint --annotations $ValComplete --image-root $ValImages --output (Join-Path $Predictions "official_val.json") --imgsz 640 --conf 0.001 --iou 0.70 --device 0 --batch 8
Invoke-PythonStep scripts\export_predictions.py --model $Checkpoint --annotations $ValAll --image-root $ValImages --output (Join-Path $Predictions "official_val_all_images.json") --imgsz 640 --conf 0.001 --iou 0.70 --device 0 --batch 8

Invoke-PythonStep scripts\build_detection_table.py --annotations $CalibrationAnnotations --predictions (Join-Path $Predictions "calibration_fit.json") --output (Join-Path $Tables "calibration_fit.csv") --iou 0.50
Invoke-PythonStep scripts\build_detection_table.py --annotations $PolicyAnnotations --predictions (Join-Path $Predictions "policy_tune.json") --output (Join-Path $Tables "policy_tune.csv") --iou 0.50
Invoke-PythonStep scripts\build_detection_table.py --annotations $ValComplete --predictions (Join-Path $Predictions "official_val.json") --output (Join-Path $Tables "official_val.csv") --iou 0.50
Invoke-PythonStep scripts\build_detection_table.py --annotations $ValAll --predictions (Join-Path $Predictions "official_val_all_images.json") --output (Join-Path $Tables "official_val_all_images.csv") --iou 0.50

Invoke-PythonStep scripts\fit_calibrators.py --calibration-table (Join-Path $Tables "calibration_fit.csv") --policy-table (Join-Path $Tables "policy_tune.csv") --evaluation-table (Join-Path $Tables "official_val.csv") --output-dir $CompleteCase --bins 15 --analysis complete-case
Invoke-PythonStep scripts\evaluate_operating_points.py --policy-table (Join-Path $CompleteCase "policy_probabilities.csv") --policy-summary (Join-Path $Tables "policy_tune.summary.json") --evaluation-table (Join-Path $CompleteCase "evaluation_probabilities.csv") --evaluation-summary (Join-Path $Tables "official_val.summary.json") --output (Join-Path $CompleteCase "operating_points.csv") --fp-per-image 0.1 0.25 0.5 1.0

Invoke-PythonStep scripts\fit_calibrators.py --calibration-table (Join-Path $Tables "calibration_fit.csv") --policy-table (Join-Path $Tables "policy_tune.csv") --evaluation-table (Join-Path $Tables "official_val_all_images.csv") --output-dir $AllImages --bins 15 --analysis all-images
Invoke-PythonStep scripts\evaluate_operating_points.py --policy-table (Join-Path $AllImages "policy_probabilities.csv") --policy-summary (Join-Path $Tables "policy_tune.summary.json") --evaluation-table (Join-Path $AllImages "evaluation_probabilities.csv") --evaluation-summary (Join-Path $Tables "official_val_all_images.summary.json") --output (Join-Path $AllImages "operating_points.csv") --fp-per-image 0.1 0.25 0.5 1.0

Invoke-PythonStep scripts\analyze_evaluation_slices.py --probabilities (Join-Path $CompleteCase "evaluation_probabilities.csv") --annotations $ValComplete --calibration-annotations $CalibrationAnnotations --policy-annotations $PolicyAnnotations --operating-points (Join-Path $CompleteCase "operating_points.csv") --output-dir $Analysis --bins 15
Invoke-PythonStep scripts\bootstrap_calibration_metrics.py --table (Join-Path $CompleteCase "evaluation_probabilities.csv") --output (Join-Path $Analysis "bootstrap_metric_differences.csv") --group-column source_group --reference prob_raw --repetitions 1000 --seed 20260803 --bins 15

Invoke-PythonStep scripts\build_public_results.py --run-dir $RunDir --training-log $TrainingLog --complete-case-dir $CompleteCase --all-images-dir $AllImages --analysis-dir $Analysis --metadata-val-summary (Join-Path $Tables "official_val.summary.json") --all-val-summary (Join-Path $Tables "official_val_all_images.summary.json") --output-dir $ResultsRoot

$Manifest = [ordered]@{
    experiment = "stable-640-noamp-calibration-sensitivity"
    completed_at_utc = [DateTime]::UtcNow.ToString("o")
    checkpoint = $Checkpoint
    checkpoint_sha256 = $ActualCheckpoint
    confidence_floor = 0.001
    nms_iou = 0.70
    calibration_iou = 0.50
    image_size = 640
    inference_batch = 8
    official_validation_used_for_checkpoint_selection = $false
    artifact_root = $ArtifactRoot
    public_results_root = $ResultsRoot
}
$Manifest | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $ArtifactRoot "run_manifest.json") -Encoding utf8
Write-Output "Stable-detector calibration sensitivity completed: $ResultsRoot"
