[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$SourceImages,

    [Parameter(Mandatory = $true)]
    [string]$SourceLabels,

    [Parameter(Mandatory = $true)]
    [string]$OutputRoot
)

$ErrorActionPreference = 'Stop'
$sourceImagePath = (Resolve-Path -LiteralPath $SourceImages).Path
$sourceLabelPath = (Resolve-Path -LiteralPath $SourceLabels).Path
$outputPath = [System.IO.Path]::GetFullPath($OutputRoot)
$imageParent = Join-Path $outputPath 'images'
$imageLink = Join-Path $imageParent 'train'
$labelTarget = Join-Path $outputPath 'labels\train'

New-Item -ItemType Directory -Force -Path $imageParent, $labelTarget | Out-Null
if (Test-Path -LiteralPath $imageLink) {
    $existing = Get-Item -Force -LiteralPath $imageLink
    if ($existing.LinkType -ne 'Junction') {
        throw "Existing image path is not a junction: $imageLink"
    }
    $existingTarget = [System.IO.Path]::GetFullPath(($existing.Target -join ''))
    if ($existingTarget -ne [System.IO.Path]::GetFullPath($sourceImagePath)) {
        throw "Existing junction points to a different source: $existingTarget"
    }
}
else {
    New-Item -ItemType Junction -Path $imageLink -Target $sourceImagePath | Out-Null
}

Copy-Item -Path (Join-Path $sourceLabelPath '*.txt') -Destination $labelTarget
$sourceNames = Get-ChildItem -File $sourceLabelPath -Filter *.txt | Select-Object -ExpandProperty Name
$copiedNames = Get-ChildItem -File $labelTarget -Filter *.txt | Select-Object -ExpandProperty Name
$difference = Compare-Object $sourceNames $copiedNames
if ($difference) {
    throw "Copied label set differs from source label set"
}

$manifest = [ordered]@{
    source_images = $sourceImagePath
    source_labels = $sourceLabelPath
    output_root = $outputPath
    image_access = 'junction'
    copied_label_files = $copiedNames.Count
    source_dataset_modified = $false
}
$manifestPath = Join-Path $outputPath 'wrapper_manifest.json'
$manifest | ConvertTo-Json | Set-Content -LiteralPath $manifestPath -Encoding utf8
$manifest | ConvertTo-Json
