[CmdletBinding()]
param(
    [string]$TargetRoot = 'D:\dataset\SeaDronesSee_ODv2',
    [ValidateRange(1, 32)]
    [int]$Workers = 8
)

$ErrorActionPreference = 'Stop'
$resolvedRoot = [System.IO.Path]::GetFullPath($TargetRoot)
$pythonScript = Join-Path $PSScriptRoot 'download_official_images.py'
& python $pythonScript --target-root $resolvedRoot --workers $Workers
if ($LASTEXITCODE -ne 0) {
    throw "Image downloader failed with exit code $LASTEXITCODE"
}
