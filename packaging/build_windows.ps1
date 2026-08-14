param(
    [string]$Version = "0.3.1-alpha"
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$AppName = "BeamXP"
$DistExe = Join-Path $Root "dist\$AppName.exe"
$StageDir = Join-Path $Root "dist\BeamXP"
$StageExe = Join-Path $StageDir "$AppName.exe"
$ReleaseDir = Join-Path $Root "release"
$ReleaseZip = Join-Path $ReleaseDir "BeamXP-$Version-windows.zip"

Set-Location $Root

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install "pyinstaller>=6.0.0"

python -m PyInstaller --noconfirm --clean ".\packaging\BeamXP.spec"

if (!(Test-Path $DistExe)) {
    throw "Expected PyInstaller output not found: $DistExe"
}

if (Test-Path $StageDir) {
    $ResolvedRoot = (Resolve-Path $Root).Path
    $ResolvedStage = (Resolve-Path $StageDir).Path
    if (!$ResolvedStage.StartsWith($ResolvedRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove staging directory outside repository root: $ResolvedStage"
    }
    Remove-Item -LiteralPath $ResolvedStage -Recurse -Force
}
New-Item -ItemType Directory -Path $StageDir -Force | Out-Null

Copy-Item -LiteralPath $DistExe -Destination $StageExe -Force
Copy-Item -LiteralPath ".\README.md" -Destination $StageDir -Force
Copy-Item -LiteralPath ".\LICENSE" -Destination $StageDir -Force

$StageExamples = Join-Path $StageDir "examples\conversion_configs"
New-Item -ItemType Directory -Path $StageExamples -Force | Out-Null
Copy-Item -Path ".\examples\conversion_configs\*.json" -Destination $StageExamples -Force

New-Item -ItemType Directory -Path $ReleaseDir -Force | Out-Null
if (Test-Path $ReleaseZip) {
    Remove-Item -LiteralPath $ReleaseZip -Force
}
Compress-Archive -LiteralPath $StageDir -DestinationPath $ReleaseZip -CompressionLevel Optimal

# Unzipped copy for local use (git-ignored; only the zip is committed). The
# zip above is the real artifact, so a locked folder (e.g. the exe is still
# running from it) must not fail the build.
$ReleaseFolder = Join-Path $ReleaseDir "BeamXP-$Version-windows"
try {
    if (Test-Path $ReleaseFolder) {
        Remove-Item -LiteralPath $ReleaseFolder -Recurse -Force -ErrorAction Stop
    }
    Copy-Item -LiteralPath $StageDir -Destination $ReleaseFolder -Recurse -Force -ErrorAction Stop
    $FolderNote = $ReleaseFolder
} catch {
    $FolderNote = "SKIPPED (in use?): $($_.Exception.Message)"
}

Write-Host "Built release archive:"
Write-Host $ReleaseZip
Write-Host "Unzipped folder version:"
Write-Host $FolderNote
