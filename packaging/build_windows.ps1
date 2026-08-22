param(
    [string]$Version = "0.3.4-alpha",
    # Signs the MSIX with this certificate (Cert:\CurrentUser\My thumbprint).
    # Left empty the package is packed unsigned, which installs in developer
    # mode with: Add-AppxPackage -Path <msix> -AllowUnsigned
    [string]$SignThumbprint = ""
)

$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$AppName = "BeamXP"
$PyInstallerDir = Join-Path $Root "dist\$AppName"
$DistExe = Join-Path $PyInstallerDir "$AppName.exe"

$StageDir = Join-Path $Root "build\release-stage\$AppName"
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
    Remove-Item -LiteralPath $StageDir -Recurse -Force
}

$StageParent = Split-Path -Parent $StageDir
New-Item -ItemType Directory -Path $StageParent -Force | Out-Null

Copy-Item -LiteralPath $PyInstallerDir -Destination $StageDir -Recurse -Force

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

# --- MSIX package ---------------------------------------------------------
# No-op unless the private manifest and Assets are both present.

function Find-SdkTool {
    param([string]$Name)

    $onPath = Get-Command $Name -ErrorAction SilentlyContinue
    if ($onPath) {
        return $onPath.Source
    }

    # Newest SDK build that ships the tool for this architecture.
    $arch = if ([Environment]::Is64BitOperatingSystem) { "x64" } else { "x86" }
    $binRoot = "C:\Program Files (x86)\Windows Kits\10\bin"
    if (!(Test-Path $binRoot)) {
        return $null
    }

    $candidate = Get-ChildItem -LiteralPath $binRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^10\.' } |
        Sort-Object { [version]$_.Name } -Descending |
        ForEach-Object { Join-Path $_.FullName "$arch\$Name" } |
        Where-Object { Test-Path $_ } |
        Select-Object -First 1

    return $candidate
}

# MSIX identities take a strictly numeric four-part version, so "0.3.4-alpha"
# packages as "0.3.4.0". The revision (fourth) part must be 0 for the Store.
$MsixVersion = $null
$VersionCore = ($Version -split '[-+]')[0]
$VersionParts = @($VersionCore -split '\.')
if ($VersionParts.Count -ge 1 -and $VersionParts.Count -le 4 -and
        ($VersionParts | Where-Object { $_ -notmatch '^\d+$' }).Count -eq 0) {
    while ($VersionParts.Count -lt 4) {
        $VersionParts += "0"
    }
    $VersionParts[3] = "0"
    $MsixVersion = $VersionParts -join '.'
}

$MsixNote = $null
$ReleaseMsix = $null

if (!(Test-Path $MsixManifest) -or !(Test-Path $MsixAssets)) {
    $MsixNote = "SKIPPED (no $MsixSourceDir manifest/Assets)"
} elseif (!$MsixVersion) {
    $MsixNote = "SKIPPED (cannot derive an MSIX version from '$Version')"
} else {
    $MakeAppx = Find-SdkTool "makeappx.exe"
    if (!$MakeAppx) {
        $MsixNote = "SKIPPED (makeappx.exe not found; install the Windows 10/11 SDK)"
    } else {
        $ReleaseMsix = Join-Path $ReleaseDir "BeamXP-$($VersionParts[0..2] -join '.')-x64.msix"

        if (Test-Path $MsixStageDir) {
            Remove-Item -LiteralPath $MsixStageDir -Recurse -Force
        }
        New-Item -ItemType Directory -Path $MsixStageDir -Force | Out-Null

        # The payload is the PyInstaller output itself: the zip's extra README
        # /LICENSE/examples are not part of the installed app.
        Copy-Item -Path (Join-Path $PyInstallerDir "*") -Destination $MsixStageDir -Recurse -Force
        Copy-Item -LiteralPath $MsixAssets -Destination $MsixStageDir -Recurse -Force

        # Stamp the build version onto a copy of the manifest so the private
        # one stays a template that needs no hand-editing per release.
        [xml]$ManifestXml = Get-Content -LiteralPath $MsixManifest -Raw
        $ManifestXml.Package.Identity.Version = $MsixVersion
        $ManifestXml.Save((Join-Path $MsixStageDir "AppxManifest.xml"))

        & $MakeAppx pack /d $MsixStageDir /p $ReleaseMsix /o
        if ($LASTEXITCODE -ne 0) {
            throw "MakeAppx.exe failed with exit code $LASTEXITCODE"
        }

        $MsixNote = $ReleaseMsix

        if ($SignThumbprint) {
            $SignTool = Find-SdkTool "signtool.exe"
            if (!$SignTool) {
                throw "signtool.exe not found; cannot sign with thumbprint $SignThumbprint"
            }
            & $SignTool sign /fd SHA256 /sha1 $SignThumbprint $ReleaseMsix
            if ($LASTEXITCODE -ne 0) {
                throw "SignTool.exe failed with exit code $LASTEXITCODE"
            }
            $MsixNote = "$ReleaseMsix (signed with $SignThumbprint)"
        }
    }
}

Write-Host "Built release archive:"
Write-Host $ReleaseZip
Write-Host "Unzipped folder version:"
Write-Host $FolderNote
Write-Host "MSIX package:"
Write-Host $MsixNote
