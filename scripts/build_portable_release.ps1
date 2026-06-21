param(
    [string]$SourceRoot = "",
    [string]$DependencyRoot = "",
    [string]$OutputRoot = "",
    [string]$Version = "v0.1.0",
    [switch]$NoZip
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $PSCommandPath
$RepoRoot = Split-Path -Parent $ScriptRoot
if ([string]::IsNullOrWhiteSpace($SourceRoot)) {
    $SourceRoot = $RepoRoot
}
if ([string]::IsNullOrWhiteSpace($DependencyRoot)) {
    $DependencyRoot = Split-Path -Parent (Split-Path -Parent $RepoRoot)
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path (Split-Path -Parent $RepoRoot) "release"
}

$ReleaseName = "rtx50-realesrgan-video-portable-$Version"
$ReleaseDir = Join-Path $OutputRoot $ReleaseName
$ZipPath = Join-Path $OutputRoot "$ReleaseName.zip"

if (Test-Path -LiteralPath $ReleaseDir) {
    Remove-Item -LiteralPath $ReleaseDir -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $ReleaseDir | Out-Null

$excludeDirs = @(".git", "tools", "work", "output", ".cache")
Get-ChildItem -LiteralPath $SourceRoot -Force | Where-Object {
    $excludeDirs -notcontains $_.Name
} | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination $ReleaseDir -Recurse -Force
}

$toolsDir = Join-Path $ReleaseDir "tools"
$ffmpegSrc = Join-Path $DependencyRoot "tools\ffmpeg\ffmpeg-8.1.1-essentials_build"
$realesrganSrc = Join-Path $DependencyRoot "tools\realesrgan"
foreach ($path in @($ffmpegSrc, $realesrganSrc)) {
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required dependency path not found: $path"
    }
}

$ffmpegDst = Join-Path $toolsDir "ffmpeg\ffmpeg-8.1.1-essentials_build"
New-Item -ItemType Directory -Force -Path (Join-Path $ffmpegDst "bin"), (Join-Path $ffmpegDst "presets") | Out-Null
Copy-Item -LiteralPath (Join-Path $ffmpegSrc "bin\ffmpeg.exe") -Destination (Join-Path $ffmpegDst "bin\ffmpeg.exe") -Force
Copy-Item -LiteralPath (Join-Path $ffmpegSrc "bin\ffprobe.exe") -Destination (Join-Path $ffmpegDst "bin\ffprobe.exe") -Force
Copy-Item -LiteralPath (Join-Path $ffmpegSrc "LICENSE") -Destination (Join-Path $ffmpegDst "LICENSE") -Force
Copy-Item -LiteralPath (Join-Path $ffmpegSrc "README.txt") -Destination (Join-Path $ffmpegDst "README.txt") -Force
if (Test-Path -LiteralPath (Join-Path $ffmpegSrc "presets")) {
    Copy-Item -LiteralPath (Join-Path $ffmpegSrc "presets\*") -Destination (Join-Path $ffmpegDst "presets") -Recurse -Force
}

$realesrganDst = Join-Path $toolsDir "realesrgan"
New-Item -ItemType Directory -Force -Path (Join-Path $realesrganDst "models") | Out-Null
foreach ($name in @("realesrgan-ncnn-vulkan.exe", "vcomp140.dll", "vcomp140d.dll", "README_windows.md")) {
    $src = Join-Path $realesrganSrc $name
    if (Test-Path -LiteralPath $src) {
        Copy-Item -LiteralPath $src -Destination (Join-Path $realesrganDst $name) -Force
    }
}
Copy-Item -LiteralPath (Join-Path $realesrganSrc "models\*.param") -Destination (Join-Path $realesrganDst "models") -Force
Copy-Item -LiteralPath (Join-Path $realesrganSrc "models\*.bin") -Destination (Join-Path $realesrganDst "models") -Force

$readmeFirst = @"
RTX50 RealESRGAN Video Portable $Version

Run:
  Start-GUI.bat

This portable package bundles FFmpeg and RealESRGAN-ncnn-vulkan.
It still requires a working GPU driver. NVIDIA AV1 NVENC output requires a GPU/driver with AV1 NVENC support.

The TensorRT/vs-mlrt benchmark path is included in the source tree, but this portable GUI release uses the ncnn Vulkan path for zero-install video processing.
"@
Set-Content -LiteralPath (Join-Path $ReleaseDir "README_FIRST.txt") -Value $readmeFirst -Encoding ASCII

if (-not $NoZip) {
    if (Test-Path -LiteralPath $ZipPath) {
        Remove-Item -LiteralPath $ZipPath -Force
    }
    Compress-Archive -Path (Join-Path $ReleaseDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal
}

[pscustomobject]@{
    ReleaseDir = $ReleaseDir
    ZipPath = if ($NoZip) { "" } else { $ZipPath }
}
