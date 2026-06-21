param(
    [string]$RepoName = "rtx50-realesrgan-video",
    [string]$Description = "Windows RTX 50 RealESRGAN anime video upscaling pipeline, TensorRT/vs-mlrt benchmarks, and SM120 experiments",
    [switch]$Public,
    [switch]$Private
)

$ErrorActionPreference = "Stop"

if ($Public -and $Private) {
    throw "Choose only one of -Public or -Private."
}
if (-not $Public -and -not $Private) {
    $Public = $true
}

$Gh = Get-Command gh -ErrorAction SilentlyContinue
if (-not $Gh) {
    throw "GitHub CLI is not installed or not on PATH. Install it with: winget install GitHub.cli"
}

$Git = Get-Command git -ErrorAction SilentlyContinue
if (-not $Git) {
    throw "Git is not installed or not on PATH."
}

& gh auth status | Out-Null

$Visibility = if ($Private) { "--private" } else { "--public" }
$ExistingOrigin = ""
try {
    $ExistingOrigin = (& git remote get-url origin 2>$null)
    if ($LASTEXITCODE -ne 0) {
        $ExistingOrigin = ""
    }
} catch {
    $ExistingOrigin = ""
}
if ([string]::IsNullOrWhiteSpace($ExistingOrigin)) {
    & gh repo create $RepoName $Visibility --source . --remote origin --push
} else {
    & git push -u origin main
}

& gh repo edit --description $Description

try {
    & gh repo edit --add-topic "realesrgan" --add-topic "tensorrt" --add-topic "vapoursynth" --add-topic "rtx5090" --add-topic "video-upscaling" --add-topic "av1"
} catch {
    Write-Warning "Repository pushed, but setting topics failed: $($_.Exception.Message)"
}
