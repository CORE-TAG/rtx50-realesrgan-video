param(
    [int]$Frames = 512,
    [int]$Requests = 4,
    [int]$Streams = 2,
    [int]$Batch = 2,
    [string]$Model = "",
    [string]$VSPipe = "C:\Program Files\SVP 4\mpv64\VSPipe.exe",
    [string]$SvpRife = "C:\Program Files\SVP 4\rife",
    [string]$SvpTrt = "",
    [string]$TrtDllPath = "",
    [string]$EngineCache = "",
    [switch]$UseTrt1016Runtime,
    [int]$MaxAuxStreams = -999,
    [int]$MaxTactics = -999,
    [int]$TilingOptimizationLevel = 0,
    [int]$L2LimitForTiling = -1,
    [int]$Cublas = 1,
    [int]$Cudnn = 1,
    [int]$EdgeMaskConvolutions = 1,
    [int]$JitConvolutions = 1
)

$ErrorActionPreference = "Stop"

$ScriptRoot = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptRoot
$Script = Join-Path $ProjectRoot "vapoursynth\benchmark_vsmlrt_realesrgan_x2.vpy"

if ([string]::IsNullOrWhiteSpace($Model)) {
    $Model = Join-Path $ProjectRoot "models\realesr-animevideov3-x2-fusedtail.onnx"
}
if ([string]::IsNullOrWhiteSpace($SvpTrt)) {
    $SvpTrt = Join-Path $SvpRife "vsmlrt-cuda"
}
if ([string]::IsNullOrWhiteSpace($EngineCache)) {
    $EngineCache = Join-Path $ProjectRoot ".cache\vsmlrt_engine_cache"
}

if ($UseTrt1016Runtime) {
    if ([string]::IsNullOrWhiteSpace($TrtDllPath)) {
        $TrtDllPath = Join-Path $ProjectRoot "tools\trt1016_venv\Lib\site-packages\tensorrt_libs"
    }
}
if (-not [string]::IsNullOrWhiteSpace($TrtDllPath)) {
    $env:TRT_DLL_PATH = $TrtDllPath
} else {
    Remove-Item Env:TRT_DLL_PATH -ErrorAction SilentlyContinue
}
$env:VSMLRT_SVP_RIFE = $SvpRife
$env:VSMLRT_SVP_TRT = $SvpTrt
$env:VSMLRT_ENGINE_CACHE = $EngineCache

foreach ($Path in @($VSPipe, $Script, $Model, $SvpRife, $SvpTrt)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path not found: $Path"
    }
}

$ArgsList = @(
    "--requests", $Requests,
    "--arg", "model=$Model",
    "--arg", "batch=$Batch",
    "--arg", "streams=$Streams",
    "--arg", "frames=$Frames",
    "--arg", "width=1920",
    "--arg", "height=1080",
    "--arg", "fp16io=1",
    "--arg", "cublas=$Cublas",
    "--arg", "cudnn=$Cudnn",
    "--arg", "edge_mask_convolutions=$EdgeMaskConvolutions",
    "--arg", "jit_convolutions=$JitConvolutions"
)

if ($MaxAuxStreams -ne -999) {
    $ArgsList += @("--arg", "max_aux_streams=$MaxAuxStreams")
}
if ($MaxTactics -ne -999) {
    $ArgsList += @("--arg", "max_tactics=$MaxTactics")
}
if ($TilingOptimizationLevel -ne 0) {
    $ArgsList += @("--arg", "tiling_optimization_level=$TilingOptimizationLevel")
    $ArgsList += @("--arg", "l2_limit_for_tiling=$L2LimitForTiling")
}
$ArgsList += @($Script, "NUL")

$OldErrorActionPreference = $ErrorActionPreference
$ErrorActionPreference = "Continue"
try {
    $VSPipeOutput = & $VSPipe @ArgsList 2>&1
    $VSPipeExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $OldErrorActionPreference
}

$VSPipeOutput | ForEach-Object {
    if ($_ -is [System.Management.Automation.ErrorRecord]) {
        $_.Exception.Message
    } else {
        $_
    }
}

if ($VSPipeExitCode -ne 0) {
    throw "VSPipe failed with exit code $VSPipeExitCode"
}
