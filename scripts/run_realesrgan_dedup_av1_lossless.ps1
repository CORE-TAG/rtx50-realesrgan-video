param(
    [Parameter(Mandatory=$true)]
    [string]$InputDir,
    [Parameter(Mandatory=$true)]
    [string]$OutputDir,
    [string]$FilePattern = "*.mkv",
    [int]$Scale = 2,
    [string]$ModelName = "realesr-animevideov3",
    [int]$ChunkSeconds = 120,
    [int]$GpuId = 0,
    [int]$RealEsrganTile = 2048,
    [string]$RealEsrganJobs = "24:12:24",
    [string]$MpdecimateOptions = "hi=768:lo=320:frac=0.33",
    [ValidateSet("libaom-lossless", "nvenc-lossless", "nvenc-highest")]
    [string]$Av1Mode = "libaom-lossless",
    [int]$LibaomThreads = 32,
    [int]$LibaomCpuUsed = 8,
    [string]$ToolsRoot = "",
    [string]$WorkRoot = "",
    [int]$MaxEpisodes = 0,
    [int]$MaxChunksPerEpisode = 0,
    [string]$OnlyNameLike = "",
    [switch]$RealEsrganVerbose,
    [switch]$KeepFrames,
    [switch]$KeepChunkVideos
)

$ErrorActionPreference = "Stop"
try {
    (Get-Process -Id $PID).PriorityClass = "High"
} catch {
    Write-Warning "Could not set process priority to High: $($_.Exception.Message)"
}

$Invariant = [Globalization.CultureInfo]::InvariantCulture
$ScriptRoot = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptRoot
if ([string]::IsNullOrWhiteSpace($ToolsRoot)) {
    $ToolsRoot = Join-Path $ProjectRoot "tools"
}
$DefaultWorkRoot = Join-Path $ProjectRoot "work\realesrgan_dedup_av1_lossless"
if ([string]::IsNullOrWhiteSpace($WorkRoot)) {
    $WorkRoot = $DefaultWorkRoot
}
$Ffmpeg = Join-Path $ToolsRoot "ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
$Ffprobe = Join-Path $ToolsRoot "ffmpeg\ffmpeg-8.1.1-essentials_build\bin\ffprobe.exe"
$RealEsrgan = Join-Path $ToolsRoot "realesrgan\realesrgan-ncnn-vulkan.exe"
$RealEsrganModels = Join-Path $ToolsRoot "realesrgan\models"

foreach ($Path in @($InputDir, $Ffmpeg, $Ffprobe, $RealEsrgan, $RealEsrganModels)) {
    if (-not (Test-Path -LiteralPath $Path)) {
        throw "Required path not found: $Path"
    }
}

New-Item -ItemType Directory -Force -Path $OutputDir, $WorkRoot | Out-Null
$WorkRootFull = [IO.Path]::GetFullPath((Resolve-Path -LiteralPath $WorkRoot).Path)

function Format-Number([double]$Value) {
    return $Value.ToString("0.########", $Invariant)
}

function Convert-RateToDouble([string]$Rate) {
    if ([string]::IsNullOrWhiteSpace($Rate) -or $Rate -eq "0/0") {
        throw "Invalid frame rate: $Rate"
    }
    if ($Rate.Contains("/")) {
        $Parts = $Rate.Split("/")
        return [double]::Parse($Parts[0], $Invariant) / [double]::Parse($Parts[1], $Invariant)
    }
    return [double]::Parse($Rate, $Invariant)
}

function Round-Frame([double]$Seconds, [double]$Fps) {
    return [int][Math]::Floor(($Seconds * $Fps) + 0.5)
}

function Assert-UnderWorkRoot([string]$Path) {
    $Full = [IO.Path]::GetFullPath($Path)
    if (-not $Full.StartsWith($WorkRootFull, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to delete outside work root: $Full"
    }
}

function Reset-WorkDirectory([string]$Path) {
    Assert-UnderWorkRoot $Path
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
}

function Remove-WorkDirectory([string]$Path) {
    Assert-UnderWorkRoot $Path
    if (Test-Path -LiteralPath $Path) {
        Remove-Item -LiteralPath $Path -Recurse -Force
    }
}

function Quote-CommandArg([string]$Arg) {
    if ($Arg -match '[\s"]') {
        return '"' + ($Arg -replace '"', '\"') + '"'
    }
    return $Arg
}

function Invoke-Logged([string]$Exe, [string[]]$ArgumentList, [string]$LogPath) {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $LogPath) | Out-Null
    $CommandLine = (Quote-CommandArg $Exe) + " " + (($ArgumentList | ForEach-Object { Quote-CommandArg $_ }) -join " ")
    Add-Content -LiteralPath $LogPath -Value ""
    Add-Content -LiteralPath $LogPath -Value "===== $(Get-Date -Format o) ====="
    Add-Content -LiteralPath $LogPath -Value $CommandLine
    $TempBase = Join-Path (Split-Path -Parent $LogPath) ([IO.Path]::GetRandomFileName())
    $StdOutPath = "$TempBase.stdout.tmp"
    $StdErrPath = "$TempBase.stderr.tmp"
    $ArgumentText = ($ArgumentList | ForEach-Object { Quote-CommandArg $_ }) -join " "
    $Process = Start-Process -FilePath $Exe -ArgumentList $ArgumentText -NoNewWindow -Wait -PassThru -RedirectStandardOutput $StdOutPath -RedirectStandardError $StdErrPath
    foreach ($OutputPath in @($StdOutPath, $StdErrPath)) {
        if (Test-Path -LiteralPath $OutputPath) {
            $Text = Get-Content -LiteralPath $OutputPath -Raw -ErrorAction SilentlyContinue
            if (-not [string]::IsNullOrEmpty($Text)) {
                Add-Content -LiteralPath $LogPath -Value ($Text -replace "`0", "")
            }
            Remove-Item -LiteralPath $OutputPath -Force -ErrorAction SilentlyContinue
        }
    }
    if ($Process.ExitCode -ne 0) {
        throw "Command failed with exit code $($Process.ExitCode). See log: $LogPath"
    }
}

function Get-FrameCount([string]$Directory) {
    if (-not (Test-Path -LiteralPath $Directory)) {
        return 0
    }
    return (Get-ChildItem -LiteralPath $Directory -Filter "*.png" -File | Measure-Object).Count
}

function Test-PngComplete([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $false
    }
    $Item = Get-Item -LiteralPath $Path -ErrorAction SilentlyContinue
    if ((-not $Item) -or ($Item.Length -lt 20)) {
        return $false
    }
    $Stream = [IO.File]::Open($Path, [IO.FileMode]::Open, [IO.FileAccess]::Read, [IO.FileShare]::ReadWrite)
    try {
        $Header = New-Object byte[] 8
        [void]$Stream.Read($Header, 0, 8)
        $ExpectedHeader = [byte[]](137,80,78,71,13,10,26,10)
        for ($Index = 0; $Index -lt 8; $Index++) {
            if ($Header[$Index] -ne $ExpectedHeader[$Index]) {
                return $false
            }
        }
        [void]$Stream.Seek(-12, [IO.SeekOrigin]::End)
        $Footer = New-Object byte[] 12
        [void]$Stream.Read($Footer, 0, 12)
        $ExpectedFooter = [byte[]](0,0,0,0,73,69,78,68,174,66,96,130)
        for ($Index = 0; $Index -lt 12; $Index++) {
            if ($Footer[$Index] -ne $ExpectedFooter[$Index]) {
                return $false
            }
        }
        return $true
    }
    finally {
        $Stream.Dispose()
    }
}

function Write-ConcatList([string[]]$Files, [string]$ListPath) {
    $Lines = foreach ($File in $Files) {
        $Escaped = $File.Replace("'", "'\''")
        "file '$Escaped'"
    }
    Set-Content -LiteralPath $ListPath -Value $Lines -Encoding ASCII
}

function Get-VideoInfo([string]$VideoPath) {
    $JsonText = & $Ffprobe -v error -select_streams v:0 -show_entries format=duration:stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:stream_tags=NUMBER_OF_FRAMES,DURATION -of json $VideoPath
    if ($LASTEXITCODE -ne 0) {
        throw "ffprobe failed for $VideoPath"
    }
    $Json = $JsonText | ConvertFrom-Json
    $Stream = $Json.streams | Select-Object -First 1
    $RateString = $Stream.avg_frame_rate
    if ([string]::IsNullOrWhiteSpace($RateString) -or $RateString -eq "0/0") {
        $RateString = $Stream.r_frame_rate
    }
    $Fps = Convert-RateToDouble $RateString
    $Duration = [double]::Parse($Json.format.duration, $Invariant)
    $ExpectedFrames = Round-Frame $Duration $Fps
    $FrameCount = 0
    if ($Stream.tags -and $Stream.tags.NUMBER_OF_FRAMES) {
        $FrameCount = [int]$Stream.tags.NUMBER_OF_FRAMES
    } elseif ($Stream.nb_frames) {
        $FrameCount = [int]$Stream.nb_frames
    } else {
        $FrameCount = $ExpectedFrames
    }
    if (($ExpectedFrames -gt 0) -and ([Math]::Abs($FrameCount - $ExpectedFrames) -gt [Math]::Max(2, [int]($ExpectedFrames * 0.05)))) {
        $FrameCount = $ExpectedFrames
    }
    return [pscustomobject]@{
        Width = [int]$Stream.width
        Height = [int]$Stream.height
        RateString = $RateString
        Fps = $Fps
        Duration = $Duration
        FrameCount = $FrameCount
    }
}

function Get-KeptFrameIndices([string]$LogPath, [double]$Fps, [int]$TargetFrames) {
    $Indices = New-Object System.Collections.Generic.List[int]
    foreach ($Line in Get-Content -LiteralPath $LogPath) {
        $CleanLine = $Line -replace "`0", ""
        if ($CleanLine -match 'pts_time:([0-9]+(?:\.[0-9]+)?)') {
            $PtsTime = [double]::Parse($Matches[1], $Invariant)
            $Index = Round-Frame $PtsTime $Fps
            if ($Index -lt 0) {
                $Index = 0
            }
            if ($Index -ge $TargetFrames) {
                $Index = $TargetFrames - 1
            }
            if (($Indices.Count -eq 0) -or ($Indices[$Indices.Count - 1] -ne $Index)) {
                $Indices.Add($Index)
            }
        }
    }
    if ($Indices.Count -eq 0) {
        throw "Could not parse kept frame timestamps from $LogPath"
    }
    return $Indices.ToArray()
}

function Complete-SrUniqueFrames(
    [string]$ChunkName,
    [string]$UniqueDir,
    [string]$SrDir,
    [string]$PendingDir,
    [int]$UniqueFrames,
    [string]$ChunkLog,
    [string]$MainLog
) {
    New-Item -ItemType Directory -Force -Path $SrDir | Out-Null
    $ValidExistingFrames = 0
    $RemovedBadFrames = 0
    for ($Frame = 1; $Frame -le $UniqueFrames; $Frame++) {
        $Name = "{0:D8}.png" -f $Frame
        $SrFrame = Join-Path $SrDir $Name
        if (Test-Path -LiteralPath $SrFrame) {
            if (Test-PngComplete $SrFrame) {
                $ValidExistingFrames++
            }
            else {
                Remove-Item -LiteralPath $SrFrame -Force
                $RemovedBadFrames++
            }
        }
    }
    if ($RemovedBadFrames -gt 0) {
        Add-Content -LiteralPath $MainLog -Value "[$ChunkName] removedBadSrFrames=$RemovedBadFrames"
    }
    if ($ValidExistingFrames -eq $UniqueFrames) {
        Add-Content -LiteralPath $MainLog -Value "[$ChunkName] srUniqueFrames=$ValidExistingFrames, skipping RealESRGAN."
        return
    }

    Reset-WorkDirectory $PendingDir
    $MissingFrames = 0
    for ($Frame = 1; $Frame -le $UniqueFrames; $Frame++) {
        $Name = "{0:D8}.png" -f $Frame
        $SrFrame = Join-Path $SrDir $Name
        if (-not (Test-Path -LiteralPath $SrFrame)) {
            $UniqueFrame = Join-Path $UniqueDir $Name
            if (-not (Test-Path -LiteralPath $UniqueFrame)) {
                throw "[$ChunkName] Missing unique frame: $UniqueFrame"
            }
            New-Item -ItemType HardLink -Path (Join-Path $PendingDir $Name) -Target $UniqueFrame | Out-Null
            $MissingFrames++
        }
    }

    Add-Content -LiteralPath $MainLog -Value "[$ChunkName] srUniqueFrames=$ValidExistingFrames, missingSrUniqueFrames=$MissingFrames"
    if ($MissingFrames -gt 0) {
        $SrArgs = @(
            "-i", $PendingDir,
            "-o", $SrDir,
            "-m", $RealEsrganModels,
            "-n", $ModelName,
            "-s", "$Scale",
            "-t", "$RealEsrganTile",
            "-g", "$GpuId",
            "-j", $RealEsrganJobs,
            "-f", "png"
        )
        if ($RealEsrganVerbose) {
            $SrArgs += "-v"
        }
        Invoke-Logged $RealEsrgan $SrArgs $ChunkLog
    }
    Remove-WorkDirectory $PendingDir
}

function Rebuild-FullFrameSequence(
    [string]$SrDir,
    [string]$FullDir,
    [int[]]$KeptIndices,
    [int]$TargetFrames,
    [string]$MapPath
) {
    Reset-WorkDirectory $FullDir
    $Lines = New-Object System.Collections.Generic.List[string]
    $Lines.Add("frame,unique_frame,kept_source_frame")
    $UniqueCursor = 0
    for ($Frame = 0; $Frame -lt $TargetFrames; $Frame++) {
        while ((($UniqueCursor + 1) -lt $KeptIndices.Count) -and ($KeptIndices[$UniqueCursor + 1] -le $Frame)) {
            $UniqueCursor++
        }
        $UniqueNumber = $UniqueCursor + 1
        $Source = Join-Path $SrDir ("{0:D8}.png" -f $UniqueNumber)
        $Dest = Join-Path $FullDir ("{0:D8}.png" -f ($Frame + 1))
        if (-not (Test-Path -LiteralPath $Source)) {
            throw "Missing SR frame for rebuild: $Source"
        }
        New-Item -ItemType HardLink -Path $Dest -Target $Source | Out-Null
        $Lines.Add(("{0},{1},{2}" -f ($Frame + 1), $UniqueNumber, ($KeptIndices[$UniqueCursor] + 1)))
    }
    Set-Content -LiteralPath $MapPath -Value $Lines -Encoding ASCII
}

function Get-EncodeArgs([string]$RateString, [string]$FullDir, [string]$ChunkVideo) {
    $InputPattern = Join-Path $FullDir "%08d.png"
    if ($Av1Mode -eq "libaom-lossless") {
        return @(
            "-y", "-hide_banner",
            "-framerate", $RateString,
            "-i", $InputPattern,
            "-c:v", "libaom-av1",
            "-crf", "0",
            "-b:v", "0",
            "-cpu-used", "$LibaomCpuUsed",
            "-row-mt", "1",
            "-threads", "$LibaomThreads",
            "-tile-columns", "2",
            "-tile-rows", "1",
            "-pix_fmt", "yuv444p10le",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-color_range", "tv",
            "-an",
            $ChunkVideo
        )
    }
    if ($Av1Mode -eq "nvenc-highest") {
        return @(
            "-y", "-hide_banner",
            "-framerate", $RateString,
            "-i", $InputPattern,
            "-c:v", "av1_nvenc",
            "-preset", "p7",
            "-tune", "uhq",
            "-rc", "constqp",
            "-qp", "1",
            "-pix_fmt", "p010le",
            "-color_primaries", "bt709",
            "-color_trc", "bt709",
            "-colorspace", "bt709",
            "-color_range", "tv",
            "-an",
            $ChunkVideo
        )
    }
    return @(
        "-y", "-hide_banner",
        "-framerate", $RateString,
        "-i", $InputPattern,
        "-c:v", "av1_nvenc",
        "-preset", "p7",
        "-tune", "lossless",
        "-rc", "constqp",
        "-qp", "0",
        "-multipass", "fullres",
        "-pix_fmt", "yuv444p",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
        "-an",
        $ChunkVideo
    )
}

function Process-Episode([string]$InputVideo) {
    $Info = Get-VideoInfo $InputVideo
    $BaseName = [IO.Path]::GetFileNameWithoutExtension($InputVideo)
    $SafeJobName = $BaseName -replace "[^A-Za-z0-9._-]+", "_"
    if ([string]::IsNullOrWhiteSpace($SafeJobName)) {
        $SafeJobName = "rezero_episode"
    }
    $EpisodeWork = Join-Path $WorkRoot $SafeJobName
    $LogRoot = Join-Path $EpisodeWork "logs"
    $MainLog = Join-Path $LogRoot "pipeline.log"
    $Av1Label = if ($Av1Mode -eq "nvenc-highest") { "AV1-NVENC-highest" } elseif ($Av1Mode -eq "nvenc-lossless") { "AV1-NVENC-lossless" } else { "AV1-lossless" }
    $OutputVideo = Join-Path $OutputDir ("{0} [RealESRGAN-{1}x-dedup][{2}].mkv" -f $BaseName, $Scale, $Av1Label)
    New-Item -ItemType Directory -Force -Path $EpisodeWork, $LogRoot | Out-Null

    if (Test-Path -LiteralPath $OutputVideo) {
        Add-Content -LiteralPath $MainLog -Value "Output already exists, skipping: $OutputVideo"
        Write-Host "SKIP existing output: $OutputVideo"
        return
    }

    Add-Content -LiteralPath $MainLog -Value ""
    Add-Content -LiteralPath $MainLog -Value "===== $(Get-Date -Format o) ====="
    Add-Content -LiteralPath $MainLog -Value "Input: $InputVideo"
    Add-Content -LiteralPath $MainLog -Value "Output: $OutputVideo"
    Add-Content -LiteralPath $MainLog -Value "Source: $($Info.Width)x$($Info.Height), fps=$($Info.RateString), frames=$($Info.FrameCount), duration=$(Format-Number $Info.Duration)"
    Add-Content -LiteralPath $MainLog -Value "Output scale: ${Scale}x -> $($Info.Width * $Scale)x$($Info.Height * $Scale)"
    Add-Content -LiteralPath $MainLog -Value "Chunk seconds: $ChunkSeconds"
    Add-Content -LiteralPath $MainLog -Value "mpdecimate: $MpdecimateOptions"
    Add-Content -LiteralPath $MainLog -Value "RealESRGAN: model=$ModelName, tile=$RealEsrganTile, jobs=$RealEsrganJobs, gpu=$GpuId"
    Add-Content -LiteralPath $MainLog -Value "AV1 mode: $Av1Mode"

    $FramesPerChunk = [Math]::Max(1, (Round-Frame $ChunkSeconds $Info.Fps))
    $ChunkVideos = New-Object System.Collections.Generic.List[string]
    $ChunkIndex = 0
    for ($ChunkStartFrame = 0; $ChunkStartFrame -lt $Info.FrameCount; $ChunkStartFrame += $FramesPerChunk) {
        if (($MaxChunksPerEpisode -gt 0) -and ($ChunkIndex -ge $MaxChunksPerEpisode)) {
            Add-Content -LiteralPath $MainLog -Value "Stopping early after MaxChunksPerEpisode=$MaxChunksPerEpisode"
            break
        }
        $TargetFrames = [Math]::Min($FramesPerChunk, $Info.FrameCount - $ChunkStartFrame)
        $StartSeconds = $ChunkStartFrame / $Info.Fps
        $ChunkDuration = $TargetFrames / $Info.Fps
        $ChunkName = "chunk_{0:D4}" -f $ChunkIndex
        $ChunkDir = Join-Path $EpisodeWork $ChunkName
        $UniqueDir = Join-Path $ChunkDir "unique"
        $SrDir = Join-Path $ChunkDir "sr_unique"
        $PendingDir = Join-Path $ChunkDir "sr_pending"
        $FullDir = Join-Path $ChunkDir "full"
        $ChunkVideo = Join-Path $ChunkDir "$ChunkName.av1.mkv"
        $DoneMarker = Join-Path $ChunkDir "done.txt"
        $MapPath = Join-Path $ChunkDir "frame_map.csv"
        $ExtractLog = Join-Path $LogRoot "$ChunkName.extract_unique.log"
        $SrLog = Join-Path $LogRoot "$ChunkName.realesrgan.log"
        $EncodeLog = Join-Path $LogRoot "$ChunkName.encode.log"

        New-Item -ItemType Directory -Force -Path $ChunkDir | Out-Null
        Add-Content -LiteralPath $MainLog -Value "[$ChunkName] startFrame=$($ChunkStartFrame + 1), targetFrames=$TargetFrames, start=$(Format-Number $StartSeconds), duration=$(Format-Number $ChunkDuration)"
        $ChunkWallStart = Get-Date

        if ((Test-Path -LiteralPath $ChunkVideo) -and (Test-Path -LiteralPath $DoneMarker)) {
            Add-Content -LiteralPath $MainLog -Value "[$ChunkName] existing encoded chunk found, skipping."
            $ChunkVideos.Add($ChunkVideo)
            $ChunkIndex++
            continue
        }

        Reset-WorkDirectory $UniqueDir
        $MpFilter = if ([string]::IsNullOrWhiteSpace($MpdecimateOptions)) { "mpdecimate" } else { "mpdecimate=$MpdecimateOptions" }
        $Filter = "setpts=PTS-STARTPTS,$MpFilter,showinfo,format=rgb24"
        $ExtractArgs = @(
            "-y", "-hide_banner",
            "-threads", "0",
            "-ss", (Format-Number $StartSeconds),
            "-t", (Format-Number $ChunkDuration),
            "-i", $InputVideo,
            "-map", "0:v:0",
            "-vf", $Filter,
            "-fps_mode", "passthrough",
            (Join-Path $UniqueDir "%08d.png")
        )
        Remove-Item -LiteralPath $ExtractLog -Force -ErrorAction SilentlyContinue
        $ExtractStart = Get-Date
        Invoke-Logged $Ffmpeg $ExtractArgs $ExtractLog
        $ExtractElapsed = ((Get-Date) - $ExtractStart).TotalSeconds
        $KeptIndices = Get-KeptFrameIndices $ExtractLog $Info.Fps $TargetFrames
        $UniqueFrames = Get-FrameCount $UniqueDir
        if ($UniqueFrames -ne $KeptIndices.Count) {
            throw "[$ChunkName] unique frame count mismatch. images=$UniqueFrames showinfo=$($KeptIndices.Count)"
        }
        $ReuseFrames = $TargetFrames - $UniqueFrames
        $ReusePercent = if ($TargetFrames -gt 0) { 100.0 * $ReuseFrames / $TargetFrames } else { 0.0 }
        Add-Content -LiteralPath $MainLog -Value "[$ChunkName] uniqueFrames=$UniqueFrames, reusedFrames=$ReuseFrames ($(Format-Number $ReusePercent)%)"

        $SrStart = Get-Date
        Complete-SrUniqueFrames $ChunkName $UniqueDir $SrDir $PendingDir $UniqueFrames $SrLog $MainLog
        $SrElapsed = ((Get-Date) - $SrStart).TotalSeconds
        $SrFrames = Get-FrameCount $SrDir
        if ($SrFrames -ne $UniqueFrames) {
            throw "[$ChunkName] RealESRGAN frame count mismatch. Expected $UniqueFrames, got $SrFrames."
        }

        $RebuildStart = Get-Date
        Rebuild-FullFrameSequence $SrDir $FullDir $KeptIndices $TargetFrames $MapPath
        $RebuildElapsed = ((Get-Date) - $RebuildStart).TotalSeconds
        $FullFrames = Get-FrameCount $FullDir
        if ($FullFrames -ne $TargetFrames) {
            throw "[$ChunkName] rebuilt frame count mismatch. Expected $TargetFrames, got $FullFrames."
        }

        $EncodeArgs = Get-EncodeArgs $Info.RateString $FullDir $ChunkVideo
        $EncodeStart = Get-Date
        Invoke-Logged $Ffmpeg $EncodeArgs $EncodeLog
        $EncodeElapsed = ((Get-Date) - $EncodeStart).TotalSeconds
        Set-Content -LiteralPath $DoneMarker -Value "done $(Get-Date -Format o)" -Encoding ASCII
        $ChunkVideos.Add($ChunkVideo)

        if (-not $KeepFrames) {
            Remove-WorkDirectory $UniqueDir
            Remove-WorkDirectory $SrDir
            Remove-WorkDirectory $PendingDir
            Remove-WorkDirectory $FullDir
        }
        $ChunkElapsed = ((Get-Date) - $ChunkWallStart).TotalSeconds
        Add-Content -LiteralPath $MainLog -Value ("[$ChunkName] completed elapsedSeconds={0}, extractSeconds={1}, srSeconds={2}, rebuildSeconds={3}, encodeSeconds={4}" -f (Format-Number $ChunkElapsed), (Format-Number $ExtractElapsed), (Format-Number $SrElapsed), (Format-Number $RebuildElapsed), (Format-Number $EncodeElapsed))
        $ChunkIndex++
    }

    if (($MaxChunksPerEpisode -gt 0) -and ($ChunkVideos.Count -lt [Math]::Ceiling($Info.FrameCount / $FramesPerChunk))) {
        Write-Host "Partial run complete for testing: $BaseName"
        return
    }

    $ConcatList = Join-Path $EpisodeWork "chunks.txt"
    Write-ConcatList $ChunkVideos.ToArray() $ConcatList

    $MuxArgs = @(
        "-y", "-hide_banner",
        "-f", "concat",
        "-safe", "0",
        "-i", $ConcatList,
        "-i", $InputVideo,
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-map", "1:t?",
        "-map_metadata", "1",
        "-map_chapters", "1",
        "-c", "copy",
        $OutputVideo
    )
    Invoke-Logged $Ffmpeg $MuxArgs (Join-Path $LogRoot "mux.log")

    $ProbeArgs = @(
        "-hide_banner",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,r_frame_rate,avg_frame_rate,pix_fmt,color_space,color_transfer,color_primaries",
        "-of", "default=nw=1",
        $OutputVideo
    )
    Invoke-Logged $Ffprobe $ProbeArgs (Join-Path $LogRoot "verify_video.log")

    $StreamProbeArgs = @(
        "-hide_banner",
        "-v", "error",
        "-show_entries", "stream=index,codec_type,codec_name:stream_tags=language,title,filename,mimetype",
        "-of", "json",
        $OutputVideo
    )
    Invoke-Logged $Ffprobe $StreamProbeArgs (Join-Path $LogRoot "verify_streams.json")
    Add-Content -LiteralPath $MainLog -Value "Completed: $OutputVideo"
    Write-Host "DONE: $OutputVideo"

    if (-not $KeepChunkVideos) {
        foreach ($ChunkVideoPath in $ChunkVideos) {
            $Dir = Split-Path -Parent $ChunkVideoPath
            Remove-WorkDirectory $Dir
        }
    }
}

$InputFiles = Get-ChildItem -LiteralPath $InputDir -Filter $FilePattern -File | Sort-Object Name
if (-not [string]::IsNullOrWhiteSpace($OnlyNameLike)) {
    $InputFiles = $InputFiles | Where-Object { $_.Name -like $OnlyNameLike }
}
if ($MaxEpisodes -gt 0) {
    $InputFiles = $InputFiles | Select-Object -First $MaxEpisodes
}
if (($InputFiles | Measure-Object).Count -eq 0) {
    throw "No input files matched."
}

Write-Host "Input files: $(($InputFiles | Measure-Object).Count)"
Write-Host "Output directory: $OutputDir"
Write-Host "Work directory: $WorkRoot"
Write-Host "RealESRGAN tile=$RealEsrganTile jobs=$RealEsrganJobs gpu=$GpuId"
Write-Host "AV1 mode=$Av1Mode"

foreach ($File in $InputFiles) {
    Process-Episode $File.FullName
}
