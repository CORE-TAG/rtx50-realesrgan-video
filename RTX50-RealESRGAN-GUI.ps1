Add-Type -AssemblyName System.Windows.Forms
Add-Type -AssemblyName System.Drawing

$ErrorActionPreference = "Stop"
$script:Root = Split-Path -Parent $PSCommandPath
$script:Worker = $null

function New-Label($Text, $X, $Y, $W = 120, $H = 22) {
    $label = New-Object System.Windows.Forms.Label
    $label.Text = $Text
    $label.Location = New-Object System.Drawing.Point($X, $Y)
    $label.Size = New-Object System.Drawing.Size($W, $H)
    $label.TextAlign = [System.Drawing.ContentAlignment]::MiddleLeft
    return $label
}

function New-TextBox($X, $Y, $W, $Text = "") {
    $box = New-Object System.Windows.Forms.TextBox
    $box.Location = New-Object System.Drawing.Point($X, $Y)
    $box.Size = New-Object System.Drawing.Size($W, 24)
    $box.Text = $Text
    return $box
}

function New-Button($Text, $X, $Y, $W = 90, $H = 28) {
    $button = New-Object System.Windows.Forms.Button
    $button.Text = $Text
    $button.Location = New-Object System.Drawing.Point($X, $Y)
    $button.Size = New-Object System.Drawing.Size($W, $H)
    return $button
}

function Select-Folder($TextBox) {
    $dialog = New-Object System.Windows.Forms.FolderBrowserDialog
    if (Test-Path -LiteralPath $TextBox.Text) {
        $dialog.SelectedPath = $TextBox.Text
    }
    if ($dialog.ShowDialog() -eq [System.Windows.Forms.DialogResult]::OK) {
        $TextBox.Text = $dialog.SelectedPath
    }
}

function Append-Log($Text) {
    if ($null -eq $script:LogBox) {
        return
    }
    $action = {
        param($Line)
        $script:LogBox.AppendText($Line + [Environment]::NewLine)
        $script:LogBox.SelectionStart = $script:LogBox.TextLength
        $script:LogBox.ScrollToCaret()
    }
    if ($script:LogBox.InvokeRequired) {
        [void]$script:LogBox.BeginInvoke($action, $Text)
    } else {
        & $action $Text
    }
}

function Set-RunningState([bool]$Running) {
    $script:StartButton.Enabled = -not $Running
    $script:StopButton.Enabled = $Running
    $script:StatusLabel.Text = if ($Running) { "Running" } else { "Idle" }
}

function Add-Arg([System.Collections.Generic.List[string]]$List, [string]$Name, [string]$Value) {
    if (-not [string]::IsNullOrWhiteSpace($Value)) {
        $List.Add($Name)
        $List.Add($Value)
    }
}

function Quote-Arg([string]$Value) {
    if ($null -eq $Value) {
        return '""'
    }
    if ($Value -notmatch '[\s"]') {
        return $Value
    }
    return '"' + ($Value -replace '"', '\"') + '"'
}

function Start-JobProcess {
    if ($script:Worker -and -not $script:Worker.HasExited) {
        [System.Windows.Forms.MessageBox]::Show("A job is already running.", "RealESRGAN GUI") | Out-Null
        return
    }

    $inputDir = $script:InputBox.Text.Trim()
    $outputDir = $script:OutputBox.Text.Trim()
    $workRoot = $script:WorkBox.Text.Trim()
    if (-not (Test-Path -LiteralPath $inputDir)) {
        [System.Windows.Forms.MessageBox]::Show("Input folder does not exist.", "RealESRGAN GUI") | Out-Null
        return
    }
    if ([string]::IsNullOrWhiteSpace($outputDir)) {
        [System.Windows.Forms.MessageBox]::Show("Choose an output folder.", "RealESRGAN GUI") | Out-Null
        return
    }

    $runner = Join-Path $script:Root "scripts\run_realesrgan_dedup_av1_lossless.ps1"
    $tools = Join-Path $script:Root "tools"
    if (-not (Test-Path -LiteralPath $runner)) {
        [System.Windows.Forms.MessageBox]::Show("Runner script not found: $runner", "RealESRGAN GUI") | Out-Null
        return
    }
    if (-not (Test-Path -LiteralPath $tools)) {
        [System.Windows.Forms.MessageBox]::Show("Portable tools folder not found: $tools", "RealESRGAN GUI") | Out-Null
        return
    }

    New-Item -ItemType Directory -Force -Path $outputDir | Out-Null
    if ([string]::IsNullOrWhiteSpace($workRoot)) {
        $workRoot = Join-Path $outputDir "_work"
        $script:WorkBox.Text = $workRoot
    }
    New-Item -ItemType Directory -Force -Path $workRoot | Out-Null

    $args = New-Object System.Collections.Generic.List[string]
    $args.Add("-NoProfile")
    $args.Add("-ExecutionPolicy")
    $args.Add("Bypass")
    $args.Add("-File")
    $args.Add($runner)
    Add-Arg $args "-InputDir" $inputDir
    Add-Arg $args "-OutputDir" $outputDir
    Add-Arg $args "-WorkRoot" $workRoot
    Add-Arg $args "-ToolsRoot" $tools
    Add-Arg $args "-FilePattern" $script:PatternBox.Text.Trim()
    Add-Arg $args "-Scale" $script:ScaleBox.SelectedItem
    Add-Arg $args "-ModelName" $script:ModelBox.Text.Trim()
    Add-Arg $args "-ChunkSeconds" $script:ChunkBox.Value.ToString()
    Add-Arg $args "-GpuId" $script:GpuBox.Value.ToString()
    Add-Arg $args "-RealEsrganTile" $script:TileBox.Value.ToString()
    Add-Arg $args "-RealEsrganJobs" $script:JobsBox.Text.Trim()
    Add-Arg $args "-MpdecimateOptions" $script:DedupBox.Text.Trim()
    Add-Arg $args "-Av1Mode" $script:Av1Box.SelectedItem
    Add-Arg $args "-LibaomThreads" $script:LibaomThreadsBox.Value.ToString()
    Add-Arg $args "-LibaomCpuUsed" $script:LibaomCpuBox.Value.ToString()
    Add-Arg $args "-OnlyNameLike" $script:OnlyBox.Text.Trim()
    if ($script:MaxEpisodesBox.Value -gt 0) {
        Add-Arg $args "-MaxEpisodes" $script:MaxEpisodesBox.Value.ToString()
    }
    if ($script:MaxChunksBox.Value -gt 0) {
        Add-Arg $args "-MaxChunksPerEpisode" $script:MaxChunksBox.Value.ToString()
    }
    if ($script:VerboseBox.Checked) {
        $args.Add("-RealEsrganVerbose")
    }
    if ($script:KeepFramesBox.Checked) {
        $args.Add("-KeepFrames")
    }
    if ($script:KeepChunksBox.Checked) {
        $args.Add("-KeepChunkVideos")
    }

    $script:LogBox.Clear()
    Append-Log "Starting portable RealESRGAN job..."
    Append-Log "Input: $inputDir"
    Append-Log "Output: $outputDir"
    Append-Log "Work: $workRoot"

    $psi = New-Object System.Diagnostics.ProcessStartInfo
    $psi.FileName = "$env:WINDIR\System32\WindowsPowerShell\v1.0\powershell.exe"
    $psi.Arguments = (($args | ForEach-Object { Quote-Arg $_ }) -join " ")
    $psi.WorkingDirectory = $script:Root
    $psi.UseShellExecute = $false
    $psi.RedirectStandardOutput = $true
    $psi.RedirectStandardError = $true
    $psi.CreateNoWindow = $true

    $process = New-Object System.Diagnostics.Process
    $process.StartInfo = $psi
    $process.EnableRaisingEvents = $true
    Register-ObjectEvent -InputObject $process -EventName OutputDataReceived -Action {
        if ($EventArgs.Data) { Append-Log $EventArgs.Data }
    } | Out-Null
    Register-ObjectEvent -InputObject $process -EventName ErrorDataReceived -Action {
        if ($EventArgs.Data) { Append-Log $EventArgs.Data }
    } | Out-Null
    Register-ObjectEvent -InputObject $process -EventName Exited -Action {
        $code = $Event.Sender.ExitCode
        Append-Log "Process exited with code $code."
        $script:Form.BeginInvoke({ Set-RunningState $false }) | Out-Null
    } | Out-Null

    $script:Worker = $process
    [void]$process.Start()
    $process.BeginOutputReadLine()
    $process.BeginErrorReadLine()
    Set-RunningState $true
}

function Stop-JobProcess {
    if ($script:Worker -and -not $script:Worker.HasExited) {
        try {
            Append-Log "Stopping job..."
            Stop-Process -Id $script:Worker.Id -Force -ErrorAction Stop
        } catch {
            Append-Log "Stop failed: $($_.Exception.Message)"
        }
    }
}

$script:Form = New-Object System.Windows.Forms.Form
$script:Form.Text = "RTX50 RealESRGAN Portable GUI"
$script:Form.StartPosition = "CenterScreen"
$script:Form.Size = New-Object System.Drawing.Size(980, 760)
$script:Form.MinimumSize = New-Object System.Drawing.Size(900, 680)

$script:Form.Controls.Add((New-Label "Input folder" 16 18))
$script:InputBox = New-TextBox 140 18 680
$script:Form.Controls.Add($script:InputBox)
$inputBrowse = New-Button "Browse" 830 16
$inputBrowse.Add_Click({ Select-Folder $script:InputBox })
$script:Form.Controls.Add($inputBrowse)

$script:Form.Controls.Add((New-Label "Output folder" 16 54))
$script:OutputBox = New-TextBox 140 54 680
$script:Form.Controls.Add($script:OutputBox)
$outputBrowse = New-Button "Browse" 830 52
$outputBrowse.Add_Click({ Select-Folder $script:OutputBox })
$script:Form.Controls.Add($outputBrowse)

$script:Form.Controls.Add((New-Label "Work folder" 16 90))
$script:WorkBox = New-TextBox 140 90 680
$script:Form.Controls.Add($script:WorkBox)
$workBrowse = New-Button "Browse" 830 88
$workBrowse.Add_Click({ Select-Folder $script:WorkBox })
$script:Form.Controls.Add($workBrowse)

$script:Form.Controls.Add((New-Label "File pattern" 16 132))
$script:PatternBox = New-TextBox 140 132 120 "*.mkv"
$script:Form.Controls.Add($script:PatternBox)

$script:Form.Controls.Add((New-Label "Only name like" 290 132))
$script:OnlyBox = New-TextBox 400 132 220 ""
$script:Form.Controls.Add($script:OnlyBox)

$script:Form.Controls.Add((New-Label "Scale" 650 132 50))
$script:ScaleBox = New-Object System.Windows.Forms.ComboBox
$script:ScaleBox.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
[void]$script:ScaleBox.Items.AddRange(@("2", "3", "4"))
$script:ScaleBox.SelectedIndex = 0
$script:ScaleBox.Location = New-Object System.Drawing.Point(705, 132)
$script:ScaleBox.Size = New-Object System.Drawing.Size(70, 24)
$script:Form.Controls.Add($script:ScaleBox)

$script:Form.Controls.Add((New-Label "GPU" 800 132 35))
$script:GpuBox = New-Object System.Windows.Forms.NumericUpDown
$script:GpuBox.Minimum = 0
$script:GpuBox.Maximum = 16
$script:GpuBox.Value = 0
$script:GpuBox.Location = New-Object System.Drawing.Point(840, 132)
$script:GpuBox.Size = New-Object System.Drawing.Size(60, 24)
$script:Form.Controls.Add($script:GpuBox)

$script:Form.Controls.Add((New-Label "Model" 16 170))
$script:ModelBox = New-TextBox 140 170 220 "realesr-animevideov3"
$script:Form.Controls.Add($script:ModelBox)

$script:Form.Controls.Add((New-Label "AV1 mode" 390 170 85))
$script:Av1Box = New-Object System.Windows.Forms.ComboBox
$script:Av1Box.DropDownStyle = [System.Windows.Forms.ComboBoxStyle]::DropDownList
[void]$script:Av1Box.Items.AddRange(@("nvenc-highest", "nvenc-lossless", "libaom-lossless"))
$script:Av1Box.SelectedIndex = 0
$script:Av1Box.Location = New-Object System.Drawing.Point(480, 170)
$script:Av1Box.Size = New-Object System.Drawing.Size(160, 24)
$script:Form.Controls.Add($script:Av1Box)

$script:Form.Controls.Add((New-Label "Chunk seconds" 670 170 105))
$script:ChunkBox = New-Object System.Windows.Forms.NumericUpDown
$script:ChunkBox.Minimum = 10
$script:ChunkBox.Maximum = 3600
$script:ChunkBox.Value = 120
$script:ChunkBox.Location = New-Object System.Drawing.Point(785, 170)
$script:ChunkBox.Size = New-Object System.Drawing.Size(80, 24)
$script:Form.Controls.Add($script:ChunkBox)

$script:Form.Controls.Add((New-Label "Tile" 16 208))
$script:TileBox = New-Object System.Windows.Forms.NumericUpDown
$script:TileBox.Minimum = 0
$script:TileBox.Maximum = 8192
$script:TileBox.Increment = 128
$script:TileBox.Value = 2048
$script:TileBox.Location = New-Object System.Drawing.Point(140, 208)
$script:TileBox.Size = New-Object System.Drawing.Size(90, 24)
$script:Form.Controls.Add($script:TileBox)

$script:Form.Controls.Add((New-Label "Jobs" 260 208 50))
$script:JobsBox = New-TextBox 315 208 120 "24:12:24"
$script:Form.Controls.Add($script:JobsBox)

$script:Form.Controls.Add((New-Label "Dedup filter" 465 208 80))
$script:DedupBox = New-TextBox 550 208 220 "hi=768:lo=320:frac=0.33"
$script:Form.Controls.Add($script:DedupBox)

$script:Form.Controls.Add((New-Label "libaom threads" 16 246 105))
$script:LibaomThreadsBox = New-Object System.Windows.Forms.NumericUpDown
$script:LibaomThreadsBox.Minimum = 1
$script:LibaomThreadsBox.Maximum = 128
$script:LibaomThreadsBox.Value = 32
$script:LibaomThreadsBox.Location = New-Object System.Drawing.Point(140, 246)
$script:LibaomThreadsBox.Size = New-Object System.Drawing.Size(90, 24)
$script:Form.Controls.Add($script:LibaomThreadsBox)

$script:Form.Controls.Add((New-Label "libaom cpu-used" 260 246 115))
$script:LibaomCpuBox = New-Object System.Windows.Forms.NumericUpDown
$script:LibaomCpuBox.Minimum = 0
$script:LibaomCpuBox.Maximum = 8
$script:LibaomCpuBox.Value = 8
$script:LibaomCpuBox.Location = New-Object System.Drawing.Point(380, 246)
$script:LibaomCpuBox.Size = New-Object System.Drawing.Size(70, 24)
$script:Form.Controls.Add($script:LibaomCpuBox)

$script:Form.Controls.Add((New-Label "Max episodes" 480 246 95))
$script:MaxEpisodesBox = New-Object System.Windows.Forms.NumericUpDown
$script:MaxEpisodesBox.Minimum = 0
$script:MaxEpisodesBox.Maximum = 10000
$script:MaxEpisodesBox.Value = 0
$script:MaxEpisodesBox.Location = New-Object System.Drawing.Point(585, 246)
$script:MaxEpisodesBox.Size = New-Object System.Drawing.Size(75, 24)
$script:Form.Controls.Add($script:MaxEpisodesBox)

$script:Form.Controls.Add((New-Label "Max chunks" 690 246 85))
$script:MaxChunksBox = New-Object System.Windows.Forms.NumericUpDown
$script:MaxChunksBox.Minimum = 0
$script:MaxChunksBox.Maximum = 10000
$script:MaxChunksBox.Value = 0
$script:MaxChunksBox.Location = New-Object System.Drawing.Point(785, 246)
$script:MaxChunksBox.Size = New-Object System.Drawing.Size(75, 24)
$script:Form.Controls.Add($script:MaxChunksBox)

$script:VerboseBox = New-Object System.Windows.Forms.CheckBox
$script:VerboseBox.Text = "Verbose SR"
$script:VerboseBox.Location = New-Object System.Drawing.Point(140, 284)
$script:VerboseBox.Size = New-Object System.Drawing.Size(110, 24)
$script:Form.Controls.Add($script:VerboseBox)

$script:KeepFramesBox = New-Object System.Windows.Forms.CheckBox
$script:KeepFramesBox.Text = "Keep frames"
$script:KeepFramesBox.Location = New-Object System.Drawing.Point(270, 284)
$script:KeepFramesBox.Size = New-Object System.Drawing.Size(110, 24)
$script:Form.Controls.Add($script:KeepFramesBox)

$script:KeepChunksBox = New-Object System.Windows.Forms.CheckBox
$script:KeepChunksBox.Text = "Keep chunk videos"
$script:KeepChunksBox.Location = New-Object System.Drawing.Point(400, 284)
$script:KeepChunksBox.Size = New-Object System.Drawing.Size(150, 24)
$script:Form.Controls.Add($script:KeepChunksBox)

$script:StartButton = New-Button "Start" 16 322 110 34
$script:StartButton.Add_Click({ Start-JobProcess })
$script:Form.Controls.Add($script:StartButton)

$script:StopButton = New-Button "Stop" 140 322 110 34
$script:StopButton.Enabled = $false
$script:StopButton.Add_Click({ Stop-JobProcess })
$script:Form.Controls.Add($script:StopButton)

$openOutput = New-Button "Open output" 270 322 110 34
$openOutput.Add_Click({
    if (Test-Path -LiteralPath $script:OutputBox.Text) {
        Start-Process explorer.exe $script:OutputBox.Text
    }
})
$script:Form.Controls.Add($openOutput)

$openWork = New-Button "Open work" 400 322 110 34
$openWork.Add_Click({
    if (Test-Path -LiteralPath $script:WorkBox.Text) {
        Start-Process explorer.exe $script:WorkBox.Text
    }
})
$script:Form.Controls.Add($openWork)

$script:StatusLabel = New-Label "Idle" 535 328 160 24
$script:Form.Controls.Add($script:StatusLabel)

$script:LogBox = New-Object System.Windows.Forms.TextBox
$script:LogBox.Location = New-Object System.Drawing.Point(16, 374)
$script:LogBox.Size = New-Object System.Drawing.Size(930, 320)
$script:LogBox.Multiline = $true
$script:LogBox.ScrollBars = [System.Windows.Forms.ScrollBars]::Both
$script:LogBox.WordWrap = $false
$script:LogBox.Font = New-Object System.Drawing.Font("Consolas", 9)
$script:Form.Controls.Add($script:LogBox)

$script:Form.Add_FormClosing({
    if ($script:Worker -and -not $script:Worker.HasExited) {
        $result = [System.Windows.Forms.MessageBox]::Show("A job is still running. Stop it and close?", "RealESRGAN GUI", [System.Windows.Forms.MessageBoxButtons]::YesNo)
        if ($result -ne [System.Windows.Forms.DialogResult]::Yes) {
            $_.Cancel = $true
            return
        }
        Stop-JobProcess
    }
})

[void]$script:Form.ShowDialog()
