# Portable GUI Release

The portable GUI package bundles:

- FFmpeg `ffmpeg.exe` and `ffprobe.exe`
- RealESRGAN-ncnn-vulkan
- RealESRGAN-ncnn-vulkan model files
- PowerShell/WinForms GUI launcher

Users run:

```text
Start-GUI.bat
```

No Python, CUDA Toolkit, TensorRT, VapourSynth, or SVP installation is required for the GUI path.

## Still Required

A working GPU driver is still required. The application cannot bundle GPU drivers.

For `nvenc-highest` or `nvenc-lossless`, the GPU and driver must support AV1 NVENC. Otherwise choose `libaom-lossless`.

## Build

From the repository root:

```powershell
.\scripts\build_portable_release.ps1 -Version v0.1.0
```

The script expects local dependency folders:

```text
tools/ffmpeg/ffmpeg-8.1.1-essentials_build/
tools/realesrgan/
```

The zip is written under `publish/release/` by default when run from this workspace layout.

