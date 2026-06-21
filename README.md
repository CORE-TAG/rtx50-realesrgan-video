# RTX50 RealESRGAN Video

Windows-first RealESRGAN anime video upscaling scripts and RTX 5090 benchmark notes.

This repository packages a practical video pipeline, a fast TensorRT/vs-mlrt benchmark harness, and experimental SM120 Tensor Core plugin work for the RealESRGAN animevideo/SRVGG family. It is not a replacement for TensorRT. The production path uses `vs-mlrt`/`vstrt` and TensorRT; the custom kernels are research artifacts.

## Highlights

- Deduplicated anime video pipeline: extract only non-duplicate frames, upscale unique frames, rebuild the original frame timeline, then encode AV1.
- TensorRT/vs-mlrt benchmark profile for RTX 5090: `batch=2`, `streams=2`, `requests=4`, FP16 I/O, CUDA Graph, TensorRT all tactics.
- RTX 5090 measurements and Nsight Compute notes, including the observed power-wall behavior.
- Experimental SRVGG SM120 TensorRT plugins and CUTLASS probes documenting what did and did not work.

## Current Best Result

On the tested RTX 5090 system, the production TensorRT/vs-mlrt path currently reproduces about:

```text
256 frames: 72.06 fps
128 frames: 72.58 fps
```

Earlier short-window peaks reached the high 70s to around 80 fps, but the repeatable steady result is around 72-73 fps. See `docs/PERFORMANCE.md`.

## Repository Layout

```text
scripts/
  bench_vsmlrt_realesrgan_best.ps1       Fast TensorRT/vs-mlrt benchmark harness
  run_realesrgan_dedup_av1_lossless.ps1  Dedup + RealESRGAN-ncnn + AV1 pipeline
  run_realesrgan_dedup_av1_pipeline.py   Overlapped Python pipeline prototype
vapoursynth/
  benchmark_vsmlrt_realesrgan_x2.vpy     BlankClip benchmark script for vs-mlrt TRT
experiments/
  trt_srvgg_plugin/                      TensorRT plugin experiments for SRVGG conv
  realesrgan_trt/                        ONNX export and CUDA/CUTLASS probes
benchmarks/raw/                          Small raw logs and timing JSON files
docs/                                    Architecture, performance, GitHub landscape
```

Large dependencies, models, TensorRT engines, videos, and virtual environments are intentionally not included.

## Requirements

- Windows 11 recommended.
- NVIDIA RTX GPU with a recent driver.
- CUDA/TensorRT stack compatible with your GPU.
- VapourSynth + SVP/vs-mlrt for the TensorRT benchmark path.
- FFmpeg and RealESRGAN-ncnn-vulkan for the dedup video pipeline.
- Python 3.10+ for the Python scripts and TensorRT plugin benches.

The project expects you to provide third-party tools under `tools/` or pass explicit paths. See `THIRD_PARTY_NOTICES.md`.

## TensorRT/vs-mlrt Benchmark

Place or generate an ONNX model, for example:

```text
models/realesr-animevideov3-x2-fusedtail.onnx
```

Then run:

```powershell
.\scripts\bench_vsmlrt_realesrgan_best.ps1 `
  -Model C:\models\realesr-animevideov3-x2-fusedtail.onnx `
  -Frames 256 `
  -Batch 2 `
  -Streams 2 `
  -Requests 4
```

If your SVP/vs-mlrt install is not in the default location, pass:

```powershell
-VSPipe C:\path\to\VSPipe.exe -SvpRife C:\path\to\rife -SvpTrt C:\path\to\vsmlrt-cuda
```

## Deduplicated Video Pipeline

This path uses FFmpeg `mpdecimate` and `realesrgan-ncnn-vulkan`.

```powershell
.\scripts\run_realesrgan_dedup_av1_lossless.ps1 `
  -InputDir D:\input-anime `
  -OutputDir D:\output-upscaled `
  -ToolsRoot D:\video-ai-tools `
  -Av1Mode nvenc-highest
```

Expected tool layout:

```text
tools/
  ffmpeg/ffmpeg-8.1.1-essentials_build/bin/ffmpeg.exe
  ffmpeg/ffmpeg-8.1.1-essentials_build/bin/ffprobe.exe
  realesrgan/realesrgan-ncnn-vulkan.exe
  realesrgan/models/
```

## Experimental SM120 Plugin Work

The plugin sources are in `experiments/trt_srvgg_plugin`. The best single-layer experiment is HWC8-only FP16 input/output with FP32 accumulation, but it is still much slower than TensorRT's production implicit convolution tactic on the full model.

Use this section as a research trail, not as a recommended production path.

## Related Projects

- [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt)
- [styler00dollar/VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker)
- [yester31/Real_ESRGAN_TRT](https://github.com/yester31/Real_ESRGAN_TRT)
- [wang-xinyu/tensorrtx real-esrgan](https://github.com/wang-xinyu/tensorrtx/tree/master/real-esrgan)
- [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass)

## License

Project glue code and documentation are MIT licensed. Some experimental CUTLASS-derived files retain NVIDIA's BSD-3-Clause headers. See `THIRD_PARTY_NOTICES.md`.

