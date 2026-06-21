# GitHub Landscape

This project overlaps with several existing repositories, but its focus is narrower: RTX 50/Windows/RealESRGAN anime video upscaling, deduplication, benchmarking, and SM120 kernel experiments.

## Closest Production Projects

- [AmusementClub/vs-mlrt](https://github.com/AmusementClub/vs-mlrt): the core VapourSynth ML runtime used by the production TensorRT path. Its `vstrt` backend is the most important upstream dependency.
- [styler00dollar/VSGAN-tensorrt-docker](https://github.com/styler00dollar/VSGAN-tensorrt-docker): a practical VapourSynth + TensorRT video enhancement stack with many model examples.
- [mafiosnik777/enhancr](https://github.com/mafiosnik777/enhancr): user-facing video enhancement app that also uses TensorRT/vs-mlrt for RealESRGAN and related models.
- [universonic/pixworker](https://github.com/universonic/pixworker): video enhancement tool with CUDA/TensorRT acceleration support.

## Lower-Level RealESRGAN TensorRT Projects

- [yester31/Real_ESRGAN_TRT](https://github.com/yester31/Real_ESRGAN_TRT): C++ TensorRT API implementation for RealESRGAN.
- [wang-xinyu/tensorrtx real-esrgan](https://github.com/wang-xinyu/tensorrtx/tree/master/real-esrgan): classic TensorRT C++ model implementation path.

These are useful references, but the current production path here is vs-mlrt/TensorRT rather than hand-building the full RealESRGAN network in C++.

## Kernel/Research References

- [NVIDIA/CUTLASS](https://github.com/NVIDIA/cutlass): primary public reference for Tensor Core GEMM and implicit GEMM convolution kernels.
- CUTLASS changelog notes Blackwell SM100 implicit GEMM convolution tests and SM120 blockwise dense GEMM support, but this does not directly provide a drop-in RTX 5090 SM120 FP16 RealESRGAN convolution engine for this project.

## What Is New Here

- A Windows RTX 5090-oriented benchmark harness for RealESRGAN animevideo/SRVGG via vs-mlrt TensorRT.
- A deduplicated anime-video workflow that reduces repeated-frame SR work.
- Local RTX 5090 performance notes with power-wall and Nsight Compute observations.
- SM120 plugin experiments documenting unsuccessful and partially successful optimization paths.

## What Is Not New Here

- The production inference runtime is not a new engine. It is TensorRT through vs-mlrt.
- The project does not redistribute RealESRGAN models, TensorRT, CUDA, FFmpeg, SVP, or third-party binaries.

