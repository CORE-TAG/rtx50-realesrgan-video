# Third-Party Notices

This repository contains glue scripts, benchmark harnesses, and experimental code. It does not redistribute third-party binaries, models, video files, TensorRT engines, virtual environments, or downloaded source trees.

## Runtime Dependencies

Users must obtain these separately:

- NVIDIA CUDA Toolkit
- NVIDIA TensorRT
- VapourSynth
- SVP/vs-mlrt
- FFmpeg
- RealESRGAN-ncnn-vulkan
- RealESRGAN model weights / ONNX exports

Each dependency is governed by its own license.

## Related Open Source Projects

- vs-mlrt: https://github.com/AmusementClub/vs-mlrt
- Real-ESRGAN: https://github.com/xinntao/Real-ESRGAN
- RealESRGAN-ncnn-vulkan: https://github.com/xinntao/Real-ESRGAN-ncnn-vulkan
- VSGAN-tensorrt-docker: https://github.com/styler00dollar/VSGAN-tensorrt-docker
- TensorRTx: https://github.com/wang-xinyu/tensorrtx
- CUTLASS: https://github.com/NVIDIA/cutlass

## CUTLASS-Derived Files

Some files in `experiments/trt_srvgg_plugin` are modified from NVIDIA CUTLASS examples and retain NVIDIA's BSD-3-Clause header in the file itself. Those file-level notices control those derived files.

## Models

No model weights are included. Users should download or export their own models and verify the redistribution terms for those models.

