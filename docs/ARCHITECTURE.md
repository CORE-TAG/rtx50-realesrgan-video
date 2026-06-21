# Architecture

## Production Benchmark Path

```text
VapourSynth BlankClip
  -> RGBH/FP16 input
  -> vs-mlrt Backend.TRT
  -> TensorRT engine cache
  -> VSPipe benchmark output
```

Primary tuning knobs:

```text
batch_size=2
num_streams=2
VSPipe --requests 4
output_format=FP16
use_cuda_graph=True
builder_optimization_level=5
use_cublas=True
use_cudnn=True
use_edge_mask_convolutions=True
use_jit_convolutions=True
```

The best local result came from allowing all major TensorRT tactics. Restricting tactics or forcing custom kernels did not beat TensorRT.

## Deduplicated Video Path

```text
Input episode
  -> FFmpeg chunk extraction
  -> mpdecimate duplicate-frame filtering
  -> unique PNG frames
  -> RealESRGAN-ncnn-vulkan
  -> rebuilt full frame sequence
  -> AV1 encode
  -> audio/subtitle/attachment remux
```

This reduces super-resolution work on anime episodes with long repeated-frame spans. It preserves the original timeline by copying the nearest corresponding upscaled unique frame back into duplicated positions.

## Experimental Kernel Path

```text
TensorRT custom plugin
  -> HWC8 FP16 input/output
  -> WMMA/HMMA Tensor Core convolution
  -> bias + PReLU epilogue
```

The current plugin demonstrates correctness and documents SM120 exploration, but it is not production faster than TensorRT's built-in implicit convolution tactic.

