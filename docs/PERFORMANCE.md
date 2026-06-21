# Performance Notes

## RTX 5090 Production Path

Best repeatable production path:

```text
Backend: vs-mlrt/vstrt + TensorRT
Model: RealESRGAN animevideo/SRVGG x2 fused-tail ONNX
Input: 1920x1080 RGBH/FP16 synthetic BlankClip
Settings: batch=2, streams=2, requests=4, FP16 I/O, CUDA Graph, TensorRT all tactics
```

Recent rerun:

```text
256 frames: 72.06 fps
128 frames: 72.58 fps
```

Historical short-window peak:

```text
High 70s to around 80 fps
```

Use the steady 72-73 fps number for planning real jobs.

## Model Math

Approximate SRVGG x2 model math:

```text
2.56794624 TFLOP/frame
72.69 fps ~= 186.7 TFLOP/s
```

Against an RTX 5090 dense FP16 Tensor theoretical figure of 419 TFLOP/s, this is about 44.5 percent model-level utilization. Against sparse marketing figures, the gap looks larger, but this model path is not automatically sparse.

## Observed System State

During the fast TensorRT path:

```text
GPU utilization: ~99 percent
Power: ~600 W power limit
Core clock: roughly 2.42-2.50 GHz
Temperature: roughly low 60s C in the captured sample
```

The system was power-wall limited rather than CPU, disk, or encoder limited in the synthetic benchmark.

## TensorRT Kernel Evidence

Nsight Compute identified the dominant TensorRT convolution kernel as an SM80-style xMMA implicit GEMM fallback on RTX 5090/SM120:

```text
sm80_xmma_fprop_implicit_gemm_f16f16_f16f16_f16_nhwckrsc_nhwc_tilesize256x64x32_stage3...
```

This means TensorRT is very strong here even without a visible SM120-native FP16 UMMA convolution path.

## Experimental Plugin Results

Single-layer SRVGG TensorRT plugin, best HWC8 version:

```text
n=1 1080p: one_layer_ms=3.844309, layer_fps=260.125, effective_tflops=39.768
n=2 1080p: one_layer_ms=7.770139, layer_fps=128.698, effective_tflops=39.351
```

Two-layer wrapper/fused plugin:

```text
n=1: effective_tflops ~= 35.0
n=2: effective_tflops ~= 34.7
```

Conclusion: the custom plugin experiments are useful for research, but the production TensorRT engine remains much faster.

## CUTLASS/SM120 Finding

Public CUTLASS Blackwell paths were useful for exploration, but a directly usable RTX 5090 SM120 FP16 dense UMMA convolution/GEMM route was not available through the tested examples. SM120 public examples were more mature for FP8/FP4/block-scaled paths than for this no-quality-loss FP16 SRVGG use case.

