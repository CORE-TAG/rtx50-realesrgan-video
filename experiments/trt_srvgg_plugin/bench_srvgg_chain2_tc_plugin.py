import argparse
import os
import sys
import time

import numpy as np
import tensorrt as trt
from cuda.bindings import runtime as cuda


PLUGIN_NAME = "SrvggConv1TcPlugin"
PLUGIN_VERSION = "1"


class Logger(trt.ILogger):
    def __init__(self):
        super().__init__()

    def log(self, severity, msg):
        if severity <= trt.ILogger.Severity.WARNING:
            print(f"[TRT] {severity.name}: {msg}")


def check_cuda(result, what):
    code = result[0]
    if code != cuda.cudaError_t.cudaSuccess:
        raise RuntimeError(f"{what} failed: {code}")
    return result[1:] if len(result) > 1 else None


def read_fp16(path, count):
    arr = np.fromfile(path, dtype=np.float16)
    if arr.size != count:
        raise RuntimeError(f"{path}: got {arr.size} fp16 values, expected {count}")
    return np.ascontiguousarray(arr)


def load_weights(weight_dir, layer):
    weight_count = 64 * 64 * 3 * 3
    vec_count = 64
    return {
        "weight": read_fp16(os.path.join(weight_dir, f"conv{layer:02d}_weight_fp16.bin"), weight_count),
        "bias": read_fp16(os.path.join(weight_dir, f"conv{layer:02d}_bias_fp16.bin"), vec_count),
        "prelu": read_fp16(os.path.join(weight_dir, f"prelu{layer:02d}_fp16.bin"), vec_count),
    }


def create_plugin(plugin_path, weights):
    registry = trt.get_plugin_registry()
    creator = registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION, "")
    if creator is None:
        creator = registry.get_plugin_creator(PLUGIN_NAME, PLUGIN_VERSION, "")
    if creator is None:
        handle = registry.load_library(plugin_path)
        if handle is None:
            raise RuntimeError(f"failed to load plugin library: {plugin_path}")
        creator = registry.get_creator(PLUGIN_NAME, PLUGIN_VERSION, "")
        if creator is None:
            creator = registry.get_plugin_creator(PLUGIN_NAME, PLUGIN_VERSION, "")
    if creator is None:
        names = [c.name for c in registry.all_creators]
        raise RuntimeError(f"plugin creator {PLUGIN_NAME} not found; registered creators include {names[:20]}")

    fields = [trt.PluginField(name, value, trt.PluginFieldType.FLOAT16) for name, value in weights.items()]
    return creator.create_plugin(PLUGIN_NAME, trt.PluginFieldCollection(fields))


def format_mask(io_format):
    if io_format == "linear":
        return 1 << int(trt.TensorFormat.LINEAR)
    if io_format == "hwc8":
        return 1 << int(trt.TensorFormat.HWC8)
    raise ValueError(f"unsupported format: {io_format}")


def describe_tensor(engine, name):
    parts = [name]
    for attr in ("get_tensor_dtype", "get_tensor_format", "get_tensor_format_desc", "get_tensor_shape"):
        try:
            value = getattr(engine, attr)(name)
        except Exception as exc:  # TensorRT minor versions differ here.
            value = f"<{type(exc).__name__}>"
        parts.append(f"{attr.replace('get_tensor_', '')}={value}")
    return " ".join(parts)


def build_engine(plugin_path, weight_dir, layer, n, h, w, io_format):
    logger = Logger()
    plugin1 = create_plugin(plugin_path, load_weights(weight_dir, layer))
    plugin2 = create_plugin(plugin_path, load_weights(weight_dir, layer + 1))
    if plugin1 is None or plugin2 is None:
        raise RuntimeError("creator.create_plugin returned None")

    builder = trt.Builder(logger)
    flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flag)
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    input_tensor = network.add_input("input", trt.float16, (n, 64, h, w))
    mask = format_mask(io_format)
    input_tensor.allowed_formats = mask
    layer1_obj = network.add_plugin_v2([input_tensor], plugin1)
    mid_tensor = layer1_obj.get_output(0)
    mid_tensor.name = "mid"
    mid_tensor.dtype = trt.float16
    mid_tensor.allowed_formats = mask
    layer2_obj = network.add_plugin_v2([mid_tensor], plugin2)
    output_tensor = layer2_obj.get_output(0)
    output_tensor.name = "output"
    network.mark_output(output_tensor)
    output_tensor.dtype = trt.float16
    output_tensor.allowed_formats = mask
    if io_format != "linear" and hasattr(trt.BuilderFlag, "DIRECT_IO"):
        config.set_flag(trt.BuilderFlag.DIRECT_IO)

    started = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - started
    if serialized is None:
        raise RuntimeError("build_serialized_network failed")

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    if engine is None:
        raise RuntimeError("deserialize_cuda_engine failed")
    return engine, build_s


def benchmark(engine, n, h, w, warmup, iters):
    check_cuda(cuda.cudaSetDevice(0), "cudaSetDevice")
    elems = n * 64 * h * w
    bytes_count = elems * np.dtype(np.float16).itemsize
    d_input = check_cuda(cuda.cudaMalloc(bytes_count), "cudaMalloc input")[0]
    d_output = check_cuda(cuda.cudaMalloc(bytes_count), "cudaMalloc output")[0]
    check_cuda(cuda.cudaMemset(d_input, 17, bytes_count), "cudaMemset input")
    check_cuda(cuda.cudaMemset(d_output, 0, bytes_count), "cudaMemset output")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context failed")
    context.set_input_shape("input", (n, 64, h, w))
    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address("output", int(d_output))

    stream = check_cuda(cuda.cudaStreamCreate(), "cudaStreamCreate")[0]
    start = check_cuda(cuda.cudaEventCreate(), "cudaEventCreate start")[0]
    stop = check_cuda(cuda.cudaEventCreate(), "cudaEventCreate stop")[0]

    for _ in range(warmup):
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("warmup execute_async_v3 failed")
    check_cuda(cuda.cudaStreamSynchronize(stream), "warmup sync")

    check_cuda(cuda.cudaEventRecord(start, stream), "event record start")
    for _ in range(iters):
        if not context.execute_async_v3(stream_handle=int(stream)):
            raise RuntimeError("execute_async_v3 failed")
    check_cuda(cuda.cudaEventRecord(stop, stream), "event record stop")
    check_cuda(cuda.cudaEventSynchronize(stop), "event sync")
    elapsed_ms = check_cuda(cuda.cudaEventElapsedTime(start, stop), "event elapsed")[0]

    check_cuda(cuda.cudaEventDestroy(start), "event destroy start")
    check_cuda(cuda.cudaEventDestroy(stop), "event destroy stop")
    check_cuda(cuda.cudaStreamDestroy(stream), "stream destroy")
    check_cuda(cuda.cudaFree(d_input), "cudaFree input")
    check_cuda(cuda.cudaFree(d_output), "cudaFree output")

    per_ms = elapsed_ms / iters
    flops = 2.0 * n * 2.0 * 64 * h * w * 64 * 3 * 3
    return per_ms, flops / (per_ms / 1000.0) / 1e12


def to_engine_layout(x_nchw, io_format):
    if io_format == "linear":
        return np.ascontiguousarray(x_nchw.astype(np.float16).reshape(-1))
    if io_format == "hwc8":
        return np.ascontiguousarray(np.transpose(x_nchw, (0, 2, 3, 1)).astype(np.float16).reshape(-1))
    raise ValueError(f"unsupported format: {io_format}")


def from_engine_layout(buf, n, h, w, io_format):
    if io_format == "linear":
        return np.ascontiguousarray(buf.reshape(n, 64, h, w))
    if io_format == "hwc8":
        return np.ascontiguousarray(buf.reshape(n, h, w, 64).transpose(0, 3, 1, 2))
    raise ValueError(f"unsupported format: {io_format}")


def execute_once(engine, n, h, w, host_input):
    check_cuda(cuda.cudaSetDevice(0), "cudaSetDevice")
    bytes_count = host_input.nbytes
    d_input = check_cuda(cuda.cudaMalloc(bytes_count), "cudaMalloc input")[0]
    d_output = check_cuda(cuda.cudaMalloc(bytes_count), "cudaMalloc output")[0]
    check_cuda(cuda.cudaMemcpy(d_input, host_input.ctypes.data, bytes_count, cuda.cudaMemcpyKind.cudaMemcpyHostToDevice), "copy input")
    check_cuda(cuda.cudaMemset(d_output, 0, bytes_count), "cudaMemset output")

    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("create_execution_context failed")
    context.set_input_shape("input", (n, 64, h, w))
    context.set_tensor_address("input", int(d_input))
    context.set_tensor_address("output", int(d_output))
    stream = check_cuda(cuda.cudaStreamCreate(), "cudaStreamCreate")[0]
    if not context.execute_async_v3(stream_handle=int(stream)):
        raise RuntimeError("execute_async_v3 failed")
    check_cuda(cuda.cudaStreamSynchronize(stream), "execute sync")
    host_output = np.empty_like(host_input)
    check_cuda(
        cuda.cudaMemcpy(host_output.ctypes.data, d_output, bytes_count, cuda.cudaMemcpyKind.cudaMemcpyDeviceToHost),
        "copy output",
    )
    check_cuda(cuda.cudaStreamDestroy(stream), "cudaStreamDestroy")
    check_cuda(cuda.cudaFree(d_input), "cudaFree input")
    check_cuda(cuda.cudaFree(d_output), "cudaFree output")
    return host_output


def reference_conv_prelu(x_nchw, weights):
    weight = weights["weight"].astype(np.float32).reshape(64, 64, 3, 3)
    bias = weights["bias"].astype(np.float32)
    prelu = weights["prelu"].astype(np.float32)
    x = x_nchw.astype(np.float32)
    n, _, h, w = x.shape
    padded = np.pad(x, ((0, 0), (0, 0), (1, 1), (1, 1)), mode="constant")
    out = np.empty((n, 64, h, w), dtype=np.float32)
    for oc in range(64):
        acc = np.full((n, h, w), bias[oc], dtype=np.float32)
        for ic in range(64):
            for ky in range(3):
                for kx in range(3):
                    acc += padded[:, ic, ky : ky + h, kx : kx + w] * weight[oc, ic, ky, kx]
        out[:, oc] = np.where(acc >= 0.0, acc, acc * prelu[oc])
    return out


def check_correctness(engine, weight_dir, layer, n, h, w, io_format):
    rng = np.random.default_rng(1234)
    x_nchw = rng.normal(0.0, 0.15, size=(n, 64, h, w)).astype(np.float16)
    host_input = to_engine_layout(x_nchw, io_format)
    host_output = execute_once(engine, n, h, w, host_input)
    y_plugin = from_engine_layout(host_output, n, h, w, io_format).astype(np.float32)
    y_mid = reference_conv_prelu(x_nchw, load_weights(weight_dir, layer)).astype(np.float16)
    y_ref = reference_conv_prelu(y_mid, load_weights(weight_dir, layer + 1)).astype(np.float16).astype(np.float32)
    diff = np.abs(y_plugin - y_ref)
    return float(diff.max()), float(diff.mean()), int(np.argmax(diff))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plugin", default=os.path.join(os.path.dirname(__file__), "srvgg_conv1_tc_plugin.dll"))
    parser.add_argument(
        "--weights",
        default=os.path.join(os.path.dirname(__file__), "..", "realesrgan_trt", "srvgg_weights"),
    )
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--h", type=int, default=1080)
    parser.add_argument("--w", type=int, default=1920)
    parser.add_argument("--format", choices=("linear", "hwc8"), default="linear")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    engine, build_s = build_engine(
        os.path.abspath(args.plugin), os.path.abspath(args.weights), args.layer, args.n, args.h, args.w, args.format
    )
    per_ms, tflops = benchmark(engine, args.n, args.h, args.w, args.warmup, args.iters)
    fps = 1000.0 / per_ms
    print(f"build_s={build_s:.3f}")
    print(describe_tensor(engine, "input"))
    print(describe_tensor(engine, "output"))
    print(
        f"plugin=srvgg_chain2_tc layers={args.layer:02d}-{args.layer + 1:02d} n={args.n} "
        f"shape={args.h}x{args.w} format={args.format} iters={args.iters}"
    )
    print(f"two_layer_ms={per_ms:.6f} pair_fps={fps:.3f} effective_tflops={tflops:.3f}")
    if args.check:
        max_diff, mean_diff, max_idx = check_correctness(
            engine, os.path.abspath(args.weights), args.layer, args.n, args.h, args.w, args.format
        )
        print(f"check_max_abs={max_diff:.6g} check_mean_abs={mean_diff:.6g} check_max_idx={max_idx}")


if __name__ == "__main__":
    sys.exit(main())
