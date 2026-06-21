import argparse
import ctypes
import os
import time
from pathlib import Path

import numpy as np


def load_cudart(cuda_bin: str):
    os.add_dll_directory(cuda_bin)
    cudart = ctypes.CDLL(str(Path(cuda_bin) / "cudart64_12.dll"))

    cudart.cudaSetDevice.argtypes = [ctypes.c_int]
    cudart.cudaSetDevice.restype = ctypes.c_int
    cudart.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cudart.cudaMalloc.restype = ctypes.c_int
    cudart.cudaFree.argtypes = [ctypes.c_void_p]
    cudart.cudaFree.restype = ctypes.c_int
    cudart.cudaMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    cudart.cudaMemset.restype = ctypes.c_int
    cudart.cudaStreamCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaStreamCreate.restype = ctypes.c_int
    cudart.cudaStreamDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamDestroy.restype = ctypes.c_int
    cudart.cudaStreamSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaStreamSynchronize.restype = ctypes.c_int
    cudart.cudaEventCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
    cudart.cudaEventCreate.restype = ctypes.c_int
    cudart.cudaEventDestroy.argtypes = [ctypes.c_void_p]
    cudart.cudaEventDestroy.restype = ctypes.c_int
    cudart.cudaEventRecord.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventRecord.restype = ctypes.c_int
    cudart.cudaEventSynchronize.argtypes = [ctypes.c_void_p]
    cudart.cudaEventSynchronize.restype = ctypes.c_int
    cudart.cudaEventElapsedTime.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_void_p, ctypes.c_void_p]
    cudart.cudaEventElapsedTime.restype = ctypes.c_int
    return cudart


def check_cuda(rc: int, what: str):
    if rc != 0:
        raise RuntimeError(f"{what} failed with cuda error {rc}")


def volume(shape):
    n = 1
    for dim in shape:
        n *= int(dim)
    return n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--shape", default="2,3,1080,1920")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--workspace-gb", type=int, default=8)
    parser.add_argument("--cuda-bin", default=r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin")
    parser.add_argument("--save-engine", default="")
    args = parser.parse_args()

    cudart = load_cudart(args.cuda_bin)
    check_cuda(cudart.cudaSetDevice(0), "cudaSetDevice")

    import tensorrt as trt

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser_trt = trt.OnnxParser(network, logger)
    onnx_bytes = Path(args.onnx).read_bytes()
    if not parser_trt.parse(onnx_bytes):
        for i in range(parser_trt.num_errors):
            print(parser_trt.get_error(i))
        raise RuntimeError("ONNX parse failed")

    input_name = network.get_input(0).name
    shape = tuple(int(x) for x in args.shape.split(","))
    config = builder.create_builder_config()
    if hasattr(trt.BuilderFlag, "FP16"):
        config.set_flag(trt.BuilderFlag.FP16)
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, args.workspace_gb << 30)
    profile = builder.create_optimization_profile()
    profile.set_shape(input_name, shape, shape, shape)
    config.add_optimization_profile(profile)

    t0 = time.perf_counter()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.perf_counter() - t0
    if serialized is None:
        raise RuntimeError("TensorRT build failed")
    if args.save_engine:
        Path(args.save_engine).write_bytes(bytes(serialized))

    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(serialized)
    context = engine.create_execution_context()
    if hasattr(context, "set_input_shape"):
        context.set_input_shape(input_name, shape)

    buffers = []
    names = []
    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        names.append(name)
        tensor_shape = tuple(context.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        nbytes = volume(tensor_shape) * np.dtype(dtype).itemsize
        ptr = ctypes.c_void_p()
        check_cuda(cudart.cudaMalloc(ctypes.byref(ptr), nbytes), f"cudaMalloc {name}")
        check_cuda(cudart.cudaMemset(ptr, 0, nbytes), f"cudaMemset {name}")
        context.set_tensor_address(name, int(ptr.value))
        buffers.append(ptr)
        print(f"tensor {name} shape={tensor_shape} dtype={np.dtype(dtype).name} bytes={nbytes}")

    stream = ctypes.c_void_p()
    start = ctypes.c_void_p()
    stop = ctypes.c_void_p()
    check_cuda(cudart.cudaStreamCreate(ctypes.byref(stream)), "cudaStreamCreate")
    check_cuda(cudart.cudaEventCreate(ctypes.byref(start)), "cudaEventCreate start")
    check_cuda(cudart.cudaEventCreate(ctypes.byref(stop)), "cudaEventCreate stop")

    for _ in range(args.warmup):
        if not context.execute_async_v3(stream_handle=int(stream.value)):
            raise RuntimeError("execute_async_v3 failed during warmup")
    check_cuda(cudart.cudaStreamSynchronize(stream), "warmup sync")

    check_cuda(cudart.cudaEventRecord(start, stream), "event start")
    wall0 = time.perf_counter()
    for _ in range(args.iters):
        if not context.execute_async_v3(stream_handle=int(stream.value)):
            raise RuntimeError("execute_async_v3 failed")
    check_cuda(cudart.cudaEventRecord(stop, stream), "event stop")
    check_cuda(cudart.cudaEventSynchronize(stop), "event sync")
    wall_s = time.perf_counter() - wall0
    gpu_ms = ctypes.c_float()
    check_cuda(cudart.cudaEventElapsedTime(ctypes.byref(gpu_ms), start, stop), "elapsed")

    batch = shape[0]
    effective_frames = args.iters * batch
    print(f"trt_version={trt.__version__}")
    print(f"build_s={build_s:.3f}")
    print(f"iters={args.iters} batch={batch} effective_frames={effective_frames}")
    print(f"wall_s={wall_s:.6f} wall_fps={effective_frames / wall_s:.3f}")
    print(f"gpu_ms={gpu_ms.value:.3f} gpu_fps={effective_frames / (gpu_ms.value / 1000.0):.3f}")
    print(f"batch_latency_gpu_ms={gpu_ms.value / args.iters:.6f}")

    check_cuda(cudart.cudaEventDestroy(start), "destroy start")
    check_cuda(cudart.cudaEventDestroy(stop), "destroy stop")
    check_cuda(cudart.cudaStreamDestroy(stream), "stream destroy")
    for ptr in buffers:
        check_cuda(cudart.cudaFree(ptr), "cudaFree")


if __name__ == "__main__":
    main()
