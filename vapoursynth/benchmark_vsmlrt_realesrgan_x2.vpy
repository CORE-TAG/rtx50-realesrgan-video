import os
import sys
import ctypes

import vapoursynth as vs


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
SVP_RIFE = os.environ.get("VSMLRT_SVP_RIFE", r"C:\Program Files\SVP 4\rife")
SVP_TRT = os.environ.get("VSMLRT_SVP_TRT", os.path.join(SVP_RIFE, "vsmlrt-cuda"))
VENV_SITE = os.environ.get("VSMLRT_PYTHON_SITE", "")
MODEL = os.environ.get(
    "VSMLRT_MODEL",
    os.path.join(PROJECT_ROOT, "models", "realesr-animevideov3-x2-fusedtail.onnx"),
)
ENGINE_CACHE = os.environ.get("VSMLRT_ENGINE_CACHE", os.path.join(PROJECT_ROOT, ".cache", "vsmlrt_engine_cache"))
TRT_DLL_PATH = os.environ.get("TRT_DLL_PATH", "")

path_parts = [p for p in (TRT_DLL_PATH, SVP_TRT, SVP_RIFE) if p]
os.environ["PATH"] = os.pathsep.join(path_parts) + os.pathsep + os.environ.get("PATH", "")
if hasattr(os, "add_dll_directory"):
    if TRT_DLL_PATH:
        os.add_dll_directory(TRT_DLL_PATH)
    if os.path.isdir(SVP_RIFE):
        os.add_dll_directory(SVP_RIFE)
    if os.path.isdir(SVP_TRT):
        os.add_dll_directory(SVP_TRT)
if TRT_DLL_PATH:
    ctypes.WinDLL(os.path.join(TRT_DLL_PATH, "nvinfer_10.dll"))
    ctypes.WinDLL(os.path.join(TRT_DLL_PATH, "nvinfer_plugin_10.dll"))
sys.path.insert(0, SVP_RIFE)
if VENV_SITE:
    sys.path.insert(0, VENV_SITE)

core = vs.core
core.num_threads = 4
core.std.LoadPlugin(os.path.join(SVP_RIFE, "vstrt.dll"))

from vsmlrt import Backend, inference  # noqa: E402


width = int(globals().get("width", 1920))
height = int(globals().get("height", 1080))
frames = int(globals().get("frames", 256))
streams = int(globals().get("streams", 1))
batch = int(globals().get("batch", 1))
model = str(globals().get("model", MODEL))
fp16io = bool(int(globals().get("fp16io", 0)))
force = bool(int(globals().get("force", 0)))
tf32 = bool(int(globals().get("tf32", 1)))
cublas = bool(int(globals().get("cublas", 1)))
cudnn = bool(int(globals().get("cudnn", 1)))
edge_mask_convolutions = bool(int(globals().get("edge_mask_convolutions", 1)))
jit_convolutions = bool(int(globals().get("jit_convolutions", 1)))
max_aux_streams_arg = str(globals().get("max_aux_streams", ""))
max_aux_streams = None if max_aux_streams_arg == "" else int(max_aux_streams_arg)
max_tactics_arg = str(globals().get("max_tactics", ""))
max_tactics = None if max_tactics_arg == "" else int(max_tactics_arg)
tiling_optimization_level = int(globals().get("tiling_optimization_level", 0))
l2_limit_for_tiling = int(globals().get("l2_limit_for_tiling", -1))

clip = core.std.BlankClip(
    width=width,
    height=height,
    length=frames,
    fpsnum=24000,
    fpsden=1001,
    format=vs.RGBH if fp16io else vs.RGBS,
    color=[0.0625, 0.0625, 0.0625],
)

backend = Backend.TRT(
    fp16=True,
    force_fp16=force,
    output_format=1 if fp16io else 0,
    tf32=tf32,
    workspace=8192,
    use_cuda_graph=True,
    num_streams=streams,
    builder_optimization_level=5,
    use_cublas=cublas,
    use_cudnn=cudnn,
    use_edge_mask_convolutions=edge_mask_convolutions,
    use_jit_convolutions=jit_convolutions,
    static_shape=True,
    engine_folder=ENGINE_CACHE,
    max_aux_streams=max_aux_streams,
    max_tactics=max_tactics,
    tiling_optimization_level=tiling_optimization_level,
    l2_limit_for_tiling=l2_limit_for_tiling,
)

clip = inference(
    clip,
    network_path=model,
    overlap=(0, 0),
    tilesize=(width, height),
    backend=backend,
    input_name="input",
    batch_size=batch,
)

clip.set_output()
