import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"
FFMPEG = TOOLS / "ffmpeg" / "ffmpeg-8.1.1-essentials_build" / "bin" / "ffmpeg.exe"
FFPROBE = TOOLS / "ffmpeg" / "ffmpeg-8.1.1-essentials_build" / "bin" / "ffprobe.exe"
REALESRGAN = TOOLS / "realesrgan" / "realesrgan-ncnn-vulkan.exe"
REALESRGAN_MODELS = TOOLS / "realesrgan" / "models"


def now_iso():
    return datetime.now().isoformat(timespec="seconds")


def safe_name(name: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    return value or "video_ai_job"


def rate_to_float(rate: str) -> float:
    if "/" in rate:
        a, b = rate.split("/", 1)
        return float(a) / float(b)
    return float(rate)


def round_frame(seconds: float, fps: float) -> int:
    return int(math.floor(seconds * fps + 0.5))


def under_root(path: Path, root: Path) -> bool:
    path_resolved = path.resolve()
    root_resolved = root.resolve()
    try:
        path_resolved.relative_to(root_resolved)
        return True
    except ValueError:
        return False


def reset_dir(path: Path, work_root: Path):
    if path.exists():
        if not under_root(path, work_root):
            raise RuntimeError(f"Refusing to remove outside work root: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def remove_dir(path: Path, work_root: Path):
    if path.exists():
        if not under_root(path, work_root):
            raise RuntimeError(f"Refusing to remove outside work root: {path}")
        shutil.rmtree(path)


def log_line(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", errors="replace") as f:
        f.write(text + "\n")


def run_capture(args):
    return subprocess.run(args, text=True, capture_output=True, check=True)


def probe_video(video: Path):
    p = run_capture([
        str(FFPROBE), "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration:stream=width,height,r_frame_rate,avg_frame_rate,nb_frames:stream_tags=NUMBER_OF_FRAMES",
        "-of", "json",
        str(video),
    ])
    data = json.loads(p.stdout)
    stream = data["streams"][0]
    rate = stream.get("avg_frame_rate") or stream.get("r_frame_rate")
    if not rate or rate == "0/0":
        rate = stream["r_frame_rate"]
    fps = rate_to_float(rate)
    duration = float(data["format"]["duration"])
    expected = round_frame(duration, fps)
    frame_count = expected
    tags = stream.get("tags") or {}
    if tags.get("NUMBER_OF_FRAMES"):
        frame_count = int(tags["NUMBER_OF_FRAMES"])
    elif stream.get("nb_frames"):
        frame_count = int(stream["nb_frames"])
    if expected > 0 and abs(frame_count - expected) > max(2, int(expected * 0.05)):
        frame_count = expected
    return {
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "rate": rate,
        "fps": fps,
        "duration": duration,
        "frame_count": frame_count,
    }


def parse_kept_indices(log_path: Path, fps: float, target_frames: int):
    text = log_path.read_text(encoding="utf-8", errors="replace").replace("\x00", "")
    indices = []
    for m in re.finditer(r"pts_time:([0-9]+(?:\.[0-9]+)?)", text):
        idx = round_frame(float(m.group(1)), fps)
        idx = max(0, min(target_frames - 1, idx))
        if not indices or indices[-1] != idx:
            indices.append(idx)
    if not indices:
        raise RuntimeError(f"Could not parse kept frame timestamps from {log_path}")
    return indices


def png_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.glob("*.png"))


def png_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size < 20:
        return False
    with path.open("rb") as f:
        data = f.read(8)
        if data != b"\x89PNG\r\n\x1a\n":
            return False
        f.seek(-12, os.SEEK_END)
        return f.read(12) == b"\x00\x00\x00\x00IEND\xaeB`\x82"


def write_concat(files, list_path: Path):
    lines = []
    for file in files:
        escaped = str(file).replace("'", "'\\''")
        lines.append(f"file '{escaped}'")
    list_path.write_text("\n".join(lines) + "\n", encoding="ascii")


class StageProcess:
    def __init__(self, name, chunk, args, log_path):
        self.name = name
        self.chunk = chunk
        self.args = args
        self.log_path = log_path
        self.start = time.time()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("ab") as f:
            f.write(f"\n===== {now_iso()} =====\n".encode("utf-8"))
            f.write((" ".join(map(str, args)) + "\n").encode("utf-8"))
        self.log_file = log_path.open("ab")
        self.proc = subprocess.Popen(args, stdout=self.log_file, stderr=subprocess.STDOUT)

    def poll(self):
        return self.proc.poll()

    def close(self):
        self.log_file.close()


def chunk_paths(ep_dir: Path, name: str):
    cdir = ep_dir / name
    return {
        "dir": cdir,
        "unique": cdir / "unique",
        "sr": cdir / "sr_unique",
        "pending": cdir / "sr_pending",
        "full": cdir / "full",
        "video": cdir / f"{name}.av1.mkv",
        "done": cdir / "done.txt",
        "meta": cdir / "pipeline_meta.json",
        "map": cdir / "frame_map.csv",
    }


def is_encoded(paths):
    return paths["done"].exists() and paths["video"].exists() and paths["video"].stat().st_size > 0


def start_extract(chunk, video, info, opts, paths, logs, work_root):
    reset_dir(paths["unique"], work_root)
    log_path = logs / f"{chunk['name']}.extract_unique.pipeline.log"
    if log_path.exists():
        log_path.unlink()
    mp = "mpdecimate" if not opts.mpdecimate_options.strip() else f"mpdecimate={opts.mpdecimate_options}"
    filt = f"setpts=PTS-STARTPTS,{mp},showinfo,format=rgb24"
    args = [
        str(FFMPEG), "-y", "-hide_banner",
        "-threads", "0",
        "-ss", f"{chunk['start']:.8f}",
        "-t", f"{chunk['duration']:.8f}",
        "-i", str(video),
        "-map", "0:v:0",
        "-vf", filt,
        "-fps_mode", "passthrough",
        str(paths["unique"] / "%08d.png"),
    ]
    return StageProcess("extract", chunk, args, log_path)


def finish_extract(sp, info, paths, main_log):
    elapsed = time.time() - sp.start
    kept = parse_kept_indices(sp.log_path, info["fps"], sp.chunk["target_frames"])
    unique = png_count(paths["unique"])
    if unique != len(kept):
        raise RuntimeError(f"{sp.chunk['name']} unique mismatch images={unique} showinfo={len(kept)}")
    reused = sp.chunk["target_frames"] - unique
    meta = {
        **sp.chunk,
        "kept_indices": kept,
        "unique_frames": unique,
        "reused_frames": reused,
        "extract_seconds": elapsed,
    }
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    log_line(main_log, f"[{sp.chunk['name']}] uniqueFrames={unique}, reusedFrames={reused} ({100.0 * reused / sp.chunk['target_frames']:.6f}%)")
    return meta


def start_sr(meta, opts, paths, logs, work_root):
    paths["sr"].mkdir(parents=True, exist_ok=True)
    reset_dir(paths["pending"], work_root)
    missing = 0
    for i in range(1, meta["unique_frames"] + 1):
        name = f"{i:08d}.png"
        sr_frame = paths["sr"] / name
        if not png_complete(sr_frame):
            if sr_frame.exists():
                sr_frame.unlink()
            src = paths["unique"] / name
            dst = paths["pending"] / name
            os.link(src, dst)
            missing += 1
    if missing == 0:
        return None
    log_path = logs / f"{meta['name']}.realesrgan.pipeline.log"
    args = [
        str(REALESRGAN),
        "-i", str(paths["pending"]),
        "-o", str(paths["sr"]),
        "-m", str(REALESRGAN_MODELS),
        "-n", opts.model_name,
        "-s", str(opts.scale),
        "-t", str(opts.realesrgan_tile),
        "-g", str(opts.gpu_id),
        "-j", opts.realesrgan_jobs,
        "-f", "png",
    ]
    return StageProcess("sr", meta, args, log_path)


def finish_sr(sp, paths, main_log):
    elapsed = 0.0 if sp is None else time.time() - sp.start
    if sp is not None:
        meta = sp.chunk
    else:
        meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    count = png_count(paths["sr"])
    if count != meta["unique_frames"]:
        raise RuntimeError(f"{meta['name']} SR count mismatch expected={meta['unique_frames']} got={count}")
    remove_dir(paths["pending"], paths["dir"].parents[1])
    meta["sr_seconds"] = elapsed
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    log_line(main_log, f"[{meta['name']}] srUniqueFrames={count}, srSeconds={elapsed:.6f}")
    return meta


def rebuild_full(meta, paths, work_root):
    reset_dir(paths["full"], work_root)
    lines = ["frame,unique_frame,kept_source_frame"]
    cursor = 0
    kept = meta["kept_indices"]
    for frame in range(meta["target_frames"]):
        while cursor + 1 < len(kept) and kept[cursor + 1] <= frame:
            cursor += 1
        unique_num = cursor + 1
        src = paths["sr"] / f"{unique_num:08d}.png"
        dst = paths["full"] / f"{frame + 1:08d}.png"
        os.link(src, dst)
        lines.append(f"{frame + 1},{unique_num},{kept[cursor] + 1}")
    paths["map"].write_text("\n".join(lines) + "\n", encoding="ascii")


def start_encode(meta, info, opts, paths, logs, work_root):
    rebuild_start = time.time()
    rebuild_full(meta, paths, work_root)
    meta["rebuild_seconds"] = time.time() - rebuild_start
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    log_path = logs / f"{meta['name']}.encode.pipeline.log"
    args = [
        str(FFMPEG), "-y", "-hide_banner",
        "-framerate", info["rate"],
        "-i", str(paths["full"] / "%08d.png"),
        "-c:v", "av1_nvenc",
        "-preset", "p7",
        "-tune", "uhq",
        "-rc", "constqp",
        "-qp", "1",
        "-pix_fmt", "p010le",
        "-color_primaries", "bt709",
        "-color_trc", "bt709",
        "-colorspace", "bt709",
        "-color_range", "tv",
        "-an",
        str(paths["video"]),
    ]
    return StageProcess("encode", meta, args, log_path)


def finish_encode(sp, paths, main_log, work_root, keep_frames=False):
    elapsed = time.time() - sp.start
    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
    if not paths["video"].exists() or paths["video"].stat().st_size == 0:
        raise RuntimeError(f"Missing encoded chunk video: {paths['video']}")
    meta["encode_seconds"] = elapsed
    total = meta["extract_seconds"] + meta.get("sr_seconds", 0.0) + meta.get("rebuild_seconds", 0.0) + elapsed
    paths["done"].write_text(f"done {now_iso()}\n", encoding="ascii")
    paths["meta"].write_text(json.dumps(meta), encoding="utf-8")
    log_line(main_log, f"[{meta['name']}] completed elapsedSeconds={total:.6f}, extractSeconds={meta['extract_seconds']:.6f}, srSeconds={meta.get('sr_seconds', 0.0):.6f}, rebuildSeconds={meta.get('rebuild_seconds', 0.0):.6f}, encodeSeconds={elapsed:.6f}")
    if not keep_frames:
        remove_dir(paths["unique"], work_root)
        remove_dir(paths["sr"], work_root)
        remove_dir(paths["pending"], work_root)
        remove_dir(paths["full"], work_root)


def process_episode(video: Path, opts):
    info = probe_video(video)
    base_name = video.stem
    ep_dir = opts.work_root / safe_name(base_name)
    logs = ep_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    main_log = logs / "pipeline.log"
    output = opts.output_dir / f"{base_name} [RealESRGAN-{opts.scale}x-dedup][AV1-NVENC-highest].mkv"
    ep_dir.mkdir(parents=True, exist_ok=True)
    if output.exists():
        print(f"SKIP existing output: {output}", flush=True)
        return
    log_line(main_log, f"\n===== {now_iso()} pipeline-optimized =====")
    log_line(main_log, f"Input: {video}")
    log_line(main_log, f"Output: {output}")
    log_line(main_log, f"Source: {info['width']}x{info['height']}, fps={info['rate']}, frames={info['frame_count']}, duration={info['duration']:.6f}")
    log_line(main_log, f"RealESRGAN: model={opts.model_name}, tile={opts.realesrgan_tile}, jobs={opts.realesrgan_jobs}, gpu={opts.gpu_id}")
    log_line(main_log, "AV1 mode: nvenc-highest")
    frames_per_chunk = max(1, round_frame(opts.chunk_seconds, info["fps"]))
    chunks = []
    idx = 0
    for start_frame in range(0, info["frame_count"], frames_per_chunk):
        target = min(frames_per_chunk, info["frame_count"] - start_frame)
        chunks.append({
            "index": idx,
            "name": f"chunk_{idx:04d}",
            "start_frame": start_frame,
            "target_frames": target,
            "start": start_frame / info["fps"],
            "duration": target / info["fps"],
        })
        idx += 1

    extract_proc = None
    sr_proc = None
    encode_proc = None
    extracted = set()
    sr_done = set()
    encoding_or_done = set()
    extract_cursor = 0

    for chunk in chunks:
        paths = chunk_paths(ep_dir, chunk["name"])
        if is_encoded(paths):
            extracted.add(chunk["index"])
            sr_done.add(chunk["index"])
            encoding_or_done.add(chunk["index"])

    while len(encoding_or_done) < len(chunks):
        # Finish processes first.
        for proc_name in ("extract", "sr", "encode"):
            sp = {"extract": extract_proc, "sr": sr_proc, "encode": encode_proc}[proc_name]
            if sp is not None and sp.poll() is not None:
                sp.close()
                if sp.proc.returncode != 0:
                    raise RuntimeError(f"{sp.name} failed for {sp.chunk['name']} rc={sp.proc.returncode}, log={sp.log_path}")
                paths = chunk_paths(ep_dir, sp.chunk["name"])
                if sp.name == "extract":
                    finish_extract(sp, info, paths, main_log)
                    extracted.add(sp.chunk["index"])
                    extract_proc = None
                elif sp.name == "sr":
                    finish_sr(sp, paths, main_log)
                    sr_done.add(sp.chunk["index"])
                    sr_proc = None
                else:
                    finish_encode(sp, paths, main_log, opts.work_root, opts.keep_frames)
                    encoding_or_done.add(sp.chunk["index"])
                    encode_proc = None

        # Start encode for earliest SR-ready chunk.
        if encode_proc is None:
            for chunk in chunks:
                if chunk["index"] in sr_done and chunk["index"] not in encoding_or_done:
                    paths = chunk_paths(ep_dir, chunk["name"])
                    if is_encoded(paths):
                        encoding_or_done.add(chunk["index"])
                        continue
                    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
                    encode_proc = start_encode(meta, info, opts, paths, logs, opts.work_root)
                    break

        # Start SR for earliest extracted chunk.
        if sr_proc is None:
            for chunk in chunks:
                if chunk["index"] in extracted and chunk["index"] not in sr_done:
                    paths = chunk_paths(ep_dir, chunk["name"])
                    meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
                    maybe = start_sr(meta, opts, paths, logs, opts.work_root)
                    if maybe is None:
                        finish_sr(None, paths, main_log)
                        sr_done.add(chunk["index"])
                    else:
                        sr_proc = maybe
                    break

        # Start extraction for next chunk not extracted/encoded.
        if extract_proc is None:
            while extract_cursor < len(chunks):
                chunk = chunks[extract_cursor]
                extract_cursor += 1
                paths = chunk_paths(ep_dir, chunk["name"])
                if chunk["index"] in extracted or is_encoded(paths):
                    continue
                log_line(main_log, f"[{chunk['name']}] startFrame={chunk['start_frame'] + 1}, targetFrames={chunk['target_frames']}, start={chunk['start']:.8f}, duration={chunk['duration']:.8f}")
                extract_proc = start_extract(chunk, video, info, opts, paths, logs, opts.work_root)
                break

        time.sleep(1)

    # mux episode
    chunk_files = [chunk_paths(ep_dir, c["name"])["video"] for c in chunks]
    concat_list = ep_dir / "chunks.txt"
    write_concat(chunk_files, concat_list)
    mux_log = logs / "mux.pipeline.log"
    mux_args = [
        str(FFMPEG), "-y", "-hide_banner",
        "-f", "concat", "-safe", "0", "-i", str(concat_list),
        "-i", str(video),
        "-map", "0:v:0",
        "-map", "1:a?",
        "-map", "1:s?",
        "-map", "1:t?",
        "-map_metadata", "1",
        "-map_chapters", "1",
        "-c", "copy",
        str(output),
    ]
    with mux_log.open("ab") as f:
        f.write(f"\n===== {now_iso()} =====\n".encode("utf-8"))
        f.write((" ".join(mux_args) + "\n").encode("utf-8"))
        subprocess.run(mux_args, stdout=f, stderr=subprocess.STDOUT, check=True)
    log_line(main_log, f"Completed: {output}")
    print(f"DONE: {output}", flush=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--work-root", required=True)
    p.add_argument("--chunk-seconds", type=int, default=120)
    p.add_argument("--scale", type=int, default=2)
    p.add_argument("--model-name", default="realesr-animevideov3")
    p.add_argument("--gpu-id", type=int, default=0)
    p.add_argument("--realesrgan-tile", type=int, default=2048)
    p.add_argument("--realesrgan-jobs", default="24:12:24")
    p.add_argument("--mpdecimate-options", default="hi=768:lo=320:frac=0.33")
    p.add_argument("--only-name-like", default="")
    p.add_argument("--keep-frames", action="store_true")
    return p.parse_args()


def main():
    opts = parse_args()
    opts.input_dir = Path(opts.input_dir)
    opts.output_dir = Path(opts.output_dir)
    opts.work_root = Path(opts.work_root)
    opts.output_dir.mkdir(parents=True, exist_ok=True)
    opts.work_root.mkdir(parents=True, exist_ok=True)
    for required in (FFMPEG, FFPROBE, REALESRGAN, REALESRGAN_MODELS):
        if not required.exists():
            raise FileNotFoundError(required)
    videos = sorted(opts.input_dir.glob("*.mkv"))
    if opts.only_name_like:
        import fnmatch
        videos = [v for v in videos if fnmatch.fnmatch(v.name, opts.only_name_like)]
    if not videos:
        raise RuntimeError("No input videos found")
    print(f"Input files: {len(videos)}", flush=True)
    print(f"Output directory: {opts.output_dir}", flush=True)
    print(f"Work directory: {opts.work_root}", flush=True)
    print("Pipeline mode: extract + RealESRGAN + NVENC overlap", flush=True)
    for video in videos:
        process_episode(video, opts)


if __name__ == "__main__":
    main()
