import argparse
import json
from pathlib import Path

import numpy as np
import torch


def tensor_to_numpy(tensor: torch.Tensor, dtype: str) -> np.ndarray:
    arr = tensor.detach().cpu().contiguous().numpy()
    if dtype == "fp16":
        return arr.astype(np.float16)
    if dtype == "fp32":
        return arr.astype(np.float32)
    raise ValueError(dtype)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--dtype", choices=("fp16", "fp32"), default="fp16")
    args = parser.parse_args()

    ckpt = torch.load(args.pth, map_location="cpu")
    state = ckpt.get("params_ema") or ckpt.get("params") or ckpt
    args.out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {"dtype": args.dtype, "layers": []}
    for idx in range(18):
        conv_base = f"body.{idx * 2 if idx < 17 else 34}"
        weight_name = f"{conv_base}.weight"
        bias_name = f"{conv_base}.bias"
        weight = tensor_to_numpy(state[weight_name], args.dtype)
        bias = tensor_to_numpy(state[bias_name], args.dtype)
        weight_path = args.out_dir / f"conv{idx:02d}_weight_{args.dtype}.bin"
        bias_path = args.out_dir / f"conv{idx:02d}_bias_{args.dtype}.bin"
        weight.tofile(weight_path)
        bias.tofile(bias_path)
        layer = {
            "index": idx,
            "weight": weight_path.name,
            "bias": bias_path.name,
            "weight_shape": list(weight.shape),
            "bias_shape": list(bias.shape),
        }
        if idx < 17:
            prelu_name = f"body.{idx * 2 + 1}.weight"
            prelu = tensor_to_numpy(state[prelu_name], args.dtype)
            prelu_path = args.out_dir / f"prelu{idx:02d}_{args.dtype}.bin"
            prelu.tofile(prelu_path)
            layer["prelu"] = prelu_path.name
            layer["prelu_shape"] = list(prelu.shape)
        manifest["layers"].append(layer)

    (args.out_dir / f"manifest_{args.dtype}.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(args.out_dir / f"manifest_{args.dtype}.json")


if __name__ == "__main__":
    main()
