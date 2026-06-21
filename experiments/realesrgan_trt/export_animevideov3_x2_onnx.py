import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    ):
        super().__init__()
        if act_type != "prelu":
            raise ValueError("This exporter is specialized for realesr-animevideov3 PReLU weights.")

        body = []
        body.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        body.append(nn.PReLU(num_parameters=num_feat))
        for _ in range(num_conv):
            body.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            body.append(nn.PReLU(num_parameters=num_feat))
        body.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        body.append(nn.PixelShuffle(upscale))
        self.body = nn.Sequential(*body)
        self.upscale = upscale

    def forward(self, x):
        out = self.body(x)
        base = F.interpolate(x, scale_factor=self.upscale, mode="nearest")
        return out + base


class RealESRGANAnimeVideoX2(nn.Module):
    def __init__(self, down_mode="bicubic"):
        super().__init__()
        self.model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
        self.down_mode = down_mode

    def forward(self, x):
        y = self.model(x)
        if self.down_mode == "nearest":
            return F.interpolate(y, scale_factor=0.5, mode="nearest")
        if self.down_mode == "bilinear":
            return F.interpolate(y, scale_factor=0.5, mode="bilinear", align_corners=False)
        if self.down_mode == "bicubic":
            return F.interpolate(y, scale_factor=0.5, mode="bicubic", align_corners=False)
        raise RuntimeError(f"Unsupported downsample mode: {self.down_mode}")


def load_state(model, pth_path):
    checkpoint = torch.load(pth_path, map_location="cpu")
    state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
    missing, unexpected = model.model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--dummy-h", type=int, default=64)
    parser.add_argument("--dummy-w", type=int, default=64)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument(
        "--down-mode",
        choices=("nearest", "bilinear", "bicubic"),
        default="bicubic",
        help="ncnn x2 param uses Interp resize_type=3 after the x4 model; bicubic is the closest export.",
    )
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    model = RealESRGANAnimeVideoX2(down_mode=args.down_mode)
    load_state(model, args.pth)
    model.eval()

    dummy = torch.randn(1, 3, args.dummy_h, args.dummy_w, dtype=torch.float32)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {2: "height", 3: "width"},
            "output": {2: "out_height", 3: "out_width"},
        },
    )
    print(f"Exported {args.onnx}")


if __name__ == "__main__":
    main()
