import argparse
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F


class SRVGGTrunk(nn.Module):
    def __init__(self):
        super().__init__()
        body = [nn.Conv2d(3, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
        for _ in range(16):
            body += [nn.Conv2d(64, 64, 3, 1, 1), nn.PReLU(num_parameters=64)]
        body.append(nn.Conv2d(64, 48, 3, 1, 1))
        self.body = nn.Sequential(*body)

    def forward(self, x):
        return self.body(x)


def original_tail(z, x):
    y = F.pixel_shuffle(z, 4)
    y = y + F.interpolate(x, scale_factor=4, mode="nearest")
    return F.interpolate(y, scale_factor=0.5, mode="bicubic", align_corners=False)


def derive_tail_kernel(in_channels, source):
    """Derive a 3x3 low-resolution kernel for source='z' or source='x'."""
    h = w = 7
    cy = cx = 3
    weight = torch.zeros(12, in_channels, 3, 3, dtype=torch.float32)

    for in_ch in range(in_channels):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                z = torch.zeros(1, 48, h, w, dtype=torch.float32)
                x = torch.zeros(1, 3, h, w, dtype=torch.float32)
                if source == "z":
                    z[0, in_ch, cy + dy, cx + dx] = 1.0
                elif source == "x":
                    x[0, in_ch, cy + dy, cx + dx] = 1.0
                else:
                    raise ValueError(source)

                out = original_tail(z, x)
                for color in range(3):
                    for py in range(2):
                        for px in range(2):
                            out_ch = color * 4 + py * 2 + px
                            weight[out_ch, in_ch, dy + 1, dx + 1] = out[
                                0, color, cy * 2 + py, cx * 2 + px
                            ]

    return weight


class FusedTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("z_weight", derive_tail_kernel(48, "z"))
        self.register_buffer("x_weight", derive_tail_kernel(3, "x"))
        top_idx = []
        bottom_idx = []
        left_idx = []
        right_idx = []
        for color in range(3):
            for vy in range(4):
                for vx in range(4):
                    ch = color * 16 + vy * 4 + vx
                    top_idx.append(color * 16 + (0 if vy == 3 else vy) * 4 + vx)
                    bottom_idx.append(color * 16 + (3 if vy == 0 else vy) * 4 + vx)
                    left_idx.append(color * 16 + vy * 4 + (0 if vx == 3 else vx))
                    right_idx.append(color * 16 + vy * 4 + (3 if vx == 0 else vx))
        self.register_buffer("top_idx", torch.tensor(top_idx, dtype=torch.long))
        self.register_buffer("bottom_idx", torch.tensor(bottom_idx, dtype=torch.long))
        self.register_buffer("left_idx", torch.tensor(left_idx, dtype=torch.long))
        self.register_buffer("right_idx", torch.tensor(right_idx, dtype=torch.long))

    def phase_pad_z(self, z):
        top = z.index_select(1, self.top_idx)[:, :, :1, :]
        bottom = z.index_select(1, self.bottom_idx)[:, :, -1:, :]
        z = torch.cat((top, z, bottom), dim=2)
        left = z.index_select(1, self.left_idx)[:, :, :, :1]
        right = z.index_select(1, self.right_idx)[:, :, :, -1:]
        return torch.cat((left, z, right), dim=3)

    def forward(self, z, x):
        z_pad = self.phase_pad_z(z)
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        low = F.conv2d(z_pad, self.z_weight) + F.conv2d(x_pad, self.x_weight)
        return F.pixel_shuffle(low, 2)


class RealESRGANAnimeVideoX2FusedTail(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = SRVGGTrunk()
        self.tail = FusedTail()

    def forward(self, x):
        return self.tail(self.trunk(x), x)


def load_state(model, pth_path):
    checkpoint = torch.load(pth_path, map_location="cpu")
    state_dict = checkpoint.get("params_ema") or checkpoint.get("params") or checkpoint
    missing, unexpected = model.trunk.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        raise RuntimeError(f"State dict mismatch. missing={missing}, unexpected={unexpected}")


def verify_tail():
    torch.manual_seed(1234)
    tail = FusedTail().eval()
    for h, w in ((5, 6), (8, 8), (17, 19)):
        z = torch.randn(1, 48, h, w)
        x = torch.randn(1, 3, h, w)
        ref = original_tail(z, x)
        got = tail(z, x)
        diff = (ref - got).abs()
        print(
            f"verify {h}x{w}: max_abs={diff.max().item():.8g} "
            f"mean_abs={diff.mean().item():.8g}"
        )
        torch.testing.assert_close(got, ref, rtol=2e-5, atol=2e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", required=True, type=Path)
    parser.add_argument("--onnx", required=True, type=Path)
    parser.add_argument("--dummy-b", type=int, default=1)
    parser.add_argument("--dummy-h", type=int, default=64)
    parser.add_argument("--dummy-w", type=int, default=64)
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--dynamic-batch", action="store_true")
    args = parser.parse_args()

    torch.set_grad_enabled(False)
    verify_tail()

    model = RealESRGANAnimeVideoX2FusedTail().eval()
    load_state(model, args.pth)

    dummy = torch.randn(args.dummy_b, 3, args.dummy_h, args.dummy_w, dtype=torch.float32)
    args.onnx.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = {
        "input": {2: "height", 3: "width"},
        "output": {2: "out_height", 3: "out_width"},
    }
    if args.dynamic_batch:
        dynamic_axes["input"][0] = "batch"
        dynamic_axes["output"][0] = "batch"

    torch.onnx.export(
        model,
        dummy,
        args.onnx,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )
    print(f"Exported {args.onnx}")


if __name__ == "__main__":
    main()
