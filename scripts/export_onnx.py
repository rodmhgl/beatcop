"""Export a Beat This! checkpoint (CPJKU, MIT license) to ONNX for beatcop.

One-time step; needs torch + beat_this, which beatcop itself does not:
    uv run --with torch --with "beat_this @ git+https://github.com/CPJKU/beat_this" \
        --with onnxscript python scripts/export_onnx.py
"""

import argparse
from pathlib import Path

import torch
from beat_this.inference import load_model

N_MELS = 128
CHUNK_FRAMES = 1500  # 30 s at 50 fps, the length the model was trained on


class OnnxWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, spect):
        out = self.m(spect)
        return out["beat"], out["downbeat"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="final0")
    parser.add_argument(
        "--out", type=Path, default=Path(__file__).parent.parent / "models" / "beatthis_final0.onnx"
    )
    args = parser.parse_args()

    model = OnnxWrapper(load_model(args.checkpoint, "cpu")).eval()
    dummy = torch.zeros(1, CHUNK_FRAMES, N_MELS)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        (dummy,),
        str(args.out),
        input_names=["spect"],
        output_names=["beat", "downbeat"],
        dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"}, "downbeat": {1: "frames"}},
        opset_version=18,
        dynamo=False,
    )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
