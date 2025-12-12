#!/usr/bin/env python3
"""
Convert a TinyYOLOv1 PyTorch checkpoint (.pt / .pth)
directly into a TensorRT .engine on Jetson.

Usage:
  python3 pth_to_engine.py \
      --input  runs/yolov1_tiny_pennfudan.pt \
      --output runs/yolov1_tiny_pennfudan.engine
"""

import argparse
import subprocess
import tempfile
from pathlib import Path

import torch

# ---- import your model ----
from src.model import TinyYOLOv1


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help="Path to .pt/.pth model")
    p.add_argument("--output", required=True, help="Path to output .engine")
    p.add_argument("--fp16", action="store_true", default=True, help="Enable FP16 (recommended on Jetson)")
    p.add_argument("--workspace", type=int, default=2048, help="TensorRT workspace (MB)")
    return p.parse_args()


def export_onnx(pt_path: Path, onnx_path: Path):
    ckpt = torch.load(pt_path, map_location="cpu")

    if "model_state" not in ckpt or "cfg" not in ckpt:
        raise RuntimeError(
            "Checkpoint must contain keys: 'model_state' and 'cfg'"
        )

    cfg = ckpt["cfg"]
    S = int(cfg.get("S", 7))
    C = int(cfg.get("C", 1))
    img_size = int(cfg.get("img_size", 224))

    model = TinyYOLOv1(S=S, C=C)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    dummy = torch.zeros(1, 3, img_size, img_size)

    torch.onnx.export(
        model,
        dummy,
        onnx_path.as_posix(),
        opset_version=13,
        input_names=["images"],
        output_names=["predictions"],
        do_constant_folding=True,
        dynamic_axes=None,   # IMPORTANT for TensorRT
    )


def build_engine(onnx_path: Path, engine_path: Path, fp16: bool, workspace: int):
    trtexec = "/usr/src/tensorrt/bin/trtexec"
    if not Path(trtexec).exists():
        raise RuntimeError("trtexec not found. Is TensorRT installed?")

    cmd = [
        trtexec,
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--workspace={workspace}",
        "--buildOnly",
    ]

    if fp16:
        cmd.append("--fp16")

    print("[INFO] Running TensorRT build:")
    print(" ".join(cmd))

    subprocess.run(cmd, check=True)


def main():
    args = parse_args()

    pt_path = Path(args.input).resolve()
    engine_path = Path(args.output).resolve()

    if not pt_path.exists():
        raise FileNotFoundError(pt_path)

    engine_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        onnx_path = Path(tmpdir) / "model.onnx"

        print("[1/2] Exporting ONNX...")
        export_onnx(pt_path, onnx_path)

        print("[2/2] Building TensorRT engine...")
        build_engine(
            onnx_path=onnx_path,
            engine_path=engine_path,
            fp16=args.fp16,
            workspace=args.workspace,
        )

    print(f"[DONE] Engine saved to: {engine_path}")


if __name__ == "__main__":
    main()
