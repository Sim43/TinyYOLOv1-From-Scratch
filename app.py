# pyright: reportMissingImports=false
"""
Streamlit app for real-time inference on images/videos using *only*:
- .pt checkpoints (TinyYOLOv1-from-scratch)
- .engine TensorRT models (optional, if TensorRT runtime is installed)

Models are loaded from: runs/
"""

from __future__ import annotations

import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from src.model import TinyYOLOv1
from src.utils import decode_predictions, nms_normalized

BASE_DIR = Path(__file__).resolve().parent
RUNS_DIR = BASE_DIR / "runs"

SUPPORTED_IMAGE_EXTS = {"bmp", "jpeg", "jpg", "png", "webp"}
SUPPORTED_VIDEO_EXTS = {"mp4", "mkv", "avi", "mov", "webm", "mpeg", "mpg"}

PAGE_STYLE = """
<style>
.stApp { background: #0b1220; color: #e5e7eb; }
section[data-testid="stSidebar"] { background-color: #0f172a; }
</style>
"""


@dataclass(frozen=True)
class ModelInfo:
    name: str
    path: Path


# --------------------------- Model listing ---------------------------

def list_models() -> list[ModelInfo]:
    if not RUNS_DIR.exists():
        return []

    allowed = {".pt", ".engine"}
    files = sorted(
        p for p in RUNS_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in allowed
    )
    return [ModelInfo(p.name, p) for p in files]


# --------------------------- TinyYOLO (.pt) runner ---------------------------

class TinyYOLOCheckpointRunner:
    def __init__(self, model_path: Path, conf: float, iou: float):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model: Optional[TinyYOLOv1] = None
        self.S = 7
        self.C = 1
        self.img_size = 224

    def load(self) -> None:
        ckpt = torch.load(str(self.model_path), map_location="cpu", weights_only=True)

        cfg = ckpt.get("cfg", {})
        self.S = int(cfg.get("S", 7))
        self.C = int(cfg.get("C", 1))
        self.img_size = int(cfg.get("img_size", 224))

        self.model = TinyYOLOv1(S=self.S, C=self.C).to(self.device)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()

    @torch.no_grad()
    def infer_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        assert self.model is not None

        h0, w0 = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rs = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)

        x = torch.from_numpy(rs).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).to(self.device)

        pred = self.model(x)[0]  # (S,S,5+C)

        boxes, scores = decode_predictions(pred, S=self.S, conf_thresh=self.conf)
        keep = nms_normalized(boxes, scores, iou_thresh=self.iou)

        boxes = boxes[keep].cpu().numpy()
        scores = scores[keep].cpu().numpy()

        out = frame_bgr.copy()
        for (x1, y1, x2, y2), sc in zip(boxes, scores):
            X1, Y1 = int(x1 * w0), int(y1 * h0)
            X2, Y2 = int(x2 * w0), int(y2 * h0)
            cv2.rectangle(out, (X1, Y1), (X2, Y2), (0, 255, 0), 2)
            cv2.putText(out, f"person {sc:.2f}", (X1, max(0, Y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        return out


# --------------------------- TensorRT (.engine) optional runner ---------------------------

class TensorRTEngineRunner:
    """
    Minimal placeholder runner.
    If you want real engine inference, install TensorRT runtime and implement
    bindings + execution context here.

    This keeps the app simple while still listing .engine models.
    """
    def __init__(self, model_path: Path):
        self.model_path = model_path

    def load(self) -> None:
        try:
            import tensorrt  # noqa: F401
        except Exception as e:
            raise RuntimeError("TensorRT not installed. Cannot run .engine models.") from e

    def infer_bgr(self, frame_bgr: np.ndarray) -> np.ndarray:
        # If you want, I can drop a minimal TRT inference implementation next.
        raise RuntimeError("TensorRT inference not implemented in this minimal app.")


# --------------------------- Streamlit caching ---------------------------

@st.cache_resource(show_spinner=False)
def get_runner(model_path: str, conf: float, iou: float) -> Union[TinyYOLOCheckpointRunner, TensorRTEngineRunner]:
    p = Path(model_path)
    if p.suffix.lower() == ".pt":
        r = TinyYOLOCheckpointRunner(p, conf=conf, iou=iou)
        r.load()
        return r
    if p.suffix.lower() == ".engine":
        r = TensorRTEngineRunner(p)
        r.load()
        return r
    raise RuntimeError("Unsupported model type")


# --------------------------- Helpers ---------------------------

def save_uploaded_file(uploaded_file) -> Path:
    suffix = Path(uploaded_file.name).suffix
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return Path(tmp.name)

def read_video_frames(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    try:
        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()

def bgr_to_rgb(frame_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)


# --------------------------- UI ---------------------------

def main() -> None:
    st.set_page_config(page_title="TinyYOLO Scratch Inference", layout="wide")
    st.markdown(PAGE_STYLE, unsafe_allow_html=True)

    st.title("TinyYOLO (From Scratch) — Inference")
    st.caption("Loads only `.pt` (from-scratch) and `.engine` models from `runs/`.")

    with st.sidebar:
        st.subheader("Settings")
        conf = st.slider("Confidence", 0.0, 1.0, 0.35, 0.01)
        iou = st.slider("NMS IoU", 0.0, 1.0, 0.50, 0.01)
        show_fps = st.checkbox("Show FPS (video)", value=True)

    models = list_models()
    if not models:
        st.warning("No `.pt` or `.engine` models found in `runs/`.")
        st.info("Put your trained checkpoint here, e.g. `runs/yolov1_tiny_pennfudan.pt`.")
        return

    model_names = [m.name for m in models]
    chosen = st.selectbox("Select model (from runs/)", model_names)
    model_info = next(m for m in models if m.name == chosen)

    uploaded = st.file_uploader(
        "Upload image or video",
        type=sorted(SUPPORTED_IMAGE_EXTS | SUPPORTED_VIDEO_EXTS),
    )

    run = st.button("Run")
    if not run:
        return
    if uploaded is None:
        st.error("Upload an image or video first.")
        return

    try:
        runner = get_runner(str(model_info.path), conf, iou)
    except Exception as e:
        st.error(str(e))
        return

    suffix = uploaded.name.lower().split(".")[-1]

    col_in, col_out = st.columns(2, gap="large")
    with col_in:
        st.markdown("**Input**")
        input_ph = st.empty()
    with col_out:
        st.markdown("**Output**")
        out_ph = st.empty()
        status_ph = st.empty()

    # ---------------- image ----------------
    if suffix in SUPPORTED_IMAGE_EXTS:
        uploaded.seek(0)
        pil = Image.open(uploaded).convert("RGB")
        input_ph.image(pil, use_container_width=True)

        frame_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        t0 = time.perf_counter()
        try:
            out = runner.infer_bgr(frame_bgr)
        except Exception as e:
            st.error(str(e))
            return
        dt = (time.perf_counter() - t0) * 1000

        out_ph.image(bgr_to_rgb(out), use_container_width=True)
        status_ph.success(f"Done. Inference: {dt:.1f} ms")
        return

    # ---------------- video ----------------
    if suffix in SUPPORTED_VIDEO_EXTS:
        tmp_path = save_uploaded_file(uploaded)
        frames = 0
        t_start = time.perf_counter()

        try:
            for frame in read_video_frames(tmp_path):
                frames += 1
                input_ph.image(bgr_to_rgb(frame), caption=f"Frame {frames}", use_container_width=True)

                t0 = time.perf_counter()
                out = runner.infer_bgr(frame)
                infer_ms = (time.perf_counter() - t0) * 1000

                out_ph.image(bgr_to_rgb(out), use_container_width=True)

                if show_fps:
                    elapsed = time.perf_counter() - t_start
                    fps = frames / elapsed if elapsed > 0 else 0.0
                    status_ph.info(f"FPS: {fps:.2f} | Inference: {infer_ms:.1f} ms")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        status_ph.success("Video done.")
        return

    st.error(f"Unsupported file type: {uploaded.name}")


if __name__ == "__main__":
    main()
