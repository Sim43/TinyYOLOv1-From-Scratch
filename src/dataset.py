from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import cv2

from .utils import build_yolo_target

class PennFudanYOLODataset(Dataset):
    """
    Uses PennFudanPed:
      - images in PNGImages/
      - instance masks in PedMasks/ (each pedestrian has a unique integer id)
    We convert masks -> bounding boxes, then boxes -> YOLO grid target.
    """
    def __init__(self, root: str, S: int, C: int, img_size: int, train: bool = True, split: float = 0.8):
        self.root = Path(root)
        self.S = S
        self.C = C
        self.img_size = img_size

        img_dir = self.root / "PNGImages"
        mask_dir = self.root / "PedMasks"
        self.images = sorted(img_dir.glob("*.png"))
        self.masks = sorted(mask_dir.glob("*.png"))
        assert len(self.images) == len(self.masks), "Images and masks count mismatch"

        # Simple deterministic split
        n = len(self.images)
        idx_split = int(n * split)
        if train:
            self.images = self.images[:idx_split]
            self.masks = self.masks[:idx_split]
        else:
            self.images = self.images[idx_split:]
            self.masks = self.masks[idx_split:]

    def __len__(self):
        return len(self.images)

    def _read_image(self, p: Path) -> np.ndarray:
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"Failed to read image: {p}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _read_mask(self, p: Path) -> np.ndarray:
        m = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise RuntimeError(f"Failed to read mask: {p}")
        return m

    def _mask_to_boxes(self, mask: np.ndarray) -> np.ndarray:
        # mask pixels are 0 for background, and 1..N for instances
        ids = np.unique(mask)
        ids = ids[ids != 0]
        boxes = []
        for obj_id in ids:
            ys, xs = np.where(mask == obj_id)
            if len(xs) == 0 or len(ys) == 0:
                continue
            x1, x2 = xs.min(), xs.max()
            y1, y2 = ys.min(), ys.max()
            boxes.append([x1, y1, x2, y2])
        return np.array(boxes, dtype=np.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img = self._read_image(self.images[idx])
        mask = self._read_mask(self.masks[idx])
        h0, w0 = img.shape[:2]

        boxes = self._mask_to_boxes(mask)  # (N,4) in original pixels

        # Resize image and scale boxes
        img_resized = cv2.resize(img, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        sx = self.img_size / float(w0)
        sy = self.img_size / float(h0)
        if boxes.size > 0:
            boxes[:, [0, 2]] *= sx
            boxes[:, [1, 3]] *= sy

        # Convert to normalized cx,cy,w,h in [0,1]
        yolo_boxes = []
        for (x1, y1, x2, y2) in boxes:
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            bw = max(1.0, (x2 - x1))
            bh = max(1.0, (y2 - y1))
            yolo_boxes.append([cx / self.img_size, cy / self.img_size, bw / self.img_size, bh / self.img_size])
        yolo_boxes = np.array(yolo_boxes, dtype=np.float32) if len(yolo_boxes) else np.zeros((0, 4), dtype=np.float32)

        # One-class (pedestrian) => class_id=0 for all boxes
        class_ids = np.zeros((yolo_boxes.shape[0],), dtype=np.int64)

        target = build_yolo_target(
            boxes=yolo_boxes,
            class_ids=class_ids,
            S=self.S,
            C=self.C,
        )  # (S,S,5+C)

        # To tensor
        img_t = torch.from_numpy(img_resized).float() / 255.0
        img_t = img_t.permute(2, 0, 1).contiguous()  # (3,H,W)

        target_t = torch.from_numpy(target).float()
        return {"image": img_t, "target": target_t}
