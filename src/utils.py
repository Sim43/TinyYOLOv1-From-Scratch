from typing import Tuple
import numpy as np
import torch
from torchvision.ops import nms

def build_yolo_target(boxes: np.ndarray, class_ids: np.ndarray, S: int, C: int) -> np.ndarray:
    """
    Build YOLOv1-style target tensor of shape (S,S,5+C):
      [tx, ty, tw, th, obj] + one-hot class(C)
    Where:
      tx,ty are in [0,1] relative to cell
      tw,th are in [0,1] relative to full image
      obj is 1 if an object assigned to the cell else 0
    If multiple objects map to same cell, keep the one with larger area.
    """
    tgt = np.zeros((S, S, 5 + C), dtype=np.float32)
    if boxes.shape[0] == 0:
        return tgt

    for b, cls in zip(boxes, class_ids):
        cx, cy, w, h = b  # all normalized to [0,1] relative to full image
        if w <= 0 or h <= 0:
            continue

        gi = int(np.clip(cx * S, 0, S - 1))
        gj = int(np.clip(cy * S, 0, S - 1))

        area = w * h
        existing_obj = tgt[gj, gi, 4]
        if existing_obj > 0:
            ex_w = tgt[gj, gi, 2]
            ex_h = tgt[gj, gi, 3]
            if (ex_w * ex_h) >= area:
                continue

        cell_x = cx * S - gi
        cell_y = cy * S - gj

        tgt[gj, gi, 0] = cell_x
        tgt[gj, gi, 1] = cell_y
        tgt[gj, gi, 2] = w
        tgt[gj, gi, 3] = h
        tgt[gj, gi, 4] = 1.0

        onehot = np.zeros((C,), dtype=np.float32)
        onehot[int(cls)] = 1.0
        tgt[gj, gi, 5:] = onehot
    return tgt

def decode_predictions(pred: torch.Tensor, S: int, conf_thresh: float = 0.4) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    pred: (S,S,5+C) for a single image
    returns:
      boxes_xyxy in [0,1] (N,4), scores (N,)
    One class only (C=1), so score = obj * class_prob.
    """
    if pred.dim() != 3:
        raise ValueError("Expected pred shape (S,S,5+C)")

    tx = pred[..., 0].sigmoid()
    ty = pred[..., 1].sigmoid()
    tw = pred[..., 2].sigmoid()
    th = pred[..., 3].sigmoid()
    obj = pred[..., 4].sigmoid()

    # class prob (C=1) -> sigmoid
    cls = pred[..., 5:].sigmoid()  # (S,S,C)
    cls_prob = cls[..., 0] if cls.numel() else torch.ones_like(obj)

    score = obj * cls_prob

    ys, xs = torch.where(score > conf_thresh)
    if ys.numel() == 0:
        return torch.zeros((0, 4), device=pred.device), torch.zeros((0,), device=pred.device)

    # Convert cell-relative to image-relative cx,cy
    cx = (xs.float() + tx[ys, xs]) / float(S)
    cy = (ys.float() + ty[ys, xs]) / float(S)
    w = tw[ys, xs]
    h = th[ys, xs]

    x1 = (cx - w / 2).clamp(0, 1)
    y1 = (cy - h / 2).clamp(0, 1)
    x2 = (cx + w / 2).clamp(0, 1)
    y2 = (cy + h / 2).clamp(0, 1)

    boxes = torch.stack([x1, y1, x2, y2], dim=1)
    scores = score[ys, xs]
    return boxes, scores

def nms_normalized(boxes_xyxy: torch.Tensor, scores: torch.Tensor, iou_thresh: float = 0.5):
    if boxes_xyxy.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes_xyxy.device)
    keep = nms(boxes_xyxy, scores, iou_thresh)
    return keep
