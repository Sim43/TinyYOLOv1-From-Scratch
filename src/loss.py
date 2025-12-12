import torch
import torch.nn as nn
import torch.nn.functional as F

class YOLOv1Loss(nn.Module):
    """
    Simple YOLOv1-like loss for B=1 box per cell.
    target/pred shape: (B,S,S,5+C)
    """
    def __init__(self, lambda_coord=5.0, lambda_noobj=0.5):
        super().__init__()
        self.lambda_coord = lambda_coord
        self.lambda_noobj = lambda_noobj

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # object mask
        obj = target[..., 4]  # (B,S,S)
        noobj = 1.0 - obj

        # coordinate losses only where obj=1
        # pred uses raw logits -> we apply sigmoid in loss to keep stable
        px = torch.sigmoid(pred[..., 0])
        py = torch.sigmoid(pred[..., 1])
        pw = torch.sigmoid(pred[..., 2])
        ph = torch.sigmoid(pred[..., 3])

        tx = target[..., 0]
        ty = target[..., 1]
        tw = target[..., 2]
        th = target[..., 3]

        # YOLOv1 uses sqrt(w,h) for stability
        coord_loss = (
            F.mse_loss(px * obj, tx * obj, reduction="sum") +
            F.mse_loss(py * obj, ty * obj, reduction="sum") +
            F.mse_loss(torch.sqrt(pw.clamp_min(1e-6)) * obj, torch.sqrt(tw.clamp_min(1e-6)) * obj, reduction="sum") +
            F.mse_loss(torch.sqrt(ph.clamp_min(1e-6)) * obj, torch.sqrt(th.clamp_min(1e-6)) * obj, reduction="sum")
        )

        # objectness: BCE on logits
        obj_loss = F.binary_cross_entropy_with_logits(pred[..., 4], obj, reduction="none")
        obj_loss = (obj_loss * obj).sum()

        noobj_loss = F.binary_cross_entropy_with_logits(pred[..., 4], obj, reduction="none")
        noobj_loss = (noobj_loss * noobj).sum()

        # class loss (C=1): BCE
        if pred.size(-1) > 5:
            class_loss = F.binary_cross_entropy_with_logits(pred[..., 5:], target[..., 5:], reduction="none")
            class_loss = (class_loss.sum(dim=-1) * obj).sum()
        else:
            class_loss = pred.new_tensor(0.0)

        total = self.lambda_coord * coord_loss + obj_loss + self.lambda_noobj * noobj_loss + class_loss

        # normalize by batch size
        total = total / max(1, pred.size(0))
        return total
