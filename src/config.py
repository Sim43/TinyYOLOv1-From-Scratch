from dataclasses import dataclass

@dataclass
class Config:
    # YOLOv1-style setup
    S: int = 7          # grid size SxS
    C: int = 1          # number of classes (pedestrian only)
    img_size: int = 224 # keep small for CPU learning

    # Training
    epochs: int = 30
    batch_size: int = 16
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Loss weights (YOLO-ish)
    lambda_coord: float = 5.0
    lambda_noobj: float = 0.5

    # Paths
    data_root: str = "data/PennFudanPed"
    out_dir: str = "runs"
    checkpoint_name: str = "yolov1_tiny_pennfudan.pt"
