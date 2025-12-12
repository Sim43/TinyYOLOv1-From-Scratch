import torch
import torch.nn as nn

class TinyYOLOv1(nn.Module):
    """
    Minimal YOLOv1-style network:
    - CNN backbone -> feature map
    - head -> S*S*(5 + C)
      where each cell predicts:
        [tx, ty, tw, th, obj] + class probs (C)
    Values are trained to represent:
      x,y in [0,1] within cell
      w,h in [0,1] relative to whole image
    """
    def __init__(self, S: int = 7, C: int = 1):
        super().__init__()
        self.S = S
        self.C = C
        out_ch = (5 + C)

        self.backbone = nn.Sequential(
            # (3,224,224)
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 112

            nn.Conv2d(16, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 56

            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 28

            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),  # 14

            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),

            # bring to ~SxS using adaptive pooling (simplifies for any img size)
            nn.AdaptiveAvgPool2d((S, S)),
        )

        self.head = nn.Sequential(
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(256, out_ch, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # output shape: (B, 5+C, S, S)
        x = self.backbone(x)
        x = self.head(x)
        # (B, S, S, 5+C)
        x = x.permute(0, 2, 3, 1).contiguous()
        return x
