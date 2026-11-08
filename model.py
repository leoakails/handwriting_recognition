"""小型卷積神經網路（CNN）：輸入 1×32×32 灰階圖，輸出 14 類分數。"""
import torch
from torch import nn


class SymbolCNN(nn.Module):
    """
    卷積區塊 1：32×32 → 擷取筆畫邊緣等局部特徵 → 最大池化成 16×16
    卷積區塊 2：16×16 → 組合成轉角、交叉、圓弧等較大形狀 → 池化成 8×8
    分類層    ：攤平 64×8×8 特徵 → 全連接 128 → 14 類分數（logits）
    """

    def __init__(self, num_classes: int = 14, dropout: float = 0.4):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32×32 → 16×16
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16×16 → 8×8
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
