from __future__ import annotations

from typing import Dict

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.segmentation.deeplabv3 import ASPP

from .backbones.resnet_se_dw import SEResNet34, seresnet34_backbone


class DeepLabSEResNet34(nn.Module):
    """DeepLabV3+ style head coupled with the SEResNet34 backbone."""

    def __init__(self, num_classes: int = 2, output_stride: int = 16, dropout: float = 0.1) -> None:
        super().__init__()
        self.backbone: SEResNet34 = seresnet34_backbone(output_stride=output_stride)
        atrous_rates = [6, 12, 18]
        if output_stride == 8:
            atrous_rates = [rate * 2 for rate in atrous_rates]
        elif output_stride == 32:
            atrous_rates = [rate // 2 for rate in atrous_rates]
        self.aspp = ASPP(512, tuple(atrous_rates), out_channels=256)
        self.aspp_proj = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.low_level_proj = nn.Sequential(
            nn.Conv2d(64, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(256 + 48, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout, inplace=False),
            nn.Conv2d(256, num_classes, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        input_size = x.shape[-2:]
        features = self.backbone(x)
        low_level = features["low_level"]
        encoder_out = features["out"]

        x = self.aspp(encoder_out)
        x = self.aspp_proj(x)
        x = F.interpolate(x, size=low_level.shape[-2:], mode="bilinear", align_corners=False)

        low = self.low_level_proj(low_level)
        x = torch.cat([x, low], dim=1)
        x = self.decoder(x)
        x = F.interpolate(x, size=input_size, mode="bilinear", align_corners=False)

        return {"out": x}


def build_segmentation_model(num_classes: int, output_stride: int = 16, dropout: float = 0.1) -> DeepLabSEResNet34:
    return DeepLabSEResNet34(num_classes=num_classes, output_stride=output_stride, dropout=dropout)

