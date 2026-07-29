"""Inverse dynamics ResNet-Lite model — efficiency-focused variant."""

from typing import override

import torch
import torch.nn as nn

from components.hud_diff import HudDiff
from components.inverse_dynamics_model import InverseDynamicsModel, ModelConfig
from components.signed_diff import SignedDiff

_RESIDUAL_DROPOUT = 0.1
_DROPOUT = 0.2
_BATCH_SIZE = 256
_VAL_BATCH_MULTIPLIER = 2
_ADAMW_WEIGHT_DECAY = 1e-2

# Values determined by empirical ablation studies.
_WINDOW_SIZE = 9
_LABEL_INDEX = 3
_STRIDE = 2


class ResidualBlock(nn.Module):
    """Standard pre-activation-style residual block with 3x3 convolutions."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.downsample: nn.Module = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(_RESIDUAL_DROPOUT)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += identity
        return self.relu(out)


class InverseDynamicsResNetLite(
    InverseDynamicsModel, nn.Module, model_name="resnet_lite"
):
    """Lightweight ResNet for inverse dynamics key prediction.

    Uses one residual block per stage, standard power-of-2 channel
    scaling (16→32→64→128→256), and a 3×3 stem convolution.
    """

    def __init__(
        self,
        window_size: int = 2,
        num_keys: int = 13,
        hud_dim: int = 0,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.signed_diff = SignedDiff()
        self.hud_diff = HudDiff()
        self.conv1 = nn.Conv2d(window_size, 16, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = ResidualBlock(16, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 128, stride=2)
        self.layer4 = ResidualBlock(128, 256, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(256 + hud_dim * 2, num_keys)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    @override
    def forward(self, x: torch.Tensor, hud: torch.Tensor) -> torch.Tensor:
        x = self.signed_diff(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        hud = self.hud_diff(hud)
        x = torch.cat([x, torch.flatten(hud, 1)], dim=1)
        return self.fc(x)

    @classmethod
    @override
    def config(cls) -> ModelConfig:
        return ModelConfig(
            batch_size=_BATCH_SIZE,
            val_batch_multiplier=_VAL_BATCH_MULTIPLIER,
            dropout=_DROPOUT,
            weight_decay=_ADAMW_WEIGHT_DECAY,
            window_size=_WINDOW_SIZE,
            label_index=_LABEL_INDEX,
            stride=_STRIDE,
            uses_channels_last=True,
        )
