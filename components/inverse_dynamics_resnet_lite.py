"""A small ResNet-like model for inverse dynamics."""

from typing import override

import torch
import torch.nn as nn

from components.helpers import (
    GroupedNeckHead,
    ResidualBlock,
    SignedDiff,
    init_resnet_weights,
)
from components.inverse_dynamics_model import InverseDynamicsModel, ModelConfig

_DROPOUT = 0.2
_BATCH_SIZE = 256
_VAL_BATCH_MULTIPLIER = 2
_ADAMW_WEIGHT_DECAY = 1e-2

# Values determined by empirical ablation studies.
_WINDOW_SIZE = 9
_LABEL_INDEX = 3
_STRIDE = 2


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
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.signed_diff = SignedDiff()
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
        self.head = GroupedNeckHead(256)

        init_resnet_weights(self)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        return self.head(x)

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
