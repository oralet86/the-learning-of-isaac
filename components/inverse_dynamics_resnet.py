"""Inverse dynamics ResNet model."""

from typing import override

import torch
import torch.nn as nn

from components.inverse_dynamics_model import InverseDynamicsModel, ModelConfig

_RESIDUAL_DROPOUT = 0.2
_DROPOUT = 0.4
_BATCH_SIZE = 256
_VAL_BATCH_MULTIPLIER = 2
_ADAMW_WEIGHT_DECAY = 1e-2


class ResidualBlock(nn.Module):
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


class InverseDynamicsResNet(InverseDynamicsModel, nn.Module, model_name="resnet"):
    def __init__(
        self,
        window_size: int = 2,
        num_keys: int = 13,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(window_size, 16, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1_block0 = ResidualBlock(16, 32, stride=1)
        self.layer1_block1 = ResidualBlock(32, 32, stride=1)
        self.layer2_block0 = ResidualBlock(32, 96, stride=2)
        self.layer2_block1 = ResidualBlock(96, 96, stride=1)
        self.layer3_block0 = ResidualBlock(96, 200, stride=2)
        self.layer3_block1 = ResidualBlock(200, 200, stride=1)
        self.layer4_block0 = ResidualBlock(200, 320, stride=2)
        self.layer4_block1 = ResidualBlock(320, 320, stride=1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(320, num_keys)

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1_block0(x)
        x = self.layer1_block1(x)
        x = self.layer2_block0(x)
        x = self.layer2_block1(x)
        x = self.layer3_block0(x)
        x = self.layer3_block1(x)
        x = self.layer4_block0(x)
        x = self.layer4_block1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)

    @classmethod
    @override
    def config(cls) -> ModelConfig:
        return ModelConfig(
            batch_size=_BATCH_SIZE,
            val_batch_multiplier=_VAL_BATCH_MULTIPLIER,
            dropout=_DROPOUT,
            weight_decay=_ADAMW_WEIGHT_DECAY,
            window_size=2,
            label_index=0,
            stride=3,
            uses_channels_last=True,
        )
