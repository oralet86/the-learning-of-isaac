"""VPT-inspired ResNet variant: narrower backbone, coarse pooled head."""

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

_DROPOUT = 0.0
_BATCH_SIZE = 256
_VAL_BATCH_MULTIPLIER = 2
_ADAMW_WEIGHT_DECAY = 1e-2

_WINDOW_SIZE = 9
_LABEL_INDEX = 3
_STRIDE = 2

_POOL_SIZE = 4
_LAYER4_CHANNELS = 96
_POOLED_DIM = _POOL_SIZE * _POOL_SIZE * _LAYER4_CHANNELS  # 1536


class InverseDynamicsResNetVPT(
    InverseDynamicsModel, nn.Module, model_name="resnet_vpt"
):
    """Narrower ResNet with a 4x4-pooled, dropout-free head.

    Four residual stages (16->32, 32->64, 64->64, 64->96), matching
    `InverseDynamicsResNetLite`'s 16-channel stem width but narrower at every
    stage. `AdaptiveAvgPool2d` is widened from 1x1 to 4x4 -- unlike a raw
    flatten, this keeps the head's `in_features` fixed at `_POOLED_DIM`
    regardless of input resolution, so (unlike the earlier full-flatten
    version of this model) no shape inference is needed at construction time.
    No dropout is applied after pooling (see `_DROPOUT`'s comment);
    `ResidualBlock`'s internal dropout still regularizes the backbone.
    """

    def __init__(
        self,
        window_size: int = 2,
    ) -> None:
        super().__init__()
        self.signed_diff = SignedDiff()
        self.conv1 = nn.Conv2d(window_size, 16, 3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.layer1 = ResidualBlock(16, 32, stride=1)
        self.layer2 = ResidualBlock(32, 64, stride=2)
        self.layer3 = ResidualBlock(64, 64, stride=2)
        self.layer4 = ResidualBlock(64, _LAYER4_CHANNELS, stride=2)

        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AdaptiveAvgPool2d((_POOL_SIZE, _POOL_SIZE))
        self.head = GroupedNeckHead(_POOLED_DIM)

        init_resnet_weights(self)

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Stem + residual stages, shared conceptually with
        `InverseDynamicsResNetLite`'s forward but factored out here for
        readability.
        """
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.signed_diff(x)
        x = self._backbone(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
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
