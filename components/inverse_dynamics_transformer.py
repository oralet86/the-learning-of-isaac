"""Inverse dynamics transformer model — shared CNN backbone + transformer encoder."""

from typing import override

import torch
import torch.nn as nn

from components.inverse_dynamics_model import InverseDynamicsModel, ModelConfig
from components.inverse_dynamics_resnet import ResidualBlock
from components.signed_diff import SignedDiff

_BATCH_SIZE = 128
_VAL_BATCH_MULTIPLIER = 2
_DROPOUT = 0.1
_WEIGHT_DECAY = 1e-2
_FEATURE_DIM = 256
_NUM_HEADS = 8
_NUM_LAYERS = 3
_WINDOW_SIZE = 11
_STRIDE = 3


class InverseDynamicsTransformer(
    InverseDynamicsModel, nn.Module, model_name="transformer"
):
    """Shared CNN backbone per frame, positional embeddings, transformer encoder.

    Each frame is independently encoded into a feature vector by a
    weight-shared CNN.  Positional embeddings are added and the sequence
    is processed by a bidirectional transformer encoder.  The embedding
    of the centre frame is passed through a classifier head to predict
    key presses.
    """

    def __init__(
        self,
        window_size: int = _WINDOW_SIZE,
        num_keys: int = 13,
        feature_dim: int = _FEATURE_DIM,
        num_heads: int = _NUM_HEADS,
        num_layers: int = _NUM_LAYERS,
        dropout: float = _DROPOUT,
    ) -> None:
        super().__init__()
        self.window_size = window_size

        self.signed_diff = SignedDiff()
        self.conv1 = nn.Conv2d(1, 32, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

        self.layer1_block0 = ResidualBlock(32, 64, stride=2)
        self.layer1_block1 = ResidualBlock(64, 64)
        self.layer2_block0 = ResidualBlock(64, 128, stride=2)
        self.layer2_block1 = ResidualBlock(128, 128)
        self.layer3_block0 = ResidualBlock(128, feature_dim, stride=2)
        self.layer3_block1 = ResidualBlock(feature_dim, feature_dim)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.pos_embed = nn.Parameter(torch.empty(1, window_size, feature_dim))
        nn.init.normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )

        head_dim = feature_dim // 2
        self.head = nn.Sequential(
            nn.Linear(feature_dim, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(head_dim, num_keys),
        )

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
        x = self.signed_diff(x)
        b, t, _h, _w = x.shape

        x = x.reshape(b * t, 1, _h, _w)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1_block0(x)
        x = self.layer1_block1(x)
        x = self.layer2_block0(x)
        x = self.layer2_block1(x)
        x = self.layer3_block0(x)
        x = self.layer3_block1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        x = x.reshape(b, t, -1)
        x = x + self.pos_embed[:, :t, :]

        x = self.transformer(x)

        center_idx = (t - 1) // 2
        x = x[:, center_idx, :]

        return self.head(x)

    @classmethod
    @override
    def config(cls) -> ModelConfig:
        return ModelConfig(
            batch_size=_BATCH_SIZE,
            val_batch_multiplier=_VAL_BATCH_MULTIPLIER,
            dropout=_DROPOUT,
            weight_decay=_WEIGHT_DECAY,
            window_size=_WINDOW_SIZE,
            label_index=_WINDOW_SIZE // 2,
            stride=_STRIDE,
        )
