"""Shared CNN backbone + one small output head per `labels.HEAD_SPECS` entry."""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

from components.helpers import GroundTruthHead, ResidualBlock, init_resnet_weights
from visual_extraction.labels import (
    DETECTION_GRID_COLS,
    DETECTION_GRID_ROWS,
    HEAD_SPECS,
)

# channels_last plus torch.compile is the standard high-throughput combo
# for conv nets -- train.py converts both the model and input batches to
# this memory format; the network itself needs no code changes to support
# either layout (nn.Conv2d/BatchNorm2d handle both transparently).
USES_CHANNELS_LAST = True

_DROPOUT = 0.2
_BACKBONE_CHANNELS = (32, 64, 128, 256)
_FEATURE_DIM = _BACKBONE_CHANNELS[-1]
_DETECTION_GRID = (DETECTION_GRID_ROWS, DETECTION_GRID_COLS)


class Backbone(nn.Module):
    """RGB image -> spatial feature map, one residual block per stage.

    Uses `components/helpers.py`'s `ResidualBlock` with its internal
    dropout disabled (`dropout=0.0`) -- this backbone already gets a
    separate, explicit dropout on the pooled features in `GroundTruthNet`,
    matching this file's pre-move behavior; the IDM models sharing
    `ResidualBlock` keep their own default internal dropout unaffected.

    Pooling is deliberately left to the caller: `GroundTruthNet` needs this
    same feature map consumed two different ways (globally pooled for
    per-sample heads, grid-pooled for per-cell detection heads), so this
    class stops at the last residual stage's raw `(N, C, H, W)` output
    rather than baking in one specific pooling choice.
    """

    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        in_channels = 16
        stages = []
        for i, out_channels in enumerate(_BACKBONE_CHANNELS):
            stride = 1 if i == 0 else 2
            stages.append(
                ResidualBlock(in_channels, out_channels, stride=stride, dropout=0.0)
            )
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        return self.stages(x)


class GroundTruthNet(nn.Module):
    """Backbone + `GroundTruthHead`, one small head per `labels.HEAD_SPECS` entry.

    `forward` returns raw logits/regression outputs keyed by head name,
    matching `labels.LabelBundle`'s field names exactly -- `losses.py`
    zips predictions against targets by that shared name, not position.

    Pools `backbone`'s feature map two different ways before handing both
    off to `GroundTruthHead`: globally to 1x1 for scalar (per-sample)
    heads, and to `DETECTION_GRID_ROWS x DETECTION_GRID_COLS` for spatial
    (per-cell) heads -- collapsing to a single vector first would throw
    away exactly the per-cell information the spatial heads need. See
    `GroundTruthHead`'s docstring (`components/helpers.py`) for how the two
    groups are split and processed.
    """

    def __init__(self, dropout: float = _DROPOUT) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.grid_pool = nn.AdaptiveAvgPool2d(_DETECTION_GRID)
        self.dropout = nn.Dropout(dropout)
        self.head = GroundTruthHead(_FEATURE_DIM, HEAD_SPECS, _DETECTION_GRID)
        init_resnet_weights(self)

    @override
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(x)
        pooled = self.dropout(torch.flatten(self.global_pool(features), 1))
        grid = self.grid_pool(features)
        return self.head(pooled, grid)
