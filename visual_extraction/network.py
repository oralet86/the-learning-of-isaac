"""Shared CNN backbone + one small output head per `labels.HEAD_SPECS` entry."""

from __future__ import annotations

from typing import override

import torch
import torch.nn as nn

from visual_extraction.labels import HEAD_SPECS

# channels_last plus torch.compile is the standard high-throughput combo
# for conv nets -- train.py converts both the model and input batches to
# this memory format; the network itself needs no code changes to support
# either layout (nn.Conv2d/BatchNorm2d handle both transparently).
USES_CHANNELS_LAST = True

_DROPOUT = 0.2
_BACKBONE_CHANNELS = (32, 64, 128, 256)
_FEATURE_DIM = _BACKBONE_CHANNELS[-1]


class ResidualBlock(nn.Module):
    """Pre-activation-style residual block with 3x3 convolutions."""

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

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)


class Backbone(nn.Module):
    """RGB image -> pooled feature vector, one residual block per stage."""

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
            stages.append(ResidualBlock(in_channels, out_channels, stride=stride))
            in_channels = out_channels
        self.stages = nn.Sequential(*stages)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class GroundTruthNet(nn.Module):
    """Backbone + one linear head per `labels.HEAD_SPECS` entry.

    `forward` returns raw logits/regression outputs keyed by head name,
    matching `labels.LabelBundle`'s field names exactly -- `losses.py`
    zips predictions against targets by that shared name, not position.

    Heads live in an `nn.ModuleList` (positional, matching `HEAD_SPECS`'s
    fixed order) rather than an `nn.ModuleDict` (string-keyed), and
    `forward` is a single static loop with no per-head name branching --
    every head's reshape (if any) comes from `HeadSpec.pred_shape`, a
    compile-time-constant attribute, not a runtime string comparison. Both
    choices are aimed at `torch.compile`: a static, uniform, branch-free
    loop over a fixed-length container is exactly what Dynamo traces most
    cleanly, with nothing here for it to guard on beyond tensor shapes.
    """

    def __init__(self, dropout: float = _DROPOUT) -> None:
        super().__init__()
        self.backbone = Backbone()
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleList(
            nn.Linear(_FEATURE_DIM, spec.out_dim) for spec in HEAD_SPECS
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
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.dropout(self.backbone(x))
        # `spec.pred_shape` is always a concrete tuple (see `labels._head`),
        # so every head takes the exact same unconditional reshape -- no
        # per-head branching in this loop.
        return {
            spec.name: head(features).view(-1, *spec.pred_shape)
            for spec, head in zip(HEAD_SPECS, self.heads, strict=True)
        }
