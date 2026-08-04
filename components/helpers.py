"""Smaller modules, meant to be used to make other models."""

from typing import override

import torch
import torch.nn as nn

from components.loss_factory import NUM_SIGMOID, NUM_SOFTMAX

_RESIDUAL_DROPOUT = 0.1
_HEAD_HIDDEN_DIM = 32


def init_resnet_weights(module: nn.Module) -> None:
    """Kaiming-init convs, constant-init BatchNorm affine params, small-normal-init
    Linear layers -- shared by every ResNet-style inverse-dynamics model so the
    init scheme can't drift out of sync between them.
    """
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0, 0.01)
            nn.init.constant_(m.bias, 0)


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


class SignedDiff(nn.Module):
    """Apply signed frame differences as a tensor operation.

    Input:  ``(B, T, H, W)`` float tensor in ``[0, 1]``.
    Output: ``(B, T, H, W)`` float tensor in ``[0, 1]``.

    Values > 0.5 mean the pixel got brighter; < 0.5 means darker.
    Equivalent to the uint8 centre-at-128 convention after normalisation.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.shape[1]
        if t > 1:
            frame0 = x[:, :1]
            diffs = x[:, 1:] - x[:, :-1]
            x = torch.cat([frame0, (0.5 + diffs).clamp(0, 1)], dim=1)
        return x


class GroupedNeckHead(nn.Module):
    """Five independent neck+head branches off the shared pooled feature.

    One branch per `loss_factory.GroupedKeyLoss` group: 1 sigmoid group of
    `NUM_SIGMOID` keys, and `NUM_SOFTMAX` 3-way softmax triples. Each branch
    is `Linear(in_features, H) -> ReLU -> Linear(H, out_dim)` so every group
    gets its own nonlinear transform of the shared pooled feature rather than
    reading off one shared linear row.

    The concatenated `(B, 17)` output's column order matches
    `loss_factory.SOFTMAX_GROUPS` exactly: sigmoid keys first, then the 4
    groups in `wasd_x, wasd_y, arrow_x, arrow_y` order. This is a positional
    contract with `GroupedKeyLoss` that this class cannot verify at runtime
    -- it must stay in sync if `SOFTMAX_GROUPS`'s order ever changes.
    """

    def __init__(self, in_features: int) -> None:
        super().__init__()
        self.sigmoid_branch = nn.Sequential(
            nn.Linear(in_features, _HEAD_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Linear(_HEAD_HIDDEN_DIM, NUM_SIGMOID),
        )
        self.softmax_branches = nn.ModuleList(
            nn.Sequential(
                nn.Linear(in_features, _HEAD_HIDDEN_DIM),
                nn.ReLU(inplace=True),
                nn.Linear(_HEAD_HIDDEN_DIM, 3),
            )
            for _ in range(NUM_SOFTMAX)
        )

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = [self.sigmoid_branch(x)]
        parts.extend(branch(x) for branch in self.softmax_branches)
        return torch.cat(parts, dim=1)
