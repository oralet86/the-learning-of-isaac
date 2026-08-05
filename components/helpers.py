"""Smaller modules, meant to be used to make other models."""

from typing import TYPE_CHECKING, Protocol, override

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Sequence

_RESIDUAL_DROPOUT = 0.1
_HEAD_HIDDEN_DIM = 32

# -------- GroupedKeyLoss constants (indices into the 13-dim label vector) --------
# Based on KEY_ORDER = ["w","a","s","d","up","down","left","right",
#                        "e","space","q","tab","ctrl"]

_SIGMOID_START: int = 8
_SIGMOID_END: int = 13

SOFTMAX_GROUPS: tuple[tuple[int, int], ...] = (
    (1, 3),  # wasd_x:  a (idx 1), d (idx 3)
    (0, 2),  # wasd_y:  w (idx 0), s (idx 2)
    (6, 7),  # arrow_x: left (idx 6), right (idx 7)
    (4, 5),  # arrow_y: up (idx 4), down (idx 5)
)

NUM_SIGMOID: int = _SIGMOID_END - _SIGMOID_START  # 5
NUM_SOFTMAX: int = len(SOFTMAX_GROUPS)  # 4

_K1_IDX_LIST: list[int] = [g[0] for g in SOFTMAX_GROUPS]
_K2_IDX_LIST: list[int] = [g[1] for g in SOFTMAX_GROUPS]
_LABEL_IDX_LIST: list[int] = [idx for g in SOFTMAX_GROUPS for idx in g]


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

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = _RESIDUAL_DROPOUT,
    ) -> None:
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
        self.dropout = nn.Dropout2d(dropout)

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

    One branch per `GroupedKeyLoss` group: 1 sigmoid group of `NUM_SIGMOID`
    keys, and `NUM_SOFTMAX` 3-way softmax triples. Each branch is
    `Linear(in_features, H) -> ReLU -> Linear(H, out_dim)` so every group
    gets its own nonlinear transform of the shared pooled feature rather than
    reading off one shared linear row.

    Returns a `(sigmoid_logits, softmax_logits)` pair rather than a
    concatenated tensor, so the head/loss split doesn't depend on hardcoded
    slice offsets. The softmax branches' stacking order still must match
    `SOFTMAX_GROUPS` order (`wasd_x, wasd_y, arrow_x, arrow_y`) -- that part
    of the contract can't be verified at runtime.
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
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sigmoid_logits = self.sigmoid_branch(x)
        softmax_logits = torch.stack(
            [branch(x) for branch in self.softmax_branches], dim=1
        )
        return sigmoid_logits, softmax_logits


class _HeadSpecLike(Protocol):
    """Structural shape `GroundTruthHead` needs from a per-head spec.

    Deliberately a `Protocol`, not an import of
    `visual_extraction.labels.HeadSpec` -- this file has no other
    dependency on `visual_extraction` (the IDM models importing
    `ResidualBlock`/`GroupedNeckHead` from here have nothing to do with
    that package), and `HeadSpec` already has every field below, so it
    satisfies this structurally with no coupling required in either
    direction.
    """

    @property
    def name(self) -> str: ...
    @property
    def out_dim(self) -> int: ...
    @property
    def kind(self) -> str: ...
    @property
    def pred_shape(self) -> tuple[int, ...]: ...
    @property
    def spatial(self) -> bool: ...


def _scalar_neck(in_features: int, out_dim: int) -> nn.Module:
    """`Linear(in, H) -> ReLU -> Linear(H, out_dim)` off a globally-pooled feature."""
    return nn.Sequential(
        nn.Linear(in_features, _HEAD_HIDDEN_DIM),
        nn.ReLU(inplace=True),
        nn.Linear(_HEAD_HIDDEN_DIM, out_dim),
    )


def _spatial_neck(in_features: int, out_channels: int) -> nn.Module:
    """`Conv1x1(in, H) -> ReLU -> Conv1x1(H, out_channels)` over a grid-pooled feature."""
    return nn.Sequential(
        nn.Conv2d(in_features, _HEAD_HIDDEN_DIM, 1),
        nn.ReLU(inplace=True),
        nn.Conv2d(_HEAD_HIDDEN_DIM, out_channels, 1),
    )


class GroundTruthHead(nn.Module):
    """One small head per spec in `specs`, split into scalar and spatial groups.

    Generalizes `GroupedNeckHead`'s "every output gets its own nonlinear
    neck off a shared pooled feature" pattern to an arbitrary, data-driven
    list of heads (rather than a fixed sigmoid/softmax split): every spec
    in `specs` gets a `Linear(in, H) -> ReLU -> Linear(H, out_dim)` or
    `Conv1x1(in, H) -> ReLU -> Conv1x1(H, out_channels)` neck, and
    `forward` returns a `dict[str, Tensor]` keyed by `spec.name`.

    `specs` splits into two static, fixed-order groups at construction
    time via `spec.spatial` (a compile-time-constant attribute, never a
    runtime name comparison):

    - Scalar heads (`spatial=False`) run off `pooled` -- a single vector
      per sample, through `_scalar_neck`.
    - Spatial heads (`spatial=True`) run off `grid` -- a
      `detection_grid`-shaped feature map, through `_spatial_neck`, so a
      per-cell prediction isn't forced through a single flattened vector
      first (which would throw away exactly the spatial information those
      heads need).

    Each group is still a single static, uniform, branch-free loop over a
    fixed-length container -- exactly what `torch.compile`'s Dynamo traces
    most cleanly -- just two such loops instead of one, since scalar and
    spatial heads genuinely need different neck architectures (linear vs.
    conv) and different pooled inputs.
    """

    def __init__(
        self,
        in_features: int,
        specs: Sequence[_HeadSpecLike],
        detection_grid: tuple[int, int],
    ) -> None:
        super().__init__()
        self._scalar_specs = tuple(spec for spec in specs if not spec.spatial)
        self._spatial_specs = tuple(spec for spec in specs if spec.spatial)
        self.scalar_heads = nn.ModuleList(
            _scalar_neck(in_features, spec.out_dim) for spec in self._scalar_specs
        )
        # A spatial head's `pred_shape` is either channel-first
        # (classification, matching F.cross_entropy's K-dimensional form)
        # or channel-last (everything else, matching a plain elementwise
        # target) -- see `forward`'s permute below. Either way, per-head
        # channel count is derived from `out_dim // grid_cells` rather
        # than hardcoded, so this stays generic across however many
        # spatial heads `specs` contains.
        grid_cells = detection_grid[0] * detection_grid[1]
        self.spatial_heads = nn.ModuleList(
            _spatial_neck(in_features, spec.out_dim // grid_cells)
            for spec in self._spatial_specs
        )

    @override
    def forward(
        self, pooled: torch.Tensor, grid: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        outputs: dict[str, torch.Tensor] = {
            spec.name: head(pooled).view(-1, *spec.pred_shape)
            for spec, head in zip(self._scalar_specs, self.scalar_heads, strict=True)
        }
        for spec, head in zip(self._spatial_specs, self.spatial_heads, strict=True):
            out = head(grid)
            if spec.kind != "classification":
                out = out.permute(0, 2, 3, 1)
            outputs[spec.name] = out.reshape(-1, *spec.pred_shape)
        return outputs


class GroupedKeyLoss(nn.Module):
    """Loss combining 5 sigmoid keys (BCE) + 4 softmax directional triples (CE).

    Usage::

        loss = GroupedKeyLoss(sigmoid_pos_weight)
        loss_fn = torch.compile(loss, mode="max-autotune")
        ...
        loss_value = loss_fn(logits, targets)
        probs = loss.to_binary_probs(logits)
    """

    sigmoid_pos_weight: torch.Tensor
    _k1_indices: torch.Tensor
    _k2_indices: torch.Tensor

    num_output_logits: int = NUM_SIGMOID + NUM_SOFTMAX * 3  # 17

    def __init__(self, sigmoid_pos_weight: torch.Tensor) -> None:
        """Args:
        sigmoid_pos_weight: ``(5,)`` pos_weight for BCE (already on device).
        """
        super().__init__()
        device = sigmoid_pos_weight.device
        self.sigmoid_pos_weight: torch.Tensor
        self.register_buffer("sigmoid_pos_weight", sigmoid_pos_weight)
        self.register_buffer(
            "_k1_indices",
            torch.tensor(_K1_IDX_LIST, dtype=torch.long, device=device),
            persistent=False,
        )
        self.register_buffer(
            "_k2_indices",
            torch.tensor(_K2_IDX_LIST, dtype=torch.long, device=device),
            persistent=False,
        )

    @classmethod
    def from_full_pos_weight(
        cls, pos_weight: torch.Tensor, device: torch.device
    ) -> "GroupedKeyLoss":
        """Build from the raw ``(13,)`` pos_weight vector loaded from disk."""
        return cls(
            sigmoid_pos_weight=pos_weight[_SIGMOID_START:_SIGMOID_END].to(device)
        )

    def forward(
        self,
        logits: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the compound loss — one BCE + one CE call.

        Args:
            logits: ``(sigmoid_logits: (B, 5), softmax_logits: (B, 4, 3))``.
            targets: ``(B, 13)`` binary labels (float).

        Returns:
            Scalar loss.
        """
        sigmoid_logits, softmax_logits = logits
        bs = sigmoid_logits.shape[0]
        device = sigmoid_logits.device

        loss = F.binary_cross_entropy_with_logits(
            sigmoid_logits,
            targets[:, _SIGMOID_START:_SIGMOID_END],
            pos_weight=self.sigmoid_pos_weight,
        )

        group_targets = targets[:, _LABEL_IDX_LIST].view(bs, NUM_SOFTMAX, 2)

        k1_active = group_targets[:, :, 0].bool()
        k2_active = group_targets[:, :, 1].bool()

        zero = torch.zeros(bs, NUM_SOFTMAX, dtype=torch.long, device=device)
        one = torch.ones(bs, NUM_SOFTMAX, dtype=torch.long, device=device)
        two = torch.full((bs, NUM_SOFTMAX), 2, dtype=torch.long, device=device)

        class_idx = one.clone()
        class_idx = torch.where(k1_active, zero, class_idx)
        class_idx = torch.where(k2_active, two, class_idx)

        loss = loss + F.cross_entropy(
            softmax_logits.reshape(-1, 3), class_idx.reshape(-1)
        )

        return loss

    @torch.no_grad()
    def to_binary_probs(
        self, logits: tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        """Reconstruct ``(B, 13)`` binary probabilities from ``(sigmoid_logits, softmax_logits)``."""
        sigmoid_logits, softmax_logits = logits
        bs = sigmoid_logits.shape[0]
        device = sigmoid_logits.device
        dtype = sigmoid_logits.dtype

        probs = torch.zeros(bs, 13, device=device, dtype=dtype)
        probs[:, _SIGMOID_START:_SIGMOID_END] = torch.sigmoid(sigmoid_logits)

        group_probs = F.softmax(softmax_logits, dim=-1).to(dtype=dtype)

        probs[:, self._k1_indices] = group_probs[:, :, 0]
        probs[:, self._k2_indices] = group_probs[:, :, 2]

        return probs
