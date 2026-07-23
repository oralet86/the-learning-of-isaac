"""Loss criterion factory for inverse dynamics training.

Supports multiple loss formulations through a unified ``(logits, targets) -> scalar``
interface. Each criterion carries its own ``num_output_logits`` attribute so the
model can size its final layer accordingly, and a ``to_binary_probs`` method so
eval/plotting code never needs to know the loss internals.

All forward implementations use only ``torch`` ops — no graph breaks under
``torch.compile(mode="max-autotune")``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812


class LossCriterion(nn.Module):
    """Abstract base for loss criteria returned by ``make_criterion``."""

    num_output_logits: int

    def to_binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


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

# ---------------------------------------------------------------------------


class BCELossWrapper(LossCriterion):
    """Standard 13-key BCE loss, wrapped to satisfy the criterion protocol."""

    num_output_logits: int = 13

    def __init__(self, pos_weight: torch.Tensor) -> None:
        """Args:
        pos_weight: ``(13,)`` pos_weight tensor (already on the correct device).
        """
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """logits: ``(B, 13)``, targets: ``(B, 13)`` binary float."""
        return self.bce(logits, targets)

    @torch.no_grad()
    def to_binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """``(B, 13)`` logits -> ``(B, 13)`` probabilities via sigmoid."""
        return torch.sigmoid(logits)


class GroupedKeyLoss(LossCriterion):
    """Loss combining 5 sigmoid keys (BCE) + 4 softmax directional triples (CE).

    Usage::

        loss = GroupedKeyLoss(sigmoid_pos_weight)
        loss_fn = torch.compile(loss, mode="max-autotune")
        ...
        loss_value = loss_fn(logits, targets)
        probs = loss.to_binary_probs(logits)
    """

    sigmoid_pos_weight: torch.Tensor
    _label_indices: torch.Tensor
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
            "_label_indices",
            torch.tensor(_LABEL_IDX_LIST, dtype=torch.long, device=device),
            persistent=False,
        )
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

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the compound loss — one BCE + one CE call.

        Args:
            logits: ``(B, 17)`` raw model output.
            targets: ``(B, 13)`` binary labels (float).

        Returns:
            Scalar loss.
        """
        bs = logits.shape[0]
        device = logits.device

        loss = F.binary_cross_entropy_with_logits(
            logits[:, :NUM_SIGMOID],
            targets[:, _SIGMOID_START:_SIGMOID_END],
            pos_weight=self.sigmoid_pos_weight,
        )

        group_logits = logits[:, NUM_SIGMOID:].view(bs, NUM_SOFTMAX, 3)
        group_targets = targets[:, self._label_indices].view(bs, NUM_SOFTMAX, 2)

        k1_active = group_targets[:, :, 0].bool()
        k2_active = group_targets[:, :, 1].bool()

        zero = torch.zeros(bs, NUM_SOFTMAX, dtype=torch.long, device=device)
        one = torch.ones(bs, NUM_SOFTMAX, dtype=torch.long, device=device)
        two = torch.full((bs, NUM_SOFTMAX), 2, dtype=torch.long, device=device)

        class_idx = one.clone()
        class_idx = torch.where(k1_active, zero, class_idx)
        class_idx = torch.where(k2_active, two, class_idx)

        loss = loss + F.cross_entropy(
            group_logits.reshape(-1, 3), class_idx.reshape(-1)
        )

        return loss

    @torch.no_grad()
    def to_binary_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """Reconstruct ``(B, 13)`` binary probabilities from ``(B, 17)`` logits."""
        bs = logits.shape[0]
        device = logits.device
        dtype = logits.dtype

        probs = torch.zeros(bs, 13, device=device, dtype=dtype)
        probs[:, _SIGMOID_START:_SIGMOID_END] = torch.sigmoid(logits[:, :NUM_SIGMOID])

        group_logits = logits[:, NUM_SIGMOID:].view(bs, NUM_SOFTMAX, 3)
        group_probs = F.softmax(group_logits, dim=-1)

        probs[:, self._k1_indices] = group_probs[:, :, 0]
        probs[:, self._k2_indices] = group_probs[:, :, 2]

        return probs


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_criterion(
    name: str,
    pos_weight: torch.Tensor,
    device: torch.device,
) -> LossCriterion:
    """Build a loss criterion by name.

    Args:
        name: ``"bce"`` or ``"grouped_ce"``.
        pos_weight: ``(13,)`` pos_weight for the full 13-key vector (raw from disk).
        device: Target ``torch.device``.

    Returns:
        An ``nn.Module`` with ``forward(logits, targets) -> scalar`` and
        ``to_binary_probs(logits) -> (B, 13)``.
    """
    if name == "bce":
        return BCELossWrapper(pos_weight=pos_weight.to(device))
    if name == "grouped_ce":
        sigmoid_pw = pos_weight[_SIGMOID_START:_SIGMOID_END].to(device)
        return GroupedKeyLoss(sigmoid_pos_weight=sigmoid_pw)

    msg = f"Unknown loss {name!r}. Choose from: bce, grouped_ce"
    raise ValueError(msg)
