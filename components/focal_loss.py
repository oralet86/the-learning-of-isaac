"""Focal loss for multi-label classification."""

import torch
import torch.nn.functional as F

DEFAULT_FOCAL_GAMMA = 2.0
POS_WEIGHT_CLAMP = (0.1, 50.0)


def focal_loss_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_weight: torch.Tensor,
    gamma: float = DEFAULT_FOCAL_GAMMA,
    reduction: str = "mean",
) -> torch.Tensor:
    pos_weight = torch.clamp(pos_weight, POS_WEIGHT_CLAMP[0], POS_WEIGHT_CLAMP[1])
    alpha = pos_weight / (pos_weight + 1.0)
    p = torch.sigmoid(logits)
    p_t = targets * p + (1.0 - targets) * (1.0 - p)
    modulating = (1.0 - p_t).pow(gamma)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    alpha_t = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss = alpha_t * modulating * bce
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss
