"""Focal loss for multi-label classification."""

import torch
from torch import Tensor
import torch.nn.functional as F

FOCAL_GAMMA = 2.0
POS_WEIGHT_CLAMP = (0.1, 50.0)


def focal_loss_with_logits(
    logits: Tensor,
    targets: Tensor,
    pos_weight: Tensor,
) -> Tensor:
    pos_weight_clamped: Tensor = torch.clamp(
        pos_weight, POS_WEIGHT_CLAMP[0], POS_WEIGHT_CLAMP[1]
    )
    alpha: Tensor = pos_weight_clamped / (pos_weight_clamped + 1.0)
    p: Tensor = torch.sigmoid(logits)
    p_t: Tensor = targets * p + (1.0 - targets) * (1.0 - p)
    modulating: Tensor = (1.0 - p_t).pow(exponent=FOCAL_GAMMA)
    bce: Tensor = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    alpha_t: Tensor = targets * alpha + (1.0 - targets) * (1.0 - alpha)
    loss: Tensor = alpha_t * modulating * bce
    return loss.mean()
