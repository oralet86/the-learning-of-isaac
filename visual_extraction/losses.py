"""Masked multi-head loss, one criterion per `labels.HEAD_SPECS` entry.

Follows `components/loss_factory.py`'s pattern of keeping per-head
math isolated behind small pure functions, adapted for this problem's
per-head masking (rather than that file's sigmoid/softmax-group split).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812

from visual_extraction.labels import HEAD_SPECS, LabelBundle

if TYPE_CHECKING:
    from collections.abc import Callable

_DEFAULT_WEIGHT = 1.0


def _masked_mean(per_element_loss: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Averages `per_element_loss` over every masked-in element.

    `mask` may have fewer trailing dims than `per_element_loss` (e.g. one
    mask value per sample against a multi-dim regression target) -- it's
    broadcast out to match before reducing. The extra dims are computed
    from `.ndim`, a static (compile-time-known) property of each tensor's
    shape, not a runtime value -- so this reshape is a fixed, traceable
    op per head rather than a data-dependent Python loop.
    """
    extra_dims = per_element_loss.ndim - mask.ndim
    mask = mask.reshape(mask.shape + (1,) * extra_dims).expand_as(per_element_loss)
    total = (per_element_loss * mask).sum()
    count = mask.sum().clamp_min(1.0)
    return total / count


def _classification_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(pred, target.long(), reduction="none")


def _regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.smooth_l1_loss(pred.reshape(target.shape), target, reduction="none")


def _bounded_regression_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(
        torch.sigmoid(pred).reshape(target.shape), target, reduction="none"
    )


def _binary_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(
        pred.reshape(target.shape), target, reduction="none"
    )


# Dispatch table keyed by `HeadSpec.kind`, rather than an if/elif chain --
# `kind` is a compile-time-constant string per head (from the static
# `HEAD_SPECS` tuple `MultiHeadLoss.forward` iterates), so this lookup
# resolves to one fixed function per head with no runtime branching either
# way, but a dict keeps that fact structurally explicit rather than
# relying on Dynamo's constant-folding of an if/elif chain.
_LOSS_FNS: dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = {
    "classification": _classification_loss,
    "regression": _regression_loss,
    "bounded_regression": _bounded_regression_loss,
    "binary": _binary_loss,
}


def _head_loss(kind: str, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-element (unreduced) loss for one head, dispatched by `HeadSpec.kind`."""
    return _LOSS_FNS[kind](pred, target)


class MultiHeadLoss(nn.Module):
    """Sums one masked criterion per head into a scalar loss.

    Per-head weights are resolved once at construction time into a
    registered buffer (`head_weight_by_index`, ordered to match
    `HEAD_SPECS`), not looked up from a Python dict inside `forward` --
    keeps `forward` pure tensor ops with no dict access for
    `torch.compile` to reason about, matching
    `components/loss_factory.py`'s `register_buffer` precedent for
    per-head constants.
    """

    def __init__(self, head_weights: dict[str, float] | None = None) -> None:
        """Args:
        head_weights: Per-head loss weight, keyed by `HeadSpec.name`.
            Defaults to `1.0` for every head -- nothing in this
            project's requirements specified relative head importance,
            so this is a tunable default, not a hardcoded assumption.
        """
        super().__init__()
        weights = head_weights or {}
        ordered_weights = torch.tensor(
            [weights.get(spec.name, _DEFAULT_WEIGHT) for spec in HEAD_SPECS],
            dtype=torch.float32,
        )
        self.head_weight_by_index: torch.Tensor
        self.register_buffer("head_weight_by_index", ordered_weights, persistent=False)

    @override
    def forward(
        self, predictions: dict[str, torch.Tensor], targets: LabelBundle
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Args:
            predictions: `GroundTruthNet`'s output dict, keyed by head name.
            targets: A batched `LabelBundle` (see `dataset.collate_label_bundles`).

        Returns:
            `(total_loss, per_head_losses)` -- the scalar sum, and each
            head's individual masked loss for logging.
        """
        per_head: dict[str, torch.Tensor] = {}
        head_losses = []
        for spec in HEAD_SPECS:
            pred = predictions[spec.name]
            target = getattr(targets, spec.name)
            mask = getattr(targets, f"{spec.name}_mask")
            per_element = _head_loss(spec.kind, pred, target)
            head_loss = _masked_mean(per_element, mask)
            per_head[spec.name] = head_loss
            head_losses.append(head_loss)

        total = (torch.stack(head_losses) * self.head_weight_by_index).sum()
        return total, per_head
