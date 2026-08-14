"""Tests for `components.helpers.broadcast_mask` and its consumers.

`broadcast_mask` is the single shared implementation of the
mask-to-prediction broadcasting used by both the loss
(`components.ground_truth_losses._masked_mean`) and the accuracy
computation (`visual_extraction.train._masked_correct`); these tests pin
its semantics so the two call sites can't drift apart.
"""

from __future__ import annotations

import pytest
import torch

from components.ground_truth_losses import _masked_mean
from components.helpers import broadcast_mask


def test_broadcast_mask_same_ndim_returns_identical_shape() -> None:
    mask = torch.tensor([1.0, 0.0, 1.0])
    result = broadcast_mask(mask, 1)
    assert result.shape == (3,)
    assert torch.equal(result, mask)


def test_broadcast_mask_appends_trailing_singletons() -> None:
    mask = torch.tensor([1.0, 0.0])
    result = broadcast_mask(mask, 4)
    assert result.shape == (2, 1, 1, 1)
    assert torch.equal(result, mask.reshape(2, 1, 1, 1))


def test_broadcast_mask_expandable_to_target() -> None:
    mask = torch.tensor([[1.0], [0.0]])
    result = broadcast_mask(mask, 3).expand_as(torch.zeros(2, 3, 3))
    assert result.shape == (2, 3, 3)
    assert result[0, :, :].all()
    assert not result[1, :, :].any()


def test_broadcast_mask_rejects_finer_than_target() -> None:
    with pytest.raises(ValueError, match="cannot be broadcast"):
        broadcast_mask(torch.zeros(2, 3), 1)


def test_masked_mean_uses_shared_broadcast_semantics() -> None:
    # (2, 3) per-element losses gated by a (2,) per-sample mask: the
    # second sample is masked out entirely, so only sample 0 counts.
    per_element = torch.tensor([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])
    mask = torch.tensor([1.0, 0.0])
    assert _masked_mean(per_element, mask).item() == pytest.approx(2.0)


def test_masked_mean_multi_dim_target_with_per_sample_mask() -> None:
    # Spatial-head case: a (2, R, C) per-cell loss gated by a single mask
    # value per sample.
    per_element = torch.full((2, 3, 4), 2.0)
    per_element[1] = 8.0
    mask = torch.tensor([1.0, 0.0])
    assert _masked_mean(per_element, mask).item() == pytest.approx(2.0)


def test_broadcast_mask_matches_callers_expand_contract() -> None:
    # The exact shape dance `_masked_correct`/`_masked_mean` perform:
    # broadcast then expand_as must preserve mask values per element.
    mask = torch.tensor([1.0, 0.0, 1.0])
    correct = torch.randn(3, 5)
    expanded = broadcast_mask(mask, correct.ndim).expand_as(correct)
    assert expanded.shape == correct.shape
    assert (expanded == mask.reshape(3, 1).expand_as(correct)).all()
