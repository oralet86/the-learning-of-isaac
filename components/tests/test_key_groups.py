"""Tests for the key grouping `components.helpers` derives from `KEY_ORDER`.

`GroupedKeyLoss` and the model heads index label vectors through these
groups, and trained checkpoints depend on their layout, so the derived
indices are pinned against the current key order.
"""

from __future__ import annotations

import torch

from components import helpers
from components.helpers import GroupedKeyLoss
from constants import KEY_ORDER


def test_groups_cover_every_key_exactly_once() -> None:
    sigmoid = list(range(helpers._SIGMOID_START, helpers._SIGMOID_END))
    softmax = [index for group in helpers.SOFTMAX_GROUPS for index in group]

    assert sorted(sigmoid + softmax) == list(range(len(KEY_ORDER)))
    assert len(KEY_ORDER) == helpers.NUM_KEYS


def test_groups_point_at_the_named_keys() -> None:
    sigmoid_names = KEY_ORDER[helpers._SIGMOID_START : helpers._SIGMOID_END]
    pair_names = [
        (KEY_ORDER[first], KEY_ORDER[second])
        for first, second in helpers.SOFTMAX_GROUPS
    ]

    assert sigmoid_names == ["e", "space", "q", "tab", "ctrl"]
    assert pair_names == [("a", "d"), ("w", "s"), ("left", "right"), ("up", "down")]


def test_layout_matches_the_one_existing_checkpoints_were_trained_with() -> None:
    assert (helpers._SIGMOID_START, helpers._SIGMOID_END) == (8, 13)
    assert helpers.SOFTMAX_GROUPS == ((1, 3), (0, 2), (6, 7), (4, 5))


def test_to_binary_probs_places_each_group_on_its_keys() -> None:
    loss = GroupedKeyLoss(torch.ones(helpers.NUM_SIGMOID))
    sigmoid_logits = torch.full((1, helpers.NUM_SIGMOID), 20.0)
    softmax_logits = torch.zeros(1, helpers.NUM_SOFTMAX, 3)
    # First pair (a, d): certain "a"; second pair (w, s): certain "s".
    softmax_logits[0, 0, 0] = 20.0
    softmax_logits[0, 1, 2] = 20.0

    probs = loss.to_binary_probs((sigmoid_logits, softmax_logits))[0]

    assert probs.shape == (len(KEY_ORDER),)
    for key in ("e", "space", "q", "tab", "ctrl", "a", "s"):
        assert probs[KEY_ORDER.index(key)] > 0.99, key
    for key in ("d", "w"):
        assert probs[KEY_ORDER.index(key)] < 0.01, key
