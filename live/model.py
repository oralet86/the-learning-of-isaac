"""RL agent — action determination for the live pipeline.

Receives preprocessed frames and returns binary action vectors
keyed by ``KEY_ORDER``.  The current placeholder returns random
presses; replace with a real PyTorch model later.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

from constants import KEY_ORDER


class RandomAgent:
    """Placeholder agent that outputs random key-press vectors.

    Returns a ``list[int]`` of length ``len(KEY_ORDER)`` where each
    entry is ``1`` (press) or ``0`` (release).  This feeds directly
    into ``ActionSender.apply_vector()``.

    Usage::

        agent = RandomAgent()
        keys = agent.predict(frame)  # list[int]
    """

    def __init__(self, press_prob: float = 0.10) -> None:
        self._n_keys = len(KEY_ORDER)
        self._press_prob = press_prob

    def predict(self, frame: np.ndarray) -> list[int]:  # noqa: ARG002
        """Return a random binary action vector.

        Args:
            frame: Preprocessed fp16 array ``(H, W, ...)``.
        """
        return [
            1 if random.random() < self._press_prob else 0 for _ in range(self._n_keys)
        ]
