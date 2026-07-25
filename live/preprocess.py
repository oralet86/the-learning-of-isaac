"""Frame preprocessing for the RL pipeline.

Transforms raw fp16 frames from StreamGame into model-ready inputs.
Each stage is a separate callable so new transforms can be inserted
without touching the loop or the agent.
"""

from __future__ import annotations

import cv2
import numpy as np

from constants import TARGET_HEIGHT, TARGET_WIDTH


class FramePreprocessor:
    """Downscale and grayscale frames for the RL model.

    Resizes to ``TARGET_WIDTH × TARGET_HEIGHT`` using area interpolation
    (anti-aliased for downscaling), then converts to single-channel
    grayscale.  Input is raw uint8 stored in fp16; output is normalised
    fp16 in ``[0, 1]``.

    Usage::

        pre = FramePreprocessor()
        obs = pre(frame)  # (H, W, 3) → (TARGET_HEIGHT, TARGET_WIDTH)
    """

    def __init__(self) -> None:
        self._target = (TARGET_WIDTH, TARGET_HEIGHT)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        """Downscale to target size and convert to grayscale.

        Args:
            frame: ``(H, W, 3)`` fp16 array, raw uint8 values in ``[0, 255]``.

        Returns:
            ``(TARGET_HEIGHT, TARGET_WIDTH)`` fp16 array in ``[0, 1]``.
        """
        img = frame.astype(np.uint8)
        small = cv2.resize(img, self._target, interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
        return gray.astype(np.float16) / 255.0
