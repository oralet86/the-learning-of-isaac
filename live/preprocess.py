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


def downscale_4x(frame: np.ndarray) -> np.ndarray:
    """Downscale an RGB frame by 4× using majority-vote (mode) pooling.

    For each 4×4 non-overlapping block of pixels, picks the most
    frequent colour per channel.  Designed for nearest-neighbour
    upscaled pixel art where a single game pixel occupies a 4×4 block
    on screen.

    Args:
        frame: ``(H, W, 3)`` uint8 or fp16 (raw uint8 values) array.

    Returns:
        ``(H//4, W//4, 3)`` uint8 array.
    """
    if frame.dtype == np.float16:
        frame = frame.astype(np.uint8)

    h, w, c = frame.shape
    new_h, new_w = h // 4, w // 4

    trimmed = frame[: new_h * 4, : new_w * 4]

    blocks = trimmed.reshape(new_h, 4, new_w, 4, c).transpose(0, 2, 1, 3, 4)
    blocks = blocks.reshape(new_h, new_w, 16, c)

    result = np.empty((new_h, new_w, c), dtype=np.uint8)
    for ch in range(c):
        flat = blocks[:, :, :, ch].reshape(-1, 16)
        modes = np.array([np.bincount(row).argmax() for row in flat])
        result[:, :, ch] = modes.reshape(new_h, new_w)

    return result
