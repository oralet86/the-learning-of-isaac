"""Frame preprocessing for the RL pipeline.

Transforms raw fp16 frames from StreamGame into model-ready inputs.
Each stage is a separate callable so new transforms can be inserted
without touching the loop or the agent.
"""

from __future__ import annotations

from functools import lru_cache

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


@lru_cache(maxsize=8)
def _center_pixel_indices(
    h: int, w: int, target_h: int, target_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the source row/col indices sampled by ``downscale_16_9_center_pixel``.

    For each output pixel, returns the index of the pixel at the center of
    its corresponding source block. Depends only on the frame shape (which
    is constant across a live capture session), so results are cached to
    avoid recomputing on every frame in the 20 Hz hot loop.

    Args:
        h: Source frame height.
        w: Source frame width.
        target_h: Output height.
        target_w: Output width.

    Returns:
        ``(row_idx, col_idx)``, each an ``np.intp`` array sized
        ``target_h``/``target_w`` respectively.
    """
    sy = h / target_h
    sx = w / target_w
    row_idx = ((np.arange(target_h) + 0.5) * sy).astype(np.intp)
    col_idx = ((np.arange(target_w) + 0.5) * sx).astype(np.intp)
    return row_idx, col_idx


def downscale_16_9_center_pixel(
    frame: np.ndarray, target_w: int = 480, target_h: int = 270
) -> np.ndarray:
    """Downscale a 16:9 frame to ``(target_h, target_w)`` via center-pixel sampling.

    Splits the frame into a ``target_h`` × ``target_w`` grid of blocks and
    picks the pixel closest to the center of each block as the output
    pixel, rather than blending (``cv2.resize``) or majority-voting across
    the block. This is fast (no per-block Python loop or interpolation)
    and well suited to nearest-neighbour-upscaled pixel art, which is why
    it's used on the hot path of live gameplay capture (~20 frames/sec).

    Args:
        frame: ``(H, W, 3)`` uint8 or fp16 (raw uint8 values) array. ``H``
            and ``W`` must be in an exact 16:9 ratio matching
            ``target_w / target_h``.
        target_w: Output width.
        target_h: Output height.

    Returns:
        ``(target_h, target_w, 3)`` uint8 array.

    Raises:
        ValueError: If *frame*'s aspect ratio isn't exactly
            ``target_w:target_h``.
    """
    if frame.dtype == np.float16:
        frame = frame.astype(np.uint8)

    h, w = frame.shape[:2]
    if w * target_h != h * target_w:
        raise ValueError(
            f"Frame resolution {w}x{h} is not exactly {target_w}:{target_h}"
        )

    row_idx, col_idx = _center_pixel_indices(h, w, target_h, target_w)
    return frame[row_idx[:, None], col_idx[None, :]]
