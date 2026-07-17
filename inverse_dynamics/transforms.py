"""Dataset-level transforms applied to frame window arrays.

Transforms operate on chunks of frame windows, mutating X in-place to minimize
memory usage. They are applied during preprocessing before train/val/test splitting.
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class DatasetTransform(Protocol):
    """Protocol for dataset-level transforms.

    A transform receives a chunk of frame windows and mutates it in-place.
    No copies are made, no values are returned.
    """

    name: str

    def __call__(self, X: np.ndarray) -> None:
        """Apply the transform to frame windows in-place.

        Args:
            X: Frame windows, shape (N, C, H, W), dtype uint8.
               Mutated in-place; no return value.
        """
        ...


class SignedDiffTransform:
    """Replace frames 1..C-1 with signed frame differences centered at 128.

    Keeps frame 0 (the reference frame) as raw grayscale. For all subsequent
    frames, computes the signed difference from the previous frame and maps it
    to [0, 255] with 128 representing no change:
        - Values > 128 indicate the pixel got brighter
        - Values < 128 indicate the pixel got darker
        - 128 means no change

    This representation explicitly encodes motion direction and magnitude,
    which can help the model focus on what changed between frames rather than
    absolute pixel values.

    All differences are computed from the original frames before any channel
    replacement, ensuring consistency.

    Works with any number of input channels (frame window size).
    """

    name: str = "signed_diff"

    def __call__(self, X: np.ndarray) -> None:
        """Apply signed diff transform to frame windows in-place.

        Args:
            X: Frame windows, shape (N, C, H, W), dtype uint8.
               Channel 0 = frame t (kept raw)
               Channel i = frame t+i (replaced with diff from t+i-1)

               Mutated in-place; no return value.
        """
        n_ch = X.shape[1]
        for i in range(1, n_ch):
            prev = X[:, i - 1].astype(np.int16)
            curr = X[:, i].astype(np.int16)
            X[:, i] = np.clip(128 + curr - prev, 0, 255).astype(np.uint8)
