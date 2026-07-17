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
    """Replace frames 1-3 with signed frame differences centered at 128.

    Keeps frame 0 (the reference frame) as raw grayscale. For frames 1, 2, 3,
    computes the signed difference from the previous frame and maps it to
    [0, 255] with 128 representing no change:
        - Values > 128 indicate the pixel got brighter
        - Values < 128 indicate the pixel got darker
        - 128 means no change

    This representation explicitly encodes motion direction and magnitude,
    which can help the model focus on what changed between frames rather than
    absolute pixel values.

    All differences are computed from the original frames before any channel
    replacement, ensuring consistency.
    """

    name: str = "signed_diff"

    def __call__(self, X: np.ndarray) -> None:
        """Apply signed diff transform to frame windows in-place.

        Args:
            X: Frame windows, shape (N, 4, H, W), dtype uint8.
               Channel 0 = frame t (kept raw)
               Channel 1 = frame t+1 (replaced with diff from t)
               Channel 2 = frame t+2 (replaced with diff from t+1)
               Channel 3 = frame t+3 (replaced with diff from t+2)

               Mutated in-place; no return value.
        """
        f0 = X[:, 0].astype(np.int16)
        f1 = X[:, 1].astype(np.int16)
        f2 = X[:, 2].astype(np.int16)
        f3 = X[:, 3].astype(np.int16)

        X[:, 1] = np.clip(128 + f1 - f0, 0, 255).astype(np.uint8)
        X[:, 2] = np.clip(128 + f2 - f1, 0, 255).astype(np.uint8)
        X[:, 3] = np.clip(128 + f3 - f2, 0, 255).astype(np.uint8)
