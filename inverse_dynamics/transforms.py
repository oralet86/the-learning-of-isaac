"""Transforms applied during preprocessing.

Two tiers:
  FrameTransform  — operates on (frames, labels) before windowing; may change
                    the sequence length (trimming, AFK removal).
  WindowTransform — operates on windows in-place after windowing; may return a
                    boolean keep mask to filter windows that carry no signal.
"""

from typing import Protocol, runtime_checkable

import numpy as np

# ---------------------------------------------------------------------------
# Protocols
# ---------------------------------------------------------------------------


@runtime_checkable
class FrameTransform(Protocol):
    """Transform that filters or mutates the raw frame sequence.

    Receives (frames, labels) and returns the potentially shorter sequence.
    No in-place mutation guarantees.
    """

    name: str

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class WindowTransform(Protocol):
    """Transform applied to window arrays after window construction.

    Mutates x in-place. Optionally returns a boolean keep mask: rows where
    the mask is False are discarded by the caller (along with corresponding
    labels and sample map entries).
    """

    name: str

    def __call__(self, x: np.ndarray) -> np.ndarray | None: ...


# ---------------------------------------------------------------------------
# FrameTransforms
# ---------------------------------------------------------------------------


class TrimEdges:
    """Trim a fraction of frames from the start and end of a run.

    Runs captured via OBS start and stop cleanly, but keeping a small trim
    margin defends against UI flicker during start/stop.
    """

    name: str = "trim_edges"

    def __init__(self, frac: float = 0.01) -> None:
        self.frac = frac

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(frames)
        trim_n = int(n * self.frac)
        if trim_n == 0:
            return frames, labels
        return frames[trim_n:-trim_n], labels[trim_n:-trim_n]


class RemoveAFK:
    """Remove runs of near-identical frames longer than *max_same*.

    Keeps the first frame of each run; subsequent identical frames are dropped.
    Intended to strip idle / AFK segments where nothing changes on screen.
    """

    name: str = "remove_afk"

    def __init__(self, max_same: int = 240, epsilon: float = 1.0) -> None:
        self.max_same = max_same
        self.epsilon = epsilon

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(frames)
        if n <= 1 or self.max_same <= 0:
            return frames, labels

        diffs = np.mean(
            np.abs(frames[1:].astype(np.float32) - frames[:-1].astype(np.float32)),
            axis=(1, 2),
        )
        boundaries = np.where(diffs > self.epsilon)[0]
        boundaries = np.concatenate([[0], boundaries + 1, [n]])

        keep_mask = np.ones(n, dtype=bool)
        for i in range(len(boundaries) - 1):
            start, end = int(boundaries[i]), int(boundaries[i + 1])
            run_len = end - start
            if run_len > self.max_same:
                keep_mask[start + 1 : end] = False

        return frames[keep_mask], labels[keep_mask]


# ---------------------------------------------------------------------------
# WindowTransforms
# ---------------------------------------------------------------------------


class SignedDiffTransform:
    """Replace frames 1..C-1 with signed frame differences centred at 128.

    Keeps frame 0 (the reference frame) as raw grayscale. For all subsequent
    frames, computes the signed difference from the previous frame and maps it
    to [0, 255] with 128 representing no change:
        - Values > 128 indicate the pixel got brighter
        - Values < 128 indicate the pixel got darker
        - 128 means no change

    This representation explicitly encodes motion direction and magnitude.
    """

    name: str = "signed_diff"

    def __call__(self, x: np.ndarray) -> None:
        n_ch = x.shape[1]
        for i in range(1, n_ch):
            prev = x[:, i - 1].astype(np.int16)
            curr = x[:, i].astype(np.int16)
            x[:, i] = np.clip(128 + curr - prev, 0, 255).astype(np.uint8)


class StaticWindowFilter:
    """Remove windows where all consecutive frame pairs are pixel-identical.

    A window whose frames are all duplicates carries no motion signal and is
    safe to discard.
    """

    name: str = "static_window_filter"

    def __init__(self, tolerance: float = 0.0) -> None:
        self.tolerance = tolerance

    def __call__(self, x: np.ndarray) -> np.ndarray | None:
        n_ch = x.shape[1]
        if n_ch < 2:
            return None

        diffs = np.abs(x[:, 1:].astype(np.float32) - x[:, :-1].astype(np.float32))
        max_diffs = diffs.max(axis=(1, 2, 3))
        keep = max_diffs > self.tolerance

        if keep.all():
            return None
        return keep
