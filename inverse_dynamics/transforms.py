"""Transforms applied during preprocessing.

FrameTransform operates on (frames, labels) before storage; it may change
the sequence length (trimming edges, AFK removal).
"""

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class FrameTransform(Protocol):
    """Transform that filters or mutates the raw frame sequence.

    Receives (frames, labels) and returns the potentially shorter sequence.
    """

    name: str

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]: ...


class TrimEdges:
    """Trim a fraction of frames from the start and/or end of a run.

    Runs captured via OBS start and stop cleanly, but keeping a small trim
    margin defends against UI flicker during start/stop.
    """

    name: str = "trim_edges"

    def __init__(self, frac: float = 0.01, side: str = "both") -> None:
        self.frac = frac
        self.side = side

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        n = len(frames)
        trim_n = int(n * self.frac)
        if trim_n == 0:
            return frames, labels
        if self.side == "start":
            return frames[trim_n:], labels[trim_n:]
        if self.side == "end":
            return frames[:-trim_n], labels[:-trim_n]
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
            np.abs(frames[1:].astype(np.int16) - frames[:-1].astype(np.int16)),
            axis=(1, 2),
        )
        boundaries = np.where(diffs > self.epsilon)[0]
        boundaries = np.concatenate([[0], boundaries + 1, [n]])

        keep_mask = np.ones(n, dtype=bool)
        run_lengths = np.diff(boundaries)
        long_runs = np.where(run_lengths > self.max_same)[0]
        for idx in long_runs:
            r_start = int(boundaries[idx]) + 1
            r_end = int(boundaries[idx + 1])
            keep_mask[r_start:r_end] = False

        return frames[keep_mask], labels[keep_mask]
