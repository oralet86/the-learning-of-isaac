"""Transforms applied during preprocessing.

FrameTransform operates on (frames, labels) before storage; it may change
the sequence length (trimming edges, AFK removal, cleaning conflicting
directions, and removing frames with too many simultaneous key presses).

Also provides mask-returning variants that produce boolean keep-masks
without copying frame data, so multiple transforms can be combined and
applied in a single pass.
"""

from typing import Protocol, runtime_checkable

import numpy as np

AFK_SLICE_SIZE = 5000


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


class RemoveConflictingDirections:
    """Drop frames where mutually exclusive direction keys are simultaneously active.

    Mutually exclusive groups:
      - WASD movement: w/s, a/d
      - Arrow shooting: up/down, left/right

    Frames with both keys in any pair set are removed.
    """

    name: str = "remove_conflicting_directions"

    def __init__(self, conflicting_pairs: list[tuple[int, int]]) -> None:
        self.conflicting_pairs = conflicting_pairs

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = np.ones(len(labels), dtype=bool)
        for i, j in self.conflicting_pairs:
            mask &= ~((labels[:, i] == 1) & (labels[:, j] == 1))
        return frames[mask], labels[mask]


class RemoveTooManyKeys:
    """Drop frames where more than *max_simul* keys are simultaneously active.

    Intended to catch rare timing artifacts where key events from adjacent
    frames bleed together due to OBS frame capture jitter.
    """

    name: str = "remove_too_many_keys"

    def __init__(self, max_simul: int) -> None:
        self.max_simul = max_simul

    def __call__(
        self, frames: np.ndarray, labels: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        mask = labels.sum(axis=1) <= self.max_simul
        return frames[mask], labels[mask]


# ---------------------------------------------------------------------------
# Mask-returning transform functions
# ---------------------------------------------------------------------------


def conflicting_mask(
    labels: np.ndarray,
    conflicting_pairs: list[tuple[int, int]],
) -> np.ndarray:
    """Return boolean keep-mask for frames without conflicting direction pairs.

    Frames where both keys in any pair from *conflicting_pairs* are active get
    ``False`` in the returned mask.
    """
    mask = np.ones(len(labels), dtype=bool)
    for i, j in conflicting_pairs:
        mask &= ~((labels[:, i] == 1) & (labels[:, j] == 1))
    return mask


def too_many_keys_mask(labels: np.ndarray, max_simul: int) -> np.ndarray:
    """Return boolean keep-mask for frames with at most *max_simul* active keys."""
    return labels.sum(axis=1) <= max_simul


def afk_mask(
    frames: np.ndarray,
    max_same: int,
    epsilon: float,
    slice_size: int = AFK_SLICE_SIZE,
) -> np.ndarray:
    """Return boolean keep-mask dropping frames in long AFK/idle runs.

    Computes per-frame-pair mean absolute difference via sliced processing to
    avoid allocating large int16 intermediate arrays.  Runs of near-identical
    frames longer than *max_same* have all but the first frame marked ``False``.
    """
    n = len(frames)
    if n <= 1 or max_same <= 0:
        return np.ones(n, dtype=bool)

    diffs = np.empty(n - 1, dtype=np.float32)
    for start in range(0, n - 1, slice_size):
        end = min(start + slice_size, n - 1)
        sl = frames[start : end + 1]
        sl_i16 = sl.astype(np.int16)
        diffs[start:end] = np.mean(
            np.abs(sl_i16[1:] - sl_i16[:-1]),
            axis=(1, 2),
        )

    boundaries = np.where(diffs > epsilon)[0]
    boundaries = np.concatenate([[0], boundaries + 1, [n]])

    keep_mask = np.ones(n, dtype=bool)
    run_lengths = np.diff(boundaries)
    long_runs = np.where(run_lengths > max_same)[0]
    for idx in long_runs:
        r_start = int(boundaries[idx]) + 1
        r_end = int(boundaries[idx + 1])
        keep_mask[r_start:r_end] = False

    return keep_mask


def edge_trim_mask(
    n: int,
    is_first: bool,
    is_last: bool,
    trim_frac: float,
) -> np.ndarray:
    """Return boolean keep-mask trimming *trim_frac* from chunk edges."""
    mask = np.ones(n, dtype=bool)
    trim_n = int(n * trim_frac)
    if trim_n <= 0:
        return mask
    if is_first:
        mask[:trim_n] = False
    if is_last:
        mask[-trim_n:] = False
    return mask


def foreign_keys_mask(
    keys_list: list[list[str]],
    key_set: set[str],
) -> np.ndarray:
    """Return boolean mask that is ``True`` for frames with keys outside *key_set*.

    The returned mask marks frames to *drop* — it is ``True`` for foreign.
    """
    mask = np.zeros(len(keys_list), dtype=bool)
    for i, frame_keys in enumerate(keys_list):
        if any(k not in key_set for k in frame_keys):
            mask[i] = True
    return mask


def clean_keys(
    keys_list: list[list[str]],
    key_set: set[str],
) -> list[list[str]]:
    """Keep only keys that belong to *key_set*, per frame."""
    return [[k for k in frame_keys if k in key_set] for frame_keys in keys_list]
