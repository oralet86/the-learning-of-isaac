"""Mask functions used during preprocessing.

Each function returns a boolean keep-mask (``True`` = keep the frame,
``False`` = drop it).  Multiple masks can be combined via ``&`` and
applied in a single pass to avoid repeated copies.
"""

import numpy as np

AFK_SLICE_SIZE = 5000


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


def empty_frames_mask(labels: np.ndarray) -> np.ndarray:
    """Return boolean keep-mask for frames where at least one key is pressed."""
    return labels.sum(axis=1) > 0


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
    total_n: int,
    is_first: bool,
    is_last: bool,
    trim_frac: float,
) -> np.ndarray:
    """Return boolean keep-mask trimming *trim_frac* from global recording edges.

    The trim amount is computed from *total_n* (total recording frames) so the
    effective percentage is consistent regardless of chunking.  Only chunks at
    the global edges (*is_first* or *is_last*) are trimmed.
    """
    mask = np.ones(n, dtype=bool)
    trim_n = int(total_n * trim_frac)
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
    """Return boolean keep-mask for frames whose keys all belong to *key_set*.

    Frames containing any key not in *key_set* get ``False``.
    """
    mask = np.ones(len(keys_list), dtype=bool)
    for i, frame_keys in enumerate(keys_list):
        if any(k not in key_set for k in frame_keys):
            mask[i] = False
    return mask


def clean_keys(
    keys_list: list[list[str]],
    key_set: set[str],
) -> list[list[str]]:
    """Keep only keys that belong to *key_set*, per frame."""
    return [[k for k in frame_keys if k in key_set] for frame_keys in keys_list]
