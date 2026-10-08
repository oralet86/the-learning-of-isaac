"""Data loading with dynamic window construction from per-run frame files.

The dataset stores each run as a contiguous .npy of frames and a parallel
.npy of per-frame labels.  A per-frame split mask (train/val/test) is also
stored per run.  WindowedDataset builds sliding-window sample indices at
construction time, filtering to a specific split.  Frame data is
memory-mapped (via `components.shard_store`) so the memory footprint does
not grow with the number of runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from components.data_loading import (
    build_loader,  # noqa: F401  (re-exported for inverse_dynamics.train)
)
from components.shard_store import ShardedArrayStore
from constants import SEED

if TYPE_CHECKING:
    from pathlib import Path

_SPLIT_TRAIN = 1
_SPLIT_VAL = 2
_SPLIT_TEST = 3

_SPLIT_MAP: dict[str, int] = {
    "train": _SPLIT_TRAIN,
    "val": _SPLIT_VAL,
    "test": _SPLIT_TEST,
}


class WindowedDataset(Dataset[tuple[Tensor, Tensor]]):
    """Dynamically builds frame windows from per-run contiguous frame files.

    Each run is stored as three files under ``processed_dir``, named
    ``{run_dir}_chunk_{id:04d}_*`` by ``inverse_dynamics.preprocess``::

        run_0000_chunk_0000_frames.npy   - (N, H, W) uint8
        run_0000_chunk_0000_labels.npy   - (N, K) uint8
        run_0000_chunk_0000_split.npy    - (N,) int8  (1=train, 2=val, 3=test)

    The dataset scans all runs, enumerates valid window starting positions,
    and retains only those whose *label frame* (determined by ``label_index``
    and ``stride``) belongs to the requested ``split``.
    """

    def __init__(
        self,
        processed_dir: Path,
        split: str,
        window_size: int,
        label_index: int,
        stride: int = 1,
        data_frac: float = 1.0,
    ) -> None:
        if split not in _SPLIT_MAP:
            raise ValueError(
                f"Unknown split {split!r}. Expected one of {list(_SPLIT_MAP)}"
            )
        self.processed_dir = processed_dir
        self.split = split
        self.split_value = _SPLIT_MAP[split]
        self.window_size = window_size
        self.label_index = label_index
        self.stride = stride
        self.data_frac = data_frac

        # Runs are `.npy` shards of three parallel arrays; the store owns
        # discovery, row counts, lazy memory-mapping, and the pickling that
        # lets `DataLoader` workers reopen the maps.
        self._store = ShardedArrayStore(
            processed_dir, ("frames", "labels", "split"), stem_prefix="run"
        )
        self._run_lengths: list[int] = []

        self._samples: list[tuple[int, int]] = []

        self._build_index()

    # ------------------------------------------------------------------
    # Sample index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        # Run lengths come from the `.npy` headers, so indexing every run
        # never faults in a byte of frame data.
        self._run_lengths = self._store.shard_lengths

        for run_idx, n_frames in enumerate(self._run_lengths):
            # Max valid window start: last frame must be in-bounds
            last_frame_pos = (self.window_size - 1) * self.stride
            max_start = n_frames - last_frame_pos - 1
            if max_start < 0:
                continue

            # Load split mask to filter by label frame
            split_mask = self._store.array(run_idx, "split")

            for frame_start in range(max_start + 1):
                label_pos = frame_start + self.label_index * self.stride
                if split_mask[label_pos] == self.split_value:
                    self._samples.append((run_idx, frame_start))

        if self.data_frac < 1.0 and self._samples:
            rng = np.random.default_rng(SEED)
            n_keep = max(1, int(len(self._samples) * self.data_frac))
            idx = rng.choice(len(self._samples), size=n_keep, replace=False)
            self._samples = [self._samples[i] for i in sorted(idx)]

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        run_idx, frame_start = self._samples[index]
        frame_mmap = self._store.array(run_idx, "frames")
        label_mmap = self._store.array(run_idx, "labels")

        # Build strided indices
        indices = np.arange(
            frame_start,
            frame_start + self.window_size * self.stride,
            self.stride,
            dtype=np.int32,
        )

        label_pos = frame_start + self.label_index * self.stride
        frames = frame_mmap[indices]
        label = label_mmap[label_pos]

        return (
            torch.from_numpy(frames),
            torch.from_numpy(label.copy()),
        )
