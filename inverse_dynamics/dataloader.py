"""Data loading utilities for chunked datasets."""

from bisect import bisect_right
from pathlib import Path  # noqa: TC003
from typing import override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from constants import NUM_WORKERS


class ChunkedDataset(Dataset[tuple[Tensor, Tensor]]):
    """Lazily loads frame window chunks from a directory.

    Expects a directory containing paired files:
        chunk_0000_X.npy, chunk_0000_y.npy
        chunk_0001_X.npy, chunk_0001_y.npy
        ...

    Caches the most recently loaded chunk to avoid redundant disk reads.
    Sequential access (DataLoader default) means each chunk is loaded once.
    """

    def __init__(self, chunk_dir: Path) -> None:
        self.chunk_dir = chunk_dir
        self.X_paths = sorted(chunk_dir.glob("chunk_*_X.npy"))
        if not self.X_paths:
            raise ValueError(f"No chunk_*_X.npy files found in {chunk_dir}")

        self.chunk_sizes: list[int] = []
        for x_path in self.X_paths:
            with open(x_path, "rb") as f:
                version = np.lib.format.read_magic(f)
                if version[0] == 1:
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
                self.chunk_sizes.append(int(shape[0]))

        self.cumulative_sizes: list[int] = []
        total = 0
        for size in self.chunk_sizes:
            total += size
            self.cumulative_sizes.append(total)

        self._cache: tuple[int, np.ndarray, np.ndarray] | None = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    @override
    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(
                f"Index {index} out of range for dataset of size {len(self)}"
            )

        chunk_idx = bisect_right(self.cumulative_sizes, index)
        local_idx = index - (
            self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else 0
        )

        if self._cache is None or self._cache[0] != chunk_idx:
            x_path = self.X_paths[chunk_idx]
            y_path = x_path.parent / x_path.name.replace("_X.npy", "_y.npy")
            x_chunk = np.load(x_path, mmap_mode="r")
            y_chunk = np.load(y_path, mmap_mode="r")
            self._cache = (chunk_idx, x_chunk, y_chunk)

        _, x_chunk, y_chunk = self._cache
        x = torch.from_numpy(x_chunk[local_idx].copy())
        y = torch.from_numpy(y_chunk[local_idx].copy())
        return x, y


def build_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    drop_last: bool = True,
) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
