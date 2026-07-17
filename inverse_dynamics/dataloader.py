"""Data loading utilities for chunked datasets."""

from bisect import bisect_right
from pathlib import Path  # noqa: TC003

import numpy as np
import torch


class ChunkedDataset(torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Lazily loads frame window chunks from a directory.

    Expects a directory containing paired files:
        chunk_0000_X.npy, chunk_0000_y.npy
        chunk_0001_X.npy, chunk_0001_y.npy
        ...

    Caches the most recently loaded chunk to avoid redundant disk reads.
    Sequential access (DataLoader default) means each chunk is loaded once.
    """

    def __init__(self, chunk_dir: Path):
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

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if idx < 0:
            idx += len(self)
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of size {len(self)}"
            )

        chunk_idx = bisect_right(self.cumulative_sizes, idx)
        local_idx = idx - (self.cumulative_sizes[chunk_idx - 1] if chunk_idx > 0 else 0)

        if self._cache is None or self._cache[0] != chunk_idx:
            x_path = self.X_paths[chunk_idx]
            y_path = x_path.parent / x_path.name.replace("_X.npy", "_y.npy")
            X_chunk = np.load(x_path, mmap_mode="r")
            y_chunk = np.load(y_path, mmap_mode="r")
            self._cache = (chunk_idx, X_chunk, y_chunk)

        _, X_chunk, y_chunk = self._cache
        x = torch.from_numpy(X_chunk[local_idx].copy())
        y = torch.from_numpy(y_chunk[local_idx].copy())
        return x, y


def build_loader(
    dataset: torch.utils.data.Dataset[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader[tuple[torch.Tensor, torch.Tensor]]:
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        persistent_workers=num_workers > 0,
        prefetch_factor=(2 if num_workers > 0 else None),
    )
