"""Data loading with dynamic window construction from per-run frame files.

The dataset stores each run as a contiguous .npy of frames and a parallel
.npy of per-frame labels.  A per-frame split mask (train/val/test) is also
stored per run.  WindowedDataset builds sliding-window sample indices at
construction time, filtering to a specific split.  Frame data is
memory-mapped so the memory footprint does not grow with the number of runs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, override

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from constants import NUM_WORKERS, SEED

if TYPE_CHECKING:
    from collections.abc import Callable
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

    Each run is stored as three files under ``processed_dir``::

        run_0000_frames.npy   - (N, H, W) uint8
        run_0000_labels.npy   - (N, K) uint8
        run_0000_split.npy    - (N,) int8  (1=train, 2=val, 3=test)

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

        self._frame_paths: list[Path] = []
        self._label_paths: list[Path] = []
        self._split_paths: list[Path] = []
        self._run_lengths: list[int] = []
        self._frame_mmaps: list[np.ndarray] = []
        self._label_mmaps: list[np.ndarray] = []
        self._split_masks: list[np.ndarray] = []

        self._samples: list[tuple[int, int]] = []

        self._discover_runs()
        self._build_index()

    # ------------------------------------------------------------------
    # Run discovery
    # ------------------------------------------------------------------

    def _discover_runs(self) -> None:
        frame_files = sorted(self.processed_dir.glob("run_*_frames.npy"))
        for frame_path in frame_files:
            stem = frame_path.name.replace("_frames.npy", "")
            label_path = self.processed_dir / f"{stem}_labels.npy"
            split_path = self.processed_dir / f"{stem}_split.npy"
            if not label_path.exists() or not split_path.exists():
                continue
            self._frame_paths.append(frame_path)
            self._label_paths.append(label_path)
            self._split_paths.append(split_path)

    # ------------------------------------------------------------------
    # Sample index
    # ------------------------------------------------------------------

    def _build_index(self) -> None:
        if not self._frame_paths:
            return

        for run_idx in range(len(self._frame_paths)):
            # Determine run length from mmap shape (load lazily here)
            frame_path = self._frame_paths[run_idx]
            with open(frame_path, "rb") as f:
                version = np.lib.format.read_magic(f)
                if version[0] == 1:
                    shape, _, _ = np.lib.format.read_array_header_1_0(f)
                else:
                    shape, _, _ = np.lib.format.read_array_header_2_0(f)
            n_frames = int(shape[0])
            self._run_lengths.append(n_frames)

            # Max valid window start: last frame must be in-bounds
            last_frame_pos = (self.window_size - 1) * self.stride
            max_start = n_frames - last_frame_pos - 1
            if max_start < 0:
                continue

            # Load split mask to filter by label frame
            split_mask = np.load(self._split_paths[run_idx], mmap_mode="r")
            self._split_masks.append(split_mask)

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
    # File I/O helpers
    # ------------------------------------------------------------------

    def _ensure_mmaps(self) -> None:
        """Open memory-maps lazily (once, after potential unpickle)."""
        if len(self._frame_mmaps) == len(self._frame_paths) > 0:
            return
        self._frame_mmaps = []
        self._label_mmaps = []
        self._split_masks = []
        for i in range(len(self._frame_paths)):
            self._frame_mmaps.append(np.load(self._frame_paths[i], mmap_mode="r"))
            self._label_mmaps.append(np.load(self._label_paths[i], mmap_mode="r"))
            self._split_masks.append(np.load(self._split_paths[i], mmap_mode="r"))

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

        self._ensure_mmaps()

        run_idx, frame_start = self._samples[index]
        frame_mmap = self._frame_mmaps[run_idx]
        label_mmap = self._label_mmaps[run_idx]

        # Build strided indices
        indices = np.arange(
            frame_start,
            frame_start + self.window_size * self.stride,
            self.stride,
            dtype=np.int32,
        )

        frames = frame_mmap[indices]
        label = label_mmap[frame_start + self.label_index * self.stride]

        return (
            torch.from_numpy(frames.copy()),
            torch.from_numpy(label.copy()),
        )

    # ------------------------------------------------------------------
    # Pickle support (for multiprocessing DataLoader workers)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # Drop mmap objects — workers reopen them via _ensure_mmaps()
        state["_frame_mmaps"] = []
        state["_label_mmaps"] = []
        state["_split_masks"] = []
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)


def build_loader(
    dataset: Dataset[tuple[Tensor, Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Callable[..., object] | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        collate_fn=collate_fn,
        persistent_workers=num_workers > 0,
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def signed_diff_collate(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Vectorized collate function that encodes motion via signed frame differences.

    Frame 0 is kept as raw grayscale.  All subsequent frames are replaced by
    the signed difference from the previous frame, centred at 128:

        diff = 128 + frame_t - frame_{t-1}

    Values > 128 mean the pixel got brighter; < 128 means darker.
    This representation makes motion direction and magnitude explicit for
    the model while keeping pixel values in [0, 255].
    """
    frames_list, labels_list = zip(*batch, strict=False)
    frames = torch.stack(frames_list)  # (B, T, H, W)
    labels = torch.stack(labels_list)

    if frames.size(1) <= 1:
        return frames, labels

    result = frames.clone()
    diffs = frames[:, 1:].to(torch.int16) - frames[:, :-1].to(torch.int16)
    result[:, 1:] = torch.clamp(128 + diffs, 0, 255).to(torch.uint8)

    return result, labels
