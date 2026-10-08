"""Sharded `.npy` storage shared by every pipeline's dataset.

Training data is stored as a directory of `.npy` shards, where one logical
shard is a group of parallel arrays sharing a row axis (frames alongside labels
alongside a split mask). Datasets memory-map those arrays so a
`__getitem__` is a page-cache read rather than a decode, and so the
resident set does not grow with the dataset.

`ShardedArrayStore` owns the parts that are identical between pipelines
-- discovery, row counts, lazy memory-mapping, and the pickle dance that
`DataLoader` workers need -- leaving each dataset to define only what a
sample *is*. `write_shard` is the matching writer, so producers and
consumers agree about file naming in exactly one place.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path


def shard_path(directory: Path, stem: str, array_name: str) -> Path:
    """Path of one array of shard `stem`: `{directory}/{stem}_{array_name}.npy`."""
    return directory / f"{stem}_{array_name}.npy"


def write_shard(
    directory: Path, stem: str, arrays: dict[str, np.ndarray]
) -> list[Path]:
    """Writes one shard's parallel arrays as `{stem}_{name}.npy` files.

    Every array must share the same length (its rows are one shard's
    samples); mismatched lengths are a producer bug that would silently
    misalign frames and labels, so it raises instead. Creates
    `directory` if needed and returns the written paths in `arrays` order.

    Args:
        directory: Output directory; created if it does not exist.
        stem: Shard stem, conventionally `f"shard_{index:04d}"`.
        arrays: Array name (e.g. `"frames"`) to the rows to store.

    Raises:
        ValueError: If `arrays` is empty or the arrays disagree on length.
    """
    if not arrays:
        raise ValueError(f"no arrays given for shard {stem!r}")
    lengths = {name: len(array) for name, array in arrays.items()}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"shard {stem!r} arrays disagree on row count: {lengths} -- parallel "
            "arrays must have one row per sample"
        )
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, array in arrays.items():
        path = shard_path(directory, stem, name)
        np.save(path, array)
        written.append(path)
    return written


def read_npy_length(path: Path) -> int:
    """Rows in the `.npy` at `path`, read from its header alone.

    Only the header is parsed, so calling this over a directory of
    multi-gigabyte shards costs a few reads instead of faulting in every
    byte -- which is what lets a dataset build its sample index up front.
    """
    with open(path, "rb") as f:
        version = np.lib.format.read_magic(f)
        if version[0] == 1:
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_1_0(f)
        else:
            shape, _fortran_order, _dtype = np.lib.format.read_array_header_2_0(f)
    if not shape:
        raise ValueError(f"{path} holds a 0-dim array, which has no rows")
    return int(shape[0])


class ShardedArrayStore:
    """Lazily memory-mapped set of `.npy` shards holding parallel arrays.

    A shard is one stem (e.g. `shard_0007`) with one file per entry of
    `array_names` (`shard_0007_frames.npy`, `shard_0007_labels.npy`, ...).
    All arrays of a shard must have the same number of rows; row `i` of
    each describes the same sample.

    Memory-maps are opened on first access rather than in `__init__`, and
    are dropped when the store is pickled, so a store built in the parent
    process re-opens cleanly inside every `DataLoader` worker.

    Args:
        directory: Directory holding the shard files.
        array_names: Array names every shard must provide. The first is
            the one discovery globs on.
        stem_prefix: Shard stem prefix to match (default `"shard"`).

    Raises:
        FileNotFoundError: If `directory` does not exist, holds no shards,
            or a shard is missing one of `array_names`.
    """

    def __init__(
        self,
        directory: Path,
        array_names: Sequence[str],
        stem_prefix: str = "shard",
    ) -> None:
        if not array_names:
            raise ValueError("array_names must name at least one array")
        self.directory = directory
        self.array_names = tuple(array_names)
        self.stem_prefix = stem_prefix
        self.stems: list[str] = self._discover()
        self._mmaps: dict[tuple[int, str], np.ndarray] = {}

    def _discover(self) -> list[str]:
        """Finds every complete shard stem under `directory`, sorted.

        Globs on the first array name and requires each match to have
        every sibling array. A half-written shard is a data-integrity
        problem, so it raises rather than skipping the shard silently.
        """
        if not self.directory.is_dir():
            raise FileNotFoundError(f"shard directory {self.directory} does not exist")
        primary = self.array_names[0]
        suffix = f"_{primary}.npy"
        stems = sorted(
            path.name[: -len(suffix)]
            for path in self.directory.glob(f"{self.stem_prefix}_*{suffix}")
        )
        if not stems:
            raise FileNotFoundError(
                f"no {self.stem_prefix}_*{suffix} shards under {self.directory}"
            )
        for stem in stems:
            for name in self.array_names[1:]:
                path = shard_path(self.directory, stem, name)
                if not path.exists():
                    raise FileNotFoundError(
                        f"shard {stem!r} is missing {path.name} -- every shard must "
                        f"provide all of {list(self.array_names)}"
                    )
        return stems

    def __len__(self) -> int:
        """Number of shards (not rows -- see `shard_lengths`)."""
        return len(self.stems)

    @property
    def shard_lengths(self) -> list[int]:
        """Row count of each shard, from `.npy` headers only."""
        return [
            read_npy_length(shard_path(self.directory, stem, self.array_names[0]))
            for stem in self.stems
        ]

    @property
    def total_rows(self) -> int:
        """Total rows across every shard."""
        return sum(self.shard_lengths)

    def path(self, shard_index: int, array_name: str) -> Path:
        """Path of `array_name` in shard `shard_index`."""
        return shard_path(self.directory, self.stems[shard_index], array_name)

    def array(self, shard_index: int, array_name: str) -> np.ndarray:
        """Memory-mapped `array_name` of shard `shard_index`.

        The map is opened on first request and cached for the lifetime of
        the store (or until it is pickled into a worker).
        """
        if array_name not in self.array_names:
            raise KeyError(
                f"unknown array {array_name!r}; this store holds "
                f"{list(self.array_names)}"
            )
        key = (shard_index, array_name)
        mmap = self._mmaps.get(key)
        if mmap is None:
            mmap = np.load(self.path(shard_index, array_name), mmap_mode="r")
            self._mmaps[key] = mmap
        return mmap

    def __getstate__(self) -> dict[str, object]:
        """Pickles without the memory-maps -- workers reopen them lazily."""
        state = self.__dict__.copy()
        state["_mmaps"] = {}
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
