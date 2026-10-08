"""Shared `torch.utils.data.DataLoader` construction for every pipeline.

This helper is the single place that decides worker-count-dependent
behavior (`persistent_workers`, `prefetch_factor`) and pinning, so tuning
learned on one pipeline transfers to any other.
"""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Dataset

from constants import NUM_WORKERS

if TYPE_CHECKING:
    from collections.abc import Callable


def build_loader[T](
    dataset: Dataset[T],
    batch_size: int,
    shuffle: bool,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Callable[..., object] | None = None,
) -> DataLoader[T]:
    """Builds a `DataLoader` with the project's shared worker settings.

    Args:
        dataset: Any `Dataset`. Sample shape is up to the dataset --
            `inverse_dynamics` yields `(frames, label)` pairs.
        batch_size: Batch size for this split.
        shuffle: Whether to shuffle batches (train split only).
        num_workers: Worker-process count for prefetching.
        pin_memory: Pin host-side batches so `non_blocking=True` device
            transfers in the training loops actually overlap compute.
        drop_last: Drop the trailing partial batch. Every split uses
            `True`: `torch.compile` + cudagraphs capture one graph per
            batch shape, and a partial last batch triggers a re-capture
            with its own private memory pool (which OOMed a val/test run).
        collate_fn: Custom batch collation; left to PyTorch's default
            when `None`.

    Returns:
        A configured `DataLoader` for `dataset`.
    """
    # Python 3.14 defaults multiprocessing to `forkserver`, whose worker
    # spawn pickles the process object -- and torch's tensor reducer shares
    # every CPU storage by forwarding a `/dev/shm` fd (`reduce_storage` ->
    # `_share_fd_cpu_`), which once blew past forkserver's 256-fd cap
    # (`ValueError: too many fds`) on a dataset that cached one tensor per
    # sample. Both datasets now hold a handful of large tensors instead, but
    # `fork` remains the right choice: it inherits them copy-on-write with
    # zero pickling, and our workers only read memmaps and collate (never
    # touch CUDA), so fork is safe.
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
        multiprocessing_context="fork" if sys.platform == "linux" else None,
    )
