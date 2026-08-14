"""Shared `torch.utils.data.DataLoader` construction for every pipeline.

`inverse_dynamics.dataloader` and `visual_extraction.train` used to build
their loaders independently; this helper is the single place that
decides worker-count-dependent behavior (`persistent_workers`,
`prefetch_factor`) and pinning, so tuning learned on one pipeline
transfers to the other.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.utils.data import DataLoader, Dataset

from constants import NUM_WORKERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def build_loader[T](
    dataset: Dataset[tuple[Tensor, T]],
    batch_size: int,
    shuffle: bool,
    num_workers: int = NUM_WORKERS,
    pin_memory: bool = True,
    drop_last: bool = True,
    collate_fn: Callable[..., object] | None = None,
) -> DataLoader[tuple[Tensor, T]]:
    """Builds a `DataLoader` with the project's shared worker settings.

    Args:
        dataset: Any `Dataset` yielding `(input_tensor, target)` pairs.
        batch_size: Batch size for this split.
        shuffle: Whether to shuffle batches (train split only).
        num_workers: Worker-process count for prefetching.
        pin_memory: Pin host-side batches so `non_blocking=True` device
            transfers in the training loops actually overlap compute.
        drop_last: Drop the trailing partial batch (train: `True`, so
            every batch has the full batch size; val/test: `False`).
        collate_fn: Custom batch collation (e.g.
            `visual_extraction.dataset.collate_label_bundles`); left to
            PyTorch's default when `None`.

    Returns:
        A configured `DataLoader` for `dataset`.
    """
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
