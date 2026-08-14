"""`torch.utils.data.Dataset` over `datagen/output_augmented/{images,labels}`.

Each label JSON's `image_id` names both files: `labels/{image_id}.json`
and `images/{image_id}.png`. `partition_label_paths` computes the
train/val/test partition once (every label file is parsed exactly once
per run) and each `GroundTruthDataset` is handed its split's pre-filtered
paths, then caches its parsed `LabelBundle`s in memory so `__getitem__`
only decodes the PNG. Split is a deterministic hash of `scenario_id`
(not `image_id`), so every viewport crop from the same room visit lands
in the same split -- otherwise near-duplicate crops of one scenario
could leak across train/val/test.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Any, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf

from constants import TEST_FRAC, VAL_FRAC
from visual_extraction.labels import LabelBundle, encode_record

if TYPE_CHECKING:
    from pathlib import Path

"""
The game natively renders at 480x270 before any upscaling,
resizing a captured frame below that loses real information (the HUD becomes unreadable),
so this matches native resolution rather than the common "square 224" CNN-input convention.
`datagen.augment` downsamples captures to this size once, so `__getitem__`
does not resize.
"""
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 270


def _assign_split(
    scenario_id: str, val_fraction: float, test_fraction: float
) -> Literal["train", "val", "test"]:
    """Deterministic 3-way split for `scenario_id`.

    Buckets the sha256 hash of `scenario_id` into `[0, 1)`: the lowest
    `val_fraction` goes to validation, the highest `test_fraction` to
    test, everything between to training -- 80/10/10 with the project's
    `constants.VAL_FRAC`/`TEST_FRAC`. Same scenario always lands in the
    same split, so augmented variants of one room stay together.
    """
    digest = hashlib.sha256(scenario_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < val_fraction:
        return "val"
    if bucket >= 1.0 - test_fraction:
        return "test"
    return "train"


def _find_records(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.glob("*.json"))


def partition_label_paths(
    labels_dir: Path,
    val_fraction: float = VAL_FRAC,
    test_fraction: float = TEST_FRAC,
) -> dict[Literal["train", "val", "test"], list[Path]]:
    """Partitions every label file under `labels_dir` by its `scenario_id`.

    Globs and parses each label JSON exactly once per call, deriving the
    split from `scenario_id` (never `image_id`, so augmented variants of
    one room stay together). Callers hand each resulting list to a
    `GroundTruthDataset` -- constructing one dataset per split then
    requires no further file reads. Three-split construction in
    `visual_extraction/train.py` previously re-parsed the whole label
    directory once per dataset, three times per run.
    """
    partitioned: dict[Literal["train", "val", "test"], list[Path]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    for path in _find_records(labels_dir):
        record = json.loads(path.read_text())
        split = _assign_split(record["scenario_id"], val_fraction, test_fraction)
        partitioned[split].append(path)
    return partitioned


class GroundTruthDataset(Dataset[tuple[torch.Tensor, LabelBundle]]):
    """One captured frame per sample: `(image_tensor, LabelBundle)`.

    Args:
        images_dir: The `datagen` output directory's `images/` subdirectory.
        label_paths: Pre-filtered label JSON paths for this dataset's
            split, as produced by `partition_label_paths` -- this class
            no longer globs or re-derives splits itself.

    Every label JSON is parsed and encoded into its `LabelBundle` exactly
    once, in `__init__`, and the resulting `(image_path, bundle)` pairs
    are kept in memory. `__getitem__` only decodes the PNG, so per-sample
    JSON parsing (previously done on every access, every epoch, in every
    worker) is eliminated at the cost of RAM. The cache is pickled to
    `DataLoader` workers once per run when `persistent_workers` is on;
    drop the cache and re-parse lazily instead if memory ever becomes a
    concern for a very large label set.
    """

    def __init__(self, images_dir: Path, label_paths: list[Path]) -> None:
        self._samples: list[tuple[Path, LabelBundle]] = []
        for label_path in label_paths:
            record: dict[str, Any] = json.loads(label_path.read_text())
            image_path = images_dir / f"{record['image_id']}.png"
            self._samples.append((image_path, encode_record(record)))

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, LabelBundle]:
        image_path, bundle = self._samples[index]
        with Image.open(image_path) as image:
            if image.size != (IMAGE_WIDTH, IMAGE_HEIGHT):
                raise ValueError(
                    f"expected {IMAGE_WIDTH}x{IMAGE_HEIGHT} images, got "
                    f"{image.size} -- `datagen.augment` downsamples captures "
                    "to that size; run it over this directory first"
                )
            image_tensor = tvf.to_tensor(image.convert("RGB"))

        return image_tensor, bundle


def collate_label_bundles(
    batch: list[tuple[torch.Tensor, LabelBundle]],
) -> tuple[torch.Tensor, LabelBundle]:
    """Stacks a list of `(image, LabelBundle)` samples into one batch.

    `LabelBundle` isn't a type `torch.utils.data.default_collate` knows
    how to batch on its own, so each of its fields is stacked explicitly.
    """
    images = torch.stack([image for image, _ in batch])
    bundles = [bundle for _, bundle in batch]
    batched_fields = {
        field.name: torch.stack([getattr(bundle, field.name) for bundle in bundles])
        for field in dataclasses.fields(LabelBundle)
    }
    return images, LabelBundle(**batched_fields)
