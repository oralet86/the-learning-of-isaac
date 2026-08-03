"""`torch.utils.data.Dataset` over `datagen/output/{images,labels}`.

Each label JSON's `image_id` names both files: `labels/{image_id}.json`
and `images/{image_id}.png`. Split is a deterministic hash of
`scenario_id` (not `image_id`), so every viewport crop from the same room
visit lands in the same split -- otherwise near-duplicate crops of one
scenario could leak across train/val.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import TYPE_CHECKING, Literal

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as tvf

from visual_extraction.labels import LabelBundle, encode_record

if TYPE_CHECKING:
    from pathlib import Path

"""
The game natively renders at 480x270 before any upscaling,
resizing a captured frame below that loses real information (the HUD becomes unreadable),
so this matches native resolution rather than the common "square 224" CNN-input convention.
"""
IMAGE_WIDTH = 480
IMAGE_HEIGHT = 270
_VAL_FRACTION = 0.1


def _is_val_split(scenario_id: str, val_fraction: float) -> bool:
    digest = hashlib.sha256(scenario_id.encode()).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return bucket < val_fraction


def _find_records(labels_dir: Path) -> list[Path]:
    return sorted(labels_dir.glob("*.json"))


class GroundTruthDataset(Dataset[tuple[torch.Tensor, LabelBundle]]):
    """One captured frame per sample: `(image_tensor, LabelBundle)`.

    Args:
        output_dir: A `datagen` run's output directory, containing
            `images/` and `labels/` subdirectories.
        split: `"train"` or `"val"`.
        val_fraction: Fraction of scenarios (not frames) held out for
            validation, split deterministically by `scenario_id`.
        image_width: Width images are resized to.
        image_height: Height images are resized to.
    """

    def __init__(
        self,
        output_dir: Path,
        split: Literal["train", "val"],
        val_fraction: float = _VAL_FRACTION,
        image_width: int = IMAGE_WIDTH,
        image_height: int = IMAGE_HEIGHT,
    ) -> None:
        self.images_dir = output_dir / "images"
        self.image_width = image_width
        self.image_height = image_height

        all_label_paths = _find_records(output_dir / "labels")
        self.label_paths = [
            path
            for path in all_label_paths
            if _is_val_split(json.loads(path.read_text())["scenario_id"], val_fraction)
            == (split == "val")
        ]

    def __len__(self) -> int:
        return len(self.label_paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, LabelBundle]:
        label_path = self.label_paths[index]
        record = json.loads(label_path.read_text())

        image_path = self.images_dir / f"{record['image_id']}.png"
        image = Image.open(image_path).convert("RGB")
        image = image.resize((self.image_width, self.image_height))
        image_tensor = tvf.to_tensor(image)

        return image_tensor, encode_record(record)


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
