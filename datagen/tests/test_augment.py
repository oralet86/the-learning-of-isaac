"""Unit tests for `datagen.augment`."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image
import pytest
import torch

from datagen.augment import (
    AugmentConfig,
    _shift_region,
    apply_compression_artifacts,
    apply_hud_jitter_batch,
    generate_variants_batch,
)
from datagen.hud_regions import get_region_config, get_region_config_for_resolution

if TYPE_CHECKING:
    from pathlib import Path

_DUMMY_CONFIG_ID = 1


def test_shift_region_moves_content_by_exact_pixel_offset() -> None:
    region = torch.zeros(1, 1, 8, 8)
    region[0, 0, 2, 2] = 1.0  # a single marker pixel at (row=2, col=2)

    dx = torch.tensor([3.0])
    dy = torch.tensor([1.0])
    scale = torch.tensor([1.0])
    shifted = _shift_region(region, dx, dy, scale)

    # dx=3 (right), dy=1 (down) -> marker should now sit at (row=3, col=5).
    assert shifted[0, 0, 3, 5].item() == 1.0
    assert shifted.sum().item() == 1.0


def test_apply_hud_jitter_batch_only_touches_named_regions() -> None:
    region_cfg = get_region_config(_DUMMY_CONFIG_ID)
    width, height = region_cfg.resolution
    images = torch.rand(2, 3, height, width) * 255
    generator = torch.Generator().manual_seed(0)

    jittered = apply_hud_jitter_batch(
        images, region_cfg, generator, jitter_max_px=4, jitter_scale_range=(1.0, 1.0)
    )

    mask = torch.ones(height, width, dtype=torch.bool)
    for bbox in region_cfg.regions.values():
        mask[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2] = False

    assert torch.equal(jittered[:, :, mask], images[:, :, mask])


def test_apply_compression_artifacts_changes_bytes_not_shape() -> None:
    rng = np.random.default_rng(0)
    image = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)

    compressed = apply_compression_artifacts(image, quality=70)

    assert compressed.shape == image.shape
    assert compressed.dtype == image.dtype
    assert not np.array_equal(compressed, image)


def _write_pair(
    tmp_path: Path, image_id: str, scenario_id: str, size: tuple[int, int]
) -> tuple[Path, Path]:
    images_dir = tmp_path / "output" / "images"
    labels_dir = tmp_path / "output" / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", size, color=(10, 20, 30))
    image_path = images_dir / f"{image_id}.png"
    image.save(image_path)

    record = {"image_id": image_id, "scenario_id": scenario_id, "seed": 1}
    label_path = labels_dir / f"{image_id}.json"
    label_path.write_text(json.dumps(record))
    return image_path, label_path


def test_generate_variants_batch_writes_expected_files(tmp_path: Path) -> None:
    region_cfg = get_region_config(_DUMMY_CONFIG_ID)
    image_path, label_path = _write_pair(
        tmp_path, "scenario1_0", "scenario1", region_cfg.resolution
    )

    cfg = AugmentConfig(
        input_dir=tmp_path / "output",
        output_dir=tmp_path / "output_augmented",
        variants_per_image=3,
        batch_size=8,
        device="cpu",
        seed=0,
    )
    generator = torch.Generator().manual_seed(0)

    generate_variants_batch([image_path], [label_path], region_cfg, cfg, generator)

    images_out = cfg.output_dir / "images"
    labels_out = cfg.output_dir / "labels"
    written = sorted(p.stem for p in images_out.glob("*.png"))
    assert written == [f"scenario1_0_aug{i}" for i in range(cfg.variants_per_image)]

    for variant in range(cfg.variants_per_image):
        record = json.loads((labels_out / f"scenario1_0_aug{variant}.json").read_text())
        assert record["image_id"] == f"scenario1_0_aug{variant}"
        assert record["scenario_id"] == "scenario1"


def test_get_region_config_for_resolution_rejects_placeholder() -> None:
    # 1920x1080 is the real capture resolution, but its region bboxes are
    # still unfilled TODO placeholders.
    with pytest.raises(ValueError, match="placeholder"):
        get_region_config_for_resolution(1920, 1080)


def test_get_region_config_for_resolution_rejects_unknown() -> None:
    with pytest.raises(KeyError):
        get_region_config_for_resolution(9999, 9999)
