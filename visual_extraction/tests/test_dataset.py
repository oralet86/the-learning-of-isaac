"""Tests for `visual_extraction.dataset`'s partition-once/cache behavior.

Builds tiny synthetic label dirs (matching the JSON contract documented
in `labels.py`) plus 480x270 PNGs, and asserts the split partition is
computed once (disjoint, deterministic, scenario-stable) and that
`GroundTruthDataset.__getitem__` serves samples from its in-memory cache
without touching the label JSONs after construction.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from PIL import Image
import torch

from visual_extraction.dataset import GroundTruthDataset, partition_label_paths

if TYPE_CHECKING:
    from pathlib import Path

_SPLITS = ("train", "val", "test")


def _record(scenario_id: str, image_id: str) -> dict[str, Any]:
    """Minimal ground-truth record matching the contract `encode_record` reads."""
    return {
        "scenario_id": scenario_id,
        "image_id": image_id,
        "stage_bucket": 3,
        "player": {"frame_position": [0.25, 0.75]},
        "player2": None,
        "active_item": {"held": False, "charge": None},
        "currency": {"coins": 5, "bombs": 2, "keys": 1},
        "consumable": {
            "kind": 0,
            "card_id": None,
            "pill_color": None,
            "pill_known": None,
        },
        "stats": {
            "damage": 3.5,
            "fire_rate": 4.2,
            "speed": 1.0,
            "range": 6.5,
            "shot_speed": 1.0,
            "luck": 0.0,
        },
        "health": {
            "unknown": False,
            "hearts": 4,
            "max_hearts": 6,
            "soul_hearts": 0,
            "black_hearts": 0,
            "bone_hearts": 0,
            "rotten_hearts": 0,
            "eternal_hearts": 0,
            "golden_hearts": 0,
            "broken_hearts": 0,
        },
        "enemies": [],
    }


def _make_dir(tmp_path: Path, records: list[dict[str, Any]]) -> tuple[Path, Path]:
    """Writes `records` as labels/{image_id}.json + a 480x270 PNG each.

    Returns `(images_dir, labels_dir)` as `tmp_path` subdirectories.
    """
    labels_dir = tmp_path / "labels"
    images_dir = tmp_path / "images"
    labels_dir.mkdir(parents=True)
    images_dir.mkdir(parents=True)
    for record in records:
        (labels_dir / f"{record['image_id']}.json").write_text(json.dumps(record))
        Image.new("RGB", (480, 270), color=(0, 0, 0)).save(
            images_dir / f"{record['image_id']}.png"
        )
    return images_dir, labels_dir


def test_partition_is_disjoint_and_total(tmp_path: Path) -> None:
    _images_dir, labels_dir = _make_dir(
        tmp_path,
        [_record(f"scenario_{i}", f"img_{i}") for i in range(50)],
    )

    partition = partition_label_paths(labels_dir)

    assert list(partition) == list(_SPLITS)
    all_paths = [path for paths in partition.values() for path in paths]
    assert len(all_paths) == 50
    assert len(set(all_paths)) == 50
    assert sum(len(paths) for paths in partition.values()) == 50


def test_partition_is_deterministic_and_scenario_stable(tmp_path: Path) -> None:
    _images_dir, labels_dir = _make_dir(
        tmp_path,
        [_record(f"scenario_{i}", f"img_{i}") for i in range(50)],
    )

    first = partition_label_paths(labels_dir)
    second = partition_label_paths(labels_dir)
    assert first == second

    # Two augmented variants of the same scenario must land in the same split.
    _images_dir, labels_dir = _make_dir(
        tmp_path / "augmented",
        [
            _record("scenario_7", "img_7_aug1"),
            _record("scenario_7", "img_7_aug2"),
        ],
    )
    partition = partition_label_paths(labels_dir)
    split_of = {
        path.name: split for split, paths in partition.items() for path in paths
    }
    assert split_of["img_7_aug1.json"] == split_of["img_7_aug2.json"]


def test_dataset_serves_from_cache_after_label_files_are_deleted(
    tmp_path: Path,
) -> None:
    images_dir, labels_dir = _make_dir(
        tmp_path,
        [_record(f"scenario_{i}", f"img_{i}") for i in range(5)],
    )
    partition = partition_label_paths(labels_dir)
    all_paths = [path for paths in partition.values() for path in paths]

    ds = GroundTruthDataset(images_dir, all_paths)
    assert len(ds) == 5

    # Construction must have parsed every JSON; deleting the labels proves
    # __getitem__ reads nothing from disk except the PNG.
    for path in labels_dir.glob("*.json"):
        path.unlink()

    image, bundle = ds[2]
    assert image.shape == (3, 270, 480)
    assert image.dtype == torch.float32
    assert bundle.stage.item() == 3
