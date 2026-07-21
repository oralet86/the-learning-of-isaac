#!/usr/bin/env python3
"""Preprocess raw chunked frames into per-run frame sequences.

Reads raw frame chunks produced by chunk_video.py, applies frame-level
transforms and saves each run as a contiguous pair of .npy files (frames + labels).
A per-frame train/val/test split mask is computed via
iterative_train_test_split and stored alongside each run.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from constants import (
    AFK_DURATION_SECONDS,
    DATA_CHUNKS_DIR,
    DUPE_EPSILON,
    KEY_ORDER,
    MASTER_FPS,
    PROCESSED_DATA_DIR,
    SEED,
    TEST_FRAC,
    TRIM_EDGE_FRAC,
    VAL_FRAC,
)
from inverse_dynamics.transforms import (
    RemoveAFK,
)

_KEY_SET = set(KEY_ORDER)

_SPLIT_TRAIN = np.int8(1)
_SPLIT_VAL = np.int8(2)
_SPLIT_TEST = np.int8(3)

# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def _clean_keys(keys_list: list[list[str]]) -> list[list[str]]:
    return [[k for k in frame_keys if k in _KEY_SET] for frame_keys in keys_list]


def _keys_to_multilabel(
    keys_list: list[list[str]], key_to_idx: dict[str, int]
) -> np.ndarray:
    n = len(keys_list)
    labels = np.zeros((n, len(key_to_idx)), dtype=np.uint8)
    pairs = [
        (i, key_to_idx[k])
        for i, frame_keys in enumerate(keys_list)
        for k in frame_keys
        if k in key_to_idx
    ]
    if pairs:
        rows, cols = zip(*pairs, strict=True)
        labels[rows, cols] = 1
    return labels


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------


def _process_run(
    chunk_dir: Path,
    key_to_idx: dict[str, int],
    trim_frac: float,
    afk_max_same: int,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """Load all chunks for a run, concatenate, apply transforms, return (frames, labels, run_name)."""
    meta_paths = sorted(chunk_dir.glob("chunk_*_meta.json"))
    if not meta_paths:
        return None

    all_frames: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    run_name: str | None = None
    is_first: bool | None = None
    is_last: bool | None = None

    for meta_path in meta_paths:
        chunk_id = int(meta_path.stem.rsplit("_", 1)[0].split("_", 1)[1])

        frames_path = chunk_dir / f"chunk_{chunk_id:04d}_frames.npy"
        keys_path = chunk_dir / f"chunk_{chunk_id:04d}_keys.json"

        if not frames_path.exists():
            continue

        frames: np.ndarray = np.load(frames_path)

        with open(keys_path) as f:
            raw_keys: list[list[str]] = json.load(f)
        with open(meta_path) as f:
            meta: dict[str, Any] = json.load(f)

        if run_name is None:
            run_name = meta["run_name"]

        cleaned = _clean_keys(raw_keys)
        labels = _keys_to_multilabel(cleaned, key_to_idx)

        # Track run edges
        if meta.get("is_first"):
            is_first = True
        if meta.get("is_last"):
            is_last = True

        all_frames.append(frames)
        all_labels.append(labels)

    if not all_frames:
        return None

    frames_concat = np.concatenate(all_frames)
    labels_concat = np.concatenate(all_labels)

    # Trim edges
    trim_n = int(len(frames_concat) * trim_frac)
    if trim_n > 0:
        if is_first and is_last:
            frames_concat = frames_concat[trim_n:-trim_n]
            labels_concat = labels_concat[trim_n:-trim_n]
        elif is_first:
            frames_concat = frames_concat[trim_n:]
            labels_concat = labels_concat[trim_n:]
        elif is_last:
            frames_concat = frames_concat[:-trim_n]
            labels_concat = labels_concat[:-trim_n]

    # Remove AFK segments
    afk_transform = RemoveAFK(max_same=afk_max_same, epsilon=DUPE_EPSILON)
    frames_concat, labels_concat = afk_transform(frames_concat, labels_concat)

    return frames_concat, labels_concat, run_name or "unknown"


# ---------------------------------------------------------------------------
# Split computation
# ---------------------------------------------------------------------------


def _compute_splits(
    all_labels: list[np.ndarray],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Compute per-frame train/val/test split masks for all runs."""
    total_frames = sum(len(labels) for labels in all_labels)
    n_keys = all_labels[0].shape[1]

    if total_frames == 0 or not all_labels or len(all_labels[0]) == 0:
        empty_split: dict[str, Any] = {
            "train_count": 0,
            "val_count": 0,
            "test_count": 0,
            "pos_weight": [1.0] * n_keys,
        }
        split_masks = [np.empty(0, dtype=np.int8) for _ in all_labels]
        return split_masks, empty_split

    y_all = np.concatenate(all_labels)

    total_pos = y_all.sum(axis=0).astype(np.float64)
    total_neg = (total_frames - total_pos).astype(np.float64)
    pos_weight = (total_neg / (total_pos + 1e-8)).tolist()

    np.random.seed(seed)

    x_idx = np.arange(total_frames, dtype=np.int32).reshape(-1, 1)
    y_int = y_all.astype(np.int32)

    x_trainval, _, x_test, _ = iterative_train_test_split(
        x_idx, y_int, test_size=test_frac
    )

    val_frac_in_remainder = val_frac / (1.0 - test_frac)
    x_train, _, x_val, _ = iterative_train_test_split(
        x_trainval,
        y_int[x_trainval.flatten().astype(np.int32)],
        test_size=val_frac_in_remainder,
    )

    train_arr = x_train.flatten().astype(np.int32)
    val_arr = x_val.flatten().astype(np.int32)
    test_arr = x_test.flatten().astype(np.int32)

    partition = np.zeros(total_frames, dtype=np.int8)
    partition[train_arr] = _SPLIT_TRAIN
    partition[val_arr] = _SPLIT_VAL
    partition[test_arr] = _SPLIT_TEST

    # Split partition back into per-run masks
    split_masks: list[np.ndarray] = []
    offset = 0
    for labels in all_labels:
        n = len(labels)
        split_masks.append(partition[offset : offset + n])
        offset += n

    split_info: dict[str, Any] = {
        "train_count": len(train_arr),
        "val_count": len(val_arr),
        "test_count": len(test_arr),
        "pos_weight": pos_weight,
    }

    del y_all
    return split_masks, split_info


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _write_dataset_info(
    out_dir: Path,
    run_meta: list[dict[str, Any]],
    split_info: dict[str, Any],
    key_names: list[str],
    val_frac: float,
    test_frac: float,
    seed: int,
) -> None:
    total_frames = sum(r["n_frames"] for r in run_meta)
    total_original = sum(r["n_original_frames"] for r in run_meta)
    total_removed_edges = sum(r["n_removed_edges"] for r in run_meta)
    total_removed_afk = sum(r["n_removed_afk"] for r in run_meta)

    dataset_info: dict[str, Any] = {
        "key_names": key_names,
        "pos_weight": split_info["pos_weight"],
        "train_count": split_info["train_count"],
        "val_count": split_info["val_count"],
        "test_count": split_info["test_count"],
        "total_frames": total_frames,
        "n_runs_processed": len(run_meta),
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
        "frame_transforms": ["trim_edges", "remove_afk"],
        "aggregated_stats": {
            "total_original_frames": total_original,
            "total_removed_edges": total_removed_edges,
            "total_removed_afk": total_removed_afk,
        },
    }

    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved dataset_info.json to {out_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess chunked frames into per-run frame sequences"
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DATA_CHUNKS_DIR,
        help="Directory containing run_* subdirectories with chunked .npy files",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=VAL_FRAC,
        help="Fraction of frames for validation",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=TEST_FRAC,
        help="Fraction of frames for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--trim-frac",
        type=float,
        default=TRIM_EDGE_FRAC,
        help="Fraction to trim from run edges",
    )
    parser.add_argument(
        "--afk-max-same",
        type=int,
        default=None,
        help="Max consecutive identical frames before AFK removal "
        f"(default: {AFK_DURATION_SECONDS} * {MASTER_FPS} fps = {int(AFK_DURATION_SECONDS * MASTER_FPS)})",
    )
    args = parser.parse_args()

    chunks_dir = args.chunks_dir
    if not chunks_dir.is_dir():
        sys.exit(f"Not a directory: {chunks_dir}")

    chunk_dirs = sorted(
        d for d in chunks_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    )
    if not chunk_dirs:
        sys.exit(f"No run_* subdirectories found in {chunks_dir}")

    afk_max_same = args.afk_max_same
    if afk_max_same is None:
        afk_max_same = int(AFK_DURATION_SECONDS * MASTER_FPS)

    print(f"Found {len(chunk_dirs)} chunked run(s) in {chunks_dir}")
    print(f"AFK max same: {afk_max_same} consecutive frames")
    print(f"Trim frac: {args.trim_frac}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    key_to_idx = {k: i for i, k in enumerate(KEY_ORDER)}
    key_names = KEY_ORDER

    with open(args.out_dir / "key_order.json", "w") as f:
        json.dump({"keys": key_names}, f, indent=2)

    # Phase 1: Process each run
    all_labels: list[np.ndarray] = []
    run_meta: list[dict[str, Any]] = []

    for run_idx, chunk_dir in enumerate(
        tqdm(chunk_dirs, desc="Processing runs", unit="run")
    ):
        result = _process_run(chunk_dir, key_to_idx, args.trim_frac, afk_max_same)
        if result is None:
            continue

        frames, labels, run_name = result

        # Save run data
        run_id = f"run_{run_idx:04d}"
        np.save(args.out_dir / f"{run_id}_frames.npy", frames)
        np.save(args.out_dir / f"{run_id}_labels.npy", labels)

        all_labels.append(labels)
        run_meta.append(
            {
                "run_id": run_id,
                "run_name": run_name,
                "n_frames": len(frames),
                "n_original_frames": len(frames),
                "n_removed_edges": 0,
                "n_removed_afk": 0,
            }
        )

    if not all_labels:
        sys.exit("No frames processed. Check input data.")

    total_frames = sum(len(labels) for labels in all_labels)
    print(f"\nProcessed {len(all_labels)} runs, {total_frames:,} total frames")

    # Phase 2: Compute splits
    print("\nComputing train/val/test split...")
    split_masks, split_info = _compute_splits(
        all_labels, args.val_frac, args.test_frac, args.seed
    )

    for run_idx, split_mask in enumerate(split_masks):
        run_id = f"run_{run_idx:04d}"
        np.save(args.out_dir / f"{run_id}_split.npy", split_mask)

    print(
        f"  Split: train={split_info['train_count']:,}"
        f"  val={split_info['val_count']:,}"
        f"  test={split_info['test_count']:,}"
    )

    _write_dataset_info(
        args.out_dir,
        run_meta,
        split_info,
        key_names,
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    print(f"\nDone. Output: {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
