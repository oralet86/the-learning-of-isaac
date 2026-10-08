#!/usr/bin/env python3
"""Preprocess individual raw-frame chunks into train/val/test-ready .npy files.

Each chunk is processed independently: foreign keys are dropped, then a
combined boolean keep-mask of transforms are computed and applied in one copy.  A per-chunk
train/val/test split is computed via iterative_train_test_split on the chunk's
own labels. Memory scales linearly with chunk size.
"""

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

from components.shard_store import shard_path, write_shard
from constants import (
    AFK_DURATION_SECONDS,
    DATA_CHUNKS_DIR,
    DUPE_EPSILON,
    FRAME_CROP_FRAC,
    KEY_ORDER,
    MASTER_FPS,
    MAX_SIMUL_KEYS,
    PROCESSED_DATA_DIR,
    SEED,
    TEST_FRAC,
    TRIM_EDGE_FRAC,
    VAL_FRAC,
)
from inverse_dynamics.chunk_video import chunk_sidecar_path, chunk_stem
from inverse_dynamics.transforms import (
    afk_mask,
    center_crop_frames,
    clean_keys,
    conflicting_mask,
    edge_trim_mask,
    empty_frames_mask,
    foreign_keys_mask,
    too_many_keys_mask,
)

_KEY_SET = set(KEY_ORDER)

_CONFLICTING_PAIRS: list[tuple[int, int]] = [
    (KEY_ORDER.index("w"), KEY_ORDER.index("s")),
    (KEY_ORDER.index("a"), KEY_ORDER.index("d")),
    (KEY_ORDER.index("up"), KEY_ORDER.index("down")),
    (KEY_ORDER.index("left"), KEY_ORDER.index("right")),
]

_TRANSFORMS: list[Any] = [
    foreign_keys_mask,
    edge_trim_mask,
    conflicting_mask,
    too_many_keys_mask,
    afk_mask,
    empty_frames_mask,
]

_SPLIT_TRAIN = np.int8(1)
_SPLIT_VAL = np.int8(2)
_SPLIT_TEST = np.int8(3)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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


def _compute_chunk_split(
    labels: np.ndarray,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute per-chunk train/val/test split mask.

    Falls back to all-train when the chunk is too small for meaningful splits.
    """
    n = len(labels)
    if n < 3:
        split = np.full(n, _SPLIT_TRAIN, dtype=np.int8)
        counts = {"train": n, "val": 0, "test": 0}
        return split, counts

    np.random.seed(seed)
    x_idx = np.arange(n, dtype=np.int32).reshape(-1, 1)
    y_int = labels.astype(np.int32)

    test_size = max(2 / n, min(test_frac, (n - 2) / n))

    x_trainval, _, x_test, _ = iterative_train_test_split(
        x_idx, y_int, test_size=test_size
    )

    n_trainval = len(x_trainval)
    if n_trainval < 2:
        train_arr = x_trainval.flatten().astype(np.int32)
        val_arr = np.array([], dtype=np.int32)
    else:
        remainder_frac = 1.0 - test_size
        val_size = max(
            1 / n_trainval,
            min(val_frac / remainder_frac, (n_trainval - 1) / n_trainval),
        )
        x_train, _, x_val, _ = iterative_train_test_split(
            x_trainval,
            y_int[x_trainval.flatten().astype(np.int32)],
            test_size=val_size,
        )
        train_arr = x_train.flatten().astype(np.int32)
        val_arr = x_val.flatten().astype(np.int32)

    test_arr = x_test.flatten().astype(np.int32)

    split = np.full(n, _SPLIT_TRAIN, dtype=np.int8)
    split[val_arr] = _SPLIT_VAL
    split[test_arr] = _SPLIT_TEST

    counts = {"train": len(train_arr), "val": len(val_arr), "test": len(test_arr)}
    return split, counts


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def _write_dataset_info(
    out_dir: Path,
    chunk_meta: list[dict[str, Any]],
    key_names: list[str],
    pos_weight: list[float],
    val_frac: float,
    test_frac: float,
    seed: int,
    transform_fns: list[Any],
) -> None:
    total_frames = sum(c["n_frames"] for c in chunk_meta)
    total_original = sum(c["n_original"] for c in chunk_meta)
    total_train = sum(c["n_train"] for c in chunk_meta)
    total_val = sum(c["n_val"] for c in chunk_meta)
    total_test = sum(c["n_test"] for c in chunk_meta)
    total_removed_foreign = sum(c["n_removed_foreign"] for c in chunk_meta)
    total_removed_edges = sum(c["n_removed_edges"] for c in chunk_meta)
    total_removed_conflicting = sum(c["n_removed_conflicting"] for c in chunk_meta)
    total_removed_too_many = sum(c["n_removed_too_many_keys"] for c in chunk_meta)
    total_removed_afk = sum(c["n_removed_afk"] for c in chunk_meta)
    total_removed_empty = sum(c["n_removed_empty"] for c in chunk_meta)

    dataset_info: dict[str, Any] = {
        "key_names": key_names,
        "pos_weight": pos_weight,
        "train_count": total_train,
        "val_count": total_val,
        "test_count": total_test,
        "total_frames": total_frames,
        "n_chunks_processed": len(chunk_meta),
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
        "crop_frac": FRAME_CROP_FRAC,
        "frame_transforms": [fn.__name__ for fn in transform_fns],
        "aggregated_stats": {
            "total_original_frames": total_original,
            "total_removed_foreign_keys": total_removed_foreign,
            "total_removed_edges": total_removed_edges,
            "total_removed_conflicting": total_removed_conflicting,
            "total_removed_too_many_keys": total_removed_too_many,
            "total_removed_afk": total_removed_afk,
            "total_removed_empty_frames": total_removed_empty,
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
        description="Preprocess chunked frames into per-chunk .npy files"
    )
    parser.add_argument(
        "--chunks-dir",
        type=Path,
        default=DATA_CHUNKS_DIR,
        help="Root directory containing per-group subdirectories with chunked .npy files",
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
        help="Fraction to trim from chunk edges",
    )
    parser.add_argument(
        "--afk-max-same",
        type=int,
        default=None,
        help="Max consecutive identical frames before AFK removal "
        f"(default: {AFK_DURATION_SECONDS} * {MASTER_FPS} fps = "
        f"{int(AFK_DURATION_SECONDS * MASTER_FPS)})",
    )
    args = parser.parse_args()

    chunks_dir = args.chunks_dir
    if not chunks_dir.is_dir():
        sys.exit(f"Not a directory: {chunks_dir}")

    meta_paths = sorted(chunks_dir.rglob("chunk_*_meta.json"))
    if not meta_paths:
        sys.exit(f"No chunk_*_meta.json files found under {chunks_dir}")

    afk_max_same = args.afk_max_same
    if afk_max_same is None:
        afk_max_same = int(AFK_DURATION_SECONDS * MASTER_FPS)

    print(f"Found {len(meta_paths)} chunk(s) under {chunks_dir}")
    print(f"AFK max same: {afk_max_same} consecutive frames")
    print(f"Trim frac: {args.trim_frac}")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    key_to_idx = {k: i for i, k in enumerate(KEY_ORDER)}
    key_names = KEY_ORDER

    with open(args.out_dir / "key_order.json", "w") as f:
        json.dump({"keys": key_names}, f, indent=2)

    chunk_meta: list[dict[str, Any]] = []
    total_frames_final = 0
    total_pos = np.zeros(len(key_names), dtype=np.float64)

    for chunk_idx, meta_path in enumerate(
        tqdm(meta_paths, desc="Processing chunks", unit="chunk")
    ):
        chunk_dir = meta_path.parent
        group_name = chunk_dir.name

        chunk_id = int(meta_path.stem.rsplit("_", 1)[0].split("_", 1)[1])
        frames_path = shard_path(chunk_dir, chunk_stem(chunk_id), "frames")
        keys_path = chunk_sidecar_path(chunk_dir, chunk_id, "keys")

        if not frames_path.exists():
            continue

        frames: np.ndarray = np.load(frames_path)

        with open(keys_path) as f:
            raw_keys: list[list[str]] = json.load(f)
        with open(meta_path) as f:
            meta: dict[str, Any] = json.load(f)

        n_original = len(frames)
        total_run_frames = meta.get("total_run_frames", n_original)

        # --- Build keep-masks (all True = keep) ---
        foreign = foreign_keys_mask(raw_keys, _KEY_SET)

        trim = edge_trim_mask(
            n_original,
            total_run_frames,
            meta.get("is_first", False),
            meta.get("is_last", False),
            args.trim_frac,
        )

        # --- Build labels from cleaned keys ---
        cleaned = clean_keys(raw_keys, _KEY_SET)
        labels = _keys_to_multilabel(cleaned, key_to_idx)

        conflict = conflicting_mask(labels, _CONFLICTING_PAIRS)
        too_many = too_many_keys_mask(labels, MAX_SIMUL_KEYS)
        afk = afk_mask(frames, afk_max_same, DUPE_EPSILON)
        empty = empty_frames_mask(labels)

        # --- Combine masks sequentially for marginal removal counts ---
        running = np.ones(n_original, dtype=bool)

        n_removed_foreign = int((running & ~foreign).sum())
        running &= foreign

        n_removed_edges = int((running & ~trim).sum())
        running &= trim

        n_removed_conflicting = int((running & ~conflict).sum())
        running &= conflict

        n_removed_too_many = int((running & ~too_many).sum())
        running &= too_many

        n_removed_afk = int((running & ~afk).sum())
        running &= afk

        n_removed_empty = int((running & ~empty).sum())
        running &= empty

        frames = frames[running]
        labels = labels[running]
        frames = center_crop_frames(frames, FRAME_CROP_FRAC)
        n_final = len(frames)

        # --- Per-chunk split ---
        chunk_seed = args.seed + chunk_idx
        split, split_counts = _compute_chunk_split(
            labels, args.val_frac, args.test_frac, chunk_seed
        )

        # --- Save ---
        out_stem = f"{group_name}_{chunk_stem(chunk_id)}"
        write_shard(
            args.out_dir,
            out_stem,
            {"frames": frames, "labels": labels, "split": split},
        )

        total_frames_final += n_final
        total_pos += labels.sum(axis=0).astype(np.float64)

        chunk_meta.append(
            {
                "stem": out_stem,
                "group_name": group_name,
                "chunk_id": chunk_id,
                "n_original": n_original,
                "n_frames": n_final,
                "n_train": split_counts["train"],
                "n_val": split_counts["val"],
                "n_test": split_counts["test"],
                "n_removed_foreign": n_removed_foreign,
                "n_removed_edges": n_removed_edges,
                "n_removed_conflicting": n_removed_conflicting,
                "n_removed_too_many_keys": n_removed_too_many,
                "n_removed_afk": n_removed_afk,
                "n_removed_empty": n_removed_empty,
            }
        )

    if not chunk_meta:
        sys.exit("No chunks processed. Check input data.")

    total_train = sum(c["n_train"] for c in chunk_meta)
    total_val = sum(c["n_val"] for c in chunk_meta)
    total_test = sum(c["n_test"] for c in chunk_meta)

    total_neg = total_frames_final - total_pos
    pos_weight = (total_neg / (total_pos + 1e-8)).tolist()

    print(f"\nProcessed {len(chunk_meta)} chunks, {total_frames_final:,} total frames")
    print(f"  Train: {total_train:,}  Val: {total_val:,}  Test: {total_test:,}")

    _write_dataset_info(
        args.out_dir,
        chunk_meta,
        key_names,
        pos_weight,
        args.val_frac,
        args.test_frac,
        args.seed,
        _TRANSFORMS,
    )

    print(f"\nDone. Output: {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
