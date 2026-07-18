#!/usr/bin/env python3
"""Preprocess recorded Isaac runs: decode video, build windows, chunk, split.

Reads master.mp4 + keys.json from each run directory, derives grayscale
training-rate frames via torchcodec (resize, subsample), applies frame-level
and window-level transforms, builds sliding windows, and writes train/val/test
chunk files.
"""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
import torch
from torchcodec import transforms as tc_transforms
from torchcodec.decoders import VideoDecoder
from tqdm import tqdm

from constants import (
    AFK_DURATION_SECONDS,
    CHUNK_SIZE,
    DUPE_EPSILON,
    INPUT_CHANNELS,
    KEY_ORDER,
    MASTER_FPS,
    PROCESSED_DATA_DIR,
    RUNS_DIR,
    SEED,
    STATIC_WINDOW_TOLERANCE,
    SUBSAMPLE_STEP,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TEST_FRAC,
    TRIM_EDGE_FRAC,
    VAL_FRAC,
)
from inverse_dynamics.transforms import (
    FrameTransform,
    RemoveAFK,
    StaticWindowFilter,
    TrimEdges,
    WindowTransform,
)

KEY_SET = set(KEY_ORDER)


# ---------------------------------------------------------------------------
# Key helpers
# ---------------------------------------------------------------------------


def clean_keys(keys_list: list[list[str]]) -> list[list[str]]:
    """Keep only keys present in the configured KEY_ORDER."""
    return [[k for k in frame_keys if k in KEY_SET] for frame_keys in keys_list]


def keys_to_multilabel(
    keys_list: list[list[str]], key_to_idx: dict[str, int]
) -> np.ndarray:
    """Convert per-frame key lists to an (N, K) binary label matrix."""
    n = len(keys_list)
    labels = np.zeros((n, len(key_to_idx)), dtype=np.uint8)
    for i, frame_keys in enumerate(keys_list):
        for k in frame_keys:
            idx = key_to_idx.get(k)
            if idx is not None:
                labels[i, idx] = 1
    return labels


# ---------------------------------------------------------------------------
# Video I/O
# ---------------------------------------------------------------------------


def read_video(
    video_path: Path, target_height: int, target_width: int
) -> tuple[np.ndarray, float]:
    """Read an mp4, resize via decode filter, convert to grayscale.

    Returns (T, H, W) uint8 array and the video's average fps.
    """
    decoder = VideoDecoder(
        str(video_path),
        transforms=[tc_transforms.Resize((target_height, target_width))],
    )
    fps_raw = decoder.metadata.average_fps
    if fps_raw is None:
        raise RuntimeError(f"No video stream in {video_path}")
    fps = float(fps_raw)
    frames = decoder[:]

    if frames.numel() == 0:
        raise RuntimeError(f"No frames read from {video_path}")

    frames = frames.float().mean(dim=1, keepdim=True)
    frames = frames.round().clamp(0, 255).to(torch.uint8)
    return frames[:, 0].numpy(), fps


# ---------------------------------------------------------------------------
# Windowing
# ---------------------------------------------------------------------------


def build_windows(
    frames: np.ndarray, labels: np.ndarray, n_frames: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build sliding windows of *n_frames* consecutive frames.

    Returns:
        windows       (W, n_frames, H, W) uint8
        window_labels (W, K) uint8
        window_map    (W, 2) int32  — col 0 = run_idx, col 1 = window_idx
    """
    n_windows = len(frames) - n_frames + 1
    if n_windows <= 0:
        return (
            np.empty((0, n_frames, *frames.shape[1:]), dtype=np.uint8),
            np.empty((0, labels.shape[1]), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int32),
        )

    view = np.lib.stride_tricks.sliding_window_view(frames, n_frames, axis=0)
    windows = np.copy(view.transpose(0, 3, 1, 2))

    window_labels = labels[:n_windows].copy()
    window_map = np.empty((n_windows, 2), dtype=np.int32)
    window_map[:, 1] = np.arange(n_windows, dtype=np.int32)

    return windows, window_labels, window_map


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------


def process_run(
    run_dir: Path,
    key_to_idx: dict[str, int],
    frame_transforms: list[FrameTransform] | None,
    window_transforms: list[WindowTransform] | None,
    subsample_step: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    """Load, clean, and window a single run.

    Returns None when the run produces zero windows after filtering.
    """
    name = run_dir.name
    keys_path = run_dir / "keys.json"
    video_path = run_dir / "master.mp4"

    if not keys_path.exists():
        print(f"  Skipping {name}: missing keys.json")
        return None
    if not video_path.exists():
        print(f"  Skipping {name}: missing master.mp4")
        return None

    with open(keys_path) as f:
        raw_keys: list[list[str]] = json.load(f)

    frames, video_fps = read_video(video_path, TARGET_HEIGHT, TARGET_WIDTH)

    if len(frames) != len(raw_keys):
        print(
            f"  WARNING {name}: frame count ({len(frames)}) != key count "
            f"({len(raw_keys)}). Using min."
        )
        n = min(len(frames), len(raw_keys))
        frames = frames[:n]
        raw_keys = raw_keys[:n]

    n_original = len(frames)

    cleaned = clean_keys(raw_keys)
    labels = keys_to_multilabel(cleaned, key_to_idx)

    if subsample_step > 1:
        frames = frames[::subsample_step]
        labels = labels[::subsample_step]
        effective_fps = video_fps / subsample_step
    else:
        effective_fps = video_fps

    n_removed_edges = 0
    n_removed_afk = 0

    if frame_transforms:
        for t in frame_transforms:
            n_before = len(frames)
            frames, labels = t(frames, labels)
            delta = n_before - len(frames)
            if isinstance(t, TrimEdges):
                n_removed_edges += delta
            elif isinstance(t, RemoveAFK):
                n_removed_afk += delta

    n_frames = len(frames)

    windows, window_labels, window_map = build_windows(frames, labels, INPUT_CHANNELS)

    if len(windows) == 0:
        return None

    n_windows_built = len(windows)
    n_removed_static = 0

    if window_transforms:
        for t in window_transforms:
            mask = t(windows)
            if mask is not None:
                n_before = len(windows)
                windows = windows[mask]
                window_labels = window_labels[mask]
                window_map = window_map[mask]
                delta = n_before - len(windows)
                if isinstance(t, StaticWindowFilter):
                    n_removed_static += delta

    if len(windows) == 0:
        return None

    kc = Counter(k for frame_keys in cleaned for k in frame_keys)
    combo_counts = Counter(tuple(sorted(k)) for k in cleaned)

    stats: dict[str, Any] = {
        "run_name": name,
        "n_original_frames": n_original,
        "n_frames": n_frames,
        "n_removed_edges": n_removed_edges,
        "n_removed_afk": n_removed_afk,
        "n_windows_built": n_windows_built,
        "n_removed_static": n_removed_static,
        "n_windows": len(windows),
        "key_frequencies": {k: int(v) for k, v in kc.most_common()},
        "empty_frames": sum(1 for k in cleaned if len(k) == 0),
        "unique_combinations": len(combo_counts),
        "video_fps": video_fps,
        "effective_fps": effective_fps,
    }

    return windows, window_labels, window_map, stats


# ---------------------------------------------------------------------------
# Chunked streaming
# ---------------------------------------------------------------------------


def stream_to_chunks(
    runs: list[Path],
    key_to_idx: dict[str, int],
    out_dir: Path,
    chunk_size: int,
    frame_transforms: list[FrameTransform] | None,
    window_transforms: list[WindowTransform] | None,
    subsample_step: int,
) -> dict[str, Any]:
    """Process every run and stream windows into fixed-size chunk files."""
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    n_keys = len(key_to_idx)
    buf_x = np.empty(
        (chunk_size, INPUT_CHANNELS, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8
    )
    buf_y = np.empty((chunk_size, n_keys), dtype=np.uint8)
    buf_map = np.empty((chunk_size, 2), dtype=np.int32)
    buf_pos = 0
    chunk_id = 0

    all_labels: list[np.ndarray] = []
    all_stats: list[dict[str, Any]] = []
    run_idx_counter = 0

    for run_dir in tqdm(runs, desc="Processing runs", unit="run"):
        result = process_run(
            run_dir,
            key_to_idx,
            frame_transforms,
            window_transforms,
            subsample_step,
        )
        if result is None:
            continue

        windows, window_labels, window_map, stats = result
        stats["run_idx"] = run_idx_counter
        window_map[:, 0] = run_idx_counter
        run_idx_counter += 1

        all_labels.append(window_labels)
        all_stats.append(stats)

        n_windows = len(windows)
        src_pos = 0

        while src_pos < n_windows:
            space = chunk_size - buf_pos
            take = min(space, n_windows - src_pos)

            buf_x[buf_pos : buf_pos + take] = windows[src_pos : src_pos + take]
            buf_y[buf_pos : buf_pos + take] = window_labels[src_pos : src_pos + take]
            buf_map[buf_pos : buf_pos + take] = window_map[src_pos : src_pos + take]
            buf_pos += take
            src_pos += take

            if buf_pos == chunk_size:
                np.save(chunks_dir / f"chunk_{chunk_id:04d}_X.npy", buf_x)
                np.save(chunks_dir / f"chunk_{chunk_id:04d}_y.npy", buf_y)
                np.save(chunks_dir / f"chunk_{chunk_id:04d}_map.npy", buf_map)
                chunk_id += 1
                buf_pos = 0

        del windows, window_labels, window_map

    if buf_pos > 0:
        np.save(chunks_dir / f"chunk_{chunk_id:04d}_X.npy", buf_x[:buf_pos])
        np.save(chunks_dir / f"chunk_{chunk_id:04d}_y.npy", buf_y[:buf_pos])
        np.save(chunks_dir / f"chunk_{chunk_id:04d}_map.npy", buf_map[:buf_pos])
        chunk_id += 1

    y_all = (
        np.concatenate(all_labels)
        if all_labels
        else np.empty((0, n_keys), dtype=np.uint8)
    )
    np.save(chunks_dir / "all_y.npy", y_all)

    total_windows = len(y_all)
    print(f"\nWrote {chunk_id} chunk(s), {total_windows:,} total windows")

    return {
        "n_chunks": chunk_id,
        "total_windows": total_windows,
        "run_stats": all_stats,
    }


# ---------------------------------------------------------------------------
# Train / val / test splitting
# ---------------------------------------------------------------------------


def split_chunks(
    out_dir: Path,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict[str, Any]:
    """Multilabel-aware split of chunked windows into train/val/test."""
    chunks_dir = out_dir / "chunks"
    y_all = np.load(chunks_dir / "all_y.npy")
    total_windows = len(y_all)
    n_keys = y_all.shape[1]

    if total_windows == 0:
        print("No windows to split.")
        return {"train_count": 0, "val_count": 0, "test_count": 0, "pos_weight": []}

    total_pos = y_all.sum(axis=0).astype(np.float64)
    total_neg = (total_windows - total_pos).astype(np.float64)
    pos_weight = total_neg / (total_pos + 1e-8)

    np.random.seed(seed)

    x_idx = np.arange(total_windows, dtype=np.int32).reshape(-1, 1)
    y_int = y_all.astype(np.int32)

    x_trainval, y_trainval, x_test, _ = iterative_train_test_split(
        x_idx, y_int, test_size=test_frac
    )

    val_frac_in_remainder = val_frac / (1.0 - test_frac)
    x_train, _, x_val, _ = iterative_train_test_split(
        x_trainval, y_trainval, test_size=val_frac_in_remainder
    )

    del y_all, x_idx, y_int, x_trainval, y_trainval

    train_arr = x_train.flatten().astype(np.int32)
    val_arr = x_val.flatten().astype(np.int32)
    test_arr = x_test.flatten().astype(np.int32)
    del x_train, x_val, x_test

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    chunk_x_paths = sorted(chunks_dir.glob("chunk_*_X.npy"))
    chunk_y_paths = sorted(chunks_dir.glob("chunk_*_y.npy"))
    chunk_map_paths = sorted(chunks_dir.glob("chunk_*_map.npy"))

    def make_buf(split_dir: Path, with_map: bool) -> dict[str, Any]:
        return {
            "dir": split_dir,
            "X": np.empty(
                (CHUNK_SIZE, INPUT_CHANNELS, TARGET_HEIGHT, TARGET_WIDTH),
                dtype=np.uint8,
            ),
            "y": np.empty((CHUNK_SIZE, n_keys), dtype=np.uint8),
            "map": np.empty((CHUNK_SIZE, 2), dtype=np.int32) if with_map else None,
            "pos": 0,
            "chunk_id": 0,
            "count": 0,
        }

    def flush(buf: dict[str, Any]) -> None:
        if buf["pos"] == 0:
            return
        cid = buf["chunk_id"]
        np.save(buf["dir"] / f"chunk_{cid:04d}_X.npy", buf["X"][: buf["pos"]])
        np.save(buf["dir"] / f"chunk_{cid:04d}_y.npy", buf["y"][: buf["pos"]])
        if buf["map"] is not None:
            np.save(
                buf["dir"] / f"chunk_{cid:04d}_sample_map.npy",
                buf["map"][: buf["pos"]],
            )
        buf["chunk_id"] += 1
        buf["pos"] = 0

    def push(
        buf: dict[str, Any], x: np.ndarray, y: np.ndarray, map_: np.ndarray | None
    ) -> None:
        n = len(x)
        buf["count"] += n
        src = 0
        while src < n:
            space = CHUNK_SIZE - buf["pos"]
            take = min(space, n - src)
            buf["X"][buf["pos"] : buf["pos"] + take] = x[src : src + take]
            buf["y"][buf["pos"] : buf["pos"] + take] = y[src : src + take]
            if map_ is not None and buf["map"] is not None:
                buf["map"][buf["pos"] : buf["pos"] + take] = map_[src : src + take]
            buf["pos"] += take
            src += take
            if buf["pos"] == CHUNK_SIZE:
                flush(buf)

    train_buf = make_buf(train_dir, with_map=False)
    val_buf = make_buf(val_dir, with_map=False)
    test_buf = make_buf(test_dir, with_map=True)

    global_offset = 0

    for x_path, y_path, map_path in tqdm(
        zip(chunk_x_paths, chunk_y_paths, chunk_map_paths, strict=True),
        total=len(chunk_x_paths),
        desc="Splitting chunks",
        unit="chunk",
    ):
        x_chunk = np.load(x_path)
        y_chunk = np.load(y_path)
        map_chunk = np.load(map_path)
        chunk_n = len(x_chunk)

        indices = np.arange(global_offset, global_offset + chunk_n, dtype=np.int32)

        train_mask = np.isin(indices, train_arr, assume_unique=True)
        val_mask = np.isin(indices, val_arr, assume_unique=True)
        test_mask = np.isin(indices, test_arr, assume_unique=True)

        if train_mask.any():
            push(train_buf, x_chunk[train_mask], y_chunk[train_mask], None)
        if val_mask.any():
            push(val_buf, x_chunk[val_mask], y_chunk[val_mask], None)
        if test_mask.any():
            push(
                test_buf,
                x_chunk[test_mask],
                y_chunk[test_mask],
                map_chunk[test_mask],
            )

        del x_chunk, y_chunk, map_chunk
        global_offset += chunk_n

    flush(train_buf)
    flush(val_buf)
    flush(test_buf)

    train_count = train_buf["count"]
    val_count = val_buf["count"]
    test_count = test_buf["count"]

    del train_arr, val_arr, test_arr, train_buf, val_buf, test_buf

    shutil.rmtree(chunks_dir)

    print(f"  Split: train={train_count:,}  val={val_count:,}  test={test_count:,}")

    return {
        "train_count": train_count,
        "val_count": val_count,
        "test_count": test_count,
        "pos_weight": pos_weight.tolist(),
    }


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def prepare_dataset_info(
    out_dir: Path,
    stream_stats: dict[str, Any],
    split_info: dict[str, Any],
    frame_transforms: list[FrameTransform] | None,
    window_transforms: list[WindowTransform] | None,
    val_frac: float,
    test_frac: float,
    seed: int,
    chunk_size: int,
    subsample_step: int,
) -> None:
    """Write dataset_info.json summarizing the preprocessing run."""
    key_order_path = out_dir / "key_order.json"
    with open(key_order_path) as f:
        key_info = json.load(f)

    total_key_freqs: Counter[str] = Counter()
    total_original = 0
    total_edges = 0
    total_afk = 0
    total_static = 0
    total_empty = 0

    for rs in stream_stats.get("run_stats", []):
        for k, v in rs.get("key_frequencies", {}).items():
            total_key_freqs[k] += v
        total_original += rs.get("n_original_frames", 0)
        total_edges += rs.get("n_removed_edges", 0)
        total_afk += rs.get("n_removed_afk", 0)
        total_static += rs.get("n_removed_static", 0)
        total_empty += rs.get("empty_frames", 0)

    frame_transform_names = [t.name for t in (frame_transforms or [])]
    window_transform_names = [t.name for t in (window_transforms or [])]

    dataset_info: dict[str, Any] = {
        "key_names": key_info["keys"],
        "pos_weight": split_info["pos_weight"],
        "train_count": split_info["train_count"],
        "val_count": split_info["val_count"],
        "test_count": split_info["test_count"],
        "total_windows": stream_stats["total_windows"],
        "n_runs_processed": len(stream_stats.get("run_stats", [])),
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
        "chunk_size": chunk_size,
        "subsample_step": subsample_step,
        "frame_transforms": frame_transform_names,
        "window_transforms": window_transform_names,
        "aggregated_stats": {
            "total_original_frames": total_original,
            "total_removed_edges": total_edges,
            "total_removed_afk": total_afk,
            "total_removed_static": total_static,
            "total_empty_frames": total_empty,
            "key_frequencies": dict(total_key_freqs.most_common()),
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
        description="Preprocess recorded Isaac runs for training"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=RUNS_DIR,
        help="Directory containing run_* subdirectories",
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
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=TEST_FRAC,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=CHUNK_SIZE,
        help="Number of windows per chunk file",
    )
    parser.add_argument(
        "--subsample-step",
        type=int,
        default=SUBSAMPLE_STEP,
        help="Take every Nth frame from the master video (1 = no subsampling)",
    )
    args = parser.parse_args()

    runs_dir = args.runs_dir
    if not runs_dir.is_dir():
        sys.exit(f"Not a directory: {runs_dir}")

    runs = sorted(
        d for d in runs_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    )
    if not runs:
        sys.exit(f"No run_* subdirectories found in {runs_dir}")

    print(f"Found {len(runs)} run(s) in {runs_dir}")
    print(f"Key order ({len(KEY_ORDER)} keys): {KEY_ORDER}")
    print("All other keys will be dropped")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    key_to_idx = {k: i for i, k in enumerate(KEY_ORDER)}
    with open(args.out_dir / "key_order.json", "w") as f:
        json.dump({"keys": KEY_ORDER}, f, indent=2)

    afk_max_same = int(AFK_DURATION_SECONDS * MASTER_FPS / args.subsample_step)

    frame_transforms: list[FrameTransform] = [
        TrimEdges(frac=TRIM_EDGE_FRAC),
        RemoveAFK(max_same=afk_max_same, epsilon=DUPE_EPSILON),
    ]
    window_transforms: list[WindowTransform] = [
        # SignedDiffTransform(),
        StaticWindowFilter(tolerance=STATIC_WINDOW_TOLERANCE),
    ]

    print(f"\nEffective fps: ~{MASTER_FPS / args.subsample_step:.0f}")
    print(f"AFK threshold: {afk_max_same} frames (~{AFK_DURATION_SECONDS}s)")
    print(f"Frame transforms: {[t.name for t in frame_transforms]}")
    print(f"Window transforms: {[t.name for t in window_transforms]}")

    print("\n--- Pass 1: Processing runs → temporary chunks ---")
    stream_stats = stream_to_chunks(
        runs,
        key_to_idx,
        args.out_dir,
        args.chunk_size,
        frame_transforms,
        window_transforms,
        args.subsample_step,
    )

    print("\n--- Pass 2: Splitting chunks → train/val/test ---")
    split_info = split_chunks(
        args.out_dir,
        args.val_frac,
        args.test_frac,
        args.seed,
    )

    prepare_dataset_info(
        args.out_dir,
        stream_stats,
        split_info,
        frame_transforms,
        window_transforms,
        args.val_frac,
        args.test_frac,
        args.seed,
        args.chunk_size,
        args.subsample_step,
    )

    print(f"\nDone. Output: {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
