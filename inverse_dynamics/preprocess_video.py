#!/usr/bin/env python3
"""Preprocess labeled Isaac runs: decode video, build windows, chunk, split."""

import argparse
from collections import Counter
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import TYPE_CHECKING

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (
    CHUNK_SIZE,
    DUPE_EPSILON,
    KEY_ORDER,
    MAX_SAME_FRAMES,
    PROCESSED_DATA_DIR,
    RUNS_DIR,
    SEED,
    TARGET_HEIGHT,
    TARGET_WIDTH,
    TEST_FRAC,
    TRIM_EDGE_FRAC,
    VAL_FRAC,
)

if TYPE_CHECKING:
    from inverse_dynamics.transforms import DatasetTransform

KEY_SET = set(KEY_ORDER)


def clean_keys(keys_list: list[list[str]]) -> list[list[str]]:
    return [[k for k in frame_keys if k in KEY_SET] for frame_keys in keys_list]


def keys_to_multilabel(
    keys_list: list[list[str]], key_to_idx: dict[str, int]
) -> np.ndarray:
    n = len(keys_list)
    labels = np.zeros((n, len(key_to_idx)), dtype=np.uint8)
    for i, frame_keys in enumerate(keys_list):
        for k in frame_keys:
            idx = key_to_idx.get(k)
            if idx is not None:
                labels[i, idx] = 1
    return labels


def decode_frames_ffmpeg(
    video_path: Path, width: int, height: int
) -> np.ndarray | None:
    frame_bytes = width * height
    cmd = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        str(video_path),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    assert proc.stdout is not None
    assert proc.stderr is not None

    chunks = []
    total_bytes = 0
    while True:
        chunk = proc.stdout.read(8 * 1024 * 1024)
        if not chunk:
            break
        chunks.append(chunk)
        total_bytes += len(chunk)

    proc.stdout.close()
    returncode = proc.wait()

    if returncode != 0:
        stderr = proc.stderr.read().decode(errors="replace")[:500]
        print(f"  ffmpeg error (code {returncode}): {stderr}")
        return None

    if total_bytes == 0:
        print("  No frame data read from video.")
        return None

    if total_bytes % frame_bytes != 0:
        print(
            f"  Warning: data size {total_bytes} not multiple of frame size {frame_bytes}."
            f" Truncating last {total_bytes % frame_bytes} bytes."
        )
        total_bytes -= total_bytes % frame_bytes

    raw = np.frombuffer(b"".join(chunks)[:total_bytes], dtype=np.uint8)
    n_frames = total_bytes // frame_bytes
    frames = raw.reshape(n_frames, height, width)
    return frames


def trim_edges(
    frames: np.ndarray, labels: np.ndarray, frac: float = 0.01
) -> tuple[np.ndarray, np.ndarray]:
    n = len(frames)
    trim_n = int(n * frac)
    if trim_n == 0:
        return frames, labels
    return frames[trim_n:-trim_n], labels[trim_n:-trim_n]


def remove_long_identical_runs(
    frames: np.ndarray, labels: np.ndarray, max_same: int = 10, epsilon: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    n: int = len(frames)
    if n <= 1 or max_same <= 0:
        return frames, labels

    keep_mask = np.ones(n, dtype=bool)
    run_start = 0
    for i in range(1, n):
        diff = float(
            np.mean(
                np.abs(frames[i].astype(np.float32) - frames[i - 1].astype(np.float32))
            )
        )
        if diff > epsilon:
            run_len = i - run_start
            if run_len > max_same:
                keep_mask[run_start + 1 : i] = False
            run_start = i

    run_len = n - run_start
    if run_len > max_same:
        keep_mask[run_start + 1 :] = False

    return frames[keep_mask], labels[keep_mask]


def build_windows(
    frames: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_windows = len(frames) - 3
    if n_windows <= 0:
        return (
            np.empty((0, 4, *frames.shape[1:]), dtype=np.uint8),
            np.empty((0, labels.shape[1]), dtype=np.uint8),
            np.empty((0, 2), dtype=np.int32),
        )

    windows = np.empty((n_windows, 4, *frames.shape[1:]), dtype=np.uint8)
    windows[:, 0] = frames[:n_windows]
    windows[:, 1] = frames[1 : n_windows + 1]
    windows[:, 2] = frames[2 : n_windows + 2]
    windows[:, 3] = frames[3 : n_windows + 3]

    window_labels = labels[1 : n_windows + 1]
    window_map = np.empty((n_windows, 2), dtype=np.int32)
    window_map[:, 1] = np.arange(n_windows, dtype=np.int32)

    return windows, window_labels, window_map


def process_run(
    run_dir: Path,
    key_to_idx: dict[str, int],
    max_same_frames: int = 10,
    trim_edge_frac: float = 0.01,
    dupe_epsilon: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict] | None:
    name = run_dir.name

    keys_path = run_dir / "keys.json"
    video_path = run_dir / "output.mp4"

    if not keys_path.exists() or not video_path.exists():
        return None

    with open(keys_path) as f:
        raw_keys = json.load(f)

    cleaned = clean_keys(raw_keys)
    labels = keys_to_multilabel(cleaned, key_to_idx)

    frames = decode_frames_ffmpeg(video_path, TARGET_WIDTH, TARGET_HEIGHT)
    if frames is None:
        return None

    n = min(len(labels), len(frames))
    if len(labels) != len(frames):
        labels = labels[:n]
        frames = frames[:n]

    n_edges_removed = 0
    n_dupes_removed = 0

    if trim_edge_frac > 0:
        frames, labels = trim_edges(frames, labels, trim_edge_frac)
        n_edges_removed = n - len(frames)
        n = len(frames)

    if max_same_frames > 0:
        n_before = n
        frames, labels = remove_long_identical_runs(
            frames, labels, max_same_frames, dupe_epsilon
        )
        n_dupes_removed = n_before - len(frames)
        n = len(frames)

    windows, window_labels, window_map = build_windows(frames, labels)

    if len(windows) == 0:
        return None

    kc: Counter[str] = Counter()
    for k in cleaned:
        kc.update(k)
    combo_counts = Counter(tuple(sorted(k)) for k in cleaned)

    stats = {
        "run_name": name,
        "n_frames": n,
        "n_original_frames": len(raw_keys),
        "n_removed_edges": n_edges_removed,
        "n_removed_duplicates": n_dupes_removed,
        "n_windows": len(windows),
        "key_frequencies": {k: int(v) for k, v in kc.most_common()},
        "empty_frames": int(sum(1 for k in cleaned if len(k) == 0)),
        "unique_combinations": len(combo_counts),
    }

    return windows, window_labels, window_map, stats


def stream_to_chunks(
    runs: list[Path],
    key_to_idx: dict[str, int],
    out_dir: Path,
    chunk_size: int,
    transforms: list["DatasetTransform"] | None,
    max_same_frames: int,
    trim_edge_frac: float,
    dupe_epsilon: float,
) -> dict:
    chunks_dir = out_dir / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    n_keys = len(key_to_idx)
    buf_X = np.empty((chunk_size, 4, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8)
    buf_y = np.empty((chunk_size, n_keys), dtype=np.uint8)
    buf_map = np.empty((chunk_size, 2), dtype=np.int32)
    buf_pos = 0
    chunk_id = 0

    all_labels: list[np.ndarray] = []
    all_stats: list[dict] = []
    run_idx_counter = 0

    for run_dir in tqdm(runs, desc="Processing runs", unit="run"):
        result = process_run(
            run_dir, key_to_idx, max_same_frames, trim_edge_frac, dupe_epsilon
        )
        if result is None:
            continue

        windows, window_labels, window_map, stats = result
        stats["run_idx"] = run_idx_counter
        window_map[:, 0] = run_idx_counter
        run_idx_counter += 1

        if transforms:
            for t in transforms:
                t(windows)

        all_labels.append(window_labels)
        all_stats.append(stats)

        n_windows = len(windows)
        src_pos = 0

        while src_pos < n_windows:
            space = chunk_size - buf_pos
            take = min(space, n_windows - src_pos)

            buf_X[buf_pos : buf_pos + take] = windows[src_pos : src_pos + take]
            buf_y[buf_pos : buf_pos + take] = window_labels[src_pos : src_pos + take]
            buf_map[buf_pos : buf_pos + take] = window_map[src_pos : src_pos + take]
            buf_pos += take
            src_pos += take

            if buf_pos == chunk_size:
                np.save(chunks_dir / f"chunk_{chunk_id:04d}_X.npy", buf_X)
                np.save(chunks_dir / f"chunk_{chunk_id:04d}_y.npy", buf_y)
                np.save(chunks_dir / f"chunk_{chunk_id:04d}_map.npy", buf_map)
                chunk_id += 1
                buf_pos = 0

        del windows, window_labels, window_map

    if buf_pos > 0:
        np.save(chunks_dir / f"chunk_{chunk_id:04d}_X.npy", buf_X[:buf_pos])
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


def split_chunks(
    out_dir: Path,
    val_frac: float,
    test_frac: float,
    seed: int,
) -> dict:
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

    X_idx = np.arange(total_windows, dtype=np.int32).reshape(-1, 1)
    y_int = y_all.astype(np.int32)

    X_trainval, y_trainval, X_test, _ = iterative_train_test_split(
        X_idx, y_int, test_size=test_frac
    )

    val_frac_in_remainder = val_frac / (1.0 - test_frac)
    X_train, _, X_val, _ = iterative_train_test_split(
        X_trainval, y_trainval, test_size=val_frac_in_remainder
    )

    del y_all, X_idx, y_int, X_trainval, y_trainval

    train_arr = X_train.flatten().astype(np.int32)
    val_arr = X_val.flatten().astype(np.int32)
    test_arr = X_test.flatten().astype(np.int32)
    del X_train, X_val, X_test

    train_dir = out_dir / "train"
    val_dir = out_dir / "val"
    test_dir = out_dir / "test"
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    chunk_X_paths = sorted(chunks_dir.glob("chunk_*_X.npy"))
    chunk_y_paths = sorted(chunks_dir.glob("chunk_*_y.npy"))
    chunk_map_paths = sorted(chunks_dir.glob("chunk_*_map.npy"))

    def make_buf(split_dir: Path, with_map: bool) -> dict:
        return {
            "dir": split_dir,
            "X": np.empty((CHUNK_SIZE, 4, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8),
            "y": np.empty((CHUNK_SIZE, n_keys), dtype=np.uint8),
            "map": np.empty((CHUNK_SIZE, 2), dtype=np.int32) if with_map else None,
            "pos": 0,
            "chunk_id": 0,
            "count": 0,
        }

    def flush(buf: dict) -> None:
        if buf["pos"] == 0:
            return
        cid = buf["chunk_id"]
        np.save(buf["dir"] / f"chunk_{cid:04d}_X.npy", buf["X"][: buf["pos"]])
        np.save(buf["dir"] / f"chunk_{cid:04d}_y.npy", buf["y"][: buf["pos"]])
        if buf["map"] is not None:
            np.save(
                buf["dir"] / f"chunk_{cid:04d}_sample_map.npy", buf["map"][: buf["pos"]]
            )
        buf["chunk_id"] += 1
        buf["pos"] = 0

    def push(buf: dict, X: np.ndarray, y: np.ndarray, map_: np.ndarray | None) -> None:
        n = len(X)
        buf["count"] += n
        src = 0
        while src < n:
            space = CHUNK_SIZE - buf["pos"]
            take = min(space, n - src)
            buf["X"][buf["pos"] : buf["pos"] + take] = X[src : src + take]
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

    for X_path, y_path, map_path in tqdm(
        zip(chunk_X_paths, chunk_y_paths, chunk_map_paths, strict=True),
        total=len(chunk_X_paths),
        desc="Splitting chunks",
        unit="chunk",
    ):
        X_chunk = np.load(X_path)
        y_chunk = np.load(y_path)
        map_chunk = np.load(map_path)
        chunk_size = len(X_chunk)

        indices = np.arange(global_offset, global_offset + chunk_size, dtype=np.int32)

        train_mask = np.isin(indices, train_arr, assume_unique=True)
        val_mask = np.isin(indices, val_arr, assume_unique=True)
        test_mask = np.isin(indices, test_arr, assume_unique=True)

        if train_mask.any():
            push(train_buf, X_chunk[train_mask], y_chunk[train_mask], None)
        if val_mask.any():
            push(val_buf, X_chunk[val_mask], y_chunk[val_mask], None)
        if test_mask.any():
            push(test_buf, X_chunk[test_mask], y_chunk[test_mask], map_chunk[test_mask])

        del X_chunk, y_chunk, map_chunk
        global_offset += chunk_size

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


def prepare_dataset_info(
    out_dir: Path,
    stream_stats: dict,
    split_info: dict,
    transforms: list["DatasetTransform"] | None,
    val_frac: float,
    test_frac: float,
    seed: int,
    chunk_size: int,
) -> None:
    key_order_path = out_dir / "key_order.json"
    with open(key_order_path) as f:
        key_info = json.load(f)

    total_key_freqs: Counter[str] = Counter()
    for rs in stream_stats.get("run_stats", []):
        for k, v in rs.get("key_frequencies", {}).items():
            total_key_freqs[k] += v

    total_original = sum(
        rs.get("n_original_frames", 0) for rs in stream_stats.get("run_stats", [])
    )
    total_edges = sum(
        rs.get("n_removed_edges", 0) for rs in stream_stats.get("run_stats", [])
    )
    total_dupes = sum(
        rs.get("n_removed_duplicates", 0) for rs in stream_stats.get("run_stats", [])
    )
    total_empty = sum(
        rs.get("empty_frames", 0) for rs in stream_stats.get("run_stats", [])
    )

    dataset_info = {
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
        "transforms": [t.name for t in (transforms or [])],
        "aggregated_stats": {
            "total_original_frames": total_original,
            "total_removed_edges": total_edges,
            "total_removed_duplicates": total_dupes,
            "total_empty_frames": total_empty,
            "key_frequencies": dict(total_key_freqs.most_common()),
        },
    }

    with open(out_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    print(f"Saved dataset_info.json to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess labeled Isaac runs for training"
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
        "--max-same-frames",
        type=int,
        default=MAX_SAME_FRAMES,
        help="Max consecutive identical frames before removing duplicates",
    )
    parser.add_argument(
        "--trim-edge-frac",
        type=float,
        default=TRIM_EDGE_FRAC,
        help="Fraction of frames to trim from start and end of each run",
    )
    parser.add_argument(
        "--dupe-epsilon",
        type=float,
        default=DUPE_EPSILON,
        help="Mean absolute pixel difference threshold for identical-frame detection",
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

    # Add transforms here to apply them during preprocessing.
    # Example:
    #   from inverse_dynamics.transforms import SignedDiffTransform
    #   transforms = [SignedDiffTransform()]
    transforms: list["DatasetTransform"] = []

    print("\n--- Pass 1: Processing runs → temporary chunks ---")
    stream_stats = stream_to_chunks(
        runs,
        key_to_idx,
        args.out_dir,
        args.chunk_size,
        transforms,
        args.max_same_frames,
        args.trim_edge_frac,
        args.dupe_epsilon,
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
        transforms,
        args.val_frac,
        args.test_frac,
        args.seed,
        args.chunk_size,
    )

    print(f"\nDone. Output: {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
