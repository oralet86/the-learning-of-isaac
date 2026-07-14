#!/usr/bin/env python3
"""Preprocess labeled Isaac runs: decode video frames, clean keys, save .npy files."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from skmultilearn.model_selection import iterative_train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import (
    DEFAULT_DUPE_EPSILON,
    DEFAULT_MAX_SAME_FRAMES,
    DEFAULT_RUNS_DIR,
    DEFAULT_SEED,
    DEFAULT_TEST_FRAC,
    DEFAULT_TRIM_EDGE_FRAC,
    DEFAULT_VAL_FRAC,
    KEY_ORDER,
    POS_WEIGHT_CLAMP,
    PROCESSED_DATA_DIR,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)

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
    """Trim ``frac`` of frames from the start and end of a run."""
    n = len(frames)
    trim_n = int(n * frac)
    if trim_n == 0:
        return frames, labels
    return frames[trim_n:-trim_n], labels[trim_n:-trim_n]


def remove_long_identical_runs(
    frames: np.ndarray, labels: np.ndarray, max_same: int = 10, epsilon: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Remove excess frames from runs of near-identical consecutive frames.

    For any run of frames where each consecutive pair has mean absolute difference
    <= ``epsilon`` (out of 255) longer than ``max_same``, keep only the first frame
    and drop the rest.
    """
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


def process_run(
    run_dir: Path,
    key_to_idx: dict[str, int],
    out_dir: Path,
    max_same_frames: int = 10,
    trim_edge_frac: float = 0.01,
    dupe_epsilon: float = 1.0,
) -> bool:
    name = run_dir.name
    print(f"\nProcessing {name} ...")

    keys_path = run_dir / "keys.json"
    video_path = run_dir / "output.mp4"
    meta_path = run_dir / "meta.json"

    if not keys_path.exists():
        print("  No keys.json, skipping.")
        return False
    if not video_path.exists():
        print("  No output.mp4, skipping.")
        return False

    with open(keys_path) as f:
        raw_keys = json.load(f)

    cleaned = clean_keys(raw_keys)
    labels = keys_to_multilabel(cleaned, key_to_idx)

    print(f"  Loaded {len(raw_keys)} key frames from keys.json")

    meta = {}
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        print(f"  Video meta: {meta.get('target_size')} @ {meta.get('fps')} fps")

    frames = decode_frames_ffmpeg(video_path, TARGET_WIDTH, TARGET_HEIGHT)
    if frames is None:
        return False

    print(f"  Decoded {len(frames)} video frames ({frames.shape})")

    n = min(len(labels), len(frames))
    if len(labels) != len(frames):
        print(f"  Mismatch: keys={len(labels)}, video={len(frames)} → using {n} frames")
        labels = labels[:n]
        frames = frames[:n]

    n_edges_removed = 0
    n_dupes_removed = 0

    if trim_edge_frac > 0:
        frames, labels = trim_edges(frames, labels, trim_edge_frac)
        n_edges_removed = n - len(frames)
        n = len(frames)
        if n_edges_removed > 0:
            print(
                f"  Trimmed {n_edges_removed} edge frames "
                f"({trim_edge_frac:.1%} from each side)"
            )

    if max_same_frames > 0:
        n_before = n
        frames, labels = remove_long_identical_runs(
            frames, labels, max_same_frames, dupe_epsilon
        )
        n_dupes_removed = n_before - len(frames)
        n = len(frames)
        if n_dupes_removed > 0:
            pct = n_dupes_removed / n_before * 100
            print(
                f"  Removed {n_dupes_removed} duplicate frames ({pct:.1f}%) "
                f"from runs >{max_same_frames} near-identical frames "
                f"(epsilon={dupe_epsilon})"
            )

    run_out = out_dir / name
    if (run_out / "frames.npy").exists() and (run_out / "labels.npy").exists():
        print("  Already processed, skipping.")
        return True
    run_out.mkdir(parents=True, exist_ok=True)

    np.save(run_out / "frames.npy", frames)
    np.save(run_out / "labels.npy", labels)

    kc: Counter[str] = Counter()
    for k in cleaned:
        kc.update(k)
    combo_counts = Counter(tuple(sorted(k)) for k in cleaned)

    run_meta = {
        "source_dir": str(run_dir),
        "n_frames": n,
        "n_original_frames": len(raw_keys),
        "n_removed_edges": n_edges_removed,
        "n_removed_duplicates": n_dupes_removed,
        "kept_keys": KEY_ORDER,
        "key_frequencies": {k: int(v) for k, v in kc.most_common()},
        "empty_frames": int(sum(1 for k in cleaned if len(k) == 0)),
        "unique_combinations": len(combo_counts),
        "original_meta": meta,
    }
    with open(run_out / "meta.json", "w") as f:
        json.dump(run_meta, f, indent=2)

    size_mb = frames.nbytes / 1e6 + labels.nbytes / 1e6
    print(f"  Saved frames.npy + labels.npy ({size_mb:.1f} MB) to {run_out}")
    return True


def prepare_training_dataset(
    out_dir: Path, val_frac: float, test_frac: float, seed: int
) -> bool:
    dataset_dir = out_dir / "dataset"
    if (dataset_dir / "dataset_info.json").exists():
        print("\nDataset already prepared, skipping.")
        return True

    runs = sorted(
        d for d in out_dir.iterdir() if d.is_dir() and d.name.startswith("run_")
    )
    if not runs:
        print("\nNo processed runs found, skipping dataset preparation.")
        return False

    print(f"\nPreparing training dataset from {len(runs)} run(s)...")

    frames_list: list[np.ndarray] = []
    labels_list: list[np.ndarray] = []
    run_names: list[str] = []
    for run_dir in runs:
        frames_path = run_dir / "frames.npy"
        labels_path = run_dir / "labels.npy"
        if not frames_path.exists() or not labels_path.exists():
            continue
        frames_list.append(np.load(frames_path))
        labels_list.append(np.load(labels_path))
        run_names.append(run_dir.name)

    if not frames_list:
        print("  No valid run data found.")
        return False

    n_keys = labels_list[0].shape[1]
    h, w = frames_list[0].shape[1], frames_list[0].shape[2]

    run_pair_counts = [max(0, len(f) - 1) for f in frames_list]
    total_pairs = sum(run_pair_counts)
    print(f"  Total frame pairs: {total_pairs:,}")

    X = np.empty((total_pairs, 2, h, w), dtype=np.uint8)
    y = np.empty((total_pairs, n_keys), dtype=np.uint8)
    sample_map = np.empty((total_pairs, 2), dtype=np.int32)

    idx = 0
    for run_idx, (frames, labels) in enumerate(
        zip(frames_list, labels_list, strict=True)
    ):
        n_pairs = run_pair_counts[run_idx]
        if n_pairs == 0:
            continue
        X[idx : idx + n_pairs, 0] = frames[:n_pairs]
        X[idx : idx + n_pairs, 1] = frames[1 : n_pairs + 1]
        y[idx : idx + n_pairs] = labels[:n_pairs]
        sample_map[idx : idx + n_pairs, 0] = run_idx
        sample_map[idx : idx + n_pairs, 1] = np.arange(n_pairs, dtype=np.int32)
        idx += n_pairs

    total_pos = np.zeros(n_keys, dtype=np.float64)
    total_neg = np.zeros(n_keys, dtype=np.float64)
    for labels in labels_list:
        total_pos += labels.sum(axis=0)
        total_neg += (1 - labels).sum(axis=0)
    pos_weight = total_neg / (total_pos + 1e-8)
    pos_weight = np.clip(pos_weight, POS_WEIGHT_CLAMP[0], POS_WEIGHT_CLAMP[1])

    np.random.seed(seed)

    X_idx = np.arange(total_pairs, dtype=np.int32).reshape(-1, 1)
    y_int = y.astype(np.int32)

    X_trainval, y_trainval, X_test, _ = iterative_train_test_split(
        X_idx, y_int, test_size=test_frac
    )

    val_frac_in_remainder = val_frac / (1.0 - test_frac)
    X_train, _, X_val, _ = iterative_train_test_split(
        X_trainval, y_trainval, test_size=val_frac_in_remainder
    )

    train_idx = X_train.flatten().astype(np.int32)
    val_idx = X_val.flatten().astype(np.int32)
    test_idx = X_test.flatten().astype(np.int32)

    dataset_dir.mkdir(parents=True, exist_ok=True)

    np.save(dataset_dir / "train_X.npy", X[train_idx])
    np.save(dataset_dir / "train_y.npy", y[train_idx])
    np.save(dataset_dir / "val_X.npy", X[val_idx])
    np.save(dataset_dir / "val_y.npy", y[val_idx])
    np.save(dataset_dir / "test_X.npy", X[test_idx])
    np.save(dataset_dir / "test_y.npy", y[test_idx])
    np.save(dataset_dir / "test_sample_map.npy", sample_map[test_idx])

    key_order_path = out_dir / "key_order.json"
    with open(key_order_path) as f:
        key_info = json.load(f)

    dataset_info = {
        "key_names": key_info["keys"],
        "pos_weight": pos_weight.tolist(),
        "run_names": run_names,
        "train_count": len(train_idx),
        "val_count": len(val_idx),
        "test_count": len(test_idx),
        "total_pairs": total_pairs,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "seed": seed,
    }
    with open(dataset_dir / "dataset_info.json", "w") as f:
        json.dump(dataset_info, f, indent=2)

    size_mb = (
        X[train_idx].nbytes
        + y[train_idx].nbytes
        + X[val_idx].nbytes
        + y[val_idx].nbytes
        + X[test_idx].nbytes
        + y[test_idx].nbytes
    ) / 1e6
    print(f"  Saved train/val/test splits ({size_mb:.1f} MB) to {dataset_dir}/")
    print(
        f"  Split: train={len(train_idx):,}  val={len(val_idx):,}  test={len(test_idx):,}"
    )

    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Preprocess labeled Isaac runs for training"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help="Directory containing run_* subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for processed .npy files",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=DEFAULT_VAL_FRAC,
        help="Fraction of data for validation",
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=DEFAULT_TEST_FRAC,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--max-same-frames",
        type=int,
        default=DEFAULT_MAX_SAME_FRAMES,
        help="Max consecutive identical frames before removing duplicates",
    )
    parser.add_argument(
        "--trim-edge-frac",
        type=float,
        default=DEFAULT_TRIM_EDGE_FRAC,
        help="Fraction of frames to trim from start and end of each run",
    )
    parser.add_argument(
        "--dupe-epsilon",
        type=float,
        default=DEFAULT_DUPE_EPSILON,
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

    success = 0
    total_frames = 0
    for run_dir in runs:
        if process_run(
            run_dir,
            key_to_idx,
            args.out_dir,
            args.max_same_frames,
            args.trim_edge_frac,
            args.dupe_epsilon,
        ):
            success += 1
            frames_path = args.out_dir / run_dir.name / "frames.npy"
            total_frames += np.load(frames_path).shape[0]

    print(
        f"\nDone. {success}/{len(runs)} runs processed, {total_frames:,} total frames."
    )
    print(f"Output: {args.out_dir.resolve()}/")

    prepare_training_dataset(args.out_dir, args.val_frac, args.test_frac, args.seed)


if __name__ == "__main__":
    main()
