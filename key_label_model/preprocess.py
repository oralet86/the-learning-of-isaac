#!/usr/bin/env python3
"""Preprocess labeled Isaac runs: decode video frames, clean keys, save .npy files."""

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from constants import KEY_ORDER, PROCESSED_DATA_DIR, TARGET_HEIGHT, TARGET_WIDTH

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


def process_run(run_dir: Path, key_to_idx: dict[str, int], out_dir: Path) -> bool:
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

    run_out = out_dir / name
    if (run_out / "frames.npy").exists() and (run_out / "labels.npy").exists():
        print(f"  Already processed, skipping.")
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


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess labeled Isaac runs for training"
    )
    parser.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("labeled_runs"),
        help="Directory containing run_* subdirectories",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=PROCESSED_DATA_DIR,
        help="Output directory for processed .npy files",
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
        if process_run(run_dir, key_to_idx, args.out_dir):
            success += 1
            frames_path = args.out_dir / run_dir.name / "frames.npy"
            total_frames += np.load(frames_path, mmap_mode="r").shape[0]

    print(
        f"\nDone. {success}/{len(runs)} runs processed, {total_frames:,} total frames."
    )
    print(f"Output: {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
