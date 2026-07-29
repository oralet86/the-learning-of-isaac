#!/usr/bin/env python3
"""Chunk master video recordings into fixed-size raw frame .npy files.

Reads master.mp4 + keys.json from each run directory, decodes video via
ffmpeg with built-in resize and grayscale, then writes
fixed-size .npy chunks with per-chunk metadata.

Frames are stored at native FPS.  Temporal spacing is deferred to the
dataloader via the stride parameter.

A second, native-resolution ffmpeg decode runs in lockstep with the
gray/downscaled one to extract HUD features (stats, hearts, charge,
collectibles) via ``components.hud_ocr.TorchHudExtractor`` before the
frame is downscaled away — by the time the gray path reaches
``preprocess.py``, HUD extraction is no longer possible (no color, no
resolution left).
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading

import numpy as np
import torch
from tqdm import tqdm

from components.hud_ocr import TorchHudExtractor, hud_field_names
from constants import (
    CHUNK_VIDEO_WORKERS,
    DATA_CHUNKS_DIR,
    HUD_OCR_BATCH_SIZE,
    MASTER_FPS,
    RAW_CHUNK_SIZE_GB,
    RUNS_DIR,
    TARGET_HEIGHT,
    TARGET_WIDTH,
)
from inverse_dynamics.hud_features import forward_fill_hud

_HUD_NATIVE_HEIGHT = 1080
_HUD_NATIVE_WIDTH = 1920
_HUD_DIM = len(hud_field_names())


@dataclass
class ChunkMetadata:
    """Metadata for a single raw-frame chunk produced by chunk_video.py."""

    run_name: str
    chunk_id: int
    is_first: bool
    is_last: bool
    total_chunks: int
    start_frame: int
    end_frame: int
    n_frames: int
    video_fps: float
    effective_fps: float


# ---------------------------------------------------------------------------
# ffmpeg helpers
# ---------------------------------------------------------------------------


def _get_video_fps(video_path: Path) -> float:
    """Return the average fps of a video, or ``MASTER_FPS`` on failure."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=avg_frame_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    fps_str = result.stdout.strip()
    if not fps_str:
        return float(MASTER_FPS)
    if "/" in fps_str:
        num, den = fps_str.split("/")
        return float(num) / float(den)
    return float(fps_str)


def _probe_video_dims(video_path: Path) -> tuple[int, int]:
    """Return ``(height, width)`` of *video_path*'s first video stream.

    Returns ``(0, 0)`` on failure, which never matches the HUD
    extraction precondition and so safely disables it for that run.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "csv=p=0:s=x",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    out = result.stdout.strip()
    if "x" not in out:
        return (0, 0)
    w_str, h_str = out.split("x")
    try:
        return (int(h_str), int(w_str))
    except ValueError:
        return (0, 0)


def _probe_frame_count(video_path: Path, fps: float) -> int | None:
    """Return the exact frame count of *video_path*'s first video stream.

    Falls back to a ``duration * fps`` estimate if ``nb_frames`` is
    unavailable (some containers never populate it), and to ``None``
    (indeterminate progress bar) if that also fails.
    """
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    nb_str = result.stdout.strip()
    if nb_str.isdigit():
        return int(nb_str)

    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    try:
        return int(float(result.stdout.strip()) * fps)
    except ValueError:
        return None


def _decode_video_rgb(video_path: Path) -> tuple[subprocess.Popen[bytes], int]:
    """Launch ffmpeg piping raw native-resolution RGB frames to stdout.

    Unlike :func:`_decode_video`, no ``scale``/``format=gray`` filter is
    applied — frames come out at the source video's native resolution
    and color, which HUD OCR requires. Must be read in lockstep with a
    ``_decode_video`` process on the same source to stay frame-aligned.

    Returns:
        ``(Popen, frame_size_in_bytes)``.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        "format=rgb24",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = _HUD_NATIVE_HEIGHT * _HUD_NATIVE_WIDTH * 3
    return proc, frame_size


def _decode_video(
    video_path: Path,
    target_w: int,
    target_h: int,
) -> tuple[subprocess.Popen[bytes], int]:
    """Launch ffmpeg piping raw grayscale frames to stdout.

    Returns (Popen, frame_size_in_bytes).  Each read of ``frame_size`` bytes
    from ``proc.stdout`` yields one grayscale frame at native FPS.
    """
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-an",
        "-vf",
        f"scale={target_w}:{target_h},format=gray",
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = target_w * target_h
    return proc, frame_size


# ---------------------------------------------------------------------------
# HUD batching
# ---------------------------------------------------------------------------


class _HudBatcher:
    """Extracts HUD features from native-res RGB frames in bounded sub-batches.

    Accumulates raw RGB frames up to ``HUD_OCR_BATCH_SIZE`` at a time
    and runs :class:`TorchHudExtractor` incrementally, so transient RGB
    memory never scales with chunk size (an RGB frame is ~108x larger
    than the grayscale frame ``frames_per_chunk`` was sized for).

    Args:
        extractor: Extractor to run, or ``None`` to skip HUD extraction
            entirely for this run (e.g. non-1920x1080 source) — every
            field is then left at the missing-value sentinel.
        gpu_lock: Serializes ``extractor.extract_batch()`` calls across
            concurrently running ``chunk_run`` threads, so the GPU never
            holds more than one batch's worth of memory at a time.
    """

    def __init__(
        self, extractor: TorchHudExtractor | None, gpu_lock: threading.Lock
    ) -> None:
        self._extractor = extractor
        self._gpu_lock = gpu_lock
        self._pending: list[np.ndarray] = []
        self._results: list[np.ndarray] = []

    def add_frame(self, rgb_frame: np.ndarray) -> None:
        """Queue one native-resolution RGB frame."""
        if self._extractor is None:
            return
        self._pending.append(rgb_frame)
        if len(self._pending) >= HUD_OCR_BATCH_SIZE:
            self._flush()

    def _flush(self) -> None:
        if not self._pending or self._extractor is None:
            self._pending = []
            return
        batch = np.stack(self._pending, axis=0)
        with self._gpu_lock:
            result = self._extractor.extract_batch(batch)
        self._results.append(result)
        self._pending = []

    def take_chunk(self, n_frames: int) -> np.ndarray:
        """Flush pending frames, forward-fill, and return this chunk's HUD array.

        Resets internal state so the next chunk starts fresh (per-chunk
        independence — forward-fill never crosses this boundary).

        Args:
            n_frames: Number of frames written to this chunk.

        Returns:
            ``(n_frames, HUD_DIM)`` float32 array.
        """
        self._flush()
        if self._extractor is None:
            hud = np.full((n_frames, _HUD_DIM), np.nan, dtype=np.float32)
        else:
            hud = np.concatenate(self._results, axis=0)[:n_frames]
        self._results = []
        return forward_fill_hud(hud)


# ---------------------------------------------------------------------------
# Chunk writing
# ---------------------------------------------------------------------------


def _write_chunk(
    chunk_dir: Path,
    chunk_id: int,
    frames: np.ndarray,
    hud: np.ndarray,
    frames_per_chunk: int,
    raw_keys: list[list[str]],
    fps: float,
    run_name: str,
    total_chunks: int,
) -> None:
    """Save one chunk to disk (frames + hud + keys + metadata)."""
    n = len(frames)

    np.save(chunk_dir / f"chunk_{chunk_id:04d}_frames.npy", frames)
    np.save(chunk_dir / f"chunk_{chunk_id:04d}_hud.npy", hud)

    start_frame = chunk_id * frames_per_chunk
    end_frame = start_frame + n
    keys_slice = raw_keys[start_frame:end_frame]
    with open(chunk_dir / f"chunk_{chunk_id:04d}_keys.json", "w") as f:
        json.dump(keys_slice, f)

    meta = ChunkMetadata(
        run_name=run_name,
        chunk_id=chunk_id,
        is_first=(chunk_id == 0),
        is_last=(chunk_id == total_chunks - 1),
        total_chunks=total_chunks,
        start_frame=start_frame,
        end_frame=end_frame,
        n_frames=n,
        video_fps=fps,
        effective_fps=fps,
    )
    with open(chunk_dir / f"chunk_{chunk_id:04d}_meta.json", "w") as f:
        json.dump(asdict(meta), f, indent=2)


# ---------------------------------------------------------------------------
# Per-run processing
# ---------------------------------------------------------------------------


def chunk_run(
    run_dir: Path,
    out_dir: Path,
    chunk_size_bytes: int,
    hud_extractor: TorchHudExtractor,
    gpu_lock: threading.Lock,
    position_pool: queue.Queue[int],
    fps_override: float | None = None,
) -> bool:
    """Decode, grayscale, and chunk one run's video at native FPS.

    A second, native-resolution ffmpeg decode of the same source runs in
    lockstep to extract HUD features via *hud_extractor* — skipped (with
    every field left at the missing-value sentinel) if the source video
    isn't ``1920x1080``, since HUD OCR's template layout assumes that
    resolution. Either way the gray/downscaled frame path is unaffected.

    May run concurrently with other ``chunk_run`` calls on other runs
    (see ``main``'s ``ThreadPoolExecutor``); *gpu_lock* serializes GPU
    access across those threads so the shared *hud_extractor* never runs
    more than one batch at a time. *position_pool* hands out a ``tqdm``
    ``position`` so each concurrently-running call gets its own stacked
    progress bar; the position is returned to the pool when this call
    finishes so a later-queued run can reuse that terminal row.

    Returns False when the run is skipped (missing files), True on success.
    """
    name = run_dir.name
    keys_path = run_dir / "keys.json"
    video_path = run_dir / "master.mp4"

    if not keys_path.exists():
        tqdm.write(f"  Skipping {name}: missing keys.json")
        return False
    if not video_path.exists():
        tqdm.write(f"  Skipping {name}: missing master.mp4")
        return False

    with open(keys_path) as f:
        raw_keys: list[list[str]] = json.load(f)

    fps = fps_override or _get_video_fps(video_path)
    total_frames = _probe_frame_count(video_path, fps)

    position = position_pool.get()
    try:
        bytes_per_frame = TARGET_HEIGHT * TARGET_WIDTH
        frames_per_chunk = max(1, int(chunk_size_bytes / bytes_per_frame))

        run_chunk_dir = out_dir / name
        run_chunk_dir.mkdir(parents=True, exist_ok=True)

        proc, frame_size = _decode_video(video_path, TARGET_WIDTH, TARGET_HEIGHT)

        native_h, native_w = _probe_video_dims(video_path)
        has_hud = (native_h, native_w) == (_HUD_NATIVE_HEIGHT, _HUD_NATIVE_WIDTH)
        if not has_hud:
            tqdm.write(
                f"  {name}: source is {native_w}x{native_h}, not "
                f"{_HUD_NATIVE_WIDTH}x{_HUD_NATIVE_HEIGHT} — HUD extraction disabled"
            )
        proc_rgb, frame_size_rgb = (
            _decode_video_rgb(video_path) if has_hud else (None, 0)
        )

        chunk_buf = np.empty(
            (frames_per_chunk, TARGET_HEIGHT, TARGET_WIDTH), dtype=np.uint8
        )
        frame_buf = bytearray(frame_size)
        frame_buf_rgb = bytearray(frame_size_rgb) if proc_rgb is not None else None
        hud_batcher = _HudBatcher(hud_extractor if has_hud else None, gpu_lock)
        filled = 0
        chunk_id = 0
        total_frames_decoded = 0

        pbar = tqdm(
            total=total_frames,
            desc=name,
            unit="frame",
            position=position,
            leave=False,
            dynamic_ncols=True,
        )
        try:
            try:
                while True:
                    n = proc.stdout.readinto(frame_buf)  # type: ignore[union-attr]
                    if n == 0:
                        break
                    if n != frame_size:
                        break

                    chunk_buf[filled] = np.frombuffer(
                        frame_buf, dtype=np.uint8
                    ).reshape(TARGET_HEIGHT, TARGET_WIDTH)

                    if proc_rgb is not None and frame_buf_rgb is not None:
                        n_rgb = proc_rgb.stdout.readinto(  # type: ignore[union-attr]
                            frame_buf_rgb
                        )
                        if n_rgb == frame_size_rgb:
                            rgb_frame = np.frombuffer(
                                frame_buf_rgb, dtype=np.uint8
                            ).reshape(_HUD_NATIVE_HEIGHT, _HUD_NATIVE_WIDTH, 3)
                            hud_batcher.add_frame(rgb_frame.copy())

                    filled += 1
                    total_frames_decoded += 1
                    pbar.update(1)

                    if filled == frames_per_chunk:
                        hud = hud_batcher.take_chunk(filled)
                        _write_chunk(
                            run_chunk_dir,
                            chunk_id,
                            chunk_buf,
                            hud,
                            frames_per_chunk,
                            raw_keys,
                            fps,
                            name,
                            0,  # placeholder, corrected below
                        )
                        filled = 0
                        chunk_id += 1

                if filled > 0:
                    hud = hud_batcher.take_chunk(filled)
                    _write_chunk(
                        run_chunk_dir,
                        chunk_id,
                        chunk_buf[:filled],
                        hud,
                        frames_per_chunk,
                        raw_keys,
                        fps,
                        name,
                        0,  # placeholder, corrected below
                    )
                    chunk_id += 1
                elif chunk_id == 0:
                    tqdm.write(f"  Skipping {name}: no frames decoded")
                    return False

                proc.wait()
                if proc_rgb is not None:
                    proc_rgb.wait()

            except Exception:
                proc.kill()
                proc.wait()
                if proc_rgb is not None:
                    proc_rgb.kill()
                    proc_rgb.wait()
                raise
        finally:
            pbar.close()

        total_chunks = chunk_id
        _patch_chunk_totals(run_chunk_dir, total_chunks, total_frames_decoded)

        tqdm.write(f"  {name}: {total_frames_decoded:,} frames → {total_chunks} chunk(s)")
        return True
    finally:
        position_pool.put(position)


def _patch_chunk_totals(
    chunk_dir: Path, total_chunks: int, total_run_frames: int
) -> None:
    """Update ``total_chunks``, ``is_last``, and ``total_run_frames`` in all chunk metadata."""
    for meta_path in sorted(chunk_dir.glob("chunk_*_meta.json")):
        with open(meta_path) as f:
            meta: dict = json.load(f)
        chunk_id = meta["chunk_id"]
        meta["total_chunks"] = total_chunks
        meta["is_last"] = chunk_id == total_chunks - 1
        meta["total_run_frames"] = total_run_frames
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk master videos into fixed-size raw frame .npy files"
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
        default=DATA_CHUNKS_DIR,
        help="Output directory for chunked frames",
    )
    parser.add_argument(
        "--chunk-size-gb",
        type=float,
        default=RAW_CHUNK_SIZE_GB,
        help="Target disk size per raw chunk in GB",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=CHUNK_VIDEO_WORKERS,
        help="Number of run_* directories to process concurrently",
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
    print(f"Chunk size: {args.chunk_size_gb} GB")
    print()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    chunk_size_bytes = int(args.chunk_size_gb * 1024**3)
    processed = 0

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"HUD extraction device: {device}")
    hud_extractor = TorchHudExtractor(device=device)
    gpu_lock = threading.Lock()

    # Bounded pool of tqdm `position` slots — always exactly matches the max
    # number of concurrently-alive chunk_run threads (never more than
    # args.workers, never more than len(runs)), so position_pool.get() never
    # actually blocks; it's a slot-assignment mechanism for stacked bars, not
    # a second concurrency throttle (ThreadPoolExecutor already provides that).
    n_positions = min(max(1, args.workers), len(runs))
    position_pool: queue.Queue[int] = queue.Queue()
    for p in range(n_positions):
        position_pool.put(p)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                chunk_run,
                run_dir,
                args.out_dir,
                chunk_size_bytes,
                hud_extractor,
                gpu_lock,
                position_pool,
            ): run_dir
            for run_dir in runs
        }
        try:
            for future in as_completed(futures):
                if future.result():
                    processed += 1
        except Exception:
            executor.shutdown(wait=False, cancel_futures=True)
            raise

    print(f"\nDone. {processed}/{len(runs)} runs processed → {args.out_dir.resolve()}/")


if __name__ == "__main__":
    main()
