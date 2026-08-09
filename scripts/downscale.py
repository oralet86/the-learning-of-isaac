"""Consolidated downscaling for frames, images, and video.

One flexible, fast entry point (`downscale`) for resizing in-memory arrays
(single frame or batch), plus thin I/O wrappers for image files and `.mp4`
video.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
import subprocess
from typing import TYPE_CHECKING

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as tf

if TYPE_CHECKING:
    from pathlib import Path


class Interpolation(StrEnum):
    """Resize algorithm selection for `downscale`."""

    AREA = "area"
    LINEAR = "linear"
    CUBIC = "cubic"
    NEAREST = "nearest"
    CENTER_PIXEL = "center_pixel"


_CV2_INTERPOLATION = {
    Interpolation.AREA: cv2.INTER_AREA,
    Interpolation.LINEAR: cv2.INTER_LINEAR,
    Interpolation.CUBIC: cv2.INTER_CUBIC,
    Interpolation.NEAREST: cv2.INTER_NEAREST,
}

_TORCH_INTERPOLATION = {
    Interpolation.AREA: "area",
    Interpolation.LINEAR: "bilinear",
    Interpolation.CUBIC: "bicubic",
    Interpolation.NEAREST: "nearest",
}


def _check_aspect_ratio(src_w: int, src_h: int, target_w: int, target_h: int) -> None:
    """Raise if source and target aspect ratios aren't exactly equal.

    Uses a cross-multiplication rather than a float comparison so the check
    is exact regardless of resolution.
    """
    if src_w * target_h != src_h * target_w:
        raise ValueError(
            f"Aspect ratio mismatch: source {src_w}x{src_h} vs target "
            f"{target_w}x{target_h}"
        )


@lru_cache(maxsize=8)
def _center_pixel_indices(
    h: int, w: int, target_h: int, target_w: int
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the source row/col indices sampled by center-pixel downscaling.

    For each output pixel, returns the index of the pixel at the center of
    its corresponding source block. Depends only on the frame shape, which
    is constant across a capture session, so results are cached to avoid
    recomputing on every frame in a hot loop.
    """
    sy = h / target_h
    sx = w / target_w
    row_idx = ((np.arange(target_h) + 0.5) * sy).astype(np.intp)
    col_idx = ((np.arange(target_w) + 0.5) * sx).astype(np.intp)
    return row_idx, col_idx


def _downscale_center_pixel(
    batch: np.ndarray, target_w: int, target_h: int
) -> np.ndarray:
    """Center-pixel nearest sampling, batch-aware via fancy indexing.

    No blending: picks the pixel closest to the center of each output
    block rather than interpolating, which is fast (no per-block Python
    loop) and well suited to nearest-neighbour-upscaled pixel art.
    """
    _n, h, w = batch.shape[:3]
    _check_aspect_ratio(w, h, target_w, target_h)
    row_idx, col_idx = _center_pixel_indices(h, w, target_h, target_w)
    return batch[:, row_idx[:, None], col_idx[None, :]]


def _downscale_cv2(
    batch: np.ndarray, target_w: int, target_h: int, interpolation: Interpolation
) -> np.ndarray:
    """Resize each frame in a batch with `cv2.resize`.

    Looped over the batch dimension: `cv2.resize` operates on a single
    image, but releases the GIL, so a Python loop has negligible overhead
    relative to the resize itself.
    """
    cv2_flag = _CV2_INTERPOLATION[interpolation]
    resized = [
        cv2.resize(frame, (target_w, target_h), interpolation=cv2_flag)
        for frame in batch
    ]
    return np.stack(resized, axis=0)


def _downscale_torch(
    batch: np.ndarray,
    target_w: int,
    target_h: int,
    interpolation: Interpolation,
    use_gpu: bool,
) -> np.ndarray:
    """Resize a batch with `torch.nn.functional.interpolate`.

    Operates on the whole batch at once (unlike the cv2 path), so transfer
    and kernel-launch overhead is amortized across frames. Falls back to
    CPU when CUDA isn't available.
    """
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("Torch Downscaling: GPU inaccessible, using CPU.")
    mode = _TORCH_INTERPOLATION[interpolation]

    grayscale = batch.ndim == 3
    tensor = torch.from_numpy(batch).to(device=device, dtype=torch.float32)
    if grayscale:
        tensor = tensor.unsqueeze(1)  # (N,H,W) -> (N,1,H,W)
    else:
        tensor = tensor.permute(0, 3, 1, 2)  # (N,H,W,C) -> (N,C,H,W)

    kwargs = {"align_corners": False} if mode in ("bilinear", "bicubic") else {}
    resized = tf.interpolate(tensor, size=(target_h, target_w), mode=mode, **kwargs)

    if grayscale:
        resized = resized.squeeze(1)
    else:
        resized = resized.permute(0, 2, 3, 1)
    return resized.cpu().numpy()


def downscale(
    frame: np.ndarray,
    target_w: int,
    target_h: int,
    interpolation: Interpolation = Interpolation.AREA,
    force_aspect_ratio: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    """Resize a single frame or a batch of frames.

    Core, I/O-free entry point for all in-repo resizing. Handles both a
    single frame (`(H,W,C)` color or `(H,W)` grayscale) and a batch
    (`(N,H,W,C)`, always 4D — pass `C=1` for a grayscale batch); output has
    the same rank, dtype, and channel layout as the input.

    Args:
        frame: Single frame or batch, `uint8`/`float16`/`float32`. `float16`
            is upcast to `float32` for resizing (neither cv2 nor torch
            reliably supports fp16 array ops on CPU) and cast back before
            returning.
        target_w: Output width.
        target_h: Output height.
        interpolation: Resize algorithm. `CENTER_PIXEL` is a fast,
            non-blending nearest-center sampler suited to pixel art, and
            always enforces an exact aspect-ratio match regardless of
            `force_aspect_ratio` (it has no defined behaviour otherwise).
        force_aspect_ratio: If True, raise `ValueError` unless the source
            and target aspect ratios match exactly.
        use_gpu: If True, dispatch batches (`N > 1`) through
            `torch.nn.functional.interpolate` on CUDA when available,
            falling back to CPU otherwise. Ignored for single frames: a
            single small frame's GPU transfer/launch overhead would
            dominate the resize itself, so single-frame calls always use
            the CPU/cv2 path regardless of this flag.

    Returns:
        Resized array, same rank/dtype as `frame`.

    Raises:
        ValueError: Aspect-ratio mismatch under `force_aspect_ratio` (or
            always, for `CENTER_PIXEL`), or an unsupported number of
            dimensions.
    """
    if frame.ndim not in (2, 3, 4):
        raise ValueError(f"Expected 2, 3, or 4 dimensions, got {frame.ndim}")

    # A batch is always 4D (N,H,W,C); a single frame is 2D (H,W) grayscale
    # or 3D (H,W,C) color. This keeps 3D unambiguous (always a single
    # color frame, never a grayscale batch).
    is_batch = frame.ndim == 4
    batch = frame if is_batch else frame[np.newaxis, ...]

    src_h, src_w = batch.shape[1], batch.shape[2]
    if force_aspect_ratio:
        _check_aspect_ratio(src_w, src_h, target_w, target_h)

    orig_dtype = frame.dtype
    working = batch.astype(np.float32) if orig_dtype == np.float16 else batch

    if interpolation == Interpolation.CENTER_PIXEL:
        result = _downscale_center_pixel(working, target_w, target_h)
    elif use_gpu and batch.shape[0] > 1:
        result = _downscale_torch(
            working, target_w, target_h, interpolation, use_gpu=True
        )
    else:
        result = _downscale_cv2(working, target_w, target_h, interpolation)

    if orig_dtype == np.float16:
        result = result.astype(np.float16)
    else:
        result = result.astype(orig_dtype)

    return result if is_batch else result[0]


_PIL_RESAMPLE = {
    Interpolation.AREA: Image.Resampling.BOX,
    Interpolation.LINEAR: Image.Resampling.BILINEAR,
    Interpolation.CUBIC: Image.Resampling.BICUBIC,
    Interpolation.NEAREST: Image.Resampling.NEAREST,
}


def _resize_pil(
    image: Image.Image, target_w: int, target_h: int, interpolation: Interpolation
) -> np.ndarray:
    """Resize a PIL image with PIL's own resampling filters.

    Used for the image-file wrappers instead of routing through the cv2/torch
    array core: PIL's filters (e.g. `BICUBIC`) aren't numerically identical
    to cv2's or torch's same-named filters.
    """
    resized = image.resize((target_w, target_h), resample=_PIL_RESAMPLE[interpolation])
    return np.array(resized, dtype=np.uint8)


def downscale_image_file(
    path: Path,
    target_w: int,
    target_h: int,
    interpolation: Interpolation = Interpolation.AREA,
    force_aspect_ratio: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    """Read one image file and downscale it.

    Args:
        path: Path to an image file readable by PIL (PNG and similar).
        target_w: Output width.
        target_h: Output height.
        interpolation: See `downscale`. `AREA`/`LINEAR`/`CUBIC`/`NEAREST`
            are resized with PIL's own filters (not cv2's or torch's) so
            output matches what direct `PIL.Image.resize` calls produced.
            `CENTER_PIXEL` has no PIL equivalent and is routed through the
            array core instead.
        force_aspect_ratio: See `downscale`.
        use_gpu: See `downscale`. A no-op for `AREA`/`LINEAR`/`CUBIC`/
            `NEAREST` (PIL is CPU-only); applies only to `CENTER_PIXEL`,
            where it's still a no-op since that's always single-frame.

    Returns:
        `(target_h, target_w, 3)` `uint8` RGB array.
    """
    image = Image.open(path).convert("RGB")
    src_w, src_h = image.size
    if force_aspect_ratio:
        _check_aspect_ratio(src_w, src_h, target_w, target_h)

    if interpolation == Interpolation.CENTER_PIXEL:
        array = np.asarray(image, dtype=np.uint8)
        return downscale(
            array,
            target_w,
            target_h,
            interpolation=interpolation,
            force_aspect_ratio=True,
            use_gpu=use_gpu,
        )

    return _resize_pil(image, target_w, target_h, interpolation)


def downscale_image_batch(
    paths: list[Path],
    target_w: int,
    target_h: int,
    interpolation: Interpolation = Interpolation.AREA,
    force_aspect_ratio: bool = False,
    use_gpu: bool = False,
) -> np.ndarray:
    """Read and downscale a batch of same-resolution image files.

    Args:
        paths: Paths to image files readable by PIL, all sharing the same
            source resolution.
        target_w: Output width.
        target_h: Output height.
        interpolation: See `downscale_image_file`.
        force_aspect_ratio: See `downscale`.
        use_gpu: See `downscale`. Only meaningful for `CENTER_PIXEL`
            (`AREA`/`LINEAR`/`CUBIC`/`NEAREST` always go through PIL, which
            is CPU-only).

    Returns:
        `(N, target_h, target_w, 3)` `uint8` RGB array.

    Raises:
        ValueError: `paths` is empty, or the images don't share a
            resolution.
    """
    if not paths:
        raise ValueError("paths must be non-empty")

    images = [Image.open(p).convert("RGB") for p in paths]
    sizes = {image.size for image in images}
    if len(sizes) > 1:
        raise ValueError(f"All images must share a resolution, got sizes: {sizes}")

    src_w, src_h = next(iter(sizes))
    if force_aspect_ratio:
        _check_aspect_ratio(src_w, src_h, target_w, target_h)

    if interpolation == Interpolation.CENTER_PIXEL:
        batch = np.stack(
            [np.asarray(image, dtype=np.uint8) for image in images], axis=0
        )
        return downscale(
            batch,
            target_w,
            target_h,
            interpolation=interpolation,
            force_aspect_ratio=True,
            use_gpu=use_gpu,
        )

    resized = [
        _resize_pil(image, target_w, target_h, interpolation) for image in images
    ]
    return np.stack(resized, axis=0)


@lru_cache(maxsize=1)
def _ffmpeg_has_cuda() -> bool:
    """Check once whether the system ffmpeg build supports CUDA hwaccel."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-hwaccels"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except OSError, subprocess.TimeoutExpired:
        return False
    return "cuda" in result.stdout.splitlines()


def downscale_video_stream(
    video_path: Path,
    target_w: int,
    target_h: int,
    grayscale: bool = False,
    use_gpu: bool = False,
) -> tuple[subprocess.Popen[bytes], int]:
    """Launch ffmpeg piping resized raw frames to stdout.

    The resize (and optional grayscale conversion) happens entirely inside
    the ffmpeg subprocess, never as an in-memory array in Python, so this
    does not call `downscale`.

    Args:
        video_path: Path to the source video.
        target_w: Output width.
        target_h: Output height.
        grayscale: If True, output single-channel `gray` frames instead of
            `rgb24`.
        use_gpu: If True and the system ffmpeg build has CUDA hwaccel
            support (checked once via `-hwaccels`), decode and scale on
            the GPU via `scale_cuda`. Falls back to the CPU `scale` filter
            when CUDA support isn't available; no mid-stream fallback is
            attempted if the GPU path fails after the process has started,
            since retrying once a pipe is being consumed is unsafe.

    Returns:
        `(proc, frame_size_bytes)`. Each read of `frame_size_bytes` bytes
        from `proc.stdout` yields one frame at native FPS.
    """
    pix_fmt = "gray" if grayscale else "rgb24"
    channels = 1 if grayscale else 3

    if use_gpu and _ffmpeg_has_cuda():
        scale_filter = f"scale_cuda={target_w}:{target_h},hwdownload,format={pix_fmt}"
        input_args = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    else:
        scale_filter = f"scale={target_w}:{target_h},format={pix_fmt}"
        input_args = []

    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel",
        "error",
        *input_args,
        "-i",
        str(video_path),
        "-an",
        "-vf",
        scale_filter,
        "-vsync",
        "0",
        "-f",
        "rawvideo",
        "-pix_fmt",
        pix_fmt,
        "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame_size = target_w * target_h * channels
    return proc, frame_size
