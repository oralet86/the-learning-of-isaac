"""Offline HUD-region jitter + light compression-artifact augmentation.

Generates `datagen/output_augmented/` from `datagen/output/`: one
augmented variant per captured frame, meant to fully replace the raw
captures for training `visual_extraction` -- clean, unaugmented captures are
a poor match for real downloaded gameplay footage, which is always
compressed and was captured at resolutions/HUD scales this project's single
1920x1080 capture setup never sees. Two augmentation families:

- HUD-region jitter: shifts and scales each named HUD element (see
  `datagen.hud_regions.HUD_REGION_NAMES`) by a few pixels/percent, as one
  rigid group -- lets the model generalize across the HUD-size/position
  variance real downloaded footage has that this project's one capture
  resolution doesn't. GPU-batched. Jitter is re-sampled -- and, in the
  last resort, neutralized -- whenever a region's magnified/shifted source
  window would reach into a neighboring region's box, so wider jitter
  ranges never smear one widget's content across another's.
- A light WebP-compression round-trip, to mimic real downloaded video
  compression. CPU, per-image -- no practical GPU WebP codec is available
  in this project's dependencies.

Two more light, whole-frame-safe augmentations (brightness/contrast,
downsample-then-upsample) round this out. Both are safe because they don't
touch pixel-level readout precision (hue/saturation, which pill-color and
heart-type reads depend on) or any label-relevant positional data
(`player`/`enemies`/`doors` `frame_position` is frame-relative and never
touched here, since nothing here does a whole-image geometric transform).

Captures are 4x-upscaled frames of a game that natively renders at 480x270,
and `visual_extraction`'s model expects 480x270 inputs, so every capture is
first downsampled to that native resolution (`TARGET_RESOLUTION`): the 4x
downsample recovers the pre-upscale frame, so no real information is lost,
and doing it once here instead of per-sample at train time shrinks every
downstream cost 16x. Image loading and the per-image CPU post-processing
(compression round-trip, label rewrite, PNG save) run on a shared thread
pool, overlapped with each other and with the GPU-batched augmentation.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
import json
import logging
import os
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as nnf

from datagen.hud_regions import (
    HUD_REGION_ANCHORS,
    HUD_REGION_NAMES,
    HudRegionConfig,
    get_region_config_for_resolution,
)
from scripts.downscale import Interpolation, downscale_image_file

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path(__file__).parent / "output"
DEFAULT_AUGMENTED_DIR = Path(__file__).parent / "output_augmented"

# The model's expected input size (`visual_extraction.dataset`): the game's
# native render resolution, exactly 1/4 of the 1920x1080 capture.
TARGET_RESOLUTION = (480, 270)

# How many times a sample whose jittered regions would overlap is re-sampled
# before `_resolve_region_overlaps` falls back to neutralizing the conflict.
_MAX_OVERLAP_RETRIES = 16


@dataclass
class AugmentConfig:
    """Settings for one `datagen.augment` run.

    Attributes:
        input_dir: A `datagen` run's output directory (`images/`/`labels/`)
            to augment.
        output_dir: Where augmented image+label pairs are written, same
            `images/`/`labels/` layout as `input_dir`.
        variants_per_image: How many independently-augmented copies to
            produce per source image.
        jitter_max_px: Max absolute per-region, per-sample pixel shift.
        jitter_scale_range: Per-region, per-sample scale factor range,
            centered on each region's own middle -- simulates the same HUD
            element looking relatively larger/smaller, since this project's
            HUD doesn't scale proportionally with resolution (see
            `hud_regions`).
        quality_range: `(low, high)` WebP quality sampled per output
            image -- kept high so compression can't corrupt HUD
            readability (JPEG at moderate quality was measurably worse).
        brightness_range: Per-sample multiplicative brightness factor range.
        contrast_range: Per-sample multiplicative contrast factor range.
        resize_scale_range: Downsample-then-upsample scale factor range,
            sampled once per augmentation pass (shared across the batch).
            A sampled value `>= 1.0` skips this augmentation for that pass.
        batch_size: Images per GPU batch, within one resolution group.
        device: Torch device string, or `None` to auto-detect (`"cuda"` if
            available, else `"cpu"` -- never assumed).
        seed: Optional seed for reproducible augmentation.
    """

    input_dir: Path = field(default_factory=lambda: DEFAULT_INPUT_DIR)
    output_dir: Path = field(default_factory=lambda: DEFAULT_AUGMENTED_DIR)
    variants_per_image: int = 1
    jitter_max_px: int = 6
    jitter_scale_range: tuple[float, float] = (0.88, 1.12)
    quality_range: tuple[int, int] = (90, 98)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    resize_scale_range: tuple[float, float] = (0.85, 1.0)
    batch_size: int = 512
    device: str | None = None
    seed: int | None = None

    def resolved_device(self) -> str:
        """`self.device`, or an auto-detected `"cuda"`/`"cpu"` fallback."""
        if self.device is not None:
            return self.device
        return "cuda" if torch.cuda.is_available() else "cpu"


def _uniform(
    n: int, value_range: tuple[float, float], generator: torch.Generator
) -> torch.Tensor:
    low, high = value_range
    return torch.empty(n, dtype=torch.float32).uniform_(low, high, generator=generator)


def _pixel_shift_scale_theta(
    dx: torch.Tensor,
    dy: torch.Tensor,
    scale: torch.Tensor,
    w: int,
    h: int,
) -> torch.Tensor:
    """Builds a `(B, 2, 3)` `affine_grid` matrix combining a per-sample scale and pixel shift.

    `dx`/`dy` are per-sample pixel offsets; positive shifts content
    right/down (the sampling grid looks up from `x_out - dx`, so output
    pixels grab source content from `dx` pixels to their left). `scale` is
    a per-sample factor about the region's own center: `< 1` zooms in
    (content appears magnified), `> 1` zooms out (content appears shrunk,
    with the newly-exposed border replicated rather than showing real
    neighboring content). Translation is independent of scale -- both
    apply in the same normalized coordinate space, so `dx`/`dy` mean the
    same pixel shift regardless of `scale`.
    """
    tx = -2.0 * dx / w
    ty = -2.0 * dy / h
    zeros = torch.zeros_like(tx)
    return torch.stack(
        [
            torch.stack([scale, zeros, tx], dim=-1),
            torch.stack([zeros, scale, ty], dim=-1),
        ],
        dim=-2,
    )


def _shift_region(
    region: torch.Tensor,
    dx: torch.Tensor,
    dy: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Rigidly shifts and scales a `(B, C, h, w)` region batch by per-sample `(dx, dy, scale)`.

    Border pixels are replicated (`grid_sample(..., padding_mode="border")`)
    rather than left as a seam/black hole, and sampling is nearest-neighbor
    so a `scale` of exactly `1.0` reproduces the region unchanged rather
    than softening pixel-art HUD icons.
    """
    _, _, h, w = region.shape
    theta = _pixel_shift_scale_theta(dx, dy, scale, w, h).to(region.device)
    grid = nnf.affine_grid(theta, list(region.shape), align_corners=False)
    return nnf.grid_sample(
        region, grid, mode="nearest", padding_mode="border", align_corners=False
    )


def _directional_range(
    anchors: tuple[str, ...], near_edge: str, far_edge: str, max_px: int
) -> tuple[int, int]:
    """Bounds for one axis's jitter, biased toward the frame's interior when a region is edge-anchored.

    `near_edge`/`far_edge` are the two opposite edge labels for one axis
    (`"left"`/`"right"` or `"top"`/`"bottom"`). A region anchored to
    `near_edge` only jitters positive (away from it, toward the frame
    interior); anchored to `far_edge` only jitters negative; anchored to
    neither jitters symmetrically, same as before regions had anchors.
    """
    if near_edge in anchors:
        return 0, max_px
    if far_edge in anchors:
        return -max_px, 0
    return -max_px, max_px


def _region_coord_tensors(
    region_cfg: HudRegionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-region `(x1, y1, x2, y2)` bbox coordinates as `(R,)` float CPU tensors.

    Order matches `HUD_REGION_NAMES`, the same order jitter is applied in.
    """
    return (
        torch.tensor(
            [region_cfg.regions[name].x1 for name in HUD_REGION_NAMES],
            dtype=torch.float32,
        ),
        torch.tensor(
            [region_cfg.regions[name].y1 for name in HUD_REGION_NAMES],
            dtype=torch.float32,
        ),
        torch.tensor(
            [region_cfg.regions[name].x2 for name in HUD_REGION_NAMES],
            dtype=torch.float32,
        ),
        torch.tensor(
            [region_cfg.regions[name].y2 for name in HUD_REGION_NAMES],
            dtype=torch.float32,
        ),
    )


def _sampled_source_rects(
    region_cfg: HudRegionConfig,
    dx: torch.Tensor,
    dy: torch.Tensor,
    scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """The `(R, B)` original-image rectangles (`x_lo`, `x_hi`, `y_lo`, `y_hi`) each region samples after jitter.

    A region's destination shows the *original* image's content from
    `[x_lo, x_hi) x [y_lo, y_hi)`. Inverting the `_pixel_shift_scale_theta`
    affine map (`source = scale * out + tx`, `tx = -2*dx/w`) gives
    `x_lo = scale*x1 - (w/2)(scale-1) - dx`, `x_hi` likewise with `x2`.
    `scale > 1` (zoom out) widens the window past the bbox edges -- where
    it can reach a neighbor's content; `scale < 1` (zoom in) narrows it;
    `dx`/`dy` shift the window either way.
    """
    x1s, y1s, x2s, y2s = _region_coord_tensors(region_cfg)
    half_ext_x = (scale - 1.0) * (x2s - x1s)[:, None] / 2.0
    half_ext_y = (scale - 1.0) * (y2s - y1s)[:, None] / 2.0
    x_lo = scale * x1s[:, None] - half_ext_x - dx
    x_hi = scale * x2s[:, None] - half_ext_x - dx
    y_lo = scale * y1s[:, None] - half_ext_y - dy
    y_hi = scale * y2s[:, None] - half_ext_y - dy
    return x_lo, x_hi, y_lo, y_hi


def _region_overlap_conflicts(
    region_cfg: HudRegionConfig,
    dx: torch.Tensor,
    dy: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Per-sample, per-ordered-pair `(B, R, R)` bool mask of jittered source-rect vs bbox overlaps.

    `mask[b, i, j]` is True when region `i`'s post-jitter sampled source
    rectangle (see `_sampled_source_rects`) reaches into region `j`'s
    bbox -- i.e. `i`'s destination would smear `j`'s content into `i`'s
    border, the "regions overlap" artifact the caller guards against. The
    diagonal is masked out: every region trivially overlaps its own bbox.
    """
    x1s, y1s, x2s, y2s = _region_coord_tensors(region_cfg)
    x_lo, x_hi, y_lo, y_hi = _sampled_source_rects(region_cfg, dx, dy, scale)
    x_lo = x_lo.t()[:, :, None]  # (B, R, 1) -- the sampled-source side
    x_hi = x_hi.t()[:, :, None]
    y_lo = y_lo.t()[:, :, None]
    y_hi = y_hi.t()[:, :, None]
    x_overlap = (x_lo < x2s[None, None, :]) & (x1s[None, None, :] < x_hi)
    y_overlap = (y_lo < y2s[None, None, :]) & (y1s[None, None, :] < y_hi)
    n_regions = len(HUD_REGION_NAMES)
    conflicts = (x_overlap & y_overlap) & ~torch.eye(n_regions, dtype=torch.bool)[None]
    return conflicts


def _sample_region_jitter(
    generator: torch.Generator,
    jitter_max_px: int,
    jitter_scale_range: tuple[float, float],
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Samples one `(dx, dy, scale)` triple per region per sample, as CPU `(R, B)` tensors.

    Translation ranges follow each region's anchors via `_directional_range`
    (edge-flush regions jitter toward the frame's interior), exactly as the
    applied jitter would.
    """
    n_regions = len(HUD_REGION_NAMES)
    dx = torch.empty(n_regions, batch_size, dtype=torch.float32)
    dy = torch.empty(n_regions, batch_size, dtype=torch.float32)
    scale = torch.empty(n_regions, batch_size, dtype=torch.float32)
    for r, name in enumerate(HUD_REGION_NAMES):
        anchors = HUD_REGION_ANCHORS[name]
        x_low, x_high = _directional_range(anchors, "left", "right", jitter_max_px)
        y_low, y_high = _directional_range(anchors, "top", "bottom", jitter_max_px)
        dx[r] = torch.randint(
            x_low, x_high + 1, (batch_size,), generator=generator
        ).float()
        dy[r] = torch.randint(
            y_low, y_high + 1, (batch_size,), generator=generator
        ).float()
        scale[r] = _uniform(batch_size, jitter_scale_range, generator)
    return dx, dy, scale


def _resolve_region_overlaps(
    region_cfg: HudRegionConfig,
    dx: torch.Tensor,
    dy: torch.Tensor,
    scale: torch.Tensor,
    generator: torch.Generator,
    jitter_max_px: int,
    jitter_scale_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Re-samples jitter for samples whose regions would sample each other's content, damping stubborn pairs to identity.

    With the default jitter ranges the HUD regions sit close enough that a
    region zoomed out (`scale > 1`) or shifted hard can reach into a
    neighbor's bbox (see `_region_overlap_conflicts`). Conflicted samples
    are re-sampled up to `_MAX_OVERLAP_RETRIES` times; any conflict still
    standing then has *both* members of every conflicting pair neutralized
    (`scale=1`, `dx=dy=0`). That always resolves it: base bboxes don't
    overlap (`hud_regions._validate_no_overlaps`), and shrinking a region's
    rect to identity can only remove conflicts with its neighbors, never
    create new ones.
    """
    batch_size = dx.shape[1]
    conflicts = _region_overlap_conflicts(region_cfg, dx, dy, scale)
    sample_conflict = conflicts.any(dim=(1, 2))
    for _ in range(_MAX_OVERLAP_RETRIES):
        if not bool(sample_conflict.any()):
            break
        fresh_dx, fresh_dy, fresh_scale = _sample_region_jitter(
            generator, jitter_max_px, jitter_scale_range, batch_size
        )
        keep = sample_conflict[:, None].t()  # (1, B) -> broadcast over (R, B)
        dx = torch.where(keep, fresh_dx, dx)
        dy = torch.where(keep, fresh_dy, dy)
        scale = torch.where(keep, fresh_scale, scale)
        conflicts = _region_overlap_conflicts(region_cfg, dx, dy, scale)
        sample_conflict = conflicts.any(dim=(1, 2))
    region_conflict = conflicts.any(dim=2) | conflicts.any(dim=1)  # (B, R)
    neutralized = region_conflict.t()  # (R, B)
    dx = torch.where(neutralized, torch.zeros_like(dx), dx)
    dy = torch.where(neutralized, torch.zeros_like(dy), dy)
    scale = torch.where(neutralized, torch.ones_like(scale), scale)
    return dx, dy, scale


def apply_hud_jitter_batch(
    images: torch.Tensor,
    region_cfg: HudRegionConfig,
    generator: torch.Generator,
    jitter_max_px: int,
    jitter_scale_range: tuple[float, float],
) -> torch.Tensor:
    """Jitters each named HUD region in `images` (`(B, C, H, W)`) by an independent per-sample, per-region shift and scale.

    Every region is jittered as one rigid group (not per-icon) -- position
    via `(dx, dy)` up to `jitter_max_px`, and size via a scale factor in
    `jitter_scale_range`, simulating the same HUD element looking
    relatively larger/smaller across different real capture resolutions
    (this project's HUD renders at its own fixed pixel size, not scaled
    proportionally with resolution -- see `hud_regions`).

    Most HUD regions sit flush against a frame edge (see
    `hud_regions.HUD_REGION_ANCHORS`), so translation is biased *away* from
    each anchored edge (toward the frame's interior) rather than allowed to
    move further into a corner/edge it's already flush with -- symmetric
    jitter there would risk sampling past where any real content exists.
    An axis with no anchor jitters symmetrically as usual.

    Jitter that would make a region's sampled source window reach into a
    neighboring region's bbox (possible at the outer ends of these ranges,
    e.g. an edge-flush region zoomed out toward its neighbor) is re-sampled
    up to `_MAX_OVERLAP_RETRIES` times, then any stubborn conflict is
    neutralized to identity -- so wider jitter ranges never smear one
    widget's content across another's (see `_resolve_region_overlaps`).

    Each region's shifted/scaled content is read from the original
    `images` batch, not from a partially-jittered working copy, so
    overlapping bboxes (there shouldn't be any -- see
    `hud_regions._validate_no_overlaps`) can't compound.
    """
    batch_size = images.shape[0]
    dx, dy, scale = _sample_region_jitter(
        generator, jitter_max_px, jitter_scale_range, batch_size
    )
    dx, dy, scale = _resolve_region_overlaps(
        region_cfg, dx, dy, scale, generator, jitter_max_px, jitter_scale_range
    )
    out = images.clone()
    for r, name in enumerate(HUD_REGION_NAMES):
        bbox = region_cfg.regions[name]
        y_slice = slice(bbox.y1, bbox.y2)
        x_slice = slice(bbox.x1, bbox.x2)
        source_region = images[:, :, y_slice, x_slice]
        out[:, :, y_slice, x_slice] = _shift_region(
            source_region,
            dx[r].to(images.device),
            dy[r].to(images.device),
            scale[r].to(images.device),
        )
    return out


def apply_brightness_contrast_batch(
    images: torch.Tensor,
    generator: torch.Generator,
    brightness_range: tuple[float, float],
    contrast_range: tuple[float, float],
) -> torch.Tensor:
    """Applies an independent per-sample brightness/contrast factor, vectorized over the batch."""
    batch_size = images.shape[0]
    brightness = _uniform(batch_size, brightness_range, generator).to(images.device)
    contrast = _uniform(batch_size, contrast_range, generator).to(images.device)
    brightness = brightness.view(-1, 1, 1, 1)
    contrast = contrast.view(-1, 1, 1, 1)
    mean = images.mean(dim=(1, 2, 3), keepdim=True)
    return (images - mean) * contrast * brightness + mean * brightness


def apply_resize_roundtrip_batch(
    images: torch.Tensor,
    generator: torch.Generator,
    scale_range: tuple[float, float],
) -> torch.Tensor:
    """Downsamples then upsamples the whole batch by one shared scale factor, sampled fresh per call.

    Simulates a lower-quality source without needing per-sample scale
    factors (`nnf.interpolate` has no such notion) -- still varies from one
    augmentation pass to the next since each call re-samples. A sampled
    scale `>= 1.0` is a no-op.
    """
    scale = float(_uniform(1, scale_range, generator).item())
    if scale >= 1.0:
        return images
    _, _, h, w = images.shape
    small = nnf.interpolate(
        images, scale_factor=scale, mode="bilinear", align_corners=False
    )
    return nnf.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def apply_compression_artifacts(image: np.ndarray, quality: int) -> np.ndarray:
    """WebP-round-trips an `(H, W, 3)` uint8 RGB `image` at `quality` (0-100).

    WebP lossy replaces the JPEG round-trip this pipeline previously
    used: JPEG's 8x8 DCT blocking visibly destroyed thin pixel-art HUD
    glyphs (some targets became unreadable even to a human), while WebP's
    prediction-based coding keeps edges far cleaner at the same quality.
    It still mimics real downloaded video compression -- the whole point
    of this step -- but at the high qualities configured here (`(90, 98)`
    by default) HUD content survives intact.

    CPU, per-image -- there's no practical GPU WebP codec in this
    project's dependencies, so this intentionally runs on the CPU/numpy
    side after the GPU-batched ops above.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(
        ".webp",
        bgr,
        [
            int(cv2.IMWRITE_WEBP_QUALITY),
            quality,
        ],
    )
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a WebP buffer")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("cv2.imdecode failed to decode the re-encoded WebP buffer")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def _load_batch(
    image_paths: list[Path],
    region_cfg: HudRegionConfig,
    executor: ThreadPoolExecutor,
) -> torch.Tensor:
    """Loads a batch downsampled to `region_cfg.resolution`, as a CPU `(B, C, H, W)` float32 tensor.

    Decoding and downsampling each image is its own executor task (PIL
    releases the GIL, so this parallelizes); results are stacked in path
    order. Returns a CPU tensor -- the caller moves it to the GPU -- so a
    prefetched batch never touches CUDA from a worker thread.
    """
    width, height = region_cfg.resolution
    futures = [
        executor.submit(downscale_image_file, path, width, height, Interpolation.AREA)
        for path in image_paths
    ]
    arrays = np.stack([future.result() for future in futures])
    return torch.from_numpy(arrays).permute(0, 3, 1, 2).float()


def _write_augmented_pair(
    augmented: torch.Tensor,
    record: dict[str, object],
    quality: int,
    images_out: Path,
    labels_out: Path,
    new_id: str,
) -> None:
    """Compression-round-trips one augmented image, rewrites its label's `image_id`, and saves both.

    Runs on a worker thread of `generate_variants_batch`'s executor: cv2,
    PIL, and `json.dumps` all release the GIL, so the per-image work
    parallelizes across threads. `record` is copied before `image_id` is
    rewritten so variants never mutate a shared label.
    """
    array = augmented.permute(1, 2, 0).numpy()  # (H, W, C)
    compressed = apply_compression_artifacts(array, quality)
    variant_record = dict(record)
    variant_record["image_id"] = new_id
    Image.fromarray(compressed, mode="RGB").save(images_out / f"{new_id}.png")
    (labels_out / f"{new_id}.json").write_text(json.dumps(variant_record))


def generate_variants_batch(
    image_paths: list[Path],
    label_paths: list[Path],
    region_cfg: HudRegionConfig,
    cfg: AugmentConfig,
    generator: torch.Generator,
) -> None:
    """Writes `cfg.variants_per_image` augmented image+label pairs per source to `cfg.output_dir`.

    `image_paths`/`label_paths` (paired by index) are each downsampled to
    `region_cfg.resolution` on load, then processed in batches of
    `cfg.batch_size`. Label JSON is copied verbatim except `image_id`
    (`-> f"{orig_id}_aug{n}"`); `scenario_id` is preserved unchanged so
    augmented variants stay grouped with their source under
    `visual_extraction.dataset`'s split-by-`scenario_id` logic.

    Runs on a shared thread pool: the next batch's decode is submitted
    before the current batch's GPU augmentation, so loading, GPU work, and
    the per-image CPU post-processing all overlap.
    """
    device = cfg.resolved_device()
    images_out = cfg.output_dir / "images"
    labels_out = cfg.output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        decode_future: Future[torch.Tensor] | None = None
        for start in range(0, len(image_paths), cfg.batch_size):
            chunk_image_paths = image_paths[start : start + cfg.batch_size]
            chunk_label_paths = label_paths[start : start + cfg.batch_size]

            if decode_future is None:
                batch = _load_batch(chunk_image_paths, region_cfg, executor)
            else:
                batch = decode_future.result()

            next_start = start + cfg.batch_size
            if next_start < len(image_paths):
                decode_future = executor.submit(
                    _load_batch,
                    image_paths[next_start : next_start + cfg.batch_size],
                    region_cfg,
                    executor,
                )
            else:
                decode_future = None

            batch = batch.to(device)
            records = [json.loads(path.read_text()) for path in chunk_label_paths]

            postproc_futures: list[Future[None]] = []
            for variant in range(cfg.variants_per_image):
                augmented = apply_hud_jitter_batch(
                    batch,
                    region_cfg,
                    generator,
                    cfg.jitter_max_px,
                    cfg.jitter_scale_range,
                )
                augmented = apply_brightness_contrast_batch(
                    augmented, generator, cfg.brightness_range, cfg.contrast_range
                )
                augmented = apply_resize_roundtrip_batch(
                    augmented, generator, cfg.resize_scale_range
                )
                augmented = augmented.clamp(0, 255).round().to(torch.uint8).cpu()

                for i, record in enumerate(records):
                    new_id = f"{record['image_id']}_aug{variant}"
                    quality = int(_uniform(1, cfg.quality_range, generator).item())
                    postproc_futures.append(
                        executor.submit(
                            _write_augmented_pair,
                            augmented[i],
                            record,
                            quality,
                            images_out,
                            labels_out,
                            new_id,
                        )
                    )

            for future in postproc_futures:
                future.result()


def _discover_pairs(input_dir: Path) -> list[tuple[Path, Path]]:
    labels_dir = input_dir / "labels"
    images_dir = input_dir / "images"
    pairs = []
    for label_path in sorted(labels_dir.glob("*.json")):
        record = json.loads(label_path.read_text())
        image_path = images_dir / f"{record['image_id']}.png"
        if image_path.exists():
            pairs.append((image_path, label_path))
    return pairs


def _group_by_resolution(
    pairs: list[tuple[Path, Path]],
) -> dict[tuple[int, int], list[tuple[Path, Path]]]:
    groups: dict[tuple[int, int], list[tuple[Path, Path]]] = {}
    for image_path, label_path in pairs:
        with Image.open(image_path) as img:
            resolution = img.size  # (width, height)
        groups.setdefault(resolution, []).append((image_path, label_path))
    return groups


def run(cfg: AugmentConfig) -> None:
    """Augments every image/label pair under `cfg.input_dir`, writing results to `cfg.output_dir`."""
    generator = torch.Generator()
    if cfg.seed is not None:
        generator.manual_seed(cfg.seed)

    pairs = _discover_pairs(cfg.input_dir)
    if not pairs:
        logger.warning("no image/label pairs found under %s", cfg.input_dir)
        return

    region_cfg = get_region_config_for_resolution(*TARGET_RESOLUTION)
    for resolution, group in _group_by_resolution(pairs).items():
        image_paths = [image_path for image_path, _ in group]
        label_paths = [label_path for _, label_path in group]
        logger.info(
            "augmenting %d image(s) at resolution %s -> %dx%d (config_id=%d)",
            len(group),
            resolution,
            TARGET_RESOLUTION[0],
            TARGET_RESOLUTION[1],
            region_cfg.config_id,
        )
        generate_variants_batch(image_paths, label_paths, region_cfg, cfg, generator)


def main() -> None:
    """CLI entry point: `python -m datagen.augment [--input DIR] [--output DIR] [--variants N] [--seed N]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUGMENTED_DIR)
    parser.add_argument("--variants", type=int, default=1)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    cfg = AugmentConfig(
        input_dir=args.input,
        output_dir=args.output,
        variants_per_image=args.variants,
        seed=args.seed,
    )
    run(cfg)


if __name__ == "__main__":
    main()
