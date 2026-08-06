"""Offline HUD-region jitter + light compression-artifact augmentation.

Generates `datagen/output_augmented/` from `datagen/output/`: several
augmented variants per captured frame, meant to fully replace the raw
captures for training `visual_extraction` -- clean, unaugmented captures are
a poor match for real downloaded gameplay footage, which is always
compressed and was captured at resolutions/HUD scales this project's single
1920x1080 capture setup never sees. Two augmentation families:

- HUD-region jitter: shifts and scales each named HUD element (see
  `datagen.hud_regions.HUD_REGION_NAMES`) by a few pixels/percent, as one
  rigid group -- lets the model generalize across the HUD-size/position
  variance real downloaded footage has that this project's one capture
  resolution doesn't. GPU-batched.
- A light JPEG-compression round-trip, to mimic real downloaded video
  compression. CPU, per-image -- no practical GPU JPEG codec is available
  in this project's dependencies.

Two more light, whole-frame-safe augmentations (brightness/contrast,
downsample-then-upsample) round this out. Both are safe because they don't
touch pixel-level readout precision (hue/saturation, which pill-color and
heart-type reads depend on) or any label-relevant positional data
(`player`/`enemies`/`doors` `frame_position` is frame-relative and never
touched here, since nothing here does a whole-image geometric transform).

Images are grouped by resolution (== `HudRegionConfig`) before processing,
since every image in a group shares the same region bbox layout -- that's
what makes GPU-batched jitter a single vectorized call per group rather
than a per-image loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import logging
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

logger = logging.getLogger(__name__)

DEFAULT_INPUT_DIR = Path(__file__).parent / "output"
DEFAULT_AUGMENTED_DIR = Path(__file__).parent / "output_augmented"


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
        jpeg_quality_range: `(low, high)` JPEG quality sampled per output
            image -- kept light so compression doesn't corrupt HUD
            readability.
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
    variants_per_image: int = 4
    jitter_max_px: int = 4
    jitter_scale_range: tuple[float, float] = (0.92, 1.08)
    jpeg_quality_range: tuple[int, int] = (60, 90)
    brightness_range: tuple[float, float] = (0.9, 1.1)
    contrast_range: tuple[float, float] = (0.9, 1.1)
    resize_scale_range: tuple[float, float] = (0.85, 1.0)
    batch_size: int = 64
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

    Each region's shifted/scaled content is read from the original
    `images` batch, not from a partially-jittered working copy, so
    overlapping bboxes (there shouldn't be any -- see
    `hud_regions._validate_no_overlaps`) can't compound.
    """
    batch_size = images.shape[0]
    out = images.clone()
    for name in HUD_REGION_NAMES:
        bbox = region_cfg.regions[name]
        anchors = HUD_REGION_ANCHORS[name]
        x_low, x_high = _directional_range(anchors, "left", "right", jitter_max_px)
        y_low, y_high = _directional_range(anchors, "top", "bottom", jitter_max_px)
        dx = (
            torch.randint(x_low, x_high + 1, (batch_size,), generator=generator)
            .float()
            .to(images.device)
        )
        dy = (
            torch.randint(y_low, y_high + 1, (batch_size,), generator=generator)
            .float()
            .to(images.device)
        )
        scale = _uniform(batch_size, jitter_scale_range, generator).to(images.device)
        y_slice = slice(bbox.y1, bbox.y2)
        x_slice = slice(bbox.x1, bbox.x2)
        source_region = images[:, :, y_slice, x_slice]
        out[:, :, y_slice, x_slice] = _shift_region(source_region, dx, dy, scale)
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
    """JPEG round-trips an `(H, W, 3)` uint8 RGB `image` at `quality` (1-100).

    CPU, per-image -- there's no practical GPU JPEG codec in this project's
    dependencies, so this intentionally runs on the CPU/numpy side after
    the GPU-batched ops above.
    """
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    ok, encoded = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("cv2.imencode failed to produce a JPEG buffer")
    decoded_bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if decoded_bgr is None:
        raise RuntimeError("cv2.imdecode failed to decode the re-encoded JPEG buffer")
    return cv2.cvtColor(decoded_bgr, cv2.COLOR_BGR2RGB)


def _load_batch(image_paths: list[Path], device: str) -> torch.Tensor:
    arrays = [
        np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)
        for path in image_paths
    ]
    stacked = np.stack(arrays)  # (B, H, W, C)
    tensor = torch.from_numpy(stacked).permute(0, 3, 1, 2).float()  # (B, C, H, W)
    return tensor.to(device)


def generate_variants_batch(
    image_paths: list[Path],
    label_paths: list[Path],
    region_cfg: HudRegionConfig,
    cfg: AugmentConfig,
    generator: torch.Generator,
) -> None:
    """Writes `cfg.variants_per_image` augmented image+label pairs per source to `cfg.output_dir`.

    `image_paths`/`label_paths` (paired by index) must all share
    `region_cfg`'s resolution -- batched `cfg.batch_size` at a time. Label
    JSON is copied verbatim except `image_id` (`-> f"{orig_id}_aug{n}"`);
    `scenario_id` is preserved unchanged so augmented variants stay grouped
    with their source under `visual_extraction.dataset`'s
    split-by-`scenario_id` logic.
    """
    device = cfg.resolved_device()
    images_out = cfg.output_dir / "images"
    labels_out = cfg.output_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)

    for start in range(0, len(image_paths), cfg.batch_size):
        chunk_image_paths = image_paths[start : start + cfg.batch_size]
        chunk_label_paths = label_paths[start : start + cfg.batch_size]
        batch = _load_batch(chunk_image_paths, device)

        for variant in range(cfg.variants_per_image):
            augmented = apply_hud_jitter_batch(
                batch, region_cfg, generator, cfg.jitter_max_px, cfg.jitter_scale_range
            )
            augmented = apply_brightness_contrast_batch(
                augmented, generator, cfg.brightness_range, cfg.contrast_range
            )
            augmented = apply_resize_roundtrip_batch(
                augmented, generator, cfg.resize_scale_range
            )
            augmented = augmented.clamp(0, 255).round().to(torch.uint8).cpu()

            for i, label_path in enumerate(chunk_label_paths):
                array = augmented[i].permute(1, 2, 0).numpy()  # (H, W, C)
                quality = int(_uniform(1, cfg.jpeg_quality_range, generator).item())
                compressed = apply_compression_artifacts(array, quality)

                record = json.loads(label_path.read_text())
                new_id = f"{record['image_id']}_aug{variant}"
                record["image_id"] = new_id

                Image.fromarray(compressed, mode="RGB").save(
                    images_out / f"{new_id}.png"
                )
                (labels_out / f"{new_id}.json").write_text(json.dumps(record))


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

    for resolution, group in _group_by_resolution(pairs).items():
        region_cfg = get_region_config_for_resolution(*resolution)
        image_paths = [image_path for image_path, _ in group]
        label_paths = [label_path for _, label_path in group]
        logger.info(
            "augmenting %d image(s) at resolution %s (config_id=%d)",
            len(group),
            resolution,
            region_cfg.config_id,
        )
        generate_variants_batch(image_paths, label_paths, region_cfg, cfg, generator)


def main() -> None:
    """CLI entry point: `python -m datagen.augment [--input DIR] [--output DIR] [--variants N] [--seed N]`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_AUGMENTED_DIR)
    parser.add_argument("--variants", type=int, default=4)
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
