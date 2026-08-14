"""Pixel-space HUD element regions for the project's one capture resolution (1920x1080).

A captured image's own pixel dimensions are enough to look up which
`HudRegionConfig` it was captured under; no extra ground-truth field is
needed for this.

Region coordinates for the real 1920x1080 config were measured against real
captures (see `datagen.find_hud_offsets` and
`datagen/finding_offsets/summary.txt`). A second, dummy config with
made-up but internally-consistent coordinates exists for tests and local
dev.
"""

from __future__ import annotations

from dataclasses import dataclass

# One rigid group per HUD region -- a region's contents are jittered
# together, matching how each of these actually renders as one contiguous
# HUD widget (or, for "vitals", several widgets whose relative layout is
# fixed regardless of resolution, so there's no benefit to jittering them
# independently).
HUD_REGION_NAMES: tuple[str, ...] = (
    "vitals",  # heart row + coins/bombs/keys + active item icon/charge, as one block
    "consumable",  # pocket card/pill icon
    "stats",  # stat display
    "item_list",  # item-tracker overlay row
)

# Which frame edge(s) each region sits flush against on screen. Jitter uses
# this to bias translation *away* from an anchored edge (toward the frame's
# interior) rather than allowing it to move further into a corner/edge it's
# already flush with -- see `datagen.augment.apply_hud_jitter_batch`.
HUD_REGION_ANCHORS: dict[str, tuple[str, ...]] = {
    "vitals": ("left", "top"),
    "stats": ("left",),
    "item_list": ("right",),
    "consumable": ("right", "bottom"),
}

_REAL_CONFIG_ID = 0
_DUMMY_CONFIG_ID = 1
_DERIVED_CONFIG_ID = 2
_REAL_RESOLUTION = (1920, 1080)
_NATIVE_SCALE_DOWN = 4  # 1920x1080 -> 480x270, the game's native render resolution


@dataclass(frozen=True)
class BBox:
    """A pixel-space rectangle, given by its top-left and bottom-right corners (exclusive)."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def w(self) -> int:
        return self.x2 - self.x1

    @property
    def h(self) -> int:
        return self.y2 - self.y1

    def is_placeholder(self) -> bool:
        """Whether this bbox is an unfilled `TODO` (never a real capture's coordinates)."""
        return self.w <= 0 or self.h <= 0


@dataclass(frozen=True)
class HudRegionConfig:
    """One capture config's HUD layout: which resolution it was captured at, and where each HUD element sits.

    Attributes:
        config_id: `0` for the real 1920x1080 config, `1` for the synthetic
            test/dev dummy, `2` for the 480x270 config derived from the
            real one (see `_NATIVE_SCALE_DOWN`).
        resolution: `(width, height)` in pixels, matching captured images'
            own dimensions exactly.
        regions: One `BBox` per name in `HUD_REGION_NAMES`.
    """

    config_id: int
    resolution: tuple[int, int]
    regions: dict[str, BBox]


def _bboxes_overlap(a: BBox, b: BBox) -> bool:
    if a.is_placeholder() or b.is_placeholder():
        return False
    return a.x1 < b.x2 and b.x1 < a.x2 and a.y1 < b.y2 and b.y1 < a.y2


def _validate_no_overlaps(config: HudRegionConfig) -> None:
    names = list(config.regions)
    for i, name_a in enumerate(names):
        for name_b in names[i + 1 :]:
            if _bboxes_overlap(config.regions[name_a], config.regions[name_b]):
                raise ValueError(
                    f"HudRegionConfig {config.config_id}: regions "
                    f"{name_a!r} and {name_b!r} overlap "
                    f"({config.regions[name_a]} vs {config.regions[name_b]})"
                )


def _div_down(value: int, divisor: int) -> int:
    """Integer division rounding toward negative infinity."""
    return value // divisor


def _div_up(value: int, divisor: int) -> int:
    """Integer division rounding toward positive infinity."""
    return -(-value // divisor)


def _downscaled_bbox(bbox: BBox, divisor: int) -> BBox:
    """Scales a `BBox` down by `divisor`, widening it so no content is clipped.

    `x1`/`y1` are floored and `x2`/`y2` are ceiled, so a boundary that
    lands between pixels sits just *outside* the region rather than
    cutting edge pixels off.
    """
    return BBox(
        x1=_div_down(bbox.x1, divisor),
        y1=_div_down(bbox.y1, divisor),
        x2=_div_up(bbox.x2, divisor),
        y2=_div_up(bbox.y2, divisor),
    )


def _build_table() -> tuple[HudRegionConfig, ...]:
    # The one real config: 1920x1080, region bboxes measured against real
    # captures via `datagen.find_hud_offsets` (median over 11 images; see
    # `datagen/finding_offsets/summary.txt`).
    real_config = HudRegionConfig(
        config_id=_REAL_CONFIG_ID,
        resolution=_REAL_RESOLUTION,
        regions={
            "vitals": BBox(x1=0, y1=30, x2=512, y2=292),
            "consumable": BBox(x1=1470, y1=984, x2=1917, y2=1071),
            "stats": BBox(x1=0, y1=356, x2=268, y2=801),
            "item_list": BBox(x1=1662, y1=235, x2=1920, y2=936),
        },
    )

    # The native-render config: `real_config` divided by the capture's 4x
    # upscale factor, bbox boundaries widened outward (see
    # `_downscaled_bbox`) so no HUD content is lost at the lower
    # resolution. Derived rather than hand-measured -- there's nothing to
    # measure, since the 480x270 frame is exactly the pre-upscale render.
    derived_config = HudRegionConfig(
        config_id=_DERIVED_CONFIG_ID,
        resolution=(
            _REAL_RESOLUTION[0] // _NATIVE_SCALE_DOWN,
            _REAL_RESOLUTION[1] // _NATIVE_SCALE_DOWN,
        ),
        regions={
            name: _downscaled_bbox(bbox, _NATIVE_SCALE_DOWN)
            for name, bbox in real_config.regions.items()
        },
    )

    dummy_config = HudRegionConfig(
        config_id=_DUMMY_CONFIG_ID,
        resolution=(320, 180),
        regions={
            "vitals": BBox(x1=8, y1=8, x2=144, y2=44),
            "consumable": BBox(x1=152, y1=8, x2=168, y2=24),
            "stats": BBox(x1=8, y1=48, x2=88, y2=96),
            "item_list": BBox(x1=8, y1=100, x2=208, y2=116),
        },
    )

    table = (real_config, derived_config, dummy_config)
    for config in table:
        _validate_no_overlaps(config)
    return table


HUD_REGION_TABLE: tuple[HudRegionConfig, ...] = _build_table()


def get_region_config(config_id: int) -> HudRegionConfig:
    """Looks up a `HudRegionConfig` by its `config_id` (0 real, 1 dummy, 2 derived)."""
    for config in HUD_REGION_TABLE:
        if config.config_id == config_id:
            return config
    raise KeyError(f"no HudRegionConfig with config_id={config_id}")


def get_region_config_for_resolution(width: int, height: int) -> HudRegionConfig:
    """Looks up a `HudRegionConfig` by a captured image's own `(width, height)`.

    Raises:
        KeyError: No config matches this resolution -- it isn't the real
            1920x1080 capture config, its 480x270 derived config, or the
            dummy.
        ValueError: A matching config exists but still has placeholder
            (unfilled) region coordinates.
    """
    for config in HUD_REGION_TABLE:
        if config.resolution == (width, height):
            if any(bbox.is_placeholder() for bbox in config.regions.values()):
                raise ValueError(
                    f"HudRegionConfig {config.config_id} (resolution "
                    f"{width}x{height}) still has placeholder region "
                    "coordinates -- fill in real measured values before "
                    "running augmentation against it"
                )
            return config
    raise KeyError(
        f"no HudRegionConfig for resolution {width}x{height} -- not one of "
        "the configured capture resolutions"
    )
