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
_REAL_RESOLUTION = (1920, 1080)


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
            test/dev dummy.
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
            "item_list": BBox(x1=1662, y1=235, x2=1920, y2=789),
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

    table = (real_config, dummy_config)
    for config in table:
        _validate_no_overlaps(config)
    return table


HUD_REGION_TABLE: tuple[HudRegionConfig, ...] = _build_table()


def get_region_config(config_id: int) -> HudRegionConfig:
    """Looks up a `HudRegionConfig` by its `config_id` (0 real, 1 dummy)."""
    for config in HUD_REGION_TABLE:
        if config.config_id == config_id:
            return config
    raise KeyError(f"no HudRegionConfig with config_id={config_id}")


def get_region_config_for_resolution(width: int, height: int) -> HudRegionConfig:
    """Looks up a `HudRegionConfig` by a captured image's own `(width, height)`.

    Raises:
        KeyError: No config matches this resolution -- it isn't the real
            1920x1080 capture config (or the dummy).
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
