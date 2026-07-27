"""HUD extraction for The Binding of Isaac: Rebirth.

Reads collectible counts, heart states, charge bar, and pause-screen
stats from a 1920×1080 game frame using pixel-font OCR and template
matching.  Frames are first downscaled 4× with majority-vote pooling
to recover the native 480×270 internal resolution before OCR runs.

Usage::

    from live.hud import HudExtractor

    hud = HudExtractor()
    state = hud.extract(frame)  # frame: (1080, 1920, 3) uint8
    print(state)

"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from live.ocr import ChargeBarReader, HeartReader, HeartState, PixelDigitOCR
from live.preprocess import downscale_4x

# ---------------------------------------------------------------------------
# Constants — all coordinates are for a downscaled 480×270 frame
# (native game resolution, ¼ of the 1920×1080 capture).
# ---------------------------------------------------------------------------

# -- Template directories --------------------------------------------------

_TEMPLATE_DIR_STATS = Path("data/templates/stats")
_TEMPLATE_DIR_COLLECTIBLES = Path("data/templates/collectibles")
_TEMPLATE_DIR_HEARTS = Path("data/templates/hearts")

# -- Layout --------------------------------------------------------------

_HUD_LEFT_X = 35

# Stat rows
_STAT_NAMES = [
    "speed",
    "firerate",
    "damage",
    "range",
    "shotspeed",
    "luck",
    "devilchance",
    "angelchance",
]
_STAT_Y_START = 107
_STAT_Y_STEP = 12
_STAT_Y_EXTRA: dict[str, int] = {
    "devilchance": 1,
    "angelchance": 1,
}

# Collectible Y positions
_COLLECTIBLE_Y: dict[str, int] = {
    "coins": 50,
    "bombs": 62,
    "keys": 74,
}

# Heart grid origin
_HEARTS_X = 61
_HEARTS_Y = 23

# Charge bar region
_CHARGE_X = 55
_CHARGE_Y = 20
_CHARGE_W = 4
_CHARGE_H = 26

# -- Glyph sizes (debug crop export) -------------------------------------

_STATS_GLYPH_W = 6
_STATS_GLYPH_H = 7
_COLLECTIBLE_GLYPH_W = 7
_COLLECTIBLE_GLYPH_H = 9
_GLYPH_OVERLAP = 1

_COLLECTIBLE_MAX_DIGITS: dict[str, int] = {
    "coins": 3,
    "bombs": 2,
    "keys": 2,
}

# -- OCR patterns --------------------------------------------------------

_STATS_DEFAULT_PATTERN = r"\d+\.\d{2}"
_STATS_CHANCE_PATTERN = r"\d+\.\d%?"
_STATS_LUCK_PATTERN = r"-?\d+\.\d{2}"

_STAT_PATTERNS: dict[str, str] = {
    "devilchance": _STATS_CHANCE_PATTERN,
    "angelchance": _STATS_CHANCE_PATTERN,
    "luck": _STATS_LUCK_PATTERN,
}


def _stat_y(name: str, index: int) -> int:
    """Compute the y-coordinate of a stat row's anchor.

    Args:
        name: Stat name, used to look up its per-stat y offset.
        index: Position of *name* within :data:`_STAT_NAMES`.

    Returns:
        Absolute y-coordinate in the downscaled 480×270 frame.
    """
    return _STAT_Y_START + index * _STAT_Y_STEP + _STAT_Y_EXTRA.get(name, 0)


class _StatsOcrKwargs(TypedDict):
    """Keyword arguments accepted by ``PixelDigitOCR.__call__`` that vary per stat."""

    percent_eligible: bool
    allow_minus: bool
    override_pattern: str | None


def _stats_ocr_kwargs(name: str) -> _StatsOcrKwargs:
    """Per-stat keyword arguments forwarded to the stats OCR call.

    Args:
        name: Stat name, e.g. ``"luck"`` or ``"devilchance"``.

    Returns:
        Keyword arguments reflecting *name*'s formatting: whether a
        trailing ``%`` is allowed, whether a leading ``-`` is allowed,
        and any stat-specific pattern override.
    """
    return {
        "percent_eligible": name in ("devilchance", "angelchance"),
        "allow_minus": name == "luck",
        "override_pattern": _STAT_PATTERNS.get(name),
    }


# ---------------------------------------------------------------------------
# HudState
# ---------------------------------------------------------------------------


@dataclass
class HudState:
    """Structured HUD reading for a single frame.

    Attributes:
        stats: Per-stat string values (e.g. ``"2.73"``, ``"5"``,
            ``"-1"``, ``"42%"``).  ``None`` when unreadable.
        bombs: Bomb count (``None`` when unreadable).
        keys: Key count (``None`` when unreadable).
        coins: Coin count (``None`` when unreadable).
        charge: Charge-bar level in ``[0, 2]``.
        hearts: List of detected heart states left-to-right,
            top-to-bottom.
    """

    stats: dict[str, str | None] = field(default_factory=dict)
    bombs: int | None = None
    keys: int | None = None
    coins: int | None = None
    charge: float | None = None
    hearts: list[HeartState] = field(default_factory=list)


# ---------------------------------------------------------------------------
# HudExtractor
# ---------------------------------------------------------------------------


class HudExtractor:
    """Run OCR on a downscaled game frame.

    Constructs one set of readers and re-uses them across calls so
    template loading happens only once.

    Args:
        stats_ocr_kwargs: Extra keyword arguments forwarded to the
            stats ``PixelDigitOCR`` (e.g. ``match_threshold``).
        collectibles_ocr_kwargs: Extra keyword arguments for the
            collectibles ``PixelDigitOCR``.
        charge_kwargs: Extra keyword arguments for ``ChargeBarReader``.
        hearts_kwargs: Extra keyword arguments for ``HeartReader``.

    Usage::

        hud = HudExtractor()
        state = hud.extract(frame)
    """

    def __init__(
        self,
        stats_ocr_kwargs: dict | None = None,
        collectibles_ocr_kwargs: dict | None = None,
        charge_kwargs: dict | None = None,
        hearts_kwargs: dict | None = None,
    ) -> None:
        so = stats_ocr_kwargs or {}
        co = collectibles_ocr_kwargs or {}
        cho = charge_kwargs or {}
        hw = hearts_kwargs or {}

        self._stats_ocr = PixelDigitOCR(
            _TEMPLATE_DIR_STATS,
            output_str=True,
            overlap=1,
            pattern=_STATS_DEFAULT_PATTERN,
            narrow_one_adjust=True,
            white_only_mask=True,
            **so,
        )
        self._collectibles_ocr = PixelDigitOCR(
            _TEMPLATE_DIR_COLLECTIBLES, overlap=1, **co
        )
        self._charge_reader = ChargeBarReader(**cho)
        self._hearts_reader = HeartReader(_TEMPLATE_DIR_HEARTS, overlap=1, **hw)

    @property
    def heart_cell_size(self) -> tuple[int, int]:
        """``(height, width)`` of the auto-detected heart cell size."""
        return self._hearts_reader.cell_size

    @property
    def heart_native_cell_size(self) -> tuple[int, int]:
        """``(height, width)`` of native heart sprites in the game frame."""
        return self._hearts_reader.native_cell_size

    @property
    def heart_reader(self) -> HeartReader:
        """The underlying ``HeartReader`` used for heart detection."""
        return self._hearts_reader

    @property
    def stats_ocr(self) -> PixelDigitOCR:
        """The ``PixelDigitOCR`` used for stat-value reading."""
        return self._stats_ocr

    @property
    def collectibles_ocr(self) -> PixelDigitOCR:
        """The ``PixelDigitOCR`` used for collectible-count reading."""
        return self._collectibles_ocr

    # -- main entry point ---------------------------------------------------

    def downscale(self, frame: np.ndarray) -> np.ndarray:
        """Downscale a raw 1920×1080 frame to the native 480×270 resolution.

        Args:
            frame: ``(1080, 1920, 3)`` uint8 or fp16 frame.

        Returns:
            ``(270, 480, 3)`` uint8 downscaled frame.
        """
        return downscale_4x(frame)

    def extract(
        self,
        frame: np.ndarray,
        block_size: int = 11,
        c_const: int = 2,
        verbose: bool = False,
    ) -> HudState:
        """Extract all HUD elements from *frame*.

        Args:
            frame: ``(1080, 1920, 3)`` uint8 or fp16 frame at
                   OBS virtual-camera resolution.
            block_size: Adaptive threshold block size forwarded to
                both ``PixelDigitOCR`` instances.
            c_const: Threshold constant forwarded to both
                ``PixelDigitOCR`` instances.
            verbose: When ``True``, print per-position match scores
                to stderr for all readers.

        Returns:
            ``HudState`` with all readable values populated.

        Raises:
            ValueError: If *frame* does not have shape ``(1080, 1920, 3)``.
        """
        if frame.shape != (1080, 1920, 3):
            raise ValueError(f"Expected frame shape (1080, 1920, 3), got {frame.shape}")

        small = self.downscale(frame)

        stats = self._read_stats(small, block_size, c_const, verbose)
        bombs = self._read_collectible(small, "bombs", block_size, c_const, verbose)
        keys = self._read_collectible(small, "keys", block_size, c_const, verbose)
        coins = self._read_collectible(small, "coins", block_size, c_const, verbose)
        charge = self._charge_reader(
            small,
            x=_CHARGE_X,
            y=_CHARGE_Y,
            width=_CHARGE_W,
            height=_CHARGE_H,
        )
        hearts = self._hearts_reader(small, x=_HEARTS_X, y=_HEARTS_Y, verbose=verbose)

        return HudState(
            stats=stats,
            bombs=bombs,
            keys=keys,
            coins=coins,
            charge=charge,
            hearts=hearts,
        )

    # -- per-element helpers ------------------------------------------------

    def _read_stats(
        self,
        frame: np.ndarray,
        block_size: int,
        c_const: int,
        verbose: bool = False,
    ) -> dict[str, str | None]:
        """Read the pause-screen stat values.

        Args:
            frame: Downscaled ``(270, 480, 3)`` uint8 frame.
            block_size: Adaptive threshold block size for OCR.
            c_const: Threshold constant for OCR.
            verbose: When ``True``, log per-position match scores.

        Returns:
            Dict mapping stat names to string values, or ``None``
            for unreadable stats.
        """
        result: dict[str, str | None] = {}
        for i, name in enumerate(_STAT_NAMES):
            raw = self._stats_ocr(
                frame,
                x=_HUD_LEFT_X,
                y=_stat_y(name, i),
                block_size=block_size,
                c_const=c_const,
                verbose=verbose,
                label=f"stats {name}",
                **_stats_ocr_kwargs(name),
            )
            result[name] = str(raw) if raw is not None else None
        return result

    def _read_collectible(
        self,
        frame: np.ndarray,
        key: str,
        block_size: int,
        c_const: int,
        verbose: bool = False,
    ) -> int | None:
        """Read a single collectible count (coins / bombs / keys).

        Args:
            frame: Downscaled ``(270, 480, 3)`` uint8 frame.
            key: One of ``"coins"``, ``"bombs"``, ``"keys"``.
            block_size: Adaptive threshold block size for OCR.
            c_const: Threshold constant for OCR.
            verbose: When ``True``, log per-position match scores.

        Returns:
            Integer count, or ``None`` if unreadable.
        """
        result = self._collectibles_ocr(
            frame,
            x=_HUD_LEFT_X,
            y=_COLLECTIBLE_Y[key],
            block_size=block_size,
            c_const=c_const,
            verbose=verbose,
            label=f"collectible {key}",
        )
        return int(result) if result is not None else None


# ---------------------------------------------------------------------------
# Debug — overlay drawing
# ---------------------------------------------------------------------------


def _crosshair(img: np.ndarray, x: int, y: int, color: tuple[int, int, int]) -> None:
    """Draw a 3×3 plus-shaped crosshair centred at *(x, y)* in-place."""
    h, w = img.shape[:2]
    for dx in (-1, 0, 1):
        px = x + dx
        if 0 <= px < w:
            img[y, px] = color
    for dy in (-1, 0, 1):
        py = y + dy
        if 0 <= py < h:
            img[py, x] = color


def _rect(
    img: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: tuple[int, int, int],
) -> None:
    """Draw a 1-pixel-wide axis-aligned rectangle in-place."""
    h, w = img.shape[:2]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w - 1))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h - 1))
    img[y1, x1:x2] = color
    img[y2, x1:x2] = color
    img[y1:y2, x1] = color
    img[y1:y2, x2] = color


def _draw_debug_overlay(
    frame: np.ndarray,
    heart_cell_w: int = 16,
    heart_cell_h: int = 16,
) -> np.ndarray:
    """Return a BGR copy of *frame* with search regions drawn on it.

    Marks:
    - Stats rows: 3×3 red crosshairs at each ``(x, y)`` anchor.
    - Collectible rows: 3×3 blue crosshairs.
    - Heart grid: green 1 px rectangle around the 7×3 grid.
    - Charge bar: cyan 1 px rectangle.

    Args:
        frame: ``(H, W, 3)`` uint8 downscaled RGB frame.
        heart_cell_w: Heart cell width in pixels.
        heart_cell_h: Heart cell height in pixels.

    Returns:
        New ``(H, W, 3)`` uint8 BGR array with overlays, ready for
        ``cv2.imwrite``.
    """
    out = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # Stats — red crosshairs
    for i, name in enumerate(_STAT_NAMES):
        _crosshair(out, _HUD_LEFT_X, _stat_y(name, i), (0, 0, 255))

    # Collectibles — blue crosshairs
    for y in _COLLECTIBLE_Y.values():
        _crosshair(out, _HUD_LEFT_X, y, (255, 0, 0))

    # Heart grid — green rectangle
    _rect(
        out,
        _HEARTS_X,
        _HEARTS_Y,
        _HEARTS_X + 7 * heart_cell_w,
        _HEARTS_Y + 3 * heart_cell_h,
        (0, 255, 0),
    )

    # Charge bar — cyan rectangle
    _rect(
        out,
        _CHARGE_X,
        _CHARGE_Y,
        _CHARGE_X + _CHARGE_W,
        _CHARGE_Y + _CHARGE_H,
        (255, 255, 0),
    )

    return out


# ---------------------------------------------------------------------------
# Debug — crop export
# ---------------------------------------------------------------------------


def _export_glyphs(
    frame: np.ndarray,
    out_dir: Path,
    prefix: str,
    x: int,
    y: int,
    glyph_w: int,
    glyph_h: int,
    overlap: int,
    max_w: int,
    ocr: PixelDigitOCR | None,
    block_size: int,
    c_const: int,
    max_positions: int = 12,
    dot_first_after: int = 999,
    percent_eligible: bool = False,
) -> None:
    """Save per-glyph-position crops, plus mask/diff when OCR is available.

    Slides horizontally by ``glyph_w - overlap``, saving each crop
    as ``{prefix}{pos}.png``.  When *ocr* is provided, also saves
    ``_mask.png`` and ``_diff.png`` using the best-matching template.

    Args:
        dot_first_after: Enable dot-first matching for positions
            ``>=`` this index.
        percent_eligible: When ``True`` and the matched char is ``%``,
            stop scanning.
    """
    pos = 0
    step = glyph_w - overlap
    dot_found = False
    while x + glyph_w <= max_w and y + glyph_h <= frame.shape[0]:
        crop = frame[y : y + glyph_h, x : x + glyph_w]
        cv2.imwrite(
            str(out_dir / f"{prefix}{pos}.png"),
            cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
        )

        advance = step
        if ocr is not None:
            _label, tmpl, preprocessed = ocr.match_glyph_for_debug(
                crop,
                block_size,
                c_const,
                dot_first=(pos >= dot_first_after),
                skip_dot=dot_found,
            )
            if tmpl is not None and preprocessed is not None:
                advance = tmpl.shape[1] - overlap
                if ocr.narrow_one_adjust and _label == "1":
                    advance -= 1
                mask = tmpl[:, :, 3] > 0
                mask_img = np.where(mask[..., None], 255, 0).astype(np.uint8)
                cv2.imwrite(
                    str(out_dir / f"{prefix}{pos}_mask.png"),
                    cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR),
                )
                tmpl_f = tmpl[:, :, :3].astype(np.float64) / 255.0
                sub = preprocessed[:, : tmpl.shape[1], :]
                diff = np.abs(sub - tmpl_f)
                diff_amp = (diff * 3.0 * 255.0).clip(0, 255).astype(np.uint8)
                cv2.imwrite(
                    str(out_dir / f"{prefix}{pos}_diff.png"),
                    cv2.cvtColor(diff_amp, cv2.COLOR_RGB2BGR),
                )

                if _label == ".":
                    dot_found = True
                if percent_eligible and _label == "%":
                    break

        x += advance
        pos += 1
        if pos >= max_positions:
            break


def save_debug_dir(
    small: np.ndarray,
    output_dir: str | Path,
    extractor: HudExtractor,
    block_size: int = 11,
    c_const: int = 2,
) -> None:
    """Export debug crops and overlay from a downscaled RGB frame.

    Creates *output_dir* and populates::

        <output_dir>/
        ├── overlay.png
        ├── stats/          (crop + mask + diff per glyph position)
        ├── collectibles/    (crop + mask + diff per glyph position)
        ├── charge.png      (5×26)
        └── hearts/         (crop + mask + diff per cell)

    Each position also gets a ``_mask.png`` and a ``_diff.png`` showing
    the best-matching template's mask and the ``|crop − template| × 3``
    difference.

    Args:
        small: ``(270, 480, 3)`` uint8 downscaled RGB frame.
        output_dir: Path to the output directory (created if needed).
        extractor: The ``HudExtractor`` whose readers should be used
            for per-position mask/diff export and whose auto-detected
            heart cell sizes determine the overlay and grid layout.
        block_size: Adaptive threshold block size forwarded to OCR.
        c_const: Threshold constant forwarded to OCR.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    padded_h, padded_w = extractor.heart_cell_size
    native_h, native_w = extractor.heart_native_cell_size
    hearts_reader = extractor.heart_reader
    stats_ocr = extractor.stats_ocr
    collectibles_ocr = extractor.collectibles_ocr

    # Overlay
    overlay = _draw_debug_overlay(small, padded_w, padded_h)
    cv2.imwrite(str(output_dir / "overlay.png"), overlay)

    h, w = small.shape[:2]

    # -- collectibles --------------------------------------------------------
    c_dir = output_dir / "collectibles"
    c_dir.mkdir(parents=True, exist_ok=True)
    for name, y0 in _COLLECTIBLE_Y.items():
        _export_glyphs(
            small,
            c_dir,
            f"{name}_",
            _HUD_LEFT_X,
            y0,
            _COLLECTIBLE_GLYPH_W,
            _COLLECTIBLE_GLYPH_H,
            _GLYPH_OVERLAP,
            w,
            collectibles_ocr,
            block_size,
            c_const,
            max_positions=_COLLECTIBLE_MAX_DIGITS[name],
        )

    # -- stats ---------------------------------------------------------------
    s_dir = output_dir / "stats"
    s_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(_STAT_NAMES):
        _export_glyphs(
            small,
            s_dir,
            f"{name}_",
            _HUD_LEFT_X,
            _stat_y(name, i),
            _STATS_GLYPH_W,
            _STATS_GLYPH_H,
            _GLYPH_OVERLAP,
            w,
            stats_ocr,
            block_size,
            c_const,
            max_positions=6,
            dot_first_after=1,
            percent_eligible=(name in ("devilchance", "angelchance")),
        )

    # -- charge bar ----------------------------------------------------------
    x1 = max(0, _CHARGE_X)
    y1 = max(0, _CHARGE_Y)
    crop = small[
        y1 : min(y1 + _CHARGE_H, h),
        x1 : min(x1 + _CHARGE_W, w),
    ]
    cv2.imwrite(
        str(output_dir / "charge.png"),
        cv2.cvtColor(crop, cv2.COLOR_RGB2BGR),
    )

    # -- hearts --------------------------------------------------------------
    h_dir = output_dir / "hearts"
    h_dir.mkdir(parents=True, exist_ok=True)
    step_x = native_w - 1
    step_y = native_h - 1
    for row in range(3):
        oy = _HEARTS_Y + row * step_y
        if oy + padded_h > h:
            break
        for col in range(7):
            ox = _HEARTS_X + col * step_x
            if ox + padded_w > w:
                break
            cell = small[oy : oy + padded_h, ox : ox + padded_w]
            cv2.imwrite(
                str(h_dir / f"r{row}_c{col}.png"),
                cv2.cvtColor(cell, cv2.COLOR_RGB2BGR),
            )

            _state, best = hearts_reader.classify_cell_for_debug(cell)
            if best is not None:
                tmpl_rgb, tmpl_mask = best
                mask_img = np.where(tmpl_mask[..., None], 255, 0).astype(np.uint8)
                cv2.imwrite(
                    str(h_dir / f"r{row}_c{col}_mask.png"),
                    cv2.cvtColor(mask_img, cv2.COLOR_RGB2BGR),
                )
                diff = cv2.absdiff(cell, tmpl_rgb)
                diff_amp = cv2.convertScaleAbs(diff, alpha=3.0)
                cv2.imwrite(
                    str(h_dir / f"r{row}_c{col}_diff.png"),
                    cv2.cvtColor(diff_amp, cv2.COLOR_RGB2BGR),
                )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _print_state(state: HudState, json_out: bool = False) -> None:
    """Pretty-print a ``HudState`` to stdout."""
    if json_out:
        payload: dict = {
            "stats": state.stats,
            "bombs": state.bombs,
            "keys": state.keys,
            "coins": state.coins,
            "charge": state.charge,
            "hearts": [h.name for h in state.hearts],
        }
        print(json.dumps(payload, indent=2))
        return

    print("Stats:")
    for name, value in state.stats.items():
        print(f"  {name}: {value}")
    print(f"\nBombs:  {state.bombs}")
    print(f"Keys:   {state.keys}")
    print(f"Coins:  {state.coins}")
    print(f"Charge: {state.charge}")
    print(f"Hearts: {[h.name for h in state.hearts]}")


def main() -> None:
    """CLI entry point: extract HUD values from a screenshot."""
    parser = argparse.ArgumentParser(
        description="Extract HUD values from a Binding of Isaac screenshot."
    )
    parser.add_argument(
        "--image",
        required=True,
        help="Path to a 1920×1080 screenshot PNG.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON instead of pretty-printed text.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=11,
        help="Adaptive threshold block size for OCR (default: 11).",
    )
    parser.add_argument(
        "--c-const",
        type=int,
        default=2,
        help="Threshold constant for OCR (default: 2).",
    )
    parser.add_argument(
        "--debug-dir",
        type=str,
        default=None,
        metavar="DIR",
        help="Export debug crops and overlay into a directory.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Print per-position match scores to stderr.",
    )
    args = parser.parse_args()

    frame = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if frame is None:
        raise SystemExit(f"Could not read image: {args.image}")
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    hud = HudExtractor()
    state = hud.extract(
        frame,
        block_size=args.block_size,
        c_const=args.c_const,
        verbose=args.verbose,
    )
    _print_state(state, json_out=args.json)

    if args.debug_dir:
        small = hud.downscale(frame)
        save_debug_dir(
            small,
            args.debug_dir,
            hud,
            block_size=args.block_size,
            c_const=args.c_const,
        )
        print(f"\nDebug export saved to {args.debug_dir}/")


if __name__ == "__main__":
    main()
