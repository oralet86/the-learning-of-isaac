"""Pixel-font digit OCR and charge-bar reading for in-game HUD elements.

OCR — adaptive thresholding + alpha-masked similarity
---------------------------------------------------
Reads digit glyph templates (RGBA PNGs) from a directory, then slides
right across a region of the frame, matching each position against the
template bank using :func:`masked_similarity`.  Templates define their
own mask via the alpha channel; only opaque pixels are compared.
Returns the parsed integer or float.

Charge bar — bottom-up HSV hue scan
-----------------------------------
Scans a vertical bar region row-by-row from the bottom.  Green fills
first; orange overlays from the bottom only once green reaches 100 %.
Returns a float in ``[0, 2]`` representing the total charge.

Heart grid — alpha-masked template matching
-------------------------------------------
Scans a 7×3 grid of 16×16 heart cells.  For each cell, compares the
opaquely-masked RGB pixels against pre-loaded heart sprites and
precomputed eternal / gold overlay composites.  Returns a list of
:class:`HeartState` with the matched template name.

Usage::

    from live.ocr_2 import PixelDigitOCR, ChargeBarReader, HeartReader

    ocr = PixelDigitOCR("data/templates/collectibles")
    bombs = ocr(frame, x=1450, y=90, block_size=11, c_const=2)  # → 7

    ocr_stats = PixelDigitOCR("data/templates/stats", output_str=True)
    strength = ocr_stats(frame, x=200, y=300, block_size=9, c_const=3)  # → "2.73"

    bar = ChargeBarReader()
    charge = bar(frame, x=700, y=80, width=14, height=100)  # → 1.5

    hearts_reader = HeartReader("data/templates/hearts")
    hearts = hearts_reader(frame, x=50, y=500)  # → [HeartState(...), ...]
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from sklearn.cluster import KMeans

_ADAPTIVE_METHODS: dict[str, int] = {
    "mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    "gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
}

# Above this white-pixel fraction, a preprocessed glyph box is treated as
# blank background rather than a real glyph. Adaptive thresholding can flip
# a near-uniform blank patch almost entirely white; real glyphs in this font
# stay well under this (~0.15-0.40 white), so the gap is wide.
_BLANK_WHITE_FRACTION = 0.55


def masked_similarity(
    crop_rgb: np.ndarray,
    template_rgba: np.ndarray,
) -> float:
    """Compute similarity between an RGB crop and an RGBA template.

    Only pixels where the template alpha is non-zero are compared;
    transparent regions are excluded.  Returns a score in ``[0, 1]``
    where 1 indicates a perfect match.

    Args:
        crop_rgb: ``(H, W, 3)`` uint8 RGB image region.
        template_rgba: ``(H, W, 4)`` uint8 RGBA template.  The alpha
            channel defines the mask: pixels with ``alpha > 0`` are
            compared, others are ignored.

    Returns:
        Similarity score ``1 / (1 + RME)`` over opaque pixels, where
        RME is root mean error: ``mean(sqrt(|crop − template|))``.
        Returns 0 if the template is fully transparent.

    Raises:
        ValueError: If ``crop_rgb`` and ``template_rgba`` have
            different spatial dimensions.
    """
    if crop_rgb.shape[:2] != template_rgba.shape[:2]:
        raise ValueError(
            f"Shape mismatch: crop {crop_rgb.shape[:2]} vs "
            f"template {template_rgba.shape[:2]}"
        )
    mask = template_rgba[:, :, 3] > 0
    if not mask.any():
        return 0.0
    tmpl_rgb = template_rgba[:, :, :3]
    crop_opaque = crop_rgb[mask].astype(np.float64)
    tmpl_opaque = tmpl_rgb[mask].astype(np.float64)
    rme = float(np.mean(np.sqrt(np.abs(crop_opaque - tmpl_opaque))))
    return 1.0 / (1.0 + rme)


def _preprocess_crop(
    crop: np.ndarray,
    block_size: int = 11,
    c_const: int = 2,
    method: str = "mean",
) -> np.ndarray:
    """Preprocess a raw glyph crop for similarity matching.

    Converts to grayscale, applies adaptive thresholding, and
    normalizes to a ``(H, W, 3)`` float64 array in ``[0, 1]``.

    Args:
        crop: ``(H, W)`` or ``(H, W, 3)`` uint8 image region.
        block_size: Adaptive threshold block size.
        c_const: Constant subtracted from threshold mean.
        method: ``"mean"`` or ``"gaussian"``.

    Returns:
        ``(H, W, 3)`` float64 array with values in ``[0, 1]``.
    """
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)

    adaptive_method = _ADAPTIVE_METHODS[method]
    binary = cv2.adaptiveThreshold(
        crop, 255, adaptive_method, cv2.THRESH_BINARY, block_size, c_const
    )
    binary_rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
    return binary_rgb.astype(np.float64) / 255.0


class PixelDigitOCR:
    """OCR for fixed-width pixel-font digits via alpha-masked similarity.

    On construction, loads ``{digit}.png``, ``dot.png``,
    ``percent.png``, and ``minus.png`` templates from *template_dir*
    in RGBA format.  The alpha channel defines the mask — opaque
    pixels are compared, transparent pixels are ignored.

    At runtime, crops a window at *(x, y)*, applies adaptive
    thresholding, and scores each template via :func:`masked_similarity`.
    Advances horizontally by the matched glyph's width minus overlap
    and repeats until no template exceeds *match_threshold*.

    Args:
        template_dir: Directory containing ``0.png`` … ``9.png`` and
            optionally ``dot.png``, ``percent.png``, ``minus.png``.
        match_threshold: Minimum similarity score (``0.0`` – ``1.0``)
            to accept a digit match.
        output_str: If ``True``, ``__call__`` returns the joined
            character string (e.g. ``"2.73"``, ``"42%"``) instead of
            a parsed ``int`` or ``float``.
        overlap: Number of pixels by which adjacent glyphs overlap.
            The horizontal advance per match is ``glyph_width - overlap``.
        pattern: Optional regex string that the final OCR result must
            fully match (e.g. ``r"\\d+\\.\\d{2}%?"``).  When set and
            the result does not match, ``__call__`` returns ``None``.
        white_only_mask: When ``True``, templates are normalized so
            only their white foreground pixels are opaque (compared)
            — any black border is excluded from the mask entirely.
            Use this when the live glyphs render semi-transparently,
            making a black border unreliable while the white interior
            stays distinctive.

    Raises:
        FileNotFoundError: If *template_dir* does not exist or contains
            no digit templates.
    """

    def __init__(
        self,
        template_dir: str | Path,
        match_threshold: float = 0.90,
        output_str: bool = False,
        overlap: int = 0,
        pattern: str | None = None,
        narrow_one_adjust: bool = False,
        white_only_mask: bool = False,
    ) -> None:
        self._match_threshold = match_threshold
        self._output_str = output_str
        self._overlap = overlap
        self._narrow_one_adjust = narrow_one_adjust
        self._white_only_mask = white_only_mask
        self._pattern: re.Pattern[str] | None = (
            None if pattern is None else re.compile(pattern)
        )
        self._templates: list[tuple[str, np.ndarray, int]] = []
        self._glyph_h = 0
        self._glyph_w = 0

        path = Path(template_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Template directory not found: {path}")

        for digit in range(10):
            p = path / f"{digit}.png"
            if not p.exists():
                continue
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            tmpl = self._normalize_template(raw, self._white_only_mask)
            h, w = tmpl.shape[:2]
            if self._glyph_w == 0:
                self._glyph_h, self._glyph_w = h, w
            self._templates.append((str(digit), tmpl, w))

        for glyph_name, label in [("dot", "."), ("percent", "%"), ("minus", "-")]:
            p = path / f"{glyph_name}.png"
            if not p.exists():
                continue
            raw = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if raw is None:
                continue
            tmpl = self._normalize_template(raw, self._white_only_mask)
            h, w = tmpl.shape[:2]
            if self._glyph_w == 0:
                self._glyph_h, self._glyph_w = h, w
            self._templates.append((label, tmpl, w))

        if not self._templates:
            raise FileNotFoundError(
                f"No digit templates (0.png … 9.png) found in {path}"
            )

    @property
    def glyph_size(self) -> tuple[int, int]:
        """``(height, width)`` of the digit glyphs."""
        return (self._glyph_h, self._glyph_w)

    @property
    def narrow_one_adjust(self) -> bool:
        """Whether the advance after a matched ``"1"`` is shortened by 1px."""
        return self._narrow_one_adjust

    # -- public API -------------------------------------------------------

    def __call__(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        block_size: int = 11,
        c_const: int = 2,
        method: str = "mean",
        verbose: bool = False,
        label: str = "",
        percent_eligible: bool = False,
        override_pattern: str | None = None,
        allow_minus: bool = False,
    ) -> int | float | str | None:
        """OCR the digits starting at *(x, y)* in *frame*.

        Args:
            frame: ``(H, W, 3)`` RGB frame (uint8 or fp16).
            x: Top-left x-coordinate of the first digit glyph.
            y: Top-left y-coordinate of the first digit glyph.
            block_size: Adaptive threshold block size (odd).
            c_const: Constant subtracted from the mean.
            method: ``"mean"`` or ``"gaussian"``.
            verbose: When ``True``, print per-position match scores
                to stderr.
            label: Human-readable prefix for verbose output (e.g.
                ``"stats speed"``).
            percent_eligible: When ``True`` and ``%`` is matched,
                stop scanning (``%`` is always the last symbol).
            override_pattern: Optional regex that overrides the
                instance-level pattern for this call.
            allow_minus: When ``True``, permit a leading ``-`` match.

        Returns:
            Parsed ``int`` (no decimal point found), ``float``
            (decimal point present), ``str`` when ``output_str=True``
            and the template bank contains non-numeric symbols, or
            ``None`` if no digit matched.

            When ``output_str=True`` the result is always ``str | None``;
            callers that construct this OCR with ``output_str=True``
            should not expect numeric return types.
        """
        if frame.dtype == np.float16:
            frame = frame.astype(np.uint8)

        h_frame, w_frame = frame.shape[:2]
        gh = self._glyph_h

        digits: list[str] = []
        dot_found = False

        while y + gh <= h_frame:
            skip_chars: set[str] = set()
            if dot_found:
                skip_chars.add(".")
            if not allow_minus:
                skip_chars.add("-")
            if not percent_eligible:
                skip_chars.add("%")

            best_char, best_score, best_width, _tmpl, _pre, all_scores = (
                self._score_position(
                    frame[y : y + gh, x:w_frame],
                    block_size,
                    c_const,
                    method,
                    skip_chars=frozenset(skip_chars),
                )
            )

            self._log_verbose_position(
                verbose, label, x, y, best_char, best_score, all_scores
            )

            if best_score < self._match_threshold:
                break

            digits.append(best_char)
            if best_char == ".":
                dot_found = True
            if percent_eligible and best_char == "%":
                break
            x += best_width - self._overlap
            if self._narrow_one_adjust and best_char == "1":
                x -= 1

        if not digits:
            return None

        text = "".join(digits)
        prefix = f"[{label}]" if label else ""
        if verbose:
            print(f'{prefix} result: "{text}"', file=sys.stderr, flush=True)

        pattern = self._resolve_pattern(override_pattern)
        if pattern is not None and not pattern.fullmatch(text):
            if verbose:
                print(
                    f'{prefix} pattern mismatch: "{text}" rejected by '
                    f"{pattern.pattern}",
                    file=sys.stderr,
                    flush=True,
                )
            return None
        if pattern is not None and verbose:
            print(
                f'{prefix} pattern OK: "{text}" matches {pattern.pattern}',
                file=sys.stderr,
                flush=True,
            )

        return self._parse_result(text)

    def match_glyph_for_debug(
        self,
        raw_crop: np.ndarray,
        block_size: int = 11,
        c_const: int = 2,
        method: str = "mean",
        dot_first: bool = False,
        skip_dot: bool = False,
    ) -> tuple[str | None, np.ndarray | None, np.ndarray | None]:
        """Match a single glyph crop against all templates, for debug export.

        Sub-crops *raw_crop* to each template's width, preprocesses,
        and scores via :meth:`_score_position` — the same logic used
        by :meth:`__call__`'s sliding window. Returns the best-match
        label, template, and preprocessed sub-crop for visual export.

        Args:
            raw_crop: ``(H, W)`` grayscale or ``(H, W, 3)`` RGB uint8
                crop of a single glyph region.
            block_size: Adaptive threshold block size.
            c_const: Constant subtracted from threshold mean.
            method: ``"mean"`` or ``"gaussian"``.
            dot_first: When ``True``, try ``%`` and ``.`` before digits.
            skip_dot: When ``True``, exclude ``.`` from candidates
                entirely (only one dot per row).

        Returns:
            ``(label, template_rgba, preprocessed_sub_crop)`` or
            ``(None, None, None)`` when no template fits.
        """
        skip_chars = {"."} if skip_dot else set()
        prefer_chars = ("%", ".") if dot_first and not skip_dot else ()

        best_char, _score, _width, best_tmpl, best_pre, _all_scores = (
            self._score_position(
                raw_crop,
                block_size,
                c_const,
                method,
                skip_chars=frozenset(skip_chars),
                prefer_chars=prefer_chars,
                require_exact_height=True,
            )
        )
        if best_tmpl is None:
            return None, None, None
        return best_char, best_tmpl, best_pre

    # -- shared scoring core ------------------------------------------------

    def _score_position(
        self,
        raw_crop: np.ndarray,
        block_size: int,
        c_const: int,
        method: str,
        *,
        skip_chars: frozenset[str] = frozenset(),
        prefer_chars: tuple[str, ...] = (),
        require_exact_height: bool = False,
    ) -> tuple[
        str, float, int, np.ndarray | None, np.ndarray | None, list[tuple[str, float]]
    ]:
        """Score every candidate template against *raw_crop*.

        Shared core used by both :meth:`__call__` and
        :meth:`match_glyph_for_debug`: sub-crops *raw_crop* to each
        template's width, preprocesses, and scores with
        :func:`masked_similarity`, tracking the best match.

        Args:
            raw_crop: Crop wide enough to contain the widest candidate
                template, starting at the glyph's top-left corner.
            block_size: Adaptive threshold block size.
            c_const: Constant subtracted from threshold mean.
            method: ``"mean"`` or ``"gaussian"``.
            skip_chars: Character labels to exclude from candidates.
            prefer_chars: When non-empty, candidates are tried in this
                order first (ties still resolved by score); used by
                the debug path to mirror ``dot_first`` ordering.
            require_exact_height: When ``True``, only consider
                templates whose height exactly matches *raw_crop*
                (used for debug export, where the crop is a fixed
                single-glyph cell rather than a full sliding window).

        Returns:
            ``(best_char, best_score, best_width, best_template_rgba,
            best_preprocessed_crop, all_scores)``. ``best_char`` is
            ``""`` and ``best_template_rgba``/``best_preprocessed_crop``
            are ``None`` when no candidate fit inside *raw_crop*.
            ``all_scores`` holds every candidate's ``(char, score)``,
            sorted descending by score, for verbose diagnostics.
        """
        if self._white_only_mask and self._is_blank_region(
            raw_crop, block_size, c_const, method
        ):
            return "", -1.0, 0, None, None, []

        candidates = [
            (char, tmpl, tw)
            for char, tmpl, tw in self._templates
            if char not in skip_chars
        ]
        if prefer_chars:
            order = {char: i for i, char in enumerate(prefer_chars)}
            candidates.sort(key=lambda c: order.get(c[0], len(prefer_chars)))

        crop_h = raw_crop.shape[0]
        best_char = ""
        best_score = -1.0
        best_width = 0
        best_tmpl: np.ndarray | None = None
        best_pre: np.ndarray | None = None
        all_scores: list[tuple[str, float]] = []

        for char, tmpl, tw in candidates:
            th = tmpl.shape[0]
            if require_exact_height and th != crop_h:
                continue
            if tw > raw_crop.shape[1]:
                continue

            crop = raw_crop[:, :tw]
            crop_f = _preprocess_crop(crop, block_size, c_const, method)
            tmpl_f = tmpl.astype(np.float64) / 255.0
            score = masked_similarity(crop_f, tmpl_f)
            all_scores.append((char, score))

            if score > best_score:
                best_score = score
                best_char = char
                best_width = tw
                best_tmpl = tmpl
                best_pre = crop_f

        all_scores.sort(key=lambda s: s[1], reverse=True)
        return best_char, best_score, best_width, best_tmpl, best_pre, all_scores

    def _is_blank_region(
        self,
        raw_crop: np.ndarray,
        block_size: int,
        c_const: int,
        method: str,
    ) -> bool:
        """Detect a blank background patch masquerading as a glyph.

        Only used when ``white_only_mask`` is set: with the black
        border excluded from template masks, a blank patch that
        adaptive thresholding flips mostly white would otherwise
        trivially match every digit's white-only mask. Real glyphs in
        this font stay well under :data:`_BLANK_WHITE_FRACTION` white;
        a patch above it is treated as background.

        Args:
            raw_crop: Crop at least :attr:`_glyph_w` wide, starting at
                the glyph's top-left corner.
            block_size: Adaptive threshold block size.
            c_const: Constant subtracted from threshold mean.
            method: ``"mean"`` or ``"gaussian"``.

        Returns:
            ``True`` if the glyph-sized box is mostly white after
            preprocessing.
        """
        w = min(self._glyph_w, raw_crop.shape[1])
        if w == 0:
            return False
        box = _preprocess_crop(raw_crop[:, :w], block_size, c_const, method)
        white_fraction = float((box[:, :, 0] > 0.5).mean())
        return white_fraction >= _BLANK_WHITE_FRACTION

    def _log_verbose_position(
        self,
        verbose: bool,
        label: str,
        x: int,
        y: int,
        best_char: str,
        best_score: float,
        all_scores: list[tuple[str, float]],
    ) -> None:
        """Print a per-position match summary to stderr when *verbose*."""
        if not verbose:
            return
        prefix = f"[{label}]" if label else ""
        pos_str = f"pos=({x},{y})"
        top5 = "  ".join(f"{c}={s:.3f}" for c, s in all_scores[:5])
        if best_score >= self._match_threshold:
            print(
                f'{prefix} {pos_str}  -> "{best_char}" score={best_score:.3f}  '
                f"top5: {top5}",
                file=sys.stderr,
            )
        else:
            print(
                f"{prefix} {pos_str}  NO MATCH "
                f'(best="{best_char}"={best_score:.3f} < '
                f"{self._match_threshold:.2f})  top5: {top5}",
                file=sys.stderr,
            )

    def _resolve_pattern(self, override: str | None) -> re.Pattern[str] | None:
        """Return the compiled pattern to validate against, if any.

        Args:
            override: Per-call pattern string that takes precedence
                over the instance-level pattern set at construction.

        Returns:
            A compiled pattern, or ``None`` if neither an override nor
            an instance-level pattern is set.
        """
        if override is not None:
            return re.compile(override)
        return self._pattern

    def _parse_result(self, text: str) -> int | float | str | None:
        """Convert the collected glyph string to its final return type.

        Args:
            text: Concatenated matched glyph characters.

        Returns:
            ``text`` unchanged when ``output_str=True``; otherwise a
            parsed ``float`` (decimal point present), ``int``, or
            ``None`` if parsing fails.
        """
        if self._output_str:
            return text
        try:
            return float(text) if "." in text else int(text)
        except ValueError:
            return None

    # -- internals --------------------------------------------------------

    @staticmethod
    def _normalize_template(raw: np.ndarray, white_only_mask: bool = False) -> np.ndarray:
        """Normalize a template image to canonical RGBA format.

        Digit pixels become opaque white ``(255, 255, 255, 255)``;
        background pixels become transparent ``(0, 0, 0, 0)``.

        Handles grayscale, RGB, and RGBA inputs.  For RGBA input the
        original alpha channel is preserved (unless *white_only_mask*
        is set) and the RGB values within opaque regions are
        binarized / inverted as needed.

        Args:
            raw: Template image in grayscale, RGB, or RGBA format.
            white_only_mask: When ``True``, only the white foreground
                pixels are kept opaque — any black border (or other
                non-foreground pixel within the original alpha mask)
                becomes transparent instead of opaque black.
        """
        if raw.ndim == 2:
            _, binary = cv2.threshold(raw, 127, 255, cv2.THRESH_BINARY)
            if np.mean(binary) > 127:
                binary = cv2.bitwise_not(binary)
            rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
            rgba[:, :, :3][binary > 0] = 255
            rgba[:, :, 3][binary > 0] = 255
            return rgba

        if raw.shape[2] == 4:
            alpha = raw[:, :, 3]
            mask = alpha > 0
            gray = cv2.cvtColor(raw[:, :, :3], cv2.COLOR_RGB2GRAY)
            rgba = np.zeros_like(raw)
            if not white_only_mask:
                rgba[:, :, 3] = alpha
            if mask.any():
                binary = gray[mask] > 127
                if binary.mean() > 0.5:
                    binary = ~binary
                rgba[mask, :3] = np.where(binary[:, None], 255, 0).astype(np.uint8)
                if white_only_mask:
                    rgba[:, :, 3][mask] = np.where(binary, 255, 0).astype(np.uint8)
            return rgba

        gray = cv2.cvtColor(raw, cv2.COLOR_RGB2GRAY)
        return PixelDigitOCR._normalize_template(gray, white_only_mask)


class ChargeBarReader:
    """Bottom-up vertical charge-bar reader via HSV hue scanning.

    Scans a vertical bar row-by-row from the bottom.  Green fills first
    (0–100 % charge); orange overlays from the bottom only after green
    is complete (100–200 %).  The bar may be divided into up to 12
    segments by 1px black horizontal lines (e.g. a charge-3 item shows
    2 dividers); divider rows are skipped rather than treated as the
    end of the fill.  The bar's on-screen position can drift slightly
    frame to frame, so its true vertical bounds are auto-detected
    within a buffered search window rather than trusted from the
    caller's nominal ``y``/``height``.

    Returns a single ``float`` in ``[0, 2]`` — *e.g.* ``0.5`` for half
    green, ``1.0`` for full green, ``1.5`` for full green + half orange.

    Args:
        green_hue_range: ``(lower, upper)`` OpenCV hue boundaries
            (0–179) for the green fill colour.
        orange_hue_range: OpenCV hue boundaries for the orange overlay.
        saturation_min: Minimum saturation (0–255) for a pixel to be
            considered coloured (ignores desaturated / grey background).
        row_fill_threshold: Fraction of pixels in a row that must match
            the colour (or be near-black, for divider detection) for
            the row to count as that category.
        y_buffer: Extra rows searched above and below the caller's
            nominal ``(y, height)`` region when locating the bar's true
            vertical bounds, to tolerate the bar's position drifting.
        track_value_max: Maximum RGB channel value for a pixel to be
            considered part of the near-black border/divider track.

    Usage::

        reader = ChargeBarReader(
            green_hue_range=(35, 85),
            orange_hue_range=(5, 25),
        )
        charge = reader(frame, x=700, y=80, width=14, height=100)  # → 1.5
    """

    def __init__(
        self,
        green_hue_range: tuple[int, int] = (35, 85),
        orange_hue_range: tuple[int, int] = (5, 25),
        saturation_min: int = 40,
        row_fill_threshold: float = 0.3,
        y_buffer: int = 10,
        track_value_max: int = 30,
    ) -> None:
        self._green_lo, self._green_hi = green_hue_range
        self._orange_lo, self._orange_hi = orange_hue_range
        self._sat_min = saturation_min
        self._row_fill_threshold = row_fill_threshold
        self._y_buffer = y_buffer
        self._track_value_max = track_value_max

    def __call__(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
    ) -> float:
        """Read the charge level from a vertical bar near *(x, y)*.

        Args:
            frame: ``(H, W, 3)`` RGB frame (uint8 or fp16).
            x: Nominal top-left x-coordinate of the bar.
            y: Nominal top-left y-coordinate of the bar; the bar's
                true vertical position is auto-detected within
                ``y_buffer`` rows of this value.
            width: Bar width in pixels.
            height: Nominal bar height in pixels, used to size the
                vertical search window.

        Returns:
            Charge level in ``[0, 2]``.  ``0`` = empty, ``1`` = full
            green, ``2`` = full green + full orange.  ``0`` is also
            returned when the bar cannot be located.
        """
        if frame.dtype == np.float16:
            frame = frame.astype(np.uint8)

        h_frame, w_frame = frame.shape[:2]
        if x < 1 or x + width + 1 > w_frame:
            return 0.0

        search_top = max(0, y - self._y_buffer)
        search_bottom = min(h_frame, y + height + self._y_buffer)
        if search_bottom <= search_top:
            return 0.0

        border_cols = frame[search_top:search_bottom, [x - 1, x + width]]
        is_track_row = np.all(border_cols.max(axis=-1) <= self._track_value_max, axis=-1)

        bar_top, bar_bottom = self._longest_run(is_track_row)
        if bar_top is None:
            return 0.0
        bar_top += search_top
        bar_bottom += search_top

        interior = frame[bar_top : bar_bottom + 1, x : x + width]
        hsv = cv2.cvtColor(interior, cv2.COLOR_RGB2HSV)
        sat_mask = hsv[:, :, 1] >= self._sat_min

        green_pixels = (
            (hsv[:, :, 0] >= self._green_lo)
            & (hsv[:, :, 0] <= self._green_hi)
            & sat_mask
        )
        orange_pixels = (
            (hsv[:, :, 0] >= self._orange_lo)
            & (hsv[:, :, 0] <= self._orange_hi)
            & sat_mask
        )
        divider_pixels = interior.max(axis=-1) <= self._track_value_max

        pixel_threshold = int(width * self._row_fill_threshold)
        is_divider_row = divider_pixels.sum(axis=-1) >= pixel_threshold

        total_rows = int((~is_divider_row).sum())
        if total_rows == 0:
            return 0.0

        visible_green_rows = 0
        orange_rows = 0
        for row in range(len(is_divider_row) - 1, -1, -1):
            if is_divider_row[row]:
                continue

            g_count = int(green_pixels[row].sum())
            o_count = int(orange_pixels[row].sum())

            if o_count >= pixel_threshold:
                orange_rows += 1
            elif g_count >= pixel_threshold:
                visible_green_rows += 1
            else:
                break

        charge = (visible_green_rows + 2 * orange_rows) / total_rows
        return min(charge, 2.0)

    @staticmethod
    def _longest_run(mask: np.ndarray) -> tuple[int | None, int | None]:
        """Return the ``(start, end)`` indices of the longest run of ``True``.

        Args:
            mask: 1-D boolean array.

        Returns:
            Inclusive ``(start, end)`` indices of the longest contiguous
            run of ``True`` values, or ``(None, None)`` if *mask* is
            all ``False``.
        """
        best_start, best_len = None, 0
        run_start, run_len = None, 0
        for i, v in enumerate(mask):
            if v:
                if run_start is None:
                    run_start = i
                run_len += 1
                if run_len > best_len:
                    best_start, best_len = run_start, run_len
            else:
                run_start, run_len = None, 0
        if best_start is None:
            return None, None
        return best_start, best_start + best_len - 1


# ---------------------------------------------------------------------------
# Heart grid reader — template-based
# ---------------------------------------------------------------------------

_SKIP_FILES = frozenset({"ui_hearts.png", ".gitkeep"})
_OVERLAY_NAMES = frozenset({"eternal_overlay_1", "gold_overlay_1"})

# unknown.png (Curse of the Unknown) renders 1px higher in-game than a
# normal heart cell's top-left origin.
_UNKNOWN_Y_OFFSET = 1


@dataclass
class HeartState:
    """A single heart cell classification result.

    Attributes:
        name: Matched template name, *e.g.* ``"red_2"``,
            ``"eternal_red_2"``, ``"bone_rotten_2"``, or
            ``"unknown"`` for Curse of the Unknown.
    """

    name: str


class HeartReader:
    """Grid-based heart-row detector using alpha-masked template matching.

    Loads 16×16 RGBA heart sprites from *template_dir*, precompiles
    eternal and gold overlay composites for every base type, then scans
    a 7×3 grid of cells from a live frame and reports the best-matching
    template for each occupied cell.

    The first cell is checked against ``unknown.png`` for Curse of the
    Unknown.  If matched, no further cells are classified.

    Args:
        template_dir: Directory containing ``red_0.png`` … etc.,
            ``eternal_overlay_1.png``, ``gold_overlay_1.png``,
            and ``unknown.png``.
        match_threshold: Minimum pixel similarity score (0–1) to
            accept a match.
        overlap: Number of pixels by which adjacent heart cells
            overlap.  Grid step is ``cell_size - overlap``.
        palette_colors: Number of colours to quantize templates and
            live cells to before matching (0 disables quantization).

    Usage::

        reader = HeartReader("data/templates/hearts")
        hearts = reader(frame, x=50, y=500)
        # → [HeartState(name="red_2"), HeartState(name="blue_1"), ...]
    """

    def __init__(
        self,
        template_dir: str | Path,
        match_threshold: float = 0.70,
        overlap: int = 0,
        palette_colors: int = 10,
    ) -> None:
        self._threshold = match_threshold
        self._overlap = overlap
        self._palette: np.ndarray | None = None

        path = Path(template_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Template directory not found: {path}")

        raw_unknown, raw_eternal, raw_gold = None, None, None
        raw_bases: dict[str, np.ndarray] = {}

        unknown_path = path / "unknown.png"
        if unknown_path.exists():
            raw_unknown = self._load_rgba(unknown_path)

        eternal_path = path / "eternal_overlay_1.png"
        if eternal_path.exists():
            raw_eternal = self._load_rgba(eternal_path)

        gold_path = path / "gold_overlay_1.png"
        if gold_path.exists():
            raw_gold = self._load_rgba(gold_path)

        for p in sorted(path.glob("*.png")):
            name = p.stem
            if p.name in _SKIP_FILES or name in _OVERLAY_NAMES or name == "unknown":
                continue
            raw_bases[name] = self._load_rgba(p)

        if not raw_bases:
            raise FileNotFoundError(f"No base heart templates found in {path}")

        # Padded cell size — max across all templates (base + overlays + unknown).
        all_raws: list[np.ndarray] = list(raw_bases.values())
        for r in (raw_unknown, raw_eternal, raw_gold):
            if r is not None:
                all_raws.append(r)
        self._cell_h = max(r.shape[0] for r in all_raws)
        self._cell_w = max(r.shape[1] for r in all_raws)

        # Native cell size — mode of base heart dimensions (actual footprint
        # in the game frame, before padding).  Used for grid stepping so
        # the rolling window does not drift.
        sizes = Counter(r.shape[:2] for r in raw_bases.values())
        (self._native_h, self._native_w), _ = sizes.most_common(1)[0]

        if raw_unknown is not None:
            padded = self._pad_rgba(raw_unknown)
            self._unknown: tuple[np.ndarray, np.ndarray] | None = (
                padded[:, :, :3],
                padded[:, :, 3] > 0,
            )
        else:
            self._unknown = None

        if raw_eternal is not None:
            padded = self._pad_rgba(raw_eternal)
            self._eternal_overlay: tuple[np.ndarray, np.ndarray] | None = (
                padded[:, :, :3],
                padded[:, :, 3] > 0,
            )
        else:
            self._eternal_overlay = None

        if raw_gold is not None:
            padded = self._pad_rgba(raw_gold)
            self._gold_overlay: tuple[np.ndarray, np.ndarray] | None = (
                padded[:, :, :3],
                padded[:, :, 3] > 0,
            )
        else:
            self._gold_overlay = None

        self._base_templates: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, rgba in raw_bases.items():
            padded = self._pad_rgba(rgba)
            self._base_templates[name] = (
                padded[:, :, :3],
                padded[:, :, 3] > 0,
            )

        if palette_colors > 0:
            self._build_palette(palette_colors)
            self._quantize_base_and_overlays()
            self._eternal_composites = self._build_composites(
                self._eternal_overlay, "eternal_"
            )
            self._gold_composites = self._build_composites(self._gold_overlay, "gold_")
            self._quantize_composites()
        else:
            self._eternal_composites = self._build_composites(
                self._eternal_overlay, "eternal_"
            )
            self._gold_composites = self._build_composites(self._gold_overlay, "gold_")

    @property
    def cell_size(self) -> tuple[int, int]:
        """``(height, width)`` of padded heart cells (max template size)."""
        return (self._cell_h, self._cell_w)

    @property
    def native_cell_size(self) -> tuple[int, int]:
        """``(height, width)`` of native heart sprites in the game frame."""
        return (self._native_h, self._native_w)

    def __call__(
        self,
        frame: np.ndarray,
        x: int,
        y: int,
        verbose: bool = False,
    ) -> list[HeartState]:
        """Scan the 7×3 heart grid at *(x, y)* in *frame*.

        Uses the cell size auto-detected from loaded templates
        (padded to the largest template dimensions).

        Args:
            frame: ``(H, W, 3)`` RGB frame (uint8 or fp16).
            x: Top-left x-coordinate of the grid.
            y: Top-left y-coordinate of the grid.
            verbose: When ``True``, print per-cell match details to
                stderr.

        Returns:
            List of :class:`HeartState` objects in row-major order
            (left-to-right, top-to-bottom), excluding empty cells.
        """
        if frame.dtype == np.float16:
            frame = frame.astype(np.uint8)

        pad_w, pad_h = self._cell_w, self._cell_h
        native_w, native_h = self._native_w, self._native_h

        grid_w = 7 * pad_w
        grid_h = 3 * pad_h
        h_frame, w_frame = frame.shape[:2]

        grid_h = min(grid_h, h_frame - y)
        grid_w = min(grid_w, w_frame - x)
        if grid_h <= 0 or grid_w <= 0:
            return []

        crop = frame[y : y + grid_h, x : x + grid_w]
        crop_rgb = crop if crop.ndim == 2 else crop[:, :, :3]

        if self._unknown is not None:
            uy = y - _UNKNOWN_Y_OFFSET
            frame_rgb = frame if frame.ndim == 2 else frame[:, :, :3]
            if uy >= 0 and uy + pad_h <= h_frame and x + pad_w <= w_frame:
                first_cell = self._maybe_quantize(frame_rgb[uy : uy + pad_h, x : x + pad_w])
                if self._masked_score(first_cell, *self._unknown) >= self._threshold:
                    return [HeartState(name="unknown")]

        result: list[HeartState] = []
        step_x = native_w - self._overlap
        step_y = native_h - self._overlap
        for row in range(3):
            row_has = False
            off_y = row * step_y
            if off_y + pad_h > crop_rgb.shape[0]:
                break
            for col in range(7):
                off_x = col * step_x
                if off_x + pad_w > crop_rgb.shape[1]:
                    break
                cell = self._maybe_quantize(
                    crop_rgb[off_y : off_y + pad_h, off_x : off_x + pad_w]
                )
                state, _tmpl = self._classify_cell(
                    cell,
                    verbose=verbose,
                    row=row,
                    col=col,
                    abs_x=x + off_x,
                    abs_y=y + off_y,
                )
                if state is not None:
                    result.append(state)
                    row_has = True
            if not row_has and row > 0:
                break
        return result

    def classify_cell_for_debug(
        self, cell_rgb: np.ndarray
    ) -> tuple[HeartState | None, tuple[np.ndarray, np.ndarray] | None]:
        """Classify a single heart cell, exposing the matched template.

        Thin public wrapper around :meth:`_classify_cell` for callers
        outside this class (debug crop / mask / diff export) that need
        both the classification and the matched template data.

        Args:
            cell_rgb: ``(cell_h, cell_w, 3)`` uint8 RGB heart cell,
                already padded to :attr:`cell_size`.

        Returns:
            Same as :meth:`_classify_cell`: ``(state, template)``.
        """
        return self._classify_cell(cell_rgb)

    # -- per-cell classification ------------------------------------------

    def _classify_cell(
        self,
        cell_rgb: np.ndarray,
        verbose: bool = False,
        row: int = 0,
        col: int = 0,
        abs_x: int = 0,
        abs_y: int = 0,
    ) -> tuple[HeartState | None, tuple[np.ndarray, np.ndarray] | None]:
        """Match a single cell against the template bank.

        Detection order:
        1. Try eternal overlay → if matched, search eternal composites.
        2. Try gold overlay → if matched, search gold composites.
        3. Otherwise, search base templates.

        Args:
            cell_rgb: ``(cell_h, cell_w, 3)`` uint8 RGB heart cell.
            verbose: When ``True``, print per-cell match diagnostics
                to stderr using *row*, *col*, *abs_x*, *abs_y*.
            row: Grid row, for verbose logging only.
            col: Grid column, for verbose logging only.
            abs_x: Absolute frame x-coordinate of this cell, for
                verbose logging only.
            abs_y: Absolute frame y-coordinate of this cell, for
                verbose logging only.

        Returns:
            ``(state, template)``.  *state* is ``None`` when the best
            score is below the match threshold.  *template* is the
            ``(rgb, mask)`` pair of the best-matching candidate (always
            populated when candidates exist, even below threshold).
        """
        e_score = -1.0
        g_score = -1.0

        if self._eternal_overlay is not None:
            e_score = self._masked_score(cell_rgb, *self._eternal_overlay)
        if self._gold_overlay is not None:
            g_score = self._masked_score(cell_rgb, *self._gold_overlay)

        if e_score >= self._threshold and e_score > g_score:
            candidates = self._eternal_composites
            kind = "eternal"
        elif g_score >= self._threshold:
            candidates = self._gold_composites
            kind = "gold"
        else:
            candidates = self._base_templates
            kind = "base"

        state, best_score, top_scores = self._best_match(cell_rgb, candidates)

        if verbose:
            self._log_heart_match(
                row,
                col,
                abs_x,
                abs_y,
                e_score,
                g_score,
                state,
                best_score,
                kind,
                top_scores,
            )

        if state is not None:
            return state, candidates[state.name]

        if top_scores:
            best_name = top_scores[0][0]
            return None, candidates[best_name]
        return None, None

    def _log_heart_match(
        self,
        row: int,
        col: int,
        abs_x: int,
        abs_y: int,
        e_score: float,
        g_score: float,
        state: HeartState | None,
        score: float,
        kind: str,
        top: list[tuple[str, float]],
    ) -> None:
        """Print per-cell heart match diagnostics to stderr.

        Args:
            row: Grid row.
            col: Grid column.
            abs_x: Absolute frame x-coordinate of the cell.
            abs_y: Absolute frame y-coordinate of the cell.
            e_score: Eternal overlay match score.
            g_score: Gold overlay match score.
            state: Matched heart state, or ``None`` if below threshold.
            score: Best candidate score within *kind*'s template set.
            kind: Which candidate set was searched (``"eternal"``,
                ``"gold"``, or ``"base"``).
            top: All candidate scores, sorted descending.
        """
        rme = 1.0 / max(score, 1e-8) - 1.0
        top5 = "  ".join(f"{n}={s:.3f}" for n, s in top[:5])
        if state is not None:
            print(
                f"[heart] r={row} c={col} pos=({abs_x},{abs_y}) "
                f"et={e_score:.3f} gold={g_score:.3f} -> "
                f'"{state.name}"={score:.3f} rme={rme:.2f} '
                f"[{kind}]  top5: {top5}",
                file=sys.stderr,
            )
        else:
            print(
                f"[heart] r={row} c={col} pos=({abs_x},{abs_y}) "
                f"et={e_score:.3f} gold={g_score:.3f} NO MATCH "
                f"(best={score:.3f} rme={rme:.2f} < "
                f"{self._threshold:.2f}) [{kind}]  top5: {top5}",
                file=sys.stderr,
            )

    # -- helpers -----------------------------------------------------------

    def _maybe_quantize(self, cell_rgb: np.ndarray) -> np.ndarray:
        """Snap every pixel to the nearest palette colour."""
        if self._palette is None:
            return cell_rgb
        flat = cell_rgb.reshape(-1, 3).astype(np.float64)
        pal = self._palette.astype(np.float64)
        dists = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(dists, axis=-1)
        return pal[nearest].reshape(cell_rgb.shape).astype(np.uint8)

    def _build_palette(self, n_clusters: int) -> None:
        """Build a reduced colour palette from all template opaque pixels."""
        pixels: list[np.ndarray] = []
        for rgb, mask in self._base_templates.values():
            pixels.append(rgb[mask])
        for pair in (self._eternal_overlay, self._gold_overlay, self._unknown):
            if pair is not None:
                rgb, mask = pair
                if mask.any():
                    pixels.append(rgb[mask])
        if not pixels:
            self._palette = None
            return
        all_pixels = np.concatenate(pixels, axis=0)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(all_pixels.astype(np.float64))
        self._palette = kmeans.cluster_centers_.astype(np.uint8)

    def _quantize_template_pair(
        self,
        rgb: np.ndarray,
        mask: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantize a template's opaque pixels to the palette."""
        if self._palette is None or not mask.any():
            return rgb, mask
        flat = rgb[mask].astype(np.float64)
        pal = self._palette.astype(np.float64)
        dists = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(dists, axis=-1)
        quantized = rgb.copy()
        quantized[mask] = pal[nearest].astype(np.uint8)
        return quantized, mask

    def _quantize_base_and_overlays(self) -> None:
        """Quantize base templates, overlays, and the unknown template."""
        if self._palette is None:
            return
        for name in list(self._base_templates):
            self._base_templates[name] = self._quantize_template_pair(
                *self._base_templates[name]
            )
        if self._eternal_overlay is not None:
            self._eternal_overlay = self._quantize_template_pair(*self._eternal_overlay)
        if self._gold_overlay is not None:
            self._gold_overlay = self._quantize_template_pair(*self._gold_overlay)
        if self._unknown is not None:
            self._unknown = self._quantize_template_pair(*self._unknown)

    def _quantize_composites(self) -> None:
        """Quantize the eternal/gold composites built from quantized bases."""
        if self._palette is None:
            return
        for name in list(self._eternal_composites):
            self._eternal_composites[name] = self._quantize_template_pair(
                *self._eternal_composites[name]
            )
        for name in list(self._gold_composites):
            self._gold_composites[name] = self._quantize_template_pair(
                *self._gold_composites[name]
            )

    def _pad_rgba(self, rgba: np.ndarray) -> np.ndarray:
        """Pad an RGBA image to ``(self._cell_h, self._cell_w)``.

        Content is top-left aligned; bottom and right regions are
        filled with transparent black ``(0, 0, 0, 0)``.
        """
        h, w = rgba.shape[:2]
        if h == self._cell_h and w == self._cell_w:
            return rgba
        padded = np.zeros((self._cell_h, self._cell_w, 4), dtype=rgba.dtype)
        padded[:h, :w] = rgba
        return padded

    @staticmethod
    def _rgba_from_pair(rgb_np: np.ndarray, mask_np: np.ndarray) -> np.ndarray:
        """Reconstruct an RGBA array from an (rgb, mask) pair."""
        alpha = np.where(mask_np, 255, 0).astype(np.uint8)
        return np.dstack([rgb_np, alpha])

    @staticmethod
    def _load_rgba(filepath: Path) -> np.ndarray:
        img = cv2.imread(str(filepath), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read: {filepath}")
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        return img

    @staticmethod
    def _masked_score(
        cell_rgb: np.ndarray,
        tmpl_rgb: np.ndarray,
        tmpl_mask: np.ndarray,
    ) -> float:
        """Compute alpha-masked similarity between a cell and a template."""
        tmpl_rgba = np.dstack([tmpl_rgb, np.where(tmpl_mask, 255, 0).astype(np.uint8)])
        return masked_similarity(cell_rgb, tmpl_rgba)

    def _best_match(
        self,
        cell_rgb: np.ndarray,
        candidates: dict[str, tuple[np.ndarray, np.ndarray]],
    ) -> tuple[HeartState | None, float, list[tuple[str, float]]]:
        """Score *cell_rgb* against every entry in *candidates*.

        Returns:
            ``(state, best_score, all_scores)`` where *all_scores* is
            sorted descending by score. *state* is ``None`` when
            *best_score* is below the match threshold.
        """
        best_score = -1.0
        best_name = ""
        all_scores: list[tuple[str, float]] = []
        for name, (tmpl_rgb, tmpl_mask) in candidates.items():
            score = self._masked_score(cell_rgb, tmpl_rgb, tmpl_mask)
            all_scores.append((name, float(score)))
            if score > best_score:
                best_score = score
                best_name = name
        all_scores.sort(key=lambda s: s[1], reverse=True)
        if best_score >= self._threshold:
            return HeartState(name=best_name), best_score, all_scores
        return None, best_score, all_scores

    @staticmethod
    def _composite(
        base_rgba: np.ndarray, overlay_rgba: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        ba = base_rgba[:, :, 3].astype(np.float64) / 255.0
        oa = overlay_rgba[:, :, 3].astype(np.float64) / 255.0

        out_a = oa + ba * (1.0 - oa)
        mask = out_a > 0.0
        out_a_safe = np.maximum(out_a, 1e-8)

        br = base_rgba[:, :, :3].astype(np.float64)
        ov = overlay_rgba[:, :, :3].astype(np.float64)

        out_rgb = (
            ov * oa[..., None] + br * ba[..., None] * (1.0 - oa[..., None])
        ) / out_a_safe[..., None]
        out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)

        return out_rgb, mask

    def _build_composites(
        self,
        overlay_pair: tuple[np.ndarray, np.ndarray] | None,
        prefix: str,
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        if overlay_pair is None:
            return {}
        overlay_rgba = self._rgba_from_pair(*overlay_pair)
        composites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for name, base_pair in self._base_templates.items():
            base_rgba = self._rgba_from_pair(*base_pair)
            composite_rgb, composite_mask = self._composite(base_rgba, overlay_rgba)
            composites[prefix + name] = (composite_rgb, composite_mask)
        return composites
