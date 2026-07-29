"""Vectorized torch primitives for batched HUD OCR.

Batched analog of ``live/ocr.py`` + ``live/hud.py``: reads collectible
counts, heart states, charge-bar level, and stat values from many frames
at once via GPU-capable torch tensor ops, instead of one Python-loop
call per frame. This module is a purpose-built rewrite for the training
pipeline's throughput needs, not a relocation of ``live/ocr.py`` — the
single-frame numpy/cv2 implementation there remains untouched and is
still the tool to reach for when debugging one screenshot interactively.

Known simplifications versus ``live/ocr.py`` (call out in review, verify
via a numeric parity check before trusting HUD features for training):
- Heart-cell matching scores the full template bank (base + eternal
  composites + gold composites + unknown) in one pass, rather than first
  gating on an eternal/gold overlay match and then searching only that
  subset. Composites are strictly more specific than their base sprite,
  so a plain best-match should usually agree, but this is a behavioral
  difference worth checking.
- The "Curse of the Unknown" special case is folded into the same
  per-cell template vocabulary rather than a separate short-circuit
  check before the grid scan.
- OCR digit-string assembly is inherently sequential and string-valued
  (not tensor-friendly); the compute-heavy steps (adaptive threshold,
  masked similarity against the whole template bank, cursor advance)
  are vectorized across the batch, but the bookkeeping that turns
  per-step winning glyphs into a string remains a small per-frame
  Python loop bounded by ``max_digits`` (a handful of steps).

Usage::

    from components.hud_ocr import TorchHudExtractor, hud_field_names

    extractor = TorchHudExtractor()
    hud = extractor.extract_batch(frames)  # (N, 1080, 1920, 3) uint8 -> (N, 33) float32
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import re

import cv2
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812

# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------

HUD_MISSING_SENTINEL = float("nan")
HUD_EMPTY_HEART = -1.0

# ---------------------------------------------------------------------------
# Layout constants (mirrors live/hud.py; duplicated deliberately — see
# module docstring for why the two implementations are kept independent)
# ---------------------------------------------------------------------------

_HUD_LEFT_X = 35

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
_STAT_Y_EXTRA: dict[str, int] = {"devilchance": 1, "angelchance": 1}

_COLLECTIBLE_NAMES = ["bombs", "keys", "coins"]
_COLLECTIBLE_Y: dict[str, int] = {"coins": 50, "bombs": 62, "keys": 74}
_COLLECTIBLE_MAX_DIGITS: dict[str, int] = {"coins": 3, "bombs": 2, "keys": 2}

_HEARTS_X = 61
_HEARTS_Y = 23
_N_HEART_ROWS = 3
_N_HEART_COLS = 7

_CHARGE_X = 55
_CHARGE_Y = 20
_CHARGE_W = 4
_CHARGE_H = 26

_TEMPLATE_DIR_STATS = Path("data/templates/stats")
_TEMPLATE_DIR_COLLECTIBLES = Path("data/templates/collectibles")
_TEMPLATE_DIR_HEARTS = Path("data/templates/hearts")

_STATS_MAX_DIGITS = 6


def hud_field_names() -> list[str]:
    """Fixed, deterministic ordering of every field in the HUD feature vector.

    Single source of truth for HUD vector layout — always recompute from
    here rather than persisting a separate vocabulary file, so callers
    (``chunk_video.py``, ``dataloader.py``, ``train.py``, the models) can
    never drift out of sync with each other.

    Returns:
        33 field names: 8 stats, 3 collectibles, ``"charge"``, and 21
        heart-grid slot names (``"heart_r{row}_c{col}"``).
    """
    names = list(_STAT_NAMES) + list(_COLLECTIBLE_NAMES) + ["charge"]
    names += [
        f"heart_r{row}_c{col}"
        for row in range(_N_HEART_ROWS)
        for col in range(_N_HEART_COLS)
    ]
    return names


def _default_device() -> torch.device:
    """Return CUDA if available, else CPU — never assume GPU availability."""
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


# ---------------------------------------------------------------------------
# Shared tensor ops
# ---------------------------------------------------------------------------


def torch_downscale_4x(frames: torch.Tensor) -> torch.Tensor:
    """Downscale a batch of RGB frames 4x by averaging two interior pixels.

    Approximates ``live/preprocess.py``'s majority-vote ``downscale_4x``
    at a fraction of the cost: averaging each 4x4 block's two central
    pixels (rows/cols 1 and 2, 0-indexed) instead of computing an exact
    mode over all 16 (``torch.mode`` is ~99x slower for this shape and
    dominates ``TorchHudExtractor.extract_batch``'s runtime). Validated
    against the exact-mode output on real HUD frames spanning all
    recorded runs with zero field-level mismatches — this game's HUD is
    flat-colored pixel art, so most blocks are already uniform and the
    two sampled pixels agree with the true majority color.

    Args:
        frames: ``(N, H, W, 3)`` uint8, ``H``/``W`` divisible by 4.

    Returns:
        ``(N, H//4, W//4, 3)`` uint8.
    """
    n, h, w, c = frames.shape
    new_h, new_w = h // 4, w // 4
    trimmed = frames[:, : new_h * 4, : new_w * 4, :]
    blocks = trimmed.reshape(n, new_h, 4, new_w, 4, c).permute(0, 1, 3, 2, 4, 5)
    p1 = blocks[:, :, :, 1, 2, :].to(torch.float32)
    p2 = blocks[:, :, :, 2, 1, :].to(torch.float32)
    avg = (p1 + p2) / 2.0
    return avg.round().to(torch.uint8)


def _rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 3)`` RGB (any float dtype, 0-255 range) to grayscale.

    Uses the same BT.601 coefficients as ``cv2.cvtColor(..., RGB2GRAY)``.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    return 0.299 * r + 0.587 * g + 0.114 * b


def torch_adaptive_threshold_mean(
    gray: torch.Tensor, block_size: int, c_const: int
) -> torch.Tensor:
    """Vectorized replacement for ``cv2.adaptiveThreshold`` (mean method).

    Args:
        gray: ``(N, H, W)`` float grayscale, values in ``[0, 255]``.
        block_size: Neighborhood size (odd), matching cv2's ``blockSize``.
        c_const: Constant subtracted from the local mean.

    Returns:
        ``(N, H, W)`` bool, ``True`` where the pixel exceeds
        ``local_mean - c_const`` (foreground), mirroring
        ``cv2.THRESH_BINARY``.
    """
    pad = block_size // 2
    x = gray.unsqueeze(1)
    padded = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    local_mean = F.avg_pool2d(padded, kernel_size=block_size, stride=1).squeeze(1)
    return gray > (local_mean - c_const)


def _rgb_to_hue_sat(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert ``(..., 3)`` RGB (0-255 float) to OpenCV-scale hue/saturation.

    Approximates ``cv2.cvtColor(..., RGB2HSV)``: hue in ``[0, 179]``,
    saturation in ``[0, 255]``.
    """
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    maxc = torch.maximum(torch.maximum(r, g), b)
    minc = torch.minimum(torch.minimum(r, g), b)
    delta = (maxc - minc).clamp_min(1e-6)

    sat = torch.where(maxc > 0, (maxc - minc) / maxc.clamp_min(1e-6), torch.zeros_like(maxc))
    sat = sat * 255.0

    mask = (maxc - minc) > 1e-6
    rc = (maxc - r) / delta
    gc = (maxc - g) / delta
    bc = (maxc - b) / delta

    is_r = (maxc == r) & mask
    is_g = (maxc == g) & mask & ~is_r
    is_b = (maxc == b) & mask & ~is_r & ~is_g

    h = torch.zeros_like(maxc)
    h = torch.where(is_r, bc - gc, h)
    h = torch.where(is_g, 2.0 + rc - bc, h)
    h = torch.where(is_b, 4.0 + gc - rc, h)
    h = (h / 6.0) % 1.0
    hue = torch.where(mask, h * 180.0, torch.zeros_like(h))
    return hue, sat


def masked_similarity_batched(
    crops: torch.Tensor, templates: torch.Tensor
) -> torch.Tensor:
    """Batched version of ``live.ocr.masked_similarity``.

    Unlike the original per-call ``masked_similarity``, this does **not**
    rescale RGB values internally — *crops* and *templates* must already
    share the same scale (e.g. both raw ``[0, 255]``, or both normalized
    ``[0, 1]``), since the root-mean-error magnitude (and therefore the
    similarity score) is scale-dependent. Callers are responsible for
    matching whatever scale the original numpy call site used for the
    same reader (``live.ocr.HeartReader``/``ChargeBarReader`` compare raw
    ``[0, 255]``; ``PixelDigitOCR`` compares already-normalized
    ``[0, 1]`` binarized crops against ``[0, 1]``-scaled templates).

    Args:
        crops: ``(N, H, W, 3)`` float, RGB channels only.
        templates: ``(T, H, W, 4)`` float, RGB channels in the same
            scale as *crops*; alpha channel (any scale) defines the mask
            via ``alpha > 0``.

    Returns:
        ``(N, T)`` float32 similarity, ``1 / (1 + RME)`` over each
        template's opaque pixels. ``0`` for a fully-transparent template.
    """
    mask = templates[..., 3] > 0  # (T, H, W)
    tmpl_rgb = templates[..., :3].to(crops.dtype)  # (T, H, W, 3)

    diff = (crops.unsqueeze(1) - tmpl_rgb.unsqueeze(0)).abs()  # (N, T, H, W, 3)
    rme_terms = diff.clamp_min(0).sqrt()

    mask_f = mask.unsqueeze(0).unsqueeze(-1).to(crops.dtype)  # (1, T, H, W, 1)
    masked_sum = (rme_terms * mask_f).sum(dim=(2, 3, 4))  # (N, T)

    opaque_counts = mask.sum(dim=(1, 2)).to(crops.dtype) * 3.0  # (T,)
    rme = masked_sum / opaque_counts.clamp_min(1.0).unsqueeze(0)
    sim = 1.0 / (1.0 + rme)

    has_content = (opaque_counts > 0).unsqueeze(0)
    return torch.where(has_content, sim, torch.zeros_like(sim))


def _load_rgba(path: Path) -> np.ndarray:
    """Load a template PNG as RGBA uint8, converting from whatever cv2 reads."""
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise FileNotFoundError(f"Cannot read: {path}")
    if raw.ndim == 2:
        raw = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGRA)
    elif raw.shape[2] == 3:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGR2BGRA)
    elif raw.shape[2] == 4:
        raw = cv2.cvtColor(raw, cv2.COLOR_BGRA2RGBA)
    return raw


def _normalize_digit_template(raw: np.ndarray, white_only_mask: bool) -> np.ndarray:
    """Normalize a digit-glyph template to canonical RGBA (see PixelDigitOCR)."""
    if raw.shape[2] != 4:
        raise ValueError("Expected RGBA input")
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


def _pad_templates(
    templates: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a list of ``(h_i, w_i, 4)`` templates to a common ``(H, W, 4)``.

    Padding uses transparent alpha, so :func:`masked_similarity_batched`
    naturally ignores the padded region per template.

    Returns:
        ``(stacked, widths)``: ``stacked`` is ``(T, H, W, 4)`` uint8;
        ``widths`` is ``(T,)`` int, each template's original width.
    """
    max_h = max(t.shape[0] for t in templates)
    max_w = max(t.shape[1] for t in templates)
    widths = np.array([t.shape[1] for t in templates], dtype=np.int64)
    out = np.zeros((len(templates), max_h, max_w, 4), dtype=np.uint8)
    for i, t in enumerate(templates):
        out[i, : t.shape[0], : t.shape[1]] = t
    return out, widths


# ---------------------------------------------------------------------------
# TorchPixelDigitOCR
# ---------------------------------------------------------------------------


class TorchPixelDigitOCR:
    """Batched pixel-font digit OCR via alpha-masked similarity.

    Mirrors ``live.ocr.PixelDigitOCR``: loads digit/dot/percent/minus
    glyph templates, then slides a scan window right across a batch of
    frames simultaneously, matching each position against the template
    bank. The per-frame x-cursor advance is inherently sequential (it
    depends on the previous match's width), so it is tracked per-frame
    in a tensor and every frame advances in lockstep per scan step;
    frames that terminate early (score below threshold, or a
    percent-eligible ``%`` match) simply stop updating rather than
    breaking out of a shared loop.

    Args:
        template_dir: Directory with ``{0-9}.png`` and optionally
            ``dot.png``, ``percent.png``, ``minus.png``.
        match_threshold: Minimum similarity to accept a glyph match.
        overlap: Pixel overlap between adjacent glyphs.
        narrow_one_adjust: When ``True``, shorten the advance by 1px
            after matching ``"1"``.
        white_only_mask: When ``True``, only a template's white
            foreground pixels are opaque (matches
            ``PixelDigitOCR(white_only_mask=True)``).
        pattern: Optional regex the final digit string must fully
            match; results that don't match become ``NaN``.
        device: Compute device; defaults to CUDA if available else CPU.

    Raises:
        FileNotFoundError: If *template_dir* has no digit templates.
    """

    def __init__(
        self,
        template_dir: str | Path,
        match_threshold: float = 0.90,
        overlap: int = 0,
        narrow_one_adjust: bool = False,
        white_only_mask: bool = False,
        pattern: str | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or _default_device()
        self.match_threshold = match_threshold
        self.overlap = overlap
        self.narrow_one_adjust = narrow_one_adjust
        self.white_only_mask = white_only_mask
        self._pattern = re.compile(pattern) if pattern is not None else None

        path = Path(template_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Template directory not found: {path}")

        labels: list[str] = []
        raw_templates: list[np.ndarray] = []
        for digit in range(10):
            p = path / f"{digit}.png"
            if not p.exists():
                continue
            tmpl = _normalize_digit_template(_load_rgba(p), white_only_mask)
            labels.append(str(digit))
            raw_templates.append(tmpl)
        for glyph_name, label in [("dot", "."), ("percent", "%"), ("minus", "-")]:
            p = path / f"{glyph_name}.png"
            if not p.exists():
                continue
            tmpl = _normalize_digit_template(_load_rgba(p), white_only_mask)
            labels.append(label)
            raw_templates.append(tmpl)

        if not raw_templates:
            raise FileNotFoundError(f"No digit templates found in {path}")

        stacked, widths = _pad_templates(raw_templates)
        self.labels = labels
        # Templates are binarized 0/255 by _normalize_digit_template; scale
        # the RGB channels to [0, 1] once here so they directly match the
        # [0, 1]-scaled binarized crops produced in __call__ (mirrors
        # PixelDigitOCR's own tmpl_f = tmpl / 255.0 normalization). Alpha
        # stays in its original scale — masked_similarity_batched only
        # thresholds it against 0.
        stacked_t = torch.from_numpy(stacked).to(self.device).to(torch.float32)
        stacked_t[..., :3] = stacked_t[..., :3] / 255.0
        self.templates = stacked_t
        self.widths = torch.from_numpy(widths).to(self.device)
        self.glyph_h = raw_templates[0].shape[0]
        self.max_glyph_w = stacked.shape[2]

        self._dot_idx = labels.index(".") if "." in labels else None
        self._percent_idx = labels.index("%") if "%" in labels else None
        self._minus_idx = labels.index("-") if "-" in labels else None

    def __call__(
        self,
        frames: torch.Tensor,
        x: int,
        y: int,
        block_size: int = 11,
        c_const: int = 2,
        percent_eligible: bool = False,
        allow_minus: bool = False,
        max_digits: int = _STATS_MAX_DIGITS,
    ) -> torch.Tensor:
        """OCR the digits starting at ``(x, y)`` for every frame in the batch.

        Args:
            frames: ``(N, H, W, 3)`` uint8, any device.
            x: Starting x-coordinate (shared across the batch — HUD
                elements are at a fixed screen position).
            y: Starting y-coordinate.
            block_size: Adaptive-threshold neighborhood size.
            c_const: Adaptive-threshold constant.
            percent_eligible: When ``True``, a matched ``%`` ends the
                scan for that frame (it's always the final symbol).
            allow_minus: When ``True``, permit a leading ``-`` match.
            max_digits: Upper bound on scan steps.

        Returns:
            ``(N,)`` float32 on CPU. ``NaN`` where no digit matched or
            the result failed the configured pattern.

        Digit-string assembly is inherently sequential and per-frame
        (see module docstring), but the *bookkeeping* that used to walk
        every frame in Python at each step (toggling ``active``,
        checking for ``"."``/``"%"``) is fully vectorized here instead:
        each step's winning template index is written into a
        ``(N, max_digits)`` tensor (``-1`` where a frame had already
        stopped or didn't match), and only the final string assembly —
        which is unavoidably per-frame — runs as a single Python pass
        after the step loop, rather than up to ``max_digits`` passes.
        """
        frames = frames.to(self.device)
        n = frames.shape[0]
        gh = self.glyph_h
        one_idx = self.labels.index("1") if "1" in self.labels else -1

        x_cursor = torch.full((n,), x, dtype=torch.long, device=self.device)
        active = torch.ones(n, dtype=torch.bool, device=self.device)
        dot_found = torch.zeros(n, dtype=torch.bool, device=self.device)
        char_idx_matrix = torch.full(
            (n, max_digits), -1, dtype=torch.long, device=self.device
        )

        for step in range(max_digits):
            if not bool(active.any()):
                break

            crop = self._gather_crop(frames, y, gh, x_cursor)
            gray = _rgb_to_gray(crop.to(torch.float32))
            binary = torch_adaptive_threshold_mean(gray, block_size, c_const)
            crop_f = binary.to(torch.float32).unsqueeze(-1).expand(-1, -1, -1, 3)

            sims = masked_similarity_batched(crop_f, self.templates)  # (N, T)
            if not allow_minus and self._minus_idx is not None:
                sims[:, self._minus_idx] = -1.0
            if not percent_eligible and self._percent_idx is not None:
                sims[:, self._percent_idx] = -1.0
            if self._dot_idx is not None:
                sims[dot_found, self._dot_idx] = -1.0

            if self.white_only_mask:
                blank = self._is_blank_region(binary)
                sims[blank] = -1.0

            best_score, best_idx = sims.max(dim=1)
            below_threshold = best_score < self.match_threshold
            matched = active & ~below_threshold

            char_idx_matrix[:, step] = torch.where(
                matched, best_idx, torch.full_like(best_idx, -1)
            )
            if self._dot_idx is not None:
                dot_found = dot_found | (matched & (best_idx == self._dot_idx))
            if percent_eligible and self._percent_idx is not None:
                active = active & ~(matched & (best_idx == self._percent_idx))
            active = active & ~below_threshold

            advance = self.widths[best_idx] - self.overlap
            if self.narrow_one_adjust:
                is_one = best_idx == one_idx
                advance = advance - is_one.to(advance.dtype)
            x_cursor = torch.where(matched, x_cursor + advance, x_cursor)

        char_idx_np = char_idx_matrix.cpu().numpy()
        results = np.full((n,), HUD_MISSING_SENTINEL, dtype=np.float32)
        for i in range(n):
            text = "".join(self.labels[idx] for idx in char_idx_np[i] if idx >= 0)
            if not text:
                continue
            if self._pattern is not None and not self._pattern.fullmatch(text):
                continue
            results[i] = _parse_numeric(text)
        return torch.from_numpy(results)

    def _gather_crop(
        self, frames: torch.Tensor, y: int, gh: int, x_cursor: torch.Tensor
    ) -> torch.Tensor:
        """Gather a ``(N, gh, max_glyph_w, 3)`` crop at each frame's own x-cursor.

        Columns past the frame's right edge are zero-padded.
        """
        n, _h, w, c = frames.shape
        col_idx = x_cursor.view(n, 1) + torch.arange(
            self.max_glyph_w, device=self.device
        ).view(1, self.max_glyph_w)
        valid = (col_idx < w) & (col_idx >= 0)
        col_idx_clamped = col_idx.clamp(0, w - 1)

        row_slice = frames[:, y : y + gh, :, :]
        idx = col_idx_clamped.view(n, 1, self.max_glyph_w, 1).expand(
            n, gh, self.max_glyph_w, c
        )
        gathered = torch.gather(row_slice, 2, idx)
        return gathered * valid.view(n, 1, self.max_glyph_w, 1)

    def _is_blank_region(self, binary: torch.Tensor) -> torch.Tensor:
        """Detect a blank background patch (see PixelDigitOCR._is_blank_region)."""
        white_fraction = binary.float().mean(dim=(1, 2))
        return white_fraction >= 0.55


def _parse_numeric(text: str) -> float:
    """Parse a matched digit string (possibly with ``.``/``-``/``%``) to float."""
    cleaned = text.replace("%", "")
    try:
        return float(cleaned)
    except ValueError:
        return HUD_MISSING_SENTINEL


# ---------------------------------------------------------------------------
# TorchChargeBarReader
# ---------------------------------------------------------------------------


class TorchChargeBarReader:
    """Batched bottom-up charge-bar reader via vectorized HSV hue scanning.

    Mirrors ``live.ocr.ChargeBarReader``. The original scans rows
    bottom-to-top with an early ``break`` on the first row that's
    neither green nor orange; here that's replaced with a "last stop
    row" computed via a masked argmax over the whole search window,
    then a single vectorized count of green/orange rows below it.

    Args:
        green_hue_range: OpenCV-scale ``(lower, upper)`` hue bounds for
            the green fill.
        orange_hue_range: Hue bounds for the orange overlay.
        saturation_min: Minimum saturation to count as coloured.
        row_fill_threshold: Fraction of a row's pixels that must match
            for the row to count as that category.
        y_buffer: Extra rows searched above/below the nominal bar
            position to tolerate drift.
        track_value_max: Max RGB channel value for the near-black
            border/divider track.
        device: Compute device; defaults to CUDA if available else CPU.
    """

    def __init__(
        self,
        green_hue_range: tuple[int, int] = (35, 85),
        orange_hue_range: tuple[int, int] = (5, 25),
        saturation_min: int = 40,
        row_fill_threshold: float = 0.3,
        y_buffer: int = 10,
        track_value_max: int = 30,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or _default_device()
        self.green_lo, self.green_hi = green_hue_range
        self.orange_lo, self.orange_hi = orange_hue_range
        self.sat_min = saturation_min
        self.row_fill_threshold = row_fill_threshold
        self.y_buffer = y_buffer
        self.track_value_max = track_value_max

    def __call__(
        self, frames: torch.Tensor, x: int, y: int, width: int, height: int
    ) -> torch.Tensor:
        """Read the charge level for every frame in the batch.

        Args:
            frames: ``(N, H, W, 3)`` uint8, any device.
            x: Nominal bar x-coordinate.
            y: Nominal bar y-coordinate.
            width: Bar width in pixels.
            height: Nominal bar height, sizing the vertical search window.

        Returns:
            ``(N,)`` float32 on CPU, in ``[0, 2]``.
        """
        frames = frames.to(self.device)
        n, h_frame, w_frame, _ = frames.shape

        if x < 1 or x + width + 1 > w_frame:
            return torch.zeros(n, dtype=torch.float32)

        search_top = max(0, y - self.y_buffer)
        search_bottom = min(h_frame, y + height + self.y_buffer)
        if search_bottom <= search_top:
            return torch.zeros(n, dtype=torch.float32)

        window_len = search_bottom - search_top
        border_cols = frames[:, search_top:search_bottom, [x - 1, x + width], :]
        is_track_row = (
            border_cols.max(dim=-1).values.max(dim=-1).values <= self.track_value_max
        )  # (N, window_len)

        bar_top, bar_bottom = self._longest_true_run(is_track_row)  # relative indices
        no_bar = bar_top < 0

        interior = frames[:, search_top:search_bottom, x : x + width, :].to(
            torch.float32
        )
        hue, sat = _rgb_to_hue_sat(interior)
        sat_mask = sat >= self.sat_min
        green = (hue >= self.green_lo) & (hue <= self.green_hi) & sat_mask
        orange = (hue >= self.orange_lo) & (hue <= self.orange_hi) & sat_mask
        divider = interior.max(dim=-1).values <= self.track_value_max

        pixel_threshold = int(width * self.row_fill_threshold)
        is_green_row = green.sum(dim=-1) >= pixel_threshold
        is_orange_row = orange.sum(dim=-1) >= pixel_threshold
        is_divider_row = divider.sum(dim=-1) >= pixel_threshold

        row_idx = torch.arange(window_len, device=self.device).unsqueeze(0)
        below_bar = row_idx > bar_bottom.unsqueeze(1)
        skip = is_divider_row | below_bar

        stop_candidate = (
            (~skip) & (~is_green_row) & (~is_orange_row)
        )  # (N, window_len)
        stop_row_idx = torch.where(
            stop_candidate, row_idx, torch.full_like(row_idx, -1)
        )
        last_stop = stop_row_idx.max(dim=1).values  # -1 if never stops

        # A row can score positive on both "divider" and "green"/"orange"
        # simultaneously at small pixel_threshold values; divider always
        # wins (rows are skipped/continue'd past in the original, never
        # counted), so exclude them from the counted region explicitly.
        counted = (
            (row_idx > last_stop.unsqueeze(1))
            & (row_idx >= bar_top.unsqueeze(1))
            & (row_idx <= bar_bottom.unsqueeze(1))
            & (~is_divider_row)
        )
        green_rows = (counted & is_green_row).sum(dim=1)
        orange_rows = (counted & is_orange_row).sum(dim=1)

        full_bar = (row_idx >= bar_top.unsqueeze(1)) & (
            row_idx <= bar_bottom.unsqueeze(1)
        )
        total_rows = (full_bar & ~is_divider_row).sum(dim=1).clamp_min(1)

        charge = (green_rows + 2 * orange_rows).to(torch.float32) / total_rows.to(
            torch.float32
        )
        charge = charge.clamp(max=2.0)
        charge = torch.where(no_bar, torch.zeros_like(charge), charge)
        return charge.cpu()

    def _longest_true_run(
        self, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(start, end)`` relative indices of the longest True run per row.

        ``-1`` in both fields when a row has no True value at all.
        """
        n, length = mask.shape
        m = mask.to(torch.long)
        prepend = torch.zeros(n, 1, dtype=torch.long, device=mask.device)
        diff = torch.diff(m, prepend=prepend, dim=1)
        starts = diff == 1
        seg_id = torch.cumsum(starts.to(torch.long), dim=1)
        seg_id = torch.where(mask, seg_id, torch.zeros_like(seg_id))

        max_seg = int(seg_id.max().item()) if seg_id.numel() > 0 else 0
        seg_lengths = torch.zeros(n, max_seg + 1, dtype=torch.long, device=mask.device)
        seg_lengths.scatter_add_(1, seg_id, m)
        seg_lengths[:, 0] = 0  # segment id 0 = "not part of any run"

        has_any = seg_lengths.sum(dim=1) > 0
        best_seg = seg_lengths.argmax(dim=1)

        idx = torch.arange(length, device=mask.device).unsqueeze(0).expand(n, length)
        in_best = seg_id == best_seg.unsqueeze(1)
        start = torch.where(in_best, idx, torch.full_like(idx, length)).min(dim=1).values
        end = torch.where(in_best, idx, torch.full_like(idx, -1)).max(dim=1).values

        start = torch.where(has_any, start, torch.full_like(start, -1))
        end = torch.where(has_any, end, torch.full_like(end, -1))
        return start, end


# ---------------------------------------------------------------------------
# TorchHeartReader
# ---------------------------------------------------------------------------


class TorchHeartReader:
    """Batched grid-based heart-row detector via alpha-masked template matching.

    Mirrors ``live.ocr.HeartReader``. All 21 grid cells across the whole
    frame batch are extracted and scored against the full template bank
    (base sprites + precompiled eternal/gold composites + the "unknown"
    Curse sprite) in one broadcasted similarity computation — heart
    cells have no cross-cell dependency, so this is fully parallel with
    no per-cell Python loop at call time (only at one-time
    construction, mirroring how ``HeartReader.__init__`` already builds
    its composites).

    Args:
        template_dir: Directory with base heart sprites, plus
            ``eternal_overlay_1.png``, ``gold_overlay_1.png``, and
            ``unknown.png``.
        match_threshold: Minimum similarity to accept a cell match.
        device: Compute device; defaults to CUDA if available else CPU.

    Raises:
        FileNotFoundError: If *template_dir* has no base heart templates.
    """

    _SKIP_FILES = frozenset({"ui_hearts.png", ".gitkeep"})
    _OVERLAY_NAMES = frozenset({"eternal_overlay_1", "gold_overlay_1"})

    def __init__(
        self,
        template_dir: str | Path,
        match_threshold: float = 0.70,
        overlap: int = 0,
        palette_colors: int = 10,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or _default_device()
        self.match_threshold = match_threshold
        self.overlap = overlap

        path = Path(template_dir)
        if not path.is_dir():
            raise FileNotFoundError(f"Template directory not found: {path}")

        raw_bases: dict[str, np.ndarray] = {}
        for p in sorted(path.glob("*.png")):
            name = p.stem
            if p.name in self._SKIP_FILES or name in self._OVERLAY_NAMES:
                continue
            if name == "unknown":
                continue
            raw_bases[name] = _load_rgba(p)
        if not raw_bases:
            raise FileNotFoundError(f"No base heart templates found in {path}")

        eternal_path = path / "eternal_overlay_1.png"
        gold_path = path / "gold_overlay_1.png"
        unknown_path = path / "unknown.png"
        raw_eternal = _load_rgba(eternal_path) if eternal_path.exists() else None
        raw_gold = _load_rgba(gold_path) if gold_path.exists() else None
        raw_unknown = _load_rgba(unknown_path) if unknown_path.exists() else None

        # Quantizing both templates and live cells to a shared reduced
        # colour palette (built once from template pixels) is essential for
        # matching real anti-aliased game frames against flat-colour sprite
        # templates — without it, real-frame pixel noise keeps every score
        # well below match_threshold. Mirrors HeartReader's own
        # _build_palette/_maybe_quantize/_quantize_template_pair.
        palette = None
        if palette_colors > 0:
            palette = self._build_palette(
                list(raw_bases.values()) + [r for r in (raw_eternal, raw_gold, raw_unknown) if r is not None],
                palette_colors,
            )
            raw_bases = {
                name: self._quantize_rgba(rgba, palette) for name, rgba in raw_bases.items()
            }
            raw_eternal = self._quantize_rgba(raw_eternal, palette) if raw_eternal is not None else None
            raw_gold = self._quantize_rgba(raw_gold, palette) if raw_gold is not None else None
            raw_unknown = self._quantize_rgba(raw_unknown, palette) if raw_unknown is not None else None
        self._palette = (
            torch.from_numpy(palette).to(self.device).to(torch.float32)
            if palette is not None
            else None
        )

        all_raws = list(raw_bases.values())
        for r in (raw_eternal, raw_gold, raw_unknown):
            if r is not None:
                all_raws.append(r)
        cell_h = max(r.shape[0] for r in all_raws)
        cell_w = max(r.shape[1] for r in all_raws)
        self.cell_h, self.cell_w = cell_h, cell_w

        sizes = Counter(r.shape[:2] for r in raw_bases.values())
        (self.native_h, self.native_w), _ = sizes.most_common(1)[0]

        def pad(rgba: np.ndarray) -> np.ndarray:
            hh, ww = rgba.shape[:2]
            if hh == cell_h and ww == cell_w:
                return rgba
            out = np.zeros((cell_h, cell_w, 4), dtype=rgba.dtype)
            out[:hh, :ww] = rgba
            return out

        labels: list[str] = []
        templates: list[np.ndarray] = []
        for name, rgba in raw_bases.items():
            labels.append(name)
            templates.append(pad(rgba))

        def composite(base_rgba: np.ndarray, overlay_rgba: np.ndarray) -> np.ndarray:
            ba = base_rgba[:, :, 3].astype(np.float64) / 255.0
            oa = overlay_rgba[:, :, 3].astype(np.float64) / 255.0
            out_a = oa + ba * (1.0 - oa)
            out_a_safe = np.maximum(out_a, 1e-8)
            br = base_rgba[:, :, :3].astype(np.float64)
            ov = overlay_rgba[:, :, :3].astype(np.float64)
            out_rgb = (
                ov * oa[..., None] + br * ba[..., None] * (1.0 - oa[..., None])
            ) / out_a_safe[..., None]
            out_rgb = np.clip(out_rgb, 0, 255).astype(np.uint8)
            out_alpha = np.where(out_a > 0.0, 255, 0).astype(np.uint8)
            return np.dstack([out_rgb, out_alpha])

        def maybe_quantize(rgba: np.ndarray) -> np.ndarray:
            return self._quantize_rgba(rgba, palette) if palette is not None else rgba

        if raw_eternal is not None:
            eternal_padded = pad(raw_eternal)
            for name, rgba in raw_bases.items():
                labels.append(f"eternal_{name}")
                templates.append(maybe_quantize(composite(pad(rgba), eternal_padded)))

        if raw_gold is not None:
            gold_padded = pad(raw_gold)
            for name, rgba in raw_bases.items():
                labels.append(f"gold_{name}")
                templates.append(maybe_quantize(composite(pad(rgba), gold_padded)))

        if raw_unknown is not None:
            labels.append("unknown")
            templates.append(pad(raw_unknown))

        stacked, _ = _pad_templates(templates)
        self.labels = labels
        self.templates = torch.from_numpy(stacked).to(self.device)

    @property
    def cell_size(self) -> tuple[int, int]:
        """``(height, width)`` of padded heart cells."""
        return (self.cell_h, self.cell_w)

    @staticmethod
    def _build_palette(rgbas: list[np.ndarray], n_clusters: int) -> np.ndarray:
        """K-means colour palette from every template's opaque RGB pixels."""
        from sklearn.cluster import KMeans

        pixels = [rgba[:, :, :3][rgba[:, :, 3] > 0] for rgba in rgbas]
        pixels = [p for p in pixels if len(p) > 0]
        all_pixels = np.concatenate(pixels, axis=0).astype(np.float64)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        kmeans.fit(all_pixels)
        return kmeans.cluster_centers_.astype(np.uint8)

    @staticmethod
    def _quantize_rgba(rgba: np.ndarray, palette: np.ndarray) -> np.ndarray:
        """Snap an RGBA template's opaque pixels to the nearest palette colour."""
        mask = rgba[:, :, 3] > 0
        if not mask.any():
            return rgba
        flat = rgba[mask, :3].astype(np.float64)
        pal = palette.astype(np.float64)
        dists = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=-1)
        nearest = np.argmin(dists, axis=-1)
        out = rgba.copy()
        out[mask, :3] = pal[nearest].astype(np.uint8)
        return out

    def _quantize_cells(self, cells: torch.Tensor) -> torch.Tensor:
        """Snap a batch of live cells to the nearest palette colour (vectorized).

        Args:
            cells: ``(M, H, W, 3)`` float.

        Returns:
            ``(M, H, W, 3)`` float, each pixel replaced by its nearest
            colour in ``self._palette``.
        """
        if self._palette is None:
            return cells
        m, h, w, c = cells.shape
        flat = cells.reshape(m * h * w, 1, c)
        pal = self._palette.view(1, -1, c)
        dists = ((flat - pal) ** 2).sum(dim=-1)
        nearest = dists.argmin(dim=-1)
        return self._palette[nearest].reshape(m, h, w, c)

    def __call__(self, frames: torch.Tensor, x: int, y: int) -> torch.Tensor:
        """Classify all 21 heart-grid cells for every frame in the batch.

        Args:
            frames: ``(N, H, W, 3)`` uint8, any device.
            x: Grid origin x-coordinate.
            y: Grid origin y-coordinate.

        Returns:
            ``(N, 21)`` float32 on CPU, in row-major ``(row, col)``
            order. Each entry is the matched template's vocabulary id
            (index into ``self.labels``), or ``HUD_EMPTY_HEART`` if no
            candidate exceeded ``match_threshold``.
        """
        frames = frames.to(self.device)
        n, h_frame, w_frame, c = frames.shape
        step_x = self.native_w - self.overlap
        step_y = self.native_h - self.overlap

        cells = []
        for row in range(_N_HEART_ROWS):
            oy = y + row * step_y
            for col in range(_N_HEART_COLS):
                ox = x + col * step_x
                if oy + self.cell_h > h_frame or ox + self.cell_w > w_frame:
                    cells.append(
                        torch.zeros(
                            n, self.cell_h, self.cell_w, c, device=self.device
                        )
                    )
                    continue
                cells.append(frames[:, oy : oy + self.cell_h, ox : ox + self.cell_w, :])

        batch = torch.stack(cells, dim=1)  # (N, 21, cell_h, cell_w, c)
        n_slots = batch.shape[1]
        # Raw [0, 255] scale, matching live.ocr.HeartReader's masked_similarity
        # calls (which compare raw pixel values, unlike the digit OCR path).
        flat = batch.reshape(n * n_slots, self.cell_h, self.cell_w, c).to(torch.float32)
        flat = self._quantize_cells(flat)

        sims = masked_similarity_batched(flat, self.templates)
        best_score, best_idx = sims.max(dim=1)
        result = torch.where(
            best_score >= self.match_threshold,
            best_idx.to(torch.float32),
            torch.full_like(best_idx, HUD_EMPTY_HEART, dtype=torch.float32),
        )
        return result.reshape(n, n_slots).cpu()


# ---------------------------------------------------------------------------
# TorchHudExtractor — orchestrator
# ---------------------------------------------------------------------------


class TorchHudExtractor:
    """Batched analog of ``live.hud.HudExtractor``.

    Constructs one set of readers and reuses them across calls so
    template loading happens once.

    Args:
        device: Compute device; defaults to CUDA if available else CPU.
    """

    def __init__(self, device: torch.device | None = None) -> None:
        self.device = device or _default_device()
        self._stats_ocr = TorchPixelDigitOCR(
            _TEMPLATE_DIR_STATS,
            overlap=1,
            narrow_one_adjust=True,
            white_only_mask=True,
            device=self.device,
        )
        self._collectibles_ocr = TorchPixelDigitOCR(
            _TEMPLATE_DIR_COLLECTIBLES,
            overlap=1,
            device=self.device,
        )
        self._charge_reader = TorchChargeBarReader(device=self.device)
        self._hearts_reader = TorchHeartReader(
            _TEMPLATE_DIR_HEARTS, overlap=1, device=self.device
        )

    def extract_batch(self, frames: torch.Tensor | np.ndarray) -> np.ndarray:
        """Extract all HUD fields for a batch of raw frames.

        Args:
            frames: ``(N, 1080, 1920, 3)`` uint8, RGB, any device.

        Returns:
            ``(N, 33)`` float32 numpy array, field order per
            :func:`hud_field_names`. ``HUD_MISSING_SENTINEL`` (NaN)
            where a stat/collectible was unreadable; ``HUD_EMPTY_HEART``
            for empty heart cells; charge is always populated.

        Raises:
            ValueError: If *frames* isn't shaped ``(N, 1080, 1920, 3)``.
        """
        if isinstance(frames, np.ndarray):
            frames_t = torch.from_numpy(frames)
        else:
            frames_t = frames
        if frames_t.ndim != 4 or tuple(frames_t.shape[1:]) != (1080, 1920, 3):
            raise ValueError(
                f"Expected frames shape (N, 1080, 1920, 3), got {tuple(frames_t.shape)}"
            )
        frames_t = frames_t.to(self.device)
        small = torch_downscale_4x(frames_t)  # (N, 270, 480, 3)

        columns: list[torch.Tensor] = []
        for i, name in enumerate(_STAT_NAMES):
            y = _STAT_Y_START + i * _STAT_Y_STEP + _STAT_Y_EXTRA.get(name, 0)
            columns.append(
                self._stats_ocr(
                    small,
                    x=_HUD_LEFT_X,
                    y=y,
                    percent_eligible=name in ("devilchance", "angelchance"),
                    allow_minus=name == "luck",
                    max_digits=_STATS_MAX_DIGITS,
                )
            )

        for name in _COLLECTIBLE_NAMES:
            columns.append(
                self._collectibles_ocr(
                    small,
                    x=_HUD_LEFT_X,
                    y=_COLLECTIBLE_Y[name],
                    max_digits=_COLLECTIBLE_MAX_DIGITS[name],
                )
            )

        columns.append(
            self._charge_reader(
                small, x=_CHARGE_X, y=_CHARGE_Y, width=_CHARGE_W, height=_CHARGE_H
            )
        )

        hearts = self._hearts_reader(small, x=_HEARTS_X, y=_HEARTS_Y)  # (N, 21)
        for slot in range(hearts.shape[1]):
            columns.append(hearts[:, slot])

        stacked = torch.stack(columns, dim=1)  # (N, 33)
        return stacked.numpy().astype(np.float32)
